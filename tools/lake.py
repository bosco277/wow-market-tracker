"""Command line for the loader and the item picker.

    python tools/lake.py plan --days 3          # how many files / MB would this touch? downloads nothing
    python tools/lake.py files --days 5         # which files exist, day by day (downloads nothing)
    python tools/lake.py inspect --days 4       # downloads each DAILY file and shows what is inside (timestamps, empty cells)
    python tools/lake.py rates --ids 13466,42253 --days 4   # our estimated units sold per day for those items
    python tools/lake.py market --ids 13466 --days 8   # price bands, TSM-style value, price wall, flip shelf
    python tools/lake.py profile --ids 13466 --days 8  # hour of day and day of week profile of price and quantity
    python tools/lake.py pick --days 4          # load recent commodity data, check it, write the experiment shortlist
    python tools/lake.py order --days 3         # is the oldest or the newest listing at a price bought first?
    python tools/lake.py item --ids 192841 --days 2   # hourly price, depth and estimated sales for one or more items
    python tools/lake.py flips --days 3         # find market flips (cheap side bought out, price jumped) and what followed

Defaults read from Hugging Face (no daily download cap). Use --store b2 only if you must; it is counted against a budget.
Environment: HF_TOKEN (private repo), B2_KEY_ID / B2_APP_KEY (only for --store b2),
             BLIZZARD_CLIENT_ID / BLIZZARD_CLIENT_SECRET (optional: item names).
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.loader import ByteBudget, NEEDED_COLUMNS, RangeFile, normalize_frame, read_parquet_file, B2Store, HFStore, LocalStore, MB, coverage_report, day_inventory, choose_files, load_snapshots, plan_load  # noqa: E402
from engine.itemview import detect_flips, gaps_in, item_history, revert_share_within
from engine.market import (find_shelf, flag_blips, flag_spikes, item_series, price_bands, profile, reach_stats, tsm_style_market_value,
                           wall_report)  # noqa: E402
from engine.ordercheck import tier_order_evidence  # noqa: E402
from engine.picker import PickerConfig, explain, item_stats, pick, publish_minutes  # noqa: E402

HF_REPO = "bosco277/wow-market-data-lake"
B2_BUCKET = "wow-market-data-lake"


def make_store(args):
    if args.store == "hf":
        return HFStore(HF_REPO, os.environ.get("HF_TOKEN"))
    if args.store == "b2":
        return B2Store(B2_BUCKET, os.environ["B2_KEY_ID"], os.environ["B2_APP_KEY"])
    return LocalStore(args.root)


def date_range(args):
    end = date.fromisoformat(args.end) if args.end else datetime.now(timezone.utc).date()
    return end - timedelta(days=args.days), end


def fetch_names(item_ids):
    """Retail item names from Blizzard's item API. Best effort: any failure just means no name."""
    cid, secret = os.environ.get("BLIZZARD_CLIENT_ID"), os.environ.get("BLIZZARD_CLIENT_SECRET")
    if not (cid and secret):
        return {}
    import requests
    try:
        tok = requests.post("https://oauth.battle.net/token", auth=(cid, secret),
                            data={"grant_type": "client_credentials"}, timeout=30).json()["access_token"]
    except Exception as e:
        print(f"(names skipped: token request failed: {e})")
        return {}
    names = {}
    for i in item_ids:
        try:
            r = requests.get(f"https://us.api.blizzard.com/data/wow/item/{int(i)}",
                             params={"namespace": "static-us", "locale": "en_US"},
                             headers={"Authorization": f"Bearer {tok}"}, timeout=20)
            if r.ok:
                names[int(i)] = r.json().get("name")
        except Exception:
            pass
    return names


def run_plan(store, args, reader=None):
    start, end = date_range(args)
    p = plan_load(store, "commodities", start, end, cache_dir=args.cache)
    print(p.describe())
    return p


def run_pick(store, args, reader=None, names_fn=fetch_names):
    start, end = date_range(args)
    cap = int(args.max_gb * 1024 * MB) if args.store == "b2" else None
    res = load_snapshots(store, "commodities", start, end, cache_dir=args.cache, max_bytes=cap, on_error="skip",
                         reader=reader, progress=print)
    print(res.plan.describe())
    print(f"downloaded this run: {res.downloaded_bytes / MB:.0f} MB in {res.requests} requests; "
          f"{len(res.snapshots)} snapshots; {res.duplicates_dropped} duplicates dropped; {len(res.skipped)} files skipped")
    for key, why in res.skipped:
        print(f"  SKIPPED {key}: {why}")
    if res.incomplete_dropped:
        print(f"  {len(res.incomplete_dropped)} snapshots had far fewer listings than usual (partial fetches) and were left out")

    rep = coverage_report(res.snapshots)
    print("\n== DATA HEALTH ==")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    _print_holes(res.snapshots)
    if len(res.snapshots) < 3:
        print("Not enough snapshots to continue.")
        return None
    last_book = res.snapshots[-1].book
    whole_silver = float((last_book["unit_price"] % 100 == 0).mean())
    print(f"\n== PRICE STEP CHECK ==\n  {whole_silver:.1%} of current listings are priced in whole silver "
          f"(if ~100%, the smallest price step is 1 silver, not 1 copper)")
    mins = publish_minutes(res.snapshots)
    print("\n== WHEN BLIZZARD'S DATA REFRESHES (minute of the hour -> how many snapshots) ==")
    if int(mins.sum()) < 12:
        print(f"  Not enough yet: only {int(mins.sum())} snapshots carry Blizzard's own publish time "
              f"(the rest only record when we fetched). Re-run once there are 24 or more.")
    else:
        print("  " + ", ".join(f":{int(m):02d} x{int(c)}" for m, c in mins.items()))

    cfg = PickerConfig(max_position_gold=args.max_position_gold, n_items=args.n_items,
                       min_floor_gold=args.min_floor_gold, max_units=args.max_units,
                       window_hours=args.window_hours, min_coverage=args.min_coverage)
    stats = item_stats(res.snapshots, cfg=cfg)
    short = pick(stats, cfg)
    if not stats.empty:
        print(f"\n== HOW MUCH DATA THE ITEMS HAD ==\n  window {cfg.window_hours:.0f} h; a typical item was observed for "
              f"{stats['observed_hours'].median():.0f} h of it (coverage {stats['coverage'].median():.2f}; needs {cfg.min_coverage:.2f} "
              f"and {cfg.min_observed_hours:.0f} h)")
    print("\n== WHY ITEMS WERE REJECTED ==")
    for k, v in explain(stats, cfg).items():
        print(f"  {k}: {v}")
    if short.empty:
        print("\nNo item passed every test. Re-run with a bigger --max-position-gold or more days of data.")
        return short

    names = names_fn(short["item_id"].tolist())
    short.insert(1, "name", short["item_id"].map(lambda i: names.get(int(i), f"item {int(i)}")))
    os.makedirs(args.out, exist_ok=True)
    short.to_csv(os.path.join(args.out, "experiment_shortlist.csv"), index=False)
    show = short[["item_id", "name", "speed", "floor_gold", "daily_est", "buy_units", "back_lane_units_ahead", "position_gold", "price_cv"]].round(2)
    print("\n== SHORTLIST ==")
    print(show.to_string(index=False))
    print(f"\nTotal money in play if you buy one batch of everything: {short['position_gold'].sum():,.0f} gold")
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        body = show.to_markdown(index=False) if _has_tabulate() else show.to_string(index=False)
        fh.write("# Experiment shortlist\n\n" + body + "\n")
    return short


def run_files(store, args, reader=None):
    """Lists what is in the store, day by day (UTC), WITHOUT downloading anything."""
    start, end = date_range(args)
    objs = store.list("commodities/")
    inv = day_inventory(objs, "commodities", start, end)
    used = {o.key for o in choose_files(objs, "commodities", start, end)}
    print(f"== FILES IN {args.store.upper()} FOR commodities, {start} to {end} (UTC days; ET = UTC minus 4 h until Nov 1, then minus 5 h) ==")
    d = start
    while d <= end:
        v = inv.get(d)
        if not v:
            print(f"  {d}: NOTHING")
        else:
            hrs = v["hours"]
            missing = [h for h in range(24) if h not in hrs] if d < end else []
            daily = f"daily file {sum(o.size for o in v['daily']) / MB:.0f} MB" if v["daily"] else "no daily file"
            use = "hourly" + (" + daily" if any(o.key in used for o in v["daily"]) else "") if v["hourly"] else "daily only"
            line = f"  {d}: {len(v['hourly'])} hourly files ({len(hrs)} different UTC hours), {daily}; loader uses: {use}"
            if missing and d < end:
                line += f"\n        UTC hours with no hourly file: {', '.join(f'{h:02d}' for h in missing)}"
            print(line)
        d += timedelta(days=1)
    return inv


def inspect_frame(df, source="commodities"):
    """What is really inside one stored file: columns, empty cells, and one line per snapshot timestamp."""
    ts = None
    if "snapshot_ts" in df.columns:
        ts = pd.to_datetime(df["snapshot_ts"])
    if "scan_timestamp" in df.columns:
        sc = pd.to_datetime(df["scan_timestamp"])
        ts = sc if ts is None else ts.fillna(sc)
    ids = pd.Series(float("nan"), index=df.index)
    for c in ("item_id", "clean_item_id"):
        if c in df.columns:
            ids = ids.fillna(pd.to_numeric(df[c], errors="coerce"))
    if "item" in df.columns:
        ids = ids.fillna(pd.to_numeric(df["item"].map(lambda x: x.get("id") if isinstance(x, dict) else None), errors="coerce"))
    by = pd.DataFrame({"ts": ts, "has_id": ids.notna(),
                       "from_snapshot_ts": df["snapshot_ts"].notna() if "snapshot_ts" in df.columns else False})
    table = by.groupby("ts").agg(rows=("has_id", "size"), rows_with_item_id=("has_id", "sum"),
                                 from_snapshot_ts=("from_snapshot_ts", "all")).reset_index().sort_values("ts")
    nulls = {c: int(df[c].isna().sum()) for c in ("snapshot_ts", "scan_timestamp", "item_id", "clean_item_id", "id", "unit_price",
                                                  "buyout", "quantity", "time_left") if c in df.columns}
    kept = len(normalize_frame(df, source))
    return dict(columns={c: str(t) for c, t in df.dtypes.items()}, rows=len(df), nulls=nulls, kept_by_loader=kept, table=table)


def run_inspect(store, args, reader=None):
    """Downloads the DAILY file of each day in range and shows what is inside (HF only; B2 would use your cap)."""
    start, end = date_range(args)
    objs = store.list("commodities/")
    inv = day_inventory(objs, "commodities", start, end)
    reader = reader or read_parquet_file
    budget = ByteBudget(store.name, None, None)
    for day in sorted(inv):
        for o in inv[day]["daily"]:
            print(f"\n== {o.key} ({o.size / MB:.0f} MB) ==")
            df = reader(RangeFile(store, o.key, o.size, budget, 4 * MB), None)
            r = inspect_frame(df)
            print(f"  columns read: {', '.join(f'{c}:{t}' for c, t in r['columns'].items())}")
            print(f"  rows: {r['rows']:,}; rows the loader keeps: {r['kept_by_loader']:,}; empty cells: {r['nulls']}")
            t = r["table"]
            print(f"  {len(t)} different snapshot timestamps (US Eastern; 'from' says which column supplied the time):")
            for _, x in t.iterrows():
                flag = "" if x["rows_with_item_id"] == x["rows"] else f"  <-- {x['rows'] - x['rows_with_item_id']:,.0f} rows have no item id"
                print(f"    {_et(x['ts'])}  rows {x['rows']:>8,.0f}  from {'snapshot_ts' if x['from_snapshot_ts'] else 'scan_timestamp (fetch time)'}{flag}")
    return inv


def _print_holes(snapshots):
    g = gaps_in(snapshots)
    print("\n== HOLES IN THE DATA (no snapshots for over 2 hours; US Eastern) ==")
    if g.empty:
        print("  none")
        return g
    for _, r in g.iterrows():
        print(f"  {_et(r['start'])} -> {_et(r['end'])}  ({r['hours']:.1f} h with nothing recorded)")
    print(f"  total: {g['hours'].sum():.1f} h. Sales inside a hole cannot be measured, so those hours are left out of every rate.")
    return g


def run_order(store, args, reader=None):
    start, end = date_range(args)
    res = load_snapshots(store, "commodities", start, end, cache_dir=args.cache, on_error="skip", reader=reader,
                         progress=print)
    print(res.plan.describe())
    print(f"{len(res.snapshots)} snapshots loaded, {len(res.skipped)} files skipped")
    ev = tier_order_evidence(res.snapshots)
    print("\n== WHICH LISTING AT THE SAME PRICE IS BOUGHT FIRST? ==")
    print("  " + ev.describe())
    return ev


def _et(ts):
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert("America/New_York").strftime("%a %m-%d %H:%M")


def _load_all(store, args, reader=None):
    start, end = date_range(args)
    res = load_snapshots(store, "commodities", start, end, cache_dir=args.cache, on_error="skip", reader=reader,
                         progress=print)
    print(res.plan.describe())
    print(f"{len(res.snapshots)} snapshots loaded, {len(res.skipped)} files skipped")
    return res.snapshots


def run_item(store, args, reader=None, names_fn=fetch_names):
    ids = [int(x) for x in str(args.ids).replace(" ", "").split(",") if x]
    snaps = _load_all(store, args, reader)
    names = names_fn(ids)
    for item in ids:
        h = item_history(snaps, item)
        print(f"\n== ITEM {item}: {names.get(item, '')} (times are US Eastern; sold_* are units estimated sold in the hour before) ==")
        if h.empty:
            print("  not in the loaded data")
            continue
        h = h.copy()
        h.insert(0, "when_et", h["ts"].map(_et))
        d = h["sold_est"].dropna()
        if len(d) and d.sum() > 0:
            print(f"  demand shape: sales in {(d >= 1).mean():.0%} of hours; the single biggest hour is {d.max() / d.sum():.0%} of all estimated sales "
                  f"({d.sum():.0f} units over {len(d)} hours)")
        show = h[["when_et", "floor_gold", "units_at_floor", "units_within_10pct", "units_listed", "new_units",
                  "sold_low", "sold_est", "sold_high"]].round(2)
        print(show.to_string(index=False))
    return snaps


def run_rates(store, args, reader=None, names_fn=fetch_names):
    """One line per item: our estimated units sold per day (low / best / high), price and depth. For comparing with other sources."""
    ids = [int(x) for x in str(args.ids).replace(" ", "").split(",") if x]
    snaps = _load_all(store, args, reader)
    _print_holes(snaps)
    cfg = PickerConfig(window_hours=args.window_hours)
    st = item_stats(snaps, cfg=cfg).set_index("item_id")
    names = names_fn(ids)
    rows = []
    for i in ids:
        if i not in st.index:
            print(f"  item {i}: not in the loaded data")
            continue
        r = st.loc[i]
        rows.append(dict(item_id=i, name=names.get(i, ""), floor_gold=r["floor_now"] / 10000.0, daily_low=r["daily_low"],
                         daily_est=r["daily_est"], daily_high=r["daily_high"], hours_observed=r["observed_hours"],
                         price_cv=r["price_cv"], active_share=r["active_share"], top_hour_share=r["top_interval_share"],
                         units_near_floor=r["depth_10pct"]))
    out = pd.DataFrame(rows)
    print(f"\n== OUR SALES ESTIMATE PER DAY (last {cfg.window_hours:.0f} h of observed time; a FLOOR on true sales because of the blind spot) ==")
    print(out.round(2).to_string(index=False) if len(out) else "  nothing to show")
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "rates.csv"), index=False)
    return out


def _rates_from_series(df, window_hours):
    """Units per HOUR (low / est / high) from the last `window_hours` of observed (not flagged) time of one item's series."""
    d = df[pd.to_datetime(df["ts"]) > pd.to_datetime(df["ts"]).max() - pd.Timedelta(hours=window_hours)]
    d = d[~(d["flag_long_gap"].fillna(False).astype(bool) | d["flag_wipe"].fillna(False).astype(bool))].dropna(subset=["gap_minutes"])
    hours = d["gap_minutes"].sum() / 60.0
    if hours <= 0:
        return None
    return dict(low=float(d["sold_low"].sum() / hours), est=float(d["sold_est"].sum() / hours), high=float(d["sold_high"].sum() / hours)), hours


def _g(x):
    return "n/a" if x is None or (isinstance(x, float) and (x != x)) else f"{x:,.2f}"


def _h(x):
    return "never" if x == float("inf") else f"{x:.1f} h"


def run_market(store, args, reader=None, names_fn=fetch_names):
    """Market rate, price bands, price wall and flip shelf for the item ids you give."""
    ids = [int(x) for x in str(args.ids).replace(" ", "").split(",") if x]
    snaps = _load_all(store, args, reader)
    _print_holes(snaps)
    names = names_fn(ids)
    series = item_series(snaps, ids)
    last = snaps[-1]
    rows = []
    for i in ids:
        df = series.get(i)
        print(f"\n== ITEM {i}: {names.get(i, '')} (as of the snapshot at {_et(last.ts)} ET) ==")
        if df is None or df.empty:
            print("  not in the loaded data")
            continue
        sp = flag_spikes(df["floor_gold"])
        b = price_bands(df, sp)
        book_i = last.book[last.book["item_id"] == i]
        if book_i.empty:
            print("  not listed in the latest snapshot")
            continue
        now = float(book_i["unit_price"].min()) / 10000.0
        r = dict(item_id=i, name=names.get(i, ""), floor_now=now, **{k: v for k, v in b.items() if k.startswith("p") or k.startswith("n")})
        print(f"  lowest price now {_g(now)} g. Typical range over {len(df)} hours (spikes and dips left out): "
              f"10%-90% = {_g(b['p10'])} - {_g(b['p90'])}, median {_g(b['p50'])}. Right now is at the {b['now_percentile']:.0%} mark of that range.")
        print(f"  {b['n_spike_hours']} spike hours and {b['n_dip_hours']} dip hours were left out (short jumps or drops that are not the normal level).")
        if b["p10"] > 0 and b["p90"] / b["p10"] > 3.0:
            print("  WARNING: the price lived at very different levels in this window (a lasting move, not a short spike). "
                  "The range above mixes them; trust 'right now', the wall and the TSM-style value more than the range.")
        tsm = tsm_style_market_value(book_i) / 10000.0
        r["tsm_style_value"] = tsm
        print(f"  TSM-style market value from this book: {_g(tsm)} g (their 'cheapest 30%' method, without their 14-day smoothing)")
        rv = _rates_from_series(df, args.window_hours)
        if rv is None:
            print("  not enough observed time for a sales rate")
            rows.append(r)
            continue
        vel, hrs = rv
        r.update({f"per_day_{k}": v * 24 for k, v in vel.items()})
        print(f"  our sales estimate (units per day, low / best / high): {vel['low'] * 24:,.0f} / {vel['est'] * 24:,.0f} / {vel['high'] * 24:,.0f} "
              f"(a floor; last {hrs:.0f} observed hours)")
        w = wall_report(book_i, vel, horizon_hours=args.horizon_hours, trade_units=args.trade_units)
        r.update({k: v for k, v in w.items() if k != "horizon_hours"})
        print(f"  PRICE WALL ({args.horizon_hours:g}-hour horizon): at the low / best / high sales rate = "
              f"{_g(w['wall_low'] / 10000)} / {_g(w['wall_est'] / 10000)} / {_g(w['wall_high'] / 10000)} g "
              f"({w['units_est']:,} units at or below the best wall)")
        print(f"  a listing AT the best wall waits (newest-first, no new cheaper listings): {_h(w['wait_high'])} / {_h(w['wait_est'])} / {_h(w['wait_low'])} "
              f"at high / best / low sales rate")
        print(f"  time to FULLY sell {args.trade_units:,.0f} units listed at the best wall (units ahead + your own): "
              f"{_h(w['sell_hours_high'])} / {_h(w['sell_hours_est'])} / {_h(w['sell_hours_low'])} at high / best / low sales rate")
        print(f"  deep market for a {args.trade_units:,.0f}-unit trade: {'YES' if w['deep_market'] else 'no'}; "
              f"does the exact sales rate change what you would do? {'YES, it matters' if w['velocity_matters'] else 'no (fully sold inside 12 h even at the low rate)'}")
        rs = reach_stats(df, w["wall_est"] / 10000.0, horizon_hours=12.0)
        if 0 < rs["n"] < 10:
            print(f"  history: only {rs['n']} starting hours with the price below {_g(w['wall_est'] / 10000)} g, too few to say how often it gets there")
        elif rs["n"]:
            print(f"  history: the lowest price reached {_g(w['wall_est'] / 10000)} g within 12 h in {rs['share_reached']:.0%} of {rs['n']} starting hours"
                  + (f" (median {rs['median_hours']:.1f} h)" if rs["share_reached"] == rs["share_reached"] and rs["median_hours"] == rs["median_hours"] else ""))
            r.update(reach_share_12h=rs["share_reached"], reach_n=rs["n"])
        shelves = find_shelf(book_i, max_cost_gold=args.max_position_gold)
        if shelves:
            sh = shelves[0]
            print(f"  FLIP SHELF in this snapshot: {sh['units']:,} units cost {_g(sh['cost_gold'])} g (avg {_g(sh['avg_price_gold'])}), next price {_g(sh['next_price_gold'])} g "
                  f"(+{sh['gap']:.0%}); net margin after the 5% cut {sh['net_margin']:.0%}, about {_g(sh['net_profit_gold'])} g. Check the live book: fast items lose these in minutes.")
            r.update(shelf_units=sh["units"], shelf_net_margin=sh["net_margin"])
        else:
            print("  no flip shelf in this snapshot (needs a 30%+ gap above cheap units, cost under the cap, 25%+ net margin)")
        rows.append(r)
    os.makedirs(args.out, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "market.csv"), index=False)
    return out


def _grid(pivot, label, fmt):
    print(f"    {label} (Mon..Sun x 4-hour blocks starting at the hour shown, US Eastern)")
    print("          " + "".join(f"{h:>7d}h" for h in pivot.columns))
    for dname, row in pivot.iterrows():
        print(f"    {dname:<5} " + "".join(f"{fmt(v):>8}" for v in row))


def run_profile(store, args, reader=None, names_fn=fetch_names):
    """Hour-of-day table and a day-of-week grid of price and listed quantity for the item ids you give."""
    ids = [int(x) for x in str(args.ids).replace(" ", "").split(",") if x]
    snaps = _load_all(store, args, reader)
    _print_holes(snaps)
    names = names_fn(ids)
    series = item_series(snaps, ids)
    os.makedirs(args.out, exist_ok=True)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i in ids:
        df = series.get(i)
        print(f"\n== ITEM {i}: {names.get(i, '')} ==")
        if df is None or df.empty:
            print("  not in the loaded data")
            continue
        sp = flag_spikes(df["floor_gold"])
        bl = flag_blips(df["units_listed"])
        hod = profile(df, sp, by="none", blips=bl)
        weeks = int(hod["weeks"].iloc[0])
        print(f"  {len(df)} hourly observations over {weeks} calendar week(s) of data; spike and dip hours ({int((sp != 0).sum())}) and "
              f"feed-blip hours ({int(bl.sum())}) are left out. "
              f"{'DRAFT: ' if weeks < 3 else ''}the day-of-week grid has about {max(1, round(len(df) / (7 * 24)))} observation(s) per slot.")
        show = hod[["block", "n_obs", "price_median", "price_vs_level", "units_median", "sold_per_hour", "sold_share"]].copy()
        show["price_vs_level"] = (show["price_vs_level"] * 100).round(1)
        show["sold_share"] = (show["sold_share"] * 100).round(1)
        show = show.round(2)
        show.columns = ["hour", "obs", "price_raw", "price_vs_day_%", "units_listed_med", "est_sold_per_hr", "share_of_sales_%"]
        print("  HOUR OF DAY (US Eastern). price_vs_day_% compares each hour with that day's own level, so a week-long trend does not")
        print("  look like an hourly effect; share_of_sales_% is each hour's part of the day's estimated sales (hourly sales are lumpy).")
        print(show.to_string(index=False))
        how = profile(df, sp, by="dow", block_hours=4, blips=bl)
        how.to_csv(os.path.join(args.out, f"profile_{i}.csv"), index=False)
        pv = how.pivot(index="dow", columns="block", values="price_vs_level") * 100.0
        pv.index = [days[d] for d in pv.index]
        _grid(pv, "price vs that day's own level (%)", lambda v: "." if v != v else f"{v:+.0f}")
        base = float(how["units_median"].median())
        if base and base == base:
            pv = (how.pivot(index="dow", columns="block", values="units_median") / base - 1.0) * 100.0
            pv.index = [days[d] for d in pv.index]
            _grid(pv, "listed quantity vs its overall median (%)", lambda v: "." if v != v else f"{v:+.0f}")
    return series


def run_reconcile(store, args, reader=None, names_fn=fetch_names):
    """Compare the buy-and-list experiment log (a CSV in the repo) with the hourly snapshots."""
    from engine.reconcile import parse_spent, read_log, reconcile, split_flips, verdict_text
    log = read_log(args.log)
    log, flips_log = split_flips(log)
    if len(flips_log):
        print(f"\n== FLIPS IN THE LOG ({len(flips_log)} rows; kept out of the lane experiment below) ==")
        fl = flips_log.copy()
        fl["posted_ET"] = fl["posted_utc"].map(_et)
        fl["spent_g"] = fl["note"].map(parse_spent)
        print(fl[["posted_ET", "item_id", "list_price", "quantity", "units_sold", "units_expired", "spent_g", "note"]].round(2).to_string(index=False))
    if log.empty:
        print(f"No usable rows in {args.log} (example rows and rows without a price or quantity are skipped).")
        return pd.DataFrame()
    for w in log.attrs.get("warnings", []):
        print(f"  CHECK THE LOG: {w}")
    first, last = log["posted_utc"].min(), log["posted_utc"].max()
    print(f"Log: {len(log)} listings, first posted {_et(first)} ET, last {_et(last)} ET. Loading the snapshots around them.")
    snaps = _load_all(store, args, reader)
    _print_holes(snaps)
    names = names_fn(sorted(log["item_id"].unique()))
    per, summ = reconcile(snaps, log)
    show = per.copy()
    show["posted_ET"] = show["posted_utc"].map(_et)
    show["name"] = show["item_id"].map(lambda i: str(names.get(int(i), ""))[:18])
    show["still_h"] = show["still_there_h"] if "still_there_h" in show.columns else np.nan
    cols = ["posted_ET", "name", "lane", "vs_floor_silver", "price_gold", "quantity", "units_sold", "units_expired", "appeared_share", "front_share",
            "still_h", "observed_hours_to_clear", "pred_lifo_h_est", "status"]
    cols = [c for c in cols if c in show.columns]
    t = show[cols].copy()
    for c in ("appeared_share", "front_share"):
        if c in t:
            t[c] = (t[c] * 100).round(0)
    t = t.rename(columns={"appeared_share": "seen_%", "observed_hours_to_clear": "gone_h", "pred_lifo_h_est": "predicted_h",
                          "units_sold": "sold", "units_expired": "expired", "vs_floor_silver": "vs_floor_s", "front_share": "at_front_%",
                          "quantity": "qty", "price_gold": "price"}).round(2)
    print("\n== EACH LISTING ==")
    print("   vs_floor_s   your price against the lowest price just before posting, in silver")
    print("   seen_%       units that appeared at your price in the next snapshot / units you posted (blank if older units already sat at your price)")
    print("   at_front_%   share of the following snapshots in which nothing was cheaper than you")
    print("   still_h / gone_h   for listings that were first in line: still there at this many hours, gone by this many (snapshots are hourly, so it is a window)")
    print("   predicted_h  hours to sell all units at the recent estimated sales rate")
    print(t.to_string(index=False))
    print("\n== SUMMARY ==")
    for line in verdict_text(summ):
        print("  " + line)
    os.makedirs(args.out, exist_ok=True)
    per.to_csv(os.path.join(args.out, "reconcile.csv"), index=False)
    return per


def run_flips(store, args, reader=None, names_fn=fetch_names):
    snaps = _load_all(store, args, reader)
    _print_holes(snaps)
    ev = detect_flips(snaps)
    print("\n== MARKET FLIPS ==")
    if ev.empty:
        print("  none found")
        return ev
    buy = ev[ev["verdict"] == "flip / buyout"]
    likely = ev[ev["likely_flip"]]
    print("  How many price jumps survive each test (each test is stricter):")
    print(f"    1. lowest price rose 30%+ and most cheap units vanished: {len(ev):,}")
    print(f"    2. ...and those units could not have just expired:        {len(buy):,}")
    print(f"       (ordinary buyers eating a thin cheap price level look like this too)")
    print(f"    3. ...and far bigger than the item's normal hour, and relisted near the new price: {len(likely):,}  <- likely flips")
    if likely.empty:
        print("  No likely flips found.")
    else:
        print(f"\n  LIKELY FLIPS ({likely['item_id'].nunique()} different items; median jump {likely['jump'].median():.2f}x, "
              f"median units bought out {likely['units_removed'].median():.0f})")
        for h in (3, 6, 12, 24):
            share, n = revert_share_within(likely, h)
            if n:
                print(f"    price back near the old level within {h:>2} h: {share:.0%} of the {n} flips we watched that long")
        sold = likely["sold_est_while_elevated"].dropna()
        if len(sold):
            print(f"    units the market bought while the price was up (estimate, watched time only): median {sold.median():.0f}, "
                  f"25th-75th pct {sold.quantile(.25):.0f}-{sold.quantile(.75):.0f}")
        cut = likely["ended_by"].value_counts().to_dict()
        print(f"    how each ended: {cut}  (hole = our data stopped before we could tell; end_of_data = still up at the last snapshot)")
    print("  These describe what prices did. They are NOT profit: nothing here says what a flipper paid or whether their units sold.")
    show = (likely if len(likely) else buy).sort_values("units_removed", ascending=False).head(15).copy()
    names = names_fn(show["item_id"].tolist())
    show.insert(0, "when_et", show["t1"].map(_et))
    show.insert(1, "name", show["item_id"].map(lambda i: names.get(int(i), "")))
    cols = ["when_et", "item_id", "name", "floor_before_gold", "floor_after_gold", "jump", "units_removed", "size_vs_normal",
            "relisted_share", "ended_by", "followup_hours", "sold_est_while_elevated"]
    print(f"\n  largest {'likely flips' if len(likely) else 'buyouts'}:")
    print(show[cols].round(2).to_string(index=False))
    os.makedirs(args.out, exist_ok=True)
    ev.to_csv(os.path.join(args.out, "flips.csv"), index=False)
    return ev


def _names_from_frame(d):
    """{item_id: name} from the dictionary table (columns id/item_id and name/item_name)."""
    idc = next(c for c in d.columns if c.lower() in ("item_id", "id"))
    nmc = next(c for c in d.columns if c.lower() in ("name", "item_name"))
    return {int(i): str(n) for i, n in zip(d[idc], d[nmc]) if n == n and n is not None}


def _dictionary_names():
    """Item names from system/item_dictionary.parquet. build_dictionary.py writes it to Backblaze (a tiny file, so reading it
    is negligible against the download cap); the Hugging Face copy is tried first in case one exists. Empty dict if neither works."""
    tok = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        return _names_from_frame(pd.read_parquet(hf_hub_download(HF_REPO, "system/item_dictionary.parquet", repo_type="dataset", token=tok)))
    except Exception:
        pass
    kid, key = os.environ.get("B2_KEY_ID"), os.environ.get("B2_APP_KEY")
    if kid and key:
        try:
            import io
            import boto3
            s3 = boto3.client("s3", endpoint_url="https://s3.us-east-005.backblazeb2.com", aws_access_key_id=kid, aws_secret_access_key=key)
            body = s3.get_object(Bucket=B2_BUCKET, Key="system/item_dictionary.parquet")["Body"].read()
            names = _names_from_frame(pd.read_parquet(io.BytesIO(body)))
            print(f"(item names: {len(names):,} read from Backblaze)")
            return names
        except Exception as e:
            print(f"(item names skipped: {e})")
    else:
        print("(item names skipped: no copy on Hugging Face and no Backblaze keys in this run)")
    return {}


def run_derive(store, args, reader=None, names_fn=None):
    """Build the small tables the front end reads (items, item_hourly, profile, flips, calibration) into --out/derived."""
    import time
    from engine.derive import DeriveConfig, derive
    from engine.reconcile import item_evidence, read_log, reconcile, summarize
    from engine import __version__ as engine_version
    print(f"[engine {engine_version}]")
    t_start = time.time()
    snaps = _load_all(store, args, reader)
    print(f"[timing] snapshots loaded in {(time.time() - t_start) / 60:.1f} min")
    _print_holes(snaps)
    names = (names_fn or (lambda ids: _dictionary_names()))([])
    watch = [int(x) for x in str(args.ids).replace(" ", "").split(",") if x]
    evidence, exp, per_exp = None, None, None
    if os.path.exists(args.log):
        try:
            log = read_log(args.log)
            if not log.empty:
                per, exp = reconcile(snaps, log)
                per_exp = per
                evidence = item_evidence(per)
                print(f"Experiment log used: {len(log)} listings, {0 if evidence is None else len(evidence)} items with checks.")
        except Exception as e:
            print(f"(experiment log skipped: {e})")
    from engine.shelves import ShelfConfig
    shelf = ShelfConfig(max_units=int(getattr(args, "max_units", 300)), max_cost_gold=float(getattr(args, "max_position_gold", 300.0)))
    cfg = DeriveConfig(window_hours=args.window_hours, horizon_hours=args.horizon_hours, shelf=shelf)
    t_calc = time.time()
    out = derive(snaps, cfg, watch=watch, evidence=evidence, experiment_summary=exp, names=names)
    print(f"[timing] tables calculated in {(time.time() - t_calc) / 60:.1f} min")
    d = os.path.join(args.out, "derived")
    os.makedirs(d, exist_ok=True)
    if per_exp is not None and len(per_exp):
        out["experiment"] = per_exp.drop(columns=[c for c in per_exp.columns if c.startswith("_")]).copy()
    for k, df in out.items():
        if len(df):
            df.to_parquet(os.path.join(d, f"{k}.parquet"), index=False)
        print(f"  {k:12s} {len(df):>7,} rows")
    sh = out["shelves"]
    print(f"\n== FLIP SHELVES (limits: {shelf.max_units} units, {shelf.max_cost_gold:.0f}g per flip; tests: price, speed, size, margin, regime, book) ==")
    try:                                       # a display problem must never stop the upload below
        needed = ["long_median_gold", "daily_est", "test_regime"]
        if len(sh) and any(c not in sh.columns for c in needed):
            print(f"  WARNING: the shelves table lacks {[c for c in needed if c not in sh.columns]}. Files from different engine versions are mixed in this "
                  f"repo (running engine {engine_version}). Re-install the latest full engine zip; the app will show blanks until then.")
        cands = sh[sh["verdict"] != "no"].head(12) if len(sh) else sh
        if len(cands):
            show = cands.assign(item=cands["name"].where(cands["name"] != "", cands["item_id"].astype(str)))
            cols = ["item", "verdict", "units", "avg_price_gold", "target_gold", "net_margin", "net_margin_stress", "profit_gold",
                    "sell_hours_low", "sell_hours_est", "sell_hours_high", "long_median_gold", "daily_est", "test_price", "test_speed",
                    "test_size", "test_margin", "test_regime", "test_book"]
            print(show[[c for c in cols if c in show.columns]].round(3).to_string(index=False))
            print("  profit_gold is at the target price, before anyone undercuts; net_margin_stress assumes selling 10% under it.")
        else:
            print("  none pass right now (that is normal; shelves are rare)")
    except Exception as e:
        print(f"  (shelf listing skipped: {e})")
    cal = out["calibration"].iloc[0]
    print(f"Confidence: {cal['n_high']} High, {cal['n_medium']} Medium, {cal['n_low']} Low; median range (high/low) {cal['median_range_ratio']:.2f}x")
    if args.upload:
        _upload_derived(d, out["calibration"], out["shelves"])
    return out


def _upload_derived(folder, cal_row, shelves=None):
    """Push the derived tables to engine/ in the HF dataset repo, and append this run to the confidence history."""
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("--upload needs HF_TOKEN")
    api = HfApi(token=tok)
    hist_path = os.path.join(folder, "calibration_history.parquet")
    try:
        old = pd.read_parquet(hf_hub_download(HF_REPO, "engine/calibration_history.parquet", repo_type="dataset", token=tok))
        hist = pd.concat([old, cal_row], ignore_index=True)
    except Exception:
        hist = cal_row.copy()
    hist = hist.drop_duplicates(subset=["data_last_ts", "engine_version"], keep="last")
    hist.to_parquet(hist_path, index=False)
    # Keep what the screener recommended at each run (candidates and weak shelves, 21 days), so flips you actually make can be
    # compared later with what the tool said at the time. The shelves table itself is overwritten every run.
    if shelves is not None and len(shelves):
        snap = shelves[shelves["verdict"] != "no"].head(300).copy()
        snap.insert(0, "data_last_ts", cal_row["data_last_ts"].iloc[0])
        try:
            old = pd.read_parquet(hf_hub_download(HF_REPO, "engine/shelves_history.parquet", repo_type="dataset", token=tok))
            snap = pd.concat([old, snap], ignore_index=True)
        except Exception:
            pass
        snap = snap.drop_duplicates(subset=["data_last_ts", "item_id", "target_gold"], keep="last")
        snap = snap[pd.to_datetime(snap["data_last_ts"]) > pd.to_datetime(snap["data_last_ts"]).max() - pd.Timedelta(days=21)]
        snap.to_parquet(os.path.join(folder, "shelves_history.parquet"), index=False)
    api.upload_folder(folder_path=folder, path_in_repo="engine", repo_id=HF_REPO, repo_type="dataset",
                      commit_message="derived layer refresh")
    print(f"Uploaded {len(os.listdir(folder))} files to engine/ in {HF_REPO} (one commit): {', '.join(sorted(os.listdir(folder)))}")


def run_backtest(store, args, reader=None, names_fn=None):
    """Replay the flip screener on the snapshots we hold and see what happened after each shelf (no gold at risk)."""
    from engine.classifier import aggregate_rates, classify_series
    from engine.shelf_backtest import backtest_shelves, summarize_backtest
    from engine.shelves import ShelfConfig
    snaps = _load_all(store, args, reader)
    _print_holes(snaps)
    cs = sorted((s for s in snaps if s.kind == "commodity"), key=lambda s: s.ts)
    rates = aggregate_rates(classify_series(cs), cs[-1].ts, args.window_hours)
    flo = cs[-1].book.groupby("item_id")["unit_price"].min() / 10000.0
    rates = rates[(rates["daily_high"] >= 20.0) & rates["item_id"].isin(flo.index)]
    rates = rates.assign(flow=rates["daily_est"].values * flo.reindex(rates["item_id"]).fillna(0.0).values)
    keep = rates.sort_values("flow", ascending=False)["item_id"].head(3000).tolist()
    shelf = ShelfConfig(max_units=int(getattr(args, "max_units", 300)), max_cost_gold=float(getattr(args, "max_position_gold", 300.0)))
    bt = backtest_shelves(cs, shelf, item_ids=keep, step=2, min_history_hours=48.0)
    os.makedirs(args.out, exist_ok=True)
    if bt.empty:
        print("No shelves found in the replay.")
        return bt
    bt.to_csv(os.path.join(args.out, "shelf_backtest.csv"), index=False)
    summ = summarize_backtest(bt)
    f = bt[bt["first_seen"]]
    print(f"\n== SHELF BACKTEST: {len(f):,} separate shelves seen ({len(bt):,} hourly sightings), "
          f"{_et(bt['t0'].min())} to {_et(bt['t0'].max())} ET ==")
    print("  reached_Nh      share whose relist price was reached by the lowest price within N hours (known outcomes only)")
    print("  median_hours    hours to get there (when it did)")
    print("  floor_ratio_4h  lowest price 4 h later / relist price (1.0 = right at it; below 1 = not there yet)")
    print("  undercut_risk   share where new cheaper listings worth 50%+ of your units appeared within 2 h")
    show = summ.copy()
    for c in ("reached_4h", "reached_12h", "undercut_risk"):
        show[c] = (show[c] * 100).round(0)
    print(show.round(2).to_string(index=False))
    print("  Spike flags are off in the replay and the cheap units stay in the book, so 'reached' is a cautious reading of the relist price.")
    summ.to_parquet(os.path.join(args.out, "backtest_summary.parquet"), index=False)
    if args.upload:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            raise SystemExit("--upload needs HF_TOKEN")
        HfApi(token=tok).upload_file(path_or_fileobj=os.path.join(args.out, "backtest_summary.parquet"), path_in_repo="engine/backtest_summary.parquet",
                                     repo_id=HF_REPO, repo_type="dataset", commit_message="shelf backtest summary")
        print("Uploaded backtest_summary.parquet to engine/.")
    return bt


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "files", "inspect", "rates", "market", "profile", "pick", "order", "item", "flips", "reconcile", "derive", "backtest"])
    ap.add_argument("--store", choices=["hf", "b2", "local"], default="hf")
    ap.add_argument("--root", default="lake", help="folder for --store local")
    ap.add_argument("--days", type=int, default=3, help="how many days back from --end")
    ap.add_argument("--end", default=None, help="last UTC date, YYYY-MM-DD (default: today)")
    ap.add_argument("--cache", default=".lake_cache")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-gb", type=float, default=0.8, help="B2 only: stop before downloading more than this per UTC day")
    ap.add_argument("--max-position-gold", type=float, default=150.0)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--ids", default="", help="item ids, comma separated (for the item command)")
    ap.add_argument("--min-floor-gold", type=float, default=1.0)
    ap.add_argument("--max-units", type=int, default=300)
    ap.add_argument("--horizon-hours", type=float, default=4.0, help="market: how many hours of sales the price wall must cover")
    ap.add_argument("--trade-units", type=float, default=500.0, help="market: the biggest trade you would make (for the deep-market test)")
    ap.add_argument("--log", default="experiment_log.csv", help="reconcile: the experiment log CSV")
    ap.add_argument("--window-hours", type=float, default=72.0, help="pick: how far back item statistics look")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="pick: share of the window that must have been observed")
    ap.add_argument("--upload", action="store_true", help="derive: push the tables to engine/ in the HF repo")
    args = ap.parse_args(argv)
    store = make_store(args)
    return {"plan": run_plan, "files": run_files, "inspect": run_inspect, "rates": run_rates, "market": run_market, "profile": run_profile, "pick": run_pick, "order": run_order, "item": run_item, "flips": run_flips, "reconcile": run_reconcile, "derive": run_derive, "backtest": run_backtest}[args.command](store, args)


if __name__ == "__main__":
    main()
