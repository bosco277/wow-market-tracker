"""Command line for the loader and the item picker.

    python tools/lake.py plan --days 3          # how many files / MB would this touch? downloads nothing
    python tools/lake.py files --days 5         # which files exist, day by day (downloads nothing)
    python tools/lake.py inspect --days 4       # downloads each DAILY file and shows what is inside (timestamps, empty cells)
    python tools/lake.py rates --ids 13466,42253 --days 4   # our estimated units sold per day for those items
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

import pandas as pd  # noqa: E402

from engine.loader import ByteBudget, NEEDED_COLUMNS, RangeFile, normalize_frame, read_parquet_file, B2Store, HFStore, LocalStore, MB, coverage_report, day_inventory, choose_files, load_snapshots, plan_load  # noqa: E402
from engine.itemview import detect_flips, gaps_in, item_history, revert_share_within  # noqa: E402
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


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "files", "inspect", "rates", "pick", "order", "item", "flips"])
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
    ap.add_argument("--window-hours", type=float, default=72.0, help="pick: how far back item statistics look")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="pick: share of the window that must have been observed")
    args = ap.parse_args(argv)
    store = make_store(args)
    return {"plan": run_plan, "files": run_files, "inspect": run_inspect, "rates": run_rates, "pick": run_pick, "order": run_order, "item": run_item, "flips": run_flips}[args.command](store, args)


if __name__ == "__main__":
    main()
