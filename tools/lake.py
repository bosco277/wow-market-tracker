"""Command line for the loader and the item picker.

    python tools/lake.py plan --days 3          # how many files / MB would this touch? downloads nothing
    python tools/lake.py pick --days 3          # load recent commodity data, check it, write the experiment shortlist
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

from engine.loader import B2Store, HFStore, LocalStore, MB, coverage_report, load_snapshots, plan_load  # noqa: E402
from engine.itemview import detect_flips, item_history  # noqa: E402
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

    rep = coverage_report(res.snapshots)
    print("\n== DATA HEALTH ==")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    if len(res.snapshots) < 3:
        print("Not enough snapshots to continue.")
        return None
    mins = publish_minutes(res.snapshots)
    print("\n== WHEN BLIZZARD'S DATA REFRESHES (minute of the hour -> how many snapshots) ==")
    if int(mins.sum()) < 12:
        print(f"  Not enough yet: only {int(mins.sum())} snapshots carry Blizzard's own publish time "
              f"(the rest only record when we fetched). Re-run once there are 24 or more.")
    else:
        print("  " + ", ".join(f":{int(m):02d} x{int(c)}" for m, c in mins.items()))

    cfg = PickerConfig(max_position_gold=args.max_position_gold, n_items=args.n_items,
                       min_floor_gold=args.min_floor_gold, max_units=args.max_units)
    stats = item_stats(res.snapshots, cfg=cfg)
    short = pick(stats, cfg)
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
        show = h[["when_et", "floor_gold", "units_at_floor", "units_within_10pct", "units_listed", "new_units",
                  "sold_low", "sold_est", "sold_high"]].round(2)
        print(show.to_string(index=False))
    return snaps


def run_flips(store, args, reader=None, names_fn=fetch_names):
    snaps = _load_all(store, args, reader)
    ev = detect_flips(snaps)
    print("\n== MARKET FLIPS: lowest price jumped 30%+ and most of the cheap units vanished ==")
    if ev.empty:
        print("  none found")
        return ev
    flips = ev[ev["verdict"] == "flip / buyout"]
    print(f"  {len(ev)} jumps found: {len(flips)} look like buyouts, {len(ev) - len(flips)} are probably expiry or cancels")
    if len(flips):
        rev = flips["revert_hours"].dropna()
        print(f"  buyouts whose price came back down within the data: {len(rev)} of {len(flips)}"
              + (f"; median {rev.median():.1f} h, 25th pct {rev.quantile(.25):.1f} h, 75th pct {rev.quantile(.75):.1f} h" if len(rev) else ""))
        print(f"  median price jump: {flips['jump'].median():.2f}x; median units bought out: {flips['units_removed'].median():.0f}")
        print(f"  median units sold while elevated (estimate): {flips['sold_est_while_elevated'].median():.0f}")
    names = names_fn(flips.sort_values("t1", ascending=False).head(25)["item_id"].tolist())
    show = flips.sort_values("t1", ascending=False).head(25).copy()
    show.insert(0, "when_et", show["t1"].map(_et))
    show.insert(1, "name", show["item_id"].map(lambda i: names.get(int(i), "")))
    cols = ["when_et", "item_id", "name", "floor_before_gold", "floor_after_gold", "jump", "units_removed",
            "revert_hours", "still_elevated_at_end", "sold_est_while_elevated"]
    print("\n  most recent buyouts:")
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
    ap.add_argument("command", choices=["plan", "pick", "order", "item", "flips"])
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
    args = ap.parse_args(argv)
    store = make_store(args)
    return {"plan": run_plan, "pick": run_pick, "order": run_order, "item": run_item, "flips": run_flips}[args.command](store, args)


if __name__ == "__main__":
    main()
