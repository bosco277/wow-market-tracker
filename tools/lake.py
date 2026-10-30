"""Command line for the loader and the item picker.

    python tools/lake.py plan --days 3          # how many files / MB would this touch? downloads nothing
    python tools/lake.py pick --days 3          # load recent commodity data, check it, write the experiment shortlist

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
    print("  " + ", ".join(f":{int(m):02d} x{int(c)}" for m, c in mins.items()))

    cfg = PickerConfig(max_position_gold=args.max_position_gold, n_items=args.n_items)
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


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "pick"])
    ap.add_argument("--store", choices=["hf", "b2", "local"], default="hf")
    ap.add_argument("--root", default="lake", help="folder for --store local")
    ap.add_argument("--days", type=int, default=3, help="how many days back from --end")
    ap.add_argument("--end", default=None, help="last UTC date, YYYY-MM-DD (default: today)")
    ap.add_argument("--cache", default=".lake_cache")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-gb", type=float, default=0.8, help="B2 only: stop before downloading more than this per UTC day")
    ap.add_argument("--max-position-gold", type=float, default=150.0)
    ap.add_argument("--n-items", type=int, default=6)
    args = ap.parse_args(argv)
    store = make_store(args)
    return (run_plan if args.command == "plan" else run_pick)(store, args)


if __name__ == "__main__":
    main()
