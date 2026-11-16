"""Buy-window test on the permanent hourly history (engine/hourly/ in the private dataset).

    python tools/buy_windows.py                          # all stored days, 3-hour window, floor price
    python tools/buy_windows.py --days 45 --window 2     # only the last 45 stored days, 2-hour window
    python tools/buy_windows.py --lookback 14            # dips measured against each item's last 14 days instead of 7
    python tools/buy_windows.py --only targets --days 60 --upload   # what the Daily chain runs: refresh the front end's Buy targets tab
    python tools/buy_windows.py --price p15              # depth price (what the cheapest 15% of units cost) instead of the single cheapest listing

It reads the day files written by the daily chain (tools/lake.py daily --upload), runs engine/windows.py (time of day, weekday, resale check) and engine/dips.py (buying dips against an
item's own range) and prints one plain report. The report is saved to out/buy_windows.txt
(a .txt on purpose: lines starting with "==" turn into #ERROR! when a spreadsheet opens them) and the shopping list to out/buy_targets.csv (engine/targets.py: three buy-below
prices per item, how often each is reached, units usually near the floor). Item groups come from engine/item_groups.parquet and daily sales (for days to sell) from engine/item_daily.parquet, if they are there.
Environment: HF_TOKEN (the private dataset). It only reads; nothing is uploaded.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from engine import dips as Dp  # noqa: E402
from engine import targets as Tg  # noqa: E402
from engine import windows as W  # noqa: E402

HF_REPO = "bosco277/wow-market-data-lake"
NEEDED = ["item_id", "ts", "floor_gold", "p15", "u5", "blip", "flag_wipe", "flag_long_gap"]
DAY_FILE = re.compile(r"^engine/hourly/h_(\d{4}-\d{2}-\d{2})\.parquet$")


def list_day_files(names):
    """Sorted [(day, repo path)] for the hourly day files among `names`."""
    out = []
    for n in names:
        m = DAY_FILE.match(n)
        if m:
            out.append((m.group(1), n))
    return sorted(out)


def read_hourly_hf(days=0):
    """Concatenate the stored hourly day files (the last `days` of them, 0 = all) from the private dataset."""
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("buy_windows needs HF_TOKEN (the private dataset).")
    files = list_day_files(HfApi(token=tok).list_repo_files(HF_REPO, repo_type="dataset"))
    if days and len(files) > days:
        files = files[-days:]
    if not files:
        return pd.DataFrame(), []
    parts = [pd.read_parquet(hf_hub_download(HF_REPO, path, repo_type="dataset", token=tok), columns=NEEDED) for _, path in files]     # only the columns the tests use: memory
    return pd.concat(parts, ignore_index=True), [d for d, _ in files]


def upload_tables(tables, extra=""):
    """Write the buy_* tables as parquet into a temporary folder and upload them to engine/ of the private dataset in ONE commit (the front end reads them there)."""
    import tempfile
    from lake import _upload_folder_to_engine
    folder = tempfile.mkdtemp(prefix="buy_upload_")
    for name, df in tables.items():
        df.to_parquet(os.path.join(folder, f"{name}.parquet"), index=False)
    _upload_folder_to_engine(folder, "buy targets" + extra)


def run(args, read_hourly=None, read_table=None, upload=None):
    """Load, analyse, print, save (and upload when asked). read_hourly(days) -> (frame, [days]), read_table(name) -> frame and upload(tables) can be replaced for tests.
    --only targets (what the Daily chain runs) skips the clock test, the weekday test and the dip test and does only the buy-below prices. Returns the analysis dict."""
    read_hourly = read_hourly or read_hourly_hf
    if read_table is None:
        from lake import _read_engine_table as read_table
    hl, days = read_hourly(args.days)
    if hl is None or hl.empty:
        print("No hourly day files found under engine/hourly/. Run the Daily chain (it now writes them), then backfill earlier days: run it again with days=7 and an earlier end date.")
        return {}
    print(f"Loaded {len(days)} stored days ({days[0]} to {days[-1]}), {len(hl):,} hourly rows.")
    groups = read_table("item_groups")
    groups = groups if groups is not None and len(groups) else None
    cfg = W.WindowConfig(price_col=args.price, window_hours=args.window, cut=args.cut)
    res, parts = {}, []
    if args.only == "all":
        res = W.analyze(hl, groups, cfg)
        parts.append(W.format_report(res))
        sales = read_table("item_daily")
        dips = Dp.analyze_dips(hl, groups, cfg, Dp.DipConfig(lookback_days=args.lookback, cut=args.cut, sale_days=cfg.sale_days, min_hours_day=cfg.min_hours_day, min_items=cfg.min_items,
                                                              boot=cfg.boot, block=cfg.block, seed=cfg.seed),
                               sales if sales is not None and len(sales) else None)
        res["dips"] = dips
        parts.append(Dp.format_dips(dips))
    tg = Tg.analyze_targets(hl, groups, cfg, Tg.TargetConfig(lookback_days=args.lookback, tier_edges=cfg.tiers, min_items=cfg.min_items))
    res["targets"] = tg
    parts.append(Tg.format_targets(tg))
    text = "\n\n".join(parts)
    print(text)
    os.makedirs(args.out, exist_ok=True)
    if len(tg["list"]):
        tg["list"].to_csv(os.path.join(args.out, "buy_targets.csv"), index=False, float_format="%.4f")
    with open(os.path.join(args.out, "buy_windows.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if args.upload:
        if not len(tg["list"]):
            print("Nothing to upload: no item has enough history for targets yet (the old tables in the dataset are left as they are).")
        else:
            (upload or upload_tables)(Tg.export_tables(tg), f" ({tg['days']} days of data)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=0, help="use only the last N stored days (0 = all)")
    ap.add_argument("--window", type=int, default=3, help="buying window length in hours (1-6)")
    ap.add_argument("--price", choices=["floor_gold", "p15"], default="floor_gold", help="price measure: floor_gold (cheapest single listing) or p15 (depth price)")
    ap.add_argument("--lookback", type=int, default=7, help="days of an item's own history that set its dip trigger line (the bottom fifth of the last N days)")
    ap.add_argument("--only", choices=["all", "targets"], default="all", help="all = the full test report; targets = only the buy-below prices (what the Daily chain runs)")
    ap.add_argument("--upload", action="store_true", help="write the buy_* tables to engine/ in the private dataset for the front end (needs HF_TOKEN)")
    ap.add_argument("--cut", type=float, default=0.05, help="AH cut for the resale check (retail 0.05)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    if not 1 <= args.window <= 6:
        raise SystemExit("--window must be between 1 and 6 hours")
    run(args)


if __name__ == "__main__":
    main()
