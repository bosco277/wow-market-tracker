"""Buy-window test on the permanent hourly history (engine/hourly/ in the private dataset).

    python tools/buy_windows.py                          # all stored days, 3-hour window, floor price
    python tools/buy_windows.py --days 45 --window 2     # only the last 45 stored days, 2-hour window
    python tools/buy_windows.py --price p15              # depth price (what the cheapest 15% of units cost) instead of the single cheapest listing

It reads the day files written by the daily chain (tools/lake.py daily --upload), runs engine/windows.py and prints a plain report. The report is saved to out/buy_windows.txt
(a .txt on purpose: lines starting with "==" turn into #ERROR! when a spreadsheet opens them). Item groups come from engine/item_groups.parquet if it is there.
Environment: HF_TOKEN (the private dataset). It only reads; nothing is uploaded.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from engine import windows as W  # noqa: E402

HF_REPO = "bosco277/wow-market-data-lake"
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
    parts = [pd.read_parquet(hf_hub_download(HF_REPO, path, repo_type="dataset", token=tok)) for _, path in files]
    return pd.concat(parts, ignore_index=True), [d for d, _ in files]


def run(args, read_hourly=None, read_table=None):
    """Load, analyse, print, save. read_hourly(days) -> (frame, [days]) and read_table(name) -> frame can be replaced for tests. Returns the analysis dict."""
    read_hourly = read_hourly or read_hourly_hf
    if read_table is None:
        from lake import _read_engine_table as read_table
    hl, days = read_hourly(args.days)
    if hl is None or hl.empty:
        print("No hourly day files found under engine/hourly/. Run the Daily chain (it now writes them), then backfill earlier days: run it again with days=7 and an earlier end date.")
        return {}
    print(f"Loaded {len(days)} stored days ({days[0]} to {days[-1]}), {len(hl):,} hourly rows.")
    groups = read_table("item_groups")
    cfg = W.WindowConfig(price_col=args.price, window_hours=args.window, cut=args.cut)
    res = W.analyze(hl, groups if groups is not None and len(groups) else None, cfg)
    text = W.format_report(res)
    print(text)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "buy_windows.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=0, help="use only the last N stored days (0 = all)")
    ap.add_argument("--window", type=int, default=3, help="buying window length in hours (1-6)")
    ap.add_argument("--price", choices=["floor_gold", "p15"], default="floor_gold", help="price measure: floor_gold (cheapest single listing) or p15 (depth price)")
    ap.add_argument("--cut", type=float, default=0.05, help="AH cut for the resale check (retail 0.05)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    if not 1 <= args.window <= 6:
        raise SystemExit("--window must be between 1 and 6 hours")
    run(args)


if __name__ == "__main__":
    main()
