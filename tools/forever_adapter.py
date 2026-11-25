"""Forever adapter check: reads the Forever files through engine/forever.py and says what it found. Nothing is written to any dataset.

    python tools/forever_adapter.py                          # the PvP set (houses 110, 113, 116), the last 2 UTC days, commodities and regular auctions
    python tools/forever_adapter.py --houses 110,114 --days 3
    python tools/forever_adapter.py --houses all --days 1
    python tools/forever_adapter.py --tables                 # also build the house-keyed hourly table (floors, units, flags) and describe it; still nothing is uploaded

It asks first: the files and megabytes a request would touch are counted before anything is downloaded, and it stops (exit code 2) when that is more than --max-files or --max-mb.
Writes ONE plain-text report, out/forever_adapter.txt (a .txt on purpose: lines starting with "=" turn into #ERROR! when a spreadsheet opens them).

What the report answers
  1 WHAT WAS READ        per house and kind: files, snapshots, first and last time, listings in a typical snapshot, repeated auction ids dropped, rows with no price, partial snapshots, unreadable files
  2 KIND CHECK           the kind is set from the source name; how many snapshots of each kind, and whether any source name disagreed with the house or kind asked for
  3 REPEATED AUCTION IDS how many rows were dropped in total, how many of those ids had copies that DISAGREED, and the files they were in
  4 GAPS                 stretches of more than 90 minutes without a snapshot, per source
  5 HOURLY TABLE         (with --tables) rows, items, days and flag shares per house

Environment: HF_TOKEN (read access to the Forever dataset); FOREVER_HF_REPO to name another dataset. The dataset's name is blanked out of everything printed, so no account name reaches a job log.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from engine import forever as FV  # noqa: E402
from engine.loader import HFStore  # noqa: E402

ET = "America/New_York"


def _ts(t):
    return "-" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%m-%d %H:%M")


def _mb(n):
    return f"{n / 1e6:,.0f}"


def window(args, now):
    end = pd.Timestamp(args.end).date() if getattr(args, "end", "") else now.date()
    return end - timedelta(days=max(int(args.days), 1) - 1), end


def table(rows, cols):
    """Plain left-aligned columns."""
    cells = [[str(c) for c in cols]] + [[str(x) for x in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(cols))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in cells)


def read_section(markets):
    rows = []
    for h, m in sorted(markets.items()):
        for sl in m.loads():
            snaps = sl.snapshots
            sizes = sorted(len(s.book) for s in snaps)
            typical = sizes[len(sizes) // 2] if sizes else 0
            r = sl.result
            rows.append((h, sl.kind, sl.files, sl.files_cached, len(snaps), _ts(snaps[0].ts) if snaps else "-", _ts(snaps[-1].ts) if snaps else "-", f"{typical:,}",
                         sl.dups.duplicate_rows, sl.unpriced_rows, len(r.incomplete_dropped), len(r.skipped)))
    return table(rows, ["house", "kind", "files", "cached", "snapshots", "first (UTC)", "last (UTC)", "listings", "repeated ids", "no price", "partial", "unreadable"])


def kind_section(markets):
    n = {"commodity": 0, "auction": 0}
    wrong = 0
    for m in markets.values():
        for sl in m.loads():
            for s in sl.snapshots:
                n[s.kind] += 1
                wrong += int(s.kind != sl.kind or s.meta.get("house") != sl.house)
    lines = [f"commodity snapshots: {n['commodity']:,}   regular-auction snapshots: {n['auction']:,}",
             "The kind comes from the end of the source name (_commodities). The retail reader alone would have called every one of the commodity snapshots 'auction'.",
             f"Snapshots whose kind or house disagreed with what was asked for: {wrong}" + (" (the load would have stopped)" if wrong else "")]
    return "\n".join(lines)


def dup_section(markets, top=15):
    tot_rows = sum(sl.dups.duplicate_rows for m in markets.values() for sl in m.loads())
    tot_conf = sum(sl.dups.conflicting_ids for m in markets.values() for sl in m.loads())
    cached = sum(sl.files_cached for m in markets.values() for sl in m.loads())
    files = sorted(((k, d, c) for m in markets.values() for sl in m.loads() for k, d, c in sl.dups.files), key=lambda x: (-x[2], -x[1], x[0]))
    lines = [f"Repeated auction ids dropped: {tot_rows:,} rows; ids whose copies disagreed (different item, price, quantity or time left): {tot_conf:,}; files with any: {len(files)}"]
    if cached:
        lines.append(f"NOTE: {cached} file(s) came from the local cache and were not read again, so they are not counted here.")
    if tot_rows == 0:
        lines.append("None in this window.")
    elif tot_conf == 0:
        lines.append("Every repeated id was an exact copy, so which copy stays does not matter.")
    else:
        lines.append("Some copies disagreed: the FIRST copy was kept. If this is more than a handful, the choice of copy needs a rule.")
    if files:
        lines.append("")
        lines.append(table([(k, d, c) for k, d, c in files[:top]], ["file", "rows dropped", "ids that disagreed"]))
        if len(files) > top:
            lines.append(f"... and {len(files) - top} more file(s)")
    return "\n".join(lines)


def gap_section(markets):
    rows = []
    for h, m in sorted(markets.items()):
        for sl in m.loads():
            for a, b, hrs in FV.snapshot_gaps(sl.snapshots):
                rows.append((h, sl.kind, _ts(a), _ts(b), f"{hrs:.1f}"))
    if not rows:
        return "No stretch of more than 90 minutes without a snapshot."
    return table(rows, ["house", "kind", "last before (UTC)", "first after (UTC)", "hours"])


def tables_section(hl):
    if hl is None or hl.empty:
        return "No hourly rows (a house needs at least 3 commodity snapshots, and the first and last 12 hours of a window are left out)."
    rows = []
    for h, g in hl.groupby("house"):
        days = pd.to_datetime(g["ts"]).dt.tz_localize("UTC").dt.tz_convert(ET).dt.strftime("%Y-%m-%d").nunique()
        rows.append((h, f"{len(g):,}", f"{g['item_id'].nunique():,}", g["ts"].nunique(), days, f"{g['spike'].mean():.1%}", f"{g['blip'].mean():.1%}", f"{g['flag_wipe'].mean():.1%}",
                     f"{g['flag_long_gap'].mean():.1%}", f"{g['u5'].notna().mean():.0%}"))
    return table(rows, ["house", "rows", "items", "hours", "ET days", "spike", "blip", "wipe", "long gap", "has u5"])


def format_report(r):
    out = ["### forever adapter check", r["head"], ""]
    if r.get("stopped"):
        out += [r["stopped"]]
        return "\n".join(out)
    out += ["### 1 what was read", read_section(r["markets"]), "",
            "### 2 kind check", kind_section(r["markets"]), "",
            "### 3 repeated auction ids", dup_section(r["markets"]), "",
            "### 4 gaps", gap_section(r["markets"])]
    for h, m in sorted(r["markets"].items()):
        for sl in m.loads():
            for k, why in sl.result.skipped[:5]:
                out.append(f"unreadable: {k}: {FV.scrub(why)}")
    if r.get("hourly") is not None:
        out += ["", "### 5 hourly table (house-keyed; not uploaded)", tables_section(r["hourly"])]
    return "\n".join(out)


def run(args, store, now=None, base_reader=None):
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    start, end = window(args, now)
    available = FV.houses_in_store(store)
    if not available:
        return dict(text="### forever adapter check\nThe Forever dataset looks empty or unreachable: its sources file was not found. Check HF_TOKEN (read access) and FOREVER_HF_REPO.", code=1)
    houses = FV.parse_houses(args.houses, available)
    kinds = ("commodity", "auction") if args.kinds == "both" else (args.kinds,)
    plans = FV.plan_houses(store, houses, start, end, kinds, getattr(args, "cache", None))
    files = sum(len(p.files) for _, _, p in plans)
    size = sum(p.total_bytes for _, _, p in plans)
    todo = sum(p.bytes_to_download_upper_bound for _, _, p in plans)
    nodata = sorted({h for h in houses if not any(len(p.files) for hh, _, p in plans if hh == h)})
    head = (f"houses: {', '.join(map(str, houses))}   window: {start} to {end} (UTC days)   kinds: {', '.join(kinds)}\n"
            f"planned: {files:,} files, {_mb(size)} MB ({_mb(todo)} MB to download)" + (f"\nhouses with no files in this window: {', '.join(map(str, nodata))}" if nodata else ""))
    if files == 0:
        return dict(text=format_report(dict(head=head, stopped="Nothing to read in this window.")), code=1)
    if files > args.max_files or size > args.max_mb * 1e6:
        why = (f"STOPPED before downloading: {files:,} files / {_mb(size)} MB is more than the limits ({args.max_files:,} files, {args.max_mb} MB). "
               f"Use fewer houses or days, or raise --max-files / --max-mb.")
        return dict(text=format_report(dict(head=head, stopped=why)), code=2)
    t0 = time.time()
    markets = FV.load_houses(store, [h for h in houses if h not in nodata], start, end, kinds, cache_dir=getattr(args, "cache", None), on_error="skip", progress=None, base_reader=base_reader)
    head += f"\nloaded in {time.time() - t0:.0f} s"
    hourly = None
    if args.tables:
        hourly = FV.hourly_by_house(markets, classify=not args.fast)
    r = dict(head=head, markets=markets, hourly=hourly)
    return dict(text=format_report(r), code=0, markets=markets, hourly=hourly)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--houses", default="", help="auction house numbers, comma separated; 'all' for every house in the dataset (default: the PvP set 110,113,116)")
    ap.add_argument("--days", type=int, default=2, help="how many UTC days, ending today (default 2)")
    ap.add_argument("--end", default="", help="last UTC day, YYYY-MM-DD (default: today)")
    ap.add_argument("--kinds", default="both", choices=["both", "commodity", "auction"])
    ap.add_argument("--tables", action="store_true", help="also build and describe the house-keyed hourly table (nothing is uploaded)")
    ap.add_argument("--fast", action="store_true", help="with --tables: skip the sales classifier (the wipe and long-gap flags then stay False)")
    ap.add_argument("--max-files", type=int, default=1500)
    ap.add_argument("--max-mb", type=int, default=800)
    ap.add_argument("--cache", default="", help="folder for the loader's cache (default: none)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    args.cache = args.cache or None
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("forever_adapter needs HF_TOKEN (read access to the Forever dataset).")
        return 1
    try:
        r = run(args, FV.RetryStore(HFStore(FV.hf_repo(), token)))
    except Exception as e:                                                       # an error message can carry the dataset's address: blank it out before it reaches a log
        print(f"forever_adapter failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_adapter.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
