"""Forever pack probe: how small can one day of a house's hourly files get WITHOUT losing anything? REPORT ONLY: it stores nothing and deletes nothing.

    python tools/forever_pack_probe.py                         # yesterday (UTC), houses 110, 116, 121
    python tools/forever_pack_probe.py --day 2026-11-26 --houses 110,113,116

For each house it downloads that day's hourly commodity files with EVERY column the fetcher wrote, packs them three ways (engine/foreverpack.py: concat,
sorted, ledger), and proves each one lossless:
  1 every hourly file rebuilt from the packed form equals the original exactly (same rows, every column, every auction id), and
  2 the sales classifier on the rebuilt day gives exactly the same sales, item by item and hour by hour, as on the original.
Then it prints the sizes (parquet, zstd) and what a year would come to for all houses. Needs HF_TOKEN (read only).
"""
import argparse
import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import forever as FV  # noqa: E402
from engine import foreverpack as P  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.classifier import classify_series  # noqa: E402
from engine.loader import HFStore, normalize_frame, plan_load  # noqa: E402
from engine.schema import snapshots_from_blizzard_v2  # noqa: E402

DEFAULT_HOUSES = "110,116,121"          # a big faction house, the PvP neutral house, a small house
HOUSES_WITH_DATA = 12
MB = 1024 * 1024
SALES = ["sold_cert", "sold_low", "sold_est", "sold_high"]


def read_raw(data: bytes) -> pd.DataFrame:
    """Every column of one stored hourly file."""
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(data)).to_pandas()


def size_of(df: pd.DataFrame) -> int:
    """Bytes as parquet with zstd compression (what a packed file would take)."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return len(buf.getvalue())


def ledger_for_size(runs: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    """The ledger as ONE table for the size count: runs, then the small per-file table's columns appended with a `_part` marker (one stored file a day)."""
    a = runs.assign(_part="run")
    b = files.assign(_part="file")
    return pd.concat([a, b], ignore_index=True, sort=False)


def sales(frames, source):
    """The sales classifier's per-item, per-hour sales for a list of raw hourly frames (read the way the Forever loader reads them)."""
    snaps = []
    for f in frames:
        clean, _, _ = FV.drop_duplicate_ids(f)
        fr = normalize_frame(clean, source)
        if len(fr):
            snaps.extend(snapshots_from_blizzard_v2(fr))
    for s in snaps:
        s.kind = "commodity"
    iv = classify_series(snaps)
    if iv.empty:
        return iv
    keep = [c for c in ["item_id", "t0", "t1"] + SALES if c in iv.columns]
    return iv[keep].sort_values(["item_id", "t0"]).reset_index(drop=True)


def run(args, store, now=None, reader=read_raw, sizer=size_of):
    now = now or datetime.now(timezone.utc)
    day = datetime.strptime(args.day, "%Y-%m-%d").date() if args.day else (now.date() - timedelta(days=1))
    houses = FV.parse_houses(args.houses or DEFAULT_HOUSES)
    lines = []
    out = lines.append
    code = 0
    out("### forever pack probe")
    out(f"run at {now.strftime('%Y-%m-%d %H:%M')} UTC; UTC day {day}; houses {', '.join(map(str, houses))}; REPORT ONLY (nothing is stored or deleted)")
    out("Lossless = every hourly file rebuilt from the packed form equals the original in every row and column (every auction id), AND the sales classifier gives")
    out("the same sales item by item and hour by hour. Sizes are parquet with zstd compression. The row order inside an hourly file is not kept (it carries no information).")
    out("")
    out("### 1 per house")
    totals = dict(raw=0, concat=0, sorted=0, ledger=0, files=0, rows=0, runs=0)
    done = 0
    for h in houses:
        source = FV.source_name(h, True)
        try:
            plan = plan_load(store, source, day, day)
            objs = [o for o in plan.files if "hour_" in o.key]
            if len(objs) < 2:
                out(f"  house {h}: {len(objs)} hourly file(s) on {day}; skipped")
                code = 1
                continue
            t0 = time.time()
            frames = [reader(store.read_range(o.key, 0, o.size)) for o in objs]
            t_read = time.time() - t0
            raw = sum(o.size for o in objs)
            cols = list(frames[0].columns)
            rows = sum(len(f) for f in frames)

            t = time.time()
            cat = P.to_concat(frames)
            s_cat = sizer(cat)
            ok_cat, why_cat = P.check(frames, P.from_rows(cat, len(frames), cols))
            t_cat = time.time() - t

            t = time.time()
            srt = P.to_sorted(frames)
            s_srt = sizer(srt)
            ok_srt, why_srt = P.check(frames, P.from_rows(srt, len(frames), cols))
            t_srt = time.time() - t

            t = time.time()
            runs, files, const = P.to_ledger(frames)
            s_led = sizer(ledger_for_size(runs, files))
            rebuilt = P.from_ledger(runs, files, const, cols, frames[0].dtypes)
            ok_led, why_led = P.check(frames, rebuilt)
            t_led = time.time() - t

            t = time.time()
            a, b = sales(frames, source), sales(rebuilt, source)
            same_sales = a.shape == b.shape and bool(np.allclose(a[SALES].to_numpy(dtype=float), b[SALES].to_numpy(dtype=float), equal_nan=True)) \
                and a[["item_id", "t0"]].equals(b[["item_id", "t0"]])
            t_sales = time.time() - t

            out(f"  house {h}: {len(objs)} hourly files, {raw / MB:.1f} MB as stored, {rows:,} rows (listing-hours); read in {t_read:.0f} s")
            out(f"    {'form':<8} {'size':>9} {'vs stored':>9} {'rows':>10} {'lossless':>9} {'secs':>5}")
            for name, size, n, ok, why, secs in (("concat", s_cat, len(cat), ok_cat, why_cat, t_cat), ("sorted", s_srt, len(srt), ok_srt, why_srt, t_srt),
                                                 ("ledger", s_led, len(runs), ok_led, why_led, t_led)):
                out(f"    {name:<8} {size / MB:>7.2f}MB {raw / max(size, 1):>8.1f}x {n:>10,} {'yes' if ok else 'NO':>9} {secs:>5.0f}" + ("" if ok else f"   <- {why}"))
                if not ok:
                    code = 1
            out(f"    sales classifier on the original day and on the day rebuilt from the ledger: {'identical' if same_sales else 'DIFFERENT'} "
                f"({len(a):,} item-hours; total certain sales {a['sold_cert'].sum() if len(a) else 0:,.0f}, conservative {a['sold_low'].sum() if len(a) else 0:,.0f}; {t_sales:.0f} s)")
            out(f"    per-file columns kept once per file in the ledger: {', '.join(const) or 'none'}")
            if not same_sales:
                code = 1
            totals.update(raw=totals["raw"] + raw, concat=totals["concat"] + s_cat, sorted=totals["sorted"] + s_srt, ledger=totals["ledger"] + s_led,
                          files=totals["files"] + len(objs), rows=totals["rows"] + rows, runs=totals["runs"] + len(runs))
            done += 1
            del frames, cat, srt, runs, rebuilt
        except Exception as e:
            code = 1
            out(f"  house {h}: FAILED: {FV.scrub(e)}")
    if done:
        out("")
        out(f"### 2 a year for all {HOUSES_WITH_DATA} houses with data (commodity files only), scaled from these {done} house-day(s)")
        scale = HOUSES_WITH_DATA * 365 / done
        out(f"  {'form':<8} {'per year':>10} {'files a year':>13}")
        out(f"  {'stored':<8} {totals['raw'] * scale / 1024 ** 3:>8.1f}GB {int(totals['files'] * scale):>13,}")
        for name in ("concat", "sorted", "ledger"):
            out(f"  {name:<8} {totals[name] * scale / 1024 ** 3:>8.1f}GB {int(HOUSES_WITH_DATA * 365):>13,}")
        out(f"  (The houses differ a lot in size, so this is rough: 121 is a small house, 110 a big one. The Hugging Face guideline is 100 GB and 100,000 files per dataset.)")
        out(f"  Ledger rows are {totals['runs'] / max(totals['rows'], 1):.1%} of the stored rows: each listing is kept once per stretch of unchanged hours instead of once per hour.")
    return dict(text="\n".join(lines), code=code)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default="", help="UTC day YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--houses", default=DEFAULT_HOUSES, help=f"house numbers separated by commas (default {DEFAULT_HOUSES})")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("forever_pack_probe needs HF_TOKEN (read access to the Forever dataset).")
        return 1
    try:
        r = run(args, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except Exception as e:
        print(f"forever_pack_probe failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_pack_probe.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
