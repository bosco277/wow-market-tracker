"""Forever latest listings: refresh each house's CURRENT prices and order book from its newest hourly snapshot (light; meant to run every hour).

    python tools/forever_latest.py               # report only
    python tools/forever_latest.py --save        # store

For each house it reads only the newest hourly commodity file (today's, or yesterday's just after midnight) and writes, in the Forever dataset:
    engine/h<house>/book_now.parquet   the order book by price level (House flips buys from it)
    engine/h<house>/now.parquet        per item: lowest price now, units and listings now, where that price sits in the item's usual range (now_percentile,
                                       against the last 72 hours of the daily build's hourly lowest prices, price spikes left out), and the snapshot time
It never rewrites the daily build's tables (items, item_hourly, ...): the app lays now.parquet over them when it is newer, so this job and the daily
"Forever derive" can never undo each other. Sales rates, bands, confidence and charts still come from the daily build. One commit for all houses; nothing is
deleted. Needs HF_TOKEN.
"""
import argparse
import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import forever as FV  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.loader import KEY_RE, HFStore, normalize_frame  # noqa: E402
from engine.schema import snapshots_from_blizzard_v2  # noqa: E402
import forever_derive as FDV  # noqa: E402
import forever_tables as FT  # noqa: E402

WINDOW_HOURS = 72.0       # now_percentile: the same window the daily build uses for rates and bands


def read_raw(data: bytes) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(data)).to_pandas()


def newest_file(store, source, now):
    """(key, size) of the source's newest hourly file: today's folder, else yesterday's."""
    for d in (now.date(), now.date() - timedelta(days=1)):
        objs = [o for o in store.list(f"{source}/year={d.year:04d}/month={d.month:02d}/day={d.day:02d}/")
                if KEY_RE.match(o.key) and KEY_RE.match(o.key).group("file").startswith("hour_")]
        if objs:
            o = max(objs, key=lambda o: o.key)
            return o.key, o.size
    return None


def snapshot_of(frame, source):
    clean, _, _ = FV.drop_duplicate_ids(frame)
    snaps = snapshots_from_blizzard_v2(normalize_frame(clean, source))
    return snaps[-1] if snaps else None


def now_table(snap, hourly: pd.DataFrame) -> pd.DataFrame:
    """Per listed item: floor_gold, units_listed, n_listings, now_percentile, now_ts."""
    bk = FDV.book_now(snap)
    if bk.empty:
        return pd.DataFrame(columns=["item_id", "floor_gold", "units_listed", "n_listings", "now_percentile", "now_ts"])
    g = bk.groupby("item_id").agg(floor_gold=("price_gold", "min"), units_listed=("quantity", "sum"), n_listings=("listings", "sum")).reset_index()
    g["now_ts"] = pd.Timestamp(snap.ts)
    pct = pd.Series(np.nan, index=g["item_id"])
    if hourly is not None and len(hourly) and {"item_id", "ts", "floor_gold"} <= set(hourly.columns):
        h = hourly[pd.to_datetime(hourly["ts"]) > pd.to_datetime(hourly["ts"]).max() - pd.Timedelta(hours=WINDOW_HOURS)]
        if "spike" in h.columns:
            h = h[h["spike"].fillna(0) == 0]
        base = h.dropna(subset=["floor_gold"]).groupby("item_id")["floor_gold"].apply(lambda s: np.sort(s.to_numpy()))
        now = g.set_index("item_id")["floor_gold"]
        for i in now.index.intersection(base.index):
            b = base[i]
            pct[i] = float(np.searchsorted(b, now[i], side="right")) / len(b) if len(b) else np.nan
    g["now_percentile"] = pct.reindex(g["item_id"]).to_numpy()
    g["units_listed"] = g["units_listed"].astype(int)
    g["n_listings"] = g["n_listings"].astype(int)
    return g[["item_id", "floor_gold", "units_listed", "n_listings", "now_percentile", "now_ts"]]


def run(args, store, now=None, reader=read_raw, commit=FT.hf_commit):
    now = now or datetime.now(timezone.utc)
    t0 = time.time()
    save = bool(getattr(args, "save", False))
    lines = []
    out = lines.append
    code = 0
    avail = [h for h in FV.houses_in_store(store) if h not in T.EMPTY_HOUSES]
    houses = [h for h in FV.parse_houses(getattr(args, "houses", "all"), avail) if h in avail]
    out("### forever latest listings")
    out(f"run at {now.strftime('%Y-%m-%d %H:%M')} UTC; houses {', '.join(map(str, houses)) or 'none'}; " + ("SAVE" if save else "REPORT ONLY (save = no)"))
    if not houses:
        out("Nothing to do: no house to read.")
        return dict(text="\n".join(lines), code=1, files={})
    out(f"  {'house':>5}  {'snapshot (ET)':<13} {'items listed':>12} {'book rows':>9} {'cheap now':>9} {'dear now':>8}")
    files = {}
    for h in houses:
        src = FV.source_name(h, True)
        try:
            nf = newest_file(store, src, now)
            if nf is None:
                out(f"  {h:>5}  no hourly file today or yesterday; skipped")
                code = 1
                continue
            snap = snapshot_of(reader(store.read_range(nf[0], 0, nf[1])), src)
            if snap is None:
                out(f"  {h:>5}  the newest file has no commodity listings; skipped")
                code = 1
                continue
            hourly = FDV.from_bytes(FT.read_key(store, FDV.key_for(h, "item_hourly")))
            nt, bk = now_table(snap, hourly), FDV.book_now(snap)
            files[FDV.key_for(h, "now")] = FDV.to_bytes(nt)
            files[FDV.key_for(h, "book_now")] = FDV.to_bytes(bk)
            cheap = int((nt["now_percentile"] <= 0.15).sum())
            dear = int((nt["now_percentile"] >= 0.85).sum())
            out(f"  {h:>5}  {FDV.et(snap.ts):<13} {len(nt):>12,} {len(bk):>9,} {cheap:>9,} {dear:>8,}")
        except Exception as e:
            code = 1
            out(f"  {h:>5}  FAILED: {FDV.cause(e)}")
    out("  cheap now / dear now: items whose lowest price is in the bottom / top 15% of their last 72 hours (the app's 'cheap / expensive for this item').")
    total = sum(len(v) for v in files.values())
    out(f"{len(files)} file(s), {FT.kb(total)}; {time.time() - t0:.0f} s.")
    if save and files:
        try:
            commit(files, f"forever latest: {len(files) // 2} house(s)")
            out("Saved in one commit.")
        except Exception as e:
            code = 1
            out(f"UPLOAD FAILED: {FDV.cause(e)}")
    elif files:
        out("Report only: nothing stored.")
    return dict(text="\n".join(lines), code=code, files=files)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--houses", default="all")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("forever_latest needs HF_TOKEN.")
        return 1
    try:
        r = run(args, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except Exception as e:
        print(f"forever_latest failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_latest.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
