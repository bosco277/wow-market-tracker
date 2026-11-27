"""Forever derive: the app's tables, one set per WoW Forever auction house (the same engine.derive the retail app was built on).

    python tools/forever_derive.py                      # report only: every house with data, the last 96 hours
    python tools/forever_derive.py --save               # store the tables in the Forever dataset
    python tools/forever_derive.py --houses 110,113,116 --save

For each house it loads that house's hourly commodity snapshots (never pooled with another house), runs engine.derive.derive with Forever settings, and adds
the latest order book by price level (book_now, for the House flips tab). Stored in the Forever dataset, one folder per house:
    engine/h<house>/items.parquet, item_hourly.parquet, profile.parquet, flips.parquet, shelves.parquet, calibration.parquet,
                    calibration_history.parquet (this run appended), book_now.parquet
    engine/houses.json   which houses were built, when, and their labels (ruleset, faction, confirmed or not) from engine/forevertables.py
One commit per house as it is built, then the houses list (v0.7.51: an upload failure loses that house only). It reads only the Forever dataset (HF_TOKEN; no Blizzard calls) and never deletes anything.
Sales figures are LOWER BOUNDS: a listing posted and sold inside one hour is in no snapshot.
"""
import argparse
import io
import json
import math
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import __version__ as ENGINE_VERSION  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.derive import DeriveConfig, derive  # noqa: E402
from engine.loader import HFStore  # noqa: E402
from engine.schema import COPPER_PER_GOLD  # noqa: E402
import forever_tables as FT  # noqa: E402

PREFIX = "engine"
HOUSES_KEY = f"{PREFIX}/houses.json"
TABLES = ("items", "item_hourly", "profile", "flips", "shelves", "calibration", "calibration_history", "book_now")
HISTORY_KEEP_DAYS = 60
# Forever settings. Retail's cut-offs were set for retail volumes (an item had to be able to sell 20 a day to be tracked, and under 500 a day counted as slow);
# Forever sells far fewer units, and our sales numbers are lower bounds, so the cut-offs are lower. Not calibrated: a first setting, shown in the report.
# ASSUMED auction house cut by house faction (Classic-era values, NOT yet checked on WoW Forever). The flip shelves of a house are worked out with its own cut;
# the app's House flips tab uses the same numbers (frontend houseflip.FEES; a test keeps them equal). Shown in the report and in the app's footnote.
AH_CUTS = dict(alliance=0.05, horde=0.05, neutral=0.15)
FOREVER_CFG = dict(window_hours=72.0, history_hours=96.0, flip_hours=96.0, min_daily_high=2.0, slow_daily_est=50.0, max_items=3000)


def key_for(house: int, table: str) -> str:
    return f"{PREFIX}/h{int(house)}/{table}.parquet"


def to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def from_bytes(data) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception:
        return pd.DataFrame()


def book_now(snap) -> pd.DataFrame:
    """The latest order book by price level: item_id, price_gold, quantity, listings (every item listed, not only tracked ones; the House flips tab buys from it)."""
    b = snap.book
    if b is None or b.empty:
        return pd.DataFrame(columns=["item_id", "price_gold", "quantity", "listings", "ts"])
    g = b.groupby(["item_id", "unit_price"]).agg(quantity=("quantity", "sum"), listings=("quantity", "size")).reset_index()
    g["price_gold"] = g["unit_price"] / COPPER_PER_GOLD
    g["ts"] = pd.Timestamp(snap.ts)
    return g[["item_id", "price_gold", "quantity", "listings", "ts"]].sort_values(["item_id", "price_gold"]).reset_index(drop=True)


def append_history(old: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    hist = pd.concat([old, cal], ignore_index=True) if old is not None and len(old) else cal.copy()
    hist = hist.drop_duplicates(subset=["data_last_ts", "engine_version"], keep="last")
    ts = pd.to_datetime(hist["data_last_ts"])
    return hist[ts > ts.max() - pd.Timedelta(days=HISTORY_KEEP_DAYS)].reset_index(drop=True)


def house_labels(houses) -> dict:
    """{house: {ruleset, faction, confidence}} from the code's house map (so a confirmed house never shows as unverified because a stored copy is old)."""
    m = T.house_map()["houses"]
    return {str(h): {k: m.get(str(h), {}).get(k) for k in ("ruleset", "faction", "confidence")} for h in houses}


def item_names(store) -> dict:
    data = FT.read_key(store, T.KEYS["items"])
    if not data:
        return {}
    try:
        t = pd.read_csv(io.BytesIO(data), compression="gzip", keep_default_na=False, na_values=[""])
        return {int(i): str(n) for i, n in zip(t["item_id"], t["name"]) if isinstance(n, str) and n}
    except Exception:
        return {}


def cause(e) -> str:
    """The error and what caused it (the Hub's own message is often only 'Error while uploading ...'), with the dataset's name blanked out."""
    parts, seen = [], set()
    while e is not None and id(e) not in seen and len(parts) < 4:
        seen.add(id(e))
        parts.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    return FV.scrub(" <- ".join(parts))[:600]


def et(ts) -> str:
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert("America/New_York").strftime("%m-%d %H:%M") if pd.Timestamp(ts).tzinfo is None else \
        pd.Timestamp(ts).tz_convert("America/New_York").strftime("%m-%d %H:%M")


def run(args, store, now=None, base_reader=None, commit=FT.hf_commit):
    now = now or datetime.now(timezone.utc)
    lines = []
    out = lines.append
    code = 0
    hours = float(args.hours)
    save = bool(getattr(args, "save", False))
    avail = FV.houses_in_store(store)
    if not avail:
        return dict(text="### forever derive\nThe sources file could not be read, so no house is known. Wrong dataset name, or a token without access to it.", code=1, files={}, houses=[])
    avail = [h for h in avail if h not in T.EMPTY_HOUSES]
    houses = [h for h in FV.parse_houses(args.houses, avail) if h in avail]
    cfg = DeriveConfig(**{**FOREVER_CFG, "window_hours": min(FOREVER_CFG["window_hours"], hours), "history_hours": hours, "flip_hours": hours})
    labels = house_labels(houses)
    end = now.date()
    start = end - timedelta(days=int(math.ceil(hours / 24.0)))
    out("### forever derive")
    out(f"run at {now.strftime('%Y-%m-%d %H:%M')} UTC; engine {ENGINE_VERSION}; last {hours:.0f} hours of each house; houses: {', '.join(map(str, houses)) or 'none'}; "
        + ("SAVE" if save else "REPORT ONLY (save = no)"))
    out(f"Forever settings (not calibrated): tracked if it could have sold {cfg.min_daily_high:g}+ a day; 'slow' under {cfg.slow_daily_est:g} a day; rates over the last {cfg.window_hours:g} h.")
    out("Sales are LOWER BOUNDS: a listing posted and sold inside one hour is in no snapshot.")
    out(f"Assumed auction house cut (not checked on Forever), used for each house's flip shelves: Alliance and Horde {AH_CUTS['alliance']:.0%}, neutral {AH_CUTS['neutral']:.0%}.")
    if not houses:
        out("Nothing to do: no house to read.")
        return dict(text="\n".join(lines), code=1, files={}, houses=[])
    names = item_names(store)
    out(f"Item names: {len(names):,} known" + ("" if names else " (run the Forever tables workflow to store them)") + ".")
    out("")
    out("### 1 houses")
    out(f"  {'house':>5} {'snapshots':>9}  {'first (ET)':<11} {'last (ET)':<11} {'items':>6} {'High':>5} {'Med':>5} {'Low':>5} {'flips':>6} {'shelves':>7} {'book items':>10} {'load s':>6} {'calc s':>6}")
    files, built, failed, saved, upload_failed = {}, {}, [], {}, []
    for h in houses:
        t0 = time.time()
        try:
            hd = FV.load_house(store, h, start, end, kinds=("commodity",), on_error="skip", base_reader=base_reader)
            t_load = time.time() - t0
            snaps = [s for s in hd.commodity.snapshots]
            if snaps:
                last_ts = snaps[-1].ts
                snaps = [s for s in snaps if s.ts > last_ts - pd.Timedelta(hours=hours)]
            if len(snaps) < 3:
                out(f"  {h:>5} {len(snaps):>9}  too few snapshots in the last {hours:.0f} hours; skipped")
                code = 1
                continue
            cut = AH_CUTS.get((labels.get(str(h)) or {}).get("faction"), AH_CUTS["alliance"])
            t1 = time.time()
            res = derive(snaps, replace(cfg, shelf=replace(cfg.shelf, ah_cut=cut)), names=names)
            res["book_now"] = book_now(snaps[-1])
            res["calibration_history"] = append_history(from_bytes(FT.read_key(store, key_for(h, "calibration_history"))), res["calibration"])
            it, cal = res["items"], res["calibration"].iloc[0]
            out(f"  {h:>5} {len(snaps):>9}  {et(snaps[0].ts):<11} {et(snaps[-1].ts):<11} {len(it):>6} {int(cal.get('n_high', 0)):>5} {int(cal.get('n_medium', 0)):>5} "
                f"{int(cal.get('n_low', 0)):>5} {len(res['flips']):>6} {len(res['shelves']):>7} {res['book_now']['item_id'].nunique():>10} {t_load:>6.0f} {time.time() - t1:>6.0f}")
            mine = {}
            for k in TABLES:
                df = res.get(k)
                if df is not None and len(df):
                    mine[key_for(h, k)] = to_bytes(df)
            files.update(mine)
            built[h] = str(pd.Timestamp(snaps[-1].ts))
            if save:                                 # each house is stored as soon as it is built: an upload failure loses that house only
                try:
                    commit(mine, f"forever derive: house {h}")
                    saved[h] = built[h]
                except Exception as e:
                    upload_failed.append(h)
                    code = 1
                    out(f"        house {h}: UPLOAD FAILED (the other houses are not affected): {cause(e)}")
            if hd.commodity.result.skipped:
                out(f"        {len(hd.commodity.result.skipped)} file(s) skipped (unreadable)")
            del snaps, hd, res
        except Exception as e:                       # one house failing must not stop the others
            failed.append(h)
            code = 1
            out(f"  {h:>5} FAILED: {cause(e)}")
    # houses.json lists every house with stored tables: the ones stored now, plus earlier ones (kept as they were), so one failed house does not vanish from the app
    listed = dict(saved) if save else dict(built)
    if save:
        try:
            old = json.loads((FT.read_key(store, HOUSES_KEY) or b"{}").decode("utf-8")).get("built") or {}
            for k, v in old.items():
                listed.setdefault(int(k), v)
        except Exception:
            pass
    houses_file = None
    if listed:
        houses_file = json.dumps(dict(engine_version=ENGINE_VERSION, built={str(k): v for k, v in sorted(listed.items())}, labels=house_labels(sorted(listed)),
                                      as_of=now.strftime("%Y-%m-%d %H:%M UTC")), indent=1, sort_keys=True).encode("utf-8")
        files[HOUSES_KEY] = houses_file
    out("load s = downloading and reading the hourly files; calc s = the derive itself (what packing the files into day files can and cannot speed up).")
    out("")
    out("### 2 storing")
    total = sum(len(v) for v in files.values())
    out(f"  {len(files)} file(s), {FT.kb(total)} in all, for {len(built)} house(s)" + (f"; failed to build: {', '.join(map(str, failed))}" if failed else ""))
    if save:
        if saved and houses_file is not None:
            try:
                commit({HOUSES_KEY: houses_file}, "forever derive: houses list")
            except Exception as e:
                code = 1
                out(f"  houses list: UPLOAD FAILED: {cause(e)}")
        out(f"Saved {len(saved)} house(s), one commit each" + (f"; upload failed for: {', '.join(map(str, upload_failed))}" if upload_failed else "") + ".")
    elif files:
        out("Report only: nothing stored. Run again with save = yes to store them.")
    return dict(text="\n".join(lines), code=code, files=files, houses=sorted(built), saved=sorted(saved))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=96.0, help="hours of snapshots per house (default 96: rates use the last 72, charts show all of it)")
    ap.add_argument("--houses", default="all", help="all (default), or house numbers separated by commas")
    ap.add_argument("--save", action="store_true", help="store the tables (default: report only)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("forever_derive needs HF_TOKEN (write access to the Forever dataset to save).")
        return 1
    try:
        r = run(args, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except SystemExit:
        raise
    except Exception as e:
        print(f"forever_derive failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_derive.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
