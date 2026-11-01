"""Backtest the flip screener on the snapshots we already hold: no gold at risk.

At every `step`-th snapshot (once there are `min_history_hours` of history) rebuild the shelf the screener would have
shown, using ONLY what was known at that moment (history up to then, that hour's book). Then look at what happened next:

  reached_Nh        did the lowest price get to the relist price (within the level tolerance) within N hours? NaN when the data
                    ends or has a hole before the answer is known
  hours_to_reach    how long it took
  floor_ratio_4h    lowest price 4 h later / relist price (1.0 = right at it)
  inflow_2h/4h      cheaper listings that APPEARED (a tier below the relist price grew) within 2 / 4 h, as a share of the units
                    you would have bought. Large = you would likely have been undercut or pushed back
  sold_est_4h       the item's estimated sales in the next 4 h as a share of the units you would have bought (all prices,
                    so a generous reading)

Caveats: the cheap units are still in the book in this replay, so "reached" means other buyers cleared them. Your own purchase
would clear them at once, which makes `reached` a cautious reading of the relist price being reachable and `inflow` the
better read of the undercut risk. A shelf that persists across hours counts once (`first_seen`). Spike flags are switched off
so nothing from the future leaks in. With a short history the regime test can only warn.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .classifier import ClassifierParams, classify_series
from .derive import _floors_and_units
from .schema import COPPER_PER_GOLD, Snapshot
from .shelves import ShelfConfig, shelves_for_item

HORIZONS = (4, 12)


def _prefilter(book: pd.DataFrame, keep: set, cfg: ShelfConfig) -> np.ndarray:
    """Items with at least one price gap that could pass the margin test (an upper bound, so nothing real is dropped)."""
    b = book[book["item_id"].isin(keep)]
    t = b.groupby(["item_id", "unit_price"], sort=True)["quantity"].sum().reset_index()
    t["prev"] = t.groupby("item_id")["unit_price"].shift()
    t["p0"] = t.groupby("item_id")["unit_price"].transform("first")
    ok = t["prev"].notna() & (t["unit_price"] / t["prev"] - 1.0 >= cfg.min_gap) & (t["unit_price"] * (1 - cfg.ah_cut) / t["p0"] - 1.0 >= cfg.min_net_margin)
    return t.loc[ok, "item_id"].unique()


def _new_cheaper_units(book_now: pd.DataFrame, book_later: pd.DataFrame, below_copper: float) -> float:
    """Units that appeared (a tier grew) at prices under `below_copper` between two books of one item."""
    now = book_now.groupby("unit_price")["quantity"].sum()
    later = book_later.groupby("unit_price")["quantity"].sum()
    later = later[later.index < below_copper]
    return float((later - now.reindex(later.index).fillna(0)).clip(lower=0).sum())


def backtest_shelves(snapshots: Sequence[Snapshot], cfg: ShelfConfig = ShelfConfig(), params: Optional[ClassifierParams] = None,
                     item_ids: Optional[Iterable[int]] = None, step: int = 2, min_history_hours: float = 48.0,
                     rate_rows: int = 72) -> pd.DataFrame:
    snaps = sorted((s for s in snapshots if s.kind == "commodity"), key=lambda s: s.ts)
    if len(snaps) < 5:
        raise ValueError("need at least 5 commodity snapshots")
    intervals = classify_series(snaps, params)
    fu = _floors_and_units(snaps, snaps[0].ts - pd.Timedelta(hours=1))
    keep = set(int(i) for i in (item_ids if item_ids is not None else fu["item_id"].unique()))
    fu = fu[fu["item_id"].isin(keep)]
    iv = intervals[intervals["item_id"].isin(keep)][["item_id", "t1", "sold_low", "sold_est", "sold_high"]].rename(columns={"t1": "ts"})
    h = fu.merge(iv, on=["item_id", "ts"], how="left").sort_values(["item_id", "ts"])
    h["floor_gold"] = h["floor"] / COPPER_PER_GOLD
    h["spike"] = False
    groups = {int(i): g.reset_index(drop=True) for i, g in h.groupby("item_id")}
    tsarr = {i: g["ts"].to_numpy() for i, g in groups.items()}
    indices = {}

    def book_of(j: int, item: int) -> pd.DataFrame:
        if j not in indices:
            indices[j] = snaps[j].book.groupby("item_id").indices
        ix = indices[j].get(item)
        return snaps[j].book.iloc[ix] if ix is not None else snaps[j].book.iloc[0:0]

    times = np.array([s.ts.to_datetime64() for s in snaps])
    last_ts = snaps[-1].ts
    start = next((i for i, s in enumerate(snaps) if (s.ts - snaps[0].ts) >= pd.Timedelta(hours=min_history_hours)), None)
    if start is None:
        raise ValueError(f"need at least {min_history_hours:.0f} hours of snapshots")
    rows: List[dict] = []
    for n, i in enumerate(range(start, len(snaps), step)):
        t0 = snaps[i].ts
        for item in _prefilter(snaps[i].book, keep, cfg):
            item = int(item)
            g = groups.get(item)
            if g is None:
                continue
            pos = int(np.searchsorted(tsarr[item], t0.to_datetime64(), side="right"))
            ser = g.iloc[:pos]
            if ser.empty or ser["ts"].iloc[-1] != t0:
                continue
            tail = ser.tail(rate_rows)
            rate = {k: float(tail[f"sold_{k}"].mean()) if tail[f"sold_{k}"].notna().any() else 0.0 for k in ("low", "est", "high")}
            band90 = float(tail["floor_gold"].quantile(0.90))
            bk = book_of(i, item)
            sh = shelves_for_item(bk, ser, rate, band90, cfg)
            if sh is None:
                continue
            row = dict(eval_n=n, t0=t0, item_id=item, **sh)
            level = sh["target_gold"] * (1 - cfg.level_tolerance)
            fut = g.iloc[pos:]
            for H in HORIZONS:
                end = t0 + pd.Timedelta(hours=H)
                w = fut[fut["ts"] <= end]
                hit = w[w["floor_gold"] >= level]
                known = bool(len(hit)) or last_ts >= end
                row[f"reached_{H}h"] = (bool(len(hit)) if known else np.nan)
                if H == HORIZONS[-1]:
                    row["hours_to_reach"] = float((hit["ts"].iloc[0] - t0).total_seconds() / 3600.0) if len(hit) else np.nan
            near = fut[(fut["ts"] - (t0 + pd.Timedelta(hours=4))).abs() <= pd.Timedelta(minutes=40)]
            row["floor_ratio_4h"] = float(near["floor_gold"].iloc[0] / sh["target_gold"]) if len(near) else np.nan
            w4 = fut[fut["ts"] <= t0 + pd.Timedelta(hours=4)]
            row["sold_est_4h"] = (float(w4["sold_est"].sum()) / sh["units"]) if last_ts >= t0 + pd.Timedelta(hours=4) else np.nan
            target_c = sh["target_gold"] * COPPER_PER_GOLD
            for H in (2, 4):
                j = int(np.abs(times - (t0 + pd.Timedelta(hours=H)).to_datetime64()).argmin())
                if abs(snaps[j].ts - (t0 + pd.Timedelta(hours=H))) <= pd.Timedelta(minutes=40) and j > i:
                    row[f"inflow_{H}h"] = _new_cheaper_units(bk, book_of(j, item), target_c) / sh["units"]
                else:
                    row[f"inflow_{H}h"] = np.nan
            rows.append(row)
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    bt["target_key"] = bt["target_gold"].round(3)
    bt = bt.sort_values(["item_id", "target_key", "eval_n"]).reset_index(drop=True)
    gap = bt.groupby(["item_id", "target_key"])["eval_n"].diff()
    bt["first_seen"] = gap.isna() | (gap != 1)
    return bt.drop(columns=["target_key"])


def _line(g: pd.DataFrame) -> dict:
    r4, r12 = g["reached_4h"].dropna().astype(float), g["reached_12h"].dropna().astype(float)
    return dict(n=int(len(g)), reached_4h=float(r4.mean()) if len(r4) else np.nan, reached_12h=float(r12.mean()) if len(r12) else np.nan,
                median_hours=float(g["hours_to_reach"].median()) if g["hours_to_reach"].notna().any() else np.nan,
                median_floor_ratio_4h=float(g["floor_ratio_4h"].median()) if g["floor_ratio_4h"].notna().any() else np.nan,
                undercut_risk=float((g["inflow_2h"].dropna() >= 0.5).mean()) if g["inflow_2h"].notna().any() else np.nan)


TESTS = ["test_price", "test_speed", "test_size", "test_margin", "test_regime", "test_book"]


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per group: all shelves, by verdict, and by each test's result. First sighting of each shelf only."""
    if bt.empty:
        return pd.DataFrame()
    f = bt[bt["first_seen"]]
    out = [dict(group="all shelves", **_line(f))]
    for v in ("candidate", "weak", "no"):
        g = f[f["verdict"] == v]
        if len(g):
            out.append(dict(group=f"verdict: {v}", **_line(g)))
    for t in TESTS:
        for res in ("pass", "warn", "fail"):
            g = f[f[t] == res]
            if len(g):
                out.append(dict(group=f"{t.replace('test_', '')}: {res}", **_line(g)))
    # What else separates shelves that worked from shelves that did not: the size of the margin, of the buy, and the selling time
    def buckets(col, edges, labels, name):
        c = pd.cut(f[col], edges, labels=labels, right=False)
        for lab in labels:
            g = f[c == lab]
            if len(g):
                out.append(dict(group=f"{name}: {lab}", **_line(g)))
    buckets("net_margin", [-np.inf, 0.25, 0.5, 1.0, np.inf], ["under 25%", "25-50%", "50-100%", "over 100%"], "margin")
    buckets("units", [0, 26, 101, np.inf], ["1-25", "26-100", "over 100"], "units")
    buckets("sell_hours_est", [0, 1, 4, 24, np.inf], ["under 1 h", "1-4 h", "4-24 h", "over 24 h"], "est. hours to sell")
    passes = (f[TESTS] == "pass").sum(axis=1)
    for k in sorted(passes.unique(), reverse=True):
        out.append(dict(group=f"checks passed: {k} of {len(TESTS)}", **_line(f[passes == k])))
    return pd.DataFrame(out)
