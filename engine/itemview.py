"""Look at one item over time, and find market "flips" (someone buys the cheap side and relists higher).

A flip candidate is one snapshot interval where an item's lowest price jumped by `min_jump` or more AND most of the
units that used to sit BELOW the new lowest price are gone. We then ask whether those units could simply have expired
(listings with LONG / VERY_LONG time left cannot expire inside an interval of an hour or so). If they could not, the
units were bought (or cancelled), which is what a flip looks like. Afterwards we measure how long the elevated price
lasted and how many units the market is estimated to have bought while it was elevated.

This is a DESCRIPTION of what happened in past data. It is not yet a recommendation to flip anything.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .classifier import ClassifierParams, classify_series
from .schema import COPPER_PER_GOLD, Snapshot

CANNOT_EXPIRE_SOON = ("LONG", "VERY_LONG")


def _only_item(s: Snapshot, item_id: int) -> Optional[Snapshot]:
    b = s.book[s.book["item_id"] == item_id]
    if b.empty:
        return None
    return Snapshot(s.source, s.ts, b.reset_index(drop=True), kind=s.kind, covered_items=frozenset({item_id}), meta=dict(s.meta))


def item_history(snapshots: Sequence[Snapshot], item_id: int, params: Optional[ClassifierParams] = None) -> pd.DataFrame:
    """One row per snapshot: price, depth and the sales bounds for the interval that ENDED at that snapshot."""
    snaps = [x for x in (_only_item(s, int(item_id)) for s in sorted(snapshots, key=lambda s: s.ts)) if x is not None]
    if not snaps:
        return pd.DataFrame()
    rows = []
    for s in snaps:
        b = s.book
        floor = int(b["unit_price"].min())
        rows.append(dict(
            ts=s.ts, floor_gold=floor / COPPER_PER_GOLD, units_listed=int(b["quantity"].sum()),
            units_at_floor=int(b.loc[b["unit_price"] == floor, "quantity"].sum()),
            units_within_10pct=int(b.loc[b["unit_price"] <= floor * 1.10, "quantity"].sum()),
            n_listings=int(len(b)),
            second_price_gold=(float(np.sort(b["unit_price"].unique())[1]) / COPPER_PER_GOLD if b["unit_price"].nunique() > 1 else np.nan),
        ))
    hist = pd.DataFrame(rows)
    if len(snaps) >= 2:
        iv = classify_series(snaps, params)[["t1", "gap_minutes", "new_units", "sold_cert", "sold_low", "sold_est", "sold_high",
                                             "flag_wipe", "flag_long_gap"]].rename(columns={"t1": "ts"})
        hist = hist.merge(iv, on="ts", how="left")
    return hist


def detect_flips(
    snapshots: Sequence[Snapshot], min_jump: float = 1.30, min_share_removed: float = 0.80, min_units_removed: int = 20,
    max_gap_minutes: float = 120.0, revert_tolerance: float = 0.10, params: Optional[ClassifierParams] = None,
) -> pd.DataFrame:
    snaps = sorted(snapshots, key=lambda s: s.ts)
    events: List[dict] = []
    for a, b in zip(snaps[:-1], snaps[1:]):
        gap = (b.ts - a.ts).total_seconds() / 60.0
        if gap > max_gap_minutes or a.kind != "commodity":
            continue
        fp = a.book.groupby("item_id")["unit_price"].min()
        fc = b.book.groupby("item_id")["unit_price"].min()
        jump = (fc / fp).dropna()
        for item in jump[jump >= min_jump].index:
            pi = a.book[a.book["item_id"] == item]
            ci = b.book[b.book["item_id"] == item].set_index("auction_id")["quantity"]
            new_floor = fc[item]
            below = pi[pi["unit_price"] < new_floor]
            if below.empty:
                continue
            now_qty = below["auction_id"].map(ci).fillna(0)
            removed = (below["quantity"] - now_qty).clip(lower=0)
            total_below, removed_units = float(below["quantity"].sum()), float(removed.sum())
            share = removed_units / total_below if total_below else 0.0
            unexp = float(removed[below["time_left"].isin(CANNOT_EXPIRE_SOON)].sum())
            share_unexp = unexp / removed_units if removed_units else 0.0
            if share < min_share_removed or removed_units < min_units_removed:
                continue
            events.append(dict(
                source=a.source, item_id=int(item), t0=a.ts, t1=b.ts, gap_minutes=gap,
                floor_before_gold=float(fp[item]) / COPPER_PER_GOLD, floor_after_gold=float(new_floor) / COPPER_PER_GOLD,
                jump=float(jump[item]), units_removed=removed_units, share_removed=share, share_unexpirable=share_unexp,
                verdict="flip / buyout" if share_unexp >= 0.8 else "probably expiry or cancels",
            ))
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev

    ev["revert_hours"] = np.nan
    ev["still_elevated_at_end"] = False
    ev["sold_est_while_elevated"] = np.nan
    for i, e in ev.iterrows():
        later = [s for s in snaps if s.ts > e["t1"]]
        base = e["floor_before_gold"] * COPPER_PER_GOLD * (1 + revert_tolerance)
        revert_ts = None
        for s in later:
            ib = s.book[s.book["item_id"] == e["item_id"]]
            if not ib.empty and ib["unit_price"].min() <= base:
                revert_ts = s.ts
                break
        end_ts = revert_ts if revert_ts is not None else (snaps[-1].ts)
        ev.at[i, "still_elevated_at_end"] = revert_ts is None
        if revert_ts is not None:
            ev.at[i, "revert_hours"] = (revert_ts - e["t1"]).total_seconds() / 3600.0
        if e["verdict"] == "flip / buyout":
            window = [x for x in (_only_item(s, int(e["item_id"])) for s in snaps if e["t1"] <= s.ts <= end_ts) if x is not None]
            if len(window) >= 2:
                iv = classify_series(window, params)
                ev.at[i, "sold_est_while_elevated"] = float(iv["sold_est"].sum())
    return ev.sort_values(["t1", "item_id"]).reset_index(drop=True)
