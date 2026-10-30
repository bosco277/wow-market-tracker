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
    max_gap_minutes: float = 120.0, revert_tolerance: float = 0.10, min_size_vs_normal: float = 3.0,
    min_relisted_share: float = 0.5, params: Optional[ClassifierParams] = None,
) -> pd.DataFrame:
    """One row per candidate. Three layers, each stricter than the last:

    1. JUMP: the lowest price rose by `min_jump` and most of the units below the new lowest price are gone.
    2. BUYOUT (`verdict`): those units could not have simply expired (LONG / VERY_LONG time left).
       Ordinary buyers eating through a thin cheap price level ALSO looks like this, which is why it is not enough.
    3. LIKELY FLIP (`likely_flip`): in addition, the amount removed is at least `min_size_vs_normal` times what the item
       normally sells in an hour, and at least `min_relisted_share` of it shows up again as NEW listings priced near the
       new lowest price in the same interval (someone buying, then relisting higher).

    Follow-up is only counted over time we actually observed. If a hole in our data comes before the price returns, the
    return time is UNKNOWN (`ended_by == "hole"`), never a guess.
    """
    snaps = sorted(snapshots, key=lambda s: s.ts)
    floors = [s.book.groupby("item_id")["unit_price"].min() for s in snaps]
    events: List[dict] = []
    for k in range(1, len(snaps)):
        a, b = snaps[k - 1], snaps[k]
        gap = (b.ts - a.ts).total_seconds() / 60.0
        if gap > max_gap_minutes or a.kind != "commodity":
            continue
        fp, fc = floors[k - 1], floors[k]
        jump = (fc / fp).dropna()
        for item in jump[jump >= min_jump].index:
            pi = a.book[a.book["item_id"] == item]
            bi = b.book[b.book["item_id"] == item]
            ci = bi.set_index("auction_id")["quantity"]
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
            fresh = bi[~bi["auction_id"].isin(pi["auction_id"])]
            relisted = float(fresh.loc[fresh["unit_price"] <= new_floor * 1.25, "quantity"].sum())
            events.append(dict(
                source=a.source, item_id=int(item), t0=a.ts, t1=b.ts, gap_minutes=gap, _k=k,
                floor_before_gold=float(fp[item]) / COPPER_PER_GOLD, floor_after_gold=float(new_floor) / COPPER_PER_GOLD,
                jump=float(jump[item]), units_removed=removed_units, share_removed=share, share_unexpirable=share_unexp,
                relisted_units=relisted, relisted_share=relisted / removed_units,
                verdict="flip / buyout" if share_unexp >= 0.8 else "probably expiry or cancels",
            ))
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev

    # what is normal for each item: the 75th percentile of its hourly estimated sales over observed intervals
    iv_all = classify_series(snaps, params)
    ok = iv_all[~(iv_all["flag_wipe"] | iv_all["flag_long_gap"])]
    normal = ok.groupby("item_id")["sold_est"].quantile(0.75).clip(lower=1.0)
    ev["normal_hourly_sold"] = ev["item_id"].map(normal).fillna(1.0)
    ev["size_vs_normal"] = ev["units_removed"] / ev["normal_hourly_sold"]
    ev["likely_flip"] = ((ev["verdict"] == "flip / buyout") & (ev["size_vs_normal"] >= min_size_vs_normal)
                         & (ev["relisted_share"] >= min_relisted_share))

    ev["revert_hours"] = np.nan
    ev["followup_hours"] = 0.0
    ev["ended_by"] = ""
    ev["still_elevated_at_end"] = False
    ev["sold_est_while_elevated"] = np.nan
    for i, e in ev.iterrows():
        k0, item = int(e["_k"]), e["item_id"]
        base = e["floor_before_gold"] * COPPER_PER_GOLD * (1 + revert_tolerance)
        prev_ts, end_k, how = e["t1"], k0, "end_of_data"
        for k in range(k0 + 1, len(snaps)):
            if (snaps[k].ts - prev_ts).total_seconds() / 60.0 > max_gap_minutes:
                how = "hole"
                break
            end_k, prev_ts = k, snaps[k].ts
            f = floors[k].get(item)
            if f is not None and f <= base:
                how = "reverted"
                break
        ev.at[i, "ended_by"] = how
        ev.at[i, "followup_hours"] = (snaps[end_k].ts - e["t1"]).total_seconds() / 3600.0
        ev.at[i, "still_elevated_at_end"] = how == "end_of_data"
        if how == "reverted":
            ev.at[i, "revert_hours"] = ev.at[i, "followup_hours"]
        if e["likely_flip"] and end_k > k0:
            window = [x for x in (_only_item(s, int(item)) for s in snaps[k0:end_k + 1]) if x is not None]
            if len(window) >= 2:
                iv = classify_series(window, params)
                iv = iv[~(iv["flag_wipe"] | iv["flag_long_gap"])]
                ev.at[i, "sold_est_while_elevated"] = float(iv["sold_est"].sum())
    return ev.drop(columns=["_k"]).sort_values(["t1", "item_id"]).reset_index(drop=True)


def revert_share_within(ev: pd.DataFrame, hours: float) -> tuple:
    """Of the events we watched for at least `hours` (or that reverted sooner), what share was back near the old price
    within `hours`? Events cut short by a hole or the end of the data before `hours` are NOT counted either way."""
    if ev.empty:
        return float("nan"), 0
    back = (ev["ended_by"] == "reverted") & (ev["revert_hours"] <= hours)
    watched = back | (ev["followup_hours"] >= hours)
    n = int(watched.sum())
    return (float(back[watched].mean()) if n else float("nan")), n


def gaps_in(snapshots: Sequence[Snapshot], min_gap_minutes: float = 120.0) -> pd.DataFrame:
    """Holes in the data: start, end, hours. The classifier cannot tell sales apart inside a hole."""
    ts = sorted(s.ts for s in snapshots)
    rows = [dict(start=a, end=b, hours=(b - a).total_seconds() / 3600.0) for a, b in zip(ts[:-1], ts[1:])
            if (b - a).total_seconds() / 60.0 > min_gap_minutes]
    return pd.DataFrame(rows, columns=["start", "end", "hours"])
