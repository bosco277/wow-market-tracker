"""Sales inference from two consecutive commodity snapshots.

The Blizzard API shows listings, not receipts. A listing that disappears may have been bought, cancelled or
expired, and anything listed AND sold between two snapshots is invisible. So this module never claims a single
number. For each item and interval it returns a range [sold_low, sold_high] and a heuristic point estimate:

  sold_cert  units that certainly sold: a surviving listing whose quantity dropped (only a purchase does that)
  sold_low   conservative end: certain sales + sweep-consistent disappearances that could NOT have expired,
             minus an allowance for undercut-cancel-relist churn
  sold_est   heuristic point estimate (calibrate `cancel_share` against simulator / real experiments)
  sold_high  optimistic end: certain sales + every disappearance consistent with a purchase sweep

The key structural fact (commodity auctions fill cheapest-first): units that were BOUGHT form a prefix of the
previous book in (price, auction_id) order. Take the first listing that survived untouched; anything that vanished
from beyond it cannot have been bought, so it was a cancel or an expiry. Anything that vanished from before it is
ambiguous (sale, cancel or expiry). `time_left` and the real gap between snapshots rule expiry in or out.

Not handled in v1 (documented limits, see docs/DATA_CONTRACT.md): sales of listings created and sold inside one
interval (blind spot: measured values are a floor on true volume), and per-listing (non-commodity) markets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .schema import Snapshot, TimeLeftBuckets

CLASSIFIER_VERSION = "v1.0"


@dataclass(frozen=True)
class ClassifierParams:
    buckets: TimeLeftBuckets = field(default_factory=TimeLeftBuckets)
    # True: assume buyers fill the earliest listing first within a price tier. False (default): do NOT assume any
    # order inside a tier, so tier-boundary disappearances are ambiguous and count only toward the high bound.
    # Validation (tools/fifo_check.py) shows False costs almost nothing if the market IS first-in-first-out and
    # protects the high bound if it is not. We cannot verify the real tie-break rule, so False is the safe default.
    fifo_within_tier: bool = False
    # Heuristic share of sweep-consistent disappearances that were cancels. Feeds sold_est only. Uncalibrated.
    cancel_share: float = 0.10
    # How much of the "new listings priced at/below the vanished price" volume to attribute to cancel-and-relist
    # undercuts when computing sold_low. 1.0 = fully conservative.
    relist_attribution: float = 1.0
    # An item whose book (almost) entirely vanished is probably a data glitch, delisting or partial scan.
    wipe_threshold: float = 0.9
    wipe_min_units: int = 50
    # Intervals longer than this are flagged: with long gaps every listing in a short bucket must be gone.
    max_gap_minutes: float = 240.0


@dataclass
class TransitionResult:
    source: str
    t0: pd.Timestamp
    t1: pd.Timestamp
    gap_minutes: float
    events: pd.DataFrame  # one row per CHANGED auction (audit trail)
    items: pd.DataFrame  # one row per item present in the previous snapshot

    def __len__(self) -> int:
        return len(self.items)


ITEM_COLUMNS = [
    "source", "item_id", "t0", "t1", "gap_minutes", "prev_units", "curr_units", "floor_prev", "floor_curr",
    "new_units", "sold_cert", "sold_low", "sold_est", "sold_high", "flag_wipe", "flag_long_gap",
    "prefix_violations", "anomaly_increases", "classifier_version",
]


def classify_transitions(prev: Snapshot, curr: Snapshot, params: Optional[ClassifierParams] = None) -> TransitionResult:
    params = params or ClassifierParams()
    if prev.source != curr.source:
        raise ValueError(f"snapshots come from different sources: {prev.source!r} vs {curr.source!r}")
    if prev.kind != "commodity" or curr.kind != "commodity":
        raise NotImplementedError(
            "classifier v1 only handles fungible commodity books. Per-listing (kind='auction') markets need a "
            "different model (no shared price floor to sweep)."
        )
    if curr.ts <= prev.ts:
        raise ValueError(f"curr.ts ({curr.ts}) must be after prev.ts ({prev.ts})")

    gap = (curr.ts - prev.ts).total_seconds() / 60.0

    # Only items both snapshots are authoritative for can show a real disappearance.
    scope = None
    if prev.covered_items is not None or curr.covered_items is not None:
        a = prev.covered_items if prev.covered_items is not None else frozenset(prev.book["item_id"].unique())
        b = curr.covered_items if curr.covered_items is not None else frozenset(curr.book["item_id"].unique())
        scope = a & b

    p = prev.book
    c = curr.book
    if scope is not None:
        p = p[p["item_id"].isin(scope)]
        c = c[c["item_id"].isin(scope)]

    p = p.sort_values(["item_id", "unit_price", "auction_id"], kind="mergesort").reset_index(drop=True)
    curr_qty = c.set_index("auction_id")["quantity"]
    p["curr_qty"] = p["auction_id"].map(curr_qty)

    gone = p["curr_qty"].isna()
    reduced = (~gone) & (p["curr_qty"] < p["quantity"])
    increased = (~gone) & (p["curr_qty"] > p["quantity"])
    same = (~gone) & (p["curr_qty"] == p["quantity"])

    # ---- sweep boundary: first untouched survivor in (price, auction_id) order -------------------------------------
    p["rank"] = p.groupby("item_id").cumcount()
    inf = np.inf
    bound_rank = p["rank"].where(same).groupby(p["item_id"]).transform("min").fillna(inf)
    bound_price = p["unit_price"].where(same).groupby(p["item_id"]).transform("min").fillna(inf)

    if params.fifo_within_tier:
        in_sweep = gone & (p["rank"] < bound_rank)
        tie = pd.Series(False, index=p.index)
        beyond_partial = reduced & (p["rank"] > bound_rank)
    else:
        in_sweep = gone & (p["unit_price"] < bound_price)
        tie = gone & (p["unit_price"] == bound_price)
        beyond_partial = reduced & (p["unit_price"] > bound_price)
    beyond = gone & ~in_sweep & ~tie

    lam = params.buckets.expiry_feasible_fraction(p["time_left"], gap)
    qty = p["quantity"].to_numpy(dtype=float)

    p["cert_q"] = np.where(reduced, p["quantity"] - p["curr_qty"].fillna(0), 0).astype(float)
    p["sweep_q"] = np.where(in_sweep, qty, 0.0)
    p["tie_q"] = np.where(tie, qty, 0.0)
    p["infeasible_q"] = np.where(in_sweep & (lam <= 0.0), qty, 0.0)
    p["est_q"] = p["sweep_q"].to_numpy() * (1.0 - lam) * (1.0 - params.cancel_share)
    p["gone_q"] = np.where(gone, qty, 0.0)
    p["lam"] = lam

    # ---- events (audit trail) ---------------------------------------------------------------------------------------
    label = np.select(
        [reduced, in_sweep, tie, beyond, increased],
        ["PARTIAL_FILL", "VANISHED_IN_SWEEP", "VANISHED_TIE", "VANISHED_BEYOND_SWEEP", "QUANTITY_INCREASED"],
        default="UNCHANGED",
    )
    p["event"] = label
    events = p.loc[p["event"] != "UNCHANGED", [
        "item_id", "auction_id", "unit_price", "time_left", "quantity", "curr_qty", "event", "lam",
    ]].rename(columns={"quantity": "prev_qty", "lam": "expiry_feasible"}).reset_index(drop=True)
    events.insert(0, "source", prev.source)

    # ---- per-item aggregation ---------------------------------------------------------------------------------------
    g = p.groupby("item_id")
    items = pd.DataFrame(
        {
            "prev_units": g["quantity"].sum(),
            "floor_prev": g["unit_price"].min(),
            "sold_cert": g["cert_q"].sum(),
            "sweep_units": g["sweep_q"].sum(),
            "tie_units": g["tie_q"].sum(),
            "infeasible_units": g["infeasible_q"].sum(),
            "est_raw": g["est_q"].sum(),
            "gone_units": g["gone_q"].sum(),
            "prefix_violations": beyond_partial.groupby(p["item_id"]).sum().astype(int),
            "anomaly_increases": increased.groupby(p["item_id"]).sum().astype(int),
        }
    )

    # new listings and the undercut allowance for sold_low
    new = c[~c["auction_id"].isin(p["auction_id"])]
    cap = p.loc[p["infeasible_q"] > 0].groupby("item_id")["unit_price"].max().rename("cap")
    relist_units = pd.Series(0.0, index=items.index)
    if len(new) and len(cap):
        nc = new.merge(cap, left_on="item_id", right_index=True, how="inner")
        relist_units = (
            nc.loc[nc["unit_price"] <= nc["cap"]].groupby("item_id")["quantity"].sum().reindex(items.index).fillna(0.0)
        )
    items["new_units"] = new.groupby("item_id")["quantity"].sum().reindex(items.index).fillna(0).astype(np.int64)
    items["curr_units"] = c.groupby("item_id")["quantity"].sum().reindex(items.index).fillna(0).astype(np.int64)
    items["floor_curr"] = c.groupby("item_id")["unit_price"].min().reindex(items.index)

    low = items["sold_cert"] + np.maximum(0.0, items["infeasible_units"] - params.relist_attribution * relist_units)
    high = items["sold_cert"] + items["sweep_units"] + items["tie_units"]
    est = items["sold_cert"] + items["est_raw"]
    items["sold_low"] = low
    items["sold_high"] = np.maximum(high, low)
    items["sold_est"] = np.minimum(np.maximum(est, low), items["sold_high"])

    removed_share = items["gone_units"] / items["prev_units"].where(items["prev_units"] > 0)
    items["flag_wipe"] = (removed_share >= params.wipe_threshold) & (items["prev_units"] >= params.wipe_min_units)
    items["flag_long_gap"] = gap > params.max_gap_minutes
    items["source"] = prev.source
    items["t0"] = prev.ts
    items["t1"] = curr.ts
    items["gap_minutes"] = gap
    items["classifier_version"] = CLASSIFIER_VERSION
    items = items.reset_index()[ITEM_COLUMNS]
    items["flag_wipe"] = items["flag_wipe"].astype(bool)

    return TransitionResult(prev.source, prev.ts, curr.ts, gap, events, items)


def aggregate_rates(
    intervals: pd.DataFrame,
    now: pd.Timestamp,
    window_hours: float = 24.0,
    exclude_flagged: bool = True,
) -> pd.DataFrame:
    """Rolls per-interval bounds up to a rate per (source, item) over the trailing window ending at `now`.

    Rates are per 24 hours of OBSERVED time (gaps in our own data are not extrapolated). `coverage` is the share
    of the window actually observed. Because listed-and-sold-inside-one-interval volume is invisible, these rates
    are a floor on true volume; see docs/DATA_CONTRACT.md ("blind spot").
    """
    now = pd.Timestamp(now)
    start = now - pd.Timedelta(hours=window_hours)
    w = intervals[(intervals["t1"] > start) & (intervals["t1"] <= now)]
    if exclude_flagged and len(w):
        w = w[~(w["flag_wipe"] | w["flag_long_gap"])]
    cols = ["source", "item_id", "daily_low", "daily_est", "daily_high", "observed_hours", "coverage", "n_intervals"]
    if w.empty:
        return pd.DataFrame(columns=cols)
    g = w.groupby(["source", "item_id"])
    hours = g["gap_minutes"].sum() / 60.0
    out = pd.DataFrame(
        {
            "daily_low": g["sold_low"].sum() / hours * 24.0,
            "daily_est": g["sold_est"].sum() / hours * 24.0,
            "daily_high": g["sold_high"].sum() / hours * 24.0,
            "observed_hours": hours,
            "coverage": (hours / window_hours).clip(upper=1.0),
            "n_intervals": g.size(),
        }
    ).reset_index()
    return out[cols]


def classify_series(snapshots: list, params: Optional[ClassifierParams] = None) -> pd.DataFrame:
    """Classifies every consecutive pair in a time-ordered list of snapshots from ONE source; returns item rows."""
    frames = []
    for a, b in zip(snapshots[:-1], snapshots[1:]):
        frames.append(classify_transitions(a, b, params).items)
    if not frames:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    return pd.concat(frames, ignore_index=True)
