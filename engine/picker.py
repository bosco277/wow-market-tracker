"""Picks the items for the buy/list experiment from your own recent data.

What a good test item looks like
- It SELLS (so a listing of ours gets bought within hours, not weeks) but not so fast that our purchase is lost in noise.
- Its price is STEADY (a test item whose price lurches around tells us about the lurching, not about sales).
- The BOOK IS DEEP enough that our purchase doesn't empty the front of it.
- The whole position is CHEAP. You get most of the money back when you relist; the real cost is the 5% AH cut plus
  the buy/sell spread. Expect to lose roughly 5-10% of the money in play, not all of it.
- We choose a spread across slow / medium / fast sellers, because the blind-spot question is different for each.

Sizing (`buy_units`): about `detect_factor` times the units the market normally sells in ONE snapshot interval. A
purchase that size should stand out from the usual hourly sales, which is what lets us check whether the classifier
credits the units we know we bought.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .classifier import ClassifierParams, aggregate_rates, classify_series
from .schema import COPPER_PER_GOLD, Snapshot


@dataclass(frozen=True)
class PickerConfig:
    window_hours: float = 72.0
    min_observed_hours: float = 20.0
    min_coverage: float = 0.5            # share of the window that was actually observed (holes in our data count against it)
    min_daily_est: float = 100.0         # units/day: slower than this and we wait too long for sales
    max_daily_est: float = 8000.0
    max_price_cv: float = 0.15           # std/mean of the hourly floor price
    detect_factor: float = 2.0           # buy this many "normal hourly sales"
    min_units: int = 10
    max_position_gold: float = 150.0     # most money in play for ONE item at once
    min_floor_gold: float = 1.0          # below this a 1-copper undercut is meaningless and the AH rounding dominates
    max_units: int = 300                 # most units you have to buy and post for one item (bag space, clicking)
    min_depth_multiple: float = 3.0      # units within +10% of the floor must be >= this x buy_units
    n_items: int = 6
    snapshot_interval_hours: float = 1.0
    max_top_interval_share: float = 0.5  # lumpy demand: one single hour may not be more than this share of all sales
    min_active_share: float = 0.3        # share of hours with at least one estimated sale


def item_stats(snapshots: Sequence[Snapshot], now: Optional[pd.Timestamp] = None, cfg: PickerConfig = PickerConfig(),
               params: Optional[ClassifierParams] = None) -> pd.DataFrame:
    """One row per item from the last `window_hours` of snapshots (one source)."""
    snaps = sorted(snapshots, key=lambda s: s.ts)
    if len(snaps) < 3:
        raise ValueError("need at least 3 snapshots")
    now = pd.Timestamp(now) if now is not None else snaps[-1].ts
    intervals = classify_series(snaps, params)
    rates = aggregate_rates(intervals, now, cfg.window_hours)
    if rates.empty:
        return pd.DataFrame()

    start = now - pd.Timedelta(hours=cfg.window_hours)
    rows = []
    for s in snaps:
        if start < s.ts <= now:
            f = s.book.groupby("item_id")["unit_price"].min()
            rows.append(pd.DataFrame({"ts": s.ts, "item_id": f.index, "floor": f.to_numpy()}))
    floors = pd.concat(rows, ignore_index=True)
    fs = floors.groupby("item_id")["floor"].agg(floor_median="median", floor_std="std", floor_mean="mean", n_floor="count")
    fs["price_cv"] = (fs["floor_std"] / fs["floor_mean"]).fillna(0.0)

    w = intervals[(intervals["t1"] > start) & (intervals["t1"] <= now) & ~(intervals["flag_wipe"] | intervals["flag_long_gap"])].copy()
    w["active"] = w["sold_est"] >= 1.0
    g = w.groupby("item_id")
    lump = pd.DataFrame({
        "top_interval_share": (g["sold_est"].max() / g["sold_est"].sum().where(g["sold_est"].sum() > 0)).fillna(1.0),
        "active_share": g["active"].mean(),
    })

    last = [s for s in snaps if s.ts <= now][-1]
    book = last.book
    cur_floor = book.groupby("item_id")["unit_price"].min().rename("floor_now")
    merged = book.merge(cur_floor, on="item_id")
    near = merged[merged["unit_price"] <= merged["floor_now"] * 1.10]
    depth = near.groupby("item_id")["quantity"].sum().rename("depth_10pct")
    listings = book.groupby("item_id").size().rename("n_listings")

    out = rates.drop(columns=["source"]).set_index("item_id").join([fs, cur_floor, depth, listings, lump], how="inner")
    out["hourly_high"] = out["daily_high"] / 24.0 * cfg.snapshot_interval_hours
    out["hourly_est"] = out["daily_est"] / 24.0 * cfg.snapshot_interval_hours
    return out.reset_index()


def _round_units(n: float) -> int:
    """Round up to a number that is easy to type: 10s up to 100, 50s up to 1000, then 100s."""
    n = max(1.0, n)
    step = 10 if n <= 100 else 50 if n <= 1000 else 100
    return int(math.ceil(n / step) * step)


def pick(stats: pd.DataFrame, cfg: PickerConfig = PickerConfig(), exclude: Sequence[int] = ()) -> pd.DataFrame:
    """Applies the filters and returns the shortlist (most useful first). Adds `reject_reason` rows only via `explain`."""
    if stats.empty:
        return stats
    df = stats.copy()
    df = df[~df["item_id"].isin(list(exclude))]
    df["buy_units"] = [_round_units(max(cfg.min_units, cfg.detect_factor * h)) for h in df["hourly_high"]]
    df["position_gold"] = df["buy_units"] * df["floor_now"] / COPPER_PER_GOLD
    ok = (
        (df["observed_hours"] >= cfg.min_observed_hours)
        & (df["coverage"] >= cfg.min_coverage)
        & df["daily_est"].between(cfg.min_daily_est, cfg.max_daily_est)
        & (df["price_cv"] <= cfg.max_price_cv)
        & (df["position_gold"] <= cfg.max_position_gold)
        & (df["floor_now"] >= cfg.min_floor_gold * COPPER_PER_GOLD)
        & (df["buy_units"] <= cfg.max_units)
        & (df["top_interval_share"] <= cfg.max_top_interval_share)
        & (df["active_share"] >= cfg.min_active_share)
        & (df["depth_10pct"] >= cfg.min_depth_multiple * df["buy_units"])
    )
    cand = df[ok].copy()
    if cand.empty:
        return cand

    # spread across slow / medium / fast sellers, steadiest price first within each group
    if len(cand) < 3:
        cand["speed"] = "medium"
    else:
        share = cand["daily_est"].rank(method="first") / len(cand)
        cand["speed"] = pd.cut(share, [0, 1 / 3, 2 / 3, 1.0], labels=["slow", "medium", "fast"]).astype(str)
    cand = cand.sort_values(["speed", "price_cv"])
    per_group = max(1, cfg.n_items // 3)
    chosen: List[pd.DataFrame] = [g.head(per_group) for _, g in cand.groupby("speed", observed=True)]
    short = pd.concat(chosen)
    if len(short) < cfg.n_items:  # top up with the steadiest remaining items
        rest = cand[~cand["item_id"].isin(short["item_id"])].sort_values("price_cv")
        short = pd.concat([short, rest.head(cfg.n_items - len(short))])
    short = short.sort_values("daily_est", ascending=False).reset_index(drop=True)
    # for the "back of the queue" listing: join the price tier with about half a day of normal sales ahead of you
    short["back_lane_units_ahead"] = [_round_units(6 * h) for h in short["hourly_est"]]
    short["floor_gold"] = short["floor_now"] / COPPER_PER_GOLD
    short["median_floor_gold"] = short["floor_median"] / COPPER_PER_GOLD
    short["wowhead"] = "https://www.wowhead.com/item=" + short["item_id"].astype(str)
    cols = ["item_id", "speed", "floor_gold", "median_floor_gold", "price_cv", "daily_low", "daily_est", "daily_high",
            "hourly_est", "active_share", "top_interval_share", "buy_units", "back_lane_units_ahead", "position_gold", "depth_10pct", "n_listings", "observed_hours", "wowhead"]
    return short[cols]


def explain(stats: pd.DataFrame, cfg: PickerConfig = PickerConfig()) -> dict:
    """How many items each filter removes (so an empty shortlist can be understood and the thresholds eased)."""
    if stats.empty:
        return {"items_seen": 0}
    df = stats.copy()
    df["buy_units"] = [_round_units(max(cfg.min_units, cfg.detect_factor * h)) for h in df["hourly_high"]]
    df["position_gold"] = df["buy_units"] * df["floor_now"] / COPPER_PER_GOLD
    return {
        "items_seen": int(len(df)),
        "too_little_data": int(((df["observed_hours"] < cfg.min_observed_hours) | (df["coverage"] < cfg.min_coverage)).sum()),
        "too_slow": int((df["daily_est"] < cfg.min_daily_est).sum()),
        "too_fast": int((df["daily_est"] > cfg.max_daily_est).sum()),
        "price_too_jumpy": int((df["price_cv"] > cfg.max_price_cv).sum()),
        "position_too_expensive": int((df["position_gold"] > cfg.max_position_gold).sum()),
        "price_too_low": int((df["floor_now"] < cfg.min_floor_gold * COPPER_PER_GOLD).sum()),
        "too_many_units": int((df["buy_units"] > cfg.max_units).sum()),
        "lumpy_demand": int(((df["top_interval_share"] > cfg.max_top_interval_share) | (df["active_share"] < cfg.min_active_share)).sum()),
        "book_too_thin": int((df["depth_10pct"] < cfg.min_depth_multiple * df["buy_units"]).sum()),
    }


def publish_minutes(snapshots: Sequence[Snapshot]) -> pd.Series:
    """How often Blizzard's data refreshes at each minute of the hour. ONLY snapshots that carry Blizzard's own
    publish time count: older files only record when we fetched, which just echoes our fetch schedule."""
    real = [s for s in snapshots if s.meta.get("ts_source") != "scan_timestamp"]
    mins = pd.Series([s.ts.minute for s in real], dtype="int64")
    return mins.value_counts().sort_index()
