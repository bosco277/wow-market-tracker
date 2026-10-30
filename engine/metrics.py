"""Price and liquidity metrics built on the ask-side book and the classifier's sales bounds.

Direction matters, so buy-side and sell-side metrics deliberately use different inputs:
- BUYING inputs: walking DEEPER into the book makes the expected cost WORSE (conservative). Size the depth from the
  HIGH end of expected sales (plus blind-spot correction).
- SELLING outputs: expecting MORE sales makes the realistic list price HIGHER (optimistic). Use the LOW or EST end.
All prices are integer copper per unit; floats appear only in averages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

AH_CUT = 0.05  # auction house cut on a sale (your ledger already assumes this)


# ---------------------------------------------------------------------------------------------------------------------
# Order-book primitives
# ---------------------------------------------------------------------------------------------------------------------
def ask_tiers(item_book: pd.DataFrame) -> pd.DataFrame:
    """Aggregates one item's listings into price tiers, cheapest first, with cumulative quantity."""
    if item_book is None or len(item_book) == 0:
        return pd.DataFrame({"price": pd.Series(dtype="int64"), "qty": pd.Series(dtype="int64"), "cum_qty": pd.Series(dtype="int64")})
    t = item_book.groupby("unit_price", as_index=False)["quantity"].sum().sort_values("unit_price")
    t = t.rename(columns={"unit_price": "price", "quantity": "qty"}).reset_index(drop=True)
    t["cum_qty"] = t["qty"].cumsum()
    return t


@dataclass(frozen=True)
class DepthQuote:
    units_requested: float
    units_filled: float
    vwap: float            # average copper per unit actually filled (nan if nothing filled)
    marginal_price: float  # price of the last unit filled
    wall_price: float      # price of the tier the cutoff lands in (== marginal_price)
    total_cost: float
    complete: bool         # False if the book was too shallow to fill the request


def depth_vwap(item_book: pd.DataFrame, depth_units: float, snap_to_tier: bool = False) -> DepthQuote:
    """Cost of buying `depth_units` by walking the ask book cheapest-first.

    Default is the EXACT partial tier at the cutoff. snap_to_tier=True widens the cutoff to include the whole tier it
    lands in; that makes the series jump whenever the cutoff crosses a wall, which is why exact is the default.
    """
    t = ask_tiers(item_book)
    if depth_units <= 0 or t.empty:
        return DepthQuote(float(depth_units), 0.0, float("nan"), float("nan"), float("nan"), 0.0, depth_units <= 0)
    qty = t["qty"].to_numpy(dtype=float)
    price = t["price"].to_numpy(dtype=float)
    cum_before = np.cumsum(qty) - qty
    take = np.clip(depth_units - cum_before, 0.0, qty)
    if snap_to_tier:
        touched = take > 0
        take = np.where(touched, qty, 0.0)
    filled = take.sum()
    if filled <= 0:
        return DepthQuote(float(depth_units), 0.0, float("nan"), float("nan"), float("nan"), 0.0, False)
    cost = float((take * price).sum())
    last = int(np.nonzero(take > 0)[0][-1])
    return DepthQuote(float(depth_units), float(filled), cost / filled, float(price[last]), float(price[last]), cost, bool(filled + 1e-9 >= depth_units))


def depth_vwap_table(book: pd.DataFrame, depth_by_item: pd.Series) -> pd.DataFrame:
    """Vectorised depth_vwap for many items at once (exact partial tier). depth_by_item is indexed by item_id."""
    b = book[["item_id", "unit_price", "quantity"]].copy()
    b = b[b["item_id"].isin(depth_by_item.index)]
    if b.empty:
        return pd.DataFrame(columns=["item_id", "units_filled", "vwap", "marginal_price", "floor"])
    b = b.sort_values(["item_id", "unit_price"], kind="mergesort")
    cum = b.groupby("item_id")["quantity"].cumsum()
    before = cum - b["quantity"]
    depth = b["item_id"].map(depth_by_item).astype(float)
    b["take"] = np.clip(depth - before, 0.0, b["quantity"].astype(float))
    b["cost"] = b["take"] * b["unit_price"]
    g = b.groupby("item_id")
    out = pd.DataFrame({"units_filled": g["take"].sum(), "cost": g["cost"].sum(), "floor": g["unit_price"].min()})
    marg = b[b["take"] > 0].groupby("item_id")["unit_price"].max()
    out["marginal_price"] = marg.reindex(out.index)
    out["vwap"] = out["cost"] / out["units_filled"].where(out["units_filled"] > 0)
    return out.drop(columns="cost").reset_index()


def target_depth(expected_units: float, blind_spot_correction: float = 1.0, safety_buffer: float = 1.0) -> float:
    """Units of book depth to average over for a BUY-side price: expected sales x blind-spot x buffer.

    `blind_spot_correction` scales observed sales up to true volume (sales of listings created and sold between
    snapshots are invisible). Estimate it from simulation (see tools/validate_classifier.py) and, better, from your
    own experiment log; 1.0 means "trust the observed floor".
    """
    return max(1.0, float(expected_units) * float(blind_spot_correction) * float(safety_buffer))


@dataclass(frozen=True)
class SellQuote:
    list_price: float
    units_ahead: float       # units queued at or below your price when you post
    expected_filled: float   # units expected to sell within the horizon
    complete: bool           # True if all my_units are expected to sell within the horizon
    undercut: bool           # True if the quote is floor-1 because every existing tier is too far down the queue


def sell_clearing_price(item_book: pd.DataFrame, expected_sales_units: float, my_units: float) -> SellQuote:
    """Highest realistic list price for `my_units` given how many units the market is expected to absorb.

    You queue BEHIND every listing priced at or below yours. Your units sell within the horizon only if
    (units ahead of you) + my_units <= expected sales. So list at the highest existing tier whose cumulative
    quantity still leaves room for you; if even the floor tier is too deep, undercut it by one copper.
    Never quotes above the current highest ask (no extrapolation beyond the observed book).
    """
    t = ask_tiers(item_book)
    my_units = float(my_units)
    if t.empty:
        return SellQuote(float("nan"), 0.0, 0.0, False, False)
    room = float(expected_sales_units) - my_units
    ok = t[t["cum_qty"] <= max(room, -1.0)] if room >= 0 else t.iloc[0:0]
    if len(ok):
        row = ok.iloc[-1]
        return SellQuote(float(row["price"]), float(row["cum_qty"]), my_units, True, False)
    floor = float(t["price"].iloc[0])
    filled = max(0.0, min(my_units, float(expected_sales_units)))
    return SellQuote(max(1.0, floor - 1.0), 0.0, filled, filled + 1e-9 >= my_units, True)


def round_trip_roi(buy: DepthQuote, sell: SellQuote, ah_cut: float = AH_CUT) -> float:
    """Net return of buying at the depth VWAP and selling at the clearing price after the AH cut (filled units only)."""
    if not buy.units_filled or np.isnan(buy.vwap) or np.isnan(sell.list_price):
        return float("nan")
    units = min(buy.units_filled, sell.expected_filled)
    if units <= 0:
        return float("nan")
    return float((sell.list_price * (1 - ah_cut) - buy.vwap) / buy.vwap)


# ---------------------------------------------------------------------------------------------------------------------
# Demand baselines from classifier intervals
# ---------------------------------------------------------------------------------------------------------------------
_COL = {"low": "sold_low", "est": "sold_est", "high": "sold_high"}


def daily_volume_quantile(
    intervals: pd.DataFrame, now: pd.Timestamp, lookback_days: int = 14, q: float = 0.75,
    assumption: str = "est", min_hours_per_day: float = 12.0,
) -> pd.DataFrame:
    """Quantile of per-day sales volume over the lookback (days with too little observed time are skipped)."""
    col = _COL[assumption]
    now = pd.Timestamp(now)
    w = intervals[(intervals["t1"] > now - pd.Timedelta(days=lookback_days)) & (intervals["t1"] <= now)
                  & ~(intervals["flag_wipe"] | intervals["flag_long_gap"])].copy()
    if w.empty:
        return pd.DataFrame(columns=["source", "item_id", "daily_q", "n_days"])
    w["day"] = w["t1"].dt.floor("D")
    d = w.groupby(["source", "item_id", "day"]).agg(sold=(col, "sum"), minutes=("gap_minutes", "sum")).reset_index()
    d = d[d["minutes"] / 60.0 >= min_hours_per_day]
    d["daily"] = d["sold"] / (d["minutes"] / 60.0) * 24.0
    g = d.groupby(["source", "item_id"])["daily"]
    return pd.DataFrame({"daily_q": g.quantile(q), "n_days": g.size()}).reset_index()


def expected_sales(
    intervals: pd.DataFrame, now: pd.Timestamp, horizon_hours: float, assumption: str = "est",
    trailing_hours: float = 24.0, seasonal_weight: float = 0.0, seasonal_weeks: int = 4,
    blind_spot_correction: float = 1.0,
) -> pd.DataFrame:
    """Expected units sold over the next `horizon_hours` per (source, item).

    trailing  = observed rate over the last `trailing_hours` x horizon.
    seasonal  = mean rate for the same hour-of-week over the last `seasonal_weeks` weeks (needs history).
    result    = (1 - w) * trailing + w * seasonal, where w = seasonal_weight (0 until you have weeks of data).
    Multiplied by blind_spot_correction (observed sales are a floor on true volume).
    """
    from .classifier import aggregate_rates

    col = _COL[assumption]
    now = pd.Timestamp(now)
    rates = aggregate_rates(intervals, now, window_hours=trailing_hours)
    if rates.empty:
        return pd.DataFrame(columns=["source", "item_id", "expected_units", "trailing_units", "seasonal_units"])
    daily = rates[{"low": "daily_low", "est": "daily_est", "high": "daily_high"}[assumption]]
    out = rates[["source", "item_id"]].copy()
    out["trailing_units"] = daily.to_numpy() * horizon_hours / 24.0
    out["seasonal_units"] = np.nan

    if seasonal_weight > 0:
        w = intervals[(intervals["t1"] > now - pd.Timedelta(weeks=seasonal_weeks)) & (intervals["t1"] <= now)
                      & ~(intervals["flag_wipe"] | intervals["flag_long_gap"])].copy()
        if len(w):
            w["rate_h"] = w[col] / (w["gap_minutes"] / 60.0)
            mid = w["t0"] + (w["t1"] - w["t0"]) / 2
            w["how"] = mid.dt.dayofweek * 24 + mid.dt.hour
            hrs = [(now + pd.Timedelta(hours=h)) for h in range(int(np.ceil(horizon_hours)))]
            hows = pd.Series([(t.dayofweek * 24 + t.hour) for t in hrs])
            by = w.groupby(["source", "item_id", "how"])["rate_h"].mean().unstack("how")
            seasonal = by.reindex(columns=sorted(set(hows))).apply(lambda r: sum(r.get(h, np.nan) for h in hows), axis=1)
            seasonal = seasonal.rename("seasonal_units").reset_index()
            out = out.drop(columns="seasonal_units").merge(seasonal, on=["source", "item_id"], how="left")
    blend = out["trailing_units"].copy()
    if seasonal_weight > 0 and "seasonal_units" in out:
        has = out["seasonal_units"].notna()
        blend[has] = (1 - seasonal_weight) * out.loc[has, "trailing_units"] + seasonal_weight * out.loc[has, "seasonal_units"]
    out["expected_units"] = blend * blind_spot_correction
    return out[["source", "item_id", "expected_units", "trailing_units", "seasonal_units"]]
