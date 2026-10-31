"""Market rate, price bands, hour-of-week profiles, price walls and flip shelves for commodity items.

Everything here works on an item's LISTINGS (the API shows listings, not sales), so it carries over to any market whose
auction house publishes its book. Sales rates come from the classifier and are passed in as low / est / high units per
hour; they are a floor (the blind spot), and for very fast items a poor one. The functions are written so that the
conclusion can be checked for sensitivity to that: see `wall_report`.

Units: prices in a BOOK are copper (10,000 copper = 1 gold, 100 copper = 1 silver, the smallest price step). Series
columns that end in `_gold` are gold.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .itemview import item_history
from .schema import COPPER_PER_GOLD, Snapshot

SILVER = COPPER_PER_GOLD // 100
AH_CUT = 0.05


# ---------------------------------------------------------------------------------------------------------------------
# Hourly series for a few items
# ---------------------------------------------------------------------------------------------------------------------
def item_series(snapshots: Sequence[Snapshot], item_ids: Iterable[int], params=None) -> Dict[int, pd.DataFrame]:
    """item -> one row per snapshot: ts, floor_gold, units_listed, n_listings, sold_low/est/high (interval ending at ts), flags."""
    ids = sorted({int(i) for i in item_ids})
    snaps = sorted(snapshots, key=lambda s: s.ts)
    small = []
    for s in snaps:  # keep only the wanted items once, so the per-item work below is cheap
        b = s.book[s.book["item_id"].isin(ids)]
        small.append(Snapshot(s.source, s.ts, b.reset_index(drop=True), kind=s.kind, covered_items=frozenset(ids), meta=dict(s.meta)))
    return {i: item_history(small, i, params) for i in ids}


# ---------------------------------------------------------------------------------------------------------------------
# Spikes and price bands
# ---------------------------------------------------------------------------------------------------------------------
def flag_spikes(floor: pd.Series, factor: float = 1.5, window: int = 25, max_hours: int = 10,
                min_move_gold: float = 0.05) -> pd.Series:
    """+1 for hours inside a short upward spike, -1 inside a short dip, 0 otherwise.

    An hour is a candidate when the price is `factor` times (or 1/factor of) the centred rolling median. Consecutive
    candidates of the same sign form an episode; episodes longer than `max_hours` are a new price LEVEL, not a spike, and
    stay at 0. Short episodes are what a flip or a one-off dump looks like, so they are kept out of the market rate.
    A move smaller than `min_move_gold` (5 silver) is never a spike: on a 4-8 silver item one price step is 25-100%.
    """
    f = pd.Series(floor, dtype="float64").reset_index(drop=True)
    med = f.rolling(window, center=True, min_periods=max(3, window // 5)).median()
    ratio = f / med
    cand = pd.Series(0, index=f.index, dtype="int64")
    big = (f - med).abs() >= float(min_move_gold)
    cand[(ratio >= factor) & big] = 1
    cand[(ratio <= 1.0 / factor) & big] = -1
    out = pd.Series(0, index=f.index, dtype="int64")
    i = 0
    n = len(cand)
    while i < n:
        if cand.iloc[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and cand.iloc[j + 1] == cand.iloc[i]:
            j += 1
        if (j - i + 1) <= max_hours:
            out.iloc[i:j + 1] = cand.iloc[i]
        i = j + 1
    out.index = pd.Series(floor).index
    return out


def flag_blips(units_listed: pd.Series, max_snapshots: int = 2, drop_ratio: float = 0.30, recover_ratio: float = 0.70,
               window: int = 25) -> pd.Series:
    """True for snapshots where the feed itself looks wrong: the units listed fall to `drop_ratio` of normal for at most
    `max_snapshots` snapshots and are back to `recover_ratio` of normal right after. Real buyouts and restocks do not
    vanish and return that fast (a real one lasts hours). Eternal Crystal at Thu 10-29 18:33 ET (685k units -> 63k -> 685k)
    is the known example, and Undermine's chart shows the same drop. An episode that runs to the end of the data is NOT
    flagged, because we cannot yet tell whether it recovers."""
    u = pd.Series(units_listed, dtype="float64").reset_index(drop=True)
    med = u.rolling(window, center=True, min_periods=5).median()
    low = (u <= drop_ratio * med).to_numpy()
    out = np.zeros(len(u), dtype=bool)
    i, n = 0, len(u)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        if (j - i + 1) <= max_snapshots and j + 1 < n and u.iloc[j + 1] >= recover_ratio * med.iloc[i]:
            out[i:j + 1] = True
        i = j + 1
    return pd.Series(out, index=pd.Series(units_listed).index)


def price_bands(series: pd.DataFrame, spikes: Optional[pd.Series] = None, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)) -> dict:
    """Percentiles of the hourly lowest price with spike / dip hours left out, and where the latest price sits in them."""
    s = series.reset_index(drop=True)
    sp = spikes.reset_index(drop=True) if spikes is not None else pd.Series(0, index=s.index)
    f = s["floor_gold"].astype("float64")
    ok = f.notna() & (sp == 0)
    base = f[ok]
    if base.empty:
        return {}
    now = float(f.dropna().iloc[-1]) if f.notna().any() else float("nan")
    out = {f"p{int(q * 100)}": float(base.quantile(q)) for q in quantiles}
    out.update(n=int(len(base)), n_spike_hours=int((sp == 1).sum()), n_dip_hours=int((sp == -1).sum()), now=now,
               now_percentile=float((base <= now).mean()))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# TSM-style market value (a baseline to compare against, built from the book we hold)
# ---------------------------------------------------------------------------------------------------------------------
def tsm_style_market_value(book_item: pd.DataFrame, top_share: float = 0.30, soft_start: float = 0.15,
                           jump: float = 0.20, n_sd: float = 1.5) -> float:
    """TSM documents AuctionDB market value as: the cheapest 30% of auctions; past the first 15%, cut at the first jump of
    20%+ between neighbouring auctions; drop prices more than 1.5 standard deviations from the mean; average the rest.
    (TSM then smooths scans over 14 days with a half-life of about 2 days; that smoothing is NOT done here.) Returns copper."""
    p = np.sort(book_item["unit_price"].to_numpy(dtype="float64"))
    n = len(p)
    if n == 0:
        return float("nan")
    k = max(1, math.ceil(top_share * n))
    p = p[:k]
    start = math.ceil(soft_start * n)
    for i in range(max(1, start), len(p)):
        if p[i] >= p[i - 1] * (1.0 + jump):
            p = p[:i]
            break
    if len(p) > 2:
        m, sd = p.mean(), p.std()
        keep = p[np.abs(p - m) <= n_sd * sd]
        if len(keep):
            p = keep
    return float(p.mean())


# ---------------------------------------------------------------------------------------------------------------------
# Hour-of-week profile
# ---------------------------------------------------------------------------------------------------------------------
def profile(series: pd.DataFrame, spikes: Optional[pd.Series] = None, tz: str = "America/New_York",
            block_hours: int = 1, by: str = "dow", min_obs: int = 3, blips: Optional[pd.Series] = None) -> pd.DataFrame:
    """Typical price, listed quantity and estimated sales per hour, by time slot (medians; spike / dip hours excluded).

    by = "dow" (7 x 24 slots), "weekpart" (weekday / weekend), or "none" (hour of day only). `block_hours` pools hours
    (4 gives six blocks a day). `n_obs` is the number of hourly observations behind each slot and `thin` marks slots
    with fewer than `min_obs`: with one week of data every day-of-week slot has ONE observation, so such a profile is a
    draft until more weeks arrive. `weeks` (same on every row) is how many distinct calendar weeks contributed.

    Three guards against misleading slots (added after the first real run):
    - `price_vs_level` is each hour's price against that day's own level (centred 25-hour median), so a price that simply
      fell over the week does not look like an evening premium. `price_median` is the raw price and mixes in the trend.
    - `sold_share` is the slot's share of the day's estimated sales (it sums to 1 within a day). Hourly sales are lumpy,
      and a share shows where the volume sits without trusting any single huge hour.
    - hours flagged in `blips` (see `flag_blips`) are left out of price, quantity and sales."""
    s = series.reset_index(drop=True).copy()
    sp = spikes.reset_index(drop=True) if spikes is not None else pd.Series(0, index=s.index)
    t = pd.to_datetime(s["ts"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    s["dow"] = t.dt.dayofweek
    s["block"] = (t.dt.hour // block_hours) * block_hours
    s["weekpart"] = np.where(s["dow"] >= 5, "weekend", "weekday")
    keys = {"dow": ["dow", "block"], "weekpart": ["weekpart", "block"], "none": ["block"]}[by]
    weeks = int(t.dt.strftime("%G-%V").nunique())
    blip = blips.reset_index(drop=True).fillna(False).astype(bool) if blips is not None else pd.Series(False, index=s.index)
    f = s["floor_gold"].astype("float64")
    ref = f.where((sp == 0) & ~blip).rolling(25, center=True, min_periods=21).median()
    s["_rel"] = f / ref - 1.0
    typical = s[(sp == 0) & ~blip]
    g = typical.groupby(keys)
    out = g.agg(n_obs=("floor_gold", "count"), price_median=("floor_gold", "median"),
                price_p25=("floor_gold", lambda x: x.quantile(0.25)), price_p75=("floor_gold", lambda x: x.quantile(0.75)),
                units_median=("units_listed", "median"), price_vs_level=("_rel", "median"))
    bad = blip.copy()
    for c in ("flag_long_gap", "flag_wipe"):
        if c in s.columns:
            bad |= s[c].fillna(False).astype(bool)
    if "sold_est" in s.columns:
        out["sold_per_hour"] = s[~bad].groupby(keys)["sold_est"].mean()
    else:
        out["sold_per_hour"] = np.nan
    out = out.reset_index()
    day_keys = keys[:-1] or ["_all"]
    out["_all"] = 0
    tot = out.groupby(day_keys)["sold_per_hour"].transform("sum")
    out["sold_share"] = out["sold_per_hour"] / tot.where(tot > 0)
    out = out.drop(columns="_all")
    out["thin"] = out["n_obs"] < min_obs
    out["weeks"] = weeks
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Price wall and queue
# ---------------------------------------------------------------------------------------------------------------------
def _tiers(book_item: pd.DataFrame) -> pd.DataFrame:
    return (book_item.groupby("unit_price")["quantity"].sum().sort_index().rename("qty").reset_index()
            .assign(cum=lambda d: d["qty"].cumsum()))


def price_wall(book_item: pd.DataFrame, units: float) -> dict:
    """Walk up the book from the cheapest listing until `units` have been added: that price is the wall. Also returns
    the units at or below it and the volume-weighted average price of the first `units`."""
    t = _tiers(book_item)
    if t.empty:
        return dict(price=float("nan"), units_at_or_below=0, vwap=float("nan"), exhausted=True)
    units = max(float(units), 1.0)
    idx = int(np.searchsorted(t["cum"].to_numpy(), units, side="left"))
    exhausted = idx >= len(t)
    idx = min(idx, len(t) - 1)
    take = t.assign(take=lambda d: np.minimum(d["qty"], np.maximum(units - (d["cum"] - d["qty"]), 0)))
    vwap = float((take["unit_price"] * take["take"]).sum() / max(take["take"].sum(), 1e-9))
    return dict(price=float(t["unit_price"].iloc[idx]), units_at_or_below=int(t["cum"].iloc[idx]), vwap=vwap, exhausted=exhausted)


def queue_wait_hours(book_item: pd.DataFrame, my_price: float, velocity_per_hour: float,
                     inflow_cheaper_per_hour: float = 0.0, tier_order: str = "lifo") -> float:
    """Hours until a listing at `my_price` reaches the front. Newest-first (lifo): only units at LOWER prices are ahead of
    you. Oldest-first (fifo): units at your price are ahead too. New cheaper listings keep joining the queue ahead of
    you, so the drain is velocity minus that inflow; if it is zero or negative the wait is infinite."""
    p = book_item["unit_price"]
    ahead = float(book_item.loc[p < my_price, "quantity"].sum() + (book_item.loc[p == my_price, "quantity"].sum() if tier_order == "fifo" else 0))
    drain = float(velocity_per_hour) - float(inflow_cheaper_per_hour)
    if ahead <= 0:
        return 0.0
    return ahead / drain if drain > 0 else float("inf")


def wall_report(book_item: pd.DataFrame, velocity_per_hour: Dict[str, float], horizon_hours: float = 4.0,
                trade_units: float = 500.0, duration_hours: float = 12.0, deep_multiple: float = 20.0,
                inflow_cheaper_per_hour: float = 0.0) -> dict:
    """The wall at the low / est / high sales rate, how long a listing AT the est wall would wait under each, and two
    plain-language flags:

    deep_market      units at or below the est wall are at least `deep_multiple` times the trade you would make, so your
                     own buying or selling hardly moves it.
    velocity_matters False when `trade_units` listed at the est wall would FULLY sell (units ahead of you plus your own)
                     within `duration_hours` even at the LOW rate (our rate is a floor), so a precise sales rate would
                     not change what you do. `wait_*` is only the time to reach the front of the line.
    """
    out: dict = {"horizon_hours": horizon_hours}
    walls = {}
    for k in ("low", "est", "high"):
        v = float(velocity_per_hour[k])
        walls[k] = price_wall(book_item, v * horizon_hours)
        out[f"wall_{k}"] = walls[k]["price"]
        out[f"units_{k}"] = walls[k]["units_at_or_below"]
    est_price = walls["est"]["price"]
    waits = {k: queue_wait_hours(book_item, est_price, float(velocity_per_hour[k]), inflow_cheaper_per_hour) for k in ("low", "est", "high")}
    out.update({f"wait_{k}": w for k, w in waits.items()})
    # Time to FULLY sell: the units ahead of you plus your own `trade_units` all have to be bought.
    ahead = float(book_item.loc[book_item["unit_price"] < est_price, "quantity"].sum())
    sell = {}
    for k in ("low", "est", "high"):
        drain = float(velocity_per_hour[k]) - float(inflow_cheaper_per_hour)
        sell[k] = (ahead + float(trade_units)) / drain if drain > 0 else float("inf")
    out.update({f"sell_hours_{k}": h for k, h in sell.items()})
    out["vwap_est"] = walls["est"]["vwap"]
    out["deep_market"] = bool(walls["est"]["units_at_or_below"] >= deep_multiple * trade_units)
    out["velocity_matters"] = bool(sell["low"] > duration_hours)
    return out


def reach_stats(series: pd.DataFrame, level_gold: float, horizon_hours: float = 12.0, max_gap_hours: float = 2.0) -> dict:
    """From history: starting at any hour when the lowest price was BELOW `level_gold`, how often did it reach the level
    within `horizon_hours`, and after how long? A listing priced at the level sells when the lowest price gets there, and
    this already includes every refill of the cheap side. Windows that cross a hole in our data are skipped. Hourly
    snapshots can miss a brief touch, so the share is slightly conservative."""
    s = series.dropna(subset=["floor_gold"]).reset_index(drop=True)
    ts = pd.to_datetime(s["ts"]).to_numpy()
    f = s["floor_gold"].to_numpy(dtype="float64")
    reached, hours, n = 0, [], 0
    for i in range(len(s)):
        if f[i] >= level_gold:
            continue
        hit, blocked, last_span = None, False, 0.0
        for j in range(i + 1, len(s)):
            if (ts[j] - ts[j - 1]) / np.timedelta64(1, "h") > max_gap_hours:
                blocked = True            # a hole in our data: we cannot say what happened
                break
            span = (ts[j] - ts[i]) / np.timedelta64(1, "h")
            if span > horizon_hours:
                break                     # watched the whole horizon without reaching it
            last_span = span
            if f[j] >= level_gold:
                hit = span
                break
        else:
            if last_span < horizon_hours:
                blocked = True            # the data ends before the horizon is up
        if hit is None and blocked:
            continue
        n += 1
        if hit is not None:
            reached += 1
            hours.append(hit)
    return dict(n=n, share_reached=(reached / n if n else float("nan")),
                median_hours=(float(np.median(hours)) if hours else float("nan")))


# ---------------------------------------------------------------------------------------------------------------------
# Flip shelves
# ---------------------------------------------------------------------------------------------------------------------
def find_shelf(book_item: pd.DataFrame, min_gap: float = 0.30, max_cost_gold: float = 150.0, min_net_margin: float = 0.25,
               ah_cut: float = AH_CUT) -> List[dict]:
    """Shelves of cheap listings worth buying: the units below a price gap of `min_gap` or more, costing at most
    `max_cost_gold` in total. The relist price is the next tier (newest-first puts you first at that price). Net margin is
    relist price x (1 - cut) over the average price paid, minus 1. Best net profit first. Prices in gold in the output."""
    t = _tiers(book_item)
    out: List[dict] = []
    if len(t) < 2:
        return out
    price, qty = t["unit_price"].to_numpy(dtype="float64"), t["qty"].to_numpy(dtype="float64")
    for k in range(1, len(t)):
        if price[k] / price[k - 1] - 1.0 < min_gap:
            continue
        units = qty[:k].sum()
        cost = float((price[:k] * qty[:k]).sum())
        if cost / COPPER_PER_GOLD > max_cost_gold:
            continue
        avg = cost / units
        relist = price[k]
        margin = relist * (1 - ah_cut) / avg - 1.0
        if margin < min_net_margin:
            continue
        out.append(dict(units=int(units), cost_gold=cost / COPPER_PER_GOLD, avg_price_gold=avg / COPPER_PER_GOLD,
                        next_price_gold=relist / COPPER_PER_GOLD, gap=float(price[k] / price[k - 1] - 1.0),
                        net_margin=float(margin), net_profit_gold=float((relist * (1 - ah_cut) - avg) * units / COPPER_PER_GOLD)))
    return sorted(out, key=lambda d: -d["net_profit_gold"])
