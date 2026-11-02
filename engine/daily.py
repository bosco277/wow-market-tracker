"""A permanent day-by-day history per item (table `item_daily`).

The hourly table that `derive` publishes only keeps 8 days, which is too short to see whether items in a group have good and bad
days together. This module turns the hourly series into ONE ROW PER ITEM PER US-EASTERN DAY and merges it into what is already stored,
so the history only grows. A day can be re-computed later (for example after a partial first run); the row with more usable hours wins.

  item_id, day (YYYY-MM-DD, US Eastern), hours (usable hours that day), sold_day (units sold in a 24 hour day, estimated from the
  usable hours), price_day (median lowest price, gold), units_day (median units listed), flagged_hours (spike / blip / wipe / gap hours
  left out), price_p15 / price_p30 (median of the hourly price at which the cheapest 15% / 30% of the units listed can be bought: depth-based prices that one cheap
  listing cannot set; empty for days stored before v0.7.26), engine_version

Usable hour = no price spike, no listing blip, not a wipe or long gap, and a sales estimate exists. A day with fewer than 18 usable
hours is stored but not scored by the group test."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .derive import _floors_and_units
from .market import flag_blips, flag_spikes
from .schema import COPPER_PER_GOLD, Snapshot

COLUMNS = ["item_id", "day", "hours", "sold_day", "price_day", "units_day", "flagged_hours", "price_p15", "price_p30", "engine_version"]
DEPTH_SHARES = (0.15, 0.30)


def depth_prices(snaps: Sequence[Snapshot], ids: Iterable[int], shares=DEPTH_SHARES) -> pd.DataFrame:
    """Per item per snapshot: item_id, ts, p15, p30 (gold) = the lowest price at which the cheapest 15% / 30% of the UNITS listed have all been reached (walk up the book from the
    cheapest listing). One cheap listing sets the floor; it cannot set these unless it holds a large share of the units."""
    ids = {int(i) for i in ids}
    parts = []
    for s in snaps:
        b = s.book
        b = b[b["item_id"].isin(ids)][["item_id", "unit_price", "quantity"]]
        b = b[b["quantity"] > 0]
        if b.empty:
            continue
        b = b.sort_values(["item_id", "unit_price"], kind="mergesort")
        g = b.groupby("item_id")["quantity"]
        frac = (g.cumsum() / g.transform("sum")).to_numpy()
        out = pd.DataFrame({"item_id": sorted(b["item_id"].unique())}).set_index("item_id")
        for share in shares:
            hit = b[frac >= share - 1e-12]
            out[f"p{int(round(share * 100))}"] = hit.groupby("item_id")["unit_price"].first() / COPPER_PER_GOLD
        out = out.reset_index()
        out["ts"] = s.ts
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["item_id", "ts", "p15", "p30"])
    return pd.concat(parts, ignore_index=True)


def build_hourly(snaps: Sequence[Snapshot], intervals: pd.DataFrame, ids: Iterable[int]) -> pd.DataFrame:
    """One row per item per snapshot (floor in gold, units listed, estimated units sold in the interval before) with the same flags the
    hourly table uses: spike, blip, flag_wipe, flag_long_gap."""
    ids = {int(i) for i in ids}
    snaps = sorted(snaps, key=lambda s: s.ts)
    if len(snaps) < 2 or not ids:
        return pd.DataFrame()
    fu = _floors_and_units(snaps, snaps[0].ts - pd.Timedelta(hours=1))
    fu = fu[fu["item_id"].isin(ids)].copy()
    if fu.empty:
        return pd.DataFrame()
    fu["floor_gold"] = fu["floor"] / COPPER_PER_GOLD
    iv = intervals[intervals["item_id"].isin(ids)][["item_id", "t1", "sold_est", "flag_wipe", "flag_long_gap"]].rename(columns={"t1": "ts"})
    h = fu.merge(iv, on=["item_id", "ts"], how="left")
    dp = depth_prices(snaps, ids)
    if not dp.empty:
        h = h.merge(dp, on=["item_id", "ts"], how="left")
    h = h.sort_values(["item_id", "ts"]).reset_index(drop=True)
    spike = np.zeros(len(h), dtype=bool)
    blip = np.zeros(len(h), dtype=bool)
    for _, pos in h.groupby("item_id").indices.items():
        ser = h.iloc[pos]
        spike[pos] = flag_spikes(ser["floor_gold"].reset_index(drop=True)).to_numpy()
        blip[pos] = flag_blips(ser["units_listed"].reset_index(drop=True)).to_numpy()
    h["spike"], h["blip"] = spike, blip
    return h


def daily_table(h: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Collapse an hourly table (item_id, ts, floor_gold, units_listed, sold_est, spike, blip, flag_wipe, flag_long_gap) to one row per item per day."""
    if h is None or h.empty:
        return pd.DataFrame(columns=COLUMNS)
    s = h.copy()
    bad = np.zeros(len(s), dtype=bool)
    for c in ("spike", "blip", "flag_wipe", "flag_long_gap"):
        if c in s.columns:
            bad |= s[c].fillna(False).astype(bool).to_numpy()
    t = pd.to_datetime(s["ts"])
    t = (t.dt.tz_localize("UTC") if t.dt.tz is None else t).dt.tz_convert(tz)
    s["day"] = t.dt.strftime("%Y-%m-%d").to_numpy()
    s["flagged"] = bad.astype(int)
    ok = (~bad) & s["sold_est"].notna().to_numpy() & (s["floor_gold"].astype(float) > 0).to_numpy()
    s["sold"] = s["sold_est"].astype(float).clip(lower=0)
    use = s[ok]
    if use.empty:
        return pd.DataFrame(columns=COLUMNS)
    g = use.groupby(["item_id", "day"])
    d = pd.DataFrame({"hours": g.size(), "sold_day": g["sold"].mean() * 24.0, "price_day": g["floor_gold"].median(), "units_day": g["units_listed"].median()})
    d["flagged_hours"] = s.groupby(["item_id", "day"])["flagged"].sum().reindex(d.index).fillna(0).astype(int)
    for col, src in (("price_p15", "p15"), ("price_p30", "p30")):
        d[col] = g[src].median() if src in use.columns else np.nan
    d = d.reset_index()
    d["item_id"] = d["item_id"].astype(int)
    d["engine_version"] = __version__
    return d[COLUMNS]


def merge_daily(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """History plus new days. For a day in both, the row with more usable hours wins (the new one if they tie)."""
    parts = []
    if old is not None and len(old):
        parts.append(old.reindex(columns=COLUMNS).assign(_src=0))
    if new is not None and len(new):
        parts.append(new.assign(_src=1))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    m = pd.concat(parts, ignore_index=True)
    m = m.sort_values(["item_id", "day", "hours", "_src"]).drop_duplicates(["item_id", "day"], keep="last")
    return m.drop(columns="_src").sort_values(["item_id", "day"]).reset_index(drop=True)[COLUMNS]


def coverage(daily: pd.DataFrame, min_hours: int = 18) -> pd.DataFrame:
    """Per day: how many items have a complete day (min_hours or more usable hours)."""
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["day", "items", "complete_items", "median_hours"])
    g = daily.groupby("day")
    return pd.DataFrame({"items": g.size(), "complete_items": g["hours"].apply(lambda x: int((x >= min_hours).sum())), "median_hours": g["hours"].median()}).reset_index()
