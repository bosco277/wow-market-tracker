"""A permanent day-by-day history per item (table `item_daily`).

The hourly table that `derive` publishes only keeps 8 days, which is too short to see whether items in a group have good and bad
days together. This module turns the hourly series into ONE ROW PER ITEM PER US-EASTERN DAY and merges it into what is already stored,
so the history only grows. A day can be re-computed later (for example after a partial first run); the row with more usable hours wins.

  item_id, day (YYYY-MM-DD, US Eastern), hours (usable hours that day), sold_day (units sold in a 24 hour day, estimated from the
  usable hours), price_day (median lowest price, gold), units_day (median units listed), flagged_hours (spike / blip / wipe / gap hours
  left out), engine_version

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

COLUMNS = ["item_id", "day", "hours", "sold_day", "price_day", "units_day", "flagged_hours", "engine_version"]


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
    h = fu.merge(iv, on=["item_id", "ts"], how="left").sort_values(["item_id", "ts"]).reset_index(drop=True)
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
    d = d.reset_index()
    d["item_id"] = d["item_id"].astype(int)
    d["engine_version"] = __version__
    return d[COLUMNS]


def merge_daily(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """History plus new days. For a day in both, the row with more usable hours wins (the new one if they tie)."""
    parts = []
    if old is not None and len(old):
        parts.append(old.assign(_src=0))
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
