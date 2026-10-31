"""The derived layer: small tables the front end reads instead of raw snapshots.

Raw hourly snapshots are about 12 MB each. A page that re-reads and re-classifies days of them on every click would be slow
and would burn download allowances. `derive` runs once (a scheduled job), reads the recent snapshots and writes a handful
of small tables:

  items          one row per tracked item, latest state: price, typical band, TSM-style value, price wall, sales per day as a
                 low / best / high range, lumpiness, and a CONFIDENCE label with the reason
  item_hourly    one row per item per snapshot: floor, units listed, sold low / best / high, flags (spike, blip, hole)
  profile        hour-of-day profile (US Eastern by default) for the busiest items and the watched ones
  flips          buyout / flip candidates found in the recent snapshots
  calibration    ONE row for this run: how wide our ranges are and how the experiment checks came out, so the front end can
                 chart error bars and confidence over time

Facts (prices, depth, wall levels) and estimates (anything that uses the sales classifier) live in the same tables but every
estimate column starts with `daily_` or `sold_`, and every row carries `engine_version`, so a change to the classifier can be
recomputed without changing what the facts mean.

CONFIDENCE (a label per item, never a number pretending to be precise):
  Low     under 24 h of data; or lumpy and slow (testing showed these were overstated, e.g. Denshroom and Laribole sold
          nothing while first in line); or the range is more than 5x wide; or the experiment saw it sell nothing while
          first in line although a sale was predicted
  Medium  everything else, and the highest label an item can have without a check
  High    the experiment checked this item and our predicted selling time fell inside what was observed, with no check
          pointing the other way
So High is earned item by item. As experiment checks accumulate, items move up and the calibration history shows it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .classifier import ClassifierParams, aggregate_rates, classify_series
from .itemview import detect_flips
from .market import flag_blips, flag_spikes, price_bands, profile, tsm_style_market_value, wall_report
from .schema import COPPER_PER_GOLD, Snapshot


@dataclass(frozen=True)
class DeriveConfig:
    window_hours: float = 72.0        # rates and bands look back this far
    history_hours: float = 192.0      # hourly series kept (8 days)
    flip_hours: float = 96.0          # flips are searched in this many recent hours
    min_daily_high: float = 20.0      # items that cannot have sold this many a day are left out (watched items stay)
    max_items: int = 3000
    horizon_hours: float = 4.0        # price wall: hours of sales the wall must cover
    top_profile_items: int = 400
    tz: str = "America/New_York"
    lumpy_top_share: float = 0.5      # one hour carrying more than this share of all sales is lumpy
    lumpy_active_share: float = 0.3   # fewer than this share of hours with a sale is lumpy
    slow_daily_est: float = 500.0     # "slow" for the lumpy-and-slow rule
    wide_ratio: float = 5.0           # high / low above this is a wide range
    min_observed_hours: float = 24.0


def confidence(row: pd.Series, cfg: DeriveConfig, evidence: Optional[pd.Series] = None) -> tuple:
    """(label, reason) for one item row. See the module docstring for the rules."""
    low = max(float(row.get("daily_low", 0.0) or 0.0), 1.0)
    ratio = float(row.get("daily_high", 0.0) or 0.0) / low
    if float(row.get("observed_hours", 0.0) or 0.0) < cfg.min_observed_hours:
        return "Low", "less than 24 hours of data"
    lumpy = (float(row.get("top_interval_share", 0.0) or 0.0) > cfg.lumpy_top_share
             or float(row.get("active_share", 1.0) if row.get("active_share") == row.get("active_share") else 1.0) < cfg.lumpy_active_share)
    if evidence is not None and evidence.get("unsold_stayed", 0) > 0 and evidence.get("inside", 0) == 0:
        return "Low", "sold nothing while first in line in the experiment, although a sale was predicted"
    if lumpy and float(row.get("daily_est", 0.0) or 0.0) < cfg.slow_daily_est:
        return "Low", "lumpy and slow: slow items were overstated in testing"
    if ratio > cfg.wide_ratio:
        return "Low", f"range is {ratio:.0f}x wide"
    if evidence is not None and evidence.get("inside", 0) > 0 and evidence.get("too_fast", 0) + evidence.get("too_slow", 0) == 0:
        return "High", f"checked in the experiment {int(evidence['inside'])}x, prediction inside what was observed"
    if evidence is not None and evidence.get("too_fast", 0) + evidence.get("too_slow", 0) > 0:
        return "Medium", "experiment found the rate off by a few times; use the range, not the middle"
    if lumpy:
        return "Medium", "lumpy demand, not yet checked"
    return "Medium", "not yet checked against your own sales"


def _floors_and_units(snaps: Sequence[Snapshot], since: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for s in snaps:
        if s.ts <= since:
            continue
        g = s.book.groupby("item_id").agg(floor=("unit_price", "min"), units_listed=("quantity", "sum"), n_listings=("quantity", "size"))
        g = g.reset_index()
        g["ts"] = s.ts
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["item_id", "floor", "units_listed", "n_listings", "ts"])
    return pd.concat(parts, ignore_index=True)


def derive(snapshots: Sequence[Snapshot], cfg: DeriveConfig = DeriveConfig(), params: Optional[ClassifierParams] = None,
           watch: Sequence[int] = (), evidence: Optional[pd.DataFrame] = None, experiment_summary: Optional[dict] = None,
           names: Optional[Dict[int, str]] = None, include_flips: bool = True) -> Dict[str, pd.DataFrame]:
    snaps = sorted((s for s in snapshots if s.kind == "commodity"), key=lambda s: s.ts)
    if len(snaps) < 3:
        raise ValueError("need at least 3 commodity snapshots")
    now = snaps[-1].ts
    watch = {int(i) for i in watch}
    names = names or {}
    intervals = classify_series(snaps, params)
    rates = aggregate_rates(intervals, now, cfg.window_hours).drop(columns=["source"]).set_index("item_id")

    since = now - pd.Timedelta(hours=cfg.history_hours)
    fu = _floors_and_units(snaps, since)
    win = fu[fu["ts"] > now - pd.Timedelta(hours=cfg.window_hours)]

    # lumpiness over the window, flagged intervals left out
    w = intervals[(intervals["t1"] > now - pd.Timedelta(hours=cfg.window_hours)) & ~(intervals["flag_wipe"] | intervals["flag_long_gap"])].copy()
    w["active"] = w["sold_est"] >= 1.0
    g = w.groupby("item_id")
    tot = g["sold_est"].sum()
    lump = pd.DataFrame({"top_interval_share": (g["sold_est"].max() / tot.where(tot > 0)).fillna(1.0), "active_share": g["active"].mean()})

    last = snaps[-1].book
    listed_now = set(last["item_id"].unique())
    cand = rates[(rates["daily_high"] >= cfg.min_daily_high) | rates.index.isin(list(watch))]
    cand = cand[cand.index.isin(list(listed_now))]
    flo = last.groupby("item_id")["unit_price"].min() / COPPER_PER_GOLD
    cand = cand.assign(flow=cand["daily_est"] * flo.reindex(cand.index).fillna(0.0))
    keep_ids = list(cand.sort_values("flow", ascending=False).index[: cfg.max_items])
    keep_ids = sorted(set(keep_ids) | (watch & listed_now))

    book_now = last[last["item_id"].isin(keep_ids)]
    by_item = {i: b for i, b in book_now.groupby("item_id")}
    ev = evidence.set_index("item_id") if evidence is not None and not evidence.empty else pd.DataFrame()

    item_rows, hourly_parts, prof_parts = [], [], []
    iv_cols = ["item_id", "t1", "sold_low", "sold_est", "sold_high", "flag_wipe", "flag_long_gap"]
    iv_small = intervals[intervals["item_id"].isin(keep_ids)][iv_cols].rename(columns={"t1": "ts"})
    fu_keep = fu[fu["item_id"].isin(keep_ids)].copy()
    fu_keep["floor_gold"] = fu_keep["floor"] / COPPER_PER_GOLD
    hourly_all = fu_keep.merge(iv_small, on=["item_id", "ts"], how="left").sort_values(["item_id", "ts"]).reset_index(drop=True)
    # profile only for the busiest items and the watched ones
    flow_rank = cand.loc[cand.index.isin(keep_ids)].sort_values("flow", ascending=False)
    profile_ids = set(flow_rank.index[: cfg.top_profile_items]) | (watch & set(keep_ids))

    for item, ser in hourly_all.groupby("item_id"):
        ser = ser.reset_index(drop=True)
        sp = flag_spikes(ser["floor_gold"])
        bl = flag_blips(ser["units_listed"])
        ser["spike"] = sp.to_numpy()
        ser["blip"] = bl.to_numpy()
        hourly_parts.append(ser)
        in_win = ser["ts"] > now - pd.Timedelta(hours=cfg.window_hours)
        bands = price_bands(ser[in_win].reset_index(drop=True), pd.Series(sp.to_numpy()[in_win.to_numpy()]))
        bi = by_item.get(item)
        if bi is None or bi.empty:
            continue
        r = rates.loc[item] if item in rates.index else None
        per_hour = {k: (float(r[f"daily_{k if k != 'est' else 'est'}"]) / 24.0 if r is not None else 0.0) for k in ("low", "est", "high")}
        wr = wall_report(bi, per_hour, horizon_hours=cfg.horizon_hours)
        row = dict(
            item_id=int(item), name=names.get(int(item), ""), watched=int(item) in watch,
            floor_gold=float(bi["unit_price"].min()) / COPPER_PER_GOLD, units_listed=int(bi["quantity"].sum()), n_listings=int(len(bi)),
            tsm_value_gold=tsm_style_market_value(bi) / COPPER_PER_GOLD,
            band_p10=bands.get("p10", np.nan), band_p50=bands.get("p50", np.nan), band_p90=bands.get("p90", np.nan),
            now_percentile=bands.get("now_percentile", np.nan), spike_hours=bands.get("n_spike_hours", 0), dip_hours=bands.get("n_dip_hours", 0),
            wall_low_gold=wr["wall_low"] / COPPER_PER_GOLD, wall_est_gold=wr["wall_est"] / COPPER_PER_GOLD, wall_high_gold=wr["wall_high"] / COPPER_PER_GOLD,
            units_at_wall=wr["units_est"], deep_market=wr["deep_market"],
        )
        if r is not None:
            for c in ("daily_low", "daily_est", "daily_high", "observed_hours", "coverage"):
                row[c] = float(r[c])
        if item in lump.index:
            row["top_interval_share"] = float(lump.loc[item, "top_interval_share"])
            row["active_share"] = float(lump.loc[item, "active_share"])
        evi = ev.loc[item] if len(ev) and item in ev.index else None
        label, why = confidence(pd.Series(row), cfg, evi)
        row["confidence"], row["confidence_reason"] = label, why
        row["range_ratio"] = row.get("daily_high", np.nan) / max(row.get("daily_low", 0.0) or 0.0, 1.0)
        item_rows.append(row)
        if item in profile_ids:
            p = profile(ser.assign(ts=ser["ts"]), pd.Series(sp.to_numpy()), tz=cfg.tz, by="none", blips=pd.Series(bl.to_numpy()))
            p.insert(0, "item_id", int(item))
            prof_parts.append(p)

    items = pd.DataFrame(item_rows)
    item_hourly = pd.concat(hourly_parts, ignore_index=True) if hourly_parts else pd.DataFrame()
    prof = pd.concat(prof_parts, ignore_index=True) if prof_parts else pd.DataFrame()
    for df in (items, prof):
        if len(df):
            df["engine_version"] = __version__
    if len(item_hourly):
        item_hourly = item_hourly.drop(columns=["floor"]).rename(columns={"sold_low": "sold_low", "sold_est": "sold_est", "sold_high": "sold_high"})
        item_hourly["engine_version"] = __version__

    flips = pd.DataFrame()
    if include_flips:
        recent = [s for s in snaps if s.ts > now - pd.Timedelta(hours=cfg.flip_hours)]
        if len(recent) >= 3:
            ev_f = detect_flips(recent, params=params)
            if len(ev_f):
                flips = ev_f[ev_f["verdict"] == "flip / buyout"].copy().tail(2000)
                flips["engine_version"] = __version__

    cal = _calibration_row(items, snaps, now, experiment_summary)
    return {"items": items, "item_hourly": item_hourly, "profile": prof, "flips": flips, "calibration": cal}


def _calibration_row(items: pd.DataFrame, snaps: Sequence[Snapshot], now: pd.Timestamp, exp: Optional[dict]) -> pd.DataFrame:
    row: dict = dict(run_ts=pd.Timestamp.now("UTC").tz_localize(None), data_last_ts=now, engine_version=__version__,
                     snapshots=len(snaps), items_tracked=int(len(items)))
    if len(items):
        row.update({f"n_{k.lower()}": int((items["confidence"] == k).sum()) for k in ("High", "Medium", "Low")})
        ratio = items["range_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        row["median_range_ratio"] = float(ratio.median()) if len(ratio) else np.nan
        row["median_coverage"] = float(items["coverage"].median()) if "coverage" in items else np.nan
    else:
        row.update(n_high=0, n_medium=0, n_low=0, median_range_ratio=np.nan, median_coverage=np.nan)
    exp = exp or {}
    for k in ("listings", "rate_check_n", "rate_inside", "rate_pred_too_fast", "rate_pred_too_slow", "detection_n",
              "median_appeared_share", "invisible_count", "detectable_sold_out", "unsold_front_n", "unsold_front_stayed"):
        row[f"exp_{k}"] = exp.get(k, np.nan)
    return pd.DataFrame([row])
