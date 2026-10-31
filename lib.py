"""Data helpers for the front end. No Streamlit or Plotly in here, so they can be tested without a browser.

Rule for every number the app shows: it comes with a range and a confidence label. `rng` formats low-best-high; the
Confidence screen shows how many items are High / Medium / Low over time and how wide the ranges are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONF_ORDER = ["High", "Medium", "Low"]
CONF_COLOR = {"High": "#2E7D4F", "Medium": "#C28A1B", "Low": "#B5483A"}


def gsc(gold) -> str:
    """Gold as 12g 34s 56c."""
    if gold is None or gold != gold:
        return "-"
    c = int(round(float(gold) * 10000))
    g, s, cc = c // 10000, (c % 10000) // 100, c % 100
    parts = []
    if g:
        parts.append(f"{g:,}g")
    if s or g:
        parts.append(f"{s}s")
    parts.append(f"{cc}c")
    return " ".join(parts)


def short_num(x) -> str:
    if x is None or x != x:
        return "-"
    x = float(x)
    for lim, suf in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= lim:
            return f"{x / lim:.1f}{suf}"
    return f"{x:.0f}"


def rng(low, est, high) -> str:
    """'1.2k - 1.5k - 1.9k' (low - best - high)."""
    return f"{short_num(low)} - {short_num(est)} - {short_num(high)}"


def market_table(items: pd.DataFrame, min_floor_gold: float = 0.0, conf: tuple = tuple(CONF_ORDER),
                 text: str = "", watched_only: bool = False, sort: str = "gold_flow") -> pd.DataFrame:
    """The screener: one row per item, with gold flow (best-estimate units/day x price) and its low/high."""
    if items is None or items.empty:
        return pd.DataFrame()
    t = items.copy()
    t["gold_flow"] = t["daily_est"] * t["floor_gold"]
    t["gold_flow_low"] = t["daily_low"] * t["floor_gold"]
    t["gold_flow_high"] = t["daily_high"] * t["floor_gold"]
    t = t[(t["floor_gold"] >= min_floor_gold) & t["confidence"].isin(conf)]
    if watched_only:
        t = t[t["watched"]]
    if text:
        q = text.strip().lower()
        t = t[t["name"].fillna("").str.lower().str.contains(q, regex=False) | t["item_id"].astype(str).str.contains(q, regex=False)]
    return t.sort_values(sort, ascending=False).reset_index(drop=True)


def price_position(row: pd.Series) -> str:
    """Where today's lowest price sits inside the typical band, in words."""
    p = row.get("now_percentile")
    if p is None or p != p:
        return "unknown"
    if p <= 0.15:
        return "cheap for this item"
    if p >= 0.85:
        return "expensive for this item"
    return "normal"


def daily_ribbon(hourly: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Units sold per rolling `hours` window as low / best / high, from the hourly rows of ONE item.
    Rolling sums need a full window, so the first hours are blank. Intervals with a hole are NaN and make the window NaN."""
    h = hourly.sort_values("ts").reset_index(drop=True)
    out = pd.DataFrame({"ts": h["ts"]})
    for k in ("low", "est", "high"):
        out[k] = h[f"sold_{k}"].rolling(hours, min_periods=hours).sum()
    return out


def hole_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hours where the estimate is missing (no usable interval before it)."""
    return hourly[hourly["sold_est"].isna()][["ts"]]


def confidence_mix(cal: pd.DataFrame) -> pd.DataFrame:
    """Calibration history -> long table of (data_last_ts, tier, items) for the stacked chart."""
    if cal is None or cal.empty:
        return pd.DataFrame(columns=["data_last_ts", "tier", "items"])
    m = cal.melt(id_vars=["data_last_ts"], value_vars=["n_high", "n_medium", "n_low"], var_name="tier", value_name="items")
    m["tier"] = m["tier"].map({"n_high": "High", "n_medium": "Medium", "n_low": "Low"})
    return m.sort_values("data_last_ts").reset_index(drop=True)


def experiment_scorecard(cal: pd.DataFrame) -> dict:
    """Latest experiment numbers as plain sentences' ingredients."""
    if cal is None or cal.empty:
        return {}
    r = cal.sort_values("data_last_ts").iloc[-1]
    g = lambda k: (None if k not in r or r[k] != r[k] else int(r[k]))
    return dict(listings=g("exp_listings"), rate_n=g("exp_rate_check_n"), inside=g("exp_rate_inside"),
                too_fast=g("exp_rate_pred_too_fast"), too_slow=g("exp_rate_pred_too_slow"),
                stayed=g("exp_unsold_front_stayed"), unsold_n=g("exp_unsold_front_n"))


def freshness(cal: pd.DataFrame, now=None) -> tuple:
    """(hours since the last snapshot the tables were built from, label). Stale data is shown, never hidden."""
    if cal is None or cal.empty:
        return None, "no data"
    last = pd.Timestamp(cal["data_last_ts"].max())
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
    h = (now - last).total_seconds() / 3600.0
    return h, ("fresh" if h <= 3 else "stale" if h <= 12 else "very stale")


MARK = {"pass": "pass", "warn": "WARN", "fail": "FAIL"}
TEST_COLS = ["test_price", "test_speed", "test_size", "test_margin", "test_book"]


def shelf_table(shelves: pd.DataFrame, verdicts: tuple = ("candidate", "weak")) -> pd.DataFrame:
    """Flip shelves worth looking at, best first (the engine already sorts by fails, warns, then profit if undercut)."""
    if shelves is None or shelves.empty:
        return pd.DataFrame()
    return shelves[shelves["verdict"].isin(verdicts)].reset_index(drop=True)


def shelf_hours(row) -> str:
    """Hours to sell everything, as best-high-low rates: '4.3 h (3.3 - 6.5)'."""
    def f(x):
        return "never" if x != x or x == float("inf") else (f"{x:.1f}" if x < 100 else ">100")
    return f"{f(row['sell_hours_est'])} h ({f(row['sell_hours_high'])} - {f(row['sell_hours_low'])})"
