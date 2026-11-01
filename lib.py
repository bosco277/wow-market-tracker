"""Data helpers for the front end. No Streamlit or Plotly in here, so they can be tested without a browser.

Rule for every number the app shows: it comes with a range and a confidence label. `rng` formats low-best-high; the
Confidence screen shows how many items are High / Medium / Low over time and how wide the ranges are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONF_ORDER = ["High", "Medium", "Low"]
CONF_COLOR = {"High": "#2E7D4F", "Medium": "#C28A1B", "Low": "#B5483A"}


def flag_mask(df: pd.DataFrame, col: str) -> np.ndarray:
    """True/False array for a flag column, whatever dtype it came back as from parquet (object, nullable, missing, all-NaN)."""
    if df is None or col not in df.columns or len(df) == 0:
        return np.zeros(0 if df is None else len(df), dtype=bool)
    return df[col].fillna(False).astype(bool).to_numpy()


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
        t = t.loc[flag_mask(t, "watched")]
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


MARK = {"pass": "pass", "warn": "WARN", "fail": "FAIL", "n/a": "n/a"}
TEST_COLS = ["test_price", "test_speed", "test_size", "test_margin", "test_regime", "test_book"]


SHELF_NUMERIC = ["units", "avg_price_gold", "breakeven_gold", "target_gold", "net_margin", "net_margin_stress", "profit_gold", "profit_stress_gold",
                 "cost_gold", "sell_hours_low", "sell_hours_est", "sell_hours_high", "long_median_gold", "long_p10_gold", "long_p90_gold",
                 "long_days", "daily_low", "daily_est", "daily_high"]


def ensure_shelf_columns(sh: pd.DataFrame) -> pd.DataFrame:
    """The shelves table written by an older engine run may lack columns the app now shows. Fill them with blanks so the app
    still opens; the Flips screen tells the user to run derive again."""
    if sh is None or sh.empty:
        return sh
    sh = sh.copy()
    for c in SHELF_NUMERIC:
        if c not in sh.columns:
            sh[c] = np.nan
    for c in TEST_COLS:
        if c not in sh.columns:
            sh[c] = "n/a"
    for c, v in (("name", ""), ("confidence", ""), ("verdict", "weak")):
        if c not in sh.columns:
            sh[c] = v
    return sh


def normal_price_text(median, p10, p90, days) -> str:
    """'1g 20s 0c (90s - 1g 50s, 5d)', or '-' when the engine run did not have it."""
    if median is None or median != median:
        return "-"
    return f"{gsc(median)} ({gsc(p10)} - {gsc(p90)}, {days:.0f}d)"


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


# What each flip check means, in plain words: (short name, what it asks, what a WARN/FAIL says)
TEST_INFO = {
    "test_price": ("Price history", "Has the lowest price actually been at the relist price recently, and is that price inside the item's usual band (not a spike)?",
                   "the relist price has rarely or never been seen lately, or it is above the item's usual band"),
    "test_speed": ("Sells fast enough", "Hours until your units, and any cheap units left ahead of you, are bought, at the sales rate seen while the price was that high. Must fit in about a day.",
                   "it may take more than a day to sell, or the sales rate was not seen at this price"),
    "test_size": ("Fits your limits", "Does the whole shelf fit your unit and gold limits? A partial buy leaves cheap units ahead of you in the queue.",
                  "you could only buy part of the shelf, so cheap units stay ahead of yours"),
    "test_margin": ("Still profitable if undercut", "Net margin after the 5% cut if you have to sell 10% below the relist price.",
                    "the profit disappears if you sell 10% under the relist price"),
    "test_regime": ("Normal price, not a spike", "Compares what you pay and the relist price with the item's longer history. Fails if you would pay well above normal, the price spiked in the last two days and is coming back down, or the relist price is far above normal.",
                    "the price looks temporary (a spike or reset) or sits well above the item's normal level"),
    "test_book": ("Complete listing snapshot", "Is the number of units listed now normal? A thin snapshot creates fake price gaps.",
                  "the snapshot looks incomplete, so the gap may not be real"),
}


def shelf_checks(row) -> str:
    """'5 of 6 pass' (checks a table from an older engine run does not have are left out of the count)."""
    have = [c for c in TEST_COLS if c in row and row[c] != "n/a"]
    return f"{sum(row[c] == 'pass' for c in have)} of {len(have)} pass"


def shelf_watch(row) -> str:
    """The checks that did not pass, in words ('FAIL: Normal price, not a spike; WARN: Price history')."""
    out = []
    for label, mark in (("FAIL", "fail"), ("WARN", "warn")):
        names = [TEST_INFO[c][0] for c in TEST_COLS if c in row and row[c] == mark]
        if names:
            out.append(f"{label}: " + ", ".join(names))
    return "; ".join(out) if out else "none"


def backtest_table(summ: pd.DataFrame) -> pd.DataFrame:
    """The backtest summary in words and percentages, for the Flips screen."""
    if summ is None or summ.empty:
        return pd.DataFrame()
    pct = lambda v: "-" if v != v else f"{v * 100:.0f}%"
    return pd.DataFrame({
        "Group": summ["group"],
        "Shelves seen": summ["n"].astype(int),
        "Reached relist price within 4 h": summ["reached_4h"].map(pct),
        "Within 12 h": summ["reached_12h"].map(pct),
        "Median hours to get there": summ["median_hours"].map(lambda v: "-" if v != v else f"{v:.1f}"),
        "Price 4 h later vs relist price": summ["median_floor_ratio_4h"].map(lambda v: "-" if v != v else f"{v:.2f}x"),
        "Undercut risk (new cheap listings within 2 h)": summ["undercut_risk"].map(pct),
    })


def backtest_rates(summ: pd.DataFrame) -> dict:
    """{'candidate': (reached 4 h, reached 12 h, undercut risk), 'weak': ...} from the backtest summary, for the Flips table."""
    out = {}
    if summ is None or summ.empty:
        return out
    for v in ("candidate", "weak"):
        r = summ[summ["group"] == f"verdict: {v}"]
        if len(r):
            r = r.iloc[0]
            out[v] = (r["reached_4h"], r["reached_12h"], r["undercut_risk"])
    return out


def past_hit_rate(verdict: str, rates: dict) -> str:
    """'61% / 82% (undercut 48%)': how often similar shelves reached the relist price in 4 h / 12 h in the replay."""
    if verdict not in rates:
        return "-"
    a, b, u = rates[verdict]
    pct = lambda v: "-" if v != v else f"{v * 100:.0f}%"
    return f"{pct(a)} / {pct(b)} (undercut {pct(u)})"
