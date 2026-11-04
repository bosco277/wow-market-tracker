"""Data helpers for the front end. No Streamlit or Plotly in here, so they can be tested without a browser.

Rule for every number the app shows: it comes with a range and a confidence label. `rng` formats low-best-high; the
Confidence screen shows how many items are High / Medium / Low over time and how wide the ranges are.
"""
from __future__ import annotations

from typing import Optional

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
                 text: str = "", watched_only: bool = False, sort: str = "gold_flow", watch_ids=None) -> pd.DataFrame:
    """The screener: one row per item, with gold flow (best-estimate units/day x price) and its low/high."""
    if items is None or items.empty:
        return pd.DataFrame()
    t = items.copy()
    t["gold_flow"] = t["daily_est"] * t["floor_gold"]
    t["gold_flow_low"] = t["daily_low"] * t["floor_gold"]
    t["gold_flow_high"] = t["daily_high"] * t["floor_gold"]
    t = t[(t["floor_gold"] >= min_floor_gold) & t["confidence"].isin(conf)]
    if watch_ids is not None:                       # the viewer's own watch list (see parse_ids)
        t["watched"] = t["item_id"].isin(set(int(i) for i in watch_ids))
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


def parse_ids(text) -> list:
    """'123, 456 789' -> [123, 456, 789]. Anything that is not a whole number is ignored; duplicates are dropped, order kept."""
    out, seen = [], set()
    for tok in str(text or "").replace(";", ",").replace("\n", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) not in seen:
            seen.add(int(tok))
            out.append(int(tok))
    return out


def item_labels(items: pd.DataFrame) -> dict:
    """{item_id: 'Name (id)'} for search boxes: typing part of the name or the id finds it."""
    if items is None or items.empty:
        return {}
    names = items["name"].fillna("") if "name" in items else pd.Series("", index=items.index)
    return {int(i): (f"{n} ({int(i)})" if n else f"item {int(i)}") for i, n in zip(items["item_id"], names)}


def _inf_to_nan(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan)


def market_numbers(t: pd.DataFrame) -> pd.DataFrame:
    """The Market table with NUMBERS in numeric columns (gold as decimals, percentages as numbers) so every column sorts properly.
    Ranges are split into low / best / high columns."""
    return pd.DataFrame({
        "Item": t["name"].where(t["name"] != "", "(no name)"),
        "Item ID": t["item_id"].astype(int),
        "Watched": flag_mask(t, "watched"),
        "Lowest price (g)": t["floor_gold"].astype(float),
        "Vs typical": t.apply(price_position, axis=1),
        "Typical low p10 (g)": t["band_p10"].astype(float),
        "Typical high p90 (g)": t["band_p90"].astype(float),
        "Sold/day low": t["daily_low"].astype(float),
        "Sold/day best": t["daily_est"].astype(float),
        "Sold/day high": t["daily_high"].astype(float),
        "Gold/day low (g)": t["gold_flow_low"].astype(float),
        "Gold/day best (g)": t["gold_flow"].astype(float),
        "Gold/day high (g)": t["gold_flow_high"].astype(float),
        "Price wall (g)": t["wall_est_gold"].astype(float),
        "Confidence": t["confidence"],
        "Why": t["confidence_reason"],
        "_id": t["item_id"].astype(int),
    })


def flip_numbers(sh: pd.DataFrame, rates: Optional[dict] = None) -> pd.DataFrame:
    """The Flips table with numbers in numeric columns, ranges split into separate columns, and the checks as counts."""
    rates = rates or {}
    pass_n = sh.apply(lambda r: sum(r[c] == "pass" for c in TEST_COLS if c in r and r[c] != "n/a"), axis=1)
    fail_n = sh.apply(lambda r: sum(r[c] == "fail" for c in TEST_COLS if c in r), axis=1)
    warn_n = sh.apply(lambda r: sum(r[c] == "warn" for c in TEST_COLS if c in r), axis=1)
    hit = lambda i: sh["verdict"].map(lambda x: (rates[x][i] * 100.0) if x in rates and rates[x][i] == rates[x][i] else np.nan)
    f = lambda c: sh[c].astype(float)
    return pd.DataFrame({
        "Item": sh["name"].where(sh["name"] != "", "(no name)"),
        "Item ID": sh["item_id"].astype(int),
        "Verdict": sh["verdict"],
        "Checks passed": pass_n.astype(int),
        "Checks failed": fail_n.astype(int),
        "Checks warned": warn_n.astype(int),
        "Watch out for": sh.apply(shelf_watch, axis=1),
        "Past 4h hit (%)": hit(0),
        "Past 12h hit (%)": hit(1),
        "Past undercut (%)": hit(2),
        "Buy (units)": sh["units"].astype(int),
        "Avg cost (g)": f("avg_price_gold"),
        "Break-even (g)": f("breakeven_gold"),
        "Relist at (g)": f("target_gold"),
        "Normal price median (g)": f("long_median_gold"),
        "Normal p10 (g)": f("long_p10_gold"),
        "Normal p90 (g)": f("long_p90_gold"),
        "History (days)": f("long_days"),
        "Sold/day low": f("daily_low"),
        "Sold/day best": f("daily_est"),
        "Sold/day high": f("daily_high"),
        "Net margin (%)": f("net_margin") * 100.0,
        "If 10% under (%)": f("net_margin_stress") * 100.0,
        "Profit at target (g)": f("profit_gold"),
        "Profit if 10% under (g)": f("profit_stress_gold"),
        "Hours to sell (best)": _inf_to_nan(f("sell_hours_est")),
        "Hours to sell (fast case)": _inf_to_nan(f("sell_hours_high")),
        "Hours to sell (slow case)": _inf_to_nan(f("sell_hours_low")),
        "Cost (g)": f("cost_gold"),
        "Item confidence": sh["confidence"],
        "_id": sh["item_id"].astype(int),
    })


DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOURS = ["12am"] + [f"{h}am" for h in range(1, 12)] + ["12pm"] + [f"{h}pm" for h in range(1, 12)]


def time_profiles(h: pd.DataFrame, tz: str = "America/New_York") -> dict:
    """Day-of-week x hour-of-day grids (7 rows Mon..Sun, 24 columns) for ONE item, from its hourly rows
    (ts, floor_gold, units_listed, sold_est, spike, blip). All times are US Eastern. Each cell is the MEDIAN of the readings that
    fall in it, and `n` says how many readings that was (about one per week, so with under a month of data most cells rest on 1 to 3).

    price   lowest price on the auction house (gold)
    units   total units listed
    sold    sales velocity: the best estimate of units sold per hour (the middle of the low - best - high range we use)
    n       readings behind each cell
    Hours flagged as a price spike or a listing blip are left out."""
    if h is None or h.empty:
        return {}
    s = h.copy()
    bad = pd.Series(flag_mask(s, "spike") | flag_mask(s, "blip"), index=s.index)
    t = pd.to_datetime(s["ts"])
    t = (t.dt.tz_localize("UTC") if t.dt.tz is None else t).dt.tz_convert(tz)
    s["dow"], s["hour"] = t.dt.dayofweek, t.dt.hour
    s["price"] = s["floor_gold"].astype(float)
    s["units"] = s["units_listed"].astype(float)
    s["sold"] = s["sold_est"].astype(float) if "sold_est" in s else np.nan
    g = s[~bad]
    if g.empty:
        return {}
    shape = lambda col, how: g.groupby(["dow", "hour"])[col].agg(how).unstack("hour").reindex(index=range(7), columns=range(24))
    return dict(price=shape("price", "median"), units=shape("units", "median"), sold=shape("sold", "median"),
                n=shape("price", "count").fillna(0).astype(int), hours=int(len(g)))


def past_events_table(f: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """One item's past buyout events (the engine's `flips` table) with plain headers and numbers in numeric columns."""
    if f is None or f.empty:
        return pd.DataFrame()
    f = f.sort_values("t1", ascending=False).reset_index(drop=True)
    t = pd.to_datetime(f["t1"])
    t = (t.dt.tz_localize("UTC") if t.dt.tz is None else t).dt.tz_convert(tz)

    def back(r):
        how = r.get("ended_by", "")
        if how == "reverted":
            return f"yes, after {r['revert_hours']:.1f} h"
        if how == "hole":
            return "unknown (gap in our data)"
        if how == "end_of_data":
            return f"not yet (watched {r.get('followup_hours', 0):.1f} h)"
        return "-"
    out = pd.DataFrame({
        "When (US Eastern)": t.dt.strftime("%a %b %d %I:%M %p"),
        "Lowest price before (g)": f["floor_before_gold"].astype(float),
        "Lowest price after (g)": f["floor_after_gold"].astype(float),
        "Price rose by (%)": (f["jump"].astype(float) - 1.0) * 100.0,
        "Units bought out": f["units_removed"].astype(float).round(0),
    })
    if "ended_by" in f:
        out["Back to old price?"] = f.apply(back, axis=1)
    if "likely_flip" in f:
        out["Looks like a flipper"] = f["likely_flip"].map(lambda v: "yes" if bool(v) else "no")
    return out


# ---------------------------------------------------------------- forecast ledger (engine/forecast_ledger, written once a day by mode forecast_ledger)
# The ledger holds one row per (day the guess was made, item, horizon, method). `actual` stays empty until the target day is complete, then is filled in once.
# Everything here is plain pandas so it can be tested without a browser. `ledger_scoreboard` repeats the engine's scoreboard rules (tests compare the two).

LEDGER_COLUMNS = ["made_day", "item_id", "horizon", "method", "guess", "lo50", "hi50", "lo80", "hi80", "origin", "jump_bin", "target_day", "actual", "made_at", "engine_version"]
METHODS = ("naive", "ema25", "kinked")
METHOD_NAME = {"naive": "Stays put", "ema25": "Recent average", "kinked": "Pullback model"}
METHOD_COLOR = {"naive": "#8A9BA3", "ema25": "#C28A1B", "kinked": "#2B4C59"}
HORIZON_NAME = {1: "1 day", 3: "3 days", 7: "7 days"}
LEVELS = ("under 1g", "1g to 10g", "10g to 100g", "100g and up")
FORECAST_COLOR = "#3A6EA5"
MIN_TRUST_CASES = 100            # a scoreboard row built on fewer graded cases is a first look, not a verdict


def clean_ledger(led: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The ledger with every column present and typed (days as 'YYYY-MM-DD' text, prices as numbers). Empty frame when there is none or it is not a ledger."""
    if led is None or led.empty or not {"item_id", "horizon", "method", "guess", "made_day", "target_day"}.issubset(led.columns):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    d = led.reindex(columns=LEDGER_COLUMNS).copy()
    for c in ("made_day", "target_day"):
        d[c] = pd.to_datetime(d[c], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ("guess", "lo50", "hi50", "lo80", "hi80", "origin", "actual"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["item_id", "horizon", "made_day", "target_day"])
    d = d[d["guess"] > 0].copy()
    d["item_id"] = d["item_id"].astype(int)
    d["horizon"] = d["horizon"].astype(int)
    return d.reset_index(drop=True)


def ledger_counts(led: pd.DataFrame) -> dict:
    """Headline numbers: rows recorded, items, how many are graded / waiting, and the days they cover."""
    if led is None or led.empty:
        return dict(made=0, items=0, graded=0, waiting=0, first_origin=None, last_origin=None, last_target=None)
    done = led["actual"].notna() & (led["actual"] > 0)
    return dict(made=len(led), items=int(led["item_id"].nunique()), graded=int(done.sum()), waiting=int((~done).sum()),
                first_origin=led["made_day"].min(), last_origin=led["made_day"].max(), last_target=led.loc[done, "target_day"].max() if done.any() else None)


def _pct(x: float) -> float:
    return (10.0 ** x - 1.0) * 100.0


def _graded(led: pd.DataFrame) -> pd.DataFrame:
    """Graded rows with the miss (in log10, so +10% and -10% weigh alike), whether the real price fell in the 50% / 80% range, and the naive method's miss on the same case."""
    g = led[led["actual"].notna() & (led["actual"] > 0)].copy()
    if g.empty:
        return g
    g["E"] = np.log10(g["actual"]) - np.log10(g["guess"])
    g["ae"] = g["E"].abs()
    g["in50"] = ((g["actual"] >= g["lo50"]) & (g["actual"] <= g["hi50"])).astype(float)
    g["in80"] = ((g["actual"] >= g["lo80"]) & (g["actual"] <= g["hi80"])).astype(float)
    g["w80"] = np.log10(g["hi80"]) - np.log10(g["lo80"])
    key = ["made_day", "item_id", "horizon"]
    nv = g[g["method"] == "naive"].drop_duplicates(key).set_index(key)["ae"].rename("en")
    g = g.join(nv, on=key)
    g["level"] = np.array(LEVELS, dtype=object)[np.digitize(np.log10(g["origin"].to_numpy(float)), [0.0, 1.0, 2.0])]
    return g


def _score(df: pd.DataFrame, by: list) -> pd.DataFrame:
    rows = []
    for keys, q in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ae, en = q["ae"].to_numpy(), q["en"].to_numpy()
        ok = np.isfinite(en)
        rows.append({**dict(zip(by, keys)), "cases": len(q), "items": int(q["item_id"].nunique()),
                     "miss_pct": _pct(float(np.median(ae))), "avg_pct": _pct(float(ae.mean())),
                     "vs_naive": float(ae[ok].sum() / en[ok].sum()) if ok.any() and en[ok].sum() > 0 else np.nan,
                     "bias_pct": _pct(float(np.median(q["E"]))), "hit50": float(q["in50"].mean()), "hit80": float(q["in80"].mean()),
                     "width80_pct": _pct(float(np.median(q["w80"])))})
    return pd.DataFrame(rows)


def ledger_scoreboard(led: pd.DataFrame) -> dict:
    """{graded: n, waiting: per-horizon frame, overall: per (horizon, method), levels: per (horizon, price level, method)}. Same numbers as the engine's own scoreboard:
    vs_naive compares each method with 'stays put' on exactly the same graded cases (below 1.00 = closer than staying put); hit50 / hit80 should sit near 0.50 / 0.80."""
    if led is None or led.empty:
        return {"graded": 0, "waiting": pd.DataFrame()}
    waiting = led[led["actual"].isna()].groupby("horizon").agg(rows=("item_id", "size"), items=("item_id", "nunique"),
                                                               first_target=("target_day", "min"), last_target=("target_day", "max")).reset_index()
    g = _graded(led)
    out = {"graded": len(g), "waiting": waiting}
    if g.empty:
        return out
    order = {m: i for i, m in enumerate(METHODS)}
    o = _score(g, ["horizon", "method"])
    out["overall"] = o.sort_values(["horizon", "method"], key=lambda s: s.map(order) if s.name == "method" else s).reset_index(drop=True)
    lv = _score(g, ["horizon", "level", "method"])
    lv["_l"] = lv["level"].map({k: i for i, k in enumerate(LEVELS)})
    lv["_m"] = lv["method"].map(order)
    out["levels"] = lv.sort_values(["horizon", "_l", "_m"]).drop(columns=["_l", "_m"]).reset_index(drop=True)
    return out


def score_table(sc: pd.DataFrame) -> pd.DataFrame:
    """A scoreboard frame (overall or by price level) with plain headers. Percent columns hold percent numbers (80.0 = 80%), so they sort and format as numbers."""
    if sc is None or sc.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Looking ahead": sc["horizon"].map(lambda h: HORIZON_NAME.get(int(h), f"{int(h)} days"))})
    if "level" in sc:
        out["Price level"] = sc["level"].to_numpy()
    out["Method"] = sc["method"].map(lambda m: METHOD_NAME.get(m, m)).to_numpy()
    out["Cases"] = sc["cases"].astype(int).to_numpy()
    out["Items"] = sc["items"].astype(int).to_numpy()
    out["Typical miss (%)"] = sc["miss_pct"].to_numpy()
    out["Average miss (%)"] = sc["avg_pct"].to_numpy()
    out["Closer than staying put?"] = sc["vs_naive"].to_numpy()
    out["Inside 50% range (%)"] = (sc["hit50"] * 100.0).to_numpy()
    out["Inside 80% range (%)"] = (sc["hit80"] * 100.0).to_numpy()
    out["Real price vs guess, typical (%)"] = sc["bias_pct"].to_numpy()
    out["Typical 80% range width (%)"] = sc["width80_pct"].to_numpy()
    out["Enough cases?"] = np.where(sc["cases"].to_numpy() >= MIN_TRUST_CASES, "yes", f"first look (under {MIN_TRUST_CASES})")
    return out


def accuracy_over_time(led: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """For one horizon: running totals as graded days arrive, per method. One row per (target day, method) with the cumulative number of graded cases, the share inside the 50% and 80% ranges
    (percent numbers) and the running 'closer than staying put' ratio over the same cases (below 1.00 is better)."""
    cols = ["target_day", "method", "cases", "inside50", "inside80", "vs_naive"]
    g = _graded(led) if led is not None and not led.empty else pd.DataFrame()
    if g.empty:
        return pd.DataFrame(columns=cols)
    g = g[g["horizon"] == int(horizon)]
    parts = []
    for m, q in g.groupby("method", sort=False):
        q = q.assign(a=q["ae"].where(q["en"].notna()), b=q["en"])
        by = q.groupby("target_day").agg(n=("ae", "size"), s50=("in50", "sum"), s80=("in80", "sum"), a=("a", "sum"), b=("b", "sum")).sort_index().cumsum()
        parts.append(pd.DataFrame({"target_day": by.index.to_numpy(), "method": m, "cases": by["n"].astype(int).to_numpy(),
                                   "inside50": (by["s50"] / by["n"] * 100.0).to_numpy(), "inside80": (by["s80"] / by["n"] * 100.0).to_numpy(),
                                   "vs_naive": np.where(by["b"].to_numpy() > 0, by["a"].to_numpy() / np.where(by["b"].to_numpy() > 0, by["b"].to_numpy(), 1.0), np.nan)}))
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


def day_to_utc(day: str, hour: int = 12, tz: str = "America/New_York") -> pd.Timestamp:
    """A forecast is for a whole US Eastern day (the median of its hourly lows); on the hourly chart (UTC, no time zone) it is drawn at midday Eastern of that day."""
    return pd.Timestamp(f"{day} {hour:02d}:00").tz_localize(tz).tz_convert("UTC").tz_localize(None)


def item_methods(led: pd.DataFrame, item_id: int) -> list:
    """Methods that have written at least one forecast for this item, in the usual order (stays put first)."""
    if led is None or led.empty:
        return []
    have = set(led.loc[led["item_id"] == int(item_id), "method"])
    return [m for m in METHODS if m in have]


def item_forecast(led: pd.DataFrame, item_id: int, method: str) -> pd.DataFrame:
    """The newest forecast for one item and one method: one row per horizon, sorted short to long. Empty when there is none."""
    if led is None or led.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    d = led[(led["item_id"] == int(item_id)) & (led["method"] == method)]
    if d.empty:
        return d
    return d[d["made_day"] == d["made_day"].max()].sort_values("horizon").reset_index(drop=True)


def item_daily_points(led: pd.DataFrame, item_id: int) -> pd.DataFrame:
    """The item's daily prices as far as the ledger knows them: the price on each day a forecast was made from (`origin`) and the real price of each graded target day (`actual`, which wins
    when both exist). Columns day, price."""
    if led is None or led.empty:
        return pd.DataFrame(columns=["day", "price"])
    d = led[led["item_id"] == int(item_id)]
    a = d[d["actual"].notna() & (d["actual"] > 0)][["target_day", "actual"]].rename(columns={"target_day": "day", "actual": "price"})
    o = d[d["origin"].notna() & (d["origin"] > 0)][["made_day", "origin"]].rename(columns={"made_day": "day", "origin": "price"})
    both = pd.concat([a.assign(_k=0), o.assign(_k=1)], ignore_index=True).sort_values(["day", "_k"])
    return both.drop_duplicates("day", keep="first").drop(columns="_k").sort_values("day").reset_index(drop=True)


def item_history_table(led: pd.DataFrame, item_id: int, limit: int = 90) -> pd.DataFrame:
    """Every forecast written for one item, newest first, with the real price once it is known and whether it landed inside the ranges."""
    if led is None or led.empty:
        return pd.DataFrame()
    d = led[led["item_id"] == int(item_id)].copy()
    if d.empty:
        return pd.DataFrame()
    d["_m"] = d["method"].map({m: i for i, m in enumerate(METHODS)})
    d = d.sort_values(["made_day", "horizon", "_m"], ascending=[False, True, True]).head(limit)
    known = d["actual"].notna() & (d["actual"] > 0)
    in50 = (d["actual"] >= d["lo50"]) & (d["actual"] <= d["hi50"])
    in80 = (d["actual"] >= d["lo80"]) & (d["actual"] <= d["hi80"])
    result = np.where(~known, "waiting", np.where(in50, "inside 50% range", np.where(in80, "inside 80% range", np.where(d["actual"] > d["hi80"], "outside: higher", "outside: lower"))))
    return pd.DataFrame({
        "Made from day": d["made_day"].to_numpy(), "For day": d["target_day"].to_numpy(),
        "Looking ahead": d["horizon"].map(lambda h: HORIZON_NAME.get(int(h), f"{int(h)} days")).to_numpy(),
        "Method": d["method"].map(lambda m: METHOD_NAME.get(m, m)).to_numpy(),
        "Price then (g)": d["origin"].to_numpy(), "Guess (g)": d["guess"].to_numpy(),
        "50% range low (g)": d["lo50"].to_numpy(), "50% range high (g)": d["hi50"].to_numpy(),
        "80% range low (g)": d["lo80"].to_numpy(), "80% range high (g)": d["hi80"].to_numpy(),
        "Real price (g)": d["actual"].to_numpy(), "Result": result})
