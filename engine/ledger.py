"""Forecast ledger: every guess is written down with its range BEFORE the day happens, then graded against what really happened.

Why: the bake-off (engine/forecast.py) is a backtest. This is the live record. Each run (after `daily`) stamps guesses for 1, 3 and 7 days ahead for every item that has enough days of
OUR OWN prices (table `item_daily`, price_day = the median of the day's usable hourly lowest prices), and grades the older guesses whose day has now arrived. Three methods are kept side by side
so the scoreboard says which one is actually closest on live data:

  naive   the price stays where it was on the last complete day (the bar everything must clear)
  ema25   an exponential average of the last days (weight 0.25 on the newest)
  kinked  the retail-trained pooled regression from the bake-off (small moves stay put, spikes pull back), fitted on Undermine's multi-year history and applied to OUR series. This is also the
          test of whether what was learned on retail carries over to a series it was not trained on (the Forever question).

Ranges: the 50% and 80% range of a guess comes from that method's own past misses on the last 180 days of the Undermine history, kept apart for five levels of recent jumpiness (the same
rule as the bake-off). They are a PRIOR: Undermine's price is not exactly our price, and the scoreboard shows whether they hold (hit50 / hit80 near 0.50 / 0.80) on our own series.

Rules that keep the record honest:
  * a guess uses only days up to and including its origin day (the latest well-covered complete day); anything later in the table is cut off before any model sees it;
  * a guess is not written if the item's target day is already complete in the table (nothing is graded the moment it is made);
  * a row, once written, is never changed except for filling in `actual` when the target day is complete (18+ usable hours); a (origin day, item, horizon, method) is written once;
  * eligibility: naive needs 5 observed days in the last 14, ema25 needs 7, kinked needs 12 (its averages need time to settle).
Only group-level numbers are ever printed; the ledger itself lives in the private dataset (engine/forecast_ledger)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import __version__
from . import forecast as FO

COLUMNS = ["made_day", "item_id", "horizon", "method", "guess", "lo50", "hi50", "lo80", "hi80", "origin", "jump_bin", "target_day", "actual", "made_at", "engine_version"]
KEY = ["made_day", "item_id", "horizon", "method"]
HORIZONS = FO.HORIZONS
METHODS = ("naive", "ema25", "kinked")
MIN_COMPLETE = 18            # a day counts only with this many usable hours
MIN_SHARE = 0.5              # the origin is the latest day on which at least this share of the best-covered day's items are complete
NEED = {"naive": 5, "ema25": 7, "kinked": 12}      # observed days within the last WINDOW days, the origin included
WINDOW = 14
OWN_MIN_PERIODS = 4          # our own series are short: jumpiness needs only 4 day-to-day changes (the calibration on Undermine's history uses 7)
CAL_DAYS = 180               # the last 180 days of the Undermine history calibrate the ranges (and hold out the pooled fit)


def _pct(x: float) -> float:
    return (10.0 ** x - 1.0) * 100.0


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _day_str(x) -> pd.Series:
    return pd.to_datetime(x).dt.strftime("%Y-%m-%d")


def _sig(Y: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    return pd.DataFrame(Y.T).diff().rolling(window, min_periods=min_periods).std().to_numpy().T


# ---------------------------------------------------------------- our own series

def own_grid(daily: pd.DataFrame, min_hours: int = MIN_COMPLETE):
    """(item ids, days, log10 gold): an items x days grid of price_day for COMPLETE days only (min_hours or more usable hours); NaN elsewhere. None when there is nothing."""
    if daily is None or daily.empty or not {"item_id", "day", "hours", "price_day"}.issubset(daily.columns):
        return None
    d = daily[daily["hours"] >= min_hours][["item_id", "day", "price_day"]].copy()
    d = d[d["price_day"] > 0]
    if d.empty:
        return None
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    d["item_id"] = d["item_id"].astype(int)
    d = d.drop_duplicates(["item_id", "day"], keep="last")
    M = d.pivot(index="item_id", columns="day", values="price_day")
    days = pd.date_range(M.columns.min(), M.columns.max(), freq="D")
    M = M.reindex(columns=days)
    return list(M.index.astype(int)), days, np.log10(M.to_numpy(float))


def origin_index(Y: np.ndarray) -> Optional[int]:
    """Column of the latest well-covered day (at least half as many items as the best-covered day), or None."""
    cnt = np.isfinite(Y).sum(axis=0)
    if len(cnt) == 0 or cnt.max() == 0:
        return None
    ok = np.nonzero(cnt >= MIN_SHARE * cnt.max())[0]
    return int(ok[-1])


# ---------------------------------------------------------------- calibration on the Undermine history

def calibrate(hist: pd.DataFrame, hs: Sequence[int] = HORIZONS, cal_days: int = CAL_DAYS) -> Optional[dict]:
    """Learn, from the Undermine daily history only: the kinked regression for each horizon (fitted on everything before the last `cal_days` days) and, for every (method, horizon), the
    q10 / q25 / q75 / q90 of its misses over the last `cal_days` days for each of five jumpiness levels. {"betas", "q", "edges", "cases", "first", "last", "items"} or None when the history is thin."""
    g = FO.build_grid(hist, "price")
    if g is None:
        return None
    ids, days, Y = g
    n, T = Y.shape
    if T < max(FO.MIN_SPAN, cal_days + FO.WARMUP + 30):
        return None
    wk = days.dayofweek.to_numpy()
    obs = np.isfinite(Y)
    first = np.where(obs.any(1), obs.argmax(1), T)
    age_ok = (np.arange(T)[None, :] - first[:, None]) >= FO.WARMUP
    sig = FO.volatility(Y)
    static, base = FO.static_forecasts(Y, wk, hs)
    feat = FO._features(Y, base, None)
    c_fit = T - cal_days
    out = {"betas": {}, "q": {}, "edges": None, "cases": {}, "first": days[c_fit], "last": days[-1], "items": n}
    min_cases = FO.MIN_BIN * FO.VOL_BINS
    for h in hs:
        valid = obs & age_ok & np.concatenate([obs[:, h:], np.zeros((n, h), bool)], axis=1)
        fit_cols = np.arange(0, max(c_fit - h + 1, 0), FO.FIT_STEP)
        cal_cols = np.arange(max(c_fit - h + 1, 0), T - h, 1)
        if len(cal_cols) == 0:
            continue
        guesses = {"naive": Y[:, cal_cols], "ema25": base["L25"][:, cal_cols]}
        beta = FO.pooled_fit(Y, base, feat, wk, valid, h, fit_cols, kind="kinked")
        if beta is not None:
            out["betas"][h] = beta
            guesses["kinked"] = FO.pooled_predict(beta, base, feat, wk, h, cal_cols, "kinked")
        Yc, sc = Y[:, cal_cols + h], sig[:, cal_cols]
        vc = valid[:, cal_cols] & np.isfinite(sc)
        if out["edges"] is None:
            if vc.sum() < min_cases:
                continue
            out["edges"] = np.quantile(sc[vc], np.linspace(0, 1, FO.VOL_BINS + 1)[1:-1])
        for m, G in guesses.items():
            E = Yc - G
            mask = vc & np.isfinite(E)
            if mask.sum() < min_cases:
                continue
            out["q"][(m, h)] = FO._bin_quantiles(E[mask], np.digitize(sc[mask], out["edges"]))
            out["cases"][(m, h)] = int(mask.sum())
    if out["edges"] is None or not out["q"]:
        return None
    return out


# ---------------------------------------------------------------- writing guesses

def make_forecasts(own, cal: dict, hs: Sequence[int] = HORIZONS, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """One row per (item, horizon, method) for the latest well-covered day of OUR series. Nothing after that day is used. Empty when there is no origin or no calibration."""
    if own is None or cal is None:
        return _empty()
    ids, days, Yfull = own
    t0 = origin_index(Yfull)
    if t0 is None:
        return _empty()
    ids = np.asarray(ids, dtype=int)
    Y = Yfull[:, :t0 + 1]                      # strictly causal: later days are cut off before any model sees them
    wk = days[:t0 + 1].dayofweek.to_numpy()
    static, base = FO.static_forecasts(Y, wk, hs)
    feat = FO._features(Y, base, None)
    sig = _sig(Y, WINDOW, OWN_MIN_PERIODS)[:, t0]
    seen = np.isfinite(Y[:, max(t0 - WINDOW + 1, 0):t0 + 1]).sum(axis=1)
    at_origin = np.isfinite(Y[:, t0])
    bins = np.digitize(np.where(np.isfinite(sig), sig, 0.0), cal["edges"])
    made_day = days[t0].strftime("%Y-%m-%d")
    stamp = (now if now is not None else pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = []
    for h in hs:
        guesses = {"naive": Y[:, t0], "ema25": base["L25"][:, t0]}
        beta = cal["betas"].get(h)
        if beta is not None:
            guesses["kinked"] = FO.pooled_predict(beta, base, feat, wk, h, np.array([t0]), "kinked")[:, 0]
        known = np.isfinite(Yfull[:, t0 + h]) if t0 + h < Yfull.shape[1] else np.zeros(len(ids), bool)     # target day already complete: nothing to forecast
        target = (days[t0] + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        for m, g in guesses.items():
            q = cal["q"].get((m, h))
            if q is None:
                continue
            idx = np.nonzero(at_origin & np.isfinite(sig) & (seen >= NEED[m]) & np.isfinite(g) & ~known)[0]
            if len(idx) == 0:
                continue
            b, gg = bins[idx], g[idx]
            parts.append(pd.DataFrame({
                "made_day": made_day, "item_id": ids[idx], "horizon": h, "method": m,
                "guess": 10.0 ** gg, "lo50": 10.0 ** (gg + q[b, 1]), "hi50": 10.0 ** (gg + q[b, 2]), "lo80": 10.0 ** (gg + q[b, 0]), "hi80": 10.0 ** (gg + q[b, 3]),
                "origin": 10.0 ** Y[idx, t0], "jump_bin": b + 1, "target_day": target, "actual": np.nan, "made_at": stamp, "engine_version": __version__}))
    return pd.concat(parts, ignore_index=True)[COLUMNS] if parts else _empty()


def merge(old: Optional[pd.DataFrame], new: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The ledger plus new rows. A (origin day, item, horizon, method) already written is kept as it was."""
    parts = [p.reindex(columns=COLUMNS) for p in (old, new) if p is not None and len(p)]
    if not parts:
        return _empty()
    m = pd.concat(parts, ignore_index=True)
    m["item_id"] = m["item_id"].astype(int)
    m["horizon"] = m["horizon"].astype(int)
    return m.drop_duplicates(KEY, keep="first").reset_index(drop=True)[COLUMNS]


# ---------------------------------------------------------------- grading

def grade(led: Optional[pd.DataFrame], daily: Optional[pd.DataFrame], min_hours: int = MIN_COMPLETE) -> Tuple[pd.DataFrame, int]:
    """Fill in `actual` for rows whose target day is now complete in item_daily (18+ usable hours). Returns (ledger, rows newly graded). Rows already graded never change."""
    if led is None or led.empty:
        return _empty(), 0
    led = led.reindex(columns=COLUMNS).copy()
    if daily is None or daily.empty:
        return led, 0
    d = daily[daily["hours"] >= min_hours][["item_id", "day", "price_day"]].copy()
    d = d[d["price_day"] > 0]
    if d.empty:
        return led, 0
    d["item_id"] = d["item_id"].astype(int)
    d["target_day"] = _day_str(d["day"])
    d = d.drop_duplicates(["item_id", "target_day"], keep="last")[["item_id", "target_day", "price_day"]].rename(columns={"price_day": "_a"})
    led["item_id"] = led["item_id"].astype(int)
    led["target_day"] = led["target_day"].astype(str)
    m = led.merge(d, on=["item_id", "target_day"], how="left")
    fill = (m["actual"].isna() & m["_a"].notna()).to_numpy()
    m.loc[fill, "actual"] = m.loc[fill, "_a"]
    return m.drop(columns="_a")[COLUMNS], int(fill.sum())


# ---------------------------------------------------------------- scoreboard

def _level(origin: pd.Series) -> pd.Series:
    lv = np.digitize(np.log10(origin.to_numpy(float)), [0.0, 1.0, 2.0])
    return pd.Series(np.array(FO.TIER_ORDER, dtype=object)[lv], index=origin.index)


def _score(df: pd.DataFrame, by: list) -> pd.DataFrame:
    rows = []
    for keys, q in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ae, en = q["ae"].to_numpy(), q["en"].to_numpy()
        ok = np.isfinite(en)
        rows.append({**dict(zip(by, keys)), "cases": len(q), "items": int(q["item_id"].nunique()),
                     "miss_pct": _pct(float(np.median(ae))), "avg_pct": _pct(float(ae.mean())),
                     "vs_naive": float(ae[ok].sum() / en[ok].sum()) if ok.any() and en[ok].sum() > 0 else np.nan,
                     "bias_pct": _pct(float(np.median(q["E"]))), "hit50": float(q["in50"].mean()), "hit80": float(q["in80"].mean()), "width80_pct": _pct(float(np.median(q["w80"])))})
    return pd.DataFrame(rows)


def scoreboard(led: Optional[pd.DataFrame]) -> dict:
    """{graded: n, waiting: per-horizon frame, overall: per (horizon, method), levels: per (horizon, price level, method), days: (first origin, last origin, last graded target)}.
    vs_naive compares each method with naive on exactly the same graded cases (below 1.00 beats 'it stays put'). hit50 / hit80 should sit near 0.50 / 0.80."""
    if led is None or led.empty:
        return {"graded": 0, "waiting": pd.DataFrame()}
    waiting = led[led["actual"].isna()].groupby("horizon").agg(rows=("item_id", "size"), items=("item_id", "nunique"), first_target=("target_day", "min"), last_target=("target_day", "max")).reset_index()
    g = led[led["actual"].notna() & (led["actual"] > 0)].copy()
    out = {"graded": len(g), "waiting": waiting, "origins": (led["made_day"].min(), led["made_day"].max())}
    if g.empty:
        return out
    g["E"] = np.log10(g["actual"]) - np.log10(g["guess"])
    g["ae"] = g["E"].abs()
    g["in50"] = ((g["actual"] >= g["lo50"]) & (g["actual"] <= g["hi50"])).astype(float)
    g["in80"] = ((g["actual"] >= g["lo80"]) & (g["actual"] <= g["hi80"])).astype(float)
    g["w80"] = np.log10(g["hi80"]) - np.log10(g["lo80"])
    key = ["made_day", "item_id", "horizon"]
    nv = g[g["method"] == "naive"].drop_duplicates(key).set_index(key)["ae"].rename("en")
    g = g.join(nv, on=key)
    g["level"] = _level(g["origin"])
    order = {m: i for i, m in enumerate(METHODS)}
    o = _score(g, ["horizon", "method"])
    out["overall"] = o.sort_values(["horizon", "method"], key=lambda s: s.map(order) if s.name == "method" else s).reset_index(drop=True)
    lv = _score(g, ["horizon", "level", "method"])
    lv["_l"] = lv["level"].map({k: i for i, k in enumerate(FO.TIER_ORDER)})
    lv["_m"] = lv["method"].map(order)
    out["levels"] = lv.sort_values(["horizon", "_l", "_m"]).drop(columns=["_l", "_m"]).reset_index(drop=True)
    out["last_target"] = g["target_day"].max()
    return out
