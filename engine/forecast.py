"""Forecast bake-off: which simple way of guessing tomorrow's (or next week's) price actually beats "it stays where it is"?

Input: the `market_history` table (item_id, day, price in copper, quantity) that `tools/lake.py undermine` stores, and the item_groups table. Daily price (lowest
price) and units listed, NOT sales. Only SETTLED items are scored: an item must be 56 days past its first appearance (a launch's settling curve belongs to engine/launch.py).

Every guess is made with only what was known on the day, then compared with what really happened (walk-forward: the history is cut in three test stretches; for each, the
guesses are tuned or calibrated on earlier days only, then scored on the stretch). Candidates, all in log10 price so a 10% miss means the same on a herb and on a flask:

  naive      tomorrow = today                                      (the bar everything else must clear)
  week_ago   the same weekday last week
  median7    the median of the last 7 days (shrugs off one-day spikes)
  ema50 / ema25 / ema10   exponential averages that follow the latest days closely / moderately / slowly
  weekday    an exponential average plus a learned weekday pattern  (the plain-arithmetic cousin of a seasonal model)
  sarima     SARIMA(p,d,q)(P,D,Q)7 fitted per item on the days before each test stretch, best of four specs by AIC (engine/sarima.py)
  pooled     one regression learned across ALL items: where today sits against the item's own averages, and the weekday of the target
  kinked     pooled, but small moves and spikes get their own slopes: it can stay put for ordinary moves yet pull back after a spike
  pooled_grp the same plus how far the item's group peers are from THEIR averages today (the stand-in for a recipe/graph model; an ablation of whether peers help)

Ranges: each guess gets a range from its own past misses, kept separately for five levels of recent jumpiness (an item that has been calm gets a narrower range than one that
has not), so error bars shrink for calm items by construction and the test below says whether they are honest (hit50 / hit80 near 0.50 / 0.80).

The same machinery runs on HOURLY data (mode forecast_hourly): the cycle is 24 hours instead of 7 days and the horizons are 1, 4, 12 and 24 hours; see HOURLY below."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import history as HI

HORIZONS = (1, 3, 7)
WARMUP = 56                 # days past an item's first appearance before it is scored (prices settle about day 38)
MIN_ITEM_DAYS = 120         # items with fewer observed days are left out
MIN_SPAN = 240              # fewer dated days than this and there is nothing honest to test
FOLDS = 3                   # the last 60% of the history is cut in three test stretches
START_FRAC = 0.4            # testing starts this far through the dated span
CAL_FRAC = 0.3              # the last 30% of the days before each test stretch calibrates the ranges; the days before that tune the pooled model
TEST_STEP = 3               # score every 3rd origin day (neighbouring origins are nearly the same test)
FIT_STEP = 2
VOL_BINS = 5
VOL_WINDOW = 14
MIN_BIN = 200
MIN_ITEM_ROWS = 20
CLIP = 0.5                  # features beyond about 3x are clipped, so one broken day cannot drive the pooled fit
BIG = float(np.log10(1.25))   # "a big miss" = off by more than 25%
MOVE = float(np.log10(1.03))  # a direction call counts only when the guess moves at least 3% from today
BEAT = 0.03                 # a model must beat naive by 3% (mean |miss|) to be named best
EMA_ALPHAS = {"ema50": 0.5, "ema25": 0.25, "ema10": 0.1}
HW_ALPHA, HW_GAMMA, HW_CLIP = 0.2, 0.15, 0.15
STATIC = ("naive", "week_ago", "median7", "ema50", "ema25", "ema10", "weekday")
SARIMA = "sarima"
POOLED = ("pooled", "kinked", "pooled_grp")
KINK = 0.1                  # log10: a deviation from the item's average beyond about 26% counts as a spike for the kinked model


@dataclass(frozen=True)
class Cfg:
    """Everything that depends on the step of the series. The defaults ARE the daily test, so the daily bake-off is unchanged; HOURLY below is the same test on hourly steps.
    season = steps in one cycle (7 days / 24 hours); ma = window of the rolling median; alphas = the exponential averages (name, weight on the newest step);
    slow = the name of a slower average used as an extra anchor by the pooled models (hourly only); phase = what a step's position in the cycle is read from (dow = day of
    week, hour = hour of day); big / move = the 'big miss' and the 'direction call' thresholds in plain ratio terms; age_by_obs = burn-in is counted in real readings an item has had, not in clock steps since its first one (hourly: a hole in the data must not use up the burn-in); labels = display names for the methods whose meaning changes."""
    unit: str = "day"
    season: int = 7
    ma: int = 7
    vol_window: int = 14
    warmup: int = WARMUP
    min_item: int = MIN_ITEM_DAYS
    min_span: int = MIN_SPAN
    folds: int = FOLDS
    start_frac: float = START_FRAC
    cal_frac: float = CAL_FRAC
    test_step: int = TEST_STEP
    fit_step: int = FIT_STEP
    min_bin: int = 200
    hw: Tuple[float, float, float] = (HW_ALPHA, HW_GAMMA, HW_CLIP)
    alphas: Tuple[Tuple[str, float], ...] = (("ema50", 0.5), ("ema25", 0.25), ("ema10", 0.1))
    slow: Optional[str] = None
    phase: str = "dow"
    big: float = 1.25
    move: float = 1.03
    age_by_obs: bool = False
    labels: Tuple[Tuple[str, str], ...] = ()

    @property
    def static(self) -> Tuple[str, ...]:
        return ("naive", "week_ago", "median7") + tuple(n for n, _ in self.alphas) + ("weekday",)

    @property
    def big_log(self) -> float:
        return float(np.log10(self.big))

    @property
    def move_log(self) -> float:
        return float(np.log10(self.move))

    def phase_of(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.hour if self.phase == "hour" else index.dayofweek).to_numpy()


DAILY = Cfg()
# Hourly: a 24-hour cycle, horizons of 1 / 4 / 12 / 24 hours. Only a few days of hourly snapshots exist, so every length is in hours and short: 48 hours of burn-in before an item is scored,
# a slow 24-hour-memory average joins the candidates, and the thresholds are tighter because hourly moves are smaller than daily ones.
HOURLY_HORIZONS = (1, 4, 12, 24)
HOURLY = Cfg(unit="hour", season=24, ma=6, vol_window=24, warmup=48, min_item=72, min_span=96, test_step=1, fit_step=1, min_bin=200, hw=(0.3, 0.2, 0.15),
             alphas=(("ema50", 0.5), ("ema25", 0.25), ("ema10", 0.1), ("ema04", 0.04)), slow="ema04", phase="hour", big=1.10, move=1.01, age_by_obs=True,
             labels=(("week_ago", "day_ago"), ("median7", "median6"), ("weekday", "hour_of_day")))


# ---------------------------------------------------------------- data

def build_grid(hist: pd.DataFrame, metric: str = "price", min_days: int = MIN_ITEM_DAYS):
    """(item ids, days, log10 values): an items x days grid with NaN for missing days. metric = price (gold) or stock (units listed)."""
    h = HI.clean(hist)
    if h.empty:
        return None
    col = "price" if metric == "price" else "quantity"
    h = h[["item_id", "day", col]].copy()
    h["day"] = h["day"].dt.normalize()
    keep = h.groupby("item_id").size()
    h = h[h["item_id"].isin(keep[keep >= min_days].index)]
    if h.empty:
        return None
    M = h.pivot(index="item_id", columns="day", values=col)
    days = pd.date_range(M.columns.min(), M.columns.max(), freq="D")
    M = M.reindex(columns=days)
    return list(M.index.astype(int)), days, np.log10(M.to_numpy(float))


def volatility(Y: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """How jumpy each item has been lately: the spread of its step-to-step changes over the last 14 steps (only steps up to and including the one in the column)."""
    d = pd.DataFrame(Y.T).diff()
    return d.rolling(window, min_periods=window // 2).std().to_numpy().T


# ---------------------------------------------------------------- models that need no fitting (each uses only days up to the origin)

def ema(Y: np.ndarray, a: float) -> np.ndarray:
    n, T = Y.shape
    L = np.full((n, T), np.nan)
    cur = np.full(n, np.nan)
    for t in range(T):
        x = Y[:, t]
        ok = np.isfinite(x)
        new = np.where(np.isfinite(cur), cur + a * (x - cur), x)
        cur = np.where(ok, new, cur)
        L[:, t] = cur
    return L


def weekday_model(Y: np.ndarray, wk: np.ndarray, hs=HORIZONS, a: float = HW_ALPHA, g: float = HW_GAMMA, clip: float = HW_CLIP, season: int = 7) -> Dict[int, np.ndarray]:
    """Level plus a weekday pattern (an hour-of-day pattern when season = 24), both updated step by step. The forecast h steps ahead is today's level plus the pattern for the
    target's position in the cycle. The pattern's updates are clipped so a spike cannot teach it a fake rhythm."""
    n, T = Y.shape
    out = {h: np.full((n, T), np.nan) for h in hs}
    level = np.full(n, np.nan)
    S = np.zeros((n, season))
    for t in range(T):
        x, w = Y[:, t], int(wk[t])
        ok = np.isfinite(x)
        if ok.any():
            xs = np.where(ok, x, 0.0)
            ds = xs - S[:, w]
            nl = np.where(np.isfinite(level), level + a * (ds - level), ds)
            innov = np.clip(xs - nl, -clip, clip)
            ns = S[:, w] + g * (innov - S[:, w])
            level = np.where(ok, nl, level)
            S[:, w] = np.where(ok, ns, S[:, w])
        for h in hs:
            out[h][:, t] = level + S[:, (w + h) % season]
    return out


def static_forecasts(Y: np.ndarray, wk: np.ndarray, hs=HORIZONS, cfg: Cfg = DAILY) -> Tuple[Dict[str, Dict[int, np.ndarray]], dict]:
    """{model: {horizon: items x steps array of the guess made AT that step for h steps later}} for the models with nothing to fit, plus the shared building blocks.
    week_ago is 'the same position in the last cycle' (needs h <= season; a longer horizon gets no guess)."""
    n, T = Y.shape
    out: Dict[str, Dict[int, np.ndarray]] = {"naive": {h: Y for h in hs}}
    wa = {}
    for h in hs:
        k = cfg.season - h
        if k == 0:
            wa[h] = Y
        else:
            F = np.full((n, T), np.nan)
            if 0 < k < T:
                F[:, k:] = Y[:, :T - k]
            wa[h] = F
    out["week_ago"] = wa
    R7 = pd.DataFrame(Y.T).rolling(cfg.ma, min_periods=(cfg.ma + 1) // 2).median().to_numpy().T
    out["median7"] = {h: R7 for h in hs}
    L = {}
    for name, a in cfg.alphas:
        L[name] = ema(Y, a)
        out[name] = {h: L[name] for h in hs}
    a_, g_, c_ = cfg.hw
    out["weekday"] = weekday_model(Y, wk, hs, a_, g_, c_, cfg.season)
    base = {"R7": R7, "L50": L["ema50"], "L25": L["ema25"], "L10": L["ema10"]}
    if cfg.slow:
        base["Lslow"] = L[cfg.slow]
    return out, base


# ---------------------------------------------------------------- pooled models (fitted on earlier days only)

def group_dev(D: np.ndarray, gi: np.ndarray, min_peers: int = 3) -> np.ndarray:
    """For each item and day, the average of how far the OTHER items in its group sit from their own averages that day (0 when fewer than 3 peers were seen)."""
    n, T = D.shape
    out = np.zeros((n, T))
    fin = np.isfinite(D)
    Z = np.where(fin, D, 0.0)
    for g in np.unique(gi[gi >= 0]):
        m = np.where(gi == g)[0]
        if len(m) < min_peers + 1:
            continue
        s, c = Z[m].sum(0), fin[m].sum(0)
        peers = c[None, :] - fin[m]
        val = (s[None, :] - Z[m]) / np.maximum(peers, 1)
        out[m] = np.where(peers >= min_peers, val, 0.0)
    return out


def _features(Y: np.ndarray, base: dict, gi: Optional[np.ndarray]) -> dict:
    d0 = np.clip(Y - base["L25"], -CLIP, CLIP)
    d0s = np.clip(d0, -KINK, KINK)
    f = {"d0": d0, "d0s": d0s, "d0b": d0 - d0s, "d1": np.clip(base["L50"] - base["L10"], -CLIP, CLIP), "d2": np.clip(base["R7"] - base["L25"], -CLIP, CLIP)}
    if "Lslow" in base:      # hourly: where the fast average sits against the slow (about 24 hour) one, so the model can learn a pull back to the daily level
        f["d3"] = np.clip(base["L25"] - base["Lslow"], -CLIP, CLIP)
    if gi is not None:
        f["gd"] = np.clip(group_dev(Y - base["L25"], gi), -CLIP, CLIP)
    return f


def _design(feat: dict, wk: np.ndarray, ii: np.ndarray, tt: np.ndarray, h: int, kind: str, season: int = 7) -> np.ndarray:
    cols = [np.ones(len(ii))]
    if kind == "kinked":
        cols.extend((feat["d0s"][ii, tt], feat["d0b"][ii, tt]))
    else:
        cols.append(feat["d0"][ii, tt])
    cols.extend((feat["d1"][ii, tt], feat["d2"][ii, tt]))
    if "d3" in feat:
        cols.append(feat["d3"][ii, tt])
    if kind == "pooled_grp":
        cols.append(feat["gd"][ii, tt])
    target_wd = (wk[tt] + h) % season
    cols.extend((target_wd == d).astype(float) for d in range(1, season))
    return np.column_stack(cols)


def _huber_fit(X: np.ndarray, y: np.ndarray, iters: int = 4, k: float = 1.345) -> np.ndarray:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-4)
        sw = np.sqrt(np.minimum(1.0, k * s / np.maximum(np.abs(r), 1e-12)))
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return beta


def _pairs(valid: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = np.nonzero(valid[:, cols])
    return ii, cols[jj]


def pooled_fit(Y, base, feat, wk, valid, h, fit_cols, kind: str = "pooled", season: int = 7):
    """Coefficients of: target (the value h steps on, minus the item's moderate average now) ~ features. None when there are too few rows."""
    ii, tt = _pairs(valid, fit_cols)
    if len(ii) < 2000:
        return None
    X = _design(feat, wk, ii, tt, h, kind, season)
    y = np.clip(Y[ii, tt + h] - base["L25"][ii, tt], -2 * CLIP, 2 * CLIP)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < 2000:
        return None
    return _huber_fit(X[ok], y[ok])


def pooled_predict(beta, base, feat, wk, h, cols, kind: str = "pooled", season: int = 7) -> np.ndarray:
    n = base["L25"].shape[0]
    ii, tt = np.repeat(np.arange(n), len(cols)), np.tile(cols, n)
    X = _design(feat, wk, ii, tt, h, kind, season)
    return (base["L25"][ii, tt] + X @ beta).reshape(n, len(cols))


# ---------------------------------------------------------------- scoring

class _Acc:
    """Everything scored for one (model, horizon), across folds. Rows hold the signed log10 miss and the range / direction results; per-item sums feed WAPE and the family table."""

    def __init__(self, n_items: int, move: float = MOVE):
        self.move = move
        self.E: List[np.ndarray] = []
        self.h50: List[np.ndarray] = []
        self.h80: List[np.ndarray] = []
        self.w80: List[np.ndarray] = []
        self.dr: List[np.ndarray] = []
        self.bn: List[np.ndarray] = []
        self.sg: List[np.ndarray] = []
        self.gold_err = np.zeros(n_items)
        self.gold_y = np.zeros(n_items)
        self.abs_sum = np.zeros(n_items)
        self.cnt = np.zeros(n_items)

    def add(self, ii, Yv, Fv, X0v, lo50, hi50, lo80, hi80, bn, sg):
        n_items = len(self.cnt)
        E = Yv - Fv
        self.E.append(E.astype(np.float32))
        self.h50.append(((Yv >= lo50) & (Yv <= hi50)).astype(np.int8))
        self.h80.append(((Yv >= lo80) & (Yv <= hi80)).astype(np.int8))
        self.w80.append((hi80 - lo80).astype(np.float32))
        self.bn.append(np.asarray(bn).astype(np.int8))
        self.sg.append(np.asarray(sg).astype(np.float32))
        moved, act = Fv - X0v, Yv - X0v
        hit = (np.sign(moved) == np.sign(act)) & (act != 0)
        self.dr.append(np.where(np.abs(moved) > self.move, hit, -1).astype(np.int8))
        gy = 10.0 ** Yv
        self.gold_err += np.bincount(ii, weights=np.abs(gy - 10.0 ** Fv), minlength=n_items)
        self.gold_y += np.bincount(ii, weights=gy, minlength=n_items)
        self.abs_sum += np.bincount(ii, weights=np.abs(E), minlength=n_items)
        self.cnt += np.bincount(ii, minlength=n_items)

    def cat(self, name):
        parts = getattr(self, name)
        return np.concatenate(parts) if parts else np.array([])


def _pct(x: float) -> float:
    return (10.0 ** x - 1.0) * 100.0


def _bin_quantiles(E: np.ndarray, bins: np.ndarray, min_bin: int = MIN_BIN) -> np.ndarray:
    """VOL_BINS x 4 table of the q10 / q25 / q75 / q90 of past misses, one row per jumpiness level (a thin level borrows the pooled quantiles)."""
    qs = (0.1, 0.25, 0.75, 0.9)
    pooled = np.quantile(E, qs)
    out = np.tile(pooled, (VOL_BINS, 1))
    for b in range(VOL_BINS):
        e = E[bins == b]
        if len(e) >= min_bin:
            out[b] = np.quantile(e, qs)
    return out


def walk_forward(Y: np.ndarray, days: pd.DatetimeIndex, gi: Optional[np.ndarray], hs=HORIZONS, sarima: bool = False, cfg: Cfg = DAILY) -> dict:
    """The whole test. Returns {acc: {(model, h): _Acc}, models, folds: [(cut date, next cut date)], rows}. `days` is the index of steps (days, or hours when cfg is HOURLY)."""
    n, T = Y.shape
    wk = cfg.phase_of(days)
    obs = np.isfinite(Y)
    first = np.where(obs.any(1), obs.argmax(1), T)
    if cfg.age_by_obs:      # readings an item has had BEFORE this step (a hole in the data does not count as burn-in)
        age_ok = (np.cumsum(obs, axis=1) - obs) >= cfg.warmup
    else:
        age_ok = (np.arange(T)[None, :] - first[:, None]) >= cfg.warmup
    sig = volatility(Y, cfg.vol_window)
    static, base = static_forecasts(Y, wk, hs, cfg)
    feat = _features(Y, base, gi)
    models = list(cfg.static) + ([SARIMA] if sarima else []) + ["pooled", "kinked"] + (["pooled_grp"] if gi is not None else [])
    acc = {(m, h): _Acc(n, cfg.move_log) for m in models for h in hs}
    acc_fit = {(m, h): _Acc(n, cfg.move_log) for m in models for h in hs} if sarima else {}     # the same cases, but only items SARIMA really fitted
    T0 = cfg.warmup
    if cfg.age_by_obs:       # testing starts once at least half of the items are burned in
        ready = np.nonzero(age_ok.mean(axis=0) >= 0.5)[0]
        T0 = int(ready[0]) if len(ready) else T
    span = T - 1 - T0
    cuts = [T0 + int(span * (cfg.start_frac + k * (1 - cfg.start_frac) / cfg.folds)) for k in range(cfg.folds)] + [T - 1]
    done, sar_info = [], []
    for k in range(cfg.folds):
        c, c_next = cuts[k], cuts[k + 1]
        c_fit = T0 + int((c - T0) * (1 - cfg.cal_frac))
        sar, sar_ok = None, None
        if sarima:
            from . import sarima as SA
            sar, info = SA.forecast_all(Y, c_fit, max(c_fit - max(hs) + 1, 0), c_next, hs)
            sar_info.append(info)
            sar_ok = info["ok"]
        for h in hs:
            valid = obs & age_ok
            valid = valid & np.concatenate([obs[:, h:], np.zeros((n, h), bool)], axis=1)
            fit_cols = np.arange(0, max(c_fit - h + 1, 0), cfg.fit_step)
            cal_cols = np.arange(max(c_fit - h + 1, 0), c - h + 1, cfg.test_step)
            test_cols = np.arange(c + 1, min(c_next, T - 1 - h) + 1, cfg.test_step)
            if len(cal_cols) == 0 or len(test_cols) == 0:
                continue
            guess_c, guess_t = {}, {}
            for m in cfg.static:
                guess_c[m], guess_t[m] = static[m][h][:, cal_cols], static[m][h][:, test_cols]
            if sar is not None:      # an item SARIMA could not be fitted for falls back to the ema25 guess, so it stays in the comparison
                fb = static["ema25"][h]
                guess_c[SARIMA] = np.where(np.isfinite(sar[h][:, cal_cols]), sar[h][:, cal_cols], fb[:, cal_cols])
                guess_t[SARIMA] = np.where(np.isfinite(sar[h][:, test_cols]), sar[h][:, test_cols], fb[:, test_cols])
            for m in models:
                if m not in POOLED:
                    continue
                beta = pooled_fit(Y, base, feat, wk, valid, h, fit_cols, kind=m, season=cfg.season)
                if beta is None:
                    continue
                guess_c[m] = pooled_predict(beta, base, feat, wk, h, cal_cols, m, cfg.season)
                guess_t[m] = pooled_predict(beta, base, feat, wk, h, test_cols, m, cfg.season)
            if any(m in POOLED and m not in guess_t for m in models):
                continue         # a pooled model had too few rows to fit in this stretch: skip the stretch for EVERY method, so all are still scored on exactly the same cases
            use = [m for m in models if m in guess_t]
            if "naive" not in use:
                continue
            Yc, Yt = Y[:, cal_cols + h], Y[:, test_cols + h]
            X0t = Y[:, test_cols]
            vc, vt = valid[:, cal_cols] & np.isfinite(sig[:, cal_cols]), valid[:, test_cols] & np.isfinite(sig[:, test_cols])
            sc = sig[:, cal_cols]
            if (vc & np.isfinite(sc)).sum() < cfg.min_bin * VOL_BINS:
                continue
            edges = np.quantile(sc[vc], np.linspace(0, 1, VOL_BINS + 1)[1:-1])
            mt = vt.copy()
            for m in use:
                mt &= np.isfinite(guess_t[m])
            ii, jj = np.nonzero(mt)
            if len(ii) == 0:
                continue
            sg_t = sig[:, test_cols][ii, jj]
            bins_t = np.digitize(sg_t, edges)
            Yv, X0v = Yt[ii, jj], X0t[ii, jj]
            okrow = sar_ok[ii] if sar_ok is not None else None
            for m in use:
                Ec = Yc - guess_c[m]
                mc = vc & np.isfinite(Ec)
                q = _bin_quantiles(Ec[mc], np.digitize(sc[mc], edges), cfg.min_bin)
                Fv = guess_t[m][ii, jj]
                lo80, hi80 = Fv + q[bins_t, 0], Fv + q[bins_t, 3]
                lo50, hi50 = Fv + q[bins_t, 1], Fv + q[bins_t, 2]
                acc[(m, h)].add(ii, Yv, Fv, X0v, lo50, hi50, lo80, hi80, bins_t, sg_t)
                if okrow is not None and okrow.any():
                    r = okrow
                    acc_fit[(m, h)].add(ii[r], Yv[r], Fv[r], X0v[r], lo50[r], hi50[r], lo80[r], hi80[r], bins_t[r], sg_t[r])
        done.append((days[min(c + 1, T - 1)], days[min(c_next, T - 1)]))
    return {"acc": acc, "acc_fit": acc_fit, "models": models, "folds": done, "horizons": tuple(hs), "sarima_info": sar_info, "cfg": cfg}


def summarize(res: dict, key: str = "acc") -> pd.DataFrame:
    """One row per (horizon, model): typical miss (median, as a percent of the price), average miss, miss against naive (below 1.0 beats it), WAPE (all items pooled, so pricey
    items weigh most) and WAPE of the median item, bias and tracking signal (positive = the real price ran above the guess), RMSE and 95th-percentile miss (the safety grades),
    the share of big misses (over 25%), direction hit rate, and whether the ranges hold. key = "acc_fit" scores only the cases where SARIMA was really fitted."""
    rows = []
    acc, models = res[key], res["models"]
    big = res["cfg"].big_log if "cfg" in res else BIG
    for h in res["horizons"]:
        nv = acc.get(("naive", h))
        if nv is None or not nv.E:
            continue
        mae_n = float(np.mean(np.abs(nv.cat("E"))))
        for m in models:
            a = acc.get((m, h))
            if a is None or not a.E:
                continue
            E = a.cat("E").astype(float)
            ae = np.abs(E)
            dr = a.cat("dr")
            dv = dr[dr >= 0]
            per_item = a.gold_err[a.cnt >= MIN_ITEM_ROWS] / np.maximum(a.gold_y[a.cnt >= MIN_ITEM_ROWS], 1e-12)
            rows.append({
                "horizon": h, "model": m, "rows": len(E), "items": int((a.cnt > 0).sum()),
                "miss_pct": _pct(float(np.median(ae))), "avg_pct": _pct(float(ae.mean())),
                "vs_naive": float(ae.mean() / mae_n) if mae_n > 0 else np.nan,
                "wape": 100.0 * float(a.gold_err.sum() / max(a.gold_y.sum(), 1e-12)),
                "wape_med": 100.0 * float(np.median(per_item)) if len(per_item) else np.nan,
                "bias_pct": _pct(float(np.median(E))), "track": float(E.mean() / ae.mean()) if ae.mean() > 0 else 0.0,
                "rmse_pct": _pct(float(np.sqrt((E ** 2).mean()))), "p95_pct": _pct(float(np.quantile(ae, 0.95))),
                "big_pct": 100.0 * float((ae > big).mean()),
                "dir_hit": float(dv.mean()) if len(dv) else np.nan, "dir_n": int(len(dv)),
                "hit50": float(a.cat("h50").mean()), "hit80": float(a.cat("h80").mean()), "width80_pct": _pct(float(np.median(a.cat("w80")))),
            })
    return pd.DataFrame(rows)


def bin_table(res: dict) -> pd.DataFrame:
    """Per horizon and level of recent jumpiness (1 = the calmest fifth of cases, 5 = the jumpiest; levels are set on each fold's calibration days): how big the swings are, how
    big naive's miss is, which method is best (naive unless 3%+ better) and how it does, and whether the ranges hold. This is where a forecast is tight enough to act on."""
    acc, models = res["acc"], res["models"]
    rows = []
    for h in res["horizons"]:
        nv = acc.get(("naive", h))
        if nv is None or not nv.E:
            continue
        bn = nv.cat("bn")
        sg = nv.cat("sg").astype(float)
        En = np.abs(nv.cat("E").astype(float))
        for b in range(VOL_BINS):
            sel = bn == b
            if sel.sum() < 50 or En[sel].mean() <= 0:
                continue
            mean_abs = {}
            for m in models:
                a = acc.get((m, h))
                if a is not None and a.E:
                    mean_abs[m] = float(np.abs(a.cat("E").astype(float))[sel].mean())
            top = min(mean_abs, key=mean_abs.get)
            best = top if top != "naive" and mean_abs[top] <= mean_abs["naive"] * (1 - BEAT) else "naive"
            a = acc[(best, h)]
            rows.append({"horizon": h, "jumpiness": b + 1, "jump_pct": _pct(float(np.median(sg[sel]))), "cases": int(sel.sum()),
                         "naive_pct": _pct(mean_abs["naive"]), "naive_typ": _pct(float(np.median(En[sel]))), "best": best, "best_pct": _pct(mean_abs[best]),
                         "vs_naive": mean_abs[best] / mean_abs["naive"], "hit80": float(a.cat("h80")[sel].mean()), "width80_pct": _pct(float(np.median(a.cat("w80")[sel])))})
    return pd.DataFrame(rows)


def best_models(summary: pd.DataFrame) -> Dict[int, str]:
    """Per horizon: naive unless another model's average miss is at least 3% smaller (complexity has to earn its keep)."""
    out = {}
    for h, t in summary.groupby("horizon"):
        t = t.set_index("model")
        if "naive" not in t.index:
            continue
        top = t["avg_pct"].idxmin()
        out[int(h)] = top if top != "naive" and t.loc[top, "vs_naive"] <= 1 - BEAT else "naive"
    return out


def family_table(res: dict, fam: np.ndarray) -> pd.DataFrame:
    """Per family and horizon: items, the naive average miss, the best model (naive unless 3%+ better) and its miss against naive."""
    acc, models = res["acc"], res["models"]
    rows = []
    fam = np.asarray(fam, dtype=object)
    for h in res["horizons"]:
        nv = acc.get(("naive", h))
        if nv is None or not nv.E:
            continue
        for f in sorted(set(fam)):
            idx = np.where((fam == f) & (nv.cnt > 0))[0]
            if len(idx) < 5:
                continue
            miss = {}
            for m in models:
                a = acc.get((m, h))
                if a is not None and a.E and a.cnt[idx].sum() > 0:
                    miss[m] = a.abs_sum[idx].sum() / a.cnt[idx].sum()
            if "naive" not in miss or miss["naive"] <= 0:
                continue
            top = min(miss, key=miss.get)
            best = top if top != "naive" and miss[top] <= miss["naive"] * (1 - BEAT) else "naive"
            rows.append({"family": f, "horizon": h, "items": len(idx), "naive_pct": _pct(miss["naive"]), "best": best, "best_pct": _pct(miss[best]), "vs_naive": miss[best] / miss["naive"]})
    return pd.DataFrame(rows)


TIER_ORDER = ("under 1g", "1g to 10g", "10g to 100g", "100g and up")


def price_tiers(Y: np.ndarray) -> np.ndarray:
    """Each item's price level from its median price over the grid (Y is log10 gold): under 1g / 1g to 10g / 10g to 100g / 100g and up. Items with no reading get ''."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Y, axis=1)
    out = np.array([""] * len(med), dtype=object)
    ok = np.isfinite(med)
    out[ok] = np.array(TIER_ORDER, dtype=object)[np.digitize(med[ok], [0.0, 1.0, 2.0])]
    return out


def tier_table(res: dict, tiers: np.ndarray) -> pd.DataFrame:
    """family_table by price level, in price order: items, naive's average miss, the best method (naive unless 3%+ better) and its miss against naive."""
    t = family_table(res, tiers)
    if t.empty:
        return t
    t = t.rename(columns={"family": "price_level"})
    t["_o"] = t["price_level"].map({k: i for i, k in enumerate(TIER_ORDER)})
    return t.sort_values(["horizon", "_o"]).drop(columns="_o").reset_index(drop=True)


def bake_off(hist: pd.DataFrame, groups: Optional[pd.DataFrame] = None, metric: str = "price", hs=HORIZONS, sarima: bool = True) -> dict:
    """The full test for one metric on the DAILY history. {} when the history is too thin. sarima=False leaves SARIMA out (it is the slow one); it is also left out, with a note, when scipy is missing."""
    g = build_grid(hist, metric)
    if g is None:
        return {}
    ids, days, Y = g
    return bake_off_grid(ids, days, Y, groups, hs, sarima, DAILY, tiers=price_tiers(Y) if metric == "price" else None)


def bake_off_grid(ids, days, Y, groups: Optional[pd.DataFrame] = None, hs=HORIZONS, sarima: bool = False, cfg: Cfg = DAILY, tiers: Optional[np.ndarray] = None) -> dict:
    """The full test on an items x steps grid of log10 values (days for DAILY, hours for HOURLY). {} when there are too few steps. SARIMA only exists for the daily cycle.
    tiers = optional price level per item (see price_tiers); adds a table by price level."""
    if len(days) < cfg.min_span:
        return {}
    if cfg.season != 7:
        sarima = False
    gi, fam = None, np.array(["Unknown"] * len(ids), dtype=object)
    if groups is not None and not groups.empty and "item_id" in groups.columns:
        k = groups.drop_duplicates("item_id").set_index("item_id")
        if "family" in k.columns:
            fam = np.array([k["family"].get(i, "Unknown") for i in ids], dtype=object)
        if "group" in k.columns:
            codes = pd.Series([k["group"].get(i, None) for i in ids]).astype("category").cat.codes.to_numpy()
            if (codes >= 0).sum() > 0:
                gi = codes.astype(int)
    note = ""
    if sarima:
        try:
            import scipy  # noqa: F401
        except ImportError:
            sarima, note = False, "SARIMA skipped: scipy is not installed (add scipy to the pip install line of the Market tools workflow)."
    res = walk_forward(Y, days, gi, hs, sarima=sarima, cfg=cfg)
    summ = summarize(res)
    if summ.empty:
        return {}
    fit = summarize(res, "acc_fit") if sarima and res["acc_fit"] else pd.DataFrame()
    out = {"summary": summ, "summary_fit": fit, "bins": bin_table(res), "best": best_models(summ), "family": family_table(res, fam), "items": len(ids), "days": len(days),
           "first": days[0], "last": days[-1], "folds": res["folds"], "grouped": gi is not None,
           "sarima_info": res["sarima_info"], "sarima_note": note}
    if tiers is not None:
        out["tiers"] = tier_table(res, np.asarray(tiers, dtype=object))
    return relabel(out, cfg)


def relabel(out: dict, cfg: Cfg) -> dict:
    """Give the methods whose meaning changes with the step their display names (hourly: week_ago -> day_ago, median7 -> median6, weekday -> hour_of_day). Internal names stay in the engine."""
    names = dict(cfg.labels)
    if not names:
        return out
    for key in ("summary", "summary_fit"):
        if isinstance(out.get(key), pd.DataFrame) and "model" in out[key].columns:
            out[key] = out[key].assign(model=out[key]["model"].replace(names))
    for key in ("bins", "family", "tiers"):
        if isinstance(out.get(key), pd.DataFrame) and "best" in out[key].columns:
            out[key] = out[key].assign(best=out[key]["best"].replace(names))
    out["best"] = {h: names.get(m, m) for h, m in out["best"].items()}
    return out


# ---------------------------------------------------------------- hourly data (own snapshots)

MIN_SNAPSHOT_SHARE = 0.5    # a snapshot with under half the usual number of listings is a partial capture and counts as missing


def hourly_frame(snaps: Sequence, ids, tz: str = "America/New_York", min_share: float = MIN_SNAPSHOT_SHARE) -> Tuple[pd.DataFrame, dict]:
    """(frame, info). frame: one row per item per clock hour (US Eastern wall clock, the same clock the daily table uses) with floor_gold = the lowest unit price and units_listed =
    the units for sale, taken from the LAST snapshot inside that hour. Partial snapshots (under half the median number of listings) are dropped, because their floors are fake.
    Items absent from a snapshot simply have no row for that hour. info: snapshots, dropped (list of (timestamp, listings)), median_listings, first, last."""
    ids = {int(i) for i in ids}
    snaps = sorted((s for s in snaps if s.kind == "commodity"), key=lambda s: s.ts)
    info = {"snapshots": len(snaps), "dropped": [], "median_listings": 0, "first": None, "last": None}
    if not snaps or not ids:
        return pd.DataFrame(columns=["item_id", "hour", "floor_gold", "units_listed"]), info
    sizes = np.array([len(s.book) for s in snaps], dtype=float)
    med = float(np.median(sizes))
    info["median_listings"] = int(med)
    parts = []
    for s, size in zip(snaps, sizes):
        if size < min_share * med:
            info["dropped"].append((s.ts, int(size)))
            continue
        b = s.book
        b = b[b["item_id"].isin(ids) & (b["quantity"] > 0)]
        if b.empty:
            continue
        g = b.groupby("item_id").agg(floor=("unit_price", "min"), units_listed=("quantity", "sum")).reset_index()
        g["ts"] = s.ts
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["item_id", "hour", "floor_gold", "units_listed"]), info
    f = pd.concat(parts, ignore_index=True)
    local = f["ts"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    f["hour"] = local.dt.floor("h")
    f = f.sort_values(["item_id", "ts"]).drop_duplicates(["item_id", "hour"], keep="last")
    f["floor_gold"] = f["floor"] / 10000.0
    f["item_id"] = f["item_id"].astype(int)
    info["first"], info["last"] = f["hour"].min(), f["hour"].max()
    return f[["item_id", "hour", "floor_gold", "units_listed"]].sort_values(["item_id", "hour"]).reset_index(drop=True), info


def hourly_grid(frame: pd.DataFrame, metric: str = "price", min_obs: int = HOURLY.min_item):
    """(item ids, hourly index, log10 values): an items x clock-hours grid, NaN where the hour has no reading. Items with fewer than min_obs readings are left out."""
    if frame is None or frame.empty:
        return None
    col = "floor_gold" if metric == "price" else "units_listed"
    keep = frame.groupby("item_id").size()
    f = frame[frame["item_id"].isin(keep[keep >= min_obs].index)]
    f = f[f[col] > 0]
    if f.empty:
        return None
    M = f.pivot(index="item_id", columns="hour", values=col)
    idx = pd.date_range(M.columns.min(), M.columns.max(), freq="h")
    M = M.reindex(columns=idx)
    return list(M.index.astype(int)), idx, np.log10(M.to_numpy(float))


def hour_profile(Y: np.ndarray, idx: pd.DatetimeIndex, window: int = 24) -> pd.DataFrame:
    """Is there a daily cycle at all? For every clock hour: how far (percent) the typical item sits from its own 24-hour centred average at that hour (median over items and days),
    and the share of items that lean the same way as the group at that hour (about 0.5 everywhere = no shared cycle; near 1.0 = every item moves together)."""
    D = pd.DataFrame(Y.T)
    dev = (D - D.rolling(window, center=True, min_periods=window * 3 // 4).mean()).to_numpy().T      # items x hours, log10
    hr = idx.hour.to_numpy()
    rows = []
    for h in range(24):
        sub = dev[:, hr == h]
        v = sub[np.isfinite(sub)]
        if v.size < 100:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # an item with no reading at this hour is an all-NaN row
            per_item = np.nanmedian(sub, axis=1)
        per_item = per_item[np.isfinite(per_item)]
        med = float(np.median(v))
        agree = float((np.sign(per_item) == np.sign(med)).mean()) if med != 0 and len(per_item) else np.nan
        rows.append({"hour": h, "vs_day_pct": _pct(med), "items_agree": agree, "readings": int(v.size)})
    return pd.DataFrame(rows)
