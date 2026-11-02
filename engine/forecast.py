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
  pooled_grp the same plus how far the item's group peers are from THEIR averages today (the stand-in for a recipe/graph model; an ablation of whether peers help)

Ranges: each guess gets a range from its own past misses, kept separately for five levels of recent jumpiness (an item that has been calm gets a narrower range than one that
has not), so error bars shrink for calm items by construction and the test below says whether they are honest (hit50 / hit80 near 0.50 / 0.80)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
POOLED = ("pooled", "pooled_grp")


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


def volatility(Y: np.ndarray) -> np.ndarray:
    """How jumpy each item has been lately: the spread of its day-to-day changes over the last 14 days (only days up to and including the one in the column)."""
    d = pd.DataFrame(Y.T).diff()
    return d.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().to_numpy().T


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


def weekday_model(Y: np.ndarray, wk: np.ndarray, hs=HORIZONS, a: float = HW_ALPHA, g: float = HW_GAMMA, clip: float = HW_CLIP) -> Dict[int, np.ndarray]:
    """Level plus a weekday pattern, both updated day by day. The forecast h days ahead is today's level plus the pattern for the target's weekday. The pattern's
    updates are clipped so a spike cannot teach it a fake rhythm."""
    n, T = Y.shape
    out = {h: np.full((n, T), np.nan) for h in hs}
    level = np.full(n, np.nan)
    S = np.zeros((n, 7))
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
            out[h][:, t] = level + S[:, (w + h) % 7]
    return out


def static_forecasts(Y: np.ndarray, wk: np.ndarray, hs=HORIZONS) -> Tuple[Dict[str, Dict[int, np.ndarray]], dict]:
    """{model: {horizon: items x days array of the guess made ON that day for h days later}} for the models with nothing to fit, plus the shared building blocks."""
    n, T = Y.shape
    out: Dict[str, Dict[int, np.ndarray]] = {"naive": {h: Y for h in hs}}
    wa = {}
    for h in hs:
        k = 7 - h
        if k == 0:
            wa[h] = Y
        else:
            F = np.full((n, T), np.nan)
            F[:, k:] = Y[:, :T - k]
            wa[h] = F
    out["week_ago"] = wa
    R7 = pd.DataFrame(Y.T).rolling(7, min_periods=4).median().to_numpy().T
    out["median7"] = {h: R7 for h in hs}
    L = {}
    for name, a in EMA_ALPHAS.items():
        L[name] = ema(Y, a)
        out[name] = {h: L[name] for h in hs}
    out["weekday"] = weekday_model(Y, wk, hs)
    return out, {"R7": R7, "L50": L["ema50"], "L25": L["ema25"], "L10": L["ema10"]}


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
    f = {"d0": d0, "d1": np.clip(base["L50"] - base["L10"], -CLIP, CLIP), "d2": np.clip(base["R7"] - base["L25"], -CLIP, CLIP)}
    if gi is not None:
        f["gd"] = np.clip(group_dev(Y - base["L25"], gi), -CLIP, CLIP)
    return f


def _design(feat: dict, wk: np.ndarray, ii: np.ndarray, tt: np.ndarray, h: int, grp: bool) -> np.ndarray:
    cols = [np.ones(len(ii)), feat["d0"][ii, tt], feat["d1"][ii, tt], feat["d2"][ii, tt]]
    if grp:
        cols.append(feat["gd"][ii, tt])
    target_wd = (wk[tt] + h) % 7
    cols.extend((target_wd == d).astype(float) for d in range(1, 7))
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


def pooled_fit(Y, base, feat, wk, valid, h, fit_cols, grp: bool):
    """Coefficients of: target (the value h days on, minus the item's moderate average today) ~ features. None when there are too few rows."""
    ii, tt = _pairs(valid, fit_cols)
    if len(ii) < 2000:
        return None
    X = _design(feat, wk, ii, tt, h, grp)
    y = np.clip(Y[ii, tt + h] - base["L25"][ii, tt], -2 * CLIP, 2 * CLIP)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < 2000:
        return None
    return _huber_fit(X[ok], y[ok])


def pooled_predict(beta, base, feat, wk, h, cols, grp: bool) -> np.ndarray:
    n = base["L25"].shape[0]
    ii, tt = np.repeat(np.arange(n), len(cols)), np.tile(cols, n)
    X = _design(feat, wk, ii, tt, h, grp)
    return (base["L25"][ii, tt] + X @ beta).reshape(n, len(cols))


# ---------------------------------------------------------------- scoring

class _Acc:
    """Everything scored for one (model, horizon), across folds. Rows hold the signed log10 miss and the range / direction results; per-item sums feed WAPE and the family table."""

    def __init__(self, n_items: int):
        self.E: List[np.ndarray] = []
        self.h50: List[np.ndarray] = []
        self.h80: List[np.ndarray] = []
        self.w80: List[np.ndarray] = []
        self.dr: List[np.ndarray] = []
        self.gold_err = np.zeros(n_items)
        self.gold_y = np.zeros(n_items)
        self.abs_sum = np.zeros(n_items)
        self.cnt = np.zeros(n_items)

    def add(self, ii, Yv, Fv, X0v, lo50, hi50, lo80, hi80):
        n_items = len(self.cnt)
        E = Yv - Fv
        self.E.append(E.astype(np.float32))
        self.h50.append(((Yv >= lo50) & (Yv <= hi50)).astype(np.int8))
        self.h80.append(((Yv >= lo80) & (Yv <= hi80)).astype(np.int8))
        self.w80.append((hi80 - lo80).astype(np.float32))
        moved, act = Fv - X0v, Yv - X0v
        hit = (np.sign(moved) == np.sign(act)) & (act != 0)
        self.dr.append(np.where(np.abs(moved) > MOVE, hit, -1).astype(np.int8))
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


def _bin_quantiles(E: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """VOL_BINS x 4 table of the q10 / q25 / q75 / q90 of past misses, one row per jumpiness level (a thin level borrows the pooled quantiles)."""
    qs = (0.1, 0.25, 0.75, 0.9)
    pooled = np.quantile(E, qs)
    out = np.tile(pooled, (VOL_BINS, 1))
    for b in range(VOL_BINS):
        e = E[bins == b]
        if len(e) >= MIN_BIN:
            out[b] = np.quantile(e, qs)
    return out


def walk_forward(Y: np.ndarray, days: pd.DatetimeIndex, gi: Optional[np.ndarray], hs=HORIZONS, sarima: bool = False) -> dict:
    """The whole test. Returns {acc: {(model, h): _Acc}, models, folds: [(cut date, next cut date)], rows}."""
    n, T = Y.shape
    wk = days.dayofweek.to_numpy()
    obs = np.isfinite(Y)
    first = np.where(obs.any(1), obs.argmax(1), T)
    age_ok = (np.arange(T)[None, :] - first[:, None]) >= WARMUP
    sig = volatility(Y)
    static, base = static_forecasts(Y, wk, hs)
    feat = _features(Y, base, gi)
    models = list(STATIC) + ([SARIMA] if sarima else []) + [POOLED[0]] + ([POOLED[1]] if gi is not None else [])
    acc = {(m, h): _Acc(n) for m in models for h in hs}
    T0 = WARMUP
    span = T - 1 - T0
    cuts = [T0 + int(span * (START_FRAC + k * (1 - START_FRAC) / FOLDS)) for k in range(FOLDS)] + [T - 1]
    done, sar_info = [], []
    for k in range(FOLDS):
        c, c_next = cuts[k], cuts[k + 1]
        c_fit = T0 + int((c - T0) * (1 - CAL_FRAC))
        sar = None
        if sarima:
            from . import sarima as SA
            sar, info = SA.forecast_all(Y, c_fit, max(c_fit - max(hs) + 1, 0), c_next, hs)
            sar_info.append(info)
        for h in hs:
            valid = obs & age_ok
            valid = valid & np.concatenate([obs[:, h:], np.zeros((n, h), bool)], axis=1)
            fit_cols = np.arange(0, max(c_fit - h + 1, 0), FIT_STEP)
            cal_cols = np.arange(max(c_fit - h + 1, 0), c - h + 1, TEST_STEP)
            test_cols = np.arange(c + 1, min(c_next, T - 1 - h) + 1, TEST_STEP)
            if len(cal_cols) == 0 or len(test_cols) == 0:
                continue
            guess_c, guess_t = {}, {}
            for m in STATIC:
                guess_c[m], guess_t[m] = static[m][h][:, cal_cols], static[m][h][:, test_cols]
            if sar is not None:      # an item SARIMA could not be fitted for falls back to the ema25 guess, so it stays in the comparison
                fb = static["ema25"][h]
                guess_c[SARIMA] = np.where(np.isfinite(sar[h][:, cal_cols]), sar[h][:, cal_cols], fb[:, cal_cols])
                guess_t[SARIMA] = np.where(np.isfinite(sar[h][:, test_cols]), sar[h][:, test_cols], fb[:, test_cols])
            for m in models:
                if m not in POOLED:
                    continue
                beta = pooled_fit(Y, base, feat, wk, valid, h, fit_cols, grp=(m == "pooled_grp"))
                if beta is None:
                    continue
                guess_c[m] = pooled_predict(beta, base, feat, wk, h, cal_cols, m == "pooled_grp")
                guess_t[m] = pooled_predict(beta, base, feat, wk, h, test_cols, m == "pooled_grp")
            use = [m for m in models if m in guess_t]
            if "naive" not in use:
                continue
            Yc, Yt = Y[:, cal_cols + h], Y[:, test_cols + h]
            X0t = Y[:, test_cols]
            vc, vt = valid[:, cal_cols] & np.isfinite(sig[:, cal_cols]), valid[:, test_cols] & np.isfinite(sig[:, test_cols])
            sc = sig[:, cal_cols]
            if (vc & np.isfinite(sc)).sum() < MIN_BIN * VOL_BINS:
                continue
            edges = np.quantile(sc[vc], np.linspace(0, 1, VOL_BINS + 1)[1:-1])
            mt = vt.copy()
            for m in use:
                mt &= np.isfinite(guess_t[m])
            ii, jj = np.nonzero(mt)
            if len(ii) == 0:
                continue
            bins_t = np.digitize(sig[:, test_cols][ii, jj], edges)
            for m in use:
                Ec = Yc - guess_c[m]
                mc = vc & np.isfinite(Ec)
                q = _bin_quantiles(Ec[mc], np.digitize(sc[mc], edges))
                Fv = guess_t[m][ii, jj]
                lo80, hi80 = Fv + q[bins_t, 0], Fv + q[bins_t, 3]
                lo50, hi50 = Fv + q[bins_t, 1], Fv + q[bins_t, 2]
                acc[(m, h)].add(ii, Yt[ii, jj], Fv, X0t[ii, jj], lo50, hi50, lo80, hi80)
        done.append((days[min(c + 1, T - 1)], days[min(c_next, T - 1)]))
    return {"acc": acc, "models": models, "folds": done, "horizons": tuple(hs), "sarima_info": sar_info}


def summarize(res: dict) -> pd.DataFrame:
    """One row per (horizon, model): typical miss (median, as a percent of the price), average miss, miss against naive (below 1.0 beats it), WAPE (all items pooled, so pricey
    items weigh most) and WAPE of the median item, bias and tracking signal (positive = the real price ran above the guess), RMSE and 95th-percentile miss (the safety grades),
    the share of big misses (over 25%), direction hit rate, and whether the ranges hold."""
    rows = []
    acc, models = res["acc"], res["models"]
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
                "big_pct": 100.0 * float((ae > BIG).mean()),
                "dir_hit": float(dv.mean()) if len(dv) else np.nan, "dir_n": int(len(dv)),
                "hit50": float(a.cat("h50").mean()), "hit80": float(a.cat("h80").mean()), "width80_pct": _pct(float(np.median(a.cat("w80")))),
            })
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


def bake_off(hist: pd.DataFrame, groups: Optional[pd.DataFrame] = None, metric: str = "price", hs=HORIZONS, sarima: bool = True) -> dict:
    """The full test for one metric. {} when the history is too thin. sarima=False leaves SARIMA out (it is the slow one); it is also left out, with a note, when scipy is missing."""
    g = build_grid(hist, metric)
    if g is None:
        return {}
    ids, days, Y = g
    if len(days) < MIN_SPAN:
        return {}
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
    res = walk_forward(Y, days, gi, hs, sarima=sarima)
    summ = summarize(res)
    if summ.empty:
        return {}
    return {"summary": summ, "best": best_models(summ), "family": family_table(res, fam), "items": len(ids), "days": len(days),
            "first": days[0], "last": days[-1], "folds": res["folds"], "grouped": gi is not None,
            "sarima_info": res["sarima_info"], "sarima_note": note}
