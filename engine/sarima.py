"""A compact SARIMA(p,d,q)(P,D,Q)7 for daily log prices, fitted by conditional least squares (the method R calls arima(method = "CSS")).

Why not statsmodels: the build sandbox cannot install it, and code that cannot be run here cannot be tested here. This module is the same model written out in about a
hundred lines of numpy / scipy, tested against known answers (it recovers planted AR / seasonal parameters, its one-step forecasts match the textbook formulas, and a (0,1,1)
fit reproduces an exponential average exactly). The differences from statsmodels: the fit is conditional least squares rather than exact likelihood, and short gaps are
bridged by interpolation. If SARIMA turns out to be the winner the exact-likelihood fit can be swapped in and compared on the same cases.

How it works. The seasonal and plain parts multiply into one AR polynomial a(B) and one MA polynomial m(B); differencing is folded into a(B) as well, so every spec is
"a(B) x_t = m(B) e_t" with x the price minus an offset (a fitted mean for stationary specs, the first value for differenced ones). The innovations e come from one filter pass
(lfilter), so they depend only on days up to t. An h-day forecast runs the recursion forward with future innovations set to zero. Nothing after the origin is ever used."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

SEASON = 7
BOUND = 0.98                # AR / MA coefficients are kept inside +-0.98 (stationary and invertible)
BURN = 40                   # the first 40 innovations are ignored (start-up transient)
MIN_FIT = 120               # an item needs this many days in the fitting window
# (p, d, q, P, D, Q): mean-reverting around a fixed level / local level with a weekly MA / the "airline" model / damped trend with weekly parts
SPECS: Tuple[Tuple[int, int, int, int, int, int], ...] = (
    (1, 0, 1, 1, 0, 1),
    (0, 1, 1, 0, 0, 1),
    (0, 1, 1, 0, 1, 1),
    (1, 1, 1, 1, 0, 1),
)


def _poly(coefs: Sequence[float], step: int = 1, sign: float = 1.0) -> np.ndarray:
    """1 + sign * (c1 B^step + c2 B^2step + ...) as a coefficient array."""
    out = np.zeros(len(coefs) * step + 1)
    out[0] = 1.0
    for i, c in enumerate(coefs, start=1):
        out[i * step] = sign * c
    return out


def polynomials(spec: Tuple[int, int, int, int, int, int], params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(a, m, mean): the full AR polynomial (differencing included), the full MA polynomial, and the fitted mean (0 for differenced specs)."""
    p, d, q, P, D, Q = spec
    i = 0
    phi, Phi, theta, Theta = params[i:i + p], params[i + p:i + p + P], params[i + p + P:i + p + P + q], params[i + p + P + q:i + p + P + q + Q]
    mean = float(params[p + P + q + Q]) if (d == 0 and D == 0) else 0.0
    a = np.convolve(_poly(phi, 1, -1.0), _poly(Phi, SEASON, -1.0))
    for _ in range(d):
        a = np.convolve(a, [1.0, -1.0])
    for _ in range(D):
        a = np.convolve(a, _poly([1.0], SEASON, -1.0))
    m = np.convolve(_poly(theta, 1, 1.0), _poly(Theta, SEASON, 1.0))
    return a, m, mean


def n_params(spec) -> int:
    p, d, q, P, D, Q = spec
    return p + P + q + Q + (1 if (d == 0 and D == 0) else 0)


def innovations(x: np.ndarray, a: np.ndarray, m: np.ndarray) -> np.ndarray:
    from scipy.signal import lfilter
    return lfilter(a, m, x)


def _offset(y: np.ndarray, spec, mean: float) -> float:
    return mean if (spec[1] == 0 and spec[4] == 0) else float(y[0])


def fit_spec(y: np.ndarray, spec) -> Optional[Tuple[np.ndarray, float]]:
    """(params, aic) for one spec on a gap-free series, or None if the optimiser gives up."""
    from scipy.optimize import minimize
    k = n_params(spec)
    stationary = spec[1] == 0 and spec[4] == 0
    x0 = np.zeros(k)
    bounds = [(-BOUND, BOUND)] * (k - (1 if stationary else 0))
    if stationary:
        x0[-1] = float(np.mean(y))
        bounds.append((float(np.min(y)) - 1.0, float(np.max(y)) + 1.0))

    def sse(params):
        a, m, mean = polynomials(spec, params)
        e = innovations(y - _offset(y, spec, mean), a, m)[BURN:]
        v = float(e @ e)
        return v if np.isfinite(v) else 1e12

    try:
        r = minimize(sse, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
    except Exception:
        return None
    rows = len(y) - BURN
    v = max(float(r.fun), 1e-12)
    return r.x, rows * np.log(v / rows) + 2 * k


def fit_best(y: np.ndarray, specs=SPECS):
    """The spec with the lowest AIC on the fitting window: (spec, params), or None when the series is too short or nothing fits."""
    if len(y) < MIN_FIT or not np.all(np.isfinite(y)):
        return None
    best = None
    for s in specs:
        r = fit_spec(y, s)
        if r is not None and (best is None or r[1] < best[2]):
            best = (s, r[0], r[1])
    return None if best is None else (best[0], best[1])


def _lag(v: np.ndarray, s: int) -> np.ndarray:
    """out[t] = v[t - s] (0 before the start)."""
    if s == 0:
        return v
    out = np.zeros_like(v)
    out[s:] = v[:-s]
    return out


def forecast_series(y: np.ndarray, spec, params: np.ndarray, hs: Sequence[int]) -> Dict[int, np.ndarray]:
    """For every origin t, the h-day-ahead forecast made with days up to t only: {h: array over t}."""
    a, m, mean = polynomials(spec, params)
    off = _offset(y, spec, mean)
    x = y - off
    e = innovations(x, a, m)
    H = max(hs)
    zf: Dict[int, np.ndarray] = {}
    for k in range(1, H + 1):
        acc = np.zeros_like(x)
        for i in range(1, len(a)):
            acc -= a[i] * (zf[k - i] if k - i >= 1 else _lag(x, i - k))
        for j in range(k, len(m)):
            acc += m[j] * _lag(e, j - k)
        zf[k] = acc
    return {h: off + zf[h] for h in hs}


def fill_gaps(y: np.ndarray) -> np.ndarray:
    """Interior gaps are bridged by straight-line interpolation (only values on both sides are used, so an observed origin never sees the future); the tail is carried forward."""
    import pandas as pd
    s = pd.Series(y)
    return s.interpolate(limit_area="inside").ffill().to_numpy()


def forecast_all(Y: np.ndarray, fit_end: int, lo: int, hi: int, hs: Sequence[int], specs=SPECS) -> Tuple[Dict[int, np.ndarray], dict]:
    """Fit each item on its days up to `fit_end`, keep the parameters fixed, and forecast from every origin in [lo, hi].
    Returns ({h: items x days array, NaN outside [lo, hi] and for items that could not be fitted}, {"fitted": n, "failed": n, "specs": {spec: count}})."""
    n, T = Y.shape
    out = {h: np.full((n, T), np.nan) for h in hs}
    info = {"fitted": 0, "failed": 0, "specs": {}}
    for i in range(n):
        row = Y[i, :hi + 1]
        obs = np.isfinite(row)
        if not obs.any():
            info["failed"] += 1
            continue
        first = int(obs.argmax())
        y = fill_gaps(row[first:])
        tr = y[:max(fit_end - first + 1, 0)]
        fit = fit_best(tr, specs)
        if fit is None:
            info["failed"] += 1
            continue
        spec, params = fit
        info["fitted"] += 1
        info["specs"][spec] = info["specs"].get(spec, 0) + 1
        f = forecast_series(y, spec, params, hs)
        a, b = max(lo - first, 0), hi - first + 1
        for h in hs:
            out[h][i, first + a:first + b] = f[h][a:b]
    return out, info
