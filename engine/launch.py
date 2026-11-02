"""Launch forecaster: what will an item's settled price and stock be, given only its first days on the auction house?

Retail history (Undermine, see history.py) holds several launches: whole weeks in which many items first appeared (an expansion or a big patch). For each item we know
how its price and stock looked in its first days compared with its own settled level (median over days 28-56 of age). That turns the question round: seen on day k at
price P, an item's settled price is P divided by "the multiple items usually have on day k". Spread in that multiple gives the range.

Three guesses are compared, always on a launch the model has NOT seen (leave one launch out):
  naive    - the settled level equals today's price (what people assume)
  age      - today's price divided by the typical day-k multiple of all other launches' items
  family   - the same, using only items of the same family when 15+ of them are known
  path     - age plus the item's own path so far: how far the price fell from day 0 to day k says something about how much of the fall is still to come
Every guess comes as a range (middle 50% and middle 80% of what happened to other items). The backtest reports how often the truth really fell inside each range and
how big a miss typically is, so the error bars are measured, not assumed.

Use on Forever: the prior table (engine/launch_prior) is applied to an item's first days; the same ranges come out, and the backtest says how far to trust them. They are
retail ranges: Forever starts from an empty economy, so treat them as a starting prior to be re-fitted on Forever's own first weeks. Aggregates only; nothing item-level is printed."""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

AGES_USED = (1, 2, 3, 5, 7, 10, 14, 21)         # ages at which a forecast is made and tested (the settled level itself is the median over days 28-56)
PRIOR_AGES = (0,) + AGES_USED
MAX_FORECAST_AGE = 24
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_TRAIN = 30                                    # items needed in the other launches to fit anything
MIN_TEST = 15                                     # items needed in a launch to test on it
MIN_FAMILY = 15                                   # items of one family needed for a family-specific guess
MIN_FIT = 20                                      # items needed to fit the path line
LAUNCH_GAP_DAYS = 14                              # burst weeks closer than this belong to one launch
MODELS = ("age", "family", "path")


def launch_labels(fm: dict) -> pd.Series:
    """item_id -> the first Monday of the launch it first appeared in (only items from burst weeks; adjacent burst weeks are one launch)."""
    bw = sorted(w for w, _ in fm.get("burst_weeks", []))
    if not bw or "wk" not in fm:
        return pd.Series(dtype="datetime64[ns]")
    lab, cur, prev = {}, None, None
    for w in bw:
        if prev is None or (w - prev).days > LAUNCH_GAP_DAYS:
            cur = w
        lab[w] = cur
        prev = w
    return fm["wk"].map(lab).dropna()


def _family(groups: Optional[pd.DataFrame], ids) -> pd.Series:
    if groups is not None and len(groups) and "family" in groups.columns:
        return groups.drop_duplicates("item_id").set_index("item_id")["family"].reindex(ids).fillna("")
    return pd.Series("", index=ids)


def _q(v) -> np.ndarray:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.quantile(v, QS) if len(v) else np.full(len(QS), np.nan)


def _line(d: np.ndarray, y: np.ndarray):
    """y = a + b d by least squares; a flat line through the mean if d does not vary."""
    if len(d) < 3 or np.std(d) < 1e-9:
        return float(np.mean(y)), 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b, a = np.polyfit(d, y, 1)
    return float(a), float(b)


def fit_path(C: pd.DataFrame, ids, k: int):
    """Line from d = log10(P_k / P_0) (known on day k) to y = log10(P_k / settled) (unknown), plus the residual quantiles; None if too few items."""
    if k not in C.columns or 0 not in C.columns:
        return None
    y, d = C.loc[C.index.intersection(ids), k], (C[k] - C[0]).reindex(C.index.intersection(ids))
    m = (y.notna() & d.notna()).to_numpy()
    if m.sum() < MIN_FIT:
        return None
    a, b = _line(d.to_numpy()[m], y.to_numpy()[m])
    return {"n": int(m.sum()), "a": a, "b": b, "q": _q(y.to_numpy()[m] - (a + b * d.to_numpy()[m]))}


def _fold(C: pd.DataFrame, fam: pd.Series, train, test, k: int, launch, metric: str) -> Optional[pd.DataFrame]:
    if k not in C.columns or 0 not in C.columns:
        return None
    y_all, d_all = C[k], C[k] - C[0]
    yt, dt = y_all.reindex(test), d_all.reindex(test)
    keep = (yt.notna() & dt.notna()).to_numpy()
    ytr = y_all.reindex(train).dropna()
    if keep.sum() < MIN_TEST or len(ytr) < MIN_TRAIN:
        return None
    yt, dt = yt[keep], dt[keep]
    q_all = _q(ytr.to_numpy())
    famq = {f: _q(s.to_numpy()) for f, s in ytr.groupby(fam.reindex(ytr.index)) if f and len(s) >= MIN_FAMILY}
    path = fit_path(C, train, k)
    rows = []
    for i, y, d in zip(yt.index, yt.to_numpy(float), dt.to_numpy(float)):
        qf = famq.get(fam.get(i, ""), q_all)
        r = {"launch": launch, "metric": metric, "age": k, "item_id": i, "e_naive": abs(y)}
        for name, mu, q in (("age", 0.0, q_all), ("family", 0.0, qf), ("path", (path["a"] + path["b"] * d) if path else np.nan, path["q"] if path else np.full(5, np.nan))):
            r[f"e_{name}"] = abs(y - (mu + q[2]))
            r[f"h50_{name}"] = float(mu + q[1] <= y <= mu + q[3]) if np.isfinite(mu + q[1]) else np.nan
            r[f"h80_{name}"] = float(mu + q[0] <= y <= mu + q[4]) if np.isfinite(mu + q[0]) else np.nan
        r["w80_age"] = q_all[4] - q_all[0]
        r["w80_family"] = qf[4] - qf[0]
        r["w80_path"] = (path["q"][4] - path["q"][0]) if path else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def backtest(fm: dict, groups: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Leave one launch out. One row per held-out item, metric (price / stock) and age, with each guess's miss (log10) and whether the 50% and 80% ranges held it."""
    lab = launch_labels(fm)
    if lab.empty:
        return pd.DataFrame()
    fam = _family(groups, fm["ids"])
    parts = []
    for L in sorted(lab.unique()):
        test, train = lab.index[lab == L], lab.index[lab != L]
        if len(test) < MIN_TEST or len(train) < MIN_TRAIN:
            continue
        for metric, C in (("price", fm["Cp"]), ("stock", fm["Cs"])):
            for k in AGES_USED:
                p = _fold(C, fam, train, test, k, pd.Timestamp(L), metric)
                if p is not None and len(p):
                    parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize(details: pd.DataFrame, by=("metric", "age")) -> pd.DataFrame:
    """Per metric and age: items tested; the typical miss of each guess as a multiple (1.0 = exact, 2.0 = off by a factor of two, median over items);
    how often the 50% / 80% ranges held the truth; the width of the 80% range as a multiple."""
    if details is None or details.empty:
        return pd.DataFrame()
    rows = []
    for key, g in details.groupby(list(by)):
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r["items"] = int(len(g))
        r["launches"] = int(g["launch"].nunique())
        r["miss_naive"] = 10 ** g["e_naive"].median()
        for m in MODELS:
            r[f"miss_{m}"] = 10 ** g[f"e_{m}"].median() if g[f"e_{m}"].notna().any() else np.nan
            r[f"hit50_{m}"] = g[f"h50_{m}"].mean()
            r[f"hit80_{m}"] = g[f"h80_{m}"].mean()
            r[f"width80_{m}"] = 10 ** g[f"w80_{m}"].median() if g[f"w80_{m}"].notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)


def best_models(summary: pd.DataFrame) -> Dict[tuple, str]:
    """For each (metric, age) the guess with the smallest typical miss; ties and tiny gains (under 3%) go to the plain age guess."""
    out = {}
    for _, r in summary.iterrows():
        cand = {m: r[f"miss_{m}"] for m in MODELS if r.get(f"miss_{m}") == r.get(f"miss_{m}")}
        pick = "age"
        if cand:
            m = min(cand, key=cand.get)
            if m != "age" and cand[m] < cand.get("age", np.inf) / 1.03:
                pick = m
        out[(r["metric"], int(r["age"]))] = pick
    return out


def build_prior(fm: dict, groups: Optional[pd.DataFrame] = None, best: Optional[dict] = None) -> pd.DataFrame:
    """The table to keep: for scope 'all' and each family with 15+ launch items, metric (price / stock), model (age / path) and age, the quantiles (log10) of the
    multiple of the settled level; for 'path' also the line a + b d. best_model says which guess the backtest favoured at that metric and age."""
    lab = launch_labels(fm)
    if lab.empty:
        return pd.DataFrame()
    ids = lab.index
    fam = _family(groups, ids)
    scopes = {"all": ids}
    for f, s in fam.groupby(fam):
        if f and len(s) >= MIN_FAMILY:
            scopes[f] = s.index
    rows = []
    for scope, sid in scopes.items():
        for metric, C in (("price", fm["Cp"]), ("stock", fm["Cs"])):
            for k in PRIOR_AGES:
                if k not in C.columns:
                    continue
                v = C.loc[C.index.intersection(sid), k].dropna().to_numpy(float)
                if len(v) >= 8:
                    q = _q(v)
                    rows.append({"scope": scope, "metric": metric, "model": "age", "age": k, "n": len(v), "a": np.nan, "b": np.nan,
                                 **{f"l{int(x * 100)}": q[j] for j, x in enumerate(QS)}})
                if k > 0:
                    p = fit_path(C, sid, k)
                    if p:
                        rows.append({"scope": scope, "metric": metric, "model": "path", "age": k, "n": p["n"], "a": p["a"], "b": p["b"],
                                     **{f"l{int(x * 100)}": p["q"][j] for j, x in enumerate(QS)}})
    out = pd.DataFrame(rows)
    if len(out):
        out["best_model"] = [(best or {}).get((m, int(a)), "age") for m, a in zip(out["metric"], out["age"])]
    return out


def settled_estimate(prior: pd.DataFrame, scope: str, metric: str, age: int, now: float, day0: Optional[float] = None, model: str = "best") -> Optional[dict]:
    """Settled price (or stock) range for an item seen at `now` on `age` days after it first appeared (`day0` = its value on its first day, for the path guess).
    Returns mid and the 50% and 80% ranges, or None if the prior cannot say (age beyond 24 days, no prior rows)."""
    if prior is None or prior.empty or not (now is not None and now > 0) or age > MAX_FORECAST_AGE:
        return None
    p = prior[prior["metric"] == metric]
    if p.empty:
        return None
    ages = np.array(sorted(p["age"].unique()))
    k = int(ages[np.abs(ages - age).argmin()])
    pk = p[p["age"] == k]
    want = (pk["best_model"].iloc[0] if "best_model" in pk.columns else "age") if model == "best" else model
    if want == "path" and (k == 0 or not (day0 is not None and day0 > 0)):
        want = "family"
    use_scope = scope if scope in set(pk["scope"]) else "all"
    if want == "path":
        row = pk[(pk["model"] == "path") & (pk["scope"] == use_scope)]
        if row.empty:
            row = pk[(pk["model"] == "path") & (pk["scope"] == "all")]
        if row.empty:
            want = "family"
    if want != "path":
        sc = use_scope if want == "family" else "all"
        row = pk[(pk["model"] == "age") & (pk["scope"] == sc)]
        if row.empty:
            row = pk[(pk["model"] == "age") & (pk["scope"] == "all")]
    if row.empty:
        return None
    r = row.iloc[0]
    mu = float(r["a"] + r["b"] * np.log10(now / day0)) if r["model"] == "path" else 0.0
    L = {q: mu + float(r[f"l{q}"]) for q in (10, 25, 50, 75, 90)}
    return {"scope": r["scope"], "model": "path" if r["model"] == "path" else ("family" if r["scope"] != "all" else "age"), "age": k, "n": int(r["n"]),
            "mult_mid": 10 ** L[50], "mid": now / 10 ** L[50], "lo50": now / 10 ** L[75], "hi50": now / 10 ** L[25], "lo80": now / 10 ** L[90], "hi80": now / 10 ** L[10]}


def launch_report(d: pd.DataFrame, prior: pd.DataFrame, groups: Optional[pd.DataFrame] = None, max_age: int = 21) -> pd.DataFrame:
    """Settled price and stock ranges for items that are young in a daily table (item_id, day, price in gold, quantity). Age counts from the first day the table sees
    the item, so the table must start at the launch for ages to mean what they did in the retail launches. Only items seen on the latest day (or the day before)."""
    need = ["item_id", "day", "price", "quantity"]
    if d is None or d.empty or any(c not in d.columns for c in need) or prior is None or prior.empty:
        return pd.DataFrame()
    g = d[need].copy()
    g["day"] = pd.to_datetime(g["day"])
    g = g[(g["price"] > 0) & (g["quantity"] > 0)].sort_values(["item_id", "day"])
    if g.empty:
        return pd.DataFrame()
    last_day = g["day"].max()
    f, l = g.groupby("item_id").first(), g.groupby("item_id").last()
    fam = _family(groups, f.index)
    rows = []
    for i in f.index:
        age = int((l.loc[i, "day"] - f.loc[i, "day"]).days)
        if age > max_age or l.loc[i, "day"] < last_day - pd.Timedelta(days=1):
            continue
        pe = settled_estimate(prior, fam.get(i, ""), "price", age, float(l.loc[i, "price"]), float(f.loc[i, "price"]))
        if pe is None:
            continue
        se = settled_estimate(prior, fam.get(i, ""), "stock", age, float(l.loc[i, "quantity"]), float(f.loc[i, "quantity"]))
        rows.append({"item_id": int(i), "family": fam.get(i, ""), "age": age, "price_now": float(l.loc[i, "price"]), "units_now": float(l.loc[i, "quantity"]),
                     "price_now_vs_settled": pe["mult_mid"], "settled_mid": pe["mid"], "settled_lo50": pe["lo50"], "settled_hi50": pe["hi50"],
                     "settled_lo80": pe["lo80"], "settled_hi80": pe["hi80"], "price_model": pe["model"],
                     "settled_units_mid": se["mid"] if se else np.nan})
    return pd.DataFrame(rows)
