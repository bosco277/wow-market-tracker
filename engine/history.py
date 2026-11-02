"""Part 4 of the group test: what years of Undermine DAILY history (price and units listed, no sales) say about groups.

Input: the `market_history` table (item_id, day, price in copper, quantity) that `tools/lake.py undermine` stores, and the item_groups table.
Price is the lowest price and quantity the units listed at the day's busiest snapshot. These are not sales. What they can show, for each group:

  4A weekday rhythm   - does the group have its own weekly pattern of price and stock, and is it the same in the early years and the late years?
  4B moving together  - do items in a group have high and low price / stock days together, beyond the whole market moving at once?
  4C fresh items      - after an item first appears, how do its price and stock settle? Items that first appear in a burst (a launch) are kept apart.

Every day is compared with the item's OWN typical level (its centred 29-day median), so quantities and long trends drop out. Only items that
are still traded today are in the history (survivorship), so shapes are trustworthy and levels are not. Aggregates only; nothing item-level is printed."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .grouptest import HELPS, MIN_PEERS, MIN_SCORED, comove_by, comove_verdict, rhythm_verdict, spike_verdict

WINDOW = 29                 # an item's typical level on a day = the median of the 29 days around it
MIN_WINDOW = 15
CLIP = 1.0                  # lifts beyond 10x either way are clipped, so one broken day cannot drive a correlation
MIN_DAYS_HIST = 365         # weekday rhythm and co-movement need an item with at least a year of days
MIN_OBS_DOW = 26            # and at least this many of each weekday (half a year)
STABLE_R = 0.5              # a group's early-years and late-years weekday patterns must correlate at least this much
MIN_SWING_PCT = 3.0         # a weekly pattern whose best and worst weekday differ by less than this is statistically visible but too small to use
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
AGES = (0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 42, 56)
PLATEAU = (28, 56)          # a fresh item's settled level = its median over these days of age
NEW_AFTER = 30              # an item is "fresh" only if it first appears this many days after tracking began ...
MIN_AFTER = 70              # ... and has this many days of history after that
BURST_MIN = 6               # a week is a launch burst if at least this many sampled items first appear in it ...
BURST_FACTOR = 2.5          # ... and at least this many times a typical week
SETTLED = np.log10(1.15)    # settled = within 15% of the settled level
SETTLE_RUN = 7              # ... for 7 days running


def clean(hist: pd.DataFrame) -> pd.DataFrame:
    """Long table item_id, day (Timestamp), price (gold), quantity; positive values only, one row per item per day."""
    need = ["item_id", "day", "price", "quantity"]
    if hist is None or hist.empty or any(c not in hist.columns for c in need):
        return pd.DataFrame(columns=need)
    h = hist[need].copy()
    h["day"] = pd.to_datetime(h["day"])
    h["price"] = pd.to_numeric(h["price"], errors="coerce") / 10000.0
    h["quantity"] = pd.to_numeric(h["quantity"], errors="coerce")
    h = h[(h["price"] > 0) & (h["quantity"] > 0)].drop_duplicates(["item_id", "day"], keep="last")
    h["item_id"] = h["item_id"].astype(int)
    return h.reset_index(drop=True)


def matrices(h: pd.DataFrame, min_days: int = MIN_DAYS_HIST) -> dict:
    """Items x days grids of price and stock for items with at least min_days days of history (missing days are NaN)."""
    if h is None or h.empty:
        return {}
    cnt = h.groupby("item_id").size()
    h = h[h["item_id"].isin(cnt[cnt >= min_days].index)]
    if h.empty:
        return {}
    days = pd.date_range(h["day"].min(), h["day"].max(), freq="D")
    P = h.pivot(index="item_id", columns="day", values="price").reindex(columns=days)
    Q = h.pivot(index="item_id", columns="day", values="quantity").reindex(columns=days)
    return {"price": P, "stock": Q, "items": int(P.shape[0]), "days": int(len(days)), "first": days[0], "last": days[-1]}


def detrended(M: pd.DataFrame) -> pd.DataFrame:
    """log10(value / the item's centred 29-day median), clipped to +-1. 0 = a normal day, 0.30 = double, -0.30 = half."""
    base = M.T.rolling(WINDOW, center=True, min_periods=MIN_WINDOW).median().T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(M / base).clip(-CLIP, CLIP)


# ------------------------------------------------------------------ 4A weekday rhythm
def weekday_profiles(L: pd.DataFrame, min_obs: int = MIN_OBS_DOW) -> pd.DataFrame:
    """Per item the median lift on each weekday (columns 0 = Monday ... 6 = Sunday), re-centred to average 0 so it describes the SHAPE of the week."""
    dow = np.asarray(L.columns.dayofweek)
    cols = []
    for d in range(7):
        sub = L.loc[:, dow == d]
        med = sub.median(axis=1, skipna=True)
        med[sub.notna().sum(axis=1) < min_obs] = np.nan
        cols.append(med)
    P = pd.concat(cols, axis=1)
    P.columns = list(range(7))
    P = P.dropna()
    return P.sub(P.mean(axis=1), axis=0)


def _key(groups: pd.DataFrame, by: str) -> pd.DataFrame:
    k = groups.drop_duplicates("item_id").set_index("item_id").assign(_all="all")
    return k[[by]] if by in k.columns else pd.DataFrame()


def weekday_by(groups: pd.DataFrame, P: pd.DataFrame, by: str = "group", early: pd.DataFrame = None, late: pd.DataFrame = None) -> pd.DataFrame:
    """Each item's weekday shape is guessed from the median shape of the OTHER items in its group (never itself). vs_flat = miss of guessing 'no weekly pattern' /
    miss of the group guess; vs_all = miss of the all-other-items guess / miss of the group guess (1.0 = no help). swing_pct = best weekday against worst in the group's
    median shape. stable_r = how well the group's shape in the early half of the history matches the late half."""
    k = _key(groups, by)
    if P is None or P.empty or k.empty:
        return pd.DataFrame()
    m = P.join(k, how="inner")
    m = m[m[by].astype(str) != ""]
    if m.empty:
        return pd.DataFrame()
    X = m[list(range(7))].to_numpy(float)
    lab = m[by].to_numpy()
    rows = []
    for g in pd.unique(lab):
        sel = lab == g
        n = int(sel.sum())
        Xg, Xo = X[sel], X[~sel]
        r = {by: g, "items": n, "swing_pct": np.nan, "peak": "", "low": "", "vs_flat": np.nan, "vs_all": np.nan, "stable_r": np.nan}
        prof = np.median(Xg, axis=0)
        r["swing_pct"] = (10 ** (prof.max() - prof.min()) - 1) * 100
        r["peak"], r["low"] = WEEKDAYS[int(prof.argmax())], WEEKDAYS[int(prof.argmin())]
        if n > MIN_PEERS:
            miss_flat = np.abs(Xg).mean(axis=1)
            miss_group = np.array([np.abs(Xg[i] - np.median(np.delete(Xg, i, axis=0), axis=0)).mean() for i in range(n)])
            denom = np.median(miss_group)
            if denom > 1e-9:
                r["vs_flat"] = np.median(miss_flat) / denom
                if by != "_all" and len(Xo) >= MIN_PEERS:
                    r["vs_all"] = np.median(np.abs(Xg - np.median(Xo, axis=0)).mean(axis=1)) / denom
        if early is not None and late is not None and len(early) and len(late):
            ids = m.index[sel]
            e, l = early.reindex(ids).dropna(), late.reindex(ids).dropna()
            both = e.index.intersection(l.index)
            if len(both) >= MIN_SCORED:
                pe, pl = np.median(e.loc[both].to_numpy(float), axis=0), np.median(l.loc[both].to_numpy(float), axis=0)
                if pe.std() > 0 and pl.std() > 0:
                    r["stable_r"] = float(np.corrcoef(pe, pl)[0, 1])
        rows.append(r)
    df = pd.DataFrame(rows)
    if by == "_all":
        df["verdict"] = ["items share a weekly pattern" if v >= HELPS else "no common weekly pattern" for v in df["vs_flat"]]
    else:
        df["verdict"] = [rhythm_verdict(n, a if a == a else 0, b if b == b else 0) for n, a, b in zip(df["items"], df["vs_flat"], df["vs_all"])]
    real = df["verdict"].isin(["group-specific rhythm", "same rhythm as every item", "items share a weekly pattern"])
    small = real & (df["swing_pct"] < MIN_SWING_PCT)
    df.loc[small, "verdict"] = f"too small to use (swing under {MIN_SWING_PCT:g}%)"
    unstable = (df["stable_r"] < STABLE_R) & df["verdict"].isin(["group-specific rhythm", "same rhythm as every item", "items share a weekly pattern"])
    df.loc[unstable, "verdict"] += " (but not stable across years)"
    return df.sort_values("items", ascending=False).reset_index(drop=True)


def rhythm_report(groups: pd.DataFrame, lifts: Dict[str, pd.DataFrame]) -> dict:
    out = {}
    for tag, L in lifts.items():
        P = weekday_profiles(L)
        if P.empty:
            continue
        half = L.shape[1] // 2
        early, late = weekday_profiles(L.iloc[:, :half], MIN_OBS_DOW // 2), weekday_profiles(L.iloc[:, half:], MIN_OBS_DOW // 2)
        out[tag] = {level: weekday_by(groups, P, by, early, late) for level, by in (("overall", "_all"), ("families", "family"), ("groups", "group"))}
    return out


# ------------------------------------------------------------------ 4B moving together over years
def comove_report(groups: pd.DataFrame, lifts: Dict[str, pd.DataFrame]) -> dict:
    """For price and stock lifts: correlation with the group's other items (corr_group) against the rest of the market (corr_market); and, for the days an item's
    price / stock is at least double its normal, how many times more likely its peers are to be up too (spike_group, spike_market)."""
    out = {}
    for level, by in (("overall", "_all"), ("families", "family"), ("groups", "group")):
        parts = []
        for tag, L in lifts.items():
            t = comove_by(groups, L, by=by)
            if len(t):
                parts.append(t.add_prefix(f"{tag}_"))
            s = comove_by(groups, L, by=by, spikes=True)
            if len(s):
                parts.append(s.add_prefix(f"{tag}_"))
        if not parts:
            continue
        df = pd.concat(parts, axis=1)
        df.index.name = by
        df = df.reset_index()
        for tag in lifts:
            if f"{tag}_items" not in df.columns:
                continue
            if by == "_all":
                df[f"{tag}_verdict"] = ["items share market-wide moves" if c >= 0.25 else "no market-wide moves" for c in df[f"{tag}_corr_group"]]
            else:
                df[f"{tag}_verdict"] = [comove_verdict(n if n == n else 0, e, cg, cm) for n, e, cg, cm in
                                        zip(df[f"{tag}_items"], df[f"{tag}_extra"], df[f"{tag}_corr_group"], df[f"{tag}_corr_market"])]
            if f"{tag}_events" in df.columns:
                df[f"{tag}_spike_verdict"] = [spike_verdict(e if e == e else 0, g, a) for e, g, a in
                                              zip(df[f"{tag}_events"], df.get(f"{tag}_spike_group", np.nan), df.get(f"{tag}_spike_market", np.nan))]
        lead = next((c for c in ("price_items", "stock_items") if c in df.columns), None)
        if lead is None:                                           # nothing outside the slice to compare with (for example only one family)
            continue
        out[level] = df.sort_values(lead, ascending=False).reset_index(drop=True)
    return out


# ------------------------------------------------------------------ 4C fresh items
def _week_start(d: pd.Series) -> pd.Series:
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()


def _settle_day(med: pd.Series):
    """First age from which the median lift stays within 15% for SETTLE_RUN days running; None if it never does within the window."""
    vals = med.reindex(range(PLATEAU[1] + 1)).to_numpy(float)
    ok = np.abs(vals) <= SETTLED
    for a in range(len(vals) - SETTLE_RUN + 1):
        if ok[a:a + SETTLE_RUN].all():
            return a
    return None


def _curve(M: pd.DataFrame) -> pd.DataFrame:
    plat = M.loc[:, PLATEAU[0]:PLATEAU[1]]
    base = plat.median(axis=1)
    good = plat.notna().sum(axis=1) >= 14
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.log10(M.div(base, axis=0))
    C.loc[~good.to_numpy()] = np.nan
    return C.dropna(how="all")


def _age_table(Cp: pd.DataFrame, Cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in AGES:
        r = {"age_days": a}
        for tag, C in (("price", Cp), ("stock", Cs)):
            if a in C.columns and C[a].notna().sum() >= 3:
                q = C[a].quantile([0.25, 0.5, 0.75])
                r[f"{tag}_x"], r[f"{tag}_low"], r[f"{tag}_high"] = 10 ** q[0.5], 10 ** q[0.25], 10 ** q[0.75]
            else:
                r[f"{tag}_x"] = r[f"{tag}_low"] = r[f"{tag}_high"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fresh_curves(h: pd.DataFrame, groups: pd.DataFrame = None) -> dict:
    """Age-aligned price and stock of items that first appear after tracking began: multiples of the item's own settled level (median over days 28-56 of age).
    Items that first appear in a launch burst are kept apart from the trickle of single new items."""
    if h is None or h.empty:
        return {}
    start, end = h["day"].min(), h["day"].max()
    first = h.groupby("item_id")["day"].min()
    new = first[(first >= start + pd.Timedelta(days=NEW_AFTER)) & (first <= end - pd.Timedelta(days=MIN_AFTER))]
    out = {"new_items": int(len(new)), "start": start, "end": end}
    if len(new) < 10:
        return out
    hn = h[h["item_id"].isin(new.index)].copy()
    hn["age"] = (hn["day"] - hn["item_id"].map(first)).dt.days
    hn = hn[hn["age"] <= PLATEAU[1]]
    ages = list(range(PLATEAU[1] + 1))
    Cp = _curve(hn.pivot(index="item_id", columns="age", values="price").reindex(columns=ages))
    Cs = _curve(hn.pivot(index="item_id", columns="age", values="quantity").reindex(columns=ages))
    ids = Cp.index.intersection(Cs.index)
    Cp, Cs = Cp.loc[ids], Cs.loc[ids]
    wk = _week_start(first.loc[ids].reset_index(drop=True))
    wk.index = ids
    counts = wk.value_counts()
    thr = max(BURST_MIN, BURST_FACTOR * float(counts.median()))
    burst_weeks = sorted(counts[counts >= thr].index)
    out["burst_weeks"] = [(pd.Timestamp(w), int(counts[w])) for w in burst_weeks]
    in_burst = wk.isin(burst_weeks)
    fam = groups.drop_duplicates("item_id").set_index("item_id")["family"] if groups is not None and "family" in groups.columns else pd.Series(dtype=str)
    cohorts = {}
    for name, sel in (("launch burst", in_burst), ("trickle (single new items)", ~in_burst), ("all new items", pd.Series(True, index=ids))):
        idx = ids[sel.reindex(ids).to_numpy()]
        if len(idx) < 8:
            continue
        cohorts[name] = {"items": int(len(idx)), "table": _age_table(Cp.loc[idx], Cs.loc[idx]),
                         "settle_price": _settle_day(Cp.loc[idx].median()), "settle_stock": _settle_day(Cs.loc[idx].median())}
    out["cohorts"] = cohorts
    if len(burst_weeks):
        fb = []
        bi = ids[in_burst.reindex(ids).to_numpy()]
        for f, idx in pd.Series(bi, index=bi).groupby(fam.reindex(bi).fillna("")).groups.items():
            if not f or len(idx) < 8:
                continue
            t = _age_table(Cp.loc[idx], Cs.loc[idx]).set_index("age_days")
            fb.append({"family": f, "items": int(len(idx)), "price_day0_x": t.loc[0, "price_x"], "price_day7_x": t.loc[7, "price_x"],
                       "price_settles": _settle_day(Cp.loc[idx].median()), "stock_day0_x": t.loc[0, "stock_x"], "stock_day7_x": t.loc[7, "stock_x"],
                       "stock_day14_x": t.loc[14, "stock_x"], "stock_settles": _settle_day(Cs.loc[idx].median())})
        out["by_family"] = pd.DataFrame(fb).sort_values("items", ascending=False).reset_index(drop=True) if fb else pd.DataFrame()
    return out


# ------------------------------------------------------------------ everything
def history_report(groups: pd.DataFrame, hist: pd.DataFrame) -> dict:
    h = clean(hist)
    if h.empty:
        return {}
    res = {"rows": int(len(h)), "all_items": int(h["item_id"].nunique())}
    mats = matrices(h)
    if mats:
        res.update({k: mats[k] for k in ("items", "days", "first", "last")})
        lifts = {"price": detrended(mats["price"]), "stock": detrended(mats["stock"])}
        res["rhythm"] = rhythm_report(groups, lifts)
        res["comove"] = comove_report(groups, lifts)
    res["fresh"] = fresh_curves(h, groups)
    return res
