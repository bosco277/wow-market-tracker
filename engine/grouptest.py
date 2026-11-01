"""Does a group earn its keep? A leave-one-out test on the items table.

Each item is guessed from the OTHER items in its group (never from itself) and we measure how far off the guess is. The same
guess is also made from the item's family and from all items; a group is only worth leaning on if it beats those. Guesses are
made in log space, so a miss is a multiple: "x2.0" means the typical item is about double or half of what its peers say.

Nothing here feeds a forecast. It only tells us which groups are safe to lean on, which are too coarse, and what would split them."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PEERS = 3          # a guess needs at least this many OTHER items in the same slice
MIN_SCORED = 6         # fewer scored items than this and there is no verdict
HELPS = 1.15           # a guess has to cut the typical miss by at least 15% to count as help
BAND = 20000           # item-id slices: a rough stand-in for the expansion an item belongs to
RELIABLE = ("Medium", "High")
ERA_EDGES = (0, 120000, 200000, 240000)       # item-id edges: a rough guess at expansion boundaries, not Blizzard data
ERA_LABELS = ("old (<120k)", "mid (120-200k)", "previous (200-240k)", "current (240k+)")
SPLITS = ("era", "id_band", "quality", "era_quality", "tiered", "use")
MIN_DAYS = 6           # an item needs this many complete days before its day-to-day lifts are scored (noisy under FULL_DAYS; the median over many items averages it out)
FULL_DAYS = 14         # below this many days in the history, the lifting-together results are provisional
MIN_DAYS_DOW = 14      # a day-of-week pattern needs every weekday seen at least twice
MIN_DAILY_SOLD = 3.0   # and a typical day of at least this many units sold (below that, a day's count is mostly luck)
SPIKE = 2.0            # a "lift" day: sales at least this many times the item's own typical day
EXTRA_CORR = 0.10      # a group has its own pattern if its correlation beats the all-items one by this much
MIN_HOURS = 96         # an item needs this many usable hourly readings (4 days) before its hour-of-day rhythm is scored


def _loo_median(df: pd.DataFrame, keys: list, col: str, min_peers: int = MIN_PEERS) -> np.ndarray:
    """For every row: the median of `col` over the OTHER rows that share `keys`; NaN if fewer than `min_peers` others."""
    out = np.full(len(df), np.nan)
    if df.empty:
        return out
    vals = df[col].to_numpy(float)
    for pos in df.groupby(keys, sort=False).indices.values():
        if len(pos) - 1 < min_peers:
            continue
        v = vals[pos]
        for k in range(len(pos)):
            out[pos[k]] = np.median(np.delete(v, k))
    return out


def prepare(groups: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    need = ["item_id", "floor_gold", "daily_est", "confidence"]
    if groups is None or items is None or groups.empty or items.empty or any(c not in items.columns for c in need):
        return pd.DataFrame()
    m = groups.merge(items[need], on="item_id", how="inner")
    m["daily_est"] = pd.to_numeric(m["daily_est"], errors="coerce")
    m["floor_gold"] = pd.to_numeric(m["floor_gold"], errors="coerce")
    m = m[(m["daily_est"] > 0) & (m["floor_gold"] > 0)].reset_index(drop=True)
    if m.empty:
        return m
    for c in ("group", "family", "use", "quality", "name"):
        if c not in m.columns:
            m[c] = ""
        m[c] = m[c].fillna("").astype(str)
    if "siblings" not in m.columns:
        m["siblings"] = 1
    if "tier_guess" not in m.columns:
        m["tier_guess"] = 0
    m["log_sales"] = np.log10(m["daily_est"].astype(float))
    m["log_price"] = np.log10(m["floor_gold"].astype(float))
    m["reliable"] = m["confidence"].isin(RELIABLE)
    m["id_band"] = ((m["item_id"].astype(int) // BAND) * BAND // 1000).astype(str) + "k"
    m["tiered"] = np.where(m["siblings"] > 1, "has tiers", "single")
    m["era"] = pd.Series(ERA_LABELS)[np.searchsorted(ERA_EDGES, m["item_id"].to_numpy(), side="right") - 1].to_numpy()
    m["era_quality"] = m["era"] + " / " + m["quality"]
    m["_all"] = "all"
    return m


def _pass(m: pd.DataFrame, by: str) -> pd.DataFrame:
    """One scoring pass over the rows of `m`: per `by` value, the typical miss (a multiple) when guessing from its group, its family, all items."""
    if m.empty:
        return pd.DataFrame()
    r = pd.DataFrame({by: m[by].to_numpy()})
    for nm, col in (("sales", "log_sales"), ("price", "log_price")):
        actual = m[col].to_numpy(float)
        for lab, keys in (("group", [by]), ("family", ["family"]), ("all", ["_all"])):
            r[f"{nm}_{lab}"] = np.abs(actual - _loo_median(m, keys, col))
    r = r[r["sales_group"].notna() & r["sales_family"].notna() & r["sales_all"].notna()]      # the same items are scored for every guess
    if r.empty:
        return pd.DataFrame()
    g = r.groupby(by)
    out = pd.DataFrame({"scored": g.size()})
    for nm in ("sales", "price"):
        med = {lab: g[f"{nm}_{lab}"].median() for lab in ("group", "family", "all")}
        for lab in med:
            out[f"{nm}_off_{lab}"] = 10 ** med[lab]
        out[f"{nm}_gain"] = (med["all"] / med["group"].clip(lower=0.005)).clip(upper=99)
        out[f"{nm}_vs_family"] = (med["family"] / med["group"].clip(lower=0.005)).clip(upper=99)
    return out


def _verdict(scored, gain_all, gain_fam, noisy, top=False) -> str:
    """`top` = the rows are families themselves, so there is no bigger group to compare with."""
    if not scored or scored < MIN_SCORED:
        return "too few to judge"
    if not gain_all >= HELPS:
        v = "no better than guessing"
    elif top:
        v = "beats guessing from all items"
    elif gain_fam >= HELPS:
        v = "adds to its family"
    else:
        v = "helps, family is as good"
    return v + (" (mostly Low-confidence numbers)" if noisy else "")


def loo_by(m: pd.DataFrame, by: str = "group") -> pd.DataFrame:
    """The leave-one-out table for every value of `by` ("group", "family" or "use"). Scored twice: on every item with numbers, and
    on the items whose own numbers are Medium/High confidence (their numbers are the trustworthy ones)."""
    if m.empty:
        return pd.DataFrame()
    base = m[m[by].astype(str) != ""] if by == "use" else m
    if base.empty:
        return pd.DataFrame()
    full = _pass(base, by)
    rel = _pass(base[base["reliable"]], by)
    n = base.groupby(by).size().rename("items")
    cols = ["scored"] + [f"{nm}_{k}" for nm in ("sales", "price") for k in ("off_group", "off_family", "off_all", "gain", "vs_family")]
    out = n.to_frame().join(full.reindex(columns=cols), how="left").join(rel.reindex(columns=cols).add_prefix("rel_"), how="left")
    out["scored"] = out["scored"].fillna(0).astype(int)
    out["rel_scored"] = out["rel_scored"].fillna(0).astype(int)
    use_rel = out["rel_scored"] >= MIN_SCORED
    out["verdict"] = [
        _verdict(r.rel_scored if ur else r.scored, r.rel_sales_gain if ur else r.sales_gain, r.rel_sales_vs_family if ur else r.sales_vs_family, not ur, top=(by == "family"))
        for r, ur in zip(out.itertuples(), use_rel)]
    return out.reset_index().sort_values("items", ascending=False).reset_index(drop=True)


def tier_report(m: pd.DataFrame) -> dict:
    """Items that share a name (quality tiers). Two questions: how do the higher tiers compare with the lowest one, and is the same-name
    sibling a better guide than the group? Every guess uses the full table for its peers."""
    out = {"ratios": pd.DataFrame(), "guess": pd.DataFrame()}
    if m.empty:
        return out
    r = m[m["reliable"] & (m["siblings"] > 1)]
    low = r[r["tier_guess"] == 1].set_index("name")[["daily_est", "floor_gold"]]
    hi = r[r["tier_guess"] > 1].join(low, on="name", rsuffix="_lowest").dropna(subset=["daily_est_lowest"])
    if len(hi):
        hi = hi.assign(s=hi["daily_est"] / hi["daily_est_lowest"], p=hi["floor_gold"] / hi["floor_gold_lowest"])
        out["ratios"] = hi.groupby("tier_guess").agg(pairs=("name", "count"), sales_vs_lowest_tier=("s", "median"),
                                                     price_vs_lowest_tier=("p", "median")).reset_index()
    tiered = np.where(m["siblings"].to_numpy() > 1)[0]
    rows = []
    for nm, col in (("sales per day", "log_sales"), ("price", "log_price")):
        actual = m[col].to_numpy(float)
        sib = np.abs(actual - _loo_median(m, ["name"], col, min_peers=1))
        grp = np.abs(actual - _loo_median(m, ["group"], col))
        al = np.abs(actual - _loo_median(m, ["_all"], col))
        ok = tiered[~np.isnan(sib[tiered]) & ~np.isnan(grp[tiered]) & ~np.isnan(al[tiered])]
        if len(ok):
            rows.append(dict(what=nm, items=len(ok), off_by_same_name_tier=10 ** np.median(sib[ok]), off_by_group=10 ** np.median(grp[ok]),
                             off_by_all_items=10 ** np.median(al[ok])))
    out["guess"] = pd.DataFrame(rows)
    return out


def _slices(part: pd.DataFrame, split: str) -> str:
    """What each slice of a split looks like: median sales per day, median price, and how many items."""
    agg = part.groupby(split).agg(sales=("daily_est", "median"), price=("floor_gold", "median"), n=("item_id", "count"))
    if split == "id_band":
        agg = agg.sort_values("n", ascending=False).head(12)
        agg = agg.loc[sorted(agg.index, key=lambda k: int(str(k)[:-1]))]                  # oldest ids first
    elif split == "era":
        agg = agg.loc[[e for e in ERA_LABELS if e in agg.index]]
    else:
        agg = agg.sort_values("n", ascending=False).head(8)
    return " | ".join(f"{k}: {r.sales:,.0f}/day at {r.price:,.2f}g (n={int(r.n)})" for k, r in agg.iterrows())


def coarse_report(m: pd.DataFrame, min_items: int = 20, spread_over: float = 10.0, splits=SPLITS) -> pd.DataFrame:
    """For the groups that are big but whose items behave very differently (wide spread): how much each extra split (era, item-id
    block, quality, era+quality, has-tiers, name use) lowers the leave-one-out miss on sales per day. gain_* = miss before / miss after;
    a slice too small to guess from falls back to the whole group."""
    cols = ["group", "items", "spread", "off_group", "best_split", "off_best"] + [f"gain_{x}" for x in splits] + ["slices"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    sp = m.groupby("group")["daily_est"].agg(items="count", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
    sp["spread"] = sp["p75"] / sp["p25"].replace(0, np.nan)
    rows = []
    for gname in sp[(sp["items"] >= min_items) & (sp["spread"] >= spread_over)].sort_values("items", ascending=False).index:
        part = m[m["group"] == gname].reset_index(drop=True)
        actual = part["log_sales"].to_numpy(float)
        base = _loo_median(part, ["group"], "log_sales")
        e0 = np.abs(actual - base)
        ok = ~np.isnan(e0)
        if not ok.any():
            continue
        row = dict(group=gname, items=int(sp.loc[gname, "items"]), spread=sp.loc[gname, "spread"], off_group=10 ** np.median(e0[ok]))
        best = None
        for s_ in splits:
            row[f"gain_{s_}"] = np.nan
            if part[s_].nunique() < 2:
                continue
            p = _loo_median(part, ["group", s_], "log_sales")
            e = np.abs(actual - np.where(np.isnan(p), base, p))
            gain = min(np.median(e0[ok]) / max(np.median(e[ok]), 0.005), 99.0)
            row[f"gain_{s_}"] = gain
            if best is None or gain > best[1]:
                best = (s_, gain, 10 ** np.median(e[ok]))
        if best is None:
            continue
        row.update(best_split=best[0], off_best=best[2], slices=_slices(part, best[0]))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------ rhythm: does a group share a daily pattern?
def hourly_profiles(h: pd.DataFrame, tz: str = "America/New_York", min_rows: int = MIN_HOURS) -> dict:
    """Per-item daily rhythm from the hourly table, US Eastern. Hours flagged as spikes, blips, wipes or gaps are left out.
    sold_hour: share of an item's sales in each hour of the day (24 numbers summing to 1);
    price_hour: log10 of the item's typical price in that hour over its overall typical price (24 numbers; 0 = no difference)."""
    need = ["item_id", "ts", "floor_gold", "sold_est"]
    if h is None or h.empty or any(c not in h.columns for c in need):
        return {}
    flags = [c for c in ("spike", "blip", "flag_wipe", "flag_long_gap") if c in h.columns]
    s = h[need + flags].copy()
    bad = np.zeros(len(s), dtype=bool)
    for c in flags:
        bad |= s[c].fillna(False).astype(bool).to_numpy()
    s = s[~bad].dropna(subset=["sold_est", "floor_gold"])
    s = s[(s["floor_gold"].astype(float) > 0)]
    cnt = s.groupby("item_id").size()
    s = s[s["item_id"].isin(cnt[cnt >= min_rows].index)].copy()
    if s.empty:
        return {}
    t = pd.to_datetime(s["ts"])
    t = (t.dt.tz_localize("UTC") if t.dt.tz is None else t).dt.tz_convert(tz)
    s["hour"] = t.dt.hour
    s["sold"] = s["sold_est"].astype(float).clip(lower=0)
    med = s.groupby("item_id")["floor_gold"].transform("median").astype(float)
    s["lp"] = np.log10(s["floor_gold"].astype(float) / med)

    def grid(col, key, k, how, min_cover):
        w = s.groupby(["item_id", key])[col].agg(how).unstack(key).reindex(columns=range(k))
        w = w[w.notna().sum(axis=1) >= min_cover]
        return w.T.fillna(w.mean(axis=1)).T

    sh = grid("sold", "hour", 24, "mean", 20)
    sh = sh[sh.sum(axis=1) > 0]
    sh = sh.div(sh.sum(axis=1), axis=0)
    ph = grid("lp", "hour", 24, "median", 20)
    return {"sold_hour": sh, "price_hour": ph, "items": int(s["item_id"].nunique()), "rows": int(len(s))}


def shape_by(groups: pd.DataFrame, prof: pd.DataFrame, by: str = "group", share: bool = True) -> pd.DataFrame:
    """Leave-one-out on rhythms: each item's profile is guessed from the average profile of the OTHER items in its group, in its
    family, in all items, and from a flat profile. miss = how far the guess is from the item's own profile: for shares, the fraction
    of the item's sales that sit in the wrong hours (0-1); for price, the typical gap in log10 price (0.05 = about 12%)."""
    if prof is None or prof.empty:
        return pd.DataFrame()
    key = groups.drop_duplicates("item_id").set_index("item_id")
    key = key.assign(_all="all")
    key = key[[c for c in dict.fromkeys([by, "family", "_all"]) if c in key.columns]]
    m = prof.join(key, how="inner")
    m = m[m[by].astype(str) != ""]
    if m.empty:
        return pd.DataFrame()
    cols = list(prof.columns)
    A = m[cols].to_numpy(float)

    def loo(keys):
        g = m.groupby(keys)
        cnt = g[cols[0]].transform("count").to_numpy(float)
        pred = (g[cols].transform("sum").to_numpy(float) - A) / np.maximum(cnt - 1, 1)[:, None]
        pred[(cnt - 1) < MIN_PEERS] = np.nan
        return pred

    def miss(pred):
        d = np.abs(A - pred)
        return 0.5 * d.sum(axis=1) if share else d.mean(axis=1)

    r = pd.DataFrame({by: m[by].to_numpy(), "m_group": miss(loo([by])), "m_all": miss(loo(["_all"])),
                      "m_flat": miss(np.full_like(A, 1.0 / len(cols)) if share else np.zeros_like(A))})
    r = r.dropna()
    if r.empty:
        return pd.DataFrame()
    g = r.groupby(by)
    out = pd.DataFrame({"items": g.size(), "miss_group": g["m_group"].median(), "miss_all": g["m_all"].median(), "miss_flat": g["m_flat"].median()})
    for col, base in (("vs_flat", "miss_flat"), ("vs_all", "miss_all")):
        out[col] = (out[base] / out["miss_group"].clip(lower=1e-4)).clip(upper=99)
        out.loc[(out[base] < 1e-4) & (out["miss_group"] < 1e-4), col] = np.nan                 # nothing to explain: no verdict
    return out


def rhythm_verdict(items, vs_flat, vs_all) -> str:
    if not items or items < MIN_SCORED:
        return "too few to judge"
    if not vs_flat >= HELPS:
        return "no better than a flat day"
    return "group-specific rhythm" if vs_all >= HELPS else "same rhythm as every item"


def dow_profile(daily: pd.DataFrame, min_days: int = MIN_DAYS_DOW, min_hours: int = 18) -> pd.DataFrame:
    """Each item's share of sales by day of the week (Monday first, 7 numbers summing to 1) from the item_daily history. Needs min_days
    complete days, a median day of MIN_DAILY_SOLD units, and every weekday seen at least twice."""
    need = ["item_id", "day", "hours", "sold_day"]
    if daily is None or daily.empty or any(c not in daily.columns for c in need):
        return pd.DataFrame()
    d = daily[(daily["hours"] >= min_hours) & daily["sold_day"].notna()].copy()
    cnt = d.groupby("item_id").size()
    d = d[d["item_id"].isin(cnt[cnt >= min_days].index)]
    if d.empty:
        return pd.DataFrame()
    d = d[d["item_id"].map(d.groupby("item_id")["sold_day"].median()) >= MIN_DAILY_SOLD]
    d["dow"] = pd.to_datetime(d["day"]).dt.dayofweek
    g = d.groupby(["item_id", "dow"])["sold_day"]
    mean = g.mean().unstack("dow").reindex(columns=range(7))
    n = g.size().unstack("dow").reindex(columns=range(7))
    mean = mean[(n >= 2).all(axis=1)]
    mean = mean[mean.sum(axis=1) > 0]
    return mean.div(mean.sum(axis=1), axis=0)


def shape_report(groups: pd.DataFrame, hourly: pd.DataFrame, daily: pd.DataFrame = None) -> dict:
    """Rhythm results per group, family and overall: hour-of-day sales pattern (hour_*) and hour-of-day price pattern (price_*) from the
    hourly table, day-of-week sales pattern (dow_*) from the daily history."""
    prof = hourly_profiles(hourly) if hourly is not None and len(hourly) else {}
    dow = dow_profile(daily) if daily is not None and len(daily) else pd.DataFrame()
    if not prof and dow.empty:
        return {}
    out = {"items": prof.get("items", 0), "rows": prof.get("rows", 0), "dow_items": int(len(dow))}
    mats = ([("hour", prof["sold_hour"], True), ("price", prof["price_hour"], False)] if prof else []) + ([("dow", dow, True)] if not dow.empty else [])
    for level, by in (("groups", "group"), ("families", "family"), ("overall", "_all")):
        parts = []
        for tag, mat, share in mats:
            t = shape_by(groups, mat, by=by, share=share)
            if len(t):
                parts.append(t.add_prefix(f"{tag}_"))
        if not parts:
            continue
        df = pd.concat(parts, axis=1)
        df.index.name = by
        df = df.reset_index()
        lead = "hour" if "hour_items" in df.columns else "dow"
        if f"{lead}_items" in df.columns:
            if by == "_all":                                     # group and all-items are the same guess here, so only "better than a flat day" is meaningful
                df[f"{lead}_verdict"] = ["items share a common daily pattern" if v >= HELPS else "no common daily pattern" for v in df[f"{lead}_vs_flat"]]
            else:
                df[f"{lead}_verdict"] = [rhythm_verdict(n if n == n else 0, a, b) for n, a, b in zip(df[f"{lead}_items"], df[f"{lead}_vs_flat"], df[f"{lead}_vs_all"])]
            df = df.sort_values(f"{lead}_items", ascending=False)
        out[level] = df.reset_index(drop=True)
    return out


# ------------------------------------------------------------------ co-movement: do items in a group lift on the same days?
def daily_series(daily: pd.DataFrame, min_hours_per_day: int = 18, min_days: int = MIN_DAYS, min_daily_sold: float = MIN_DAILY_SOLD) -> dict:
    """Each item's day-by-day lift against ITS OWN typical day, so quantities drop out, from the item_daily history.
    sales: log10((units sold that day + 1) / (the item's median day + 1)); 0 = a normal day, 0.30 = double, -0.30 = half.
    price: log10(the day's typical price / the item's typical price).
    A day counts only if it has min_hours_per_day usable hours. Items need min_days such days and a median day of min_daily_sold units."""
    need = ["item_id", "day", "hours", "sold_day", "price_day"]
    if daily is None or daily.empty or any(c not in daily.columns for c in need):
        return {}
    d = daily[(daily["hours"] >= min_hours_per_day) & daily["sold_day"].notna() & (daily["price_day"].astype(float) > 0)].copy()
    cnt = d.groupby("item_id").size()
    d = d[d["item_id"].isin(cnt[cnt >= min_days].index)]
    if d.empty:
        return {}
    base = d.groupby("item_id")["sold_day"].median()
    d = d[d["item_id"].isin(base[base >= min_daily_sold].index)].copy()
    if d.empty:
        return {}
    d["sales_lift"] = np.log10((d["sold_day"].astype(float) + 1.0) / (d["item_id"].map(base) + 1.0))
    d["price_lift"] = np.log10(d["price_day"].astype(float) / d.groupby("item_id")["price_day"].transform("median").astype(float))
    return {"sales": d.pivot(index="item_id", columns="day", values="sales_lift"), "price": d.pivot(index="item_id", columns="day", values="price_lift"),
            "items": int(d["item_id"].nunique()), "days": int(d["day"].nunique())}


def _peer_mean(A: np.ndarray, V: np.ndarray, frame: pd.DataFrame, keys: list) -> np.ndarray:
    """Per item and day: the mean of the OTHER items' values that day (NaN with fewer than MIN_PEERS valid peers)."""
    cols = list(range(A.shape[1]))
    k = frame[keys].reset_index(drop=True)
    S = pd.concat([k, pd.DataFrame(np.where(V, A, 0.0), columns=cols)], axis=1).groupby(keys)[cols].transform("sum").to_numpy(float)
    C = pd.concat([k, pd.DataFrame(V.astype(float), columns=cols)], axis=1).groupby(keys)[cols].transform("sum").to_numpy(float)
    n = C - V
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (S - np.where(V, A, 0.0)) / n
    out[n < MIN_PEERS] = np.nan
    return out


def _outside_mean(A: np.ndarray, V: np.ndarray, frame: pd.DataFrame, keys: list) -> np.ndarray:
    """Per item and day: the mean over every item OUTSIDE the item's own slice (the rest of the market), so a strong group cannot inflate its own control."""
    cols = list(range(A.shape[1]))
    k = frame[keys].reset_index(drop=True)
    Av = np.where(V, A, 0.0)
    S = pd.concat([k, pd.DataFrame(Av, columns=cols)], axis=1).groupby(keys)[cols].transform("sum").to_numpy(float)
    C = pd.concat([k, pd.DataFrame(V.astype(float), columns=cols)], axis=1).groupby(keys)[cols].transform("sum").to_numpy(float)
    n = V.sum(axis=0).astype(float)[None, :] - C
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (Av.sum(axis=0)[None, :] - S) / n
    out[n < MIN_PEERS] = np.nan
    return out


def _rowcorr(X: np.ndarray, Y: np.ndarray, min_n: int = 10) -> np.ndarray:
    ok = ~np.isnan(X) & ~np.isnan(Y)
    n = ok.sum(axis=1)
    X0, Y0 = np.where(ok, X, 0.0), np.where(ok, Y, 0.0)
    nn = np.maximum(n, 1)
    dx = np.where(ok, X0 - (X0.sum(axis=1) / nn)[:, None], 0.0)
    dy = np.where(ok, Y0 - (Y0.sum(axis=1) / nn)[:, None], 0.0)
    vx, vy = (dx ** 2).sum(axis=1), (dy ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (dx * dy).sum(axis=1) / np.sqrt(vx * vy)
    r[(n < min_n) | (vx <= 0) | (vy <= 0)] = np.nan
    return r


def comove_by(groups: pd.DataFrame, mat: pd.DataFrame, by: str = "group", spikes: bool = False) -> pd.DataFrame:
    """Leave-one-out on day-by-day lifts. For each item, its lift series is compared with the average of the OTHER items in its group
    (corr_group) and with the average of every item OUTSIDE the group (corr_market). extra = how much more the group tracks the item than the rest of the market does.
    With spikes=True: on the days the item lifts (>= SPIKE times its own typical day), how often do its peers lift too, compared with
    how often they lift on any day (a ratio; 1.0 = no link), for the group (spike_group) and for the rest of the market (spike_market)."""
    if mat is None or mat.empty:
        return pd.DataFrame()
    key = groups.drop_duplicates("item_id").set_index("item_id").assign(_all="all")
    key = key[[c for c in dict.fromkeys([by, "family", "_all"]) if c in key.columns]]
    m = mat.join(key, how="inner")
    m = m[m[by].astype(str) != ""]
    if m.empty:
        return pd.DataFrame()
    A = m[list(mat.columns)].to_numpy(float)
    V = ~np.isnan(A)
    if not spikes:
        rg = _rowcorr(A, _peer_mean(A, V, m, [by]))
        ra = rg if by == "_all" else _rowcorr(A, _outside_mean(A, V, m, [by]))      # nothing lies outside "all items"
        r = pd.DataFrame({by: m[by].to_numpy(), "corr_group": rg, "corr_market": ra}).dropna()
        if r.empty:
            return pd.DataFrame()
        r["extra"] = r["corr_group"] - r["corr_market"]
        g = r.groupby(by)
        return pd.DataFrame({"items": g.size(), "corr_group": g["corr_group"].median(), "corr_market": g["corr_market"].median(), "extra": g["extra"].median()})
    L = np.where(V, A >= np.log10(SPIKE), False).astype(float)
    out = {}
    out["group"] = _peer_mean(L, V, m, [by])                          # share of the other items in the group lifting that day
    out["market"] = out["group"] if by == "_all" else _outside_mean(L, V, m, [by])   # share of the items outside the group lifting that day
    rows = []
    byv = m[by].to_numpy()
    for gval in pd.unique(byv):
        sel = byv == gval
        ev = (L[sel] == 1) & ~np.isnan(out["group"][sel]) & ~np.isnan(out["market"][sel])
        n_ev = int(ev.sum())
        res = dict(events=n_ev)
        for lab in ("group", "market"):
            rt = out[lab][sel]
            valid = V[sel] & ~np.isnan(rt)
            base = rt[valid].mean() if valid.any() else np.nan
            cond = rt[ev].mean() if n_ev else np.nan
            res[f"spike_{lab}"] = cond / base if base and base > 0 else np.nan
        res[by] = gval
        rows.append(res)
    return pd.DataFrame(rows).set_index(by)


def comove_verdict(items, extra, corr_group, corr_market) -> str:
    if not items or items < MIN_SCORED:
        return "too few to judge"
    if extra >= EXTRA_CORR and corr_group >= 0.25:
        return "moves together beyond the market"
    if corr_market >= 0.25 and extra < EXTRA_CORR:
        return "moves with the whole market"
    return "no shared movement"


def comove_report(groups: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Co-movement results per group, family and overall, for sales lifts (sales_*), price lifts (price_*) and same-day spikes (spike_*)."""
    ds = daily_series(daily)
    if not ds:
        return {}
    out = {"items": ds["items"], "days": ds["days"]}
    for level, by in (("groups", "group"), ("families", "family"), ("overall", "_all")):
        parts = []
        for tag, mat in (("sales", ds["sales"]), ("price", ds["price"])):
            t = comove_by(groups, mat, by=by)
            if len(t):
                parts.append(t.add_prefix(f"{tag}_"))
        sp = comove_by(groups, ds["sales"], by=by, spikes=True)
        if len(sp):
            parts.append(sp)
        if not parts:
            continue
        df = pd.concat(parts, axis=1)
        df.index.name = by
        df = df.reset_index()
        if "sales_items" in df.columns:
            if by == "_all":
                df["sales_verdict"] = ["items share market-wide lifts" if c >= 0.25 else "no market-wide lifts" for c in df["sales_corr_group"]]
            else:
                df["sales_verdict"] = [comove_verdict(n if n == n else 0, e, cg, ca) for n, e, cg, ca in
                                       zip(df["sales_items"], df["sales_extra"], df["sales_corr_group"], df["sales_corr_market"])]
            df = df.sort_values("sales_items", ascending=False)
        if "spike_events" in df.columns or "events" in df.columns:
            df = df.rename(columns={"events": "spike_events"})
            df["spike_verdict"] = [spike_verdict(e if e == e else 0, g, a) for e, g, a in zip(df["spike_events"], df.get("spike_group", np.nan), df.get("spike_market", np.nan))]
        out[level] = df.reset_index(drop=True)
    return out


def spike_verdict(events, spike_group, spike_market) -> str:
    if not events or events < 10:
        return "too few spike days"
    if spike_group >= 1.5 and spike_group >= 1.15 * (spike_market if spike_market == spike_market else 0):
        return "spikes together beyond the market"
    if spike_market >= 1.5:
        return "spikes with the whole market"
    return "no shared spikes"


def grouptest(groups: pd.DataFrame, items: pd.DataFrame, hourly: pd.DataFrame = None, daily: pd.DataFrame = None) -> dict:
    m = prepare(groups, items)
    if m.empty:
        return {}
    res = {"groups": loo_by(m, "group"), "families": loo_by(m, "family"), "uses": loo_by(m, "use"),
           **{f"tiers_{k}": v for k, v in tier_report(m).items()}, "coarse": coarse_report(m), "n_items": len(m)}
    res["rhythm"] = shape_report(groups, hourly, daily)
    res["comove"] = comove_report(groups, daily) if daily is not None and len(daily) else {}
    return res
