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
SPLITS = ("id_band", "quality", "tiered", "use")


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


def coarse_report(m: pd.DataFrame, min_items: int = 20, spread_over: float = 10.0, splits=SPLITS) -> pd.DataFrame:
    """For the groups that are big but whose items behave very differently (wide spread), which extra split (item-id band, quality,
    has-tiers, name use) would make the peers more alike? Reported as the drop in the leave-one-out miss on sales per day."""
    cols = ["group", "items", "spread", "best_split", "off_group", "off_with_split", "gain", "slices"]
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
        best = None
        for s in splits:
            if part[s].nunique() < 2:
                continue
            p = _loo_median(part, ["group", s], "log_sales")
            p = np.where(np.isnan(p), base, p)                      # a slice too small to guess from falls back to the whole group
            e = np.abs(actual - p)
            ok = ~np.isnan(e0) & ~np.isnan(e)
            if not ok.any():
                continue
            gain = np.median(e0[ok]) / max(np.median(e[ok]), 0.005)
            if best is None or gain > best[1]:
                best = (s, gain, 10 ** np.median(e0[ok]), 10 ** np.median(e[ok]))
        if best is None:
            continue
        agg = part.groupby(best[0])["daily_est"].agg(["median", "count"])
        if best[0] == "id_band":
            agg = agg.loc[sorted(agg.index, key=lambda k: int(str(k)[:-1]))]                  # oldest ids first
        else:
            agg = agg.sort_values("count", ascending=False)
        agg = agg.head(6)
        slices = " | ".join(f"{k}: {r['median']:,.0f}/day (n={int(r['count'])})" for k, r in agg.iterrows())
        rows.append(dict(group=gname, items=int(sp.loc[gname, "items"]), spread=sp.loc[gname, "spread"], best_split=best[0],
                         off_group=best[2], off_with_split=best[3], gain=min(best[1], 99.0), slices=slices))
    return pd.DataFrame(rows, columns=cols)


def grouptest(groups: pd.DataFrame, items: pd.DataFrame) -> dict:
    m = prepare(groups, items)
    if m.empty:
        return {}
    return {"groups": loo_by(m, "group"), "families": loo_by(m, "family"), "uses": loo_by(m, "use"),
            **{f"tiers_{k}": v for k, v in tier_report(m).items()}, "coarse": coarse_report(m), "n_items": len(m)}
