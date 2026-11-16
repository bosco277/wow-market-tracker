"""Buy-below prices: what is a reasonable minimum price for an item, how often is it actually reached, and is there a time when it tends to show up?

For gathering inventory (crafting inputs, stock you will use), not for reselling. From an item's own recent hourly prices the engine sets three targets:
  good  = the bottom 25% of its hourly prices over the previous 7 days,
  great = the bottom 10%,
  steal = the bottom 3% (about the lowest price it touches in a week, without trusting a single freak hour).
Setting a target proves nothing, so the engine CALIBRATES it the only honest way: at the start of every US Eastern day it sets the targets from the 7 days BEFORE that day, then checks
whether any hourly snapshot of the next 1, 3 and 7 days was at or under the target, how many hours later the first one came, how many units were listed near the floor at that moment (u5),
and how much cheaper that first hit was than the average price over the same window. Results are pooled by how deep the target sits under the item's 7-day median (under 3%, 3-10%,
10-20%, 20%+) and by price tier, because "10% under the median" behaves alike across items while individual items have too little history.

WHEN: for every hourly snapshot the engine asks whether the price was at or under that day's target, by US Eastern hour (only items whose target is at least 3% under their median, so a
flat item cannot hit all day). The hit rate by hour is shown as a multiple of the average (1.0 = no pattern), and the best 3-hour window is learned on the first half of the days and
graded on the second half, so a pattern that does not repeat shows as about 1.0. Weekdays need about 6 weeks of days.

The tool also writes a shopping list (buy_targets.csv): for the next day, each item's three targets, how deep each is, the calibrated chance of reaching it within 1 / 3 / 7 days, and the
units usually near the floor when it was that low. Then the in-game scan decides whether the price is actually on the market.

Prices are the floor (cheapest listing) or the depth price p15, as chosen for the whole report. Nothing here reads or writes the network.
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import windows as W

TIERS = {"good": 0.25, "great": 0.10, "steal": 0.03}
DEPTH_EDGES = [0.0, 0.03, 0.10, 0.20, 10.0]
DEPTH_NAMES = ["under 3%", "3-10%", "10-20%", "20%+"]


@dataclass
class TargetConfig:
    lookback_days: int = 7
    tiers: dict = field(default_factory=lambda: dict(TIERS))
    horizons: tuple = (1, 3, 7)         # days ahead a target is checked over
    window_hours: int = 3               # length of the best-window test for the hit hours
    min_items: int = 3
    min_depth: float = 0.03             # the hit-hour test only counts items whose target is at least this far under their median
    min_pool_days: int = 8              # days needed before the halves of the hit-hour test mean anything
    weekday_days: int = 42              # days needed before the weekday split is shown
    tz: str = W.TZ
    tier_edges: tuple = (1.0, 100.0)


def depth_name(depth: float) -> str:
    for lo, hi, name in zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:], DEPTH_NAMES):
        if lo <= depth < hi:
            return name
    return DEPTH_NAMES[0]


# ------------------------------------------------------------------ the grid

def _wide(u: pd.DataFrame, col: str, index=None, columns=None) -> pd.DataFrame:
    w = u.pivot_table(index="t", columns="item_id", values=col, aggfunc="mean", dropna=False)
    if index is None:
        index = pd.date_range(w.index.min(), w.index.max(), freq="h")
    return w.reindex(index=index, columns=columns if columns is not None else w.columns)


def build_grid(p: pd.DataFrame, cfg: Optional[TargetConfig] = None) -> Optional[dict]:
    """Hourly price and u5 matrices (hours x items, NaN where unusable) with each tier's trailing target and the trailing median at every hour (looking only at the previous
    `lookback_days`), plus the position of every US Eastern midnight. None if there is nothing to work with."""
    cfg = cfg or TargetConfig()
    u = p[p["usable"]]
    if u.empty:
        return None
    price = _wide(u, "price")
    u5 = _wide(u, "u5", price.index, price.columns)
    L = cfg.lookback_days * 24
    mp = int(L * 0.75)
    prev = price.shift(1)
    thr = {name: prev.rolling(L, min_periods=mp).quantile(q).to_numpy() for name, q in cfg.tiers.items()}
    med = prev.rolling(L, min_periods=mp).median().to_numpy()
    starts = []
    for d in sorted(p["day"].unique()):
        t = pd.Timestamp(d).tz_localize(cfg.tz).tz_convert("UTC").tz_localize(None)
        pos = price.index.get_indexer([t])[0]
        if pos >= 0:
            starts.append((pd.Timestamp(d), int(pos)))
    return {"index": price.index, "items": price.columns.to_numpy(), "P": price.to_numpy(), "U": u5.to_numpy(), "THR": thr, "MED": med, "starts": starts, "L": L}


# ------------------------------------------------------------------ calibration

def reach_events(grid: Optional[dict], cfg: Optional[TargetConfig] = None) -> pd.DataFrame:
    """One row per item, US Eastern day, tier and horizon with a full forward window: reached (any hourly snapshot at or under the target), first_hour (hours after midnight of
    the first one), price_first, u5_first (units near the floor then), mean_window (average usable price over the window), thr, med."""
    cfg = cfg or TargetConfig()
    cols = ["item_id", "day", "tier", "H", "reached", "first_hour", "price_first", "u5_first", "mean_window", "thr", "med"]
    if grid is None:
        return pd.DataFrame(columns=cols)
    P, U, MED, items = grid["P"], grid["U"], grid["MED"], grid["items"]
    n_rows = P.shape[0]
    parts = []
    for day, i0 in grid["starts"]:
        for H in cfg.horizons:
            end = i0 + 24 * H
            if end > n_rows:
                continue
            Wp, Wu = P[i0:end], U[i0:end]
            cnt = np.sum(~np.isnan(Wp), axis=0)
            mean_w = np.nansum(Wp, axis=0) / np.maximum(cnt, 1)
            ok = (cnt >= 0.75 * 24 * H) & ~np.isnan(MED[i0])
            for name in cfg.tiers:
                T = grid["THR"][name][i0]
                idx = np.where(ok & ~np.isnan(T))[0]
                if not len(idx):
                    continue
                with np.errstate(invalid="ignore"):
                    cond = Wp[:, idx] <= T[idx][None, :]
                reached = cond.any(axis=0)
                first = cond.argmax(axis=0)
                parts.append(pd.DataFrame({"item_id": items[idx], "day": day, "tier": name, "H": H, "reached": reached, "first_hour": np.where(reached, first, np.nan),
                                           "price_first": np.where(reached, Wp[first, idx], np.nan), "u5_first": np.where(reached, Wu[first, idx], np.nan),
                                           "mean_window": mean_w[idx], "thr": T[idx], "med": MED[i0][idx]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def calibration(events: pd.DataFrame, pools: pd.DataFrame, cfg: Optional[TargetConfig] = None) -> pd.DataFrame:
    """Per broad pool (everything, price tiers), target tier, depth of the target under the 7-day median and horizon: item-days, the share that reached it, the median wait in hours
    (among those that did), the median units near the floor at the first hit, the median saving of the first hit against the window's average price, and the median depth."""
    cols = ["pool", "tier", "depth", "H", "n", "reached", "wait_med", "units_med", "saving_med", "depth_med"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    broad = pools[(pools["pool"] == "everything") | pools["pool"].str.startswith("price")]
    e = events.merge(broad, on="item_id")
    e["dd"] = (1.0 - e["thr"] / e["med"]).clip(lower=0.0)
    e["depth"] = pd.cut(e["dd"], bins=DEPTH_EDGES, labels=DEPTH_NAMES, right=False)
    e["saving"] = 1.0 - e["price_first"] / e["mean_window"]
    out = []
    for (pool, tier, depth, H), d in e.groupby(["pool", "tier", "depth", "H"], observed=True):
        hit = d[d["reached"]]
        out.append({"pool": pool, "tier": tier, "depth": depth, "H": H, "n": len(d), "reached": float(d["reached"].mean()),
                    "wait_med": float(hit["first_hour"].median()) if len(hit) else np.nan, "units_med": float(hit["u5_first"].median()) if len(hit) else np.nan,
                    "saving_med": float(hit["saving"].median()) if len(hit) else np.nan, "depth_med": float(d["dd"].median())})
    return pd.DataFrame(out, columns=cols)


# ------------------------------------------------------------------ when

def hit_hours(grid: Optional[dict], pools: pd.DataFrame, cfg: Optional[TargetConfig] = None) -> pd.DataFrame:
    """Per broad pool, tier, US Eastern day and hour of day: hits (item-hours at or under that day's target) and obs (item-hours observed), counting only items whose target is at least
    `min_depth` under their median."""
    cfg = cfg or TargetConfig()
    cols = ["pool", "tier", "day", "hour", "hits", "obs"]
    if grid is None:
        return pd.DataFrame(columns=cols)
    P, MED, items = grid["P"], grid["MED"], grid["items"]
    broad = pools[(pools["pool"] == "everything") | pools["pool"].str.startswith("price")]
    masks = {pool: np.isin(items, d["item_id"].to_numpy()) for pool, d in broad.groupby("pool")}
    rows = []
    for day, i0 in grid["starts"]:
        end = i0 + 24
        if end > P.shape[0]:
            continue
        Wp = P[i0:end]
        for name in cfg.tiers:
            T = grid["THR"][name][i0]
            with np.errstate(invalid="ignore"):
                deep = (1.0 - T / MED[i0]) >= cfg.min_depth
                valid = deep[None, :] & ~np.isnan(Wp)
                cond = (Wp <= T[None, :]) & valid
            for pool, m in masks.items():
                if m.sum() < cfg.min_items:
                    continue
                hits, obs = cond[:, m].sum(axis=1), valid[:, m].sum(axis=1)
                for h in range(24):
                    rows.append((pool, name, day, h, int(hits[h]), int(obs[h])))
    return pd.DataFrame(rows, columns=cols)


def _lifts(d: pd.DataFrame) -> np.ndarray:
    """Hit rate of each hour as a multiple of the overall hit rate (24 values; NaN where nothing was observed)."""
    g = d.groupby("hour")[["hits", "obs"]].sum().reindex(range(24))
    rate = g["hits"] / g["obs"].replace(0, np.nan)
    total = g["hits"].sum() / max(g["obs"].sum(), 1)
    return (rate / total).to_numpy() if total > 0 else np.full(24, np.nan)


def when_summary(hh: pd.DataFrame, cfg: Optional[TargetConfig] = None) -> pd.DataFrame:
    """Per broad pool and tier: the hit-rate multiple by hour over all days, the best `window_hours` window learned on the FIRST half of the days, its multiple on the SECOND half
    (1.0 = no gain; this is the number to trust), the same window's multiple on all days, and the correlation of the 24 hourly multiples between the two halves."""
    cfg = cfg or TargetConfig()
    cols = ["pool", "tier", "days", "lifts", "window_start", "lift_later", "lift_all", "replication"]
    if hh.empty:
        return pd.DataFrame(columns=cols)
    out = []
    w = cfg.window_hours
    for (pool, tier), d in hh.groupby(["pool", "tier"]):
        days = sorted(d["day"].unique())
        lifts_all = _lifts(d)
        row = {"pool": pool, "tier": tier, "days": len(days), "lifts": lifts_all, "window_start": np.nan, "lift_later": np.nan, "lift_all": np.nan, "replication": np.nan}
        if len(days) >= cfg.min_pool_days:
            half = len(days) // 2
            a, b = d[d["day"].isin(days[:half])], d[d["day"].isin(days[half:])]
            la, lb = _lifts(a), _lifts(b)
            best, best_s = -np.inf, -1
            for s in range(0, 24 - w + 1):
                seg = la[s:s + w]
                if np.isnan(seg).any():
                    continue
                if seg.mean() > best:
                    best, best_s = seg.mean(), s
            if best_s >= 0:
                row["window_start"] = best_s
                row["lift_later"] = float(np.nanmean(lb[best_s:best_s + w]))
                row["lift_all"] = float(np.nanmean(lifts_all[best_s:best_s + w]))
            ok = ~np.isnan(la) & ~np.isnan(lb)
            if ok.sum() >= 12 and np.std(la[ok]) > 0 and np.std(lb[ok]) > 0:
                row["replication"] = float(np.corrcoef(la[ok], lb[ok])[0, 1])
        out.append(row)
    return pd.DataFrame(out, columns=cols)


def weekday_summary(hh: pd.DataFrame, cfg: Optional[TargetConfig] = None) -> Optional[pd.DataFrame]:
    """Per pool (tier 'great'): the hit rate of each weekday as a multiple of the average, once there are `weekday_days` days. None before that."""
    cfg = cfg or TargetConfig()
    if hh.empty or hh["day"].nunique() < cfg.weekday_days:
        return None
    d = hh[hh["tier"] == "great"].groupby(["pool", "day"])[["hits", "obs"]].sum().reset_index()
    d["rate"] = d["hits"] / d["obs"].replace(0, np.nan)
    d["dow"] = pd.to_datetime(d["day"]).dt.dayofweek
    g = d.groupby(["pool", "dow"])["rate"].mean().unstack("dow")
    return (g.div(g.mean(axis=1), axis=0)).rename(columns=dict(enumerate(W.WEEKDAYS)))


# ------------------------------------------------------------------ the shopping list

def current_targets(grid: Optional[dict], pools: pd.DataFrame, calib: pd.DataFrame, groups: Optional[pd.DataFrame] = None, cfg: Optional[TargetConfig] = None) -> pd.DataFrame:
    """One row per item with enough history: targets for the NEXT day from the last `lookback_days` of hourly prices (the same rule the calibration graded), their depth under the
    median, the calibrated chance of reaching each within 1 / 3 / 7 days (from the item's price tier and the target's depth), and the units usually near the floor when the price was that low."""
    cfg = cfg or TargetConfig()
    if grid is None:
        return pd.DataFrame()
    P, U, items, L = grid["P"], grid["U"], grid["items"], grid["L"]
    if P.shape[0] < L:
        return pd.DataFrame()
    Wp, Wu = P[-L:], U[-L:]
    cnt = np.sum(~np.isnan(Wp), axis=0)
    ok = cnt >= int(L * 0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Wp, axis=0)
        last = np.array([Wp[~np.isnan(Wp[:, j]), j][-1] if ok[j] else np.nan for j in range(Wp.shape[1])])
        tcols = {}
        for name, q in cfg.tiers.items():
            T = np.nanquantile(Wp, q, axis=0)
            with np.errstate(invalid="ignore"):
                near = np.where(Wp <= T[None, :], Wu, np.nan)
            tcols[name] = (T, np.nanmedian(near, axis=0))
    tier_of = pools[pools["pool"].str.startswith("price")].drop_duplicates("item_id").set_index("item_id")["pool"]
    lookup = {(r.pool, r.tier, r.depth, r.H): r.reached for r in calib.itertuples()} if len(calib) else {}
    rows = []
    for j in np.where(ok)[0]:
        iid = int(items[j])
        pool = tier_of.get(iid, W.tier_label(med[j], cfg.tier_edges))
        row = {"item_id": iid, "price_tier": pool, "med7_g": med[j], "last_g": last[j]}
        for name in cfg.tiers:
            T, units = tcols[name]
            dd = max(0.0, 1.0 - T[j] / med[j]) if med[j] > 0 else 0.0
            dn = depth_name(dd)
            row[f"{name}_g"], row[f"{name}_copper"], row[f"{name}_depth"], row[f"{name}_depth_band"] = T[j], int(round(T[j] * 10000)), dd, dn
            for H in cfg.horizons:
                row[f"{name}_p{H}d"] = lookup.get((pool, name, dn, H), np.nan)
            row[f"{name}_units"] = units[j]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score_g"] = out["great_units"].fillna(0) * (out["med7_g"] - out["great_g"])
    if groups is not None and len(groups):
        keep = [c for c in ("item_id", "name", "group") if c in groups.columns]
        out = out.merge(groups.drop_duplicates("item_id")[keep], on="item_id", how="left")
    return out.sort_values("score_g", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ everything together

def analyze_targets(hl: pd.DataFrame, groups: Optional[pd.DataFrame] = None, wcfg: Optional[W.WindowConfig] = None, cfg: Optional[TargetConfig] = None) -> dict:
    """Calibration, when-tables and the shopping list for an hourly table (HOURLY_COLUMNS)."""
    wcfg = wcfg or W.WindowConfig()
    cfg = cfg or TargetConfig(tier_edges=wcfg.tiers, min_items=wcfg.min_items)
    p = W.prepare(hl, wcfg)
    res = {"cfg": cfg, "calibration": pd.DataFrame(), "when": pd.DataFrame(), "weekday": None, "list": pd.DataFrame(), "days": 0, "items": 0, "first_day": None, "last_day": None, "last_hour": None}
    if p.empty or not p["usable"].any():
        return res
    pools = W.label_items(p, groups, wcfg)
    grid = build_grid(p, cfg)
    if grid is None:
        return res
    events = reach_events(grid, cfg)
    calib = calibration(events, pools, cfg)
    hh = hit_hours(grid, pools, cfg)
    res.update(days=int(p["day"].nunique()), items=int(p["item_id"].nunique()), first_day=p["day"].min(), last_day=p["day"].max(), last_hour=p["t"].max(),
               calibration=calib, when=when_summary(hh, cfg), weekday=weekday_summary(hh, cfg), list=current_targets(grid, pools, calib, groups, cfg))
    return res


# ------------------------------------------------------------------ what the app reads

TABLES = ("buy_targets", "buy_calibration", "buy_when", "buy_meta")


def export_tables(res: dict, built_at=None) -> dict:
    """The four small tables the front end reads from engine/ in the private dataset (all plain columns, no lists or categories):
    buy_targets (the shopping list), buy_calibration (how often each depth of target was reached), buy_when (hit-rate multiples by hour of day, one column per hour) and
    buy_meta (one row: when this was built, which days of data it saw). Empty frames when there was nothing to compute."""
    cfg = res["cfg"]
    lst = res["list"].copy()
    cal = res["calibration"].copy()
    if len(cal):
        cal["depth"] = cal["depth"].astype(str)
    when = res["when"].copy()
    if len(when):
        lifts = pd.DataFrame(np.vstack(when["lifts"].to_list()), columns=[f"lift_{h:02d}" for h in range(24)], index=when.index)
        when = pd.concat([when.drop(columns=["lifts"]), lifts], axis=1)
    built = pd.Timestamp(built_at) if built_at is not None else pd.Timestamp.now("UTC").tz_localize(None)
    meta = pd.DataFrame([{"built_at": built, "first_day": str(res["first_day"])[:10] if res["first_day"] is not None else "", "last_day": str(res["last_day"])[:10] if res["last_day"] is not None else "",
                          "last_hour": res["last_hour"] if res["last_hour"] is not None else pd.NaT, "days": int(res["days"]), "items": int(res["items"]), "listed": int(len(lst)),
                          "lookback_days": int(cfg.lookback_days), "window_hours": int(cfg.window_hours)}])
    return {"buy_targets": lst, "buy_calibration": cal, "buy_when": when, "buy_meta": meta}


def format_targets(res: dict, top: int = 30) -> str:
    cfg, calib, when, lst = res["cfg"], res["calibration"], res["when"], res["list"]
    pct = lambda x: "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:5.0f}%"
    num = lambda x, f="{:,.0f}": "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)
    L = [f"== BUY-BELOW PRICES: targets from the item's own last {cfg.lookback_days} days of hourly prices. good = bottom {cfg.tiers['good'] * 100:.0f}% of its hours, great = bottom {cfg.tiers['great'] * 100:.0f}%, "
         f"steal = bottom {cfg.tiers['steal'] * 100:.0f}% (about the lowest price it touches in a week)."]
    if calib.empty:
        L.append(f"   Not enough history yet: the targets need {cfg.lookback_days} days of hourly prices, then at least one day after that to check them on.")
        return "\n".join(L)
    L.append("   CALIBRATION, walk-forward: at midnight of each US Eastern day the targets are set from the previous days only, then the next 1 / 3 / 7 days are checked. 'Reached' = at least one hourly snapshot")
    L.append("   at or under the target. Wait = median hours until the first such snapshot. Units = median units listed within 5% of the floor at that moment. Saving = how much cheaper that first snapshot was than the average price over")
    L.append("   the same days. Depth = how far under the item's 7-day median the target sits. A shallow target is reached almost always and saves almost nothing; a deep one is the real bargain and is reached less often.")
    L.append(f"   {'pool':20s} {'target':>6s} {'depth':>9s} {'item-days':>9s} | {'reached in 1d':>13s} {'3d':>6s} {'7d':>6s} | {'wait h (1d/3d)':>14s} | {'units then':>10s} | {'saving (3d)':>11s}")
    for (pool, tier, depth), d in calib.groupby(["pool", "tier", "depth"], observed=True, sort=False):
        d = d.set_index("H")
        g = lambda col, H: d[col].get(H, np.nan)
        L.append(f"   {pool[:20]:20s} {tier:>6s} {str(depth):>9s} {int(d['n'].max()):9d} | {pct(g('reached', 1)):>13s} {pct(g('reached', 3)):>6s} {pct(g('reached', 7)):>6s} | "
                 f"{num(g('wait_med', 1)):>6s} /{num(g('wait_med', 3)):>6s} | {num(g('units_med', 3)):>10s} | {pct(g('saving_med', 3)):>11s}")
    L.append("")
    L.append(f"== WHEN IS THE PRICE AT OR UNDER THE TARGET? Hit rate by US Eastern hour as a multiple of the average (1.0 = no pattern); only items whose target is at least {cfg.min_depth * 100:.0f}% under their median.")
    if when.empty:
        L.append("   No data yet.")
    else:
        for _, r in when[(when["pool"] == "everything") & (when["tier"] == "great")].iterrows():
            cells = [f"{h:02d}h {v:4.2f}" for h, v in enumerate(r["lifts"])]
            for i in range(0, 24, 8):
                L.append("   " + "   ".join(cells[i:i + 8]))
        L.append(f"   Best {cfg.window_hours}-hour window learned on the FIRST half of the days, graded on the SECOND half (1.00 = no gain; this is the number to trust). 'Replication' = correlation of the 24 hourly multiples between the halves (about 0 = noise, near 1 = a repeating rhythm).")
        L.append(f"   {'pool':20s} {'target':>6s} {'days':>5s} {'window (ET)':>12s} {'later days':>10s} {'all days':>8s} {'replication':>11s}")
        for _, r in when.sort_values(["pool", "tier"]).iterrows():
            ws = "n/a" if np.isnan(r["window_start"]) else f"{int(r['window_start']):02d}-{int(r['window_start']) + cfg.window_hours:02d}h"
            L.append(f"   {r['pool'][:20]:20s} {r['tier']:>6s} {int(r['days']):5d} {ws:>12s} {num(r['lift_later'], '{:.2f}'):>10s} {num(r['lift_all'], '{:.2f}'):>8s} {num(r['replication'], '{:+.2f}'):>11s}")
        if res.get("weekday") is None:
            L.append(f"   Weekdays: shown once there are {cfg.weekday_days} days of data (about 6 weeks); until then treat the weekday as unknown.")
        else:
            wd = res["weekday"]
            for pool, r in wd.iterrows():
                L.append(f"   Weekday hit rate, {pool}: " + "  ".join(f"{k} {v:.2f}" for k, v in r.items()))
    L.append("")
    L.append(f"== SHOPPING LIST (top {top} by units x discount at the 'great' target; the full list for every item is buy_targets.csv). Targets are for the next day, from the last {cfg.lookback_days} days.")
    if lst.empty:
        L.append("   No items with enough history.")
    else:
        L.append(f"   {'item':34s} {'group':22s} {'7d median':>10s} {'great target':>12s} {'depth':>6s} {'reach 1d/3d/7d':>16s} {'units':>6s} {'worth g':>9s} | {'good target':>11s} {'steal target':>12s}")
        name = lst["name"] if "name" in lst.columns else lst["item_id"].astype(str)
        grp = lst["group"] if "group" in lst.columns else pd.Series([""] * len(lst))
        for i in range(min(top, len(lst))):
            r = lst.iloc[i]
            L.append(f"   {str(name.iloc[i])[:34]:34s} {str(grp.iloc[i])[:22]:22s} {r['med7_g']:10.2f} {r['great_g']:12.2f} {r['great_depth'] * 100:5.1f}% "
                     f"{pct(r['great_p1d'])}{pct(r['great_p3d'])}{pct(r['great_p7d'])} {num(r['great_units']):>6s} {r['score_g']:9,.0f} | {r['good_g']:11.2f} {r['steal_g']:12.2f}")
        L.append("   Prices in gold. 'reach' = how often a target this deep (for an item in this price tier) was reached within 1 / 3 / 7 days when it was graded; 'units' = typical units near the floor when it was that low;")
        L.append("   'worth' = units x (7-day median - great target), the gold you save against paying the median for those units. Scan in game: the list says what is worth looking for, the scan says whether it is there.")
    return "\n".join(L)
