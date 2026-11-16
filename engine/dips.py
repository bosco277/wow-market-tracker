"""Dip test: when an item's price falls into the bottom of its OWN recent range, does buying then beat buying at a random hour of the same days, and what does it sell for later?

This is the question the clock test (engine/windows.py) pointed at: a learned hour of day is worth 1-3% at best, but a resale needs about 10.5% between buying and selling
(5% profit left after a 5% AH cut), so the edge, if there is one, has to come from buying when the item is cheap against itself.

The rule is causal, so the test needs no hold-out: at each hour the trigger line is the bottom `quantile` (default a fifth) of the item's usable hourly prices over the
PREVIOUS `lookback_days` (default 7), and the item is "in a dip" when its price is at or under that line AND at least `min_discount` (default 3%) under the item's median over the
same days (the second condition stops a flat item with tied prices from "dipping" all day). The buy is the FIRST dip hour of each US Eastern day. For each such item-day the test
compares, with the same sale price k days later (the item's median price that day, minus the AH cut):
  * buying at the dip hour,
  * buying at the average hour of the same item-day (a random-hour buy: it isolates the timing),
  * buying at the average hour of every item-day, dip or not (the unconditional market).
It also reports how many units were listed near the floor at the dip hour (u5: within 5% of the cheapest listing), which is how many a plan could actually take. The edge is the
extra margin of the dip buy over the random-hour buy (the median over a day's buys, then the average over days), with a 90% range from a block bootstrap over days.

Three questions about whether a dip can be ACTED on (v0.7.39):
  * Can you still buy it? Arriving 1, 2 or 3 hours after the first dip hour, is the price still within `arrive_tol` (5%) of the dip price, how many units are near the floor then,
    and what is the margin buying at THAT hour's price?
  * Can you sell it? Units near the floor against how many the item sells a day (its median `sold_day` over the previous 7 days from item_daily): days to sell them, and a typical
    plan (at most `plan_days` days of sales) with its cost and expected profit in gold.
  * How often does it happen? Events per day across the pool.
A resale "loss" means the price fell by `min_profit` or more (a margin of -(cut + min_profit) or worse); a flat price lands exactly at the cut and counts as neither gain nor loss.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import windows as W


@dataclass
class DipConfig:
    lookback_days: int = 7
    quantile: float = 0.20
    min_discount: float = 0.03
    cut: float = 0.05
    min_profit: float = 0.05
    sale_days: tuple = (1, 2, 3)
    min_hours_day: int = 18
    min_items: int = 3
    boot: int = 400
    block: int = 3
    seed: int = 7
    min_days: int = 10                  # graded days needed for a verdict
    min_edge: float = 0.01              # an edge under one point of margin is not worth acting on
    lags: tuple = (1, 2, 3)             # hours after the first dip hour at which the buyer arrives
    arrive_tol: float = 0.05            # "still there": the price is within this much above the dip price
    plan_days: int = 2                  # a plan buys at most this many days of the item's normal sales

    @property
    def loss_margin(self) -> float:
        return -(self.cut + self.min_profit)


def _wide(u: pd.DataFrame, col: str) -> pd.DataFrame:
    w = u.pivot_table(index="t", columns="item_id", values=col, aggfunc="mean")
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))


def trigger_lines(u: pd.DataFrame, cfg: DipConfig) -> pd.DataFrame:
    """Per item and hour: item_id, t, thr (bottom-`quantile` price of the previous lookback), med (median of the same hours). Needs 75% of the lookback hours."""
    L = int(cfg.lookback_days * 24)
    prev = _wide(u, "price").shift(1)
    mp = int(L * 0.75)
    thr = prev.rolling(L, min_periods=mp).quantile(cfg.quantile)
    med = prev.rolling(L, min_periods=mp).median()
    for x in (thr, med):
        x.index.name, x.columns.name = "t", "item_id"
    a = thr.stack().rename("thr")
    b = med.stack().rename("med")
    out = pd.concat([a[a.notna()], b[b.notna()]], axis=1).dropna().reset_index()
    return out


def find_events(p: pd.DataFrame, cfg: Optional[DipConfig] = None, lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per item and US Eastern day that had a dip: item_id, day, t, price (at the first dip hour), thr, med, u5, b_rnd (mean usable price that day)."""
    cfg = cfg or DipConfig()
    u = p[p["usable"]]
    cols = ["item_id", "day", "t", "price", "thr", "med", "u5", "b_rnd"]
    if u.empty:
        return pd.DataFrame(columns=cols)
    lines = trigger_lines(u, cfg) if lines is None else lines
    g = u.groupby(["item_id", "day"])["price"]
    v = u.assign(n=g.transform("size"), b_rnd=g.transform("mean"))
    v = v[v["n"] >= cfg.min_hours_day].merge(lines, on=["item_id", "t"], how="inner")
    dip = v[(v["price"] <= v["thr"]) & (v["price"] <= v["med"] * (1.0 - cfg.min_discount))]
    first = dip.sort_values("t").groupby(["item_id", "day"], as_index=False).first()
    look = u[["item_id", "t", "price", "u5"]].drop_duplicates(["item_id", "t"])
    for h in cfg.lags:
        lh = look.assign(t=look["t"] - pd.Timedelta(hours=h)).rename(columns={"price": f"lagp{h}", "u5": f"lagu{h}"})
        first = first.merge(lh, on=["item_id", "t"], how="left")
    return first[cols + [f"lag{c}{h}" for h in cfg.lags for c in "pu"]].reset_index(drop=True)


def graded_days(p: pd.DataFrame, cfg: DipConfig, lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every item-day that could have had a dip (has trigger lines and enough hours): item_id, day, b_rnd = its mean usable price. The unconditional baseline."""
    u = p[p["usable"]]
    if u.empty:
        return pd.DataFrame(columns=["item_id", "day", "b_rnd"])
    lines = trigger_lines(u, cfg) if lines is None else lines
    v = u.assign(n=u.groupby(["item_id", "day"])["price"].transform("size"))
    v = v[v["n"] >= cfg.min_hours_day].merge(lines[["item_id", "t"]], on=["item_id", "t"], how="inner")
    return v.groupby(["item_id", "day"], as_index=False)["price"].mean().rename(columns={"price": "b_rnd"})


def sold_baseline(sales: Optional[pd.DataFrame], window: int = 7, min_periods: int = 4) -> pd.DataFrame:
    """item_id, day (midnight), sold7: the median of an item's `sold_day` over the previous `window` days (the day itself excluded), from the item_daily table."""
    cols = ["item_id", "day", "sold7"]
    if sales is None or sales.empty or not {"item_id", "day", "sold_day"} <= set(sales.columns):
        return pd.DataFrame(columns=cols)
    s = sales[["item_id", "day", "sold_day"]].copy()
    s["day"] = pd.to_datetime(s["day"])
    wide = s.pivot_table(index="day", columns="item_id", values="sold_day", aggfunc="mean")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    base = wide.shift(1).rolling(window, min_periods=min_periods).median()
    base.index.name, base.columns.name = "day", "item_id"
    long = base.stack()
    return long[long.notna()].rename("sold7").reset_index()[cols]


def resale_verdict(median_margin: float, gain: float, loss: float, cfg: DipConfig) -> str:
    """Does buying these dips and selling k days later pay? The TYPICAL buy (median) has to clear the profit bar after the cut, and winners have to outnumber losers."""
    if np.isnan(median_margin):
        return "no data"
    if median_margin >= cfg.min_profit and gain > loss:
        return f"resale clears {cfg.min_profit * 100:.0f}%"
    if median_margin > 0:
        return f"resale under the {cfg.min_profit * 100:.0f}% bar"
    return "resale loses money"


def dip_tables(p: pd.DataFrame, pools: pd.DataFrame, daily: pd.DataFrame, cfg: Optional[DipConfig] = None, sold7: Optional[pd.DataFrame] = None):
    """(table, depth, lag).
    `table`: per pool and sale delay k: dip-days, share of item-days with a dip, units near the floor at the dip (median, 25th percentile), median discount against the 7-day median,
    the share of buys that gain (`min_profit` or more left after the cut) and that lose (the price fell `min_profit` or more) for the dip, a random hour of the same days and every item-day,
    median margins, the edge of the dip over the random hour with its 90% range and verdict, and the resale verdict.
    `depth`: for the broad pools, the same buys split by how deep the dip was (3-6%, 6-10%, 10-20%, 20%+ under the item's median): events, events per day, units near the floor, gain / loss,
    median margin, days to sell the units near the floor at the item's normal sales rate, and a typical plan (units, cost, expected profit in gold).
    `lag`: for the broad pools and each depth, arriving h hours after the first dip hour: share still within `arrive_tol` of the dip price, units near the floor then, median margin buying at that
    hour's price, gain share."""
    cfg = cfg or DipConfig()
    dcols = ["pool", "k", "depth", "rank", "events", "per_day", "units_med", "gain", "loss", "median", "sell_known", "days_to_sell", "plan_qty", "plan_cost", "plan_profit"]
    lcols = ["pool", "k", "depth", "rank", "lag", "events", "still", "units_med", "median", "gain"]
    cols = ["pool", "k", "dip_days", "dip_share", "units_med", "units_p25", "discount_med", "gain_dip", "gain_random", "gain_all", "loss_dip", "loss_random", "loss_all",
            "median_dip", "median_random", "edge", "lo", "hi", "days", "verdict", "resale"]
    empty = (pd.DataFrame(columns=cols), pd.DataFrame(columns=dcols), pd.DataFrame(columns=lcols))
    u = p[p["usable"]]
    if u.empty or daily.empty:
        return empty
    lines = trigger_lines(u, cfg)
    ev = find_events(p, cfg, lines)
    if ev.empty:
        return empty
    alld = graded_days(p, cfg, lines)
    lvl = daily[["item_id", "day", "level"]]
    ev = ev.merge(pools, on="item_id")
    alld = alld.merge(pools, on="item_id")
    if sold7 is not None and len(sold7):
        ev = ev.merge(sold7, on=["item_id", "day"], how="left")
    else:
        ev["sold7"] = np.nan
    rows, drows, lrows = [], [], []
    edges = [cfg.min_discount, 0.06, 0.10, 0.20, 10.0]
    names = [f"{edges[0] * 100:.0f}-6% under", "6-10% under", "10-20% under", "20%+ under"]
    keep = 1.0 - cfg.cut
    ok = ev["sold7"].where(ev["sold7"] > 0)
    ev = ev.assign(dts=ev["u5"] / ok, plan_qty=np.minimum(ev["u5"], cfg.plan_days * ok))
    for k in cfg.sale_days:
        later = lvl.assign(day=lvl["day"] - pd.Timedelta(days=k)).rename(columns={"level": "sale"})
        e = ev.merge(later, on=["item_id", "day"])
        a = alld.merge(later, on=["item_id", "day"])
        if e.empty:
            continue
        e = e.assign(md=e["sale"] * keep / e["price"] - 1.0, mr=e["sale"] * keep / e["b_rnd"] - 1.0, disc=1.0 - e["price"] / e["med"])
        e["bucket"] = pd.cut(e["disc"], bins=edges, labels=names, right=False)
        e["plan_cost"] = e["plan_qty"] * e["price"]
        e["plan_profit"] = e["plan_cost"] * e["md"]
        a = a.assign(ma=a["sale"] * keep / a["b_rnd"] - 1.0)
        n_all = a.groupby("pool")["item_id"].size()
        for pool, d in e.groupby("pool"):
            if d["item_id"].nunique() < cfg.min_items:
                continue
            al = a[a["pool"] == pool]
            diff = (d["md"].clip(-1, 5) - d["mr"].clip(-1, 5)).groupby(d["day"]).median()
            m, lo, hi = W.block_bootstrap_mean(diff.to_numpy(), cfg.boot, cfg.block, cfg.seed)
            g_dip, l_dip = float((d["md"] >= cfg.min_profit).mean()), float((d["md"] <= cfg.loss_margin).mean())
            rows.append({"pool": pool, "k": k, "dip_days": len(d), "dip_share": len(d) / max(int(n_all.get(pool, 0)), 1),
                         "units_med": float(d["u5"].median()), "units_p25": float(d["u5"].quantile(0.25)),
                         "discount_med": float((1.0 - d["price"] / d["med"]).median()),
                         "gain_dip": g_dip, "gain_random": float((d["mr"] >= cfg.min_profit).mean()),
                         "gain_all": float((al["ma"] >= cfg.min_profit).mean()) if len(al) else np.nan,
                         "loss_dip": l_dip, "loss_random": float((d["mr"] <= cfg.loss_margin).mean()),
                         "loss_all": float((al["ma"] <= cfg.loss_margin).mean()) if len(al) else np.nan,
                         "median_dip": float(d["md"].median()), "median_random": float(d["mr"].median()),
                         "edge": m, "lo": lo, "hi": hi, "days": len(diff), "verdict": W.verdict(m, lo, hi, len(diff), cfg.min_days, cfg.min_edge),
                         "resale": resale_verdict(float(d["md"].median()), g_dip, l_dip, cfg)})
            if pool == "everything" or pool.startswith("price"):
                n_days = max(d["day"].nunique(), 1)
                for name, bd in d.groupby("bucket", observed=True):
                    known = bd["plan_qty"].notna()
                    drows.append({"pool": pool, "k": k, "depth": name, "rank": names.index(name), "events": len(bd), "per_day": len(bd) / n_days,
                                  "units_med": float(bd["u5"].median()), "gain": float((bd["md"] >= cfg.min_profit).mean()), "loss": float((bd["md"] <= cfg.loss_margin).mean()),
                                  "median": float(bd["md"].median()), "sell_known": float(known.mean()),
                                  "days_to_sell": float(bd["dts"].median()) if known.any() else np.nan,
                                  "plan_qty": float(bd["plan_qty"].median()) if known.any() else np.nan,
                                  "plan_cost": float(bd["plan_cost"].median()) if known.any() else np.nan,
                                  "plan_profit": float(bd["plan_profit"].median()) if known.any() else np.nan})
                    for h in cfg.lags:
                        ph = bd[f"lagp{h}"]
                        still = ph.notna() & (ph <= bd["price"] * (1.0 + cfg.arrive_tol))
                        sub = bd[still]
                        lm = sub["sale"] * keep / sub[f"lagp{h}"] - 1.0
                        lrows.append({"pool": pool, "k": k, "depth": name, "rank": names.index(name), "lag": h, "events": len(bd), "still": float(still.mean()),
                                      "units_med": float(sub[f"lagu{h}"].median()) if len(sub) else np.nan,
                                      "median": float(lm.median()) if len(sub) else np.nan, "gain": float((lm >= cfg.min_profit).mean()) if len(sub) else np.nan})
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(drows, columns=dcols), pd.DataFrame(lrows, columns=lcols)


def dip_table(p: pd.DataFrame, pools: pd.DataFrame, daily: pd.DataFrame, cfg: Optional[DipConfig] = None) -> pd.DataFrame:
    return dip_tables(p, pools, daily, cfg)[0]


TIMING = {"holds on later days": "cheaper than a random hour", "no pattern visible": "no timing edge visible", "worse than random": "worse than a random hour", "too few test days": "too few days"}


def format_dips(res: dict, top_groups: int = 10) -> str:
    cfg, tab, depth, lag = res["cfg"], res["table"], res["depth"], res.get("lag", pd.DataFrame())
    L = [f"== DIP TEST: buy at the first hour of a day when the price is in the bottom {cfg.quantile * 100:.0f}% of the item's own last {cfg.lookback_days} days AND at least {cfg.min_discount * 100:.0f}% under its median of those days"]
    pct = lambda x: "  n/a" if x is None or np.isnan(x) else f"{x * 100:5.1f}%"
    sgn = lambda x: "   n/a" if x is None or np.isnan(x) else f"{x * 100:+6.1f}%"
    num = lambda x, f="{:,.0f}": "n/a" if x is None or np.isnan(x) else f.format(x)
    if tab.empty:
        L.append(f"   No dips to grade: either there are fewer than {cfg.lookback_days + 2} days of history (the trigger line needs {cfg.lookback_days} days, then a sale day after the dip), or no item fell {cfg.min_discount * 100:.0f}% under its own median into its bottom fifth.")
        return "\n".join(L)
    L.append(f"   Sale = the item's median price k days later minus the {cfg.cut * 100:.0f}% cut. 'dip' = bought at the dip hour; 'random' = bought at the average hour of the SAME item-days; 'all' = every item-day.")
    L.append(f"   Gain = at least {cfg.min_profit * 100:.0f}% left after the cut. Loss = the price fell {cfg.min_profit * 100:.0f}% or more ({-cfg.loss_margin * 100:.0f}% lost after the cut). A flat price lands exactly at the cut ({-cfg.cut * 100:.0f}%) and is neither.")
    L.append(f"   Edge = how much extra margin the dip hour gave over the average hour of the same days (median of each day's buys, averaged over days; 90% range over days; needs {cfg.min_days} days and {cfg.min_edge * 100:.0f} point for a verdict).")
    L.append(f"   Edge alone says nothing about resale: any hour that happens to be low is lower than the average hour. Resale pays only if the TYPICAL (median) buy clears {cfg.min_profit * 100:.0f}% after the cut and gains outnumber losses.")
    head = f"   {'pool':34s} {'k':>1s} {'dip-days':>8s} {'of days':>7s} {'units near floor':>17s} {'discount':>8s} | {'gain: dip':>9s} {'random':>7s} {'all':>6s} | {'loss: dip':>9s} {'random':>7s} {'all':>6s} | {'median: dip':>11s} {'random':>7s} | {'edge (90% range)':>24s}  verdict"
    broad = tab[(tab["pool"] == "everything") | tab["pool"].str.startswith("price")]
    groups = tab[tab["pool"].str.startswith("group:")]

    def row(r):
        return (f"   {r['pool'][:34]:34s} {int(r['k']):1d} {int(r['dip_days']):8d} {pct(r['dip_share'])} {r['units_med']:8.0f} /{r['units_p25']:7.0f} {pct(r['discount_med']):>8s} | "
                f"{pct(r['gain_dip']):>9s} {pct(r['gain_random']):>7s} {pct(r['gain_all']):>6s} | {pct(r['loss_dip']):>9s} {pct(r['loss_random']):>7s} {pct(r['loss_all']):>6s} | "
                f"{sgn(r['median_dip']):>11s} {sgn(r['median_random']):>7s} | {sgn(r['edge'])} ({sgn(r['lo']).strip()} to {sgn(r['hi']).strip()})  timing: {TIMING.get(r['verdict'], r['verdict'])}; {r['resale']}")
    L.append(head)
    for _, r in broad.sort_values(["pool", "k"]).iterrows():
        L.append(row(r))
    g2 = groups[groups["k"] == 2].sort_values("edge", ascending=False)
    if len(g2):
        L.append(f"   Item groups, best {min(top_groups, len(g2))} of {len(g2)} by edge at k = 2 (sorted best-first, so some look good by luck: judge the broad rows first):")
        for _, r in g2.head(top_groups).iterrows():
            L.append(row(r))
    if len(depth):
        L.append("")
        L.append("== BY HOW DEEP THE DIP WAS (the same buys, split by the discount against the item's 7-day median). A resale needs about a 10.5% rise, so only deep dips can pay.")
        L.append(f"   Days to sell = units near the floor / the item's normal daily sales (median of the previous 7 days). Typical plan = at most {cfg.plan_days} days of normal sales, never more than the units near the floor: units, cost and expected profit in gold (medians).")
        L.append(f"   {'pool':34s} {'k':>1s} {'dip depth':>13s} {'buys':>7s} {'per day':>7s} {'units near floor':>16s} | {'gain':>6s} {'even':>6s} {'loss':>6s} | {'median margin':>13s} | {'days to sell':>12s} {'plan units':>10s} {'cost g':>8s} {'profit g':>9s}")
        for _, r in depth.sort_values(["pool", "k", "rank"]).iterrows():
            even = 1.0 - r["gain"] - r["loss"]
            L.append(f"   {r['pool'][:34]:34s} {int(r['k']):1d} {str(r['depth']):>13s} {int(r['events']):7d} {r['per_day']:7.0f} {r['units_med']:16.0f} | {pct(r['gain']):>6s} {pct(even):>6s} {pct(r['loss']):>6s} | "
                     f"{sgn(r['median']):>13s} | {num(r['days_to_sell'], '{:.1f}'):>12s} {num(r['plan_qty']):>10s} {num(r['plan_cost']):>8s} {num(r['plan_profit']):>9s}")
    if len(lag):
        L.append("")
        L.append(f"== CAN YOU STILL BUY IT? Arriving h hours after the first dip hour. 'Still there' = the price is within {cfg.arrive_tol * 100:.0f}% of the dip price at that hour. Margin = buying at THAT hour's price (those still there), selling k days later.")
        ks = [k for k in (2, 3) if k in set(lag["k"])] or sorted(set(lag["k"]))[-2:]
        L.append(f"   {'pool':34s} {'dip depth':>13s} {'h':>1s} {'still there':>11s} {'units near floor':>16s} | " + " ".join(f"{'median margin k=' + str(k):>19s}" for k in ks))
        sel = lag[(lag["rank"] >= 2) & lag["k"].isin(ks)]
        base = sel[sel["k"] == ks[0]].set_index(["pool", "rank", "depth", "lag"])
        for (pool, rank, depth_name, h), r in base.sort_index().iterrows():
            ms = []
            for k in ks:
                x = sel[(sel["k"] == k) & (sel["pool"] == pool) & (sel["rank"] == rank) & (sel["lag"] == h)]
                ms.append(sgn(x["median"].iloc[0]) if len(x) else "   n/a")
            L.append(f"   {pool[:34]:34s} {str(depth_name):>13s} {int(h):1d} {pct(r['still']):>11s} {num(r['units_med']):>16s} | " + " ".join(f"{m:>19s}" for m in ms))
        L.append("   If 'still there' falls fast with h, the dip is gone before a person can act and only an instant alert helps. If it stays high, a check once or twice a day catches it.")
    L.append("   Read this as: for a RESALE look at the median margin against the bar, and at gain against loss. For a CRAFTING INPUT (copper ore now, tin later), the edge is what waiting for a dip saves over")
    L.append("   buying at a random time, and the units column is how many you could take near that price. A dip that is really a lasting price drop does not come back, which the median margin shows.")
    return "\n".join(L)


def analyze_dips(hl: pd.DataFrame, groups: Optional[pd.DataFrame] = None, wcfg: Optional[W.WindowConfig] = None, cfg: Optional[DipConfig] = None,
                 sales: Optional[pd.DataFrame] = None) -> dict:
    """{'table', 'depth', 'lag', 'cfg'} for an hourly table (HOURLY_COLUMNS), pooled like the clock test: everything, price tiers, item groups. `wcfg` carries the price column and tiers.
    `sales` is the item_daily table (item_id, day, sold_day), used for days to sell and the plan size; without it those columns are empty."""
    wcfg = wcfg or W.WindowConfig()
    cfg = cfg or DipConfig(cut=wcfg.cut, min_profit=wcfg.min_profit, sale_days=wcfg.sale_days, min_hours_day=wcfg.min_hours_day, min_items=wcfg.min_items, boot=wcfg.boot,
                           block=wcfg.block, seed=wcfg.seed)
    p = W.prepare(hl, wcfg)
    if p.empty or not p["usable"].any():
        t, d, g = dip_tables(p, pd.DataFrame(), pd.DataFrame(), cfg)
    else:
        t, d, g = dip_tables(p, W.label_items(p, groups, wcfg), W.daily_lows(p, wcfg), cfg, sold_baseline(sales))
    return {"table": t, "depth": d, "lag": g, "cfg": cfg}
