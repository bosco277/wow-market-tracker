"""Buy windows: is there a time of day, or a day of the week, when buying a commodity is reliably cheaper, and does it hold on days the test has never seen?

Input: the permanent hourly table (engine/daily.py `hourly_long`: item_id, ts, floor_gold, units_listed, u5, u10, p15, p30, flags; ts is naive UTC) and, optionally,
the item_groups table (item_id, group, quality) used to pool items of one family so a pattern has many items to show up in.

The test is WALK-FORWARD, so it cannot flatter itself: for every test day it learns the cheapest window only from days BEFORE that day, then looks up what that window
cost on the test day, against the average over all hours of that same day (what buying at a random hour pays). Savings are shown as a range (a block bootstrap over days,
because neighbouring days resemble each other), never as one number. A pattern that only exists in the days it was learned from scores about zero here.

Three questions, all answered per item group (and per price tier, and for everything together):
  1. Hour of day: the best `window_hours` in a row (US Eastern), learned from earlier days, against a random hour.
  2. Day of week: the cheapest weekday (by each day's low price against the centred 7-day level), learned from earlier weeks. Needs about 7 weeks.
  3. Resale check: buying in the learned hour window, then selling at the market price 1, 2 or 3 days later: how often does that leave at least `min_profit` after the AH cut?

Prices are the floor (the single cheapest listing, which may be one unit) or the depth price p15 (what the cheapest 15% of the units cost: closer to what buying a
quantity pays). Hours where the feed looked wrong (blip), the book was wiped, or data was missing for long (long gap) are left out; short price dips stay IN, because a
dip is what a buyer is looking for. Nothing here reads or writes the network: tools/buy_windows.py loads the tables and prints the report.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TZ = "America/New_York"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class WindowConfig:
    tz: str = TZ
    price_col: str = "floor_gold"       # "floor_gold" or "p15" (depth price, falls back to the floor where there is no depth price)
    window_hours: int = 3               # length of the buying window, in hours
    min_hours_day: int = 18             # an item-day needs this many usable hours
    min_items: int = 3                  # a group needs this many items in an hour for that hour to count
    min_train_days: int = 10            # the hour-of-day test starts after this many days of learning
    min_train_dow: int = 28             # the weekday test learns from at least this many days
    clip: float = 1.0                   # log price ratios are clipped to +-clip (a 2.7x move), so one broken price cannot drive a group
    cut: float = 0.05                   # AH cut taken from a sale
    min_profit: float = 0.05            # the resale check wants at least this much left after the cut
    sale_days: tuple = (1, 2, 3)
    tiers: tuple = (1.0, 100.0)         # price tiers by the item's median price in gold: under 1g, 1g to 100g, 100g and up
    boot: int = 400
    block: int = 3                      # days per block in the bootstrap
    seed: int = 7


def tier_label(gold: float, tiers=(1.0, 100.0)) -> str:
    if gold < tiers[0]:
        return f"price under {tiers[0]:g}g"
    if gold < tiers[1]:
        return f"price {tiers[0]:g}g to {tiers[1]:g}g"
    return f"price {tiers[1]:g}g and up"


# ------------------------------------------------------------------ preparing the hours

def prepare(hl: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """Hourly rows with: item_id, day (US Eastern date, midnight, no zone), hour (0-23 ET), dow (0 = Monday), price (the chosen price column), usable."""
    cfg = cfg or WindowConfig()
    if hl is None or hl.empty:
        return pd.DataFrame(columns=["item_id", "day", "hour", "dow", "price", "usable"])
    h = hl.copy()
    t = pd.to_datetime(h["ts"])
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    et = t.dt.tz_convert(cfg.tz)
    h["day"] = et.dt.tz_localize(None).dt.normalize()
    h["hour"] = et.dt.hour
    h["dow"] = et.dt.dayofweek
    price = h["floor_gold"].astype(float)
    if cfg.price_col == "p15" and "p15" in h.columns:
        price = h["p15"].astype(float).fillna(price)
    h["price"] = price
    bad = pd.Series(False, index=h.index)
    for c in ("blip", "flag_wipe", "flag_long_gap"):
        if c in h.columns:
            bad |= h[c].fillna(False).astype(bool)
    h["usable"] = ~bad & h["price"].notna() & (h["price"] > 0)
    return h[["item_id", "day", "hour", "dow", "price", "usable"]].reset_index(drop=True)


def relative_hours(p: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """Per usable hour: log of its price over the median price of the same item on the same day. Item-days with too few usable hours are dropped."""
    cfg = cfg or WindowConfig()
    u = p[p["usable"]]
    if u.empty:
        return pd.DataFrame(columns=["item_id", "day", "hour", "dow", "price", "ref", "rel"])
    g = u.groupby(["item_id", "day"])["price"]
    ref = g.transform("median")
    n = g.transform("size")
    out = u.assign(ref=ref, n=n)
    out = out[out["n"] >= cfg.min_hours_day].copy()
    out["rel"] = np.log(out["price"] / out["ref"]).clip(-cfg.clip, cfg.clip)
    return out[["item_id", "day", "hour", "dow", "price", "ref", "rel"]].reset_index(drop=True)


def label_items(p: pd.DataFrame, groups: Optional[pd.DataFrame], cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """item_id -> list of pools it belongs to: its item group, its price tier, and 'everything'. Returns long form (item_id, pool)."""
    cfg = cfg or WindowConfig()
    ids = pd.DataFrame({"item_id": sorted(p["item_id"].unique())})
    med = p[p["usable"]].groupby("item_id")["price"].median().rename("med")
    ids = ids.merge(med, on="item_id", how="left")
    rows = [pd.DataFrame({"item_id": ids["item_id"], "pool": "everything"})]
    ok = ids.dropna(subset=["med"])
    rows.append(pd.DataFrame({"item_id": ok["item_id"], "pool": [tier_label(x, cfg.tiers) for x in ok["med"]]}))
    if groups is not None and len(groups) and "group" in groups.columns:
        g = groups.drop_duplicates("item_id")[["item_id", "group"]]
        g = g[g["item_id"].isin(ids["item_id"])]
        rows.append(pd.DataFrame({"item_id": g["item_id"], "pool": "group: " + g["group"].astype(str)}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ hour of day

def hour_matrices(rel: pd.DataFrame, pools: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> dict:
    """{pool: DataFrame(index day, columns hours 0-23)}: the mean log price ratio across the pool's items, NaN where fewer than min_items items had that hour."""
    cfg = cfg or WindowConfig()
    m = rel.merge(pools, on="item_id")
    out = {}
    for pool, d in m.groupby("pool"):
        if d["item_id"].nunique() < cfg.min_items:
            continue
        g = d.groupby(["day", "hour"])["rel"].agg(["mean", "size"]).reset_index()
        g.loc[g["size"] < cfg.min_items, "mean"] = np.nan
        mat = g.pivot(index="day", columns="hour", values="mean").reindex(columns=range(24)).sort_index()
        out[pool] = mat
    return out


def _best_start(profile: np.ndarray, w: int) -> int:
    """Start hour of the w consecutive hours (no wrap past midnight) with the lowest mean of the profile; -1 if no window has data."""
    best, best_s = np.inf, -1
    for s in range(0, 24 - w + 1):
        seg = profile[s:s + w]
        if np.isnan(seg).any():
            continue
        v = seg.mean()
        if v < best:
            best, best_s = v, s
    return best_s


def walk_forward_hours(mats: dict, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """One row per pool and test day: pool, day, start (the window learned from earlier days), chosen (its mean log ratio that day), baseline (all hours that day),
    oracle (the best window of that day, which no one can know in advance: an upper limit)."""
    cfg = cfg or WindowConfig()
    w = cfg.window_hours
    rows = []
    for pool, mat in mats.items():
        vals = mat.to_numpy(dtype=float)
        days = list(mat.index)
        for i in range(len(days)):
            train = vals[:i]
            if np.sum(~np.isnan(train).all(axis=1)) < cfg.min_train_days:
                continue
            counts = np.sum(~np.isnan(train), axis=0)
            if (counts < cfg.min_train_days // 2).any():
                continue
            prof = np.nanmean(train, axis=0)
            s = _best_start(prof, w)
            today = vals[i]
            if s < 0 or np.isnan(today).all():
                continue
            seg = today[s:s + w]
            if np.isnan(seg).any():
                continue
            wins = [today[k:k + w].mean() for k in range(0, 24 - w + 1) if not np.isnan(today[k:k + w]).any()]
            rows.append({"pool": pool, "day": days[i], "start": s, "chosen": float(seg.mean()),
                         "baseline": float(np.nanmean(today)), "oracle": float(min(wins)) if wins else np.nan})
    return pd.DataFrame(rows, columns=["pool", "day", "start", "chosen", "baseline", "oracle"])


def block_bootstrap_mean(x: np.ndarray, boot: int = 400, block: int = 3, seed: int = 7, lo: float = 5, hi: float = 95):
    """Mean of x and its (lo, hi) percentile range, resampling blocks of `block` neighbouring values (days) so autocorrelation widens the range as it should."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n < 2 * block:
        block = 1
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(boot)
    for b in range(boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    return float(x.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def verdict(mean: float, lo: float, hi: float, n: int, min_n: int = 10, min_saving: float = 0.005) -> str:
    """Plain words for a saving (as a fraction) and its 90% range."""
    if n < min_n or np.isnan(mean):
        return "too few test days"
    if lo > 0 and mean >= min_saving:
        return "holds on later days"
    if hi < 0:
        return "worse than random"
    return "no pattern visible"


def summarize_hours(wf: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """Per pool: test days, mean saving of the learned window against a random hour (fraction, with 90% range), share of days it beat random, the oracle limit, the most
    often chosen window and how steadily it was chosen."""
    cfg = cfg or WindowConfig()
    rows = []
    for pool, d in wf.groupby("pool"):
        sav = 1.0 - np.exp(d["chosen"].to_numpy() - d["baseline"].to_numpy())
        orc = 1.0 - np.exp(d["oracle"].to_numpy() - d["baseline"].to_numpy())
        m, lo, hi = block_bootstrap_mean(sav, cfg.boot, cfg.block, cfg.seed)
        mode = d["start"].mode().iloc[0]
        rows.append({"pool": pool, "test_days": len(d), "saving": m, "lo": lo, "hi": hi, "beat_random": float((sav > 0).mean()),
                     "oracle": float(np.nanmean(orc)), "window_start": int(mode), "window_steady": float((d["start"] == mode).mean()),
                     "verdict": verdict(m, lo, hi, len(d))})
    return pd.DataFrame(rows).sort_values(["verdict", "saving"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["pool", "test_days", "saving", "lo", "hi", "beat_random", "oracle", "window_start", "window_steady", "verdict"])


def hour_profile(mats: dict, pool: str = "everything") -> Optional[pd.Series]:
    """The average % above (+) or below (-) the day's median price for each ET hour, over all days of one pool."""
    mat = mats.get(pool)
    if mat is None or mat.empty:
        return None
    return (np.exp(mat.mean(axis=0)) - 1.0).rename("pct_vs_day_median")


# ------------------------------------------------------------------ day of week

def daily_lows(p: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """Per item and day: low (10th percentile of the day's usable hourly prices: about the 2-3 cheapest hours), level (their median), hours."""
    cfg = cfg or WindowConfig()
    u = p[p["usable"]]
    if u.empty:
        return pd.DataFrame(columns=["item_id", "day", "low", "level", "n"])
    g = u.groupby(["item_id", "day"])["price"]
    d = pd.DataFrame({"low": g.quantile(0.10), "level": g.median(), "n": g.size()}).reset_index()
    return d[d["n"] >= cfg.min_hours_day].reset_index(drop=True)


def weekday_matrices(daily: pd.DataFrame, pools: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> dict:
    """{pool: Series indexed by day}: mean over items of log(low / centred 7-day geometric mean of the low). Needs 7 consecutive days around a day."""
    cfg = cfg or WindowConfig()
    if daily.empty:
        return {}
    pieces = []
    for iid, d in daily.groupby("item_id"):
        s = np.log(d.set_index("day")["low"].sort_index())
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        rel = (full - full.rolling(7, center=True, min_periods=7).mean()).clip(-cfg.clip, cfg.clip).dropna()
        if len(rel):
            pieces.append(pd.DataFrame({"item_id": iid, "day": rel.index, "rel": rel.to_numpy()}))
    if not pieces:
        return {}
    rel = pd.concat(pieces, ignore_index=True).merge(pools, on="item_id")
    out = {}
    for pool, d in rel.groupby("pool"):
        if d["item_id"].nunique() < cfg.min_items:
            continue
        g = d.groupby("day")["rel"].agg(["mean", "size"])
        g = g[g["size"] >= cfg.min_items]["mean"].sort_index()
        if len(g):
            out[pool] = g
    return out


def weekday_profile(series: pd.Series) -> pd.DataFrame:
    """Per weekday: mean % above (+) or below (-) the 7-day level, number of days, and the standard error of that mean in %-points."""
    if series is None or series.empty:
        return pd.DataFrame(columns=["weekday", "pct", "days", "se"])
    d = pd.DataFrame({"v": series.to_numpy(), "dow": series.index.dayofweek})
    g = d.groupby("dow")["v"].agg(["mean", "std", "size"])
    out = pd.DataFrame({"weekday": [WEEKDAYS[i] for i in g.index], "pct": (np.exp(g["mean"]) - 1.0) * 100.0, "days": g["size"].to_numpy(),
                        "se": (g["std"] / np.sqrt(g["size"])).fillna(0.0).to_numpy() * 100.0})
    return out.reset_index(drop=True)


def walk_forward_weekday(wk: dict, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """One row per pool: days tested, test days that fell on the weekday learned from earlier days (only days at least 4 before, since a day's 7-day level reaches 3 days
    ahead), mean saving on those days against the average of all test days (fraction, 90% range)."""
    cfg = cfg or WindowConfig()
    rows = []
    for pool, s in wk.items():
        days, vals = list(s.index), s.to_numpy()
        chosen, tested = [], []
        for i, t in enumerate(days):
            cutoff = t - pd.Timedelta(days=4)
            tr = s[s.index <= cutoff]
            if len(tr) < cfg.min_train_dow:
                continue
            prof = tr.groupby(tr.index.dayofweek).agg(["mean", "size"])
            if len(prof) < 7 or (prof["size"] < 3).any():
                continue
            tested.append(vals[i])
            if t.dayofweek == int(prof["mean"].idxmin()):
                chosen.append(vals[i])
        if not tested:
            continue
        base = float(np.mean(tested))
        sav = 1.0 - np.exp(np.asarray(chosen) - base) if chosen else np.array([])
        m, lo, hi = block_bootstrap_mean(sav, cfg.boot, 1, cfg.seed) if len(sav) else (np.nan, np.nan, np.nan)
        rows.append({"pool": pool, "test_days": len(tested), "learned_days": len(chosen), "saving": m, "lo": lo, "hi": hi,
                     "verdict": verdict(m, lo, hi, len(chosen), min_n=6)})
    return pd.DataFrame(rows, columns=["pool", "test_days", "learned_days", "saving", "lo", "hi", "verdict"])


# ------------------------------------------------------------------ resale check

def resale_check(p: pd.DataFrame, rel: pd.DataFrame, wf: pd.DataFrame, pools: pd.DataFrame, daily: pd.DataFrame, cfg: Optional[WindowConfig] = None) -> pd.DataFrame:
    """Buy each item in its pool's learned window on each test day, sell at that item's median price k days later (k in cfg.sale_days) minus the AH cut. Per pool and k:
    items-days tested, mean margin, and the share that left at least min_profit, for the learned window and for a random-hour buy (all hours that day)."""
    cfg = cfg or WindowConfig()
    if wf.empty or daily.empty:
        return pd.DataFrame(columns=["pool", "k", "n", "margin_window", "margin_random", "share_window", "share_random"])
    w = cfg.window_hours
    m = rel.merge(pools, on="item_id").merge(wf[["pool", "day", "start"]], on=["pool", "day"])
    inwin = (m["hour"] >= m["start"]) & (m["hour"] < m["start"] + w)
    win = m[inwin].groupby(["pool", "item_id", "day"])["price"].mean().rename("b_win")
    rnd = m.groupby(["pool", "item_id", "day"])["price"].mean().rename("b_rnd")
    buy = pd.concat([win, rnd], axis=1).dropna().reset_index()
    lvl = daily[["item_id", "day", "level"]]
    rows = []
    for k in cfg.sale_days:
        later = lvl.assign(day=lvl["day"] - pd.Timedelta(days=k)).rename(columns={"level": "sale"})
        b = buy.merge(later, on=["item_id", "day"])
        if b.empty:
            continue
        net = b["sale"] * (1.0 - cfg.cut)
        b["mw"] = net / b["b_win"] - 1.0
        b["mr"] = net / b["b_rnd"] - 1.0
        for pool, d in b.groupby("pool"):
            rows.append({"pool": pool, "k": k, "n": len(d), "margin_window": float(d["mw"].clip(-1, 5).mean()), "margin_random": float(d["mr"].clip(-1, 5).mean()),
                         "share_window": float((d["mw"] >= cfg.min_profit).mean()), "share_random": float((d["mr"] >= cfg.min_profit).mean())})
    return pd.DataFrame(rows, columns=["pool", "k", "n", "margin_window", "margin_random", "share_window", "share_random"])


# ------------------------------------------------------------------ everything together

def analyze(hl: pd.DataFrame, groups: Optional[pd.DataFrame] = None, cfg: Optional[WindowConfig] = None) -> dict:
    """Run all three questions on an hourly table. Returns the tables plus how many days and items went in."""
    cfg = cfg or WindowConfig()
    p = prepare(hl, cfg)
    res = {"cfg": cfg, "days": 0, "items": 0, "hours": 0}
    if p.empty or not p["usable"].any():
        return {**res, "profile": None, "hours_summary": pd.DataFrame(), "weekday_summary": pd.DataFrame(), "weekday_profile": None, "resale": pd.DataFrame(), "walk": pd.DataFrame()}
    res.update(days=int(p["day"].nunique()), items=int(p["item_id"].nunique()), hours=int(len(p)), first_day=p["day"].min(), last_day=p["day"].max())
    pools = label_items(p, groups, cfg)
    rel = relative_hours(p, cfg)
    mats = hour_matrices(rel, pools, cfg)
    wf = walk_forward_hours(mats, cfg)
    daily = daily_lows(p, cfg)
    wk = weekday_matrices(daily, pools, cfg)
    res.update(profile=hour_profile(mats), walk=wf, hours_summary=summarize_hours(wf, cfg), weekday_profile=weekday_profile(wk.get("everything")),
               weekday_summary=walk_forward_weekday(wk, cfg), resale=resale_check(p, rel, wf, pools, daily, cfg))
    return res


# ------------------------------------------------------------------ the report

def _pct(x, digits=1):
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{digits}f}%"


def format_report(res: dict, top: int = 25) -> str:
    cfg = res["cfg"]
    L = []
    if not res.get("days"):
        return "No usable hourly rows. Run the daily chain (with upload) so engine/hourly/ fills, and backfill earlier days with an earlier end date."
    L.append(f"BUY WINDOWS: {res['items']:,} items, {res['hours']:,} hourly rows, {res['days']} US Eastern days ({res['first_day']:%Y-%m-%d} to {res['last_day']:%Y-%m-%d}). "
             f"Price used: {'depth price (cheapest 15% of units)' if cfg.price_col == 'p15' else 'floor (cheapest single listing)'}. Window: {cfg.window_hours} hours in a row.")
    L.append("Every result below is walk-forward: the window is learned only from days BEFORE the day it is graded on. 'Saving' is how much cheaper the learned window was than buying at a random hour of that day;")
    L.append("the range is 90%. The oracle column is the best window of each day, which nobody can know in advance: it shows how much is there to find, not what you can capture.")
    L.append("")
    prof = res.get("profile")
    if prof is not None:
        L.append("== HOUR OF DAY (all items): average price against that day's median, by US Eastern hour")
        cells = [f"{h:02d}h {v * 100:+.1f}%" for h, v in prof.items()]
        for i in range(0, 24, 6):
            L.append("   " + "   ".join(cells[i:i + 6]))
        L.append("")
    hs = res.get("hours_summary", pd.DataFrame())
    L.append(f"== HOUR-OF-DAY BUY WINDOW, tested on later days (first {cfg.min_train_days} days are learning only)")
    if hs.empty:
        L.append(f"   Not enough days yet: needs more than {cfg.min_train_days} days of hourly history. Backfill earlier days or wait.")
    else:
        L.append(f"   {'pool':42s} {'days':>5s} {'saving':>8s} {'90% range':>17s} {'beat random':>11s} {'oracle':>8s}  window (ET)    verdict")
        order = {"holds on later days": 0, "no pattern visible": 1, "worse than random": 2, "too few test days": 3}
        for _, r in hs.assign(o=hs["verdict"].map(order)).sort_values(["o", "saving"], ascending=[True, False]).head(top).iterrows():
            L.append(f"   {r['pool'][:42]:42s} {r['test_days']:5d} {_pct(r['saving']):>8s} {_pct(r['lo']):>8s} to {_pct(r['hi']):>6s} {r['beat_random'] * 100:10.0f}% {_pct(r['oracle']):>8s}  "
                     f"{int(r['window_start']):02d}-{int(r['window_start']) + cfg.window_hours:02d}h ({r['window_steady'] * 100:.0f}%)  {r['verdict']}")
        L.append("   (window (x%): the hours most often chosen and the share of test days that chose them. A window that keeps moving is noise.)")
    L.append("")
    wp = res.get("weekday_profile")
    L.append("== DAY OF WEEK (all items): each day's low price (about the 3 cheapest hours) against the centred 7-day level")
    if wp is None or wp.empty:
        L.append("   Not enough consecutive days yet (needs 7 in a row per item).")
    else:
        for _, r in wp.iterrows():
            L.append(f"   {r['weekday']}: {r['pct']:+.2f}%  (+/- {r['se']:.2f}, {int(r['days'])} days)")
    ws = res.get("weekday_summary", pd.DataFrame())
    L.append("")
    L.append(f"== DAY-OF-WEEK BUY DAY, tested on later days (learns from at least {cfg.min_train_dow} days, then grades on days at least 4 later)")
    if ws.empty:
        L.append(f"   Not enough days yet: the weekday test needs about {cfg.min_train_dow + 4 + 14} days of hourly history, which means about 7 weeks. Until then, treat the weekday as unknown.")
    else:
        for _, r in ws.sort_values("saving", ascending=False).head(top).iterrows():
            L.append(f"   {r['pool'][:42]:42s} test days {r['test_days']:4d}, on the learned weekday {r['learned_days']:3d}: saving {_pct(r['saving'])} ({_pct(r['lo'])} to {_pct(r['hi'])})  {r['verdict']}")
    L.append("")
    rs = res.get("resale", pd.DataFrame())
    L.append(f"== RESALE CHECK: buy in the learned window, sell at the median price k days later, after a {cfg.cut * 100:.0f}% AH cut. Share of item-days leaving at least {cfg.min_profit * 100:.0f}%")
    if rs.empty:
        L.append("   Nothing to check yet (needs test days that have a later sale day).")
    else:
        L.append(f"   {'pool':42s} {'k':>2s} {'item-days':>10s} {'learned window':>15s} {'random hour':>12s}   average margin (window / random)")
        for _, r in rs[rs["pool"].isin(["everything"] + [x for x in rs["pool"].unique() if x.startswith("price")])].sort_values(["pool", "k"]).iterrows():
            L.append(f"   {r['pool'][:42]:42s} {int(r['k']):2d} {int(r['n']):10d} {r['share_window'] * 100:14.1f}% {r['share_random'] * 100:11.1f}%   {_pct(r['margin_window'])} / {_pct(r['margin_random'])}")
        L.append("   Read this as: if the learned window is not clearly better than a random hour, time of day alone does not make a resale profit; a price DIP against the item's own range has to do the work.")
    return "\n".join(L)
