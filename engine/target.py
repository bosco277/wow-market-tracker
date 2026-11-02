"""Which daily price is calmer: Undermine's, or ours?

Undermine's daily price is the lowest price at the day's busiest snapshot, so one cheap listing can set it. Our own `item_daily.price_day` is the MEDIAN of at least 18 hourly
lowest prices with spike hours already dropped. The forecast bake-off on Undermine's series found that about 40% of cases come from items whose price swings 60% or more in a day,
which says the target is noisy, not that models are weak. This module puts the two measures side by side on the days and items they share (consecutive days only) and says how
much of the day-to-day movement in Undermine's series is real, meaning our own measure moves with it, and how much is a one-snapshot blip that our measure smooths away.

Group-level numbers only. Days are matched by calendar label: ours are US Eastern, Undermine's are probably UTC, so a few hours of offset is possible; each series' own swings are
compared, which that offset hardly touches."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import history as HI

MIN_HOURS = 18              # a day of ours counts only with this many usable hours
MIN_PAIRS = 200             # fewer paired day-to-day changes than this and there is nothing to compare
MIN_ITEM_CHANGES = 7        # an item's own jumpiness needs this many paired changes
BIG = float(np.log10(1.25))
CLIP = 1.0


def paired_changes(own: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """One row per item per day where BOTH measures exist on that day and the day before: item_id, day, own (gold), theirs (gold), d_own, d_theirs (log10 change from the day before)."""
    if own is None or own.empty or hist is None or hist.empty:
        return pd.DataFrame()
    o = own[own["hours"] >= MIN_HOURS][["item_id", "day", "price_day"]].rename(columns={"price_day": "own"}).copy()
    o["day"] = pd.to_datetime(o["day"]).dt.normalize()
    o["item_id"] = o["item_id"].astype(int)
    t = HI.clean(hist)[["item_id", "day", "price"]].rename(columns={"price": "theirs"})
    t["day"] = t["day"].dt.normalize()
    m = o.merge(t, on=["item_id", "day"])
    m = m[(m["own"] > 0) & (m["theirs"] > 0)].sort_values(["item_id", "day"]).reset_index(drop=True)
    if m.empty:
        return m
    g = m.groupby("item_id")
    consecutive = (m["day"] - g["day"].shift(1)) == pd.Timedelta(days=1)
    m["d_own"] = np.log10(m["own"]) - np.log10(g["own"].shift(1))
    m["d_theirs"] = np.log10(m["theirs"]) - np.log10(g["theirs"].shift(1))
    return m[consecutive & m["d_own"].notna() & m["d_theirs"].notna()].reset_index(drop=True)


def _pct(x: float) -> float:
    return (10.0 ** x - 1.0) * 100.0


def _stats(p: pd.DataFrame) -> dict:
    do, dt = p["d_own"].to_numpy(), p["d_theirs"].to_numpy()
    ao, at = np.abs(do), np.abs(dt)
    per = p.groupby("item_id").agg(n=("d_own", "size"), so=("d_own", "std"), st=("d_theirs", "std"))
    per = per[per["n"] >= MIN_ITEM_CHANGES]
    big_t = at > BIG
    confirmed = big_t & (np.sign(do) == np.sign(dt)) & (ao >= 0.5 * at)
    blip = big_t & (ao < np.log10(1.05))
    ratio = np.log10(p["own"].to_numpy() / p["theirs"].to_numpy())
    co, ct = np.clip(do, -CLIP, CLIP), np.clip(dt, -CLIP, CLIP)
    return {
        "changes": len(p), "items": int(p["item_id"].nunique()),
        "own_avg": _pct(float(ao.mean())), "theirs_avg": _pct(float(at.mean())),
        "own_typ": _pct(float(np.median(ao))), "theirs_typ": _pct(float(np.median(at))),
        "own_big": 100.0 * float((ao > BIG).mean()), "theirs_big": 100.0 * float(big_t.mean()),
        "own_jump": _pct(float(per["so"].median())) if len(per) else np.nan, "theirs_jump": _pct(float(per["st"].median())) if len(per) else np.nan,
        "calmer_x": float(at.mean() / ao.mean()) if ao.mean() > 0 else np.nan,
        "level_gap": _pct(float(np.median(ratio))), "within10": 100.0 * float((np.abs(ratio) <= np.log10(1.10)).mean()),
        "corr": float(np.corrcoef(co, ct)[0, 1]) if len(p) > 2 and co.std() > 0 and ct.std() > 0 else np.nan,
        "confirmed": 100.0 * float(confirmed.sum() / big_t.sum()) if big_t.any() else np.nan,
        "blip": 100.0 * float(blip.sum() / big_t.sum()) if big_t.any() else np.nan,
    }


def compare_targets(own: pd.DataFrame, hist: pd.DataFrame, groups: Optional[pd.DataFrame] = None) -> dict:
    """{} when the shared days are too few; otherwise {overall: one-row frame, family: frame, days, first, last}.
    Columns: own_* is our measure, theirs_* is Undermine's. avg / typ = average / median |day-to-day change| in percent; big = share of changes over 25%; jump = the median item's
    spread of daily changes; calmer_x = theirs_avg / own_avg (above 1 means ours is calmer); level_gap = median own/theirs - 1 and within10 = share of item-days within 10% of each other;
    corr = how closely the two measures' daily changes move together; confirmed = of Undermine's 25%+ moves, the share ours also made (same direction, at least half as big);
    blip = of those moves, the share ours barely noticed (under 5%)."""
    p = paired_changes(own, hist)
    if len(p) < MIN_PAIRS:
        return {"pairs": len(p)}
    out = {"pairs": len(p), "days": int(p["day"].nunique()), "first": p["day"].min(), "last": p["day"].max(), "overall": pd.DataFrame([_stats(p)])}
    if groups is not None and not groups.empty and {"item_id", "family"}.issubset(groups.columns):
        fam = groups.drop_duplicates("item_id").set_index("item_id")["family"]
        p = p.assign(family=p["item_id"].map(fam).fillna("Unknown"))
        rows = []
        for f, q in p.groupby("family"):
            if len(q) >= MIN_PAIRS // 2:
                rows.append({"family": f, **_stats(q)})
        out["family"] = pd.DataFrame(rows)
    return out
