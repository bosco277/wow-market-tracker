"""Flip shelves: cheap units sitting below a gap in the price tiers, worth buying and relisting at the tier above the gap.

Example: 30 units at 1.00g, 100 at 1.10g, 1000 at 1.50g. Buy the 130, relist at 1.50g. Newest-first puts you at the
front of the 1.50g tier. Net of the 5% cut the relist brings 1.425g against an average cost of about 1.077g.

A shelf is only a CANDIDATE. Each one gets these tests, each marked pass / warn / fail (the numbers behind them are in
the table so you can judge for yourself):

  price     has the lowest price actually been at the target level in the recent past, and is the target inside the
            item's typical band (not a spike)?
  speed     how many hours until your units AND any cheaper units left ahead of you are bought, at the item's sales rate
            while the price was at that level (low / best / high). Falls back to the overall rate, with a warning.
  size      fits the unit and gold limits you set (inventory is finite). A partial buy leaves cheap units ahead of you,
            which slows the sale, and the speed test counts them.
  margin    net margin at the target, and again if you have to sell 10% under it. Break-even is cost / 0.95.
  book      the book looks complete: units listed now are not far below normal (a partial book makes fake gaps).

verdict:  candidate = no fail and at most one warn;  weak = no fail;  no = at least one fail.
These describe prices and rates. They do not know whether someone else will undercut you, or whether the cheap units are
another flipper's inventory that will be relisted. Check the live auction house before buying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .market import AH_CUT, _tiers
from .schema import COPPER_PER_GOLD


@dataclass(frozen=True)
class ShelfConfig:
    min_gap: float = 0.15            # the tier above the shelf must be this much dearer than the tier below it
    max_units: int = 300             # most units you will buy for one flip (inventory space)
    max_cost_gold: float = 300.0     # most gold in one flip
    min_net_margin: float = 0.10     # drop shelves that would not net this at the target price
    stress_undercut: float = 0.10    # "if I must sell this much under the target"
    min_stress_margin: float = 0.03
    max_sell_hours: float = 24.0     # a listing lasts about this long
    level_tolerance: float = 0.02    # the price counts as "at the level" within this of the target
    min_level_hours: int = 6         # hours at the level needed to trust the sales rate seen there
    min_level_share: float = 0.10    # share of recent hours the price should have been at the level
    min_book_share: float = 0.70     # units now / normal units below this = partial book
    ah_cut: float = AH_CUT


def _tier_test(ok: bool, warn: bool = False) -> str:
    return "pass" if ok else ("warn" if warn else "fail")


def level_stats(series: pd.DataFrame, target_gold: float, tol: float) -> dict:
    """From the item's hourly rows (floor_gold, sold_low/est/high, spike): share of hours with the lowest price at or above
    the target, and the sales rate seen in those hours. The sold_* of an hour is the interval ending at it, so this is
    approximate at the edges."""
    s = series.dropna(subset=["floor_gold"])
    if "spike" in s:
        s = s[~s["spike"].astype(bool)]
    if s.empty:
        return dict(share=0.0, hours=0, rate=None)
    at = s["floor_gold"] >= target_gold * (1 - tol)
    got = s[at & s["sold_est"].notna()]
    rate = None
    if len(got):
        rate = {k: float(got[f"sold_{k}"].mean()) for k in ("low", "est", "high")}
    return dict(share=float(at.mean()), hours=int(len(got)), rate=rate)


def shelves_for_item(book: pd.DataFrame, series: pd.DataFrame, rate_per_hour: Dict[str, float], band_p90: float,
                     cfg: ShelfConfig = ShelfConfig()) -> Optional[dict]:
    """The best shelf for one item (by profit if you must undercut), or None. `book` is the latest listings of the item."""
    t = _tiers(book)
    if len(t) < 2:
        return None
    price, qty = t["unit_price"].to_numpy(dtype="float64"), t["qty"].to_numpy(dtype="float64")
    units_now = float(book["quantity"].sum())
    med_units = float(series["units_listed"].median()) if len(series) and "units_listed" in series else units_now
    best, best_key = None, -np.inf
    for k in range(1, len(t)):
        if price[k] / price[k - 1] - 1.0 < cfg.min_gap:
            continue
        shelf_units = float(qty[:k].sum())
        # buy the cheapest units first, until the unit or gold limit is hit
        bought, cost = 0.0, 0.0
        for p, q in zip(price[:k], qty[:k]):
            take = min(q, cfg.max_units - bought, (cfg.max_cost_gold * COPPER_PER_GOLD - cost) / p)
            if take < 1:
                break
            bought += take
            cost += take * p
        bought = float(int(bought))
        if bought < 1:
            continue
        # recompute cost for the whole units actually bought
        rem, cost = bought, 0.0
        for p, q in zip(price[:k], qty[:k]):
            take = min(q, rem)
            cost += take * p
            rem -= take
            if rem <= 0:
                break
        avg = cost / bought
        target = price[k]
        net_target = target * (1 - cfg.ah_cut)
        margin = net_target / avg - 1.0
        if margin < cfg.min_net_margin:
            continue
        stress_price = target * (1 - cfg.stress_undercut)
        margin_stress = stress_price * (1 - cfg.ah_cut) / avg - 1.0
        ahead = shelf_units - bought                      # cheap units left in the book, ahead of your relisted units
        tg = target / COPPER_PER_GOLD
        lv = level_stats(series, tg, cfg.level_tolerance)
        use_level = lv["rate"] is not None and lv["hours"] >= cfg.min_level_hours
        rates = lv["rate"] if use_level else {k_: float(v) for k_, v in rate_per_hour.items()}
        sell = {k_: ((ahead + bought) / rates[k_] if rates[k_] > 0 else float("inf")) for k_ in ("low", "est", "high")}
        t_price = _tier_test(
            lv["share"] >= cfg.min_level_share and (band_p90 != band_p90 or tg <= band_p90 * 1.05),
            warn=lv["share"] > 0 or (band_p90 == band_p90 and tg <= band_p90 * 1.05))
        t_speed = _tier_test(sell["low"] <= cfg.max_sell_hours, warn=sell["est"] <= cfg.max_sell_hours)
        if not use_level and t_speed == "pass":
            t_speed = "warn"                              # the rate was not seen at this price
        t_size = _tier_test(bought >= shelf_units, warn=bought >= 0.5 * shelf_units)
        t_margin = _tier_test(margin_stress >= cfg.min_stress_margin, warn=margin_stress > 0)
        book_share = units_now / med_units if med_units > 0 else 1.0
        t_book = _tier_test(book_share >= cfg.min_book_share)
        tests = dict(test_price=t_price, test_speed=t_speed, test_size=t_size, test_margin=t_margin, test_book=t_book)
        fails, warns = sum(v == "fail" for v in tests.values()), sum(v == "warn" for v in tests.values())
        verdict = "candidate" if fails == 0 and warns <= 1 else ("weak" if fails == 0 else "no")
        row = dict(
            units=int(bought), shelf_units=int(shelf_units), cost_gold=cost / COPPER_PER_GOLD, avg_price_gold=avg / COPPER_PER_GOLD,
            target_gold=tg, gap=float(price[k] / price[k - 1] - 1.0), breakeven_gold=avg / (1 - cfg.ah_cut) / COPPER_PER_GOLD,
            net_margin=float(margin), net_margin_stress=float(margin_stress),
            profit_gold=float((net_target - avg) * bought / COPPER_PER_GOLD),
            profit_stress_gold=float((stress_price * (1 - cfg.ah_cut) - avg) * bought / COPPER_PER_GOLD),
            sell_hours_low=sell["low"], sell_hours_est=sell["est"], sell_hours_high=sell["high"],
            rate_from_level=bool(use_level), hours_at_level=lv["hours"], share_at_level=lv["share"], cheap_units_ahead=int(ahead),
            book_share=float(book_share), verdict=verdict, fails=fails, warns=warns, **tests)
        key = {"candidate": 2, "weak": 1, "no": 0}[verdict] * 1e9 + row["profit_stress_gold"]
        if key > best_key:
            best, best_key = row, key
    return best
