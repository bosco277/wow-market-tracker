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
  regime    is this a normal price or a temporary one? Compares what you would pay and the target with the item's longer
            history (median of everything we have, up to 8 days). Fails when you would pay well above the normal price
            (cost over 1.25x the median), when the last 2 days ran well above the longer-run price and the price is now coming back down (a market
            reset: the target is likely to normalize), or when the target is over 1.5x the normal price; warns when the
            target is over 1.25x the normal price or we have under 3 days of history.
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
    regime_cost_x: float = 1.25      # paying more than this times the long-run median is buying into a spike
    regime_target_warn_x: float = 1.25
    regime_target_fail_x: float = 1.5
    regime_spike_x: float = 1.3      # the last 48 h median this far above the longer-run median...
    regime_falling: float = 0.85     # ...with the price now below this share of the last 48 h p90 = a reset in progress
    regime_min_days: float = 3.0
    min_book_share: float = 0.70     # units now / normal units below this = partial book
    ah_cut: float = AH_CUT


def _tier_test(ok: bool, warn: bool = False) -> str:
    return "pass" if ok else ("warn" if warn else "fail")


class _LevelData:
    """The item's hourly rows as numpy arrays (spike hours and hours without a price left out), so the per-target questions are
    cheap: the screener asks them once per price gap, and the backtest asks them for every shelf in every hour."""

    def __init__(self, series: pd.DataFrame):
        f = series["floor_gold"].to_numpy(dtype="float64")
        keep = ~np.isnan(f)
        if "spike" in series:
            keep &= ~series["spike"].fillna(False).astype(bool).to_numpy()
        self.f = f[keep]
        self.sold = {k: series[f"sold_{k}"].to_numpy(dtype="float64")[keep] for k in ("low", "est", "high")}

    def stats(self, target_gold: float, tol: float) -> dict:
        if len(self.f) == 0:
            return dict(share=0.0, hours=0, rate=None)
        at = self.f >= target_gold * (1 - tol)
        got = at & ~np.isnan(self.sold["est"])
        n = int(got.sum())
        rate = None
        if n:
            rate = {}
            for k, v in self.sold.items():
                x = v[got]
                x = x[~np.isnan(x)]
                rate[k] = float(x.mean()) if len(x) else float("nan")
        return dict(share=float(at.mean()), hours=n, rate=rate)


def level_stats(series: pd.DataFrame, target_gold: float, tol: float) -> dict:
    """From the item's hourly rows (floor_gold, sold_low/est/high, spike): share of hours with the lowest price at or above
    the target, and the sales rate seen in those hours. The sold_* of an hour is the interval ending at it, so this is
    approximate at the edges."""
    return _LevelData(series).stats(target_gold, tol)


def long_run(series: pd.DataFrame) -> dict:
    """The item's longer price history from its hourly rows: median / p10 / p90 of the lowest price, the days covered, the
    median and p90 of the lowest price over the last 48 h, and the lowest price now. (The 48 h median, not the single highest
    hour, so an item that merely flips between two prices is not mistaken for a spike.)"""
    f = series["floor_gold"].to_numpy(dtype="float64")
    ok = ~np.isnan(f)
    nan = float("nan")
    if not ok.any():
        return dict(median=nan, p10=nan, p90=nan, days=0.0, recent_med=nan, recent_p90=nan, now=nan)
    ts = pd.to_datetime(series["ts"]).to_numpy()[ok]
    f = f[ok]
    if len(ts) > 1 and (ts[1:] < ts[:-1]).any():
        order = np.argsort(ts, kind="stable")
        ts, f = ts[order], f[order]
    recent = f[ts > ts[-1] - np.timedelta64(48, "h")]
    return dict(median=float(np.median(f)), p10=float(np.quantile(f, 0.10)), p90=float(np.quantile(f, 0.90)),
                days=float((ts[-1] - ts[0]) / np.timedelta64(1, "D")), recent_med=float(np.median(recent)),
                recent_p90=float(np.quantile(recent, 0.90)), now=float(f[-1]))


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
    lr = long_run(series)
    ld = _LevelData(series)
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
        lv = ld.stats(tg, cfg.level_tolerance)
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
        avg_g = avg / COPPER_PER_GOLD
        med = lr["median"]
        if med != med or lr["days"] < cfg.regime_min_days:
            t_regime = "warn"
        else:
            reset = lr["recent_med"] >= cfg.regime_spike_x * med and lr["now"] <= cfg.regime_falling * lr["recent_p90"]
            if avg_g > cfg.regime_cost_x * med or reset or tg > cfg.regime_target_fail_x * med:
                t_regime = "fail"
            else:
                t_regime = "warn" if tg > cfg.regime_target_warn_x * med else "pass"
        tests = dict(test_price=t_price, test_speed=t_speed, test_size=t_size, test_margin=t_margin, test_regime=t_regime, test_book=t_book)
        fails, warns = sum(v == "fail" for v in tests.values()), sum(v == "warn" for v in tests.values())
        verdict = "candidate" if fails == 0 and warns <= 1 else ("weak" if fails == 0 else "no")
        row = dict(
            units=int(bought), shelf_units=int(shelf_units), shelf_top_gold=float(price[k - 1]) / COPPER_PER_GOLD, cost_gold=cost / COPPER_PER_GOLD, avg_price_gold=avg / COPPER_PER_GOLD,
            target_gold=tg, gap=float(price[k] / price[k - 1] - 1.0), breakeven_gold=avg / (1 - cfg.ah_cut) / COPPER_PER_GOLD,
            net_margin=float(margin), net_margin_stress=float(margin_stress),
            profit_gold=float((net_target - avg) * bought / COPPER_PER_GOLD),
            profit_stress_gold=float((stress_price * (1 - cfg.ah_cut) - avg) * bought / COPPER_PER_GOLD),
            sell_hours_low=sell["low"], sell_hours_est=sell["est"], sell_hours_high=sell["high"],
            rate_from_level=bool(use_level), hours_at_level=lv["hours"], share_at_level=lv["share"], cheap_units_ahead=int(ahead),
            book_share=float(book_share), long_median_gold=med, long_p10_gold=lr["p10"], long_p90_gold=lr["p90"], long_days=lr["days"],
            recent_48h_median_gold=lr["recent_med"], cost_vs_long=(avg_g / med if med == med and med > 0 else float("nan")),
            target_vs_long=(tg / med if med == med and med > 0 else float("nan")), verdict=verdict, fails=fails, warns=warns, **tests)
        key = {"candidate": 2, "weak": 1, "no": 0}[verdict] * 1e9 + row["profit_stress_gold"]
        if key > best_key:
            best, best_key = row, key
    return best
