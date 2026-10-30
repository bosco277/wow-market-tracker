import dataclasses
import unittest

import pandas as pd

from helpers import T0, mk, minutes
from engine.backtest import (
    BacktestConfig, DipBuyer, Order, run_backtest, run_backtest_range, summarize_trades,
)
from engine.schema import Snapshot
from engine.simulator import MarketSimulator, default_profiles


class BuyOnce:
    def __init__(self, item=1, units=100):
        self.item, self.units, self.done = item, units, False

    def decide(self, ctx):
        if self.done:
            return []
        self.done = True
        return [Order(self.item, self.units)]


class BuyWhenFree:
    """Rate-independent strategy: same entries under every sales assumption."""
    def __init__(self, item, units):
        self.item, self.units = item, units

    def decide(self, ctx):
        return [Order(self.item, self.units)]


class TestKnownPnL(unittest.TestCase):
    def snapshots(self):
        s0 = mk([(1, 1, 100, 1000, "LONG"), (2, 1, 100, 1100, "LONG")], ts=minutes(0))
        s1 = mk([(2, 1, 100, 1100, "LONG"), (3, 1, 100, 1500, "LONG")], ts=minutes(60))  # A sold out, a pricier listing arrived
        s2 = mk([(2, 1, 100, 1100, "LONG"), (3, 1, 100, 1500, "LONG")], ts=minutes(120))
        return [s0, s1, s2]

    def test_hand_computed_round_trip(self):
        cfg = BacktestConfig(warmup_hours=0, hold_hours=1, exit_horizon_hours=12, sales_assumption="est")
        res = run_backtest(self.snapshots(), BuyOnce(1, 100), cfg)
        t = res.trades.iloc[0]
        self.assertEqual(len(res.trades), 1)
        self.assertEqual(t["cost"], 100 * 1000)                  # swept the 1000-copper listing
        # classifier: A (100 units) vanished from before the first untouched survivor and cannot have expired
        # -> 100 sold in 1h -> 2400/day -> 1200 expected over 12h; room for 100 more units behind 1100's tier
        self.assertEqual(t["list_price"], 1500)
        self.assertEqual(t["filled_units"], 100)
        self.assertAlmostEqual(t["proceeds"], 100 * 1500 * 0.95)
        self.assertAlmostEqual(t["pnl"], 100 * 1500 * 0.95 - 100_000)
        self.assertAlmostEqual(res.summary["roi_on_cost"], 0.425)

    def test_unsold_units_are_haircut_not_ignored(self):
        cfg = BacktestConfig(warmup_hours=0, hold_hours=1, exit_horizon_hours=12, sales_assumption="low")
        s = self.snapshots()
        # make the market absorb nothing: A survives, so no sales are observed
        flat = [mk([(1, 1, 100, 1000, "LONG"), (2, 1, 100, 1100, "LONG")], ts=minutes(m)) for m in (0, 60, 120)]
        res = run_backtest(flat, BuyOnce(1, 50), cfg)
        t = res.trades.iloc[0]
        self.assertEqual(t["filled_units"], 0)
        self.assertAlmostEqual(t["proceeds"], 50 * 1000 * 0.95 * cfg.unfilled_haircut)
        self.assertLess(t["pnl"], 0)


class TestSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = MarketSimulator(default_profiles(), seed=11).run(72)

    def test_context_contains_no_future_data(self):
        seen = []

        class Spy:
            def decide(self, ctx):
                assert all(s.ts <= ctx.now for s in ctx.history)
                assert ctx.history[-1].ts == ctx.now
                assert len(ctx.intervals) == 0 or ctx.intervals["t1"].max() <= ctx.now
                assert ctx.metrics_history["ts"].max() <= ctx.now
                seen.append(ctx.now)
                return []

        run_backtest(self.res.snapshots, Spy(), BacktestConfig(warmup_hours=2))
        self.assertGreater(len(seen), 10)

    def test_future_data_cannot_change_earlier_decisions(self):
        snaps = self.res.snapshots
        m = 40  # decisions at index <= m must not depend on anything after m

        def perturb(s: Snapshot) -> Snapshot:
            b = s.book.copy()
            b["unit_price"] = (b["unit_price"] * 7 + 3).astype("int64")
            b["quantity"] = (b["quantity"] // 2 + 1).astype("int64")
            return Snapshot(s.source, s.ts, b)

        altered = snaps[: m + 1] + [perturb(s) for s in snaps[m + 1:]]
        cfg = BacktestConfig(warmup_hours=24)
        a = run_backtest(snaps, DipBuyer(dip=1.0), cfg).orders_log
        b = run_backtest(altered, DipBuyer(dip=1.0), cfg).orders_log
        early_a = [o for o in a if o[0] <= m]
        early_b = [o for o in b if o[0] <= m]
        self.assertGreater(len(early_a), 0, "test is only meaningful if the strategy traded before the cut")
        self.assertEqual(early_a, early_b)

    def test_cash_is_never_overspent(self):
        # Profits are reinvested, so a trade may cost more than the STARTING bankroll. What must hold is that every
        # entry fits the cash on hand at that moment. Replay the cash flows and check.
        cfg = BacktestConfig(bankroll=2_000_000.0, warmup_hours=2, hold_hours=3)
        res = run_backtest(self.res.snapshots, BuyWhenFree(1001, 10_000_000), cfg)
        t = res.trades
        self.assertGreater(len(t), 0)
        events = []
        for _, row in t.iterrows():
            events.append((row["entry_ts"], 1, row["cost"], 0.0))
            events.append((row["exit_ts"], 0, 0.0, row["proceeds"]))
        events.sort(key=lambda e: (e[0], e[1]))  # exits are processed before entries at the same timestamp
        cash = cfg.bankroll
        for ts, kind, cost, proceeds in events:
            if kind == 1:
                self.assertLessEqual(cost, cash + 1e-6, f"entry at {ts} costs {cost} but only {cash} was available")
                cash -= cost
            else:
                cash += proceeds
            self.assertGreaterEqual(cash, -1e-6)
        # and a tiny bankroll genuinely limits position size
        small = run_backtest(self.res.snapshots, BuyWhenFree(1001, 10_000_000), BacktestConfig(bankroll=50_000.0, warmup_hours=2, hold_hours=3))
        self.assertLessEqual(small.trades["cost"].iloc[0], 50_000.0)

    def test_one_open_position_per_item(self):
        cfg = BacktestConfig(warmup_hours=2, hold_hours=5)
        res = run_backtest(self.res.snapshots, BuyWhenFree(1002, 5), cfg)
        t = res.trades.sort_values("entry_ts")
        for (_, a), (_, b) in zip(t.iloc[:-1].iterrows(), t.iloc[1:].iterrows()):
            self.assertGreaterEqual(b["entry_ts"], a["exit_ts"])

    def test_pnl_range_is_ordered_low_to_high(self):
        cfg = BacktestConfig(warmup_hours=24, hold_hours=4)
        r = run_backtest_range(self.res.snapshots, lambda: BuyWhenFree(1002, 15), cfg)
        pnl = {k: v.summary["total_pnl"] for k, v in r.items()}
        self.assertLessEqual(pnl["low"], pnl["est"] + 1e-6)
        self.assertLessEqual(pnl["est"], pnl["high"] + 1e-6)
        self.assertEqual({v.summary["n_trades"] for v in r.values()}.__len__(), 1)  # same entries under every assumption


class TestTierOrder(unittest.TestCase):
    def test_newest_first_never_lists_lower_than_oldest_first_for_the_same_entries(self):
        res = MarketSimulator(default_profiles(), seed=11).run(72)
        out = {}
        for rule in ("fifo", "lifo"):
            cfg = BacktestConfig(warmup_hours=24, hold_hours=4, tier_order=rule)
            out[rule] = run_backtest(res.snapshots, BuyWhenFree(1002, 5), cfg).trades.sort_values("entry_ts").reset_index(drop=True)
        self.assertEqual(len(out["fifo"]), len(out["lifo"]))
        self.assertTrue((out["fifo"]["cost"] == out["lifo"]["cost"]).all())  # identical entries
        self.assertTrue((out["lifo"]["list_price"] >= out["fifo"]["list_price"] - 1e-9).all())


class TestSummary(unittest.TestCase):
    def test_drawdown_and_hit_rate(self):
        trades = pd.DataFrame({
            "exit_ts": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"]),
            "pnl": [100.0, -40.0, -30.0, 50.0], "cost": [1000.0] * 4, "roi": [0.1, -0.04, -0.03, 0.05],
            "filled_units": [10, 10, 10, 10], "units": [10, 10, 10, 10],
        })
        s = summarize_trades(trades, BacktestConfig(bankroll=10_000.0))
        self.assertEqual(s["n_trades"], 4)
        self.assertAlmostEqual(s["total_pnl"], 80.0)
        self.assertAlmostEqual(s["hit_rate"], 0.5)
        self.assertAlmostEqual(s["max_drawdown"], 70.0)  # peak 100 -> trough 30
        self.assertAlmostEqual(s["max_drawdown_pct_of_bankroll"], 0.007)

    def test_empty_trade_list(self):
        s = summarize_trades(pd.DataFrame(columns=["exit_ts", "pnl", "cost", "roi", "filled_units", "units"]), BacktestConfig())
        self.assertEqual(s["n_trades"], 0)


if __name__ == "__main__":
    unittest.main()
