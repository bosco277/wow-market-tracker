import unittest

import pandas as pd

from helpers import T0, mk, minutes
from engine.ordercheck import tier_order_evidence
from engine.simulator import MarketSimulator, default_profiles


class TestHandMade(unittest.TestCase):
    def tier(self, qtys_after):
        """Five listings at one price, ids 1..5 (5 = newest). qtys_after: id -> quantity after (None = gone)."""
        rows = [(i, 7, 100, 1000, "LONG") for i in range(1, 6)]
        before = mk(rows, ts=minutes(0))
        after_rows = [(i, 7, q, 1000, "LONG") for i, q in qtys_after.items() if q is not None]
        return [before, mk(after_rows, ts=minutes(60))]

    def test_newest_bought_first_is_lifo(self):
        ev = tier_order_evidence(self.tier({1: 100, 2: 100, 3: 40, 4: None, 5: None}))
        self.assertEqual((ev.lifo_consistent, ev.fifo_consistent, ev.mixed), (1, 0, 0))

    def test_oldest_bought_first_is_fifo(self):
        ev = tier_order_evidence(self.tier({1: None, 2: None, 3: 40, 4: 100, 5: 100}))
        self.assertEqual((ev.lifo_consistent, ev.fifo_consistent, ev.mixed), (0, 1, 0))

    def test_scattered_disappearances_are_mixed(self):
        ev = tier_order_evidence(self.tier({1: None, 2: 100, 3: 40, 4: 100, 5: None}))
        self.assertEqual((ev.lifo_consistent, ev.fifo_consistent, ev.mixed), (0, 0, 1))

    def test_listings_that_may_have_expired_are_ignored(self):
        rows = [(i, 7, 100, 1000, "SHORT") for i in range(1, 6)]
        before = mk(rows, ts=minutes(0))
        after = mk([(3, 7, 40, 1000, "SHORT"), (4, 7, 100, 1000, "SHORT"), (5, 7, 100, 1000, "SHORT")], ts=minutes(60))
        self.assertEqual(tier_order_evidence([before, after]).tiers_examined, 0)

    def test_big_gaps_are_ignored(self):
        s = self.tier({1: 100, 2: 100, 3: 40, 4: None, 5: None})
        s[1].ts = minutes(600)
        self.assertEqual(tier_order_evidence(s).tiers_examined, 0)


class TestOnSimulatedMarkets(unittest.TestCase):
    def run_sim(self, **kw):
        # many listings per tier so there is something to compare
        res = MarketSimulator(default_profiles(), seed=4, **kw).run(96)
        return tier_order_evidence(res.snapshots)

    def test_identifies_each_rule(self):
        lifo = self.run_sim(lifo_within_tier=True)
        fifo = self.run_sim(fifo_within_tier=True)
        rand = self.run_sim(fifo_within_tier=False)
        for name, ev in (("lifo", lifo), ("fifo", fifo), ("rand", rand)):
            print(name, ev.describe())
        self.assertGreater(lifo.lifo_consistent, 3 * max(1, lifo.fifo_consistent))
        self.assertGreater(fifo.fifo_consistent, 3 * max(1, fifo.lifo_consistent))
        # a random order should not look strongly like either rule
        self.assertLess(rand.lifo_consistent, 0.6 * max(1, rand.lifo_consistent + rand.fifo_consistent + rand.mixed))
        self.assertLess(rand.fifo_consistent, 0.6 * max(1, rand.lifo_consistent + rand.fifo_consistent + rand.mixed))


if __name__ == "__main__":
    unittest.main()
