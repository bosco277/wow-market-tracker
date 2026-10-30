import unittest

import numpy as np
import pandas as pd

from helpers import T0  # noqa: F401  (sets sys.path)
from engine.schema import TimeLeftBuckets
from engine.simulator import ItemProfile, MarketSimulator, default_profiles


def small_run(seed=3, **kw):
    return MarketSimulator(default_profiles(), seed=seed).run(hours=24, **kw)


class TestSimulator(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        a, b = small_run(5), small_run(5)
        pd.testing.assert_frame_equal(a.ledger, b.ledger)
        self.assertEqual(len(a.snapshots), len(b.snapshots))
        c = small_run(6)
        self.assertFalse(a.ledger.equals(c.ledger))

    def test_ledger_explains_every_snapshot_quantity(self):
        res = small_run(2, burn_in_hours=24)
        led = res.ledger
        for snap in res.snapshots[::5]:
            upto = led[led["ts"] <= snap.ts]
            listed = upto[upto["type"] == "LIST"].set_index("auction_id")["qty"]
            removed = upto[upto["type"].isin(["SALE", "CANCEL", "EXPIRE"])].groupby("auction_id")["qty"].sum()
            remaining = listed.sub(removed.reindex(listed.index).fillna(0))
            live = snap.book.set_index("auction_id")["quantity"]
            # every live auction: listed minus removed equals what the snapshot shows
            self.assertTrue((remaining.reindex(live.index) == live).all())
            # every non-live auction has been fully removed
            dead = remaining.drop(live.index, errors="ignore")
            self.assertTrue((dead == 0).all())

    def test_snapshot_invariants(self):
        res = small_run(4, miss_prob=0.2)
        times = [s.ts for s in res.snapshots]
        self.assertEqual(times, sorted(set(times)))
        full = small_run(4)
        self.assertLess(len(res.snapshots), len(full.snapshots))
        for s in res.snapshots:
            self.assertTrue((s.book["unit_price"] > 0).all() and (s.book["quantity"] > 0).all())

    def test_time_left_values_are_valid_and_varied(self):
        sim = MarketSimulator(default_profiles(), seed=1)
        res = sim.run(hours=6, burn_in_hours=24)
        snap = res.snapshots[-1]
        self.assertTrue(set(snap.book["time_left"]) <= {"SHORT", "MEDIUM", "LONG", "VERY_LONG"})
        self.assertGreater(snap.book["time_left"].nunique(), 1)

    def test_truth_between_splits_observable_and_blind_spot(self):
        res = small_run(2, burn_in_hours=24)
        prev, curr = res.snapshots[3], res.snapshots[4]
        t = res.truth_between(prev, curr)
        self.assertTrue((t["sold_total"] >= t["sold_observable"]).all())
        self.assertGreater(t["sold_total"].sum(), 0)

    def test_buyers_sweep_cheapest_first(self):
        prof = ItemProfile(1, 1000, 0.0, 10, 0.0, 10)
        sim = MarketSimulator([prof], seed=0)
        book = sim._books[1]
        book[1] = [10, 900, 10_000]
        book[2] = [10, 950, 10_000]
        book[3] = [10, 1200, 10_000]
        sim._buy(prof, 0, 15, limit=2000)
        self.assertNotIn(1, book)                  # cheapest fully bought
        self.assertEqual(book[2][0], 5)            # next partially bought
        self.assertEqual(book[3][0], 10)           # most expensive untouched
        sales = [(e[2], e[4]) for e in sim._log if e[3] == "SALE"]
        self.assertEqual(sales, [(1, 10), (2, 5)])

    def test_buyer_limit_price_is_respected(self):
        prof = ItemProfile(1, 1000, 0.0, 10, 0.0, 10)
        sim = MarketSimulator([prof], seed=0)
        sim._books[1][1] = [10, 5000, 10_000]
        sim._buy(prof, 0, 10, limit=1500)
        self.assertEqual(sim._books[1][1][0], 10)  # too expensive: nothing bought


if __name__ == "__main__":
    unittest.main()
