import unittest

import numpy as np
import pandas as pd

from helpers import T0, mk, minutes
from engine.itemview import detect_flips, item_history
from engine.simulator import MarketSimulator, default_profiles

LOW = [(i, 7, 10, 1000 + i, "LONG") for i in range(1, 11)]  # 10 listings x 10 units, 1001..1010 copper
HIGH = [(50, 7, 500, 3000, "LONG")]


def book(extra=(), low=LOW, t=0):
    return mk(list(low) + list(HIGH) + list(extra), ts=minutes(t))


class TestFlips(unittest.TestCase):
    def test_buyout_and_relist_is_detected_with_revert_and_volume(self):
        s0 = book(t=0)
        s1 = mk(HIGH + [(60, 7, 80, 2600, "LONG")], ts=minutes(60))       # cheap side gone; flipper relists at 2600
        s2 = mk(HIGH + [(60, 7, 40, 2600, "LONG")], ts=minutes(120))      # 40 of the flipper's units sold at the high price
        s3 = mk(HIGH + [(61, 7, 50, 1100, "LONG"), (60, 7, 40, 2600, "LONG")], ts=minutes(180))  # new cheap seller: price reverts
        ev = detect_flips([s0, s1, s2, s3])
        self.assertEqual(len(ev), 1)
        e = ev.iloc[0]
        self.assertEqual((e["item_id"], e["verdict"]), (7, "flip / buyout"))
        self.assertAlmostEqual(e["jump"], 2600 / 1001, places=3)
        self.assertEqual(e["units_removed"], 100)
        self.assertAlmostEqual(e["revert_hours"], 2.0)          # back near 1001 two hours after the flip
        self.assertFalse(e["still_elevated_at_end"])
        self.assertGreaterEqual(e["sold_est_while_elevated"], 40)

    def test_listings_that_could_have_expired_are_not_called_a_flip(self):
        short = [(i, 7, 10, 1000 + i, "SHORT") for i in range(1, 11)]
        s0 = book(low=short, t=0)
        s1 = mk(HIGH, ts=minutes(60))
        ev = detect_flips([s0, s1])
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev.iloc[0]["verdict"], "probably expiry or cancels")

    def test_price_that_stays_up_is_reported_as_still_elevated(self):
        s0 = book(t=0)
        s1 = mk(HIGH, ts=minutes(60))
        s2 = mk(HIGH, ts=minutes(120))
        ev = detect_flips([s0, s1, s2])
        self.assertTrue(ev.iloc[0]["still_elevated_at_end"])
        self.assertTrue(np.isnan(ev.iloc[0]["revert_hours"]))

    def test_small_moves_and_partial_removals_are_ignored(self):
        s0 = book(t=0)
        s1 = book(low=LOW[:5] + [(i, 7, 10, 1000 + i, "LONG") for i in range(6, 11)], t=60)   # nothing happened
        s2 = book(low=LOW[:2], t=120)   # floor unchanged (cheapest listings remain)
        self.assertTrue(detect_flips([s0, s1, s2]).empty)

    def test_big_gaps_are_not_judged(self):
        self.assertTrue(detect_flips([book(t=0), mk(HIGH, ts=minutes(600))]).empty)


class TestHistory(unittest.TestCase):
    def test_one_row_per_snapshot_with_bounds(self):
        res = MarketSimulator(default_profiles(), seed=3).run(24)
        h = item_history(res.snapshots, 1002)
        self.assertEqual(len(h), len(res.snapshots))
        self.assertTrue((h["sold_low"].dropna() <= h["sold_est"].dropna() + 1e-9).all())
        self.assertTrue((h["units_at_floor"] <= h["units_listed"]).all())
        self.assertTrue(item_history(res.snapshots, 999999).empty)


if __name__ == "__main__":
    unittest.main()
