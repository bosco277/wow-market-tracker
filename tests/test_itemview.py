import unittest

import numpy as np
import pandas as pd

from helpers import T0, mk, minutes
from engine.itemview import detect_flips, gaps_in, item_history, revert_share_within
from engine.simulator import MarketSimulator, default_profiles

LOW = [(i, 7, 10, 1000 + i, "LONG") for i in range(1, 11)]  # 10 listings x 10 units, 1001..1010 copper
HIGH = [(50, 7, 500, 3000, "LONG")]


def book(extra=(), low=LOW, t=0):
    return mk(list(low) + list(HIGH) + list(extra), ts=minutes(t))


def quiet(n, start=-8):
    """n unchanged hourly snapshots before the event: the item's 'normal' (no sales)."""
    return [book(t=60 * (start + j)) for j in range(n)]


class TestFlips(unittest.TestCase):
    def test_buyout_and_relist_is_detected_with_revert_and_volume(self):
        s0 = book(t=0)
        s1 = mk(HIGH + [(60, 7, 80, 2600, "LONG")], ts=minutes(60))       # cheap side gone; flipper relists at 2600
        s2 = mk(HIGH + [(60, 7, 40, 2600, "LONG")], ts=minutes(120))      # 40 of the flipper's units sold at the high price
        s3 = mk(HIGH + [(61, 7, 50, 1100, "LONG"), (60, 7, 40, 2600, "LONG")], ts=minutes(180))  # new cheap seller: price reverts
        ev = detect_flips(quiet(8) + [s0, s1, s2, s3])
        self.assertEqual(len(ev), 1)
        e = ev.iloc[0]
        self.assertEqual((e["item_id"], e["verdict"]), (7, "flip / buyout"))
        self.assertAlmostEqual(e["jump"], 2600 / 1001, places=3)
        self.assertEqual(e["units_removed"], 100)
        self.assertAlmostEqual(e["revert_hours"], 2.0)          # back near 1001 two hours after the flip
        self.assertEqual(e["ended_by"], "reverted")
        self.assertFalse(e["still_elevated_at_end"])
        self.assertTrue(e["likely_flip"])
        self.assertAlmostEqual(e["relisted_share"], 0.8)
        self.assertGreaterEqual(e["size_vs_normal"], 3)
        self.assertGreaterEqual(e["sold_est_while_elevated"], 40)

    def test_buyout_without_relisting_is_not_a_likely_flip(self):
        """Ordinary demand eating the cheap level looks the same, minus the relisting."""
        s0 = book(t=0)
        s1 = mk(HIGH, ts=minutes(60))
        ev = detect_flips(quiet(8) + [s0, s1])
        self.assertEqual(ev.iloc[0]["verdict"], "flip / buyout")
        self.assertFalse(ev.iloc[0]["likely_flip"])
        self.assertTrue(np.isnan(ev.iloc[0]["sold_est_while_elevated"]))

    def test_a_buyout_the_item_often_sees_is_not_a_likely_flip(self):
        """If the item normally sells about this much every hour, removing that much is not unusual."""
        busy = []
        for j in range(8):                      # every hour ~100 units go and 100 new ones arrive at the same price
            busy.append(mk(HIGH + [(100 + j, 7, 100, 1001, "LONG")], ts=minutes(60 * (j - 8))))
        s0 = book(t=0)
        s1 = mk(HIGH + [(60, 7, 80, 2600, "LONG")], ts=minutes(60))
        ev = detect_flips(busy + [s0, s1])
        self.assertFalse(ev.iloc[0]["likely_flip"])

    def test_a_hole_before_the_price_comes_back_means_unknown_not_a_number(self):
        s0 = book(t=0)
        s1 = mk(HIGH, ts=minutes(60))
        s2 = mk(HIGH, ts=minutes(120))
        s3 = book(t=60 * 30)                    # 28 hours later (a hole): the price is back near the old one
        ev = detect_flips(quiet(8) + [s0, s1, s2, s3])
        e = ev.iloc[0]
        self.assertEqual(e["ended_by"], "hole")
        self.assertTrue(np.isnan(e["revert_hours"]))
        self.assertFalse(e["still_elevated_at_end"])
        self.assertAlmostEqual(e["followup_hours"], 1.0)   # only the hour we really saw

    def test_revert_share_counts_only_events_watched_long_enough(self):
        ev = pd.DataFrame([
            dict(ended_by="reverted", revert_hours=2.0, followup_hours=2.0),     # back within 6 h
            dict(ended_by="reverted", revert_hours=9.0, followup_hours=9.0),     # back, but later than 6 h
            dict(ended_by="end_of_data", revert_hours=np.nan, followup_hours=7.0),   # watched 7 h, still up
            dict(ended_by="hole", revert_hours=np.nan, followup_hours=1.0),      # cut short: not counted
        ])
        share, n = revert_share_within(ev, 6.0)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(share, 1 / 3)

    def test_gaps_in_lists_the_holes(self):
        g = gaps_in([book(t=0), book(t=60), book(t=60 * 26), book(t=60 * 27)])
        self.assertEqual(len(g), 1)
        self.assertAlmostEqual(g.iloc[0]["hours"], 25.0)

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
