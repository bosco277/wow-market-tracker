import unittest

import pandas as pd

from helpers import mk, minutes
from engine.classifier import ClassifierParams, aggregate_rates, classify_transitions


def one(result, item=1):
    return result.items[result.items["item_id"] == item].iloc[0]


class TestSweepLogic(unittest.TestCase):
    def test_sweep_with_partial_fill(self):
        prev = mk([(1, 1, 100, 10, "LONG"), (2, 1, 100, 11, "LONG"), (3, 1, 100, 12, "LONG"), (4, 1, 100, 13, "LONG")])
        curr = mk([(2, 1, 40, 11, "LONG"), (3, 1, 100, 12, "LONG"), (4, 1, 100, 13, "LONG")], ts=minutes(60))
        r = one(classify_transitions(prev, curr))
        self.assertEqual(r["sold_cert"], 60)           # B went 100 -> 40: certainly bought
        self.assertEqual(r["sold_high"], 160)          # + A vanished from before the first untouched survivor
        self.assertEqual(r["sold_low"], 160)           # LONG cannot expire within 60 min, no undercut relists
        self.assertEqual(r["sold_est"], 160)
        self.assertLessEqual(r["sold_low"], r["sold_est"])
        self.assertLessEqual(r["sold_est"], r["sold_high"])

    def test_disappearance_beyond_the_sweep_is_never_a_sale(self):
        prev = mk([(1, 1, 100, 10, "LONG"), (2, 1, 100, 11, "LONG"), (3, 1, 100, 12, "LONG")])
        res = classify_transitions(prev, mk([(1, 1, 100, 10, "LONG"), (2, 1, 100, 11, "LONG")], ts=minutes(60)))
        r = one(res)
        self.assertEqual((r["sold_low"], r["sold_est"], r["sold_high"]), (0, 0, 0))
        self.assertEqual(list(res.events["event"]), ["VANISHED_BEYOND_SWEEP"])

    def test_expiry_feasibility_separates_low_from_high(self):
        prev = mk([(1, 1, 50, 10, "SHORT"), (2, 1, 100, 11, "LONG")])
        curr = mk([(2, 1, 100, 11, "LONG")], ts=minutes(60))
        r = one(classify_transitions(prev, curr))
        self.assertEqual(r["sold_low"], 0)    # a SHORT listing is certainly gone after 60 minutes: says nothing
        self.assertEqual(r["sold_est"], 0)
        self.assertEqual(r["sold_high"], 50)  # but it may have been bought

    def test_undercut_relist_allowance_lowers_sold_low(self):
        prev = mk([(1, 1, 100, 10, "LONG"), (2, 1, 100, 11, "LONG")])
        curr = mk([(2, 1, 100, 11, "LONG"), (9, 1, 100, 9, "LONG")], ts=minutes(60))  # same volume reappears cheaper
        r = one(classify_transitions(prev, curr))
        self.assertEqual(r["sold_low"], 0)
        self.assertEqual(r["sold_high"], 100)
        r0 = one(classify_transitions(prev, curr, ClassifierParams(relist_attribution=0.0)))
        self.assertEqual(r0["sold_low"], 100)

    def test_tier_boundary_depends_on_fifo_assumption(self):
        prev = mk([(1, 1, 100, 10, "LONG"), (2, 1, 100, 10, "LONG"), (3, 1, 100, 12, "LONG")])
        curr = mk([(1, 1, 100, 10, "LONG"), (3, 1, 100, 12, "LONG")], ts=minutes(60))  # later listing in the tier vanished
        fifo = one(classify_transitions(prev, curr, ClassifierParams(fifo_within_tier=True)))
        tier = one(classify_transitions(prev, curr, ClassifierParams(fifo_within_tier=False)))
        self.assertEqual(fifo["sold_high"], 0)   # FIFO: the earlier listing survived, so the later one was not bought
        self.assertEqual(tier["sold_high"], 100)  # no order assumed: it might have been bought
        self.assertEqual(tier["sold_low"], 0)

    def test_everything_gone_is_flagged(self):
        prev = mk([(1, 1, 600, 10, "LONG"), (2, 1, 600, 11, "LONG")])
        curr = mk([(77, 2, 5, 10, "LONG")], ts=minutes(60))
        res = classify_transitions(prev, curr)
        r = one(res)
        self.assertTrue(r["flag_wipe"])
        self.assertEqual(len(aggregate_rates(res.items, minutes(60))), 0)  # excluded from rates

    def test_quantity_increase_is_an_anomaly_not_a_sale(self):
        prev = mk([(1, 1, 100, 10, "LONG")])
        curr = mk([(1, 1, 150, 10, "LONG")], ts=minutes(60))
        res = classify_transitions(prev, curr)
        r = one(res)
        self.assertEqual(r["anomaly_increases"], 1)
        self.assertEqual(r["sold_high"], 0)
        self.assertEqual(list(res.events["event"]), ["QUANTITY_INCREASED"])

    def test_long_gap_flag(self):
        prev = mk([(1, 1, 10, 10, "LONG")])
        curr = mk([(1, 1, 10, 10, "LONG")], ts=minutes(300))
        self.assertTrue(one(classify_transitions(prev, curr))["flag_long_gap"])

    def test_scan_coverage_limits_scope(self):
        prev = mk([(1, 1, 10, 10, "LONG"), (2, 2, 10, 10, "LONG")], covered_items=[1, 2])
        curr = mk([(1, 1, 10, 10, "LONG")], ts=minutes(60), covered_items=[1])  # item 2 was not scanned this time
        res = classify_transitions(prev, curr)
        self.assertEqual(list(res.items["item_id"]), [1])  # nothing is inferred about item 2

    def test_guards(self):
        a = mk([(1, 1, 10, 10, "LONG")])
        with self.assertRaises(ValueError):
            classify_transitions(a, mk([(1, 1, 10, 10, "LONG")], ts=minutes(60), source="other"))
        with self.assertRaises(ValueError):
            classify_transitions(a, mk([(1, 1, 10, 10, "LONG")], ts=a.ts))
        with self.assertRaises(NotImplementedError):
            classify_transitions(
                mk([(1, 1, 1, 10, "LONG")], kind="auction"), mk([(1, 1, 1, 10, "LONG")], ts=minutes(60), kind="auction")
            )

    def test_bounds_are_ordered_on_random_books(self):
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(3, 30))
            prices = rng.integers(5, 20, n)
            prev_rows = [(i + 1, 1, int(rng.integers(1, 200)), int(prices[i]), str(rng.choice(["SHORT", "MEDIUM", "LONG", "VERY_LONG"]))) for i in range(n)]
            keep = [r for r in prev_rows if rng.random() < 0.6]
            curr_rows = [(a, i, max(1, int(q * rng.uniform(0.3, 1.0))), p, t) for a, i, q, p, t in keep]
            new = [(1000 + j, 1, int(rng.integers(1, 100)), int(rng.integers(3, 20)), "LONG") for j in range(int(rng.integers(0, 4)))]
            if not curr_rows and not new:
                continue
            res = classify_transitions(mk(prev_rows), mk(curr_rows + new, ts=minutes(int(rng.integers(20, 200)))))
            for _, row in res.items.iterrows():
                self.assertLessEqual(row["sold_cert"], row["sold_low"] + 1e-9)
                self.assertLessEqual(row["sold_low"], row["sold_est"] + 1e-9)
                self.assertLessEqual(row["sold_est"], row["sold_high"] + 1e-9)
                self.assertGreaterEqual(row["sold_cert"], 0)
                self.assertLessEqual(row["sold_high"], row["prev_units"] + 1e-9)


class TestAggregateRates(unittest.TestCase):
    def test_rates_use_observed_hours_only(self):
        s = [mk([(1, 1, 100, 10, "LONG"), (2, 1, 100, 11, "LONG")], ts=minutes(0))]
        s.append(mk([(2, 1, 100, 11, "LONG")], ts=minutes(60)))            # A (100 units) bought in hour 1
        s.append(mk([(2, 1, 100, 11, "LONG")], ts=minutes(120)))           # nothing in hour 2
        frames = [classify_transitions(a, b).items for a, b in zip(s[:-1], s[1:])]
        rates = aggregate_rates(pd.concat(frames, ignore_index=True), minutes(120), window_hours=24)
        row = rates.iloc[0]
        self.assertAlmostEqual(row["observed_hours"], 2.0)
        self.assertAlmostEqual(row["daily_high"], 100 / 2.0 * 24)
        self.assertAlmostEqual(row["coverage"], 2.0 / 24.0)



class TestTierOrder(unittest.TestCase):
    """Five 100-unit listings at one price (ids 1..5, 5 newest). Buyers take 250 units."""
    def books(self, survivors):
        before = mk([(i, 7, 100, 1000, "LONG") for i in range(1, 6)], ts=minutes(0))
        after = mk(survivors, ts=minutes(60))
        return before, after

    def test_lifo_knows_the_newest_were_the_ones_bought(self):
        before, after = self.books([(1, 7, 100, 1000, "LONG"), (2, 7, 100, 1000, "LONG"), (3, 7, 50, 1000, "LONG")])
        r = classify_transitions(before, after, ClassifierParams(tier_order="lifo")).items.iloc[0]
        self.assertEqual(r["sold_cert"], 50)
        self.assertEqual(r["sold_high"], 250)  # 50 partial + listings 4 and 5

    def test_unknown_order_cannot_tell_but_still_brackets_the_truth(self):
        before, after = self.books([(1, 7, 100, 1000, "LONG"), (2, 7, 100, 1000, "LONG"), (3, 7, 50, 1000, "LONG")])
        r = classify_transitions(before, after, ClassifierParams()).items.iloc[0]
        self.assertLessEqual(r["sold_low"], 250)
        self.assertGreaterEqual(r["sold_high"], 250)

    def test_a_wrong_fifo_assumption_is_visible_as_prefix_violations(self):
        before, after = self.books([(1, 7, 100, 1000, "LONG"), (2, 7, 100, 1000, "LONG"), (3, 7, 50, 1000, "LONG")])
        r = classify_transitions(before, after, ClassifierParams(tier_order="fifo")).items.iloc[0]
        self.assertGreaterEqual(r["prefix_violations"], 1)  # survivors 1 and 2 are older than the "sold" ones

    def test_bad_value_is_rejected(self):
        before, after = self.books([(1, 7, 100, 1000, "LONG")])
        with self.assertRaises(ValueError):
            classify_transitions(before, after, ClassifierParams(tier_order="newest"))


if __name__ == "__main__":
    unittest.main()
