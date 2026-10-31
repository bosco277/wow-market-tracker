import unittest

from helpers import T0  # noqa: F401  (puts the engine on the path)

import numpy as np
import pandas as pd

from engine.shelves import ShelfConfig, level_stats, shelves_for_item

RATE = {"low": 20.0, "est": 30.0, "high": 40.0}


def book(tiers):
    rows = []
    for gold, n in tiers:
        rows += [(int(round(gold * 10000)), 1)] * n
    return pd.DataFrame(rows, columns=["unit_price", "quantity"]).assign(item_id=1)


def series(floors, units=1130, sold=(20.0, 30.0, 40.0)):
    n = len(floors)
    return pd.DataFrame({"ts": pd.date_range("2026-10-30", periods=n, freq="h"), "floor_gold": floors, "units_listed": units,
                         "sold_low": sold[0], "sold_est": sold[1], "sold_high": sold[2], "spike": False})


EX = book([(1.0, 30), (1.1, 100), (1.5, 1000)])
OFTEN = series(np.where(np.arange(72) % 3 == 0, 1.5, 1.0))


class TestShelves(unittest.TestCase):
    def test_the_example_in_the_plan(self):
        r = shelves_for_item(EX, OFTEN, RATE, 1.5)
        self.assertEqual((r["units"], r["shelf_units"], r["cheap_units_ahead"]), (130, 130, 0))
        self.assertAlmostEqual(r["avg_price_gold"], 140 / 130, places=6)
        self.assertAlmostEqual(r["net_margin"], 1.5 * 0.95 / (140 / 130) - 1, places=6)
        self.assertAlmostEqual(r["breakeven_gold"], 140 / 130 / 0.95, places=6)
        self.assertLess(r["net_margin_stress"], r["net_margin"])
        self.assertEqual(r["verdict"], "candidate")

    def test_no_gap_no_shelf(self):
        self.assertIsNone(shelves_for_item(book([(1.0, 50), (1.05, 50), (1.08, 50)]), OFTEN, RATE, 1.5))
        self.assertIsNone(shelves_for_item(book([(1.0, 50)]), OFTEN, RATE, 1.5))

    def test_unit_cap_buys_the_cheapest_and_leaves_cheap_units_ahead(self):
        r = shelves_for_item(EX, OFTEN, RATE, 1.5, ShelfConfig(max_units=40))
        self.assertEqual((r["units"], r["shelf_units"], r["cheap_units_ahead"]), (40, 130, 90))
        self.assertAlmostEqual(r["avg_price_gold"], (30 * 1.0 + 10 * 1.1) / 40, places=6)
        full = shelves_for_item(EX, OFTEN, RATE, 1.5)
        self.assertGreater(r["sell_hours_est"], 0)
        self.assertEqual(r["test_size"], "fail")          # under half the shelf
        self.assertEqual(r["verdict"], "no")
        self.assertAlmostEqual(r["sell_hours_est"], (90 + 40) / 30.0)
        self.assertAlmostEqual(full["sell_hours_est"], 130 / 30.0)

    def test_gold_cap(self):
        r = shelves_for_item(EX, OFTEN, RATE, 1.5, ShelfConfig(max_cost_gold=35.0))
        self.assertEqual(r["units"], 34)                  # 30 x 1.00g = 30g, then 4 x 1.10g = 4.4g; a fifth would pass 35g
        self.assertLessEqual(r["cost_gold"], 35.0 + 1e-9)

    def test_price_never_at_the_level_fails(self):
        r = shelves_for_item(EX, series([1.0] * 72), RATE, 1.2)
        self.assertEqual(r["test_price"], "fail")
        self.assertEqual(r["verdict"], "no")

    def test_target_inside_band_but_rare_is_only_a_warning(self):
        r = shelves_for_item(EX, series([1.0] * 72), RATE, 1.5)
        self.assertEqual(r["test_price"], "warn")

    def test_slow_item_fails_the_speed_test(self):
        slow = series(np.where(np.arange(72) % 3 == 0, 1.5, 1.0), sold=(0.5, 1.0, 2.0))
        r = shelves_for_item(EX, slow, {"low": 0.5, "est": 1.0, "high": 2.0}, 1.5)
        self.assertEqual(r["test_speed"], "fail")

    def test_rate_falls_back_to_overall_with_a_warning_when_never_seen_at_level(self):
        r = shelves_for_item(EX, series([1.0] * 72), RATE, 1.5)
        self.assertFalse(r["rate_from_level"])
        self.assertIn(r["test_speed"], ("warn", "fail"))

    def test_partial_book_fails_the_book_test(self):
        r = shelves_for_item(EX, series(np.where(np.arange(72) % 3 == 0, 1.5, 1.0), units=5000), RATE, 1.5)
        self.assertEqual(r["test_book"], "fail")

    def test_thin_margin_is_dropped(self):
        self.assertIsNone(shelves_for_item(book([(1.0, 50), (1.16, 500)]), OFTEN, RATE, 1.5, ShelfConfig(min_net_margin=0.20)))

    def test_level_stats_ignores_spikes(self):
        s = series([1.5, 1.5, 1.0, 1.0])
        s.loc[0, "spike"] = True
        self.assertAlmostEqual(level_stats(s, 1.5, 0.02)["share"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
