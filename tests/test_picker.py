import unittest

import numpy as np
import pandas as pd

from helpers import T0  # noqa: F401
from engine.picker import PickerConfig, explain, item_stats, pick, publish_minutes, _round_units
from engine.simulator import MarketSimulator, default_profiles


def stats_row(item, daily_est=500.0, floor=10_000, cv=0.05, depth=5000, hours=40.0, cov=0.9):
    return dict(item_id=item, daily_low=daily_est * 0.7, daily_est=daily_est, daily_high=daily_est * 1.4,
                observed_hours=hours, coverage=cov, n_intervals=40, floor_median=floor, floor_std=floor * cv,
                floor_mean=floor, n_floor=40, price_cv=cv, floor_now=floor, depth_10pct=depth, n_listings=30,
                hourly_high=daily_est * 1.4 / 24, hourly_est=daily_est / 24)


class TestPick(unittest.TestCase):
    def test_filters_remove_the_right_items(self):
        s = pd.DataFrame([
            stats_row(1),                       # fine
            stats_row(2, daily_est=20),         # too slow
            stats_row(3, cv=0.5),               # jumpy price
            stats_row(4, floor=900_000_000),    # far too expensive
            stats_row(5, depth=10),             # book too thin
            stats_row(6, hours=3, cov=0.1),     # not enough data
            stats_row(7, daily_est=50_000),     # too fast
            stats_row(8, floor=100, daily_est=2000.0),   # 1 silver item: price too low (and would need thousands of units)
            stats_row(9, floor=10_000, daily_est=4000.0),  # fine price but would need more than 300 units
        ])
        got = pick(s)
        self.assertEqual(list(got["item_id"]), [1])
        why = explain(s)
        self.assertEqual((why["too_slow"], why["price_too_jumpy"], why["too_little_data"], why["too_fast"]), (1, 1, 1, 1))
        self.assertEqual((why["price_too_low"], why["too_many_units"]), (1, 2))  # items 7 and 9 both need > 300 units

    def test_position_cost_and_size_rules_hold_for_everything_picked(self):
        rng = np.random.default_rng(3)
        s = pd.DataFrame([stats_row(i, daily_est=float(rng.integers(100, 6000)), floor=int(rng.integers(1_000, 30_000)),
                                    cv=float(rng.random() * 0.14), depth=int(rng.integers(500, 90_000))) for i in range(200)])
        cfg = PickerConfig(max_position_gold=1000.0)
        got = pick(s, cfg)
        self.assertEqual(len(got), cfg.n_items)
        self.assertTrue((got["position_gold"] <= cfg.max_position_gold + 1e-9).all())
        self.assertTrue((got["buy_units"] >= cfg.min_units).all())
        self.assertTrue((got["depth_10pct"] >= cfg.min_depth_multiple * got["buy_units"]).all())
        self.assertTrue(got["item_id"].is_unique)
        self.assertTrue((got["back_lane_units_ahead"] > 0).all())
        self.assertEqual(set(got["speed"]), {"slow", "medium", "fast"})  # a spread, not eight of the same kind

    def test_buy_units_cover_about_two_normal_hours_of_sales(self):
        got = pick(pd.DataFrame([stats_row(1, daily_est=1200.0, floor=10_000)]))
        hourly_high = 1200.0 * 1.4 / 24
        self.assertGreaterEqual(got["buy_units"].iloc[0], 2 * hourly_high)
        self.assertLess(got["buy_units"].iloc[0], 2 * hourly_high + 50)

    def test_exclude_and_empty_inputs(self):
        s = pd.DataFrame([stats_row(1), stats_row(2)])
        self.assertEqual(list(pick(s, exclude=[1])["item_id"]), [2])
        self.assertTrue(pick(pd.DataFrame()).empty)
        self.assertTrue(pick(pd.DataFrame([stats_row(1, daily_est=1.0)])).empty)

    def test_round_units(self):
        self.assertEqual([_round_units(x) for x in (3, 11, 101, 1001)], [10, 20, 150, 1100])


class TestItemStats(unittest.TestCase):
    def test_runs_on_simulated_market(self):
        res = MarketSimulator(default_profiles(), seed=2).run(72)
        st = item_stats(res.snapshots)
        self.assertEqual(set(st["item_id"]), {1001, 1002, 1003, 1004})
        self.assertFalse(st[["daily_est", "price_cv", "floor_now", "depth_10pct"]].isna().any().any())
        self.assertTrue((st["daily_low"] <= st["daily_est"] + 1e-9).all())
        self.assertTrue((st["daily_est"] <= st["daily_high"] + 1e-9).all())
        # the very liquid item should rank faster than the thin one
        d = st.set_index("item_id")["daily_est"]
        self.assertGreater(d[1004], d[1003])

    def test_publish_minutes_ignores_snapshots_that_only_know_the_fetch_time(self):
        res = MarketSimulator(default_profiles(), seed=2).run(10)
        self.assertEqual(int(publish_minutes(res.snapshots).sum()), len(res.snapshots))
        for s in res.snapshots[:4]:
            s.meta["ts_source"] = "scan_timestamp"
        self.assertEqual(int(publish_minutes(res.snapshots).sum()), len(res.snapshots) - 4)


if __name__ == "__main__":
    unittest.main()
