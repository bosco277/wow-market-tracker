import math
import unittest

import numpy as np
import pandas as pd

from helpers import T0  # noqa: F401
from engine.market import (find_shelf, flag_spikes, item_series, price_bands, price_wall, profile, queue_wait_hours,
                           reach_stats, tsm_style_market_value, wall_report)
from engine.simulator import MarketSimulator, default_profiles

G = 10_000  # copper per gold
S = 100     # copper per silver


def book(rows):
    """rows: (price_copper, quantity) -> one listing per row"""
    return pd.DataFrame({"auction_id": range(len(rows)), "item_id": 1, "unit_price": [r[0] for r in rows],
                         "quantity": [r[1] for r in rows], "time_left": "LONG", "variant_key": "1"})


class TestSpikes(unittest.TestCase):
    def test_short_spike_and_dip_are_flagged_but_a_new_level_is_not(self):
        f = pd.Series([100.0] * 60)
        f.iloc[20:25] = 400.0       # 5-hour spike
        f.iloc[40:42] = 30.0        # 2-hour dip
        got = flag_spikes(f)
        self.assertEqual(list(got.iloc[20:25]), [1] * 5)
        self.assertEqual(list(got.iloc[40:42]), [-1] * 2)
        self.assertEqual(int((got != 0).sum()), 7)
        shift = pd.Series([100.0] * 40 + [300.0] * 40)       # a lasting move is a new level, not a spike
        self.assertEqual(int((flag_spikes(shift) != 0).sum()), 0)

    def test_bands_leave_out_spikes_and_place_now(self):
        f = pd.Series([100.0] * 60)
        f.iloc[20:25] = 400.0
        df = pd.DataFrame({"floor_gold": f})
        b = price_bands(df, flag_spikes(f))
        self.assertAlmostEqual(b["p50"], 100.0)
        self.assertAlmostEqual(b["p90"], 100.0)           # the spike does not drag the upper band
        self.assertEqual(b["n_spike_hours"], 5)
        self.assertAlmostEqual(b["now_percentile"], 1.0)


class TestTsmStyleValue(unittest.TestCase):
    def test_cheapest_30_percent_cut_at_a_jump(self):
        rows = [(100, 1)] * 10 + [(110, 1)] * 10 + [(1000, 1)] * 80
        self.assertAlmostEqual(tsm_style_market_value(book(rows)), 105.0)

    def test_empty_and_single(self):
        self.assertTrue(math.isnan(tsm_style_market_value(book([]))))
        self.assertEqual(tsm_style_market_value(book([(500, 3)])), 500.0)


class TestWall(unittest.TestCase):
    def setUp(self):
        self.b = book([(100, 50), (110, 100), (200, 1000)])

    def test_wall_is_the_price_where_cumulative_units_reach_the_target(self):
        w = price_wall(self.b, 120)
        self.assertEqual((w["price"], w["units_at_or_below"]), (110, 150))
        self.assertAlmostEqual(w["vwap"], (100 * 50 + 110 * 70) / 120)
        self.assertFalse(w["exhausted"])
        self.assertTrue(price_wall(self.b, 10_000)["exhausted"])
        self.assertEqual(price_wall(self.b, 10)["price"], 100)

    def test_queue_wait_newest_first_counts_only_cheaper_units(self):
        self.assertAlmostEqual(queue_wait_hours(self.b, 110, 50.0), 1.0)                 # 50 units cheaper
        self.assertAlmostEqual(queue_wait_hours(self.b, 110, 50.0, tier_order="fifo"), 3.0)  # + the 100 at the same price
        self.assertAlmostEqual(queue_wait_hours(self.b, 110, 50.0, inflow_cheaper_per_hour=25.0), 2.0)
        self.assertEqual(queue_wait_hours(self.b, 110, 50.0, inflow_cheaper_per_hour=50.0), float("inf"))
        self.assertEqual(queue_wait_hours(self.b, 90, 50.0), 0.0)                       # below the floor: you are first

    def test_deep_market_where_velocity_does_not_matter_like_linen_cloth(self):
        linen = book([(6 * S, 559), (7 * S, 9613), (8 * S, 8598), (9 * S, 305_642), (10 * S, 1242)])
        r = wall_report(linen, dict(low=2449.0, est=2449.0, high=34_481.0), horizon_hours=4.0, trade_units=500)
        self.assertEqual(r["wall_est"], 7 * S)
        self.assertTrue(r["deep_market"])
        self.assertFalse(r["velocity_matters"])
        self.assertLess(r["wait_low"], 12.0)
        self.assertLessEqual(r["wait_high"], r["wait_est"])

    def test_velocity_matters_when_the_low_rate_gives_a_long_wait(self):
        thin = book([(1000, 100), (2000, 1000)])
        r = wall_report(thin, dict(low=5.0, est=100.0, high=300.0), horizon_hours=4.0, trade_units=500)
        self.assertEqual(r["wall_est"], 2000)
        self.assertAlmostEqual(r["wait_low"], 20.0)
        self.assertTrue(r["velocity_matters"])
        self.assertFalse(r["deep_market"])


class TestReach(unittest.TestCase):
    def series(self, floors, step_h=1.0, gap_after=None, gap_h=10):
        ts, t = [], pd.Timestamp("2026-10-27 00:00")
        for i, _ in enumerate(floors):
            ts.append(t)
            t += pd.Timedelta(hours=step_h + (gap_h if gap_after == i else 0))
        return pd.DataFrame({"ts": ts, "floor_gold": floors})

    def test_share_and_time_to_reach(self):
        df = self.series([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        r = reach_stats(df, 2.0, horizon_hours=3)
        # starts below 2.0: hours 0,1,4,5,6 -> reach times 2,1,3,2,1 (all within 3 h)
        self.assertEqual(r["n"], 5)
        self.assertAlmostEqual(r["share_reached"], 1.0)
        self.assertAlmostEqual(r["median_hours"], 2.0)

    def test_never_reached_counts_only_when_watched_long_enough(self):
        df = self.series([1.0] * 10)
        r = reach_stats(df, 2.0, horizon_hours=3)
        self.assertEqual(r["n"], 7)                 # starts at hours 0-6 were watched a full 3 hours; hours 7-9 were not
        self.assertEqual(r["share_reached"], 0.0)

    def test_windows_across_a_hole_are_skipped(self):
        df = self.series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], gap_after=2)
        r = reach_stats(df, 2.0, horizon_hours=5)
        self.assertEqual(r["n"], 0)


class TestShelf(unittest.TestCase):
    def test_the_eternal_crystal_shelf_is_found(self):
        b = book([(99 * S, 43), (100 * S, 47), (173 * S, 6), (175 * S, 269), (209 * S, 64), (211 * S, 997)])
        s = find_shelf(b)
        self.assertGreaterEqual(len(s), 1)
        best = s[0]
        self.assertEqual(best["units"], 90)
        self.assertAlmostEqual(best["cost_gold"], (43 * 0.99 + 47 * 1.00), places=3)
        self.assertAlmostEqual(best["next_price_gold"], 1.73)
        self.assertAlmostEqual(best["net_margin"], 1.73 * 0.95 / ((43 * 0.99 + 47) / 90) - 1, places=4)
        self.assertGreater(best["net_margin"], 0.6)

    def test_small_gaps_big_costs_and_thin_margins_are_rejected(self):
        self.assertEqual(find_shelf(book([(100, 10), (120, 10), (130, 10)])), [])                       # gaps under 30%
        self.assertEqual(find_shelf(book([(10 * G, 100), (20 * G, 100)]), max_cost_gold=150), [])      # costs 1,000 gold
        self.assertEqual(len(find_shelf(book([(100, 10), (135, 10)]), min_net_margin=0.25)), 1)       # 135*.95/100-1 = 28%: passes
        self.assertEqual(find_shelf(book([(100, 10), (131, 10)]), min_net_margin=0.25), [])           # 131*.95/100-1 = 24.5%: too thin


class TestProfileAndSeries(unittest.TestCase):
    def frame(self, days=14):
        ts = pd.date_range("2026-10-05 00:00", periods=24 * days, freq="h")
        price = np.where(ts.hour == 16, 130.0, 100.0)
        df = pd.DataFrame({"ts": ts, "floor_gold": price, "units_listed": np.where(ts.dayofweek == 0, 5000, 3000),
                           "sold_est": 10.0, "flag_long_gap": False, "flag_wipe": False})
        return df

    def test_hour_of_day_and_week_slots_with_weeks_and_thin_flags(self):
        df = self.frame(14)
        hod = profile(df, tz="UTC", by="none")
        self.assertEqual(len(hod), 24)
        self.assertAlmostEqual(float(hod.loc[hod["block"] == 16, "price_median"].iloc[0]), 130.0)
        self.assertAlmostEqual(float(hod.loc[hod["block"] == 3, "price_median"].iloc[0]), 100.0)
        self.assertEqual(int(hod["n_obs"].min()), 14)
        self.assertEqual(int(hod["weeks"].iloc[0]), 2)
        how = profile(df, tz="UTC", by="dow", min_obs=3)
        self.assertEqual(len(how), 7 * 24)
        self.assertTrue((how["n_obs"] == 2).all())
        self.assertTrue(how["thin"].all())                        # two weeks: two observations per slot, still a draft
        monday = how[how["dow"] == 0]
        self.assertTrue((monday["units_median"] == 5000).all())
        blocks = profile(df, tz="UTC", by="weekpart", block_hours=4)
        self.assertEqual(len(blocks), 2 * 6)

    def test_spike_hours_do_not_move_the_profile(self):
        df = self.frame(14)
        df.loc[df.index[40], "floor_gold"] = 9999.0              # one wild hour
        sp = flag_spikes(df["floor_gold"])
        got = profile(df, spikes=sp, tz="UTC", by="none")
        self.assertTrue((got["price_median"] < 200).all())

    def test_item_series_runs_on_the_simulator(self):
        res = MarketSimulator(default_profiles(), seed=3).run(40)
        ser = item_series(res.snapshots, [1002, 1004])
        self.assertEqual(set(ser), {1002, 1004})
        for df in ser.values():
            self.assertEqual(len(df), len(res.snapshots))
            for c in ("ts", "floor_gold", "units_listed", "sold_est", "flag_long_gap"):
                self.assertIn(c, df.columns)
        sp = flag_spikes(ser[1002]["floor_gold"])
        self.assertEqual(len(sp), len(ser[1002]))
        p = profile(ser[1002], spikes=sp, by="none")
        self.assertGreater(len(p), 0)


if __name__ == "__main__":
    unittest.main()
