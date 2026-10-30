import unittest

import numpy as np
import pandas as pd

from helpers import T0, mk
from engine.metrics import (
    ask_tiers, daily_volume_quantile, depth_vwap, depth_vwap_table, expected_sales, round_trip_roi,
    sell_clearing_price, target_depth,
)

BOOK = pd.DataFrame({"unit_price": [11, 10, 12, 10], "quantity": [100, 60, 300, 40]})  # tiers: 10:100, 11:100, 12:300


class TestDepth(unittest.TestCase):
    def test_tiers_aggregate_and_sort(self):
        t = ask_tiers(BOOK)
        self.assertEqual(list(t["price"]), [10, 11, 12])
        self.assertEqual(list(t["qty"]), [100, 100, 300])
        self.assertEqual(list(t["cum_qty"]), [100, 200, 500])

    def test_exact_partial_tier(self):
        q = depth_vwap(BOOK, 150)
        self.assertAlmostEqual(q.vwap, (100 * 10 + 50 * 11) / 150)
        self.assertEqual((q.units_filled, q.marginal_price, q.complete), (150, 11, True))

    def test_snap_to_tier_takes_the_whole_wall(self):
        q = depth_vwap(BOOK, 150, snap_to_tier=True)
        self.assertEqual(q.units_filled, 200)
        self.assertAlmostEqual(q.vwap, 10.5)

    def test_shallow_book_and_degenerate_inputs(self):
        q = depth_vwap(BOOK, 900)
        self.assertEqual((q.units_filled, q.complete), (500, False))
        self.assertTrue(depth_vwap(BOOK, 0).complete)
        e = depth_vwap(BOOK.iloc[0:0], 10)
        self.assertEqual((e.units_filled, e.complete), (0.0, False))

    def test_vwap_is_continuous_in_depth(self):
        # exact partial tiers: no jump when the cutoff crosses a tier boundary
        a, b = depth_vwap(BOOK, 99.0).vwap, depth_vwap(BOOK, 101.0).vwap
        self.assertLess(abs(b - a), 0.02)
        snapped = abs(depth_vwap(BOOK, 101.0, snap_to_tier=True).vwap - depth_vwap(BOOK, 99.0, snap_to_tier=True).vwap)
        self.assertGreater(snapped, abs(b - a))

    def test_vectorised_table_matches_scalar(self):
        rng = np.random.default_rng(1)
        rows = []
        for item in (1, 2, 3):
            for a in range(int(rng.integers(5, 25))):
                rows.append((len(rows) + 1, item, int(rng.integers(1, 300)), int(rng.integers(5, 40)), "LONG"))
        snap = mk(rows)
        depth = pd.Series({1: 120.0, 2: 800.0, 3: 5.0})
        tab = depth_vwap_table(snap.book, depth).set_index("item_id")
        for item, d in depth.items():
            q = depth_vwap(snap.item_book(item), d)
            self.assertAlmostEqual(tab.loc[item, "vwap"], q.vwap)
            self.assertAlmostEqual(tab.loc[item, "units_filled"], q.units_filled)
            self.assertAlmostEqual(tab.loc[item, "marginal_price"], q.marginal_price)

    def test_target_depth(self):
        self.assertEqual(target_depth(1000, 1.5, 1.2), 1800)
        self.assertEqual(target_depth(0), 1.0)


class TestSellSideLifo(unittest.TestCase):
    """Newest-first at a price (retail): only strictly cheaper listings are ahead of you. BOOK tiers: 10:100, 11:100, 12:300."""

    def test_joins_a_higher_tier_because_its_own_tier_is_not_ahead_of_you(self):
        q = sell_clearing_price(BOOK, 250, 40, tier_order="lifo")  # room 210; units strictly below: 0, 100, 200
        self.assertEqual((q.list_price, q.units_ahead, q.undercut), (12, 200, False))
        self.assertEqual(sell_clearing_price(BOOK, 250, 40, tier_order="fifo").list_price, 11)

    def test_never_needs_an_undercut(self):
        q = sell_clearing_price(BOOK, 100, 40, tier_order="lifo")  # FIFO would undercut to 9
        self.assertEqual((q.list_price, q.units_ahead, q.undercut, q.complete), (10, 0, False, True))
        q = sell_clearing_price(BOOK, 30, 40, tier_order="lifo")   # market absorbs less than my stack
        self.assertEqual((q.list_price, q.expected_filled, q.complete, q.undercut), (10, 30, False, False))

    def test_lifo_price_is_never_below_fifo_and_never_above_the_top_ask(self):
        for e in (0, 20, 60, 100, 150, 250, 400, 600, 5000):
            lifo = sell_clearing_price(BOOK, e, 40, tier_order="lifo").list_price
            fifo = sell_clearing_price(BOOK, e, 40, tier_order="fifo").list_price
            self.assertGreaterEqual(lifo, fifo)
            self.assertLessEqual(lifo, 12)

    def test_rejects_unknown_rule(self):
        with self.assertRaises(ValueError):
            sell_clearing_price(BOOK, 100, 40, tier_order="random")


class TestSellSide(unittest.TestCase):
    def test_joins_the_deepest_tier_that_still_leaves_room(self):
        q = sell_clearing_price(BOOK, expected_sales_units=250, my_units=40)  # room 210 -> tier 11 (cum 200)
        self.assertEqual((q.list_price, q.units_ahead, q.complete, q.undercut), (11, 200, True, False))

    def test_undercuts_when_the_floor_tier_is_too_deep(self):
        q = sell_clearing_price(BOOK, expected_sales_units=100, my_units=40)  # room 60 < cum 100
        self.assertEqual((q.list_price, q.expected_filled, q.undercut, q.complete), (9, 40, True, True))

    def test_partial_fill_when_market_absorbs_less_than_my_stack(self):
        q = sell_clearing_price(BOOK, expected_sales_units=30, my_units=40)
        self.assertEqual((q.expected_filled, q.complete, q.undercut), (30, False, True))

    def test_never_quotes_above_the_highest_ask(self):
        q = sell_clearing_price(BOOK, expected_sales_units=10_000, my_units=40)
        self.assertEqual(q.list_price, 12)

    def test_more_expected_sales_never_lowers_the_price(self):
        prices = [sell_clearing_price(BOOK, e, 40).list_price for e in (20, 60, 100, 150, 250, 400, 600, 5000)]
        self.assertEqual(prices, sorted(prices))

    def test_round_trip_roi(self):
        buy = depth_vwap(BOOK, 150)
        sell = sell_clearing_price(BOOK, 250, 40)
        roi = round_trip_roi(buy, sell)
        self.assertAlmostEqual(roi, (11 * 0.95 - buy.vwap) / buy.vwap)


def intervals(rows):
    df = pd.DataFrame(rows, columns=["t1", "sold"])
    df["source"] = "s"; df["item_id"] = 1
    df["t1"] = pd.to_datetime(df["t1"]); df["t0"] = df["t1"] - pd.Timedelta(hours=1)
    df["gap_minutes"] = 60.0
    df["sold_low"] = df["sold"] * 0.5; df["sold_est"] = df["sold"]; df["sold_high"] = df["sold"] * 1.5
    df["flag_wipe"] = False; df["flag_long_gap"] = False
    return df


class TestDemand(unittest.TestCase):
    def test_expected_sales_trailing_and_blind_spot(self):
        iv = intervals([(T0 + pd.Timedelta(hours=h), 10.0) for h in range(1, 25)])  # 10/h for a day
        now = T0 + pd.Timedelta(hours=24)
        e = expected_sales(iv, now, horizon_hours=6, assumption="est")
        self.assertAlmostEqual(e["expected_units"].iloc[0], 60.0)
        e2 = expected_sales(iv, now, horizon_hours=6, assumption="high", blind_spot_correction=2.0)
        self.assertAlmostEqual(e2["expected_units"].iloc[0], 60.0 * 1.5 * 2.0)

    def test_daily_volume_quantile(self):
        rows = []
        for day, per_hour in enumerate((10, 20, 30)):
            for h in range(1, 25):
                rows.append((pd.Timestamp("2026-10-20") + pd.Timedelta(days=day, hours=h) - pd.Timedelta(seconds=1), float(per_hour)))
        iv = intervals(rows)
        now = pd.Timestamp("2026-10-23")
        q50 = daily_volume_quantile(iv, now, lookback_days=7, q=0.5)
        q75 = daily_volume_quantile(iv, now, lookback_days=7, q=0.75)
        self.assertAlmostEqual(q50["daily_q"].iloc[0], 480.0)
        self.assertAlmostEqual(q75["daily_q"].iloc[0], 600.0)
        self.assertEqual(q50["n_days"].iloc[0], 3)

    def test_seasonal_component_uses_same_hour_of_week(self):
        rows = []
        start = pd.Timestamp("2026-10-12 00:00")  # a Monday
        for hour in range(0, 14 * 24):
            t0 = start + pd.Timedelta(hours=hour)
            rows.append((t0 + pd.Timedelta(hours=1), 10.0 if (hour % 24) < 12 else 50.0))
        iv = intervals(rows)
        now = pd.Timestamp("2026-10-26 11:30")  # hour 11; next three hours are hours 11, 12, 13 -> 10 + 50 + 50
        s = expected_sales(iv, now, horizon_hours=3, assumption="est", seasonal_weight=1.0)
        self.assertAlmostEqual(s["seasonal_units"].iloc[0], 110.0)
        self.assertAlmostEqual(s["expected_units"].iloc[0], 110.0)
        t = expected_sales(iv, now, horizon_hours=3, assumption="est", seasonal_weight=0.0)
        self.assertNotAlmostEqual(t["expected_units"].iloc[0], 110.0)


if __name__ == "__main__":
    unittest.main()
