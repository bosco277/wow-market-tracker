import os
import sys
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import targets as Tg  # noqa: E402
from engine import windows as W  # noqa: E402
from tests.test_windows import market, two_groups  # noqa: E402


def named_groups(items=8):
    g = two_groups(items)
    g["name"] = [f"Item {i}" for i in g["item_id"]]
    return g


def when_row(res, tier="great", pool="everything"):
    w = res["when"]
    return w[(w["pool"] == pool) & (w["tier"] == tier)].iloc[0]


def calib_row(res, tier, H, pool="everything"):
    c = res["calibration"]
    return c[(c["pool"] == pool) & (c["tier"] == tier) & (c["H"] == H)]


class TestTargets(unittest.TestCase):
    def test_targets_are_set_from_earlier_hours_only(self):
        hl = market(days=20, noise=0.04, day_walk=0.02)
        cfg = Tg.TargetConfig()
        g0 = Tg.build_grid(W.prepare(hl), cfg)
        day, i0 = g0["starts"][12]
        changed = hl.copy()
        cut = g0["index"][i0]
        changed.loc[changed["ts"] >= cut, "floor_gold"] *= 20.0                    # everything from that midnight on goes wild
        g1 = Tg.build_grid(W.prepare(changed), cfg)
        self.assertEqual(g1["starts"][12], (day, i0))
        for name in cfg.tiers:
            np.testing.assert_array_equal(g0["THR"][name][i0], g1["THR"][name][i0])      # the day's targets never saw the day
        np.testing.assert_array_equal(g0["MED"][i0], g1["MED"][i0])

    def test_steal_is_deeper_and_reached_less_often_than_good(self):
        res = Tg.analyze_targets(market(days=40, noise=0.04, day_walk=0.02), named_groups())
        c = res["calibration"]
        e = c[(c["pool"] == "everything") & (c["H"] == 1)].groupby("tier").apply(lambda d: pd.Series({"reached": np.average(d["reached"], weights=d["n"]), "depth": np.average(d["depth_med"], weights=d["n"])}), include_groups=False)
        self.assertGreater(e.loc["good", "reached"], e.loc["great", "reached"])
        self.assertGreater(e.loc["great", "reached"], e.loc["steal", "reached"])
        self.assertLess(e.loc["good", "depth"], e.loc["steal", "depth"])
        for tier in ("good", "great", "steal"):                                      # a longer wait always reaches at least as often
            r = calib_row(res, tier, 1)["reached"].mean(), calib_row(res, tier, 3)["reached"].mean(), calib_row(res, tier, 7)["reached"].mean()
            self.assertLessEqual(r[0], r[1] + 1e-9)
            self.assertLessEqual(r[1], r[2] + 1e-9)

    def test_a_steadily_rising_market_never_returns_to_its_old_low_prices(self):
        res = Tg.analyze_targets(market(days=30, noise=0.01, slide=0.03), named_groups())
        c = res["calibration"]
        self.assertLess(c[c["H"] == 1]["reached"].max(), 0.05)                      # the 7-day-old lows are not coming back: the calibration says so instead of promising them

    def test_a_cheap_window_that_repeats_is_found_and_replicates(self):
        res = Tg.analyze_targets(market(days=40, noise=0.03, hour_effect={3: -0.08, 4: -0.08, 5: -0.08}), named_groups())
        r = when_row(res)
        self.assertEqual(r["window_start"], 3)
        self.assertGreater(r["lift_later"], 1.5)
        self.assertGreater(r["replication"], 0.5)
        self.assertGreater(r["lifts"][4], 1.5)
        self.assertLess(r["lifts"][12], 1.0)

    def test_noise_has_no_rhythm_on_the_later_days(self):
        res = Tg.analyze_targets(market(days=40, noise=0.04, day_walk=0.02), named_groups())
        r = when_row(res)
        self.assertLess(r["lift_later"], 1.3)                                        # the best-looking window of the first half is just average on the second
        self.assertLess(r["replication"], 0.5)

    def test_a_flat_item_cannot_hit_all_day(self):
        res = Tg.analyze_targets(market(days=20, noise=0.0), named_groups())
        hh = res["when"]
        self.assertTrue(hh["lifts"].apply(lambda v: np.isnan(v).all()).all())        # tied prices: no item has a target 3% under its median, nothing is counted
        self.assertIn("WHEN IS THE PRICE", Tg.format_targets(res))                   # and the report still prints

    def test_the_shopping_list_matches_the_calibration_and_is_in_order(self):
        res = Tg.analyze_targets(market(days=40, noise=0.04, day_walk=0.02), named_groups())
        lst = res["list"]
        self.assertEqual(len(lst), 8)
        self.assertTrue((lst["steal_g"] <= lst["great_g"] + 1e-9).all() and (lst["great_g"] <= lst["good_g"] + 1e-9).all() and (lst["good_g"] <= lst["med7_g"] + 1e-9).all())
        self.assertTrue((lst["great_copper"] == (lst["great_g"] * 10000).round()).all())
        self.assertTrue(lst["score_g"].is_monotonic_decreasing)
        self.assertIn("name", lst.columns)
        self.assertTrue(lst["name"].str.startswith("Item").all())
        r = lst.iloc[0]
        c = res["calibration"]
        want = c[(c["pool"] == r["price_tier"]) & (c["tier"] == "great") & (c["depth"] == r["great_depth_band"]) & (c["H"] == 3)]["reached"]
        if len(want):
            self.assertAlmostEqual(r["great_p3d"], float(want.iloc[0]))              # the chance printed for an item is the calibrated one for its price tier and depth
        self.assertTrue(lst[["great_p1d", "great_p3d", "great_p7d"]].dropna().stack().between(0, 1).all())
        self.assertEqual(r["great_units"], 50)

    def test_too_little_history_says_so(self):
        res = Tg.analyze_targets(market(days=6, noise=0.04), named_groups())
        self.assertTrue(res["calibration"].empty)
        self.assertIn("Not enough history yet", Tg.format_targets(res))
        self.assertTrue(Tg.analyze_targets(pd.DataFrame())["list"].empty)

    def test_weekdays_wait_for_six_weeks(self):
        self.assertIsNone(Tg.analyze_targets(market(days=40, noise=0.04, day_walk=0.02), named_groups())["weekday"])
        wd = Tg.analyze_targets(market(days=60, noise=0.04, day_walk=0.02), named_groups())["weekday"]
        self.assertIsNotNone(wd)
        self.assertEqual(list(wd.columns), W.WEEKDAYS)

    def test_the_report_has_its_three_parts(self):
        text = Tg.format_targets(Tg.analyze_targets(market(days=40, noise=0.04, day_walk=0.02, hour_effect={3: -0.08}), named_groups()))
        for part in ("BUY-BELOW PRICES", "CALIBRATION, walk-forward", "WHEN IS THE PRICE AT OR UNDER THE TARGET", "SHOPPING LIST", "buy_targets.csv", "Replication", "Scan in game"):
            self.assertIn(part, text)


def with_clusters(hl, item_ids, cluster_days, hours=(10, 18), factor=0.02, start="2026-06-01"):
    """Copy of an hourly table in which each listed item's floor drops to `factor` of itself for a few US Eastern hours on each listed day (day numbers from `start`) and is normal otherwise:
    the price that swings between two levels (an 11,695g median and a 48g floor), whose cheap hours are a one-off."""
    out = hl.copy()
    et = out["ts"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    day_no = ((et.dt.normalize().dt.tz_localize(None) - pd.Timestamp(start)).dt.days).to_numpy()
    hr = et.dt.hour.to_numpy()
    mask = out["item_id"].isin(item_ids).to_numpy() & np.isin(day_no, list(cluster_days)) & (hr >= hours[0]) & (hr < hours[1])
    out.loc[mask, "floor_gold"] *= factor
    return out


class TestDeepTargets(unittest.TestCase):
    """The v0.7.40 'Tavern Mug' problem: one open-ended 20%+ band gave a target 99% under the median the hit rate of one 21% under."""
    @classmethod
    def setUpClass(cls):
        hl = with_clusters(market(days=45, noise=0.03, day_walk=0.01), [1000, 1001, 1002, 1003], (5, 15, 25, 35, 42))
        cls.res = Tg.analyze_targets(hl, named_groups())

    def test_the_bands_split_the_deep_end(self):
        self.assertEqual(Tg.DEPTH_NAMES, ["under 3%", "3-10%", "10-20%", "20-35%", "35-50%", "50%+"])
        self.assertEqual(len(Tg.DEPTH_EDGES), len(Tg.DEPTH_NAMES) + 1)
        for d, name in ((0.0, "under 3%"), (0.029, "under 3%"), (0.03, "3-10%"), (0.199, "10-20%"), (0.2, "20-35%"), (0.35, "35-50%"), (0.499, "35-50%"), (0.5, "50%+"), (0.996, "50%+")):
            self.assertEqual(Tg.depth_name(d), name)

    def test_a_one_off_cheap_cluster_makes_a_deep_target_that_is_not_reached(self):
        c = self.res["calibration"]
        row = c[(c["pool"] == "everything") & (c["tier"] == "steal") & (c["depth"] == "50%+") & (c["H"] == 1)]
        self.assertEqual(len(row), 1)
        self.assertGreaterEqual(int(row["n"].iloc[0]), 30)
        self.assertLess(float(row["reached"].iloc[0]), 0.05)                          # the cheap hours were over before the window: the deep target is almost never met again
        shallow = c[(c["pool"] == "everything") & (c["tier"] == "steal") & (c["depth"].isin(["under 3%", "3-10%"])) & (c["H"] == 3)]
        self.assertGreater(float(np.average(shallow["reached"], weights=shallow["n"])), 0.5)      # while ordinary shallow targets are

    def test_an_item_in_that_state_now_gets_the_deep_bands_chance_not_a_shallow_one(self):
        lst = self.res["list"].set_index("item_id")
        r = lst.loc[1000]                                                              # its last cluster was day 42: inside the last 7 days of a 45-day table
        self.assertGreaterEqual(r["steal_depth"], 0.5)
        self.assertEqual(r["steal_depth_band"], "50%+")
        c = self.res["calibration"]
        want = c[(c["pool"] == r["price_tier"]) & (c["tier"] == "steal") & (c["depth"] == "50%+") & (c["H"] == 3)]
        if len(want):
            self.assertAlmostEqual(r["steal_p3d"], float(want["reached"].iloc[0]))
            self.assertLess(r["steal_p3d"], 0.2)
        else:
            self.assertTrue(np.isnan(r["steal_p3d"]))                                  # or no chance at all, never somebody else's

    def test_a_thin_cell_shows_no_chance_instead_of_a_shaky_one(self):
        hl = market(days=40, noise=0.04, day_walk=0.02)
        none = Tg.analyze_targets(hl, named_groups(), cfg=Tg.TargetConfig(min_cal_n=10 ** 6))
        self.assertFalse(none["calibration"].empty)
        self.assertTrue(none["list"][["good_p1d", "good_p3d", "great_p7d", "steal_p3d"]].isna().all().all())
        some = Tg.analyze_targets(hl, named_groups(), cfg=Tg.TargetConfig(min_cal_n=1))
        self.assertTrue(some["list"][["good_p1d", "good_p3d"]].notna().any().any())


class TestExport(unittest.TestCase):
    """The four tables the Daily chain uploads for the front end's Buy targets tab."""
    @classmethod
    def setUpClass(cls):
        cls.res = Tg.analyze_targets(market(days=40, noise=0.04, day_walk=0.02, hour_effect={3: -0.08, 4: -0.08, 5: -0.08}), named_groups())
        cls.t = Tg.export_tables(cls.res, built_at="2026-11-17 06:30")

    def test_four_tables_with_the_names_the_app_loads(self):
        self.assertEqual(set(self.t), set(Tg.TABLES))
        self.assertEqual(set(self.t), {"buy_targets", "buy_calibration", "buy_when", "buy_meta"})

    def test_the_shopping_list_is_unchanged(self):
        pd.testing.assert_frame_equal(self.t["buy_targets"], self.res["list"])

    def test_every_column_is_a_plain_type_parquet_can_store(self):
        for name, df in self.t.items():
            for c in df.columns:
                kind = df[c].dtype.kind
                self.assertIn(kind, "iufMOUbS", (name, c, df[c].dtype))                 # no categoricals, intervals or lists
                if kind in "OUS":
                    self.assertTrue(all(isinstance(x, str) for x in df[c].dropna()), (name, c))

    def test_hour_lifts_are_one_column_per_hour_and_match_the_analysis(self):
        w = self.t["buy_when"]
        self.assertNotIn("lifts", w.columns)
        cols = [f"lift_{h:02d}" for h in range(24)]
        self.assertTrue(set(cols) <= set(w.columns))
        src = self.res["when"]
        for i in range(len(src)):
            self.assertTrue(np.allclose(w.loc[w.index[i], cols].to_numpy(float), np.asarray(src["lifts"].iloc[i], float), equal_nan=True))

    def test_calibration_depths_are_text(self):
        c = self.t["buy_calibration"]
        self.assertTrue(set(c["depth"]) <= set(Tg.DEPTH_NAMES))
        self.assertEqual(len(c), len(self.res["calibration"]))

    def test_meta_says_when_and_from_what(self):
        m = self.t["buy_meta"]
        self.assertEqual(len(m), 1)
        r = m.iloc[0]
        self.assertEqual(pd.Timestamp(r["built_at"]), pd.Timestamp("2026-11-17 06:30"))
        self.assertEqual(r["days"], 40)
        self.assertEqual(r["listed"], len(self.res["list"]))
        self.assertEqual(r["first_day"], "2026-06-01")
        self.assertEqual(r["last_day"], "2026-07-10")
        self.assertEqual(r["lookback_days"], 7)
        self.assertEqual(r["window_hours"], 3)
        self.assertEqual(pd.Timestamp(r["last_hour"]), self.res["last_hour"])

    def test_nothing_to_export_gives_empty_tables_and_a_meta_row(self):
        t = Tg.export_tables(Tg.analyze_targets(pd.DataFrame()))
        self.assertTrue(t["buy_targets"].empty and t["buy_calibration"].empty and t["buy_when"].empty)
        self.assertEqual(len(t["buy_meta"]), 1)
        self.assertEqual(t["buy_meta"]["listed"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
