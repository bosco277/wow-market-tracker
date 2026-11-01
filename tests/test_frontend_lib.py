import os
import sys
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_root, "frontend"), _root):      # lib.py sits in "frontend" here and in the main folder in the app repo
    if os.path.exists(os.path.join(_p, "lib.py")):
        sys.path.insert(0, _p)
        break
import lib  # noqa: E402


def items():
    return pd.DataFrame({
        "item_id": [1, 2, 3], "name": ["Iceweb Spider Silk", "Saronite Ore", ""], "watched": [False, True, False],
        "floor_gold": [1.0, 20.0, 0.5], "daily_low": [900, 5, 100], "daily_est": [1000, 10, 150], "daily_high": [1100, 40, 200],
        "confidence": ["High", "Low", "Medium"], "now_percentile": [0.5, 0.95, 0.05]})


class TestLib(unittest.TestCase):
    def test_gsc(self):
        self.assertEqual(lib.gsc(12.3456), "12g 34s 56c")
        self.assertEqual(lib.gsc(0.0099), "99c")
        self.assertEqual(lib.gsc(float("nan")), "-")

    def test_rng_and_short_num(self):
        self.assertEqual(lib.rng(900, 1500, 2100000), "900 - 1.5k - 2.1M")

    def test_market_table_ranks_by_gold_flow_and_filters(self):
        t = lib.market_table(items())
        self.assertEqual(list(t["item_id"]), [1, 2, 3])          # 1000, 200, 75 gold/day
        self.assertEqual(list(lib.market_table(items(), conf=("Low",))["item_id"]), [2])
        self.assertEqual(list(lib.market_table(items(), text="iceweb")["item_id"]), [1])
        self.assertEqual(list(lib.market_table(items(), text="3")["item_id"]), [3])
        self.assertEqual(list(lib.market_table(items(), watched_only=True)["item_id"]), [2])
        self.assertTrue((t["gold_flow_low"] <= t["gold_flow"]).all() and (t["gold_flow"] <= t["gold_flow_high"]).all())

    def test_price_position(self):
        self.assertEqual([lib.price_position(r) for _, r in items().iterrows()], ["normal", "expensive for this item", "cheap for this item"])

    def test_daily_ribbon_needs_a_full_window_and_keeps_order(self):
        ts = pd.date_range("2026-10-30", periods=30, freq="h")
        h = pd.DataFrame({"ts": ts, "sold_low": 1.0, "sold_est": 2.0, "sold_high": 3.0})
        h.loc[28, "sold_est"] = np.nan
        r = lib.daily_ribbon(h)
        self.assertTrue(r["est"].iloc[:23].isna().all())
        self.assertEqual(r["low"].iloc[23], 24.0)
        self.assertEqual(r["est"].iloc[23], 48.0)
        self.assertTrue(np.isnan(r["est"].iloc[28]))
        ok = r["est"].notna()
        self.assertTrue((r.loc[ok, "low"] <= r.loc[ok, "est"]).all())
        self.assertEqual(len(lib.hole_hours(h)), 1)

    def test_confidence_mix_and_scorecard_and_freshness(self):
        cal = pd.DataFrame({"data_last_ts": pd.to_datetime(["2026-10-30 12:00", "2026-10-31 12:00"]), "n_high": [0, 2], "n_medium": [5, 4],
                            "n_low": [1, 1], "exp_rate_check_n": [np.nan, 7], "exp_rate_inside": [np.nan, 3],
                            "exp_rate_pred_too_fast": [np.nan, 2], "exp_rate_pred_too_slow": [np.nan, 2]})
        m = lib.confidence_mix(cal)
        self.assertEqual(len(m), 6)
        self.assertEqual(m[(m["tier"] == "High") & (m["data_last_ts"] == pd.Timestamp("2026-10-31 12:00"))]["items"].iloc[0], 2)
        self.assertEqual(lib.experiment_scorecard(cal)["inside"], 3)
        self.assertIsNone(lib.experiment_scorecard(cal)["listings"])
        h, label = lib.freshness(cal, now="2026-10-31 13:00")
        self.assertAlmostEqual(h, 1.0)
        self.assertEqual(label, "fresh")
        self.assertEqual(lib.freshness(cal, now="2026-11-01 12:00")[1], "very stale")


if __name__ == "__main__":
    unittest.main()


class TestShelfLib(unittest.TestCase):
    def test_shelf_table_filters_and_hours_text(self):
        sh = pd.DataFrame({"verdict": ["candidate", "weak", "no"], "sell_hours_est": [4.3, 30.0, 1.0], "sell_hours_high": [3.2, 20.0, 1.0],
                           "sell_hours_low": [6.5, float("inf"), 1.0]})
        self.assertEqual(len(lib.shelf_table(sh)), 2)
        self.assertEqual(len(lib.shelf_table(sh, ("candidate",))), 1)
        self.assertEqual(lib.shelf_hours(sh.iloc[0]), "4.3 h (3.2 - 6.5)")
        self.assertEqual(lib.shelf_hours(sh.iloc[1]), "30.0 h (20.0 - never)")
        self.assertTrue(lib.shelf_table(pd.DataFrame()).empty)


class TestFlagMask(unittest.TestCase):
    def test_flag_mask_handles_odd_dtypes_and_missing_columns(self):
        df = pd.DataFrame({"a": [True, False, None], "b": pd.Series([1.0, np.nan, 0.0]), "c": pd.Series(["x", "y", "z"]).astype(object).map(lambda v: v == "y")})
        self.assertEqual(list(lib.flag_mask(df, "a")), [True, False, False])
        self.assertEqual(list(lib.flag_mask(df, "b")), [True, False, False])
        self.assertEqual(list(lib.flag_mask(df, "c")), [False, True, False])
        self.assertEqual(list(lib.flag_mask(df, "missing")), [False, False, False])
        self.assertEqual(len(lib.flag_mask(pd.DataFrame(), "a")), 0)
        self.assertEqual(list(df.loc[lib.flag_mask(df, "a")].index), [0])

    def test_watched_filter_survives_object_dtype(self):
        it = items().assign(watched=pd.Series([None, True, False], dtype=object))
        self.assertEqual(list(lib.market_table(it, watched_only=True)["item_id"]), [2])


class TestShelfWords(unittest.TestCase):
    def test_checks_and_watch_text(self):
        row = pd.Series({c: "pass" for c in lib.TEST_COLS})
        self.assertEqual(lib.shelf_checks(row), "6 of 6 pass")
        self.assertEqual(lib.shelf_watch(row), "none")
        row["test_regime"], row["test_price"] = "fail", "warn"
        self.assertEqual(lib.shelf_checks(row), "4 of 6 pass")
        self.assertEqual(lib.shelf_watch(row), "FAIL: Normal price, not a spike; WARN: Price history")

    def test_every_check_has_an_explanation(self):
        for c in lib.TEST_COLS:
            self.assertEqual(len(lib.TEST_INFO[c]), 3)


class TestBacktestTable(unittest.TestCase):
    def test_formats_percentages_and_blanks(self):
        s = pd.DataFrame({"group": ["all shelves"], "n": [12], "reached_4h": [0.5], "reached_12h": [0.75], "median_hours": [float("nan")],
                          "median_floor_ratio_4h": [0.98], "undercut_risk": [0.25]})
        t = lib.backtest_table(s)
        self.assertEqual(t.iloc[0]["Reached relist price within 4 h"], "50%")
        self.assertEqual(t.iloc[0]["Median hours to get there"], "-")
        self.assertEqual(t.iloc[0]["Price 4 h later vs relist price"], "0.98x")
        self.assertTrue(lib.backtest_table(pd.DataFrame()).empty)


class TestBacktestRates(unittest.TestCase):
    def test_rates_by_verdict(self):
        s = pd.DataFrame({"group": ["all shelves", "verdict: candidate", "verdict: weak"], "n": [10, 5, 3],
                          "reached_4h": [0.1, 0.61, 0.41], "reached_12h": [0.2, 0.82, 0.6], "median_hours": [2, 2, 2],
                          "median_floor_ratio_4h": [0, 0.9, 0.8], "undercut_risk": [0.6, 0.48, 0.52]})
        r = lib.backtest_rates(s)
        self.assertEqual(lib.past_hit_rate("candidate", r), "61% / 82% (undercut 48%)")
        self.assertEqual(lib.past_hit_rate("no", r), "-")
        self.assertEqual(lib.backtest_rates(pd.DataFrame()), {})


class TestOldShelfTable(unittest.TestCase):
    def old_table(self):
        """What the v0.6.2 engine wrote: no long-run columns, no regime test, no sales per day."""
        return pd.DataFrame({"item_id": [1, 2], "name": ["A", "B"], "verdict": ["candidate", "weak"], "units": [10, 20],
                             "avg_price_gold": [1.0, 2.0], "breakeven_gold": [1.05, 2.1], "target_gold": [1.5, 3.0], "net_margin": [0.3, 0.4],
                             "net_margin_stress": [0.2, 0.3], "profit_gold": [4.0, 9.0], "profit_stress_gold": [2.0, 5.0], "cost_gold": [10.0, 40.0],
                             "sell_hours_low": [3.0, 5.0], "sell_hours_est": [2.0, 4.0], "sell_hours_high": [1.0, 3.0], "confidence": ["Medium"] * 2,
                             **{c: ["pass", "warn"] for c in ("test_price", "test_speed", "test_size", "test_margin", "test_book")}})

    def test_missing_columns_are_filled_and_the_app_pieces_still_work(self):
        t = lib.ensure_shelf_columns(self.old_table())
        for c in lib.SHELF_NUMERIC + lib.TEST_COLS:
            self.assertIn(c, t.columns)
        row = t.iloc[0]
        self.assertEqual(lib.shelf_checks(row), "5 of 5 pass")
        self.assertEqual(lib.shelf_watch(t.iloc[1]), "WARN: Price history, Sells fast enough, Fits your limits, Still profitable if undercut, Complete listing snapshot")
        self.assertEqual(lib.normal_price_text(row["long_median_gold"], row["long_p10_gold"], row["long_p90_gold"], row["long_days"]), "-")
        self.assertEqual(lib.normal_price_text(1.2, 0.9, 1.5, 5.0), "1g 20s 0c (90s 0c - 1g 50s 0c, 5d)")
        self.assertEqual(len(lib.shelf_table(t)), 2)
        self.assertTrue(lib.ensure_shelf_columns(pd.DataFrame()).empty)


class TestNumericTables(unittest.TestCase):
    def market(self):
        it = items().assign(band_p10=[0.8, 15.0, 0.4], band_p90=[1.2, 25.0, 0.7], gold_flow=0, wall_est_gold=[1.1, 22.0, 0.6],
                            daily_low=[900, 5, 100], daily_est=[1000, 10, 150], daily_high=[1100, 40, 200],
                            confidence_reason=["a", "b", "c"])
        return lib.market_table(it)

    def test_market_numbers_are_numeric_and_sort(self):
        t = lib.market_numbers(self.market())
        for c in ("Lowest price (g)", "Sold/day low", "Sold/day best", "Sold/day high", "Gold/day best (g)", "Price wall (g)", "Item ID"):
            self.assertTrue(pd.api.types.is_numeric_dtype(t[c]), c)
        self.assertEqual(list(t["Item ID"]), [1, 2, 3])
        self.assertTrue(pd.api.types.is_bool_dtype(t["Watched"]))

    def test_flip_numbers_split_ranges_and_handle_inf(self):
        sh = TestOldShelfTable().old_table()
        sh.loc[1, "sell_hours_low"] = float("inf")
        t = lib.flip_numbers(lib.ensure_shelf_columns(sh), {"candidate": (0.61, 0.82, 0.48)})
        for c in ("Net margin (%)", "Profit at target (g)", "Normal price median (g)", "Sold/day best", "Hours to sell (slow case)",
                  "Checks passed", "Past 4h hit (%)", "Avg cost (g)"):
            self.assertTrue(pd.api.types.is_numeric_dtype(t[c]), c)
        self.assertAlmostEqual(t["Net margin (%)"].iloc[0], 30.0)
        self.assertAlmostEqual(t["Past 4h hit (%)"].iloc[0], 61.0)
        self.assertTrue(np.isnan(t["Past 4h hit (%)"].iloc[1]))              # weak has no rates here
        self.assertTrue(np.isnan(t["Hours to sell (slow case)"].iloc[1]))    # inf shown as blank, not text
        self.assertEqual(int(t["Checks passed"].iloc[0]), 5)
        self.assertEqual(int(t["Checks warned"].iloc[1]), 5)

    def test_watch_ids_and_labels(self):
        self.assertEqual(lib.parse_ids("12, 5 12;7\nx 9.5"), [12, 5, 7])
        self.assertEqual(lib.parse_ids(None), [])
        t = lib.market_table(items(), watch_ids=[1])
        self.assertEqual(list(t.loc[t["watched"], "item_id"]), [1])
        self.assertEqual(list(lib.market_table(items(), watched_only=True, watch_ids=[3])["item_id"]), [3])
        lab = lib.item_labels(items())
        self.assertEqual(lab[2], "Saronite Ore (2)")
        self.assertEqual(lab[3], "item 3")


class TestTimeProfiles(unittest.TestCase):
    def hourly(self, days=10):   # ends before daylight saving time ends (Nov 1), so the -4 h offset holds
        ts = pd.date_range("2026-10-20 00:00", periods=days * 24, freq="h")
        hour_et = (ts - pd.Timedelta(hours=4)).hour
        return pd.DataFrame({"ts": ts, "floor_gold": np.where(hour_et >= 20, 8.0, 10.0), "units_listed": np.where(hour_et >= 18, 2000, 1000),
                             "sold_est": np.where(hour_et == 21, 5.0, 1.0), "spike": False, "blip": False})

    def test_grids_are_7_by_24_in_eastern_time(self):
        tp = lib.time_profiles(self.hourly())
        for k in ("price", "units", "sold", "n"):
            self.assertEqual(tp[k].shape, (7, 24))
        self.assertTrue((tp["price"][20] == 8.0).all() and (tp["price"][12] == 10.0).all())    # 8 PM Eastern, not UTC
        self.assertTrue((tp["units"][19] == 2000).all() and (tp["units"][9] == 1000).all())
        self.assertTrue((tp["sold"][21] == 5.0).all())
        self.assertEqual(int(tp["n"].to_numpy().sum()), tp["hours"])
        self.assertEqual(len(lib.HOURS), 24)
        self.assertEqual((lib.HOURS[0], lib.HOURS[12], lib.HOURS[23]), ("12am", "12pm", "11pm"))

    def test_day_of_week_row_order(self):
        h = self.hourly()
        tp = lib.time_profiles(h.assign(floor_gold=[float(t.dayofweek) for t in (h["ts"] - pd.Timedelta(hours=4))]))
        for d in range(7):
            self.assertTrue((tp["price"].loc[d].dropna() == d).all())                       # row 0 = Monday ... row 6 = Sunday

    def test_flagged_hours_excluded_blank_cells_and_timezone_aware_input(self):
        h = self.hourly(2)
        h.loc[5, ["floor_gold", "spike"]] = [1000.0, True]
        tp = lib.time_profiles(h)
        self.assertEqual(tp["hours"], len(h) - 1)
        self.assertTrue(tp["price"].isna().any().any())                                   # two days cannot fill a week
        self.assertEqual(lib.time_profiles(h.assign(ts=h["ts"].dt.tz_localize("UTC")))["hours"], len(h) - 1)
        self.assertEqual(lib.time_profiles(pd.DataFrame()), {})


class TestPastEvents(unittest.TestCase):
    def test_plain_headers_numbers_and_wording(self):
        f = pd.DataFrame({"t1": pd.to_datetime(["2026-10-31 17:33:42", "2026-11-01 06:33:45", "2026-11-01 08:00:00"]),
                          "floor_before_gold": [2.92, 1.74, 2.0], "floor_after_gold": [3.96, 3.95, 4.0], "jump": [1.3562, 2.2701, 2.0],
                          "units_removed": [93.0, 65.0, 40.0], "ended_by": ["reverted", "hole", "end_of_data"],
                          "revert_hours": [3.5, np.nan, np.nan], "followup_hours": [3.5, 2.0, 1.5], "likely_flip": [True, False, False]})
        t = lib.past_events_table(f)
        self.assertEqual(list(t["When (US Eastern)"])[0], "Sun Nov 01 03:00 AM")        # newest first, DST ended Nov 1 (UTC-5 -> 03:00)
        self.assertAlmostEqual(t["Price rose by (%)"].iloc[2], 35.62, places=2)
        self.assertTrue(pd.api.types.is_numeric_dtype(t["Units bought out"]))
        self.assertEqual(list(t["Back to old price?"]), ["not yet (watched 1.5 h)", "unknown (gap in our data)", "yes, after 3.5 h"])
        self.assertEqual(list(t["Looks like a flipper"]), ["no", "no", "yes"])
        self.assertTrue(lib.past_events_table(pd.DataFrame()).empty)
        self.assertNotIn("Back to old price?", lib.past_events_table(f.drop(columns=["ended_by"])).columns)
