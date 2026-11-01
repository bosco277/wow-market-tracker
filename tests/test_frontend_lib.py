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
