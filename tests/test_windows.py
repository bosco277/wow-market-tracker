import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import daily as D  # noqa: E402
from engine import windows as W  # noqa: E402
from tests.helpers import T0, mk  # noqa: E402


def market(days=40, items=8, start="2026-06-01", seed=1, hour_effect=None, dow_effect=None, noise=0.03, day_walk=0.0, drift=0.0, base=10.0):
    """A synthetic hourly table (HOURLY_COLUMNS) whose answer is known. hour_effect: {ET hour: log effect} or a function of the day number returning that dict;
    dow_effect: {weekday: log effect}; drift: log growth per day. June to September: no clock change."""
    rng = np.random.default_rng(seed)
    et = pd.date_range(start, periods=days * 24, freq="h", tz="America/New_York")
    day_no = np.repeat(np.arange(days), 24)
    hrs, dows = et.hour.to_numpy(), et.dayofweek.to_numpy()
    rows = []
    for it in range(items):
        walk = np.cumsum(rng.normal(0, day_walk, days))[day_no] if day_walk else np.zeros(len(et))
        eff = np.zeros(len(et))
        for i in range(len(et)):
            he = hour_effect(day_no[i]) if callable(hour_effect) else (hour_effect or {})
            eff[i] = he.get(hrs[i], 0.0) + (dow_effect or {}).get(dows[i], 0.0)
        price = base * (1 + it) * np.exp(walk + eff + drift * day_no + rng.normal(0, noise, len(et)))
        rows.append(pd.DataFrame({"item_id": 1000 + it, "ts": et.tz_convert("UTC").tz_localize(None), "floor_gold": price, "units_listed": 100, "u5": 50, "u10": 80,
                                  "p15": price * 1.02, "p30": price * 1.04, "spike": False, "blip": False, "flag_wipe": False, "flag_long_gap": False}))
    return pd.concat(rows, ignore_index=True)[D.HOURLY_COLUMNS]


def two_groups(items=8):
    return pd.DataFrame({"item_id": [1000 + i for i in range(items)], "group": ["metal" if i < items // 2 else "cloth" for i in range(items)]})


class TestHourlyTable(unittest.TestCase):
    def test_depth_units_count_units_within_5_and_10_percent_of_the_floor(self):
        # item 1: 5 units at 10g, 20 at 10.4g, 30 at 10.9g, 100 at 20g. Floor 10g: within +5% = 25 units, within +10% = 55 units.
        snap = mk([(1, 1, 5, 100000, "SHORT"), (2, 1, 20, 104000, "SHORT"), (3, 1, 30, 109000, "SHORT"), (4, 1, 100, 200000, "SHORT"), (5, 2, 7, 50000, "SHORT")])
        du = D.depth_units([snap], [1, 2]).set_index("item_id")
        self.assertEqual((du.loc[1, "u5"], du.loc[1, "u10"]), (25, 55))
        self.assertEqual((du.loc[2, "u5"], du.loc[2, "u10"]), (7, 7))
        self.assertTrue(D.depth_units([snap], [99]).empty)

    def test_hourly_long_has_the_columns_in_order_and_naive_utc_times(self):
        snap = mk([(1, 1, 5, 100000, "SHORT"), (2, 1, 30, 108000, "SHORT")])
        h = pd.DataFrame({"item_id": [1], "ts": [pd.Timestamp(T0, tz="UTC")], "floor_gold": [10.0], "units_listed": [35], "sold_est": [1.0],
                          "spike": [False], "blip": [True], "flag_wipe": [False], "flag_long_gap": [False]})
        out = D.hourly_long(h, [snap], [1])
        self.assertEqual(list(out.columns), D.HOURLY_COLUMNS)
        self.assertEqual(out.loc[0, "u10"], 35)
        self.assertTrue(bool(out.loc[0, "blip"]))
        self.assertIsNone(pd.to_datetime(out["ts"]).dt.tz)
        self.assertEqual(out.loc[0, "ts"], T0)
        self.assertTrue(D.hourly_long(pd.DataFrame(), [snap], [1]).empty)

    def test_days_split_on_the_us_eastern_midnight_and_merge_without_shrinking(self):
        hl = market(days=3, items=1, noise=0.0)
        days = D.split_by_day(hl)
        self.assertEqual(list(days), ["2026-06-01", "2026-06-02", "2026-06-03"])
        self.assertTrue(all(len(v) == 24 for v in days.values()))
        old = days["2026-06-02"]
        partial = old.head(6).assign(floor_gold=1.0)
        merged = D.merge_hourly(old, partial)
        self.assertEqual(len(merged), 24)                                        # the old, fuller day is kept
        self.assertTrue((merged.sort_values("ts").head(6)["floor_gold"] == 1.0).all())   # same hour: the new row wins
        self.assertEqual(len(D.merge_hourly(None, partial)), 6)
        self.assertTrue(D.merge_hourly(None, pd.DataFrame()).empty)

    def test_edges_of_the_window_are_trimmed_so_flags_are_written_with_full_context(self):
        hl = market(days=3, items=1, noise=0.0)
        first, last = hl["ts"].min(), hl["ts"].max()
        t = D.trim_edges(hl, first, last, hours=12)
        self.assertEqual(t["ts"].min(), first + pd.Timedelta(hours=12))
        self.assertEqual(t["ts"].max(), last - pd.Timedelta(hours=12))
        self.assertEqual(len(t), 72 - 24)


class TestPrepare(unittest.TestCase):
    def test_hours_are_eastern_and_bad_hours_are_not_usable(self):
        hl = market(days=2, items=1, noise=0.0)
        hl.loc[5, "blip"] = True
        hl.loc[6, "flag_wipe"] = True
        hl.loc[7, "flag_long_gap"] = True
        hl.loc[8, "spike"] = True                                                # a short dip or spike stays usable: it is what a buyer looks for
        p = W.prepare(hl)
        self.assertEqual(list(p["hour"].head(4)), [0, 1, 2, 3])
        self.assertEqual(int((~p["usable"]).sum()), 3)
        self.assertTrue(p.loc[8, "usable"])
        self.assertEqual(p["dow"].iloc[0], 0)                                    # 2026-06-01 is a Monday

    def test_depth_price_is_used_when_asked_and_falls_back_to_the_floor(self):
        hl = market(days=1, items=1, noise=0.0)
        hl.loc[0, "p15"] = np.nan
        p = W.prepare(hl, W.WindowConfig(price_col="p15"))
        self.assertAlmostEqual(p.loc[0, "price"], hl.loc[0, "floor_gold"])
        self.assertAlmostEqual(p.loc[1, "price"], hl.loc[1, "p15"])
        self.assertTrue(W.prepare(pd.DataFrame()).empty)

    def test_item_days_with_too_few_usable_hours_are_dropped(self):
        hl = market(days=2, items=1, noise=0.0)
        hl.loc[:9, "blip"] = True                                                # day 1 keeps 14 usable hours: below the 18 needed
        rel = W.relative_hours(W.prepare(hl))
        self.assertEqual(set(rel["day"]), {pd.Timestamp("2026-06-02")})


class TestHourOfDay(unittest.TestCase):
    def _run(self, hl, **kw):
        return W.analyze(hl, two_groups(hl["item_id"].nunique()), W.WindowConfig(**kw))

    def test_a_cheap_window_that_really_exists_is_found_and_holds_on_later_days(self):
        res = self._run(market(days=40, hour_effect={3: -0.08, 4: -0.08, 5: -0.08}))
        hs = res["hours_summary"].set_index("pool")
        r = hs.loc["everything"]
        self.assertEqual(r["window_start"], 3)
        self.assertGreater(r["saving"], 0.05)
        self.assertLess(r["saving"], 0.09)
        self.assertGreater(r["lo"], 0.0)
        self.assertEqual(r["verdict"], "holds on later days")
        self.assertGreater(r["window_steady"], 0.9)
        self.assertEqual(r["test_days"], 30)                                      # 40 days, the first 10 are learning only
        self.assertIn("group: metal", hs.index)
        self.assertEqual(hs.loc["group: cloth", "window_start"], 3)
        self.assertLess(res["profile"][3], -0.06)                                 # the profile shows the dip at 03h

    def test_pure_noise_does_not_look_like_a_pattern(self):
        res = self._run(market(days=40))
        r = res["hours_summary"].set_index("pool").loc["everything"]
        self.assertLess(abs(r["saving"]), 0.01)
        self.assertNotEqual(r["verdict"], "holds on later days")
        self.assertGreater(r["oracle"], r["saving"])                              # the oracle always finds something in noise: that is why it is not a result

    def test_a_cheap_hour_that_moves_every_day_is_not_a_pattern(self):
        rng = np.random.default_rng(5)
        starts = rng.integers(0, 21, 60)
        res = self._run(market(days=60, hour_effect=lambda d: {int(starts[d]) + k: -0.08 for k in range(3)}))
        r = res["hours_summary"].set_index("pool").loc["everything"]
        self.assertLess(r["saving"], 0.01)
        self.assertNotEqual(r["verdict"], "holds on later days")
        self.assertLess(r["window_steady"], 0.5)

    def test_too_few_days_says_so_instead_of_guessing(self):
        res = self._run(market(days=10, hour_effect={3: -0.08}))
        self.assertTrue(res["hours_summary"].empty)
        self.assertIn("Not enough days yet", W.format_report(res))

    def test_the_window_length_is_respected(self):
        res = self._run(market(days=40, hour_effect={3: -0.08, 4: -0.08}), window_hours=2)
        r = res["hours_summary"].set_index("pool").loc["everything"]
        self.assertEqual(r["window_start"], 3)
        self.assertGreater(r["saving"], 0.06)


class TestDayOfWeek(unittest.TestCase):
    def test_a_cheap_tuesday_is_seen_and_holds_on_later_weeks(self):
        hl = market(days=100, items=6, dow_effect={1: -0.06}, noise=0.02, day_walk=0.01)
        res = W.analyze(hl, None, W.WindowConfig())
        wp = res["weekday_profile"].set_index("weekday")
        self.assertEqual(wp["pct"].idxmin(), "Tue")
        self.assertLess(wp.loc["Tue", "pct"], -3.0)
        ws = res["weekday_summary"].set_index("pool").loc["everything"]
        self.assertGreaterEqual(ws["learned_days"], 6)
        self.assertGreater(ws["saving"], 0.03)
        self.assertEqual(ws["verdict"], "holds on later days")

    def test_no_weekday_effect_is_not_reported_as_one(self):
        res = W.analyze(market(days=100, items=6, noise=0.02, day_walk=0.01), None, W.WindowConfig())
        ws = res["weekday_summary"].set_index("pool").loc["everything"]
        self.assertNotEqual(ws["verdict"], "holds on later days")
        self.assertLess(np.nan_to_num(ws["saving"]), 0.02)

    def test_a_few_weeks_are_not_enough_and_the_report_says_how_long_it_needs(self):
        res = W.analyze(market(days=30, items=6, dow_effect={1: -0.06}), None, W.WindowConfig())
        self.assertTrue(res["weekday_summary"].empty)
        self.assertIn("about 7 weeks", W.format_report(res))


class TestResaleCheck(unittest.TestCase):
    def test_a_flat_market_loses_the_cut_and_a_rising_one_clears_the_bar(self):
        cfg = W.WindowConfig()
        flat = W.analyze(market(days=40, noise=0.0), None, cfg)["resale"]
        r = flat[(flat["pool"] == "everything") & (flat["k"] == 1)].iloc[0]
        self.assertAlmostEqual(r["margin_window"], -0.05, places=3)                # sell at the same price minus the 5% cut
        self.assertEqual(r["share_window"], 0.0)
        rising = W.analyze(market(days=40, noise=0.0, drift=np.log(1.12)), None, cfg)["resale"]
        r1 = rising[(rising["pool"] == "everything") & (rising["k"] == 1)].iloc[0]
        r2 = rising[(rising["pool"] == "everything") & (rising["k"] == 2)].iloc[0]
        self.assertAlmostEqual(r1["margin_window"], 1.12 * 0.95 - 1, places=3)
        self.assertEqual(r1["share_window"], 1.0)
        self.assertAlmostEqual(r2["margin_window"], 1.12 ** 2 * 0.95 - 1, places=3)

    def test_the_cut_and_the_profit_bar_come_from_the_config(self):
        res = W.analyze(market(days=40, noise=0.0, drift=np.log(1.12)), None, W.WindowConfig(cut=0.15, min_profit=0.05))
        r1 = res["resale"].query("pool == 'everything' and k == 1").iloc[0]
        self.assertAlmostEqual(r1["margin_window"], 1.12 * 0.85 - 1, places=3)
        self.assertEqual(r1["share_window"], 0.0)                                  # a 15% cut eats a 12% rise


class TestStatsAndReport(unittest.TestCase):
    def test_block_bootstrap_range_contains_the_mean_and_is_repeatable(self):
        x = np.random.default_rng(0).normal(0.02, 0.05, 60)
        m, lo, hi = W.block_bootstrap_mean(x)
        self.assertLess(lo, m)
        self.assertGreater(hi, m)
        self.assertEqual((m, lo, hi), W.block_bootstrap_mean(x))
        self.assertTrue(np.isnan(W.block_bootstrap_mean(np.array([]))[0]))

    def test_verdict_words(self):
        self.assertEqual(W.verdict(0.03, 0.01, 0.05, 30), "holds on later days")
        self.assertEqual(W.verdict(0.03, -0.01, 0.07, 30), "no pattern visible")
        self.assertEqual(W.verdict(-0.03, -0.05, -0.01, 30), "worse than random")
        self.assertEqual(W.verdict(0.03, 0.01, 0.05, 5), "too few test days")
        self.assertEqual(W.verdict(0.001, 0.0005, 0.002, 30), "no pattern visible")     # real but too small to matter

    def test_the_report_names_each_question_and_handles_no_data(self):
        res = W.analyze(market(days=40, hour_effect={3: -0.08, 4: -0.08, 5: -0.08}), two_groups(), W.WindowConfig())
        text = W.format_report(res)
        for part in ("HOUR OF DAY", "HOUR-OF-DAY BUY WINDOW", "DAY OF WEEK", "RESALE CHECK", "holds on later days", "03-06h", "walk-forward"):
            self.assertIn(part, text)
        self.assertIn("No usable hourly rows", W.format_report(W.analyze(pd.DataFrame())))
        self.assertEqual(W.analyze(pd.DataFrame())["hours_summary"].shape[0], 0)


class TestTool(unittest.TestCase):
    def test_day_files_are_found_in_order_and_other_files_ignored(self):
        names = ["engine/hourly/h_2026-11-03.parquet", "engine/item_daily.parquet", "engine/hourly/h_2026-11-01.parquet", "engine/hourly/readme.txt", "data/h_2026-11-02.parquet"]
        self.assertEqual([d for d, _ in __import__("buy_windows").list_day_files(names)], ["2026-11-01", "2026-11-03"])

    def test_run_prints_saves_and_passes_the_options_through(self):
        import buy_windows
        hl = market(days=40, hour_effect={3: -0.08, 4: -0.08, 5: -0.08})
        with tempfile.TemporaryDirectory() as d:
            args = buy_windows.argparse.Namespace(days=0, window=3, price="p15", cut=0.05, out=d)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = buy_windows.run(args, read_hourly=lambda n: (hl, ["2026-06-01", "2026-07-10"]), read_table=lambda n: two_groups())
            with open(os.path.join(d, "buy_windows.txt"), encoding="utf-8") as f:
                saved = f.read()
        self.assertEqual(res["cfg"].price_col, "p15")
        self.assertIn("depth price", saved)
        self.assertIn("Loaded 2 stored days", buf.getvalue())

    def test_run_with_nothing_stored_tells_you_what_to_do(self):
        import buy_windows
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = buy_windows.run(buy_windows.argparse.Namespace(days=0, window=3, price="floor_gold", cut=0.05, out="x"), read_hourly=lambda n: (pd.DataFrame(), []),
                                  read_table=lambda n: pd.DataFrame())
        self.assertEqual(res, {})
        self.assertIn("backfill", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
