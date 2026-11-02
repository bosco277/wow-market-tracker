import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import ledger as LG  # noqa: E402
from tests.test_forecast import world  # noqa: E402

START = pd.Timestamp("2026-10-01")
NOW = pd.Timestamp("2026-11-01T12:00:00Z")


def own_world(n=100, D=30, seed=7, noise=0.05):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 3.0, n)[:, None]
    return base + rng.normal(0, noise, (n, D))


def daily_frame(Y, hours=24, start=START, first_day=None):
    """item_daily-style frame from an items x days log10 matrix; NaN = no row. hours may be a scalar or an items x days matrix."""
    n, D = Y.shape
    days = pd.date_range(start, periods=D)
    H = np.full(Y.shape, hours) if np.isscalar(hours) else hours
    rows = [(i + 1, days[k].strftime("%Y-%m-%d"), int(H[i, k]), float(10 ** Y[i, k])) for i in range(n) for k in range(D) if np.isfinite(Y[i, k])]
    return pd.DataFrame(rows, columns=["item_id", "day", "hours", "price_day"])


def day(k):
    return (START + pd.Timedelta(days=k)).strftime("%Y-%m-%d")


_CAL = {}


def calibration(kind="noisy"):
    if kind not in _CAL:
        hist, _, _ = world(kind, n=300, days=700, seed=1)
        _CAL[kind] = (hist, LG.calibrate(hist))
    return _CAL[kind]


class TestOwnSeries(unittest.TestCase):
    def test_grid_keeps_only_complete_days_in_log_gold(self):
        Y = own_world(n=5, D=6)
        H = np.full(Y.shape, 24)
        H[0, 2] = 10                                  # an incomplete day
        d = daily_frame(Y, hours=H)
        d.loc[(d["item_id"] == 2) & (d["day"] == day(3)), "price_day"] = 0.0         # a non-positive price
        ids, days, G = LG.own_grid(d)
        self.assertEqual(ids, [1, 2, 3, 4, 5])
        self.assertEqual(len(days), 6)
        self.assertTrue(np.isnan(G[0, 2]))
        self.assertTrue(np.isnan(G[1, 3]))
        np.testing.assert_allclose(G[2], Y[2], atol=1e-9)
        self.assertIsNone(LG.own_grid(pd.DataFrame()))
        self.assertIsNone(LG.own_grid(d.assign(hours=3)))

    def test_origin_is_the_latest_well_covered_day(self):
        Y = own_world(n=40, D=10)
        Y[:, 9] = np.nan
        Y[:4, 9] = 1.0                                # day 9 has only 10% of the items
        _, _, G = LG.own_grid(daily_frame(Y))
        self.assertEqual(LG.origin_index(G), 8)
        Y[:25, 9] = 1.0                               # now 62%
        _, _, G = LG.own_grid(daily_frame(Y))
        self.assertEqual(LG.origin_index(G), 9)
        self.assertIsNone(LG.origin_index(np.full((3, 3), np.nan)))


class TestCalibration(unittest.TestCase):
    def test_tables_for_every_method_and_horizon_and_the_quantiles_are_ordered(self):
        _, cal = calibration()
        self.assertEqual(sorted(cal["betas"]), [1, 3, 7])
        self.assertEqual(len(cal["edges"]), 4)
        self.assertEqual(set(cal["q"]), {(m, h) for m in LG.METHODS for h in LG.HORIZONS})
        for key, q in cal["q"].items():
            self.assertEqual(q.shape, (5, 4), key)
            self.assertTrue((np.diff(q, axis=1) > 0).all(), key)      # q10 < q25 < q75 < q90 for every jumpiness level

    def test_a_method_that_forecasts_better_gets_narrower_ranges(self):
        _, cal = calibration()
        width = lambda m, h: float(np.mean(cal["q"][(m, h)][:, 3] - cal["q"][(m, h)][:, 0]))
        for h in LG.HORIZONS:
            self.assertLess(width("ema25", h), width("naive", h) * 0.9)

    def test_thin_history_gives_nothing(self):
        hist, _, _ = world("noisy", n=40, days=200)
        self.assertIsNone(LG.calibrate(hist))
        self.assertIsNone(LG.calibrate(pd.DataFrame()))


class TestGuesses(unittest.TestCase):
    def _made(self, daily):
        _, cal = calibration()
        return LG.make_forecasts(LG.own_grid(daily), cal, now=NOW)

    def test_rows_columns_and_ranges(self):
        d = daily_frame(own_world(n=60, D=30))
        led = self._made(d)
        self.assertEqual(list(led.columns), LG.COLUMNS)
        self.assertEqual(len(led), 60 * 3 * 3)
        self.assertEqual(set(led["made_day"]), {day(29)})
        self.assertEqual(sorted(led["target_day"].unique()), [day(30), day(32), day(36)])
        self.assertTrue(led["actual"].isna().all())
        self.assertTrue((led["lo80"] <= led["lo50"]).all() and (led["hi50"] <= led["hi80"]).all() and (led["lo80"] < led["hi80"]).all())
        self.assertTrue(led["jump_bin"].between(1, 5).all())
        self.assertEqual(set(led["made_at"]), {"2026-11-01T12:00:00Z"})
        naive = led[(led["method"] == "naive") & (led["horizon"] == 1)].set_index("item_id")
        o = daily_frame(own_world(n=60, D=30))
        last = o[o["day"] == day(29)].set_index("item_id")["price_day"]
        np.testing.assert_allclose(naive["guess"], last.reindex(naive.index))

    def test_history_length_decides_which_methods_an_item_gets(self):
        Y = own_world(n=40, D=30)
        Y[:10, :26] = np.nan            # 4 observed days
        Y[10:20, :24] = np.nan          # 6 observed days
        Y[20:30, :21] = np.nan          # 9 observed days
        led = self._made(daily_frame(Y))                                     # items 31-40 have all 30 days
        ids = lambda m: set(led[(led["method"] == m) & (led["horizon"] == 1)]["item_id"])
        self.assertEqual(ids("naive"), set(range(11, 41)))
        self.assertEqual(ids("ema25"), set(range(21, 41)))
        self.assertEqual(ids("kinked"), set(range(31, 41)))

    def test_nothing_after_the_origin_day_is_ever_used(self):
        Y = own_world(n=60, D=30)
        base = self._made(daily_frame(Y))
        extra = daily_frame(np.vstack([np.full((6, 1), 6.0), np.full((54, 1), np.nan)]), start=START + pd.Timedelta(days=30))   # day 30: only 10% of items, absurd prices
        both = pd.concat([daily_frame(Y), extra], ignore_index=True)
        later = self._made(both)
        self.assertEqual(set(later["made_day"]), {day(29)})
        cols = ["item_id", "horizon", "method", "guess", "lo80", "hi80", "jump_bin"]
        # the six items whose day 30 is already complete are not forecast one day ahead (their day is known); every other row is identical
        full = base
        base = base[~((base["horizon"] == 1) & (base["item_id"] <= 6))]
        self.assertEqual(len(base), len(later))
        pd.testing.assert_frame_equal(base[cols].reset_index(drop=True), later[cols].reset_index(drop=True))
        Y2 = Y.copy()
        Y2[:, 29] += 0.3
        self.assertFalse(np.allclose(self._made(daily_frame(Y2))["guess"].to_numpy(), full["guess"].to_numpy()))

    def test_a_target_day_that_is_already_complete_is_not_forecast(self):
        Y = own_world(n=50, D=31)
        Y[:20, 30] = np.nan                       # day 30 is complete for 60% of the items, so the origin is day 30 ...
        led = self._made(daily_frame(Y))
        self.assertEqual(set(led["made_day"]), {day(30)})
        Y = own_world(n=50, D=31)
        Y[:30, 30] = np.nan                       # ... and with only 40% complete the origin stays on day 29 and 20 items already have their day-30 price
        led = self._made(daily_frame(Y))
        self.assertEqual(set(led["made_day"]), {day(29)})
        h1 = led[(led["method"] == "naive") & (led["horizon"] == 1)]
        h3 = led[(led["method"] == "naive") & (led["horizon"] == 3)]
        self.assertEqual(len(h1), 30)
        self.assertEqual(len(h3), 50)

    def test_no_calibration_no_guesses(self):
        d = daily_frame(own_world(n=20, D=30))
        self.assertTrue(LG.make_forecasts(LG.own_grid(d), None).empty)
        self.assertTrue(LG.make_forecasts(None, calibration()[1]).empty)

    def test_a_row_is_written_once(self):
        d = daily_frame(own_world(n=30, D=30))
        led = self._made(d)
        again = led.assign(guess=led["guess"] * 2, made_at="later")
        m = LG.merge(led, again)
        self.assertEqual(len(m), len(led))
        pd.testing.assert_frame_equal(m.sort_values(LG.KEY).reset_index(drop=True), led.sort_values(LG.KEY).reset_index(drop=True))
        self.assertTrue(LG.merge(None, None).empty)


class TestGrading(unittest.TestCase):
    def setUp(self):
        self.Y = own_world(n=50, D=45)
        _, cal = calibration()
        self.led = LG.make_forecasts(LG.own_grid(daily_frame(self.Y[:, :30])), cal, now=NOW)

    def test_a_guess_is_graded_when_its_day_is_complete_and_not_before(self):
        d = daily_frame(self.Y[:, :31])
        led, n = LG.grade(self.led, d)
        self.assertEqual(n, int((self.led["horizon"] == 1).sum()))
        g = led[led["actual"].notna()]
        self.assertEqual(set(g["horizon"]), {1})
        np.testing.assert_allclose(g["actual"], 10 ** self.Y[g["item_id"].to_numpy() - 1, 30], rtol=1e-9)
        H = np.full((50, 31), 24)
        H[:, 30] = 10                                   # day 30 only partly seen: not graded
        led2, n2 = LG.grade(self.led, daily_frame(self.Y[:, :31], hours=H))
        self.assertEqual(n2, 0)
        self.assertTrue(led2["actual"].isna().all())

    def test_graded_rows_never_change_and_everything_is_graded_in_time(self):
        led, _ = LG.grade(self.led, daily_frame(self.Y[:, :31]))
        d2 = daily_frame(self.Y[:, :40])
        d2.loc[d2["day"] == day(30), "price_day"] *= 7.0                        # a restated price must not rewrite a graded row
        led2, n2 = LG.grade(led, d2)
        done = led["actual"].notna()
        pd.testing.assert_series_equal(led2.loc[done, "actual"], led.loc[done, "actual"])
        self.assertEqual(n2, int((self.led["horizon"] >= 3).sum()))
        self.assertTrue(led2["actual"].notna().all())

    def test_empty_inputs_are_harmless(self):
        e, n = LG.grade(None, pd.DataFrame())
        self.assertTrue(e.empty and n == 0)
        led, n = LG.grade(self.led, pd.DataFrame())
        self.assertEqual(n, 0)
        self.assertEqual(len(led), len(self.led))


class TestScoreboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run the ledger day by day over a stationary-noise world: forecast at each origin from the data up to it, grade with what has arrived."""
        _, cal = calibration()
        Y = own_world(n=100, D=40, seed=9)
        led = LG._empty()
        for k in range(14, 33):
            cut = daily_frame(Y[:, :k + 1])
            led = LG.merge(led, LG.make_forecasts(LG.own_grid(cut), cal, now=NOW))
            led, _ = LG.grade(led, cut)
        led, _ = LG.grade(led, daily_frame(Y))
        cls.led, cls.sb = led, LG.scoreboard(led)

    def test_every_guess_was_written_before_its_day_and_graded_after(self):
        led = self.led
        self.assertTrue((pd.to_datetime(led["target_day"]) > pd.to_datetime(led["made_day"])).all())
        self.assertEqual(self.sb["graded"], int(led["actual"].notna().sum()))
        self.assertGreater(self.sb["graded"], 5000)

    def test_ranges_hold_on_a_series_like_the_one_they_were_calibrated_on(self):
        o = self.sb["overall"]
        self.assertTrue(o["hit80"].between(0.72, 0.88).all(), o[["horizon", "method", "hit80"]].values.tolist())
        self.assertTrue(o["hit50"].between(0.42, 0.58).all(), o[["horizon", "method", "hit50"]].values.tolist())

    def test_smoothing_beats_staying_put_on_noise_and_naive_scores_one(self):
        o = self.sb["overall"].set_index(["horizon", "method"])
        for h in LG.HORIZONS:
            self.assertAlmostEqual(o.loc[(h, "naive"), "vs_naive"], 1.0)
            self.assertLess(o.loc[(h, "ema25"), "vs_naive"], 0.9)
            self.assertLess(o.loc[(h, "kinked"), "vs_naive"], 0.9)

    def test_tables_are_ordered_and_labelled(self):
        o = self.sb["overall"]
        self.assertEqual(list(o["method"][:3]), list(LG.METHODS))
        self.assertEqual(sorted(o["horizon"].unique()), [1, 3, 7])
        lv = self.sb["levels"]
        self.assertTrue(set(lv["level"]) <= set(["under 1g", "1g to 10g", "10g to 100g", "100g and up"]))
        for h, t in lv.groupby("horizon"):
            order = [["under 1g", "1g to 10g", "10g to 100g", "100g and up"].index(x) for x in t["level"]]
            self.assertEqual(order, sorted(order))
        self.assertIn("waiting", self.sb)

    def test_empty_ledger(self):
        self.assertEqual(LG.scoreboard(LG._empty())["graded"], 0)
        self.assertEqual(LG.scoreboard(None)["graded"], 0)
        sb = LG.scoreboard(self.led.assign(actual=np.nan))
        self.assertEqual(sb["graded"], 0)
        self.assertGreater(len(sb["waiting"]), 0)


class TestRunner(unittest.TestCase):
    def _run(self, tables, upload=False, saved=None):
        import lake
        old = lake._store_private
        lake._store_private = lambda t, msg, up: (saved.append((t, msg, up)) if saved is not None else None) or True
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = lake.run_forecast_ledger(None, type("A", (), {"upload": upload})(), read_table=lambda name: tables.get(name, pd.DataFrame()))
        finally:
            lake._store_private = old
        return res, buf.getvalue()

    def _tables(self, D=30):
        hist, _ = calibration()
        return {"item_daily": daily_frame(own_world(n=60, D=D)), "market_history": hist}

    def test_first_run_writes_guesses_and_says_what_is_waiting(self):
        saved = []
        res, out = self._run(self._tables(), upload=True, saved=saved)
        for s in ("FORECAST LEDGER", "Origin day: 2026-10-30", "PRIOR 80% RANGE WIDTH", "WRITTEN NOW", "naive", "ema25", "kinked", "Nothing graded yet", "STILL WAITING"):
            self.assertIn(s, out)
        self.assertEqual(res["added"], 60 * 9)
        self.assertEqual(len(saved), 1)
        self.assertEqual(list(saved[0][0]), ["forecast_ledger"])
        self.assertEqual(len(saved[0][0]["forecast_ledger"]), 60 * 9)
        self.assertFalse(any(line.startswith("==") for line in out.splitlines()))

    def test_second_run_the_same_day_adds_nothing_and_uploads_nothing(self):
        t = self._tables()
        first, _ = self._run(t)
        t["forecast_ledger"] = first["ledger"]
        saved = []
        res, out = self._run(t, upload=True, saved=saved)
        self.assertEqual(res["added"], 0)
        self.assertEqual(saved, [])
        self.assertIn("Nothing new was written", out)
        self.assertIn("Nothing changed, so nothing was uploaded", out)

    def test_later_days_grade_old_guesses_and_print_the_scoreboard(self):
        t = self._tables(D=30)
        first, _ = self._run(t)
        t2 = self._tables(D=40)
        t2["forecast_ledger"] = first["ledger"]
        saved = []
        res, out = self._run(t2, upload=True, saved=saved)
        self.assertGreater(res["graded"], 0)
        self.assertEqual(res["graded"], 60 * 9)               # all three horizons of the first origin are now in the past
        for s in ("SCOREBOARD", "hit80", "vs_naive", "BY PRICE LEVEL"):
            self.assertIn(s, out)
        self.assertEqual(len(saved), 1)

    def test_not_ticking_upload_warns_that_the_guesses_are_lost(self):
        _, out = self._run(self._tables())
        self.assertIn("Not saved", out)

    def test_missing_tables_say_what_to_run(self):
        res, out = self._run({})
        self.assertEqual(res, {})
        self.assertIn("mode daily", out)
        t = self._tables()
        res, out = self._run({"item_daily": t["item_daily"]})
        self.assertIn("mode undermine", out)
        self.assertEqual(res["added"], 0)

    def test_workflow_lists_the_mode_and_runs_without_a_store(self):
        for rel in (".github/workflows/pick_items.yml", "tools/pick_items.yml"):
            with open(os.path.join(_root, rel)) as f:
                text = f.read()
            self.assertIn("forecast_ledger", text.split("options:")[1].splitlines()[0])
            self.assertIn("inputs.mode != 'forecast_ledger'", text)
        with open(os.path.join(_root, "tools", "lake.py")) as f:
            src = f.read()
        line = [l for l in src.splitlines() if l.strip().startswith("store = None if args.command in")][0]
        self.assertIn("forecast_ledger", line)


if __name__ == "__main__":
    unittest.main()
