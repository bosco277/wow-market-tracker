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
from engine import forecast as FO  # noqa: E402


def world(kind, n=96, days=700, seed=1):
    """Synthetic daily history. spiky = a slowly drifting level plus one-day spikes of +0.4 (log10) on 6% of days; mixed = half the items calm, half jumpy. Returns (history frame in the stored layout, groups frame, true log10 matrix).
    noisy  = a fixed level plus day-to-day noise; walk = a random walk; weekly = a fixed level plus a strong weekday rhythm;
    group  = a fixed level plus an AR(1) shock shared by the item's group of 12 plus noise of its own."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=days)   # starts on a Monday
    wd = dates.dayofweek.to_numpy()
    base = rng.uniform(0.5, 3.0, n)[:, None]
    if kind == "noisy":
        Y = base + rng.normal(0, 0.05, (n, days))
    elif kind == "walk":
        Y = base + np.cumsum(rng.normal(0, 0.02, (n, days)), axis=1)
    elif kind == "weekly":
        rhythm = np.array([0, 0.08, -0.04, -0.04, 0, 0.02, -0.02])
        Y = base + rhythm[wd][None, :] + rng.normal(0, 0.02, (n, days))
    elif kind == "group":
        G = n // 12
        common = np.zeros((G, days))
        for t in range(1, days):
            common[:, t] = 0.8 * common[:, t - 1] + rng.normal(0, 0.06, G)
        Y = base + np.repeat(common, 12, axis=0) + rng.normal(0, 0.05, (n, days))
    elif kind == "spiky":
        Y = base + np.cumsum(rng.normal(0, 0.01, (n, days)), axis=1) + (rng.random((n, days)) < 0.06) * 0.4
    elif kind == "mixed":
        Y = base + rng.normal(0, 1, (n, days)) * np.where(np.arange(n) < n // 2, 0.02, 0.15)[:, None]
    else:
        raise ValueError(kind)
    ids = np.arange(1, n + 1) + 1000
    hist = pd.DataFrame({"item_id": np.repeat(ids, days), "day": np.tile(dates.strftime("%Y-%m-%d"), n), "price": (10 ** Y * 10000.0).ravel(), "quantity": 100.0})
    groups = pd.DataFrame({"item_id": ids, "group": [f"Fam: g{i // 12}" for i in range(n)], "family": "Fam"})
    return hist, groups, Y


def run(kind, sarima=False, **kw):
    hist, groups, _ = world(kind, **kw)
    return FO.bake_off(hist, groups, "price", sarima=sarima)


def row(res, model, h):
    t = res["summary"]
    return t[(t["model"] == model) & (t["horizon"] == h)].iloc[0]


class TestModelsAreCausal(unittest.TestCase):
    def test_fixed_models_ignore_the_future(self):
        _, _, Y = world("weekly", n=20, days=300)
        wk = pd.date_range("2022-01-03", periods=300).dayofweek.to_numpy()
        a, _ = FO.static_forecasts(Y, wk)
        Y2 = Y.copy()
        Y2[:, 200:] += 5.0
        b, _ = FO.static_forecasts(Y2, wk)
        for m in FO.STATIC:
            for h in FO.HORIZONS:
                np.testing.assert_allclose(a[m][h][:, :200], b[m][h][:, :200], err_msg=f"{m} h={h} used days after the origin")

    def test_pooled_fit_ignores_days_after_its_rows(self):
        _, groups, Y = world("group", n=48, days=400)
        wk = pd.date_range("2022-01-03", periods=400).dayofweek.to_numpy()
        gi = np.repeat(np.arange(4), 12)
        h, last = 3, 250
        out = []
        for M in (Y, np.where(np.arange(400)[None, :] > last + h, Y + 9.0, Y)):
            _, base = FO.static_forecasts(M, wk)
            feat = FO._features(M, base, gi)
            obs = np.isfinite(M)
            valid = obs & np.concatenate([obs[:, h:], np.zeros((48, h), bool)], axis=1)
            valid[:, :FO.WARMUP] = False
            out.append(FO.pooled_fit(M, base, feat, wk, valid, h, np.arange(0, last + 1, 2), kind="pooled_grp"))
        self.assertIsNotNone(out[0])
        np.testing.assert_allclose(out[0], out[1])


class TestScoring(unittest.TestCase):
    def test_smoothing_beats_naive_on_noise_and_loses_on_a_random_walk(self):
        r = run("noisy")
        self.assertLess(row(r, "ema10", 1)["vs_naive"], 0.85)
        self.assertLess(row(r, "median7", 1)["vs_naive"], 0.95)
        w = run("walk")
        self.assertGreater(row(w, "ema10", 7)["vs_naive"], 1.1)
        self.assertEqual(w["best"][7], "naive")

    def test_weekday_model_finds_the_rhythm_and_gets_named_best(self):
        r = run("weekly")
        for h in (1, 3):
            self.assertLess(row(r, "weekday", h)["vs_naive"], 0.7)
        self.assertIn(r["best"][1], ("weekday", "pooled", "kinked", "pooled_grp"))
        self.assertNotEqual(r["best"][3], "naive")

    def test_peers_help_when_a_shared_shock_drives_the_group(self):
        r = run("group")
        self.assertLess(row(r, "pooled_grp", 1)["avg_pct"], row(r, "pooled", 1)["avg_pct"] * 0.97)

    def test_kinked_stays_put_for_small_moves_and_pulls_back_after_spikes(self):
        r = run("spiky")
        for h in (1, 3):
            k, p = row(r, "kinked", h), row(r, "pooled", h)
            self.assertLess(k["vs_naive"], 0.75, f"h={h}")
            self.assertLess(k["avg_pct"], p["avg_pct"], f"h={h}")
            self.assertLessEqual(k["miss_pct"], p["miss_pct"], f"h={h}")

    def test_results_by_jumpiness_split_calm_from_jumpy_items(self):
        b = run("mixed")["bins"]
        one = b[b["horizon"] == 1].sort_values("jumpiness")
        self.assertEqual(len(one), FO.VOL_BINS)
        self.assertGreater(one["naive_pct"].iloc[-1], 2 * one["naive_pct"].iloc[0])
        self.assertTrue((one["jump_pct"].diff().dropna() >= 0).all())
        self.assertTrue(set(one["best"]).issubset({"naive", "week_ago", "median7", "ema50", "ema25", "ema10", "weekday", "pooled", "kinked", "pooled_grp"}))

    def test_ranges_are_honest_on_stationary_noise(self):
        r = run("noisy")
        for m in ("naive", "ema25", "weekday", "pooled"):
            for h in FO.HORIZONS:
                x = row(r, m, h)
                self.assertTrue(0.40 <= x["hit50"] <= 0.60, f"{m} h={h} hit50 {x['hit50']:.2f}")
                self.assertTrue(0.72 <= x["hit80"] <= 0.88, f"{m} h={h} hit80 {x['hit80']:.2f}")

    def test_summary_columns_and_sanity(self):
        r = run("weekly")
        t = r["summary"]
        for c in ("miss_pct", "avg_pct", "vs_naive", "wape", "wape_med", "bias_pct", "track", "rmse_pct", "p95_pct", "big_pct", "dir_hit", "dir_n", "hit50", "hit80", "width80_pct"):
            self.assertIn(c, t.columns)
        n = t[t["model"] == "naive"]
        np.testing.assert_allclose(n["vs_naive"], 1.0)
        self.assertTrue((t["rmse_pct"] >= t["avg_pct"] - 1e-9).all())      # RMSE is never below the mean absolute miss
        self.assertTrue((t["p95_pct"] >= t["miss_pct"]).all())
        self.assertTrue(((t["track"] >= -1) & (t["track"] <= 1)).all())
        self.assertTrue((t.groupby("horizon")["rows"].nunique() == 1).all())  # every model scored on the same rows

    def test_misses_grow_with_the_horizon_on_a_random_walk(self):
        n = run("walk")["summary"].query("model == 'naive'")["avg_pct"].values
        self.assertTrue((n[:-1] < n[1:]).all())

    def test_bias_sign_follows_the_error_convention(self):
        hist, groups, Y = world("noisy", n=48)
        # a slow upward drift: the real price runs above any average of the past
        n, T = Y.shape
        Y = Y + np.linspace(0, 0.6, T)[None, :]
        hist["price"] = (10 ** Y * 10000.0).ravel()
        r = FO.bake_off(hist, groups, "price", sarima=False)
        self.assertGreater(row(r, "ema10", 7)["bias_pct"], 0)
        self.assertGreater(row(r, "ema10", 7)["track"], 0.1)

    def test_young_items_are_scored_only_after_their_first_56_days(self):
        hist, _, _ = world("noisy", n=48, days=520)
        young = hist[(hist["item_id"] == 1001) & (pd.to_datetime(hist["day"]) >= pd.Timestamp("2022-12-01"))].iloc[:150]
        both = pd.concat([hist[hist["item_id"] != 1001], young])
        ids, days, Y = FO.build_grid(both)
        i = ids.index(1001)
        self.assertEqual(int(np.isfinite(Y[i]).sum()), 150)
        res = FO.walk_forward(Y, days, None)
        a = res["acc"][("naive", 1)]
        self.assertGreater(a.cnt[i], 0)
        self.assertLessEqual(a.cnt[i], 150 - FO.WARMUP)
        self.assertGreater(a.cnt[ids.index(1002)], 3 * a.cnt[i] / 2)    # the old items are scored on many more days

    def test_too_little_history_returns_nothing(self):
        hist, groups, _ = world("noisy", n=24, days=150)
        self.assertEqual(FO.bake_off(hist, groups, "price", sarima=False), {})
        self.assertEqual(FO.bake_off(pd.DataFrame(), None), {})

    def test_stock_metric_and_missing_groups(self):
        hist, _, _ = world("noisy", n=48, days=400)
        r = FO.bake_off(hist, None, "stock", sarima=False)
        self.assertFalse(r["grouped"])
        self.assertNotIn("pooled_grp", set(r["summary"]["model"]))

    def test_family_table(self):
        r = run("weekly")
        f = r["family"]
        self.assertTrue(len(f) > 0)
        self.assertTrue(set(f["best"]).issubset(set(r["summary"]["model"])))
        self.assertTrue((f["vs_naive"] <= 1.0 + 1e-9).all())    # the best is naive (1.0) or something at least 3% better


class TestSarimaInTheBakeOff(unittest.TestCase):
    def test_sarima_is_scored_on_the_same_cases_and_finds_a_weekday_rhythm(self):
        r = run("weekly", sarima=True, n=36, days=520)
        t = r["summary"]
        self.assertIn("sarima", set(t["model"]))
        self.assertTrue((t.groupby("horizon")["rows"].nunique() == 1).all())
        for h in (1, 3):
            self.assertLess(row(r, "sarima", h)["vs_naive"], 0.75, f"h={h}")
        self.assertEqual(sum(sum(i["specs"].values()) for i in r["sarima_info"]) + sum(i["failed"] for i in r["sarima_info"]), 36 * FO.FOLDS)

    def test_fitted_only_table_leaves_out_items_that_fell_back(self):
        hist, groups, _ = world("weekly", n=36, days=520)
        late = hist["item_id"].isin(range(1001, 1007)) & (pd.to_datetime(hist["day"]) < pd.Timestamp("2022-01-03") + pd.Timedelta(days=400))
        r = FO.bake_off(hist[~late], groups, "price", sarima=True)
        full, fit = r["summary"], r["summary_fit"]
        a = full[(full["model"] == "sarima") & (full["horizon"] == 1)].iloc[0]
        b = fit[(fit["model"] == "sarima") & (fit["horizon"] == 1)].iloc[0]
        self.assertLess(b["rows"], a["rows"])
        self.assertGreater(sum(i["failed"] for i in r["sarima_info"]), 0)
        self.assertTrue((fit.groupby("horizon")["rows"].nunique() == 1).all())

    def test_sarima_does_not_beat_naive_on_a_random_walk(self):
        r = run("walk", sarima=True, n=36, days=520)
        self.assertGreater(row(r, "sarima", 7)["vs_naive"], 0.97)

    def test_without_scipy_the_rest_still_runs_and_says_why(self):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "scipy":
                raise ImportError("no scipy")
            return real(name, *a, **k)
        hist, groups, _ = world("noisy", n=36, days=420)
        builtins.__import__ = fake
        try:
            r = FO.bake_off(hist, groups, "price", sarima=True)
        finally:
            builtins.__import__ = real
        self.assertNotIn("sarima", set(r["summary"]["model"]))
        self.assertIn("scipy", r["sarima_note"])


class TestRunner(unittest.TestCase):
    def test_sarima_is_left_out_unless_asked_for(self):
        import lake
        hist, groups, _ = world("weekly", n=48, days=420)
        tables = {"market_history": hist, "item_groups": groups}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_forecast_test(None, type("A", (), {"ids": ""})(), read_table=lambda name: tables.get(name, pd.DataFrame()))
        out = buf.getvalue()
        self.assertNotIn("sarima", set(res["price"]["summary"]["model"]))
        self.assertNotIn("SARIMA CHECK", out)
        self.assertIn("put the word sarima in item_ids", out)

    def test_forecast_test_mode_prints_every_model_and_a_verdict(self):
        import lake
        hist, groups, _ = world("weekly", n=48, days=420)
        tables = {"market_history": hist, "item_groups": groups}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lake.run_forecast_test(None, type("A", (), {"ids": "sarima"})(), read_table=lambda name: tables.get(name, pd.DataFrame()))
        out = buf.getvalue()
        for s in ("FORECAST BAKE-OFF", "naive", "ema25", "weekday", "sarima", "kinked", "pooled_grp", "BEST GUESS", "BEST GUESS BY PRICE LEVEL", "hit80", "BY HOW JUMPY", "SARIMA CHECK"):
            self.assertIn(s, out)
        self.assertFalse(any(line.startswith("==") for line in out.splitlines()))

    def test_workflow_lists_the_mode_and_skips_saving_results(self):
        for rel in (".github/workflows/pick_items.yml", "tools/pick_items.yml"):
            with open(os.path.join(_root, rel)) as f:
                text = f.read()
            self.assertIn("forecast_test", text.split("options:")[1].splitlines()[0])
            self.assertIn("inputs.mode != 'forecast_test'", text)
            self.assertIn("boto3 scipy", text)      # SARIMA needs scipy on the runner

    def test_missing_tables_say_what_to_run(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_forecast_test(None, type("A", (), {})(), read_table=lambda name: pd.DataFrame())
        self.assertEqual(res, {})
        self.assertIn("mode undermine", buf.getvalue())


# ---------------------------------------------------------------- hourly


def hworld(kind, n=100, hours=216, seed=3, hole=True):
    """Synthetic hourly history (US Eastern wall clock). cycle = a fixed level plus a 24-hour sine; walk = a random walk; noise = a fixed level plus noise. A hole of 25 hours near the start
    (like the partial day in the real data). Returns (ids, hourly index, log10 grid)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-25 20:00", periods=hours, freq="h")
    hr = idx.hour.to_numpy()
    base = rng.uniform(0.5, 3.0, n)[:, None]
    if kind == "cycle":
        Y = base + (0.04 * np.sin(2 * np.pi * (hr - 3) / 24))[None, :] + rng.normal(0, 0.01, (n, hours))
    elif kind == "walk":
        Y = base + np.cumsum(rng.normal(0, 0.01, (n, hours)), axis=1)
    elif kind == "noise":
        Y = base + rng.normal(0, 0.03, (n, hours))
    else:
        raise ValueError(kind)
    if hole:
        Y[:, 5:30] = np.nan
    return list(range(1, n + 1)), idx, Y


def hrun(kind, **kw):
    ids, idx, Y = hworld(kind, **kw)
    groups = pd.DataFrame({"item_id": ids, "group": [f"Fam: g{i // 10}" for i in range(len(ids))], "family": "Fam"})
    return FO.bake_off_grid(ids, idx, Y, groups, FO.HOURLY_HORIZONS, False, FO.HOURLY)


class TestHourlyConfig(unittest.TestCase):
    def test_the_daily_settings_are_still_the_original_constants(self):
        d = FO.DAILY
        self.assertEqual((d.season, d.ma, d.vol_window, d.warmup, d.min_item, d.min_span), (7, 7, 14, FO.WARMUP, FO.MIN_ITEM_DAYS, FO.MIN_SPAN))
        self.assertEqual(d.static, FO.STATIC)
        self.assertIsNone(d.slow)
        self.assertEqual(dict(d.alphas), FO.EMA_ALPHAS)

    def test_hourly_phase_is_the_clock_hour_and_daily_phase_is_the_weekday(self):
        idx = pd.date_range("2026-10-26 00:00", periods=30, freq="h")           # a Monday
        np.testing.assert_array_equal(FO.HOURLY.phase_of(idx), idx.hour.to_numpy())
        np.testing.assert_array_equal(FO.DAILY.phase_of(idx), np.zeros(30, int) + np.where(idx.day == 26, 0, 1))

    def test_hourly_models_ignore_the_future(self):
        ids, idx, Y = hworld("cycle", n=20, hours=300, hole=False)
        wk = FO.HOURLY.phase_of(idx)
        a, base_a = FO.static_forecasts(Y, wk, FO.HOURLY_HORIZONS, FO.HOURLY)
        Y2 = Y.copy()
        Y2[:, 200:] += 5.0
        b, _ = FO.static_forecasts(Y2, wk, FO.HOURLY_HORIZONS, FO.HOURLY)
        for m in FO.HOURLY.static:
            for h in FO.HOURLY_HORIZONS:
                np.testing.assert_allclose(a[m][h][:, :200], b[m][h][:, :200], err_msg=f"{m} h={h} used hours after the origin")
        self.assertIn("Lslow", base_a)

    def test_day_ago_is_the_same_clock_hour_yesterday(self):
        ids, idx, Y = hworld("cycle", n=10, hours=120, hole=False)
        a, _ = FO.static_forecasts(Y, FO.HOURLY.phase_of(idx), FO.HOURLY_HORIZONS, FO.HOURLY)
        # a guess made at hour t for hour t+4 is the value at the same clock hour a day before the target: t+4-24 = t-20
        np.testing.assert_allclose(a["week_ago"][4][:, 20:], Y[:, :-20])
        np.testing.assert_allclose(a["week_ago"][24], Y)                      # a 24-hour horizon: the same clock hour yesterday is today's reading
        self.assertTrue(np.isnan(a["week_ago"][4][:, :20]).all())

    def test_burn_in_counts_real_readings_so_a_hole_cannot_use_it_up(self):
        import dataclasses
        ids, idx, Y = hworld("noise")                                         # 25 empty hours near the start
        by_obs = FO.walk_forward(Y, idx, None, (1,), cfg=FO.HOURLY)["acc"][("naive", 1)]
        by_slot = FO.walk_forward(Y, idx, None, (1,), cfg=dataclasses.replace(FO.HOURLY, age_by_obs=False))["acc"][("naive", 1)]
        self.assertTrue(FO.HOURLY.age_by_obs)
        self.assertFalse(FO.DAILY.age_by_obs)
        # 5 readings before the hole + 43 after it are needed: nothing can be scored before slot 73
        self.assertLessEqual(by_obs.cnt.max(), len(idx) - 73)
        self.assertLess(by_obs.cnt.sum(), by_slot.cnt.sum())

    def test_the_slow_anchor_feature_exists_only_for_hourly(self):
        ids, idx, Y = hworld("noise", n=12, hours=120, hole=False)
        _, base = FO.static_forecasts(Y, FO.HOURLY.phase_of(idx), (1,), FO.HOURLY)
        self.assertIn("d3", FO._features(Y, base, None))
        w = pd.date_range("2022-01-03", periods=120)
        _, base_d = FO.static_forecasts(Y, w.dayofweek.to_numpy(), (1,))
        self.assertNotIn("d3", FO._features(Y, base_d, None))


class TestHourlyBakeOff(unittest.TestCase):
    def test_a_daily_cycle_is_found_and_the_methods_that_can_learn_it_win(self):
        r = hrun("cycle")
        t = r["summary"]
        row_ = lambda m, h: t[(t["model"] == m) & (t["horizon"] == h)].iloc[0]
        self.assertLess(row_("pooled", 1)["vs_naive"], 0.8)
        self.assertLess(row_("hour_of_day", 12)["vs_naive"], 0.85)
        self.assertNotEqual(r["best"][1], "naive")
        self.assertNotEqual(r["best"][12], "naive")

    def test_on_a_random_walk_nothing_beats_naive_and_the_ranges_hold(self):
        r = hrun("walk")
        self.assertEqual(set(r["best"].values()), {"naive"})
        t = r["summary"]
        nv = t[t["model"] == "naive"]
        self.assertTrue(nv["hit80"].between(0.72, 0.88).all(), nv["hit80"].tolist())
        self.assertTrue(nv["hit50"].between(0.42, 0.58).all(), nv["hit50"].tolist())

    def test_smoothing_helps_on_noise(self):
        r = hrun("noise")
        t = r["summary"]
        self.assertLess(t[(t["model"] == "ema10") & (t["horizon"] == 1)]["vs_naive"].iloc[0], 0.85)

    def test_display_names_replace_the_weekly_ones(self):
        r = hrun("noise")
        names = set(r["summary"]["model"])
        for m in ("day_ago", "median6", "hour_of_day", "ema04", "naive", "pooled", "kinked", "pooled_grp"):
            self.assertIn(m, names)
        for m in ("week_ago", "median7", "weekday", "sarima"):
            self.assertNotIn(m, names)
        self.assertTrue(set(r["best"].values()) <= names)

    def test_a_day_ago_guess_at_24_hours_is_exactly_naive(self):
        r = hrun("cycle")
        t = r["summary"]
        a, b = t[(t["model"] == "day_ago") & (t["horizon"] == 24)].iloc[0], t[(t["model"] == "naive") & (t["horizon"] == 24)].iloc[0]
        self.assertAlmostEqual(a["avg_pct"], b["avg_pct"])
        self.assertAlmostEqual(a["vs_naive"], 1.0)

    def test_every_method_is_scored_on_the_same_cases_even_when_a_fit_has_too_few_rows(self):
        ids, idx, Y = hworld("cycle", n=40, hours=200)          # too thin for the pooled fit in the first stretch at 24 hours
        r = FO.bake_off_grid(ids, idx, Y, None, FO.HOURLY_HORIZONS, False, FO.HOURLY)
        t = r["summary"]
        for h, th in t.groupby("horizon"):
            self.assertEqual(th["rows"].nunique(), 1, f"h={h}: {th[['model', 'rows']].values.tolist()}")

    def test_horizons_are_1_4_12_and_24_hours(self):
        r = hrun("noise")
        self.assertEqual(sorted(r["summary"]["horizon"].unique()), [1, 4, 12, 24])

    def test_the_hole_does_not_leak_into_the_scored_cases(self):
        ids, idx, Y = hworld("noise")
        res = FO.walk_forward(Y, idx, None, FO.HOURLY_HORIZONS, cfg=FO.HOURLY)
        a = res["acc"][("naive", 1)]
        self.assertTrue(np.isfinite(a.cat("E")).all())
        self.assertLessEqual(a.cnt.max(), len(idx) - FO.HOURLY.warmup)

    def test_too_few_hours_returns_nothing(self):
        ids, idx, Y = hworld("noise", hours=80, hole=False)
        self.assertEqual(FO.bake_off_grid(ids, idx, Y, None, FO.HOURLY_HORIZONS, False, FO.HOURLY), {})

    def test_sarima_is_never_run_on_hourly_data(self):
        ids, idx, Y = hworld("noise", n=60)
        r = FO.bake_off_grid(ids, idx, Y, None, FO.HOURLY_HORIZONS, True, FO.HOURLY)
        self.assertNotIn("sarima", set(r["summary"]["model"]))
        self.assertEqual(r["sarima_info"], [])


class TestPriceTiers(unittest.TestCase):
    def test_levels_follow_the_median_price(self):
        Y = np.array([[-1.0, -1.0, np.nan], [0.5, 0.5, 0.5], [1.5, 1.5, 9.0], [3.0, 3.0, 3.0], [np.nan, np.nan, np.nan]])
        t = FO.price_tiers(Y)
        self.assertEqual(t.tolist(), ["under 1g", "1g to 10g", "10g to 100g", "100g and up", ""])

    def test_the_table_is_in_price_order_with_display_names_and_is_added_for_both_steps(self):
        ids, idx, Y = hworld("walk", n=120)
        tiers = FO.price_tiers(Y)
        r = FO.bake_off_grid(ids, idx, Y, None, FO.HOURLY_HORIZONS, False, FO.HOURLY, tiers=tiers)
        t = r["tiers"]
        self.assertIn("price_level", t.columns)
        self.assertEqual(set(t["price_level"]), set(tiers) - {""})
        for h, th in t.groupby("horizon"):
            order = [FO.TIER_ORDER.index(x) for x in th["price_level"]]
            self.assertEqual(order, sorted(order))
        self.assertTrue(set(t["best"]) <= set(r["summary"]["model"]))
        self.assertNotIn("tiers", FO.bake_off_grid(ids, idx, Y, None, FO.HOURLY_HORIZONS, False, FO.HOURLY))
        d = run("walk", n=48, days=420)
        self.assertIn("tiers", d)

    def test_a_tier_with_a_different_behaviour_is_told_apart(self):
        # cheap items bounce around by +-0.1 log10 (noise: smoothing helps); dear items walk (naive wins)
        rng = np.random.default_rng(11)
        ids, idx, _ = hworld("walk", n=100)
        n, T = 100, len(idx)
        Y = np.where(np.arange(n)[:, None] < 50, 0.3 + rng.normal(0, 0.1, (n, T)), 2.0 + np.cumsum(rng.normal(0, 0.01, (n, T)), axis=1))
        r = FO.bake_off_grid(ids, idx, Y, None, FO.HOURLY_HORIZONS, False, FO.HOURLY, tiers=FO.price_tiers(Y))
        t = r["tiers"][r["tiers"]["horizon"] == 1].set_index("price_level")
        self.assertLess(t.loc["1g to 10g", "vs_naive"], 0.9)
        self.assertEqual(t.loc["100g and up", "best"], "naive")


def _snap(ts, items, price_scale=1.0, first_id=0):
    """A small commodity book: three listings per item (floor, +10%, +30%); items = {item_id: floor in gold}."""
    from tests.helpers import mk
    rows, a = [], first_id
    for it, gold in items.items():
        for mult, q in ((1.0, 5), (1.1, 10), (1.3, 20)):
            a += 1
            rows.append((a, it, q, int(round(gold * mult * price_scale * 10000)), "LONG"))
    return mk(rows, ts=pd.Timestamp(ts))


class TestHourlyFrame(unittest.TestCase):
    def test_hours_are_us_eastern_and_the_last_reading_in_an_hour_wins(self):
        snaps = [_snap("2026-10-27 12:10", {7: 10.0}), _snap("2026-10-27 12:50", {7: 20.0}, first_id=100), _snap("2026-10-27 13:30", {7: 30.0}, first_id=200)]
        f, info = FO.hourly_frame(snaps, [7])
        self.assertEqual(f["hour"].tolist(), [pd.Timestamp("2026-10-27 08:00"), pd.Timestamp("2026-10-27 09:00")])          # EDT is UTC-4
        self.assertEqual(f["floor_gold"].tolist(), [20.0, 30.0])
        self.assertEqual(f["units_listed"].tolist(), [35, 35])
        self.assertEqual(info["snapshots"], 3)

    def test_the_clock_going_back_an_hour_merges_the_repeated_hour(self):
        snaps = [_snap("2026-11-01 05:30", {7: 10.0}), _snap("2026-11-01 06:30", {7: 12.0}, first_id=100), _snap("2026-11-01 07:30", {7: 14.0}, first_id=200)]
        f, _ = FO.hourly_frame(snaps, [7])
        self.assertEqual(f["hour"].tolist(), [pd.Timestamp("2026-11-01 01:00"), pd.Timestamp("2026-11-01 02:00")])
        self.assertEqual(f["floor_gold"].tolist(), [12.0, 14.0])

    def test_partial_snapshots_are_dropped_and_reported(self):
        snaps = [_snap(f"2026-10-27 {12 + k}:30", {i: 5.0 + i for i in range(1, 21)}, first_id=1000 * k) for k in range(5)]
        snaps.append(_snap("2026-10-27 17:30", {1: 99.0, 2: 99.0}, first_id=9000))                       # 6 listings against a usual 60
        f, info = FO.hourly_frame(snaps, range(1, 21))
        self.assertEqual(len(info["dropped"]), 1)
        self.assertEqual(info["dropped"][0][1], 6)
        self.assertNotIn(99.0, f["floor_gold"].tolist())
        self.assertEqual(info["median_listings"], 60)

    def test_only_the_asked_for_items_and_nothing_when_empty(self):
        snaps = [_snap("2026-10-27 12:30", {1: 5.0, 2: 6.0, 3: 7.0})]
        f, _ = FO.hourly_frame(snaps, [2])
        self.assertEqual(f["item_id"].tolist(), [2])
        f0, info0 = FO.hourly_frame([], [2])
        self.assertTrue(f0.empty)
        self.assertEqual(info0["snapshots"], 0)

    def test_grid_has_nan_for_hours_with_no_reading_and_drops_thin_items(self):
        hours = pd.date_range("2026-10-27 08:00", periods=100, freq="h")
        rows = []
        for it, keep in ((1, [h for i, h in enumerate(hours) if i % 10 != 3]), (2, list(hours[:20]))):
            rows += [(it, h, 10.0, 50) for h in keep]
        frame = pd.DataFrame(rows, columns=["item_id", "hour", "floor_gold", "units_listed"])
        ids, idx, Y = FO.hourly_grid(frame, "price", min_obs=72)
        self.assertEqual(ids, [1])
        self.assertEqual(len(idx), 100)
        self.assertEqual(int(np.isnan(Y[0]).sum()), 10)
        np.testing.assert_allclose(Y[0][np.isfinite(Y[0])], 1.0)
        ids2, _, Y2 = FO.hourly_grid(frame, "stock", min_obs=72)
        np.testing.assert_allclose(Y2[0][np.isfinite(Y2[0])], np.log10(50))
        self.assertIsNone(FO.hourly_grid(frame, "price", min_obs=500))


class TestHourProfile(unittest.TestCase):
    def test_a_shared_cycle_shows_up_with_the_right_sign_and_everyone_agrees(self):
        ids, idx, Y = hworld("cycle", hole=False, hours=216)
        p = FO.hour_profile(Y, idx).set_index("hour")
        self.assertEqual(len(p), 24)
        self.assertGreater(p.loc[9, "vs_day_pct"], 5.0)
        self.assertLess(p.loc[21, "vs_day_pct"], -5.0)
        self.assertGreater(p.loc[9, "items_agree"], 0.9)

    def test_without_a_cycle_items_do_not_agree(self):
        ids, idx, Y = hworld("noise", hole=False, hours=216)
        p = FO.hour_profile(Y, idx)
        self.assertLess(float(p["items_agree"].mean()), 0.7)
        self.assertLess(float(p["vs_day_pct"].abs().max()), 3.0)


class TestHourlyRunner(unittest.TestCase):
    def _world_snaps(self, hours=216, n=100, partial_at=40):
        rng = np.random.default_rng(5)
        start = pd.Timestamp("2026-10-25 23:31")                                               # UTC, like the real run: first reading in the evening
        base = rng.uniform(0.5, 2.5, n)
        snaps = []
        for k in range(hours):
            ts = start + pd.Timedelta(hours=k)
            hr_et = (ts - pd.Timedelta(hours=4)).hour
            lvl = base + 0.04 * np.sin(2 * np.pi * (hr_et - 3) / 24) + rng.normal(0, 0.01, n)
            items = {i + 1: float(10 ** lvl[i]) for i in range(n)}
            if k == partial_at:
                items = {i + 1: 999.0 for i in range(3)}
            snaps.append(_snap(ts, items, first_id=100000 * k))
        return snaps, n

    def _run(self, snaps, n, ids_arg=""):
        import lake
        groups = pd.DataFrame({"item_id": range(1, n + 1), "group": [f"Fam: g{i // 10}" for i in range(n)], "family": "Fam"})
        tables = {"items": pd.DataFrame({"item_id": range(1, n + 1)}), "item_groups": groups}
        old = lake._load_all
        lake._load_all = lambda store, args, reader=None: snaps
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = lake.run_forecast_hourly(object(), type("A", (), {"ids": ids_arg})(), read_table=lambda name: tables.get(name, pd.DataFrame()))
        finally:
            lake._load_all = old
        return res, buf.getvalue()

    def test_prints_every_method_the_cycle_table_and_the_partial_snapshot_note(self):
        snaps, n = self._world_snaps()
        res, out = self._run(snaps, n)
        for s in ("HOURLY FORECAST BAKE-OFF (PRELIMINARY)", "naive", "day_ago", "median6", "ema04", "hour_of_day", "kinked", "pooled_grp", "BEST GUESS", "hit80", "BY HOW JUMPY",
                  "IS THERE A DAILY CYCLE", "partial snapshots left out", "BEST GUESS BY PRICE LEVEL", "real readings of burn-in", "1 hour ahead", "4 hours ahead", "12 hours ahead", "24 hours ahead", "PRICE", "STOCK"):
            self.assertIn(s, out)
        self.assertFalse(any(line.startswith("==") for line in out.splitlines()))
        self.assertEqual(set(res), {"price", "stock"})
        self.assertNotIn("sarima", out.lower().replace("sarima is not included", ""))

    def test_the_cycle_is_visible_in_the_results(self):
        snaps, n = self._world_snaps()
        res, _ = self._run(snaps, n)
        t = res["price"]["summary"]
        self.assertLess(t[(t["model"] == "pooled") & (t["horizon"] == 1)]["vs_naive"].iloc[0], 0.85)

    def test_no_items_and_too_few_snapshots_say_what_to_do(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_forecast_hourly(object(), type("A", (), {"ids": ""})(), read_table=lambda name: pd.DataFrame())
        self.assertEqual(res, {})
        self.assertIn("No items to cover", buf.getvalue())
        snaps, n = self._world_snaps(hours=10, partial_at=99)
        res, out = self._run(snaps, n)
        self.assertEqual(res, {})
        self.assertIn("at least 24", out)

    def test_workflow_lists_the_mode_and_skips_saving_results(self):
        for rel in (".github/workflows/pick_items.yml", "tools/pick_items.yml"):
            with open(os.path.join(_root, rel)) as f:
                text = f.read()
            self.assertIn("forecast_hourly", text.split("options:")[1].splitlines()[0])
            self.assertIn("inputs.mode != 'forecast_hourly'", text)
        with open(os.path.join(_root, "tools", "lake.py")) as f:
            src = f.read()
        # the mode reads snapshots, so it must be given a store (it is NOT in the list of modes that run without one)
        line = [l for l in src.splitlines() if l.strip().startswith("store = None if args.command in")][0]
        self.assertNotIn("forecast_hourly", line)


if __name__ == "__main__":
    unittest.main()
