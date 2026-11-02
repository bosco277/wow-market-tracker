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
    def test_forecast_test_mode_prints_every_model_and_a_verdict(self):
        import lake
        hist, groups, _ = world("weekly", n=48, days=420)
        tables = {"market_history": hist, "item_groups": groups}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lake.run_forecast_test(None, type("A", (), {})(), read_table=lambda name: tables.get(name, pd.DataFrame()))
        out = buf.getvalue()
        for s in ("FORECAST BAKE-OFF", "naive", "ema25", "weekday", "sarima", "kinked", "pooled_grp", "BEST GUESS", "hit80", "BY HOW JUMPY", "SARIMA CHECK"):
            self.assertIn(s, out)
        self.assertFalse(any(line.startswith("==") for line in out.splitlines()))

    def test_workflow_lists_the_mode_and_skips_saving_results(self):
        for rel in (".github/workflows/pick_items.yml", "tools/pick_items.yml"):
            with open(os.path.join(_root, rel)) as f:
                text = f.read()
            self.assertIn("forecast_test]", text)
            self.assertIn("inputs.mode != 'forecast_test'", text)
            self.assertIn("boto3 scipy", text)      # SARIMA needs scipy on the runner

    def test_missing_tables_say_what_to_run(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_forecast_test(None, type("A", (), {})(), read_table=lambda name: pd.DataFrame())
        self.assertEqual(res, {})
        self.assertIn("mode undermine", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
