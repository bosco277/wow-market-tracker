import os
import sys
import unittest

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
from engine import sarima as SA  # noqa: E402


def simulate(n, phi=(), Phi=(), theta=(), Theta=(), mean=2.0, seed=0, sd=1.0):
    """x_t = sum phi_i x_{t-i} + sum Phi_j x_{t-7j} + e_t + sum theta_i e_{t-i} + sum Theta_j e_{t-7j}; returns mean + x and the innovations."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sd, n + 200)
    x = np.zeros(n + 200)
    for t in range(n + 200):
        v = e[t]
        for i, c in enumerate(phi, 1):
            v += c * (x[t - i] if t - i >= 0 else 0)
        for j, c in enumerate(Phi, 1):
            v += c * (x[t - 7 * j] if t - 7 * j >= 0 else 0)
        for i, c in enumerate(theta, 1):
            v += c * (e[t - i] if t - i >= 0 else 0)
        for j, c in enumerate(Theta, 1):
            v += c * (e[t - 7 * j] if t - 7 * j >= 0 else 0)
        x[t] = v
    return mean + x[200:], e[200:]


class TestEstimation(unittest.TestCase):
    def test_recovers_ar1_and_the_mean(self):
        y, _ = simulate(3000, phi=(0.7,), mean=2.0, seed=1, sd=0.05)
        params, aic = SA.fit_spec(y, (1, 0, 0, 0, 0, 0))
        self.assertAlmostEqual(params[0], 0.7, delta=0.06)
        self.assertAlmostEqual(params[1], 2.0, delta=0.05)

    def test_recovers_a_seasonal_ar(self):
        y, _ = simulate(3000, Phi=(0.6,), mean=1.0, seed=2, sd=0.05)
        params, _ = SA.fit_spec(y, (0, 0, 0, 1, 0, 0))
        self.assertAlmostEqual(params[0], 0.6, delta=0.07)

    def test_recovers_an_ma1(self):
        y, _ = simulate(3000, theta=(0.5,), mean=0.0, seed=3, sd=0.05)
        params, _ = SA.fit_spec(y, (0, 0, 1, 0, 0, 0))
        self.assertAlmostEqual(params[0], 0.5, delta=0.07)

    def test_aic_prefers_the_true_structure(self):
        y, _ = simulate(2500, Phi=(0.6,), mean=1.0, seed=4, sd=0.05)
        with_season = SA.fit_spec(y, (0, 0, 0, 1, 0, 0))[1]
        without = SA.fit_spec(y, (0, 0, 0, 0, 0, 0))[1]
        self.assertLess(with_season, without - 50)

    def test_fit_best_needs_a_long_clean_series(self):
        self.assertIsNone(SA.fit_best(np.ones(50)))
        y, _ = simulate(400, phi=(0.5,), seed=5, sd=0.05)
        y[100] = np.nan
        self.assertIsNone(SA.fit_best(y))


class TestForecasts(unittest.TestCase):
    def test_ar1_forecasts_follow_the_textbook_formula(self):
        y, _ = simulate(500, phi=(0.7,), mean=2.0, seed=6, sd=0.05)
        spec, params = (1, 0, 0, 0, 0, 0), np.array([0.7, 2.0])
        f = SA.forecast_series(y, spec, params, (1, 3, 7))
        for h in (1, 3, 7):
            np.testing.assert_allclose(f[h][50:], 2.0 + 0.7 ** h * (y[50:] - 2.0), atol=1e-9)

    def test_ma1_two_days_ahead_is_just_the_mean(self):
        y, _ = simulate(500, theta=(0.5,), mean=2.0, seed=7, sd=0.05)
        f = SA.forecast_series(y, (0, 0, 1, 0, 0, 0), np.array([0.5, 2.0]), (1, 2, 5))
        np.testing.assert_allclose(f[2][60:], 2.0, atol=1e-9)
        np.testing.assert_allclose(f[5][60:], 2.0, atol=1e-9)
        # one day ahead uses the latest innovation: its error is the innovation of the next day, whose spread is the noise level
        err = y[61:] - f[1][60:-1]
        self.assertAlmostEqual(float(err.std()), 0.05, delta=0.005)

    def test_local_level_spec_reproduces_an_exponential_average(self):
        rng = np.random.default_rng(8)
        y = 2.0 + np.cumsum(rng.normal(0, 0.03, 400))
        f = SA.forecast_series(y, (0, 1, 1, 0, 0, 0), np.array([-0.8]), (1, 3, 7))
        level = np.empty_like(y)
        level[0] = y[0]
        for t in range(1, len(y)):
            level[t] = 0.2 * y[t] + 0.8 * level[t - 1]
        for h in (1, 3, 7):
            np.testing.assert_allclose(f[h], level, atol=1e-9)

    def test_forecasts_never_use_days_after_the_origin(self):
        y, _ = simulate(400, phi=(0.5,), Phi=(0.4,), theta=(0.2,), mean=2.0, seed=9, sd=0.05)
        for spec, params in (((1, 0, 1, 1, 0, 0), np.array([0.5, 0.4, 0.2, 2.0])), ((0, 1, 1, 0, 1, 1), np.array([-0.6, -0.5])), ((1, 1, 1, 1, 0, 1), np.array([0.3, 0.3, -0.5, -0.2]))):
            y2 = y.copy()
            y2[250:] += 3.0
            a, b = SA.forecast_series(y, spec, params, (1, 3, 7)), SA.forecast_series(y2, spec, params, (1, 3, 7))
            for h in (1, 3, 7):
                np.testing.assert_allclose(a[h][:250], b[h][:250], atol=1e-9, err_msg=f"{spec} h={h}")

    def test_seasonal_differencing_predicts_the_same_weekday(self):
        rhythm = np.array([0, 0.3, -0.2, -0.2, 0, 0.1, -0.1])
        y = 2.0 + np.tile(rhythm, 60)
        f = SA.forecast_series(y, (0, 0, 0, 0, 1, 0), np.array([]), (1, 3, 7))
        t = np.arange(20, 400)
        for h in (1, 3, 7):
            np.testing.assert_allclose(f[h][t], 2.0 + rhythm[(t + h) % 7], atol=1e-9)


class TestHelpers(unittest.TestCase):
    def test_fill_gaps_bridges_only_inside(self):
        y = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, 5.0, np.nan])
        out = SA.fill_gaps(y)
        self.assertTrue(np.isnan(out[0]))
        np.testing.assert_allclose(out[1:6], [1, 2, 3, 4, 5])
        self.assertEqual(out[6], 5.0)

    def test_forecast_all_fills_the_window_only_and_reports_failures(self):
        y1, _ = simulate(600, phi=(0.6,), mean=2.0, seed=10, sd=0.05)
        y2, _ = simulate(600, phi=(0.6,), mean=1.0, seed=11, sd=0.05)
        short = np.full(600, np.nan)
        short[500:] = 1.0
        Y = np.vstack([y1, y2, short])
        out, info = SA.forecast_all(Y, 400, 380, 599, (1, 7))
        self.assertEqual(info["fitted"], 2)
        self.assertEqual(info["failed"], 1)
        self.assertTrue(np.isnan(out[1][:, :380]).all())
        self.assertTrue(np.isfinite(out[1][:2, 380:]).all())
        self.assertTrue(np.isnan(out[1][2]).all())
        self.assertEqual(sum(info["specs"].values()), 2)

    def test_params_are_fitted_on_the_fitting_window_only(self):
        y, _ = simulate(600, phi=(0.6,), mean=2.0, seed=12, sd=0.05)
        y2 = y.copy()
        y2[450:] += 5.0
        a, _ = SA.forecast_all(np.vstack([y]), 400, 380, 599, (1,))
        b, _ = SA.forecast_all(np.vstack([y2]), 400, 380, 599, (1,))
        np.testing.assert_allclose(a[1][0, 380:449], b[1][0, 380:449])   # same parameters, and days before 450 are identical


if __name__ == "__main__":
    unittest.main()
