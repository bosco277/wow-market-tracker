"""Statistical checks of the classifier against simulated ground truth.

These document measured behaviour on SIMULATED markets. They guard against regressions in the logic; they do not
prove the bounds hold on the real auction house (that needs real event data, see docs/DATA_CONTRACT.md).
"""
import dataclasses
import unittest

import pandas as pd

from helpers import T0  # noqa: F401  (sets sys.path)
from engine.classifier import ClassifierParams
from engine.simulator import MarketSimulator, default_profiles
from engine.validation import evaluate_run, summarize


def run_rows(seeds, hours, params=None, profiles=None, **sim_kw):
    frames = []
    for seed in seeds:
        res = MarketSimulator(profiles or default_profiles(), seed=seed, **sim_kw).run(hours)
        frames.append(evaluate_run(res, params))
    return pd.concat(frames, ignore_index=True)


class TestBaselineBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = summarize(run_rows((1, 2), 72)).set_index("item_id")

    def test_bounds_bracket_truth_most_of_the_time(self):
        self.assertTrue((self.summary["coverage"] >= 0.93).all(), self.summary["coverage"].to_dict())

    def test_high_bound_is_not_violated(self):
        self.assertTrue((self.summary["high_viol"] <= 0.01).all(), self.summary["high_viol"].to_dict())

    def test_point_estimate_is_close_in_aggregate_on_busy_items(self):
        for item in (1001, 1004):
            ratio = self.summary.loc[item, "est_ratio"]
            self.assertTrue(0.85 <= ratio <= 1.15, (item, ratio))

    def test_blind_spot_is_large_on_hot_items(self):
        # Sales of listings created and sold inside one interval are invisible to snapshot diffs.
        self.assertGreater(self.summary.loc[1004, "blind_spot_ratio"], 1.5)


class TestStressBehaviour(unittest.TestCase):
    def test_heavy_cancel_churn_degrades_the_conservative_bound(self):
        base = summarize(run_rows((1, 2), 72)).set_index("item_id")
        churn_profiles = [dataclasses.replace(p, cancel_rate_per_hr=p.cancel_rate_per_hr * 6, p_relist=0.9) for p in default_profiles()]
        churn = summarize(run_rows((1, 2), 72, profiles=churn_profiles)).set_index("item_id")
        self.assertGreater(churn["low_viol"].mean(), base["low_viol"].mean())  # the validator notices worse markets

    def test_tier_aware_default_protects_the_high_bound_when_market_is_not_fifo(self):
        fifo = summarize(run_rows((1, 2), 72, ClassifierParams(fifo_within_tier=True), fifo_within_tier=False))
        tier = summarize(run_rows((1, 2), 72, ClassifierParams(fifo_within_tier=False), fifo_within_tier=False))
        self.assertGreater(fifo["high_viol"].mean(), tier["high_viol"].mean())
        self.assertLessEqual(tier["high_viol"].mean(), 0.01)


if __name__ == "__main__":
    unittest.main()
