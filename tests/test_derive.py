import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from helpers import T0  # noqa: F401
from engine.derive import DeriveConfig, confidence, derive
from engine.loader import LocalStore
from engine.reconcile import item_evidence
from engine.simulator import MarketSimulator, default_profiles
from test_cli import TestCli as _Cli  # noqa: F401  (reuses the _lake helper)
from test_loader import fake_reader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import lake  # noqa: E402

CFG = DeriveConfig(min_daily_high=0.0)


class TestDerive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snaps = MarketSimulator(default_profiles(), seed=9).run(80).snapshots
        cls.out = derive(cls.snaps, CFG, names={1002: "Test Herb"})

    def test_tables_and_ranges_are_ordered(self):
        it = self.out["items"]
        self.assertEqual(set(it["item_id"]), {1001, 1002, 1003, 1004})
        self.assertTrue((it["daily_low"] <= it["daily_est"] + 1e-9).all())
        self.assertTrue((it["daily_est"] <= it["daily_high"] + 1e-9).all())
        self.assertTrue((it["wall_low_gold"] <= it["wall_high_gold"] + 1e-9).all())
        self.assertEqual(it.set_index("item_id").loc[1002, "name"], "Test Herb")
        self.assertIn("engine_version", it.columns)

    def test_nothing_is_high_without_an_experiment_check(self):
        self.assertNotIn("High", set(self.out["items"]["confidence"]))

    def test_hourly_and_profile_cover_items(self):
        h = self.out["item_hourly"]
        self.assertEqual(set(h["item_id"]), {1001, 1002, 1003, 1004})
        self.assertTrue({"floor_gold", "units_listed", "sold_est", "spike", "blip"} <= set(h.columns))
        self.assertGreater(len(self.out["profile"]), 0)

    def test_calibration_is_one_row_with_counts(self):
        c = self.out["calibration"]
        self.assertEqual(len(c), 1)
        self.assertEqual(int(c.iloc[0][["n_high", "n_medium", "n_low"]].sum()), len(self.out["items"]))

    def test_too_few_snapshots_is_refused(self):
        with self.assertRaises(ValueError):
            derive(self.snaps[:2], CFG)

    def test_watch_list_keeps_items_the_filter_would_drop(self):
        strict = DeriveConfig(min_daily_high=1e12)
        none = derive(self.snaps, strict, include_flips=False)["items"]
        self.assertEqual(len(none), 0)
        kept = derive(self.snaps, strict, watch=[1002], include_flips=False)["items"]
        self.assertEqual(list(kept["item_id"]), [1002])
        self.assertTrue(kept.iloc[0]["watched"])


class TestConfidence(unittest.TestCase):
    base = dict(daily_low=900.0, daily_est=1000.0, daily_high=1100.0, observed_hours=72.0, top_interval_share=0.05, active_share=0.9)

    def c(self, evidence=None, **kw):
        return confidence(pd.Series({**self.base, **kw}), DeriveConfig(), None if evidence is None else pd.Series(evidence))[0]

    def test_default_is_medium(self):
        self.assertEqual(self.c(), "Medium")

    def test_short_history_is_low(self):
        self.assertEqual(self.c(observed_hours=10.0), "Low")

    def test_lumpy_and_slow_is_low_but_lumpy_and_fast_is_not(self):
        self.assertEqual(self.c(top_interval_share=0.8, daily_est=100.0, daily_low=80.0, daily_high=120.0), "Low")
        self.assertEqual(self.c(top_interval_share=0.8), "Medium")

    def test_wide_range_is_low(self):
        self.assertEqual(self.c(daily_low=100.0, daily_high=900.0), "Low")

    def test_high_needs_a_check_and_no_contradiction(self):
        self.assertEqual(self.c(dict(inside=2, too_fast=0, too_slow=0, unsold_stayed=0)), "High")
        self.assertEqual(self.c(dict(inside=2, too_fast=1, too_slow=0, unsold_stayed=0)), "Medium")

    def test_unsold_at_the_front_with_no_confirming_check_is_low(self):
        self.assertEqual(self.c(dict(inside=0, too_fast=0, too_slow=0, unsold_stayed=1)), "Low")


class TestItemEvidence(unittest.TestCase):
    def test_counts_inside_fast_and_stayed(self):
        def row(item, sold, qty, still, gone, pred, front):
            return dict(item_id=item, status="ok", cheapest_at_posting=True, done=True, abnormal=False, units_sold=sold, quantity=qty,
                        still_there_h=still, observed_hours_to_clear=gone, pred_lifo_h_est=pred, front_share=front)
        per = pd.DataFrame([row(1, 10, 10, 1.0, 3.0, 2.0, 1.0), row(1, 10, 10, 4.0, 6.0, 2.0, 1.0), row(1, 10, 10, 1.0, 3.0, 8.0, 1.0),
                            row(2, 0, 10, 12.0, np.nan, 5.0, 1.0), row(2, 0, 10, 12.0, np.nan, 5.0, 0.2)])
        e = item_evidence(per).set_index("item_id")
        self.assertEqual((e.loc[1, "inside"], e.loc[1, "too_slow"], e.loc[1, "too_fast"]), (1, 1, 1))
        self.assertEqual((e.loc[2, "unsold_stayed"], e.loc[2, "unsold_pushed"]), (1, 1))


class TestDeriveCli(unittest.TestCase):
    def test_derive_writes_parquet_files(self):
        snaps = MarketSimulator(default_profiles(), seed=9).run(60).snapshots
        with tempfile.TemporaryDirectory() as d:
            args = _Cli()._lake(d, snaps)
            args.log = os.path.join(d, "none.csv")
            args.ids = ""
            args.upload = False
            args.horizon_hours = 4.0
            args.window_hours = 72.0
            try:
                import pyarrow  # noqa: F401
                real = True
            except ImportError:      # this sandbox has no pyarrow: write pickles under the same names
                real = False
            patch = mock.patch.object(pd.DataFrame, "to_parquet", lambda self, path, **kw: self.to_pickle(path)) if not real else mock.MagicMock()
            with patch:
                out = lake.run_derive(LocalStore(args.root), args, reader=fake_reader, names_fn=lambda ids: {1002: "Test Herb"})
            folder = os.path.join(d, "out", "derived")
            for k in ("items", "item_hourly", "calibration"):
                self.assertTrue(os.path.exists(os.path.join(folder, f"{k}.parquet")), k)
            self.assertGreater(len(out["items"]), 0)
            if real:
                self.assertGreater(len(pd.read_parquet(os.path.join(folder, "items.parquet"))), 0)
            self.assertEqual(len(out["calibration"]), 1)


if __name__ == "__main__":
    unittest.main()
