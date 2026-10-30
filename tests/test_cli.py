import argparse
import os
import pickle
import sys
import tempfile
import unittest

import pandas as pd

from helpers import T0  # noqa: F401
from engine.loader import LocalStore
from engine.simulator import MarketSimulator, default_profiles
from test_loader import fake_reader, key_for, write
from test_pipeline import as_fetch_v2_frame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import lake  # noqa: E402


class TestCli(unittest.TestCase):
    def test_pick_end_to_end_on_local_files(self):
        snaps = MarketSimulator(default_profiles(), seed=9).run(60).snapshots
        with tempfile.TemporaryDirectory() as d:
            for s in snaps:
                write(os.path.join(d, "lake"), key_for(s.ts), as_fetch_v2_frame([s]))
            args = argparse.Namespace(
                store="local", root=os.path.join(d, "lake"), days=3, end=str(snaps[-1].ts.date()),
                cache=os.path.join(d, "cache"), out=os.path.join(d, "out"), max_gb=0.8,
                max_position_gold=1e9, n_items=8, min_floor_gold=0.0, max_units=10**9,
                window_hours=72.0, min_coverage=0.5)
            plan = lake.run_plan(LocalStore(args.root), args)
            self.assertEqual(len(plan.files), len(snaps))
            short = lake.run_pick(LocalStore(args.root), args, reader=fake_reader, names_fn=lambda ids: {1002: "Test Herb"})
            self.assertIsNotNone(short)
            self.assertGreater(len(short), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "out", "experiment_shortlist.csv")))
            back = pd.read_csv(os.path.join(d, "out", "experiment_shortlist.csv"))
            self.assertIn("buy_units", back.columns)
            self.assertIn("Test Herb", set(back["name"]))

    def test_order_command_runs(self):
        snaps = MarketSimulator(default_profiles(), seed=9, lifo_within_tier=True).run(60).snapshots
        with tempfile.TemporaryDirectory() as d:
            for s in snaps:
                write(os.path.join(d, "lake"), key_for(s.ts), as_fetch_v2_frame([s]))
            args = argparse.Namespace(store="local", root=os.path.join(d, "lake"), days=3, end=str(snaps[-1].ts.date()),
                                      cache=os.path.join(d, "cache"))
            ev = lake.run_order(LocalStore(args.root), args, reader=fake_reader)
            self.assertGreater(ev.lifo_consistent, ev.fifo_consistent)


    def _lake(self, d, snaps):
        for s in snaps:
            write(os.path.join(d, "lake"), key_for(s.ts), as_fetch_v2_frame([s]))
        return argparse.Namespace(store="local", root=os.path.join(d, "lake"), days=3, end=str(snaps[-1].ts.date()),
                                  cache=os.path.join(d, "cache"), out=os.path.join(d, "out"), ids="1002,1003")

    def test_item_and_flips_commands_run(self):
        snaps = MarketSimulator(default_profiles(), seed=9).run(40).snapshots
        with tempfile.TemporaryDirectory() as d:
            args = self._lake(d, snaps)
            out = lake.run_item(LocalStore(args.root), args, reader=fake_reader, names_fn=lambda ids: {1002: "Test Herb"})
            self.assertEqual(len(out), len(snaps))
            ev = lake.run_flips(LocalStore(args.root), args, reader=fake_reader, names_fn=lambda ids: {})
            self.assertIsNotNone(ev)

    def test_files_command_lists_days_and_missing_hours_without_downloading(self):
        import contextlib
        import io
        snaps = MarketSimulator(default_profiles(), seed=9).run(30).snapshots
        with tempfile.TemporaryDirectory() as d:
            args = self._lake(d, snaps[::3])
            args.store = "local"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                inv = lake.run_files(LocalStore(args.root), args)
            self.assertIn("UTC hours with no hourly file", buf.getvalue())
            self.assertTrue(inv)
            self.assertFalse(os.path.exists(os.path.join(d, "cache")))

    def test_pick_survives_a_big_hole_in_the_data_and_says_so(self):
        """The real failure: ~21 h with no snapshots made every item look like 'too little data'."""
        import contextlib
        import io
        snaps = MarketSimulator(default_profiles(), seed=9).run(110).snapshots
        snaps = snaps[:60] + snaps[83:]          # a 23 hour hole inside the last 72 hours
        with tempfile.TemporaryDirectory() as d:
            for sn in snaps:
                write(os.path.join(d, "lake"), key_for(sn.ts), as_fetch_v2_frame([sn]))
            args = argparse.Namespace(
                store="local", root=os.path.join(d, "lake"), days=5, end=str(snaps[-1].ts.date()),
                cache=os.path.join(d, "cache"), out=os.path.join(d, "out"), max_gb=0.8,
                max_position_gold=1e9, n_items=8, min_floor_gold=0.0, max_units=10**9,
                window_hours=72.0, min_coverage=0.5)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                short = lake.run_pick(LocalStore(args.root), args, reader=fake_reader, names_fn=lambda ids: {})
            text = buf.getvalue()
            self.assertIn("HOLES IN THE DATA", text)
            self.assertIn("with nothing recorded", text)
            self.assertIn("HOW MUCH DATA THE ITEMS HAD", text)
            self.assertGreater(len(short), 0)


if __name__ == "__main__":
    unittest.main()
