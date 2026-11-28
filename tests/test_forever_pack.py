"""engine/foreverpack.py (packing one house-day losslessly) and tools/forever_pack_probe.py (the report-only probe) on hand-made files and on a pretend
Forever dataset (simulated snapshots written the way the fetcher writes them). Parquet is stood in for by pickle (pyarrow is not installed here)."""
import json
import os
import pickle
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from helpers import mk  # noqa: F401  (sets sys.path)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import forever_pack_probe as PP  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine import foreverpack as P  # noqa: E402
from engine.loader import LocalStore  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402
from test_forever_adapter import write, write_house  # noqa: E402

TS = [pd.Timestamp("2026-11-26 00:32") + pd.Timedelta(hours=k) for k in range(5)]


def f(k, rows):
    """One hourly file: rows of (id, item, quantity, unit_price, time_left)."""
    return pd.DataFrame({"id": [r[0] for r in rows], "item": [{"id": r[1]} for r in rows], "quantity": [r[2] for r in rows],
                         "unit_price": [float(r[3]) for r in rows], "time_left": [r[4] for r in rows], "item_id": [r[1] for r in rows],
                         "source": "pretend-commodities", "snapshot_ts": TS[k], "schema_version": 2})


def day():
    return [f(0, [(1, 5, 10, 100, "LONG"), (2, 6, 3, 50.25, "SHORT")]),
            f(1, [(1, 5, 10, 100, "LONG"), (2, 6, 1, 50.25, "SHORT")]),          # 2 sold two units
            f(2, [(1, 5, 10, 100, "MEDIUM")]),                                  # 1 dropped a time-left bucket; 2 is missing this hour
            f(3, [(1, 5, 10, 100, "MEDIUM"), (2, 6, 1, 50.25, "SHORT"), (3, 7, 20, 9, "VERY_LONG")])]


class TestLedger(unittest.TestCase):
    def roundtrip(self, frames):
        cols = list(frames[0].columns)
        runs, files, const = P.to_ledger(frames)
        return runs, const, P.check(frames, P.from_ledger(runs, files, const, cols, [f.dtypes for f in frames]))

    def test_runs_split_on_any_change_or_a_missing_hour(self):
        runs, const, (ok, why) = self.roundtrip(day())
        self.assertTrue(ok, why)
        self.assertEqual(const, ["source", "snapshot_ts", "schema_version"])
        self.assertEqual(len(runs), 6)                                                        # 1: LONG then MEDIUM; 2: q3, q1, (gap) q1; 3
        r2 = runs[runs["id"] == 2][["quantity", "k_first", "k_last"]].values.tolist()
        self.assertEqual(r2, [[3, 0, 0], [1, 1, 1], [1, 3, 3]])

    def test_sorted_and_concat_rebuild_exactly(self):
        frames = day()
        cols = list(frames[0].columns)
        for packed in (P.to_concat(frames), P.to_sorted(frames)):
            ok, why = P.check(frames, P.from_rows(packed, len(frames), cols))
            self.assertTrue(ok, why)

    def test_a_listing_twice_in_one_file_an_empty_hour_and_missing_values(self):
        frames = day()
        frames[1] = pd.concat([frames[1], frames[1].iloc[[0]]], ignore_index=True)           # the same auction id twice in one file
        frames.insert(2, f(4, []).astype(frames[0].dtypes.to_dict()))                         # an hour with no listings
        frames[3].loc[0, "unit_price"] = np.nan
        _, _, (ok, why) = self.roundtrip(frames)
        self.assertTrue(ok, why)

    def test_a_wrong_rebuild_is_caught(self):
        frames = day()
        cols = list(frames[0].columns)
        runs, files, const = P.to_ledger(frames)
        bad = P.from_ledger(runs, files, const, cols)
        bad[1].loc[bad[1]["id"] == 2, "quantity"] = 3                                         # the partial sale lost
        ok, why = P.check(frames, bad)
        self.assertFalse(ok)
        self.assertIn("quantity", why)
        bad = P.from_ledger(runs, files, const, cols)
        bad[3] = bad[3].iloc[:-1]
        self.assertFalse(P.check(frames, bad)[0])
        bad = P.from_ledger(runs, files, const, cols)
        bad[0]["item"] = [{"id": 5, "extra": 1}, {"id": 6}]
        self.assertFalse(P.check(frames, bad)[0])

    def test_a_column_that_is_whole_numbers_in_one_hour_and_decimals_in_another(self):
        # house 125, regular auctions, Nov 13: a bid-only auction (no buyout) makes pandas read that hour's buyout column as decimals; other hours are whole numbers
        def g(k, buyouts):
            return pd.DataFrame({"id": range(10 * k, 10 * k + len(buyouts)), "item_id": 5, "quantity": 1, "buyout": buyouts, "snapshot_ts": TS[k]})
        frames = [g(0, [100.0, np.nan]), g(1, np.array([100, 200], dtype="int64")), g(2, [300.0, np.nan])]
        self.assertEqual([str(x["buyout"].dtype) for x in frames], ["float64", "int64", "float64"])
        runs, meta = P.pack(frames, ["a", "b", "c"])
        ok, why = P.check(frames, P.unpack(runs, meta))
        self.assertTrue(ok, why)
        self.assertFalse(P.check(frames, P.from_ledger(runs, meta["files"], meta["const"], meta["columns"], meta["dtypes"][0]))[0])   # the v0.7.53 rule fails

    def test_prices_keep_every_decimal(self):
        frames = day()
        frames[0].loc[1, "unit_price"] = 50.123456789
        frames[1].loc[1, "unit_price"] = 50.123456789
        runs, _, (ok, why) = self.roundtrip(frames)
        self.assertTrue(ok, why)


class TestProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snaps = MarketSimulator(default_profiles(), seed=5).run(30).snapshots

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "lake")
        sources = {}
        for h in (110, 116):
            write_house(self.root, h, self.snaps)
            sources[FV.source_name(h)] = dict(house_id=h)
        write(self.root, FV.SOURCES_KEY, json.dumps(sources).encode())
        self.store = LocalStore(self.root)

    def go(self, **kw):
        a = SimpleNamespace(**{**dict(day="2026-10-01", houses="110,116"), **kw})
        return PP.run(a, self.store, now=datetime(2026, 10, 2, 12, tzinfo=timezone.utc), reader=pickle.loads, sizer=lambda df: len(pickle.dumps(df)))

    def test_every_form_is_lossless_and_sales_are_identical(self):
        r = self.go()
        self.assertEqual(r["code"], 0, r["text"])
        self.assertEqual(r["text"].count("identical"), 2)
        for form in ("concat", "sorted", "ledger"):
            self.assertEqual(r["text"].count(f"    {form:<8}"), 2)
        self.assertNotIn(" NO ", r["text"])
        self.assertIn("REPORT ONLY", r["text"])
        self.assertIn("a year for all 12 houses", r["text"])

    def test_the_ledger_has_far_fewer_rows_than_the_stored_files(self):
        r = self.go(houses="110")
        line = next(l for l in r["text"].splitlines() if "Ledger rows are" in l)
        self.assertLess(float(line.split("Ledger rows are ")[1].split("%")[0]), 50.0)

    def test_a_ledger_that_loses_something_is_reported_and_fails_the_run(self):
        orig = P.from_ledger

        def lossy(runs, files, const, cols, dtypes=None):
            out = orig(runs, files, const, cols, dtypes)
            out[3] = out[3].iloc[1:].reset_index(drop=True)
            return out
        with mock.patch.object(P, "from_ledger", lossy):
            r = self.go(houses="110")
        self.assertEqual(r["code"], 1)
        led = next(l for l in r["text"].splitlines() if l.strip().startswith("ledger"))
        self.assertIn("NO", led)                                      # the exact file check catches it (one missing listing-hour need not change the sales totals)

    def test_nothing_is_written_to_the_dataset(self):
        before = sorted(os.path.relpath(os.path.join(d, n), self.root) for d, _, ns in os.walk(self.root) for n in ns)
        self.go()
        after = sorted(os.path.relpath(os.path.join(d, n), self.root) for d, _, ns in os.walk(self.root) for n in ns)
        self.assertEqual(before, after)

    def test_a_house_without_files_that_day_is_skipped(self):
        r = self.go(houses="110,113")
        self.assertIn("house 113", r["text"])
        self.assertEqual(r["code"], 1)
        self.assertEqual(r["text"].count("identical"), 1)


class TestWorkflow(unittest.TestCase):
    def test_both_copies_match_and_it_cannot_save(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "tools", "forever_pack_probe.yml"), encoding="utf-8") as fh:
            a = fh.read()
        with open(os.path.join(root, ".github", "workflows", "forever_pack_probe.yml"), encoding="utf-8") as fh:
            b = fh.read()
        self.assertEqual(a, b)
        self.assertNotIn("--save", a)
        self.assertNotIn("schedule:", a)


if __name__ == "__main__":
    unittest.main()
