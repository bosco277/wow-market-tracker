"""tools/forever_compact.py (copy-only ledger compaction) on a pretend Forever dataset: houses 110 and 116, commodity and regular-auction hourly files written the
way the fetcher writes them, 54 hours = Oct 1 and Oct 2 complete, Oct 3 still running. Parquet is stood in for by pickle (pyarrow is not installed here); the
stand-in writer keeps the runs and meta apart exactly as the parquet writer does, so what is checked is what would be stored."""
import json
import os
import pickle
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from helpers import mk  # noqa: F401  (sets sys.path)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import forever_compact as FC  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine import foreverpack as P  # noqa: E402
from engine.loader import LocalStore  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402
from test_forever_adapter import write, write_house  # noqa: E402

NOW = datetime(2026, 10, 3, 6, 0, tzinfo=timezone.utc)


def pwriter(runs, meta):
    return pickle.dumps((runs, {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in meta.items()}))


def preader(data):
    return pickle.loads(data)


class CompactFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snaps = MarketSimulator(default_profiles(), seed=5).run(54).snapshots

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "lake")
        sources = {}
        for h in (110, 116):
            write_house(self.root, h, self.snaps, commodity=True)
            write_house(self.root, h, self.snaps, commodity=False)
            sources[FV.source_name(h, True)] = dict(house_id=h)
            sources[FV.source_name(h, False)] = dict(house_id=h)
        write(self.root, FV.SOURCES_KEY, json.dumps(sources).encode())
        self.store = LocalStore(self.root)
        self.commits = []

    def commit(self, files, message):
        self.commits.append((message, sorted(files)))
        for k, v in files.items():
            write(self.root, k, v)

    def go(self, now=NOW, **kw):
        a = SimpleNamespace(**{**dict(houses="all", kinds="both", days=60, minutes=90, save=False), **kw})
        return FC.run(a, self.store, now=now, reader=pickle.loads, writer=pwriter, preader=preader, commit=self.commit)

    def tree(self):
        return sorted(os.path.relpath(os.path.join(d, n), self.root) for d, _, ns in os.walk(self.root) for n in ns)


class TestCompact(CompactFixture):
    def test_report_only_packs_and_checks_every_complete_day_and_stores_nothing(self):
        before = self.tree()
        r = self.go()
        self.assertEqual(self.tree(), before)
        self.assertEqual(self.commits, [])
        self.assertEqual(r["passed"], 8)                          # 2 houses x 2 kinds x 2 complete days (Oct 3 is still running)
        self.assertEqual(r["failed"], [])
        self.assertIn("### 2026-10-01", r["text"])
        self.assertIn("### 2026-10-02", r["text"])
        self.assertNotIn("### 2026-10-03", r["text"])
        self.assertIn("Hourly files deleted: 0", r["text"])

    def test_save_stores_one_commit_per_day_outside_the_raw_folders_and_deletes_nothing(self):
        before = self.tree()
        r = self.go(save=True)
        self.assertEqual([m for m, _ in self.commits], ["forever compact: 2026-10-01 (4 pack(s), copy-only)", "forever compact: 2026-10-02 (4 pack(s), copy-only)"])
        after = self.tree()
        self.assertTrue(set(before) <= set(after))                # every hourly file is still there
        new = sorted(set(after) - set(before))
        self.assertEqual(len(new), 8)
        self.assertTrue(all(k.startswith("packed/") and k.endswith("/ledger.parquet") for k in new))
        self.assertIn("packed/" + FV.source_name(110, True) + "/year=2026/month=10/day=01/ledger.parquet", new)
        self.assertIn("each read back byte for byte identical", r["text"])
        self.assertEqual(len(r["stored"]), 8)

    def test_a_stored_pack_rebuilds_the_days_hourly_files_exactly(self):
        self.go(save=True)
        src = FV.source_name(116, False)
        with open(os.path.join(self.root, P.pack_key(src, "2026-10-02")), "rb") as fh:
            runs, meta = preader(fh.read())
        files = FC.hourly_days(self.store, src)[datetime(2026, 10, 2).date()]
        originals = [pickle.loads(self.store.read_range(k, 0, s)) for k, s in files]
        self.assertEqual(meta["keys"], [k for k, _ in files])
        ok, why = P.check(originals, P.unpack(runs, meta))
        self.assertTrue(ok, why)
        self.assertLess(len(runs), sum(len(f) for f in originals))

    def test_the_next_run_skips_packed_days_and_packs_the_new_one(self):
        self.go(save=True)
        self.commits.clear()
        r = self.go(save=True)
        self.assertEqual(self.commits, [])
        self.assertIn("Days waiting to be packed: 0", r["text"])
        r = self.go(now=datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc), save=True)      # Oct 3 is now complete
        self.assertEqual([m for m, _ in self.commits], ["forever compact: 2026-10-03 (4 pack(s), copy-only)"])

    def test_a_day_is_not_complete_until_two_hours_after_midnight(self):
        r = self.go(now=datetime(2026, 10, 3, 1, 0, tzinfo=timezone.utc))
        self.assertNotIn("### 2026-10-02", r["text"])
        self.assertIn("### 2026-10-01", r["text"])

    def test_a_pack_that_does_not_rebuild_exactly_is_not_stored(self):
        orig = P.unpack

        def lossy(runs, meta):
            out = orig(runs, meta)
            out[5] = out[5].iloc[1:].reset_index(drop=True)
            return out
        with mock.patch.object(P, "unpack", lossy):
            r = self.go(save=True)
        self.assertEqual(r["passed"], 0)
        self.assertEqual(len(r["failed"]), 8)
        self.assertEqual(self.commits, [])
        self.assertIn("NOT kept", r["text"])
        self.assertEqual(r["code"], 1)

    def test_a_stored_file_that_reads_back_different_is_reported(self):
        def corrupting(files, message):
            self.commits.append((message, sorted(files)))
            for k, v in files.items():
                write(self.root, k, v[:-1] + b"x")
        r = FC.run(SimpleNamespace(houses="110", kinds="commodity", days=1, minutes=90, save=True), self.store, now=NOW, reader=pickle.loads,
                   writer=pwriter, preader=preader, commit=corrupting)
        self.assertEqual(r["code"], 1)
        self.assertIn("did NOT read back identical", r["text"])
        self.assertEqual(r["stored"], [])

    def test_days_and_time_budget_limit_a_run(self):
        r = self.go(days=1)
        self.assertIn("days still waiting after this run: 1", r["text"])
        ticks = iter([0, 0] + [10_000] * 100)
        r = FC.run(SimpleNamespace(houses="all", kinds="both", days=60, minutes=1, save=False), self.store, now=NOW, reader=pickle.loads,
                   writer=pwriter, preader=preader, commit=self.commit, clock=lambda: next(ticks))
        self.assertIn("Time budget reached", r["text"])

    def test_kinds_and_houses_options(self):
        r = self.go(kinds="commodity", houses="116")
        self.assertEqual(r["passed"], 2)

    def test_the_job_has_no_way_to_delete(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "tools", "forever_compact.py"), encoding="utf-8") as fh:
            src = fh.read()
        for word in ("CommitOperationDelete", "delete_file", "delete_folder", "os.remove", "unlink", "rmtree"):
            self.assertNotIn(word, src)


class TestWorkflow(unittest.TestCase):
    def test_both_copies_match_and_the_schedule_is_gated(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "tools", "forever_compact.yml"), encoding="utf-8") as fh:
            a = fh.read()
        with open(os.path.join(root, ".github", "workflows", "forever_compact.yml"), encoding="utf-8") as fh:
            b = fh.read()
        self.assertEqual(a, b)
        self.assertIn("vars.FOREVER_COMPACT_BUILD == 'yes'", a)
        self.assertIn("pyarrow", a)


if __name__ == "__main__":
    unittest.main()
