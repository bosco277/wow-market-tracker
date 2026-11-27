"""tools/forever_derive.py on a pretend Forever dataset: houses 110, 113 (prices doubled) and 116 (prices x 1.5), 60 hours of simulated snapshots each.
Parquet is stood in for by pickle (pyarrow is not installed where the tests run; the workflow installs it)."""
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

import forever_derive as FDV  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.loader import LocalStore  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402
from test_forever_adapter import base_reader, write, write_house  # noqa: E402

NOW = datetime(2026, 10, 3, 14, 0, tzinfo=timezone.utc)
MULT = {110: 1.0, 113: 2.0, 116: 1.5}


def scaled(snaps, mult):
    out = []
    for s in snaps:
        b = s.book.copy()
        b["unit_price"] = (b["unit_price"] * mult).round().astype(s.book["unit_price"].dtype)
        out.append(type(s)(ts=s.ts, book=b, kind=s.kind, source=s.source, meta=dict(s.meta)))
    return out


class DeriveFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snaps = MarketSimulator(default_profiles(), seed=5).run(60).snapshots

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "lake")
        sources = {}
        for h, m in MULT.items():
            write_house(self.root, h, scaled(self.snaps, m))
            sources[FV.source_name(h)] = dict(house_id=h)
            sources[FV.source_name(h, False)] = dict(house_id=h)
        write(self.root, FV.SOURCES_KEY, json.dumps(sources).encode())
        items = T.tidy(pd.DataFrame({"item_id": [1001, 1002], "name": ["Pretend Ore", "Pretend Herb"]}), "items")
        write(self.root, T.KEYS["items"], T.to_csv_gz(items))
        self.store = LocalStore(self.root)
        self.commits = []
        p1 = mock.patch.object(FDV, "to_bytes", lambda df: pickle.dumps(df))
        p2 = mock.patch.object(FDV, "from_bytes", lambda b: pickle.loads(b) if b else pd.DataFrame())
        p1.start(), p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def commit(self, files, message):
        self.commits.append(message)
        for k, v in files.items():
            write(self.root, k, v)

    def go(self, **kw):
        a = SimpleNamespace(**{**dict(hours=96.0, houses="all", save=False), **kw})
        return FDV.run(a, self.store, now=NOW, base_reader=base_reader, commit=self.commit)

    def table(self, house, name):
        with open(os.path.join(self.root, FDV.key_for(house, name)), "rb") as f:
            return pickle.loads(f.read())


class TestForeverDerive(DeriveFixture):
    def test_report_only_builds_every_table_for_every_house_and_stores_nothing(self):
        r = self.go()
        self.assertEqual(r["houses"], [110, 113, 116])
        self.assertEqual(self.commits, [])
        for h in MULT:
            for t in ("items", "item_hourly", "calibration", "calibration_history", "book_now"):
                self.assertIn(FDV.key_for(h, t), r["files"], (h, t))
        self.assertIn(FDV.HOUSES_KEY, r["files"])
        self.assertIn("REPORT ONLY", r["text"])

    def test_save_stores_each_house_in_its_own_commit_then_the_houses_list(self):
        r = self.go(save=True)
        self.assertEqual(self.commits, ["forever derive: house 110", "forever derive: house 113", "forever derive: house 116", "forever derive: houses list"])
        self.assertEqual(r["saved"], [110, 113, 116])

    def test_an_upload_failure_loses_that_house_only_and_the_list_keeps_what_is_stored(self):
        self.go(save=True)                                                      # every house stored once
        self.commits.clear()
        orig = self.commit

        def flaky(files, message):
            if message.endswith("house 113"):
                raise RuntimeError("Error while uploading 'engine/h113/items.parquet' to the Hub.") from ConnectionError("connection reset by peer")
            orig(files, message)
        with mock.patch.object(self, "commit", flaky):
            r = FDV.run(SimpleNamespace(hours=96.0, houses="all", save=True), self.store, now=NOW, base_reader=base_reader, commit=self.commit)
        self.assertEqual(r["code"], 1)
        self.assertEqual(r["saved"], [110, 116])
        self.assertIn("house 113: UPLOAD FAILED", r["text"])
        self.assertIn("connection reset by peer", r["text"])                    # the cause is shown, not only the Hub's short message
        with open(os.path.join(self.root, FDV.HOUSES_KEY), encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)["built"]), ["110", "113", "116"])   # 113's earlier tables are still stored, so it stays in the app

    def test_the_tables_match_each_house(self):
        self.go(save=True)
        a, b = self.table(110, "items").set_index("item_id"), self.table(113, "items").set_index("item_id")
        common = a.index.intersection(b.index)
        self.assertGreater(len(common), 0)
        self.assertTrue(((b.loc[common, "floor_gold"] / a.loc[common, "floor_gold"]).round(2) == 2.0).all())     # houses are never pooled
        self.assertIn("Pretend Ore", set(a["name"]))
        for col in ("floor_gold", "band_p50", "daily_est", "active_share", "confidence"):
            self.assertIn(col, a.columns)

    def test_book_now_is_the_latest_snapshot_by_price_level(self):
        self.go(save=True)
        bk = self.table(116, "book_now")
        last = scaled(self.snaps, 1.5)[-1].book
        self.assertEqual(int(bk["quantity"].sum()), int(last["quantity"].sum()))
        self.assertEqual(int(bk["listings"].sum()), len(last))
        one = bk[bk["item_id"] == bk["item_id"].iloc[0]]
        self.assertTrue(one["price_gold"].is_monotonic_increasing)

    def test_the_houses_file_takes_labels_from_the_code_so_116_is_confirmed(self):
        self.go(save=True)
        with open(os.path.join(self.root, FDV.HOUSES_KEY), encoding="utf-8") as f:
            hj = json.load(f)
        self.assertEqual(sorted(hj["built"]), ["110", "113", "116"])
        self.assertEqual(hj["labels"]["116"], dict(ruleset="pvp", faction="neutral", confidence="confirmed"))
        self.assertEqual(hj["labels"]["113"]["faction"], "horde")

    def test_the_calibration_history_grows_run_by_run(self):
        self.go(save=True)
        self.assertEqual(len(self.table(110, "calibration_history")), 1)
        with mock.patch("engine.derive.__version__", "9.9.9"):                     # a second build of the same data by another engine version is a new row
            self.go(save=True)
        self.assertEqual(len(self.table(110, "calibration_history")), 2)

    def test_one_house_failing_does_not_stop_the_others(self):
        orig = FDV.derive

        def boom(snaps, cfg, **kw):
            if snaps[0].meta.get("house") == 113:
                raise RuntimeError("boom")
            return orig(snaps, cfg, **kw)
        with mock.patch.object(FDV, "derive", boom):
            r = self.go(save=True)
        self.assertEqual(r["houses"], [110, 116])
        self.assertEqual(r["code"], 1)
        self.assertIn("113 FAILED", r["text"])

    def test_houses_option_and_report_privacy(self):
        r = self.go(houses="110,116")
        self.assertEqual(r["houses"], [110, 116])
        self.assertNotIn(FV.hf_repo().split("/")[0], r["text"])

    def test_each_house_works_out_its_flip_shelves_with_its_own_cut(self):
        seen = {}
        orig = FDV.derive

        def spy(snaps, cfg, **kw):
            seen[snaps[0].meta.get("house")] = cfg.shelf.ah_cut
            return orig(snaps, cfg, **kw)
        with mock.patch.object(FDV, "derive", spy):
            r = self.go()
        self.assertEqual(seen, {110: 0.05, 113: 0.05, 116: 0.15})
        self.assertIn("neutral 15%", r["text"])

    def test_forever_settings_are_used(self):
        r = self.go()
        self.assertIn("2+ a day", r["text"])
        self.assertEqual(FDV.FOREVER_CFG["min_daily_high"], 2.0)


class TestEndToEnd(DeriveFixture):
    """The derive job's real output for three pretend houses, read by the app (against the Streamlit stand-ins): every tab gets tables it can use."""

    def test_the_app_runs_on_what_the_derive_job_writes(self):
        import test_app_smoke as S
        r = self.go(save=True)
        data = {f"h{h}": {t: self.table(h, t) for t in FDV.TABLES if os.path.exists(os.path.join(self.root, FDV.key_for(h, t)))} for h in r["houses"]}
        with open(os.path.join(self.root, FDV.HOUSES_KEY), encoding="utf-8") as f:
            hj = json.load(f)
        S.Any.calls.clear()
        st = S.run_app(data, houses_json=hj, selected_row=0)
        market = next(d for d, k in st.seen if "Sold/day best" in d)
        self.assertGreater(len(market), 0)
        self.assertTrue(any(r_.startswith("h110") for r_ in st.reads))
        hf = next((d for d, k in st.seen if "Profit after cut (g)" in d), None)       # 116 is priced x1.5, so buying there to resell on 110 rarely pays
        errors = [a for n, a, k in S.Any.calls if n == "error"]
        self.assertEqual(errors, [])
        # Horde house: the neutral house is cheaper (x1.5 against x2), so House flips finds items to buy on 116 and resell on 113
        S.Any.calls.clear()
        st = S.run_app(data, query={"house": "113"}, houses_json=hj, selected_row=None)
        hf = next(d for d, k in st.seen if "Profit after cut (g)" in d)
        self.assertGreater(len(hf), 0)
        self.assertTrue((hf["Margin after cut (%)"] >= 15 - 1e-9).all())
        self.assertTrue((hf["Profit after cut (g)"] > 0).all())


class TestWorkflow(unittest.TestCase):
    def test_both_copies_match_and_the_schedule_is_gated(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "tools", "forever_derive.yml"), encoding="utf-8") as f:
            a = f.read()
        with open(os.path.join(root, ".github", "workflows", "forever_derive.yml"), encoding="utf-8") as f:
            b = f.read()
        self.assertEqual(a, b)
        self.assertIn("vars.FOREVER_DERIVE_BUILD == 'yes'", a)
        self.assertIn("pyarrow", a)


if __name__ == "__main__":
    unittest.main()
