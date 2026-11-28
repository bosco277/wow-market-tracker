"""tools/forever_latest.py (hourly current prices and order book) and the app's overlay of now.parquet on the daily build (frontend/forever_lib.apply_now), on the
pretend dataset of test_forever_derive (houses 110, 113 x2, 116 x1.5; 60 simulated hours from Oct 1 00:00)."""
import os
import pickle
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from helpers import mk  # noqa: F401  (sets sys.path)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "tools"))
sys.path.insert(0, os.path.join(_root, "frontend") if os.path.exists(os.path.join(_root, "frontend", "app.py")) else _root)

import forever_latest as FL  # noqa: E402
import forever_derive as FDV  # noqa: E402
import forever_lib as F  # noqa: E402
from test_forever_adapter import write_house  # noqa: E402
from test_forever_derive import DeriveFixture, scaled  # noqa: E402

NOW = datetime(2026, 10, 3, 12, 50, tzinfo=timezone.utc)


class TestLatest(DeriveFixture):
    def latest(self, now=NOW, **kw):
        a = SimpleNamespace(**{**dict(houses="all", save=False), **kw})
        return FL.run(a, self.store, now=now, reader=pickle.loads, commit=self.commit)

    def test_report_only_reads_the_newest_snapshot_of_every_house_and_stores_nothing(self):
        r = self.latest()
        self.assertEqual(self.commits, [])
        self.assertEqual(r["code"], 0, r["text"])
        for h in (110, 113, 116):
            self.assertIn(FDV.key_for(h, "now"), r["files"])
            self.assertIn(FDV.key_for(h, "book_now"), r["files"])
        nt = pickle.loads(r["files"][FDV.key_for(110, "now")])
        last = self.snaps[-1].book
        want = (last.groupby("item_id")["unit_price"].min() / 10000.0)
        got = nt.set_index("item_id")["floor_gold"]
        self.assertTrue(np.allclose(got.loc[want.index], want))
        self.assertEqual(pd.Timestamp(nt["now_ts"].iloc[0]), pd.Timestamp(self.snaps[-1].ts))
        self.assertTrue((nt.set_index("item_id")["units_listed"] == last.groupby("item_id")["quantity"].sum()).all())

    def test_a_new_hour_with_cheaper_listings_shows_as_cheap_now(self):
        self.go(save=True)                                                     # the daily build, for the 72-hour ranges
        newer = scaled([self.snaps[-1]], 0.5)[0]
        newer = type(newer)(ts=newer.ts + pd.Timedelta(hours=1), book=newer.book, kind=newer.kind, source=newer.source, meta=dict(newer.meta))
        write_house(self.root, 110, [newer])
        r = self.latest(save=True)
        self.assertEqual(self.commits[-1][0] if isinstance(self.commits[-1], tuple) else self.commits[-1], "forever latest: 3 house(s)")
        nt = self.table(110, "now")
        self.assertEqual(pd.Timestamp(nt["now_ts"].iloc[0]), pd.Timestamp(newer.ts))
        self.assertTrue((nt["now_percentile"].dropna() <= 0.15).mean() > 0.9)   # half price: nearly everything is at the bottom of its range
        other = self.table(113, "now")
        self.assertEqual(pd.Timestamp(other["now_ts"].iloc[0]), pd.Timestamp(self.snaps[-1].ts))
        self.assertIn("cheap now", r["text"])

    def test_just_after_midnight_it_uses_yesterdays_newest_file(self):
        r = self.latest(now=datetime(2026, 10, 4, 0, 10, tzinfo=timezone.utc))
        nt = pickle.loads(r["files"][FDV.key_for(116, "now")])
        self.assertEqual(pd.Timestamp(nt["now_ts"].iloc[0]), pd.Timestamp(self.snaps[-1].ts))

    def test_without_a_daily_build_the_rank_is_blank_but_prices_are_there(self):
        r = self.latest()
        nt = pickle.loads(r["files"][FDV.key_for(110, "now")])
        self.assertTrue(nt["now_percentile"].isna().all())
        self.assertTrue(nt["floor_gold"].notna().all())

    def test_it_never_writes_the_daily_build_tables(self):
        r = self.latest(save=True)
        self.assertTrue(all(k.endswith(("/now.parquet", "/book_now.parquet")) for k in r["files"]))


class TestOverlay(unittest.TestCase):
    def items(self):
        return pd.DataFrame({"item_id": [1, 2, 3], "floor_gold": [10.0, 5.0, 1.0], "units_listed": [100, 50, 10], "n_listings": [5, 4, 3],
                             "now_percentile": [0.5, 0.5, 0.5], "daily_est": [20.0, 30.0, 40.0]})

    def now(self, ts="2026-11-28 01:32"):
        return pd.DataFrame({"item_id": [1, 2], "floor_gold": [8.0, 6.0], "units_listed": [90, 0 + 70], "n_listings": [4, 6], "now_percentile": [0.1, 0.9],
                             "now_ts": pd.Timestamp(ts)})

    def test_newer_current_numbers_are_laid_over_the_build(self):
        out, ts = F.apply_now(self.items(), self.now(), pd.Timestamp("2026-11-27 12:32"))
        self.assertEqual(ts, pd.Timestamp("2026-11-28 01:32"))
        self.assertEqual(list(out["floor_gold"]), [8.0, 6.0, 1.0])               # item 3 is not listed now: it keeps the build's numbers
        self.assertEqual(list(out["now_percentile"]), [0.1, 0.9, 0.5])
        self.assertEqual(list(out["daily_est"]), [20.0, 30.0, 40.0])             # rates stay the build's

    def test_older_current_numbers_are_ignored(self):
        out, ts = F.apply_now(self.items(), self.now("2026-11-27 10:32"), pd.Timestamp("2026-11-27 12:32"))
        self.assertIsNone(ts)
        self.assertEqual(list(out["floor_gold"]), [10.0, 5.0, 1.0])

    def test_nothing_to_lay_over(self):
        out, ts = F.apply_now(self.items(), pd.DataFrame(), None)
        self.assertIsNone(ts)
        self.assertEqual(list(out["floor_gold"]), [10.0, 5.0, 1.0])


class TestWorkflow(unittest.TestCase):
    def test_both_copies_match_and_the_schedule_is_gated(self):
        with open(os.path.join(_root, "tools", "forever_latest.yml"), encoding="utf-8") as fh:
            a = fh.read()
        with open(os.path.join(_root, ".github", "workflows", "forever_latest.yml"), encoding="utf-8") as fh:
            b = fh.read()
        self.assertEqual(a, b)
        self.assertIn("vars.FOREVER_LATEST_BUILD == 'yes'", a)


if __name__ == "__main__":
    unittest.main()
