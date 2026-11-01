import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import daily as D  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402


def hourly(days=3, item=7, start="2026-10-05", sold=2.0, spike_hours=()):
    ts = pd.date_range(start, periods=days * 24, freq="h", tz="America/New_York").tz_convert("UTC")
    h = pd.DataFrame({"item_id": item, "ts": ts, "floor_gold": 10.0, "units_listed": 100, "sold_est": sold, "spike": False, "blip": False,
                      "flag_wipe": False, "flag_long_gap": False})
    for k in spike_hours:
        h.loc[k, "spike"] = True
    return h


class TestDailyTable(unittest.TestCase):
    def test_one_row_per_item_per_eastern_day_scaled_to_24_hours(self):
        d = D.daily_table(hourly(days=3))
        self.assertEqual(list(d["day"]), ["2026-10-05", "2026-10-06", "2026-10-07"])
        self.assertTrue((d["hours"] == 24).all())
        self.assertTrue(np.allclose(d["sold_day"], 48.0))                              # 2 units an hour for 24 hours
        self.assertTrue(np.allclose(d["price_day"], 10.0))
        self.assertEqual(list(d.columns), D.COLUMNS)

    def test_flagged_and_unmeasured_hours_are_left_out_and_counted(self):
        h = hourly(days=1, spike_hours=(3, 4, 5))
        h.loc[0, "sold_est"] = np.nan                                                   # the first hour has no sales estimate
        d = D.daily_table(h).iloc[0]
        self.assertEqual(d["hours"], 20)
        self.assertEqual(d["flagged_hours"], 3)
        self.assertAlmostEqual(d["sold_day"], 48.0)                                     # scaled up from the 20 usable hours
        self.assertTrue(D.daily_table(pd.DataFrame()).empty)

    def test_merge_keeps_history_and_prefers_the_fuller_day(self):
        old = D.daily_table(hourly(days=2, sold=1.0))
        partial = D.daily_table(hourly(days=2, sold=5.0).head(30))                      # day 1 full, day 2 only 6 hours
        merged = D.merge_daily(old, partial)
        self.assertEqual(len(merged), 2)
        m = merged.set_index("day")
        self.assertAlmostEqual(m.loc["2026-10-05", "sold_day"], 120.0)                  # tie on hours: the new row wins
        self.assertAlmostEqual(m.loc["2026-10-06", "sold_day"], 24.0)                   # the old, fuller day is kept
        later = D.daily_table(hourly(days=1, start="2026-10-20"))
        self.assertEqual(len(D.merge_daily(merged, later)), 3)                          # new days are added
        self.assertEqual(len(D.merge_daily(None, later)), 1)
        self.assertTrue(D.merge_daily(None, pd.DataFrame()).empty)

    def test_coverage(self):
        d = pd.concat([D.daily_table(hourly(days=2, item=1)), D.daily_table(hourly(days=1, item=2))], ignore_index=True)
        c = D.coverage(d).set_index("day")
        self.assertEqual(c.loc["2026-10-05", "complete_items"], 2)
        self.assertEqual(c.loc["2026-10-06", "complete_items"], 1)


class TestRunDaily(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snaps = MarketSimulator(default_profiles(), seed=9).run(80).snapshots

    def _run(self, tables, upload=False):
        import lake
        items = pd.DataFrame({"item_id": [1001, 1002, 1003, 1004]})
        store = dict(tables)
        store.setdefault("items", items)
        buf = io.StringIO()
        uploaded = []
        with tempfile.TemporaryDirectory() as d:
            args = lake.argparse.Namespace(ids="", out=d, upload=upload)
            with mock.patch.object(lake, "_load_all", return_value=self.snaps), \
                    mock.patch.object(pd.DataFrame, "to_parquet", lambda self_, path, **kw: open(path, "wb").close()), \
                    mock.patch.object(lake, "_upload_folder_to_engine", side_effect=lambda folder, msg: uploaded.append(sorted(os.listdir(folder)))), \
                    contextlib.redirect_stdout(buf):
                out = lake.run_daily(None, args, read_table=lambda n: store.get(n, pd.DataFrame()))
            wrote = os.path.exists(os.path.join(d, "item_daily.csv"))
        return out, buf.getvalue(), uploaded, wrote

    def test_builds_the_history_from_snapshots_and_says_how_many_days_are_still_needed(self):
        out, text, uploaded, wrote = self._run({})
        self.assertGreater(len(out), 0)
        self.assertEqual(list(out.columns), D.COLUMNS)
        self.assertTrue(wrote)
        self.assertEqual(uploaded, [])                                                  # no upload unless asked
        self.assertIn("DAYS STORED", text)
        self.assertIn("more well-covered days needed", text)

    def test_merges_with_what_is_stored_and_uploads_when_asked(self):
        first, _, _, _ = self._run({})
        old = first.assign(day="2026-09-01")                                            # an earlier day already in the history
        out, text, uploaded, _ = self._run({"item_daily": old}, upload=True)
        self.assertIn("2026-09-01", set(out["day"]))
        self.assertGreater(len(set(out["day"])), 1)
        self.assertEqual(uploaded, [["item_daily.parquet"]])

    def test_refuses_without_items(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = lake.run_daily(None, lake.argparse.Namespace(ids="", out="x", upload=False), read_table=lambda n: pd.DataFrame())
        self.assertTrue(out.empty)
        self.assertIn("No items to cover", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
