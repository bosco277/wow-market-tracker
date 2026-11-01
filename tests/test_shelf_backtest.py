import unittest

import numpy as np
import pandas as pd

from helpers import T0, mk
from engine.shelf_backtest import backtest_shelves, summarize_backtest
from engine.shelves import ShelfConfig

G = 10000


def rows_for(h):
    r, a = [], iter(range(1, 100000))

    def add(item, qty, gold):
        r.append((next(a), item, qty, int(round(gold * G)), "VERY_LONG"))
    # 7001: a 30-unit cheap shelf every 6th hour that is gone an hour later
    if h % 6 == 0:
        add(7001, 30, 1.0)
    add(7001, 1000, 1.5)
    # 7002: the cheap shelf never clears
    add(7002, 30, 1.0)
    add(7002, 1000, 1.5)
    # 7003: a shelf every 6th hour that is joined by 40 new cheaper units in the next two hours
    if h % 6 in (0, 1, 2):
        add(7003, 30, 1.0)
    if h % 6 in (1, 2):
        add(7003, 40, 0.95)
    add(7003, 1000, 1.5)
    return r


def snaps(n=130):
    return [mk(rows_for(h), ts=T0 + pd.Timedelta(hours=h)) for h in range(n)]


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bt = backtest_shelves(snaps(), ShelfConfig(), step=1, min_history_hours=48.0)

    def test_no_future_leaks_and_columns(self):
        bt = self.bt
        self.assertGreater(len(bt), 0)
        self.assertTrue((bt["t0"] >= T0 + pd.Timedelta(hours=48)).all())
        for c in ("reached_4h", "reached_12h", "hours_to_reach", "floor_ratio_4h", "inflow_2h", "first_seen", "verdict", "test_regime"):
            self.assertIn(c, bt.columns)

    def test_a_shelf_that_clears_is_reached_quickly(self):
        a = self.bt[(self.bt["item_id"] == 7001) & self.bt["first_seen"]]
        self.assertGreater(len(a), 5)
        self.assertTrue((a["reached_4h"] == 1.0).all() or a["reached_4h"].astype(float).eq(1.0).all())
        self.assertTrue((a["hours_to_reach"] == 1.0).all())
        known = a["floor_ratio_4h"].dropna()
        self.assertGreater(len(known), 5)
        self.assertTrue(((known - 1.0).abs() < 1e-9).all())      # the last shelf has no +4 h data yet: NaN, not a guess

    def test_a_shelf_that_never_clears_is_never_reached_and_counts_once(self):
        b = self.bt[self.bt["item_id"] == 7002]
        self.assertGreater(len(b), 20)
        self.assertEqual(int(b["first_seen"].sum()), 1)
        first = b[b["first_seen"]]
        self.assertEqual(float(first["reached_12h"].iloc[0]), 0.0)
        self.assertTrue(np.isnan(first["hours_to_reach"].iloc[0]))

    def test_new_cheaper_listings_show_up_as_inflow(self):
        c = self.bt[(self.bt["item_id"] == 7003) & self.bt["first_seen"]]
        self.assertGreater(len(c), 5)
        self.assertTrue(((c["inflow_2h"] - 40 / 30.0).abs() < 1e-6).all())
        a = self.bt[(self.bt["item_id"] == 7001) & self.bt["first_seen"]]
        self.assertTrue((a["inflow_2h"] == 0).all())

    def test_outcomes_are_unknown_near_the_end_of_the_data(self):
        late = self.bt[self.bt["t0"] > T0 + pd.Timedelta(hours=129 - 4)]
        self.assertTrue(late.empty or late[late["item_id"] == 7002]["reached_4h"].isna().all())

    def test_summary_groups(self):
        s = summarize_backtest(self.bt)
        self.assertIn("all shelves", set(s["group"]))
        r = s.set_index("group").loc["all shelves"]
        self.assertGreater(r["n"], 0)
        self.assertTrue(0.0 <= r["reached_12h"] <= 1.0)

    def test_too_little_data_is_refused(self):
        with self.assertRaises(ValueError):
            backtest_shelves(snaps(3), ShelfConfig())
        with self.assertRaises(ValueError):
            backtest_shelves(snaps(20), ShelfConfig(), min_history_hours=48.0)


if __name__ == "__main__":
    unittest.main()


class TestSummaryBuckets(unittest.TestCase):
    def test_extra_groups_exist(self):
        bt = backtest_shelves(snaps(), ShelfConfig(), step=1, min_history_hours=48.0)
        groups = set(summarize_backtest(bt)["group"])
        self.assertTrue(any(g.startswith("margin:") for g in groups))
        self.assertTrue(any(g.startswith("checks passed:") for g in groups))
        self.assertTrue(any(g.startswith("units:") for g in groups))
