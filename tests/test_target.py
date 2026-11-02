import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import target as TG  # noqa: E402


def shared_world(n=60, days=40, seed=1, hours=24):
    """A true slowly drifting level per item. theirs = that level plus a one-snapshot blip (+0.4 log10) on 6% of days; own = the same level with a little noise (a median of many hours)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-09-20", periods=days)
    base = rng.uniform(0.5, 3.0, n)[:, None]
    level = base + np.cumsum(rng.normal(0, 0.005, (n, days)), axis=1)
    theirs = level + (rng.random((n, days)) < 0.06) * 0.4 + rng.normal(0, 0.01, (n, days))
    own = level + rng.normal(0, 0.003, (n, days))
    ids = np.arange(n) + 5001
    day = np.tile(dates.strftime("%Y-%m-%d"), n)
    iid = np.repeat(ids, days)
    hist = pd.DataFrame({"item_id": iid, "day": day, "price": (10 ** theirs * 10000.0).ravel(), "quantity": 100.0})
    own_df = pd.DataFrame({"item_id": iid, "day": day, "hours": hours, "price_day": (10 ** own).ravel(), "units_day": 100.0})
    groups = pd.DataFrame({"item_id": ids, "group": "G: x", "family": np.where(np.arange(n) % 2 == 0, "Herbs", "Gems")})
    return own_df, hist, groups


class TestCompareTargets(unittest.TestCase):
    def test_a_median_of_hours_is_calmer_and_misses_one_snapshot_blips(self):
        own, hist, groups = shared_world()
        r = TG.compare_targets(own, hist, groups)
        o = r["overall"].iloc[0]
        self.assertEqual(r["days"], 39)                      # the first day has no day before it
        self.assertGreater(o["calmer_x"], 2.5)
        self.assertLess(o["own_big"], 1.0)
        self.assertGreater(o["theirs_big"], 5.0)
        self.assertGreater(o["blip"], 80.0)                  # nearly all of their big moves are blips that ours barely registers
        self.assertLess(o["confirmed"], 10.0)
        self.assertGreater(o["within10"], 85.0)              # the levels agree
        self.assertLess(abs(o["level_gap"]), 2.0)

    def test_two_identical_measures_look_identical(self):
        own, hist, _ = shared_world()
        own["price_day"] = hist["price"] / 10000.0
        o = TG.compare_targets(own, hist)["overall"].iloc[0]
        self.assertAlmostEqual(o["calmer_x"], 1.0, places=6)
        self.assertAlmostEqual(o["corr"], 1.0, places=6)
        self.assertAlmostEqual(o["confirmed"], 100.0, places=6)

    def test_real_moves_are_confirmed(self):
        own, hist, _ = shared_world()
        # a real 60% jump up on day 20 that both measures see, for every item
        d = pd.Timestamp("2026-10-10")
        own.loc[pd.to_datetime(own["day"]) >= d, "price_day"] *= 1.6
        hist.loc[pd.to_datetime(hist["day"]) >= d, "price"] *= 1.6
        o = TG.compare_targets(own, hist)["overall"].iloc[0]
        self.assertGreater(o["confirmed"], 10.0)

    def test_incomplete_days_and_gaps_are_left_out(self):
        own, hist, _ = shared_world(days=30)
        full = TG.paired_changes(own, hist)
        own2 = own.copy()
        own2.loc[pd.to_datetime(own2["day"]) == pd.Timestamp("2026-10-01"), "hours"] = 6    # a thin day: dropped, and it breaks the chain around it
        thin = TG.paired_changes(own2, hist)
        self.assertEqual(len(full) - len(thin), 2 * own["item_id"].nunique())
        self.assertTrue((thin["day"] != pd.Timestamp("2026-10-01")).all())
        self.assertTrue((thin["day"] != pd.Timestamp("2026-10-02")).all())

    def test_family_table_and_thin_overlap(self):
        own, hist, groups = shared_world()
        fam = TG.compare_targets(own, hist, groups)["family"]
        self.assertEqual(set(fam["family"]), {"Herbs", "Gems"})
        thin = TG.compare_targets(own[own["item_id"] < 5004], hist)
        self.assertNotIn("overall", thin)
        self.assertEqual(TG.compare_targets(pd.DataFrame(), hist), {"pairs": 0})


def own_with_depth(n=40, days=30, seed=3, calm=True):
    """item_daily with a floor that jumps (one cheap listing) and depth prices that do not."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-10-01", periods=days)
    base = rng.uniform(0.5, 3.0, n)[:, None]
    level = base + np.cumsum(rng.normal(0, 0.005, (n, days)), axis=1)
    floor = level - (rng.random((n, days)) < 0.2) * 0.3 + rng.normal(0, 0.005, (n, days))
    p15 = level + 0.05 + (rng.normal(0, 0.005, (n, days)) if calm else rng.normal(0, 0.3, (n, days)))
    p30 = level + 0.08 + rng.normal(0, 0.004, (n, days))
    ids = np.repeat(np.arange(n) + 7001, days)
    return pd.DataFrame({"item_id": ids, "day": np.tile(dates.strftime("%Y-%m-%d"), n), "hours": 24, "price_day": (10 ** floor).ravel(), "units_day": 100.0,
                         "price_p15": (10 ** p15).ravel(), "price_p30": (10 ** p30).ravel()})


class TestOwnMeasures(unittest.TestCase):
    def test_depth_prices_are_calmer_than_a_floor_set_by_single_listings(self):
        m = TG.own_measures(own_with_depth()).set_index("measure")
        self.assertEqual(set(m.index), {"price_day", "price_p15", "price_p30"})
        self.assertEqual(m["changes"].nunique(), 1)                                   # every definition scored on the same item-days
        self.assertGreater(m.loc["price_p15", "calmer_x"], 3.0)
        self.assertAlmostEqual(m.loc["price_day", "calmer_x"], 1.0)
        self.assertAlmostEqual(m.loc["price_p15", "vs_floor"], 12.0, delta=6.0)       # depth prices sit above the floor (about +0.05 log10 here, more when the floor dips)
        self.assertLess(m.loc["price_p15", "big"], m.loc["price_day", "big"])

    def test_needs_the_depth_columns_and_enough_days(self):
        own = own_with_depth()
        self.assertTrue(TG.own_measures(own.drop(columns=["price_p15", "price_p30"])).empty)
        self.assertTrue(TG.own_measures(own.assign(price_p15=np.nan, price_p30=np.nan)).empty)
        self.assertTrue(TG.own_measures(pd.DataFrame()).empty)
        thin = own.copy()
        thin.loc[pd.to_datetime(thin["day"]).dt.day % 2 == 0, "hours"] = 3                # every other day incomplete: no consecutive pairs
        self.assertTrue(TG.own_measures(thin).empty)


class TestRunner(unittest.TestCase):
    def test_runner_prints_the_comparison_and_a_readout(self):
        import lake
        own, hist, groups = shared_world()
        tables = {"item_daily": own, "market_history": hist, "item_groups": groups}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_target_test(None, type("A", (), {})(), read_table=lambda n: tables.get(n, pd.DataFrame()))
        out = buf.getvalue()
        self.assertIn("overall", res)
        for s in ("PRICE TARGET TEST", "ALL SHARED ITEMS", "BY FAMILY", "calmer_x", "Read-out:", "barely register"):
            self.assertIn(s, out)
        self.assertFalse(any(line.startswith("==") for line in out.splitlines()))

    def test_runner_prints_our_own_definitions_even_without_undermine(self):
        import lake
        tables = {"item_daily": own_with_depth()}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_target_test(None, type("A", (), {})(), read_table=lambda n: tables.get(n, pd.DataFrame()))
        out = buf.getvalue()
        self.assertIn("own", res)
        for s in ("OUR PRICE DEFINITIONS", "cheapest 15% of units", "Read-out: the cheapest", "mode undermine"):
            self.assertIn(s, out)

    def test_runner_tells_you_to_rerun_daily_when_depth_prices_are_missing(self):
        import lake
        own, hist, _ = shared_world()
        tables = {"item_daily": own, "market_history": hist}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lake.run_target_test(None, type("A", (), {})(), read_table=lambda n: tables.get(n, pd.DataFrame()))
        out = buf.getvalue()
        self.assertIn("Run mode daily (upload ticked) with this version", out)
        self.assertIn("PRICE TARGET TEST vs UNDERMINE", out)

    def test_runner_says_what_to_run_when_a_table_is_missing_or_overlap_is_thin(self):
        import lake
        own, hist, _ = shared_world()
        for tables, needle in (({"market_history": hist}, "mode daily"), ({"item_daily": own}, "mode undermine"), ({"item_daily": own[own["item_id"] < 5003], "market_history": hist}, "too few consecutive days")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lake.run_target_test(None, type("A", (), {})(), read_table=lambda n, t=tables: t.get(n, pd.DataFrame()))
            self.assertIn(needle, buf.getvalue())

    def test_workflow_lists_the_mode_and_skips_saving_results(self):
        for rel in (".github/workflows/pick_items.yml", "tools/pick_items.yml"):
            with open(os.path.join(_root, rel)) as f:
                text = f.read()
            self.assertIn("target_test]", text)
            self.assertIn("inputs.mode != 'target_test'", text)


if __name__ == "__main__":
    unittest.main()
