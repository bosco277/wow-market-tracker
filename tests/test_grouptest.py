import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import grouptest as GT  # noqa: E402


def fake_tables(seed=3):
    """Three groups of 30: Tight (alike), Noise (items unrelated to each other), Mixed (two id bands that behave very differently)."""
    rng = np.random.default_rng(seed)
    rows = []
    iid = 1000

    def add(group, family, sales, price, n, id0, conf="High", name=None, tier=0, sib=1, quality="COMMON", use=""):
        nonlocal iid
        for k in range(n):
            iid += 1
            rows.append(dict(item_id=id0 + k, name=name or f"{group} {iid}", group=group, family=family, use=use, quality=quality, siblings=sib,
                             tier_guess=tier, daily_est=float(sales(k)), floor_gold=float(price(k)), confidence=conf if k % 5 else "Low"))

    add("Consumables: Tight", "Consumables", lambda k: 5000 * 10 ** rng.normal(0, 0.05), lambda k: 200 * 10 ** rng.normal(0, 0.05), 30, 300000)
    add("Consumables: Noise", "Consumables", lambda k: 10 ** rng.uniform(0, 4), lambda k: 10 ** rng.uniform(0, 3), 30, 310000)
    add("Crafting materials: Mixed", "Crafting materials", lambda k: 1000 * 10 ** rng.normal(0, 0.05) if k < 20 else 10 * 10 ** rng.normal(0, 0.05),
        lambda k: 5.0, 40, 10000)                                   # item ids are rewritten below so the busy half sits in a different id band
    items = pd.DataFrame(rows)
    # make the Mixed group split cleanly by id band: first 20 in band "0k", last 20 in band "40k"
    mixed = items["group"] == "Crafting materials: Mixed"
    ids = np.where(np.arange(mixed.sum()) < 20, 1000 + np.arange(mixed.sum()), 40000 + np.arange(mixed.sum()))
    items.loc[mixed, "item_id"] = ids
    # three-tier names: tier 2 costs double and sells half of tier 1
    t = []
    for n in range(8):
        for tier in (1, 2, 3):
            t.append(dict(item_id=500000 + n * 10 + tier, name=f"Flask {n}", group="Consumables: Flasks", family="Consumables", use="", quality="COMMON",
                          siblings=3, tier_guess=tier, daily_est=100.0 / tier, floor_gold=10.0 * tier, confidence="High"))
    items = pd.concat([items, pd.DataFrame(t)], ignore_index=True)
    groups = items[["item_id", "name", "group", "family", "use", "quality", "siblings", "tier_guess"]].copy()
    return groups, items[["item_id", "floor_gold", "daily_est", "confidence"]].copy()


class TestLooMedian(unittest.TestCase):
    def test_never_uses_the_item_itself(self):
        df = pd.DataFrame({"k": ["a"] * 5, "v": [1.0, 2.0, 3.0, 4.0, 100.0]})
        out = GT._loo_median(df, ["k"], "v")
        self.assertAlmostEqual(out[4], 2.5)            # the 100 is left out of its own guess
        self.assertAlmostEqual(out[0], 3.5)            # median of 2,3,4,100

    def test_too_few_peers_gives_nothing(self):
        df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
        self.assertTrue(np.isnan(GT._loo_median(df, ["k"], "v")).all())
        self.assertEqual(len(GT._loo_median(pd.DataFrame({"k": [], "v": []}), ["k"], "v")), 0)


class TestGroupTest(unittest.TestCase):
    def setUp(self):
        self.groups, self.items = fake_tables()
        self.res = GT.grouptest(self.groups, self.items)

    def test_a_tight_group_beats_guessing_and_noise_does_not(self):
        g = self.res["groups"].set_index("group")
        self.assertLess(g.loc["Consumables: Tight", "sales_off_group"], 1.2)
        self.assertGreater(g.loc["Consumables: Tight", "sales_gain"], 3)
        self.assertTrue(g.loc["Consumables: Tight", "verdict"].startswith(("adds", "helps")))
        self.assertGreater(g.loc["Consumables: Noise", "sales_off_group"], 2)           # unrelated items: the group says little
        self.assertNotIn("adds to its family", g.loc["Consumables: Noise", "verdict"])

    def test_small_groups_get_no_verdict(self):
        groups = pd.concat([self.groups, pd.DataFrame([dict(item_id=900001, name="Lonely", group="Key: Key", family="Other", use="", quality="COMMON", siblings=1, tier_guess=0)])])
        items = pd.concat([self.items, pd.DataFrame([dict(item_id=900001, floor_gold=5.0, daily_est=3.0, confidence="High")])])
        g = GT.grouptest(groups, items)["groups"].set_index("group")
        self.assertEqual(g.loc["Key: Key", "verdict"], "too few to judge")
        self.assertEqual(g.loc["Key: Key", "scored"], 0)

    def test_low_confidence_numbers_are_flagged_when_reliable_items_are_too_few(self):
        items = self.items.copy()
        keep = items["item_id"].between(310000, 310999)
        items.loc[keep, "confidence"] = "Low"
        g = GT.grouptest(self.groups, items)["groups"].set_index("group")
        self.assertIn("Low-confidence", g.loc["Consumables: Noise", "verdict"])

    def test_coarse_group_is_split_by_id_band(self):
        c = self.res["coarse"].set_index("group")
        self.assertIn("Crafting materials: Mixed", c.index)
        row = c.loc["Crafting materials: Mixed"]
        self.assertEqual(row["best_split"], "id_band")
        self.assertGreater(row["gain"], 3)
        self.assertIn("0k", row["slices"])
        self.assertNotIn("Consumables: Tight", c.index)                                # alike items are not "coarse"

    def test_tiers(self):
        r = self.res["tiers_ratios"].set_index("tier_guess")
        self.assertAlmostEqual(r.loc[2, "sales_vs_lowest_tier"], 0.5)
        self.assertAlmostEqual(r.loc[3, "price_vs_lowest_tier"], 3.0)
        guess = self.res["tiers_guess"].set_index("what")
        self.assertIn("sales per day", guess.index)
        self.assertGreaterEqual(guess.loc["sales per day", "items"], 24)

    def test_empty_and_missing_inputs_do_not_crash(self):
        self.assertEqual(GT.grouptest(pd.DataFrame(), self.items), {})
        self.assertEqual(GT.grouptest(self.groups, pd.DataFrame({"item_id": [1]})), {})
        bad = self.items.assign(daily_est=0.0)
        self.assertEqual(GT.grouptest(self.groups, bad), {})


class TestCommand(unittest.TestCase):
    def test_prints_and_writes_files(self):
        import lake
        groups, items = fake_tables()
        tables = {"items": items, "item_groups": groups}
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            args = lake.argparse.Namespace(out=d)
            with contextlib.redirect_stdout(buf):
                res = lake.run_grouptest(None, args, read_table=lambda n: tables.get(n, pd.DataFrame()))
            self.assertTrue(res)
            for f in ("grouptest_groups.csv", "grouptest_families.csv", "grouptest_coarse.csv", "grouptest_tiers.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
        text = buf.getvalue()
        for s in ("GROUP TEST", "BY GROUP", "Consumables: Tight", "SAME-NAME QUALITY TIERS", "WHAT SPLITS THEM", "split by id_band"):
            self.assertIn(s, text)

    def test_missing_tables_say_what_to_run(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_grouptest(None, lake.argparse.Namespace(out="x"), read_table=lambda n: pd.DataFrame())
        self.assertEqual(res, {})
        self.assertIn("groups with upload ticked", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
