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
from engine import groups as G  # noqa: E402

FAKE = {
    1: dict(id=1, name="Iceweb Spider Silk", item_class={"id": 7, "name": "Tradeskill"}, item_subclass={"id": 5, "name": "Cloth"}, quality={"type": "COMMON"},
            level=1, sell_price=100, purchase_price=0, is_stackable=True, max_count=0),
    2: dict(id=2, name="Algari Healing Potion", item_class={"id": 0, "name": "Consumable"}, item_subclass={"id": 1, "name": "Potion"}, quality={"type": "COMMON"},
            level=80, sell_price=2500),
    3: dict(id=3, name="Algari Healing Potion", item_class={"id": 0, "name": "Consumable"}, item_subclass={"id": 1, "name": "Potion"}, quality={"type": "UNCOMMON"},
            level=80, sell_price=2500),
    4: dict(id=4, name="Saronite Ore", item_class={"id": 7, "name": "Tradeskill"}, item_subclass={"id": 7, "name": "Metal & Stone"}, quality={"type": "COMMON"}, sell_price=40000),
    5: dict(id=5, name="Mystery", item_class={"id": 7, "name": "Tradeskill"}),                      # no subclass, no price: must not break
}


def fake_get(calls=None):
    def get(url):
        i = int(url.split("/item/")[1].split("?")[0])
        if calls is not None:
            calls.append(i)
        if i == 99:
            return 404, None
        if i == 98:
            return 500, None
        return (200, FAKE[i]) if i in FAKE else (404, None)
    return get


class TestGroups(unittest.TestCase):
    def test_parse_and_tag(self):
        df = G.fetch_items([1, 2, 3, 4, 5], fake_get(), workers=2, per_second=1000)
        t = G.tag(df).set_index("item_id")
        self.assertEqual(t.loc[1, "group"], "Crafting materials: Cloth")
        self.assertEqual(t.loc[1, "family"], "Crafting materials")
        self.assertEqual(t.loc[2, "group"], "Consumables: Potion")
        self.assertEqual(t.loc[2, "use"], "Healing potion")
        self.assertEqual(t.loc[4, "use"], "Ore or bar")
        self.assertAlmostEqual(t.loc[4, "vendor_sell_gold"], 4.0)
        self.assertEqual(t.loc[5, "group"], "Crafting materials")                                  # missing subclass handled
        self.assertTrue(np.isnan(t.loc[5, "vendor_sell_gold"]))

    def test_quality_tier_siblings(self):
        t = G.tag(G.fetch_items([2, 3, 1], fake_get(), workers=1, per_second=1000)).set_index("item_id")
        self.assertEqual((t.loc[2, "siblings"], t.loc[3, "siblings"], t.loc[1, "siblings"]), (2, 2, 1))
        self.assertEqual((t.loc[2, "tier_guess"], t.loc[3, "tier_guess"], t.loc[1, "tier_guess"]), (1, 2, 0))

    def test_unknown_and_failing_items_are_skipped_without_crashing(self):
        df = G.fetch_items([1, 99, 98], fake_get(), workers=2, per_second=1000)
        self.assertEqual(list(df["item_id"]), [1])

    def test_refresh_only_asks_about_new_ids(self):
        first = G.refresh(None, [1, 2], fake_get(), workers=1, per_second=1000)
        calls = []
        second = G.refresh(first, [1, 2, 4], fake_get(calls), workers=1, per_second=1000)
        self.assertEqual(calls, [4])
        self.assertEqual(sorted(second["item_id"]), [1, 2, 4])
        self.assertEqual(sorted(G.refresh(second, [1], fake_get([]), workers=1, per_second=1000)["item_id"]), [1])    # items no longer wanted drop out

    def test_group_summary(self):
        g = G.refresh(None, [1, 2, 3, 4], fake_get(), workers=1, per_second=1000)
        items = pd.DataFrame({"item_id": [1, 2, 3, 4], "floor_gold": [1.0, 2.0, 3.0, 4.0], "daily_est": [100.0, 400.0, 600.0, 10.0],
                              "confidence": ["High", "Low", "Medium", "Low"]})
        s = G.group_summary(g, items).set_index("group")
        self.assertEqual(s.loc["Consumables: Potion", "items"], 2)
        self.assertAlmostEqual(s.loc["Consumables: Potion", "sold_per_day_median"], 500.0)
        self.assertAlmostEqual(s.loc["Consumables: Potion", "share_low_confidence"], 0.5)
        self.assertTrue(G.group_summary(pd.DataFrame(), items).empty)

    def test_command_prints_summary_and_never_needs_a_snapshot_store(self):
        import lake
        items = pd.DataFrame({"item_id": [1, 2, 3, 4], "floor_gold": [1.0, 2.0, 3.0, 4.0], "daily_est": [100.0, 400.0, 600.0, 10.0],
                              "confidence": ["High", "Low", "Medium", "Low"]})
        tables = {"items": items}
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            args = lake.argparse.Namespace(ids="", out=d, upload=False)
            with contextlib.redirect_stdout(buf):
                out = lake.run_groups(None, args, get=fake_get(), read_table=lambda n: tables.get(n, pd.DataFrame()))
            self.assertEqual(len(out), 4)
            self.assertTrue(os.path.exists(os.path.join(d, "item_groups.csv")))
        text = buf.getvalue()
        self.assertIn("FAMILIES", text)
        self.assertIn("Crafting materials: Cloth", text)
        self.assertIn("FINER USES", text)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lake.run_groups(None, lake.argparse.Namespace(ids="", out="x", upload=False), get=fake_get(), read_table=lambda n: pd.DataFrame())
        self.assertIn("No item ids", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
