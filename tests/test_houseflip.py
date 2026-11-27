"""House flips (frontend/houseflip.py) and the ruleset / house picker helpers (frontend/forever_lib.py), on hand-made tables."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

from helpers import mk  # noqa: F401  (sets sys.path)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_app_dir = os.path.join(_root, "frontend") if os.path.exists(os.path.join(_root, "frontend", "app.py")) else _root
sys.path.insert(0, _app_dir)
sys.path.insert(0, os.path.join(_root, "tools"))

import forever_lib as F  # noqa: E402
import houseflip as HFL  # noqa: E402


def sell(**over):
    base = dict(item_id=[1, 2, 3], name=["Ore", "Herb", "Cloth"], floor_gold=[10.0, 5.0, 1.0], band_p50=[12.0, 4.0, 1.0], daily_est=[20.0, 50.0, 0.4],
                active_share=[0.8, 0.5, 0.9], confidence=["Medium", "Low", "Medium"])
    base.update(over)
    return pd.DataFrame(base)


def book(rows):
    return pd.DataFrame(rows, columns=["item_id", "price_gold", "quantity"])


class TestFlipTable(unittest.TestCase):
    def test_resell_price_is_the_lower_of_now_and_typical(self):
        p = HFL.resell_price(sell())
        self.assertEqual(list(p), [10.0, 4.0, 1.0])

    def test_the_cut_and_margin_set_the_most_you_can_pay(self):
        t = HFL.flip_table(book([(1, 7.0, 5), (1, 8.0, 5), (1, 8.5, 5)]), sell(), 0.05, 0.15)
        r = t.set_index("Item ID").loc[1]
        self.assertAlmostEqual(r["Pay at most (g)"], 9.5 / 1.15)                        # 8.26g: the 8.5g units are too dear
        self.assertEqual(r["Units at or under (units)"], 10)
        self.assertAlmostEqual(r["Cost (g)"], 75.0)
        self.assertAlmostEqual(r["Profit after cut (g)"], 10 * 9.5 - 75.0)
        self.assertAlmostEqual(r["Margin after cut (%)"], (9.5 / 7.5 - 1) * 100)
        self.assertAlmostEqual(r["Days to sell"], 10 / 20.0)

    def test_units_are_capped_at_what_the_resell_house_sells_in_the_chosen_days(self):
        t = HFL.flip_table(book([(2, 1.0, 500)]), sell(), 0.05, 0.15, days_to_sell=2)
        r = t.set_index("Item ID").loc[2]
        self.assertEqual(r["Buy (units)"], 100)                                         # 50 a day x 2 days, though 500 are cheap enough
        self.assertEqual(r["Units at or under (units)"], 500)

    def test_an_item_that_sells_under_one_unit_in_the_period_or_is_too_dear_is_left_out(self):
        t = HFL.flip_table(book([(3, 0.1, 50), (1, 9.9, 5)]), sell(), 0.05, 0.15, days_to_sell=2)
        self.assertTrue(t.empty)

    def test_the_neutral_cut_is_bigger(self):
        a = HFL.flip_table(book([(1, 7.0, 5)]), sell(), HFL.FEES["faction_cut"], 0.0).iloc[0]
        b = HFL.flip_table(book([(1, 7.0, 5)]), sell(), HFL.FEES["neutral_cut"], 0.0).iloc[0]
        self.assertAlmostEqual(a["Profit after cut (g)"], 5 * (9.5 - 7.0))
        self.assertAlmostEqual(b["Profit after cut (g)"], 5 * (8.5 - 7.0))

    def test_reverse_rule_big_margin_or_a_fast_seller(self):
        bk = book([(1, 6.5, 5), (2, 2.0, 5)])
        # item 1: resell 10g, 15% cut -> 8.5g; paying 6.5g is a 31% margin. Item 2: resell 4g -> 3.4g; paying 2g is 70%, but item 2 sells in only 50% of hours
        t = HFL.flip_table(bk, sell(), 0.15, 0.80, fast_share=0.75, fast_margin=0.15).set_index("Item ID")
        self.assertEqual(list(t.index), [1])                                             # item 1 is a fast seller (80% of hours): 15% is enough
        self.assertIn("fast seller", t.loc[1, "Why it qualifies"])
        t2 = HFL.flip_table(bk, sell(), 0.15, 0.30, fast_share=0.75, fast_margin=0.15).set_index("Item ID")
        self.assertEqual(sorted(t2.index), [1, 2])
        self.assertIn("margin 30%", t2.loc[2, "Why it qualifies"])

    def test_empty_inputs_give_an_empty_table_with_the_columns(self):
        for t in (HFL.flip_table(pd.DataFrame(), sell(), 0.05, 0.15), HFL.flip_table(book([(1, 1.0, 1)]), pd.DataFrame(), 0.05, 0.15)):
            self.assertTrue(t.empty)
            self.assertIn("Profit after cut (g)", t.columns)

    def test_names_come_from_the_given_names_then_the_items_table(self):
        t = HFL.flip_table(book([(1, 7.0, 5), (2, 1.0, 5)]), sell(), 0.05, 0.15, names={2: "Peacebloom"}).set_index("Item ID")
        self.assertEqual((t.loc[1, "Item"], t.loc[2, "Item"]), ("Ore", "Peacebloom"))

    def test_the_footnote_says_what_is_assumed(self):
        text = " ".join(HFL.assumptions())
        self.assertIn("5%", text)
        self.assertIn("15%", text)
        self.assertIn("not yet checked", text)

    def test_the_app_and_the_derive_job_assume_the_same_cuts(self):
        import forever_derive as FDV
        self.assertEqual(HFL.FEES["faction_cut"], FDV.AH_CUTS["alliance"])
        self.assertEqual(HFL.FEES["faction_cut"], FDV.AH_CUTS["horde"])
        self.assertEqual(HFL.FEES["neutral_cut"], FDV.AH_CUTS["neutral"])


class TestPicker(unittest.TestCase):
    def setUp(self):
        self.data = (b'{"built": {"110": "t", "113": "t", "116": "t", "111": "t", "117": "t"}, "labels": {"110": {"ruleset": "pvp", "faction": "alliance", '
                     b'"confidence": "confirmed"}, "113": {"ruleset": "pvp", "faction": "horde", "confidence": "matched"}, "116": {"ruleset": "pvp", "faction": "neutral", '
                     b'"confidence": "confirmed"}, "111": {"ruleset": "pve", "faction": "alliance", "confidence": "likely"}, "117": {"ruleset": "pve", "faction": "neutral", '
                     b'"confidence": "assumed"}}}')

    def test_houses_file_gives_the_houses_and_their_labels(self):
        built, labels, when = F.parse_houses_file(self.data)
        self.assertEqual(built, [110, 111, 113, 116, 117])
        self.assertEqual(F.picker_label(116, labels), "116 · Blackwater (neutral)")       # confirmed: no "(unverified)"
        self.assertEqual(F.picker_label(117, labels), "117 · Blackwater (neutral) (unverified)")
        self.assertEqual(F.picker_label(113, labels), "113 · Horde")
        self.assertEqual(F.parse_houses_file(None), ([], {}, {}))
        self.assertEqual(F.parse_houses_file(b"not json"), ([], {}, {}))

    def test_rulesets_then_houses(self):
        built, labels, _ = F.parse_houses_file(self.data)
        self.assertEqual(F.rulesets_of(built, labels), ["pvp", "pve"])
        self.assertEqual(F.houses_of("pvp", built, labels), [110, 113, 116])
        self.assertEqual(F.houses_of("pve", built, labels), [111, 117])
        self.assertEqual(F.neutral_of(113, labels, built), 116)
        self.assertEqual(F.neutral_of(111, labels, built), 117)
        self.assertIsNone(F.neutral_of(110, labels, [110, 113]))


if __name__ == "__main__":
    unittest.main()
