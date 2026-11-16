import os
import sys
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import dips as Dp  # noqa: E402
from engine import windows as W  # noqa: E402
from tests.test_windows import market, two_groups  # noqa: E402


def dumps(days=40, seed=3, depth=-0.3, **kw):
    """A quiet market (1% noise) where every day a 2-hour dump of `depth` (log) hits at a RANDOM hour and the price then returns to normal."""
    starts = np.random.default_rng(seed).integers(0, 22, days)
    return market(days=days, noise=0.01, hour_effect=lambda d: {int(starts[d]): depth, int(starts[d]) + 1: depth}, **kw)


def row(res, pool="everything", k=1):
    t = res["table"]
    return t[(t["pool"] == pool) & (t["k"] == k)].iloc[0]


class TestDips(unittest.TestCase):
    def test_a_dump_that_comes_back_is_found_and_the_resale_clears_the_bar(self):
        res = Dp.analyze_dips(dumps(), two_groups())
        r = row(res)
        self.assertGreater(r["dip_days"], 150)                                      # about one dip per item per day after the first week
        self.assertGreater(r["gain_dip"], 0.95)                                     # a dump of log -0.3 (26% under), back to normal next day: 0.95 / 0.74 - 1 = +28%
        self.assertGreater(r["median_dip"], 0.25)
        self.assertEqual(r["resale"], "resale clears 5%")
        self.assertGreater(r["edge"], 0.2)                                          # far cheaper than an average hour of the same days
        self.assertEqual(r["verdict"], "holds on later days")
        self.assertLess(r["gain_random"], 0.05)                                     # the average hour of those days never clears the cut
        self.assertEqual(r["units_med"], 50)
        d = res["depth"]
        self.assertIn("20%+ under", set(d["depth"]))
        top = d[(d["pool"] == "everything") & (d["k"] == 1)].sort_values("events", ascending=False).iloc[0]
        self.assertEqual(top["depth"], "20%+ under")                                # almost every buy is the deep dump
        self.assertGreater(top["gain"], 0.95)

    def test_a_lasting_price_drop_looks_like_a_dip_but_does_not_come_back(self):
        res = Dp.analyze_dips(market(days=40, noise=0.01, step=(20, -0.2)), two_groups())
        r = row(res)
        self.assertGreater(r["dip_days"], 8)                                        # the first days after the drop trip the trigger
        self.assertLess(r["median_dip"], 0.0)                                       # the later price is the new, lower price: the cut is all that is left
        self.assertNotEqual(r["resale"], "resale clears 5%")
        self.assertLess(r["gain_dip"], 0.1)

    def test_a_quiet_market_has_no_dips_and_says_so(self):
        res = Dp.analyze_dips(market(days=30, noise=0.004), two_groups())
        self.assertTrue(res["table"].empty)
        self.assertIn("No dips to grade", Dp.format_dips(res))

    def test_too_little_history_gives_an_empty_table(self):
        res = Dp.analyze_dips(dumps(days=6), two_groups())
        self.assertTrue(res["table"].empty)

    def test_the_trigger_line_uses_only_earlier_hours(self):
        hl = dumps(days=20)
        cfg = Dp.DipConfig()
        before = Dp.trigger_lines(W.prepare(hl)[lambda x: x["usable"]], cfg)
        spiked = hl.copy()
        cut = spiked["ts"].sort_values().iloc[-30]
        spiked.loc[spiked["ts"] >= cut, "floor_gold"] *= 50.0                       # the last 30 hours go wild
        after = Dp.trigger_lines(W.prepare(spiked)[lambda x: x["usable"]], cfg)
        t_cut = pd.Timestamp(cut).floor("h")
        a = before[before["t"] <= t_cut].set_index(["item_id", "t"]).sort_index()
        b = after[after["t"] <= t_cut].set_index(["item_id", "t"]).sort_index()
        pd.testing.assert_frame_equal(a, b)                                         # nothing at or before the change moved: the line never looks ahead

    def test_a_flat_item_with_tied_prices_never_dips(self):
        res = Dp.analyze_dips(market(days=30, noise=0.0), two_groups())
        self.assertTrue(res["table"].empty)

    def test_the_report_names_the_parts_and_the_depth_split(self):
        text = Dp.format_dips(Dp.analyze_dips(dumps(), two_groups()))
        for part in ("DIP TEST", "BY HOW DEEP THE DIP WAS", "units near floor", "resale clears 5%", "20%+ under", "TYPICAL (median) buy", "CRAFTING INPUT"):
            self.assertIn(part, text)
        self.assertLess(text.index("3-6% under"), text.index("20%+ under"))        # depth rows are in depth order, not alphabetical

    def test_a_two_hour_dump_is_still_there_one_hour_later_but_not_two(self):
        res = Dp.analyze_dips(dumps(), two_groups())
        lag = res["lag"]
        x = lag[(lag["pool"] == "everything") & (lag["k"] == 2) & (lag["depth"] == "20%+ under")].set_index("lag")
        self.assertGreater(x.loc[1, "still"], 0.95)                                 # the dump lasts two hourly snapshots
        self.assertLess(x.loc[2, "still"], 0.05)                                    # and is gone at the third
        self.assertLess(x.loc[3, "still"], 0.05)
        self.assertGreater(x.loc[1, "median"], 0.25)                                # arriving one hour late still gets the bargain
        self.assertEqual(x.loc[1, "units_med"], 50)

    def test_sell_through_turns_units_near_the_floor_into_days_to_sell_and_a_plan(self):
        hl = dumps()
        days = pd.date_range("2026-06-01", periods=40, freq="D").strftime("%Y-%m-%d")
        sales = lambda per_day: pd.DataFrame([{"item_id": 1000 + i, "day": d, "sold_day": per_day} for i in range(8) for d in days])
        fast = Dp.analyze_dips(hl, two_groups(), sales=sales(100.0))["depth"]
        r = fast[(fast["pool"] == "everything") & (fast["k"] == 2) & (fast["depth"] == "20%+ under")].iloc[0]
        self.assertAlmostEqual(r["days_to_sell"], 0.5)                              # 50 units near the floor, 100 sold a day
        self.assertEqual(r["plan_qty"], 50)                                         # two days of sales is 200: the 50 units near the floor are the limit
        self.assertGreater(r["plan_profit"], 0)
        self.assertAlmostEqual(r["sell_known"], 1.0)
        slow = Dp.analyze_dips(hl, two_groups(), sales=sales(10.0))["depth"]
        r = slow[(slow["pool"] == "everything") & (slow["k"] == 2) & (slow["depth"] == "20%+ under")].iloc[0]
        self.assertAlmostEqual(r["days_to_sell"], 5.0)                              # 50 units, 10 sold a day
        self.assertEqual(r["plan_qty"], 20)                                         # capped at two days of sales
        none = Dp.analyze_dips(hl, two_groups())["depth"]                           # no sales table: the columns are empty, nothing breaks
        self.assertTrue(none["days_to_sell"].isna().all())

    def test_events_per_day_and_the_loss_definition(self):
        res = Dp.analyze_dips(dumps(), two_groups())
        d = res["depth"]
        r = d[(d["pool"] == "everything") & (d["k"] == 1) & (d["depth"] == "20%+ under")].iloc[0]
        self.assertGreater(r["per_day"], 7.0)                                        # eight items, about one dump each a day
        self.assertLess(r["per_day"], 8.5)
        falling = Dp.analyze_dips(market(days=40, noise=0.01, step=(20, -0.2)), two_groups())["table"]
        f = falling[(falling["pool"] == "everything") & (falling["k"] == 1)].iloc[0]
        self.assertAlmostEqual(f["median_dip"], -0.05, places=1)                    # the price stays at its new level: only the cut is lost
        self.assertLess(f["loss_dip"], 0.1)                                         # a flat price is not a loss

    def test_the_report_has_the_acting_sections(self):
        text = Dp.format_dips(Dp.analyze_dips(dumps(), two_groups()))
        for part in ("CAN YOU STILL BUY IT", "still there", "days to sell", "plan units", "per day", "neither"):
            self.assertIn(part, text)

    def test_resale_verdict_words(self):
        cfg = Dp.DipConfig()
        self.assertEqual(Dp.resale_verdict(0.08, 0.5, 0.1, cfg), "resale clears 5%")
        self.assertEqual(Dp.resale_verdict(0.08, 0.2, 0.4, cfg), "resale under the 5% bar")       # a good median but losers outnumber winners
        self.assertEqual(Dp.resale_verdict(0.02, 0.2, 0.1, cfg), "resale under the 5% bar")
        self.assertEqual(Dp.resale_verdict(-0.04, 0.1, 0.4, cfg), "resale loses money")
        self.assertEqual(Dp.resale_verdict(float("nan"), 0, 0, cfg), "no data")


if __name__ == "__main__":
    unittest.main()
