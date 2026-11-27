"""The WoW Forever page (forever_page.py, forever_lib.py). Not wired into app.py: kept as a module whose helpers the port of the app to Forever data reuses.

Helpers on hand-made tables, then the whole page run against stand-ins for Streamlit and Plotly on a pretend Forever dataset folder whose day files are
built by the real daily-table code (engine/foreverdaily.py) from simulated snapshots for houses 110, 113 (prices doubled) and 116 (prices x 1.5).
It cannot judge how the page looks; it catches typos, wrong column names and crashes."""
import copy
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from helpers import mk  # noqa: F401  (sets sys.path)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_app_dir = os.path.join(_root, "frontend") if os.path.exists(os.path.join(_root, "frontend", "app.py")) else _root
sys.path.insert(0, _app_dir)

import forever_lib as F  # noqa: E402
from engine import foreverdaily as FD  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402
from test_app_smoke import Any, fake_streamlit  # noqa: E402
import types  # noqa: E402
from unittest import mock  # noqa: E402


def run_app(forever_dir=None, **kw):
    """Runs forever_page.render on its own (it is not in app.py yet) against the stand-ins for Streamlit and Plotly."""
    st = fake_streamlit(**kw)
    st.query_params.pop("watch", None)

    def stop():
        raise SystemExit(0)
    st.stop = stop
    if forever_dir:
        os.environ["FOREVER_LOCAL_DIR"] = forever_dir
    px = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    for name in ("Figure", "Scatter", "Bar", "Heatmap"):
        setattr(go, name, Any(name))
    px.graph_objects = go
    for m in ("forever_lib", "forever_page"):
        sys.modules.pop(m, None)
    try:
        with mock.patch.dict(sys.modules, {"streamlit": st, "plotly": px, "plotly.graph_objects": go}):
            try:
                import forever_page
                forever_page.render({"width": "stretch"})
            except SystemExit:
                pass
    finally:
        os.environ.pop("FOREVER_LOCAL_DIR", None)
        for m in ("forever_lib", "forever_page"):
            sys.modules.pop(m, None)
        import forever_lib  # noqa: F401  (put the helpers back for the other tests)
    return st

HMAP = F.parse_house_map(T.house_map_bytes())


def row(house, item, day, hours=24, price=1.0, units=100.0, sold=10.0, cert=5.0, gold=None):
    return dict(house=house, kind="commodity", item_id=item, day=day, hours=hours, flagged_hours=0, price_day=price, price_min=price * 0.9, price_p15=price,
                price_p30=price * 1.05, buy20=price * 1.01, buy200=price * 1.1, units_day=units, listings_day=5.0, gold_day=price * units if gold is None else gold,
                sold_cert_day=cert, sold_low_day=sold, sold_day=sold * 1.2, sold_high_day=sold * 1.5, turnover_day=sold / units, active_share=0.5, engine_version="0.7.5")


def frame(rows):
    return F.read_daily([FD.to_csv_gz(pd.DataFrame(rows))])


class TestReading(unittest.TestCase):
    def test_stored_day_files_come_back_as_numbers(self):
        d = frame([row(110, 1, "2026-11-20"), row(113, 1, "2026-11-20", price=2.0)])
        self.assertEqual(len(d), 2)
        self.assertTrue(pd.api.types.is_integer_dtype(d["house"]) and pd.api.types.is_integer_dtype(d["item_id"]))
        self.assertAlmostEqual(d.loc[d["house"] == 113, "price_day"].iloc[0], 2.0)

    def test_an_unreadable_file_is_skipped_and_a_duplicate_day_keeps_the_fuller_row(self):
        a = FD.to_csv_gz(pd.DataFrame([row(110, 1, "2026-11-20", hours=10)]))
        b = FD.to_csv_gz(pd.DataFrame([row(110, 1, "2026-11-20", hours=22)]))
        d = F.read_daily([a, b"not gzip", b])
        self.assertEqual(len(d), 1)
        self.assertEqual(int(d["hours"].iloc[0]), 22)

    def test_nothing_readable_gives_an_empty_table_with_the_columns(self):
        d = F.read_daily([])
        self.assertTrue(d.empty)
        for c in ("house", "item_id", "day", "price_day", "sold_low_day"):
            self.assertIn(c, d.columns)

    def test_item_names_from_the_stored_item_table(self):
        items = T.tidy(pd.DataFrame({"item_id": [2592, 4306], "name": ["Wool Cloth", "Silk Cloth"], "item_class": ["Trade Goods", "Trade Goods"]}), "items")
        t = F.read_items(T.to_csv_gz(items))
        self.assertEqual(F.item_names(t), {2592: "Wool Cloth", 4306: "Silk Cloth"})
        self.assertEqual(F.item_classes(t)[2592], "Trade Goods")
        self.assertTrue(F.read_items(None).empty)
        self.assertEqual(F.name_of(999, {}), "Item 999")

    def test_day_keys(self):
        self.assertEqual(F.day_of_key("daily/d_2026-11-20.csv.gz"), "2026-11-20")
        self.assertIsNone(F.day_of_key("daily/d_2026-11-20.csv"))
        self.assertIsNone(F.day_of_key("tables/forever_items.csv.gz"))

    def test_error_messages_do_not_name_the_dataset_or_its_owner(self):
        t = F.scrub("404 for https://huggingface.co/datasets/bosco277/wow-forever-data-lake/tree/main/daily (bosco277)")
        self.assertNotIn("bosco277", t)
        self.assertNotIn("wow-forever-data-lake", t)


class TestHouses(unittest.TestCase):
    def test_labels_say_ruleset_and_faction_and_flag_unverified_houses(self):
        self.assertEqual(F.house_label(110, HMAP), "110 · PvP · Alliance")
        self.assertEqual(F.house_label(116, HMAP), "116 · PvP · Neutral")                  # confirmed by the Nov 25 listing test
        self.assertEqual(F.house_label(117, HMAP), "117 · PvE · Neutral (unverified)")
        self.assertEqual(F.house_label(124, HMAP), "124 · RP-PvP · Horde")

    def test_the_same_ruleset_is_the_alliance_horde_and_neutral_house(self):
        avail = [110, 111, 113, 114, 116, 117, 119, 120, 121, 123, 124, 125]
        self.assertEqual(F.same_ruleset(110, HMAP, avail), [110, 113, 116])
        self.assertEqual(F.same_ruleset(116, HMAP, avail), [110, 113, 116])
        self.assertEqual(F.same_ruleset(120, HMAP, avail), [119, 120, 121])
        self.assertEqual(F.same_ruleset(110, HMAP, [110, 116]), [110, 116])

    def test_houses_are_ordered_by_ruleset_then_faction(self):
        self.assertEqual(F.house_order([125, 116, 110, 111, 113], HMAP), [110, 113, 116, 111, 125])

    def test_without_a_stored_map_the_slot_order_is_used(self):
        m = F.parse_house_map(None)
        self.assertEqual(F.same_ruleset(113, m, [110, 113, 116]), [110, 113, 116])
        self.assertIn("(unverified)", F.house_label(110, m))
        self.assertEqual(F.parse_house_map(b"{not json"), m)


class TestDays(unittest.TestCase):
    def daily(self):
        rows = []
        for day, hours in (("2026-11-20", 10), ("2026-11-21", 24), ("2026-11-22", 24), ("2026-11-23", 8)):
            for item in range(1, 5):
                rows.append(row(110, item, day, hours=hours if item != 4 else 6))
        return frame(rows)

    def test_the_default_day_is_the_newest_one_mostly_complete(self):
        s = F.day_summary(self.daily(), 110)
        self.assertEqual(list(s["day"]), ["2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23"])
        self.assertEqual(F.default_day(s), "2026-11-22")
        self.assertIn("partial", F.day_label("2026-11-23", s))
        self.assertEqual(F.day_label("2026-11-22", s), "Sun Nov 22")

    def test_a_house_with_only_partial_days_falls_back_to_its_newest_day(self):
        s = F.day_summary(frame([row(110, 1, "2026-11-20", hours=5), row(110, 1, "2026-11-21", hours=7)]), 110)
        self.assertEqual(F.default_day(s), "2026-11-21")
        self.assertIsNone(F.default_day(F.day_summary(self.daily(), 999)))

    def test_freshness_warns_when_the_daily_job_stopped(self):
        d = self.daily()
        text, stale = F.freshness(d, datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc))
        self.assertFalse(stale)
        self.assertIn("4 days stored", text)
        text, stale = F.freshness(d, datetime(2026, 11, 28, 20, 0, tzinfo=timezone.utc))
        self.assertTrue(stale)
        self.assertIn("FOREVER_DAILY_BUILD", text)


class TestTables(unittest.TestCase):
    def daily(self):
        return frame([
            row(110, 1, "2026-11-22", price=10.0, units=100, sold=50),       # 500 g/day seen traded
            row(110, 2, "2026-11-22", price=1.0, units=1000, sold=900),      # 900 g/day
            row(110, 3, "2026-11-22", price=100.0, units=5, sold=0),         # 0
            row(110, 4, "2026-11-22", hours=9, price=5.0, sold=400),         # partial
            row(113, 1, "2026-11-22", price=12.0, units=80), row(116, 1, "2026-11-22", price=9.0, units=60),
            row(113, 2, "2026-11-22", price=1.5, units=10), row(116, 2, "2026-11-22", price=0.8, units=500),
            row(113, 3, "2026-11-22", price=90.0, units=50, hours=10),
            row(110, 1, "2026-11-21", price=11.0), row(110, 1, "2026-11-20", hours=8, price=12.0)])

    def test_the_market_table_is_ranked_by_gold_seen_traded_and_keeps_numbers(self):
        names = {1: "Arcane Crystal", 2: "Wool Cloth", 3: "Mooncloth"}
        t = F.market_view(self.daily(), 110, "2026-11-22", names, {1: "Trade Goods", 2: "Trade Goods", 3: "Trade Goods"})
        self.assertEqual(list(t["Item"]), ["Wool Cloth", "Arcane Crystal", "Mooncloth"])     # the partial item 4 is left out by default
        self.assertEqual(list(t["Seen traded (g/day)"]), [900.0, 500.0, 0.0])
        self.assertAlmostEqual(t.loc[0, "Turnover (%)"], 90.0)
        self.assertEqual(list(t["_id"]), [2, 1, 3])
        for c in F.MARKET_COLUMNS:
            self.assertIn(c, t.columns)
            if c not in ("Item", "Class"):
                self.assertTrue(pd.api.types.is_numeric_dtype(t[c]), c)

    def test_market_filters(self):
        d = self.daily()
        self.assertEqual(len(F.market_view(d, 110, "2026-11-22", {}, complete_only=False)), 4)
        self.assertEqual(list(F.market_view(d, 110, "2026-11-22", {}, min_gold=600)["_id"]), [2, 1])     # gold in play 1,000 / 1,000 / 500: item 3 is left out
        self.assertEqual(list(F.market_view(d, 110, "2026-11-22", {2: "Wool Cloth"}, query="wool")["_id"]), [2])
        self.assertEqual(list(F.market_view(d, 110, "2026-11-22", {}, query="3")["_id"]), [3])
        self.assertEqual(len(F.market_view(d, 110, "2026-11-22", {}, {1: "Cloth"}, class_filter=["Cloth"])), 1)
        self.assertTrue(F.market_view(d, 110, "2026-11-19", {}).empty)

    def test_one_item_across_days_and_houses(self):
        d = self.daily()
        h = F.item_history(d, 110, 1)
        self.assertEqual(list(h["day"]), ["2026-11-20", "2026-11-21", "2026-11-22"])
        self.assertEqual(list(h["complete"]), [False, True, True])
        self.assertEqual(F.history_table(h)["Day"].iloc[0], "2026-11-22")                  # newest first
        a = F.item_across_houses(d, "2026-11-22", 1, HMAP)
        self.assertEqual(list(a["House"]), ["110 · PvP · Alliance", "113 · PvP · Horde", "116 · PvP · Neutral"])
        self.assertEqual(list(a["Price (g)"]), [10.0, 12.0, 9.0])

    def test_houses_side_by_side_show_where_an_item_is_cheapest(self):
        cv = F.compare_view(self.daily(), "2026-11-22", [110, 113, 116], HMAP, {1: "Arcane Crystal", 2: "Wool Cloth"})
        self.assertEqual(list(cv["Item"]), ["Arcane Crystal", "Wool Cloth"])              # biggest gap first: item 1 is 12g vs 9g (33%); item 2 is 1.0g vs 0.8g (25%; 113's 10 units do not count)
        r = cv.set_index("Item ID")
        self.assertEqual((int(r.loc[1, "Cheapest at"]), int(r.loc[1, "Dearest at"])), (116, 113))
        self.assertAlmostEqual(r.loc[1, "Gap (%)"], (12.0 / 9.0 - 1) * 100)
        self.assertTrue(np.isnan(r.loc[2, "Horde 113 (g)"]))                                # too few units in 113 to count
        self.assertAlmostEqual(r.loc[2, "Gap (%)"], (1.0 / 0.8 - 1) * 100)
        self.assertNotIn(3, r.index)                                                        # item 3: partial in 113, so only one house counts

    def test_the_headline_numbers(self):
        hd = F.market_headline(self.daily(), 110, "2026-11-22")
        self.assertEqual((hd["items"], hd["complete"]), (4, 3))
        self.assertAlmostEqual(hd["traded"], 1400.0)
        self.assertAlmostEqual(hd["selling_share"], 2 / 3)


def pretend_dataset(root, names=True):
    """daily/ files for houses 110, 113 and 116 built by engine/foreverdaily.py, the item table and the house map, in a folder laid out like the dataset."""
    snaps = MarketSimulator(default_profiles(), seed=5).run(4 * 24 + 1).snapshots
    rows = []
    for house, mult in ((110, 1.0), (113, 2.0), (116, 1.5)):
        hs = []
        for s in snaps:
            c = copy.deepcopy(s)
            c.book["unit_price"] = (c.book["unit_price"] * mult).round().astype(c.book["unit_price"].dtype)
            c.meta["house"] = house
            hs.append(c)
        rows.append(FD.build_house(hs, house))
    allrows = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.join(root, "daily"))
    os.makedirs(os.path.join(root, "tables"))
    for day, g in FD.split_days(allrows).items():
        with open(os.path.join(root, FD.day_key(day)), "wb") as f:
            f.write(FD.to_csv_gz(g))
    if names:
        ids = sorted(allrows["item_id"].unique())
        items = T.tidy(pd.DataFrame({"item_id": ids, "name": [f"Pretend Reagent {i}" for i in ids], "item_class": ["Trade Goods"] * len(ids)}), "items")
        with open(os.path.join(root, T.KEYS["items"]), "wb") as f:
            f.write(T.to_csv_gz(items))
    with open(os.path.join(root, T.KEYS["house_map"]), "wb") as f:
        f.write(T.house_map_bytes())
    return allrows


class TestForeverPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = os.path.join(cls.tmp.name, "lake")
        cls.rows = pretend_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def calls(self, label):
        return [(a, k) for n, a, k in Any.calls if n == label]

    def test_the_forever_page_shows_the_market_table_with_numbers_and_opens_an_item(self):
        Any.calls.clear()
        st = run_app(forever_dir=self.root, selected_row=0)
        market = next(d for d, k in st.seen if "Seen traded (g/day)" in d)
        self.assertGreater(len(market), 0)
        for c in ("Price (g)", "Buy 20 avg (g)", "Seen sold/day (units)", "Turnover (%)"):
            self.assertTrue(pd.api.types.is_numeric_dtype(market[c]), c)
        self.assertTrue(market["Item"].str.startswith("Pretend Reagent").all())
        keys = [k.get("key") for a, k in self.calls("plotly_chart")]
        self.assertIn("fv_market_price", keys)                                            # the clicked row's charts
        self.assertIn("fv_market_sales", keys)
        self.assertTrue(any("House" in d and "Price (g)" in d for d, k in st.seen))        # the item in every house
        for d, k in st.seen:
            self.assertEqual(k.get("width"), "stretch")

    def test_the_compare_tab_lines_up_the_three_houses_of_the_ruleset(self):
        st = run_app(forever_dir=self.root, selected_row=None)
        cv = next(d for d, k in st.seen if "Gap (%)" in d)
        for c in ("Alliance 110 (g)", "Horde 113 (g)", "Neutral 116 (g)"):
            self.assertIn(c, cv.columns)
        self.assertTrue((cv["Cheapest at"] == 110).all())                                 # 113 is priced x2 and 116 x1.5
        self.assertTrue((cv["Dearest at"] == 113).all())

    def test_the_house_goes_into_the_page_address(self):
        st = run_app(forever_dir=self.root, selected_row=None)
        self.assertEqual(st.query_params.get("house"), "110")
        self.assertNotIn("game", st.query_params)

    def test_a_house_in_the_address_is_used(self):
        Any.calls.clear()
        st = run_app(forever_dir=self.root, selected_row=None)
        st2_params = {"house": "116"}
        global fake_streamlit
        orig = fake_streamlit

        def fake(**kw):
            s = orig(**kw)
            s.query_params.update(st2_params)
            return s
        fake_streamlit = fake
        try:
            st2 = run_app(forever_dir=self.root, selected_row=None)
        finally:
            fake_streamlit = orig
        m1 = next(d for d, k in st.seen if "Seen traded (g/day)" in d)
        m2 = next(d for d, k in st2.seen if "Seen traded (g/day)" in d)
        self.assertEqual(st2.query_params.get("house"), "116")
        self.assertGreater(m2["Price (g)"].median(), m1["Price (g)"].median())            # house 116 is priced x1.5

    def test_no_names_stored_and_no_days_stored(self):
        with tempfile.TemporaryDirectory() as t:
            root = os.path.join(t, "lake")
            pretend_dataset(root, names=False)
            Any.calls.clear()
            st = run_app(forever_dir=root, selected_row=0)
            market = next(d for d, k in st.seen if "Seen traded (g/day)" in d)
            self.assertTrue(market["Item"].str.startswith("Item ").all())
            self.assertTrue(any("Item names are not stored yet" in str(a[0]) for a, k in self.calls("caption")))
        with tempfile.TemporaryDirectory() as t:
            Any.calls.clear()
            st = run_app(forever_dir=t, selected_row=0)
            self.assertFalse(st.seen)
            self.assertTrue(any("No daily tables stored yet" in str(a[0]) for a, k in self.calls("info")))


if __name__ == "__main__":
    unittest.main()
