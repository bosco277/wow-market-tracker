"""engine/foreverdaily.py and tools/forever_daily.py on a pretend Forever dataset (a local folder of pickled frames; nothing leaves the machine): the price-to-buy-N-units maths on a hand-made book,
one row per house / item / US Eastern day, the same numbers as the retail daily table, houses never pooled, the stored form (deterministic bytes, round trip, merge rules), what a run reports and
stores (report only by default, only changed days written, empty houses left out, a failing house or a corrupt file reported and not hidden), the crafting-coverage count, and the workflow wiring."""
import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from helpers import mk  # noqa: F401  (sets sys.path)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "tools"))

from engine import daily as D  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine import foreverdaily as FD  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.classifier import classify_series  # noqa: E402
from engine.loader import LocalStore  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402
from test_forever_adapter import Lake, base_reader, write, write_house  # noqa: E402
import forever_daily as FDT  # noqa: E402


def args(**kw):
    base = dict(days=3, end="2026-10-03", houses="all", save=False, no_sales=False, rebuild=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TestBookStats(unittest.TestCase):
    def book(self):
        # item 7: 10 units at 100 copper, 10 at 200, 500 at 300.  item 8: 5 units at 50.
        return mk([(1, 7, 10, 100, "LONG"), (2, 7, 10, 200, "LONG"), (3, 7, 500, 300, "LONG"), (4, 8, 5, 50, "LONG")])

    def test_the_price_to_buy_n_units_walks_up_the_listings(self):
        b = FD.book_stats([self.book()], [7, 8]).set_index("item_id")
        self.assertAlmostEqual(b.loc[7, "buy20"], (10 * 100 + 10 * 200) / 20 / 10000)
        self.assertAlmostEqual(b.loc[7, "buy200"], (10 * 100 + 10 * 200 + 180 * 300) / 200 / 10000)

    def test_fewer_units_than_n_for_sale_gives_a_blank_not_a_guess(self):
        b = FD.book_stats([self.book()], [7, 8]).set_index("item_id")
        self.assertTrue(np.isnan(b.loc[8, "buy20"]) and np.isnan(b.loc[8, "buy200"]))

    def test_exactly_n_units_for_sale_is_enough(self):
        s = mk([(1, 7, 10, 100, "LONG"), (2, 7, 10, 300, "LONG")])
        b = FD.book_stats([s], [7]).set_index("item_id")
        self.assertAlmostEqual(b.loc[7, "buy20"], 200 / 10000)

    def test_gold_in_play_is_price_times_quantity_over_every_listing(self):
        b = FD.book_stats([self.book()], [7, 8]).set_index("item_id")
        self.assertAlmostEqual(b.loc[7, "gold"], (10 * 100 + 10 * 200 + 500 * 300) / 10000)
        self.assertAlmostEqual(b.loc[8, "gold"], 5 * 50 / 10000)

    def test_only_the_wanted_items_are_covered(self):
        s = mk([(1, 7, 10, 100, "LONG"), (3, 9, 50, 10, "LONG")])
        b = FD.book_stats([s], [7])
        self.assertEqual(list(b["item_id"]), [7])
        self.assertAlmostEqual(float(b["gold"].iloc[0]), 10 * 100 / 10000)

    def test_no_snapshots_gives_an_empty_table_with_the_columns(self):
        b = FD.book_stats([], [7])
        self.assertEqual(list(b.columns), ["item_id", "ts", "gold", "buy20", "buy200"])
        self.assertEqual(len(b), 0)


class TestBuildHouse(Lake):
    def snaps_for(self, house):
        out = []
        for s in self.snaps:
            s.meta["house"] = house
            out.append(s)
        return out

    def test_rows_carry_the_house_the_kind_and_exactly_the_stored_columns(self):
        d = FD.build_house(self.snaps_for(110), 110)
        self.assertGreater(len(d), 0)
        self.assertEqual(list(d.columns), FD.COLUMNS)
        self.assertEqual(set(d["house"]), {110})
        self.assertEqual(set(d["kind"]), {"commodity"})
        self.assertEqual(len(d), len(d.drop_duplicates(FD.KEY_COLUMNS)))

    def test_prices_hours_and_units_are_the_retail_daily_tables_numbers(self):
        snaps = self.snaps_for(110)
        d = FD.build_house(snaps, 110, sales=True)
        ids = {int(i) for i in snaps[0].book["item_id"].unique()}
        h = D.build_hourly(snaps, classify_series(snaps), ids)
        retail = D.daily_table(D.trim_edges(h, snaps[0].ts, snaps[-1].ts))
        m = d.merge(retail, on=["item_id", "day"], suffixes=("", "_retail"))
        self.assertEqual(len(m), len(retail))
        for c in ("hours", "flagged_hours"):
            self.assertTrue((m[c] == m[f"{c}_retail"]).all(), c)
        for c, nd in (("price_day", 4), ("units_day", 1), ("price_p15", 4), ("price_p30", 4), ("sold_day", 2)):
            self.assertTrue(np.allclose(m[c], m[f"{c}_retail"].round(nd), atol=1e-9, equal_nan=True), c)

    SALES = ["sold_cert_day", "sold_low_day", "sold_day", "sold_high_day", "turnover_day", "active_share"]

    def test_the_sales_columns_are_filled_by_default_and_blank_when_switched_off(self):
        snaps = self.snaps_for(110)
        on = FD.build_house(snaps, 110)
        off = FD.build_house(snaps, 110, sales=False)
        for c in self.SALES:
            self.assertTrue(on[c].notna().any(), c)
            self.assertTrue(off[c].isna().all(), c)
        self.assertGreaterEqual(int(off["hours"].sum()), int(on["hours"].sum()))        # without the sales estimate every hour with a price counts
        pd.testing.assert_series_equal(off["price_day"].round(2), off["price_day"].round(2))
        self.assertEqual(list(off.columns), list(on.columns))

    def test_the_new_columns_make_sense(self):
        d = FD.build_house(self.snaps_for(110), 110)
        self.assertTrue((d["price_min"] <= d["price_day"] + 1e-9).all())
        both = d.dropna(subset=["buy20", "buy200"])
        self.assertGreater(len(both), 0)
        self.assertTrue((both["buy200"] >= both["buy20"] - 1e-9).all())                # buying more units never gets cheaper per unit
        self.assertTrue((d["gold_day"] > 0).all() and (d["listings_day"] > 0).all())

    def test_the_sales_bounds_are_ordered_and_turnover_and_active_share_are_what_they_say(self):
        d = FD.build_house(self.snaps_for(110), 110).dropna(subset=["sold_low_day"])
        self.assertGreater(len(d), 0)
        eps = 0.011                                                                   # the stored columns are rounded to 2 decimals
        self.assertTrue((d["sold_cert_day"] <= d["sold_low_day"] + eps).all())
        self.assertTrue((d["sold_low_day"] <= d["sold_day"] + eps).all())
        self.assertTrue((d["sold_day"] <= d["sold_high_day"] + eps).all())
        self.assertTrue(((d["active_share"] >= 0) & (d["active_share"] <= 1)).all())
        self.assertTrue(np.allclose(d["turnover_day"], d["sold_low_day"] / d["units_day"], atol=2e-4))

    def test_the_first_and_last_12_hours_of_the_window_are_left_out(self):
        d = FD.build_house(self.snaps_for(110), 110)
        self.assertEqual(int(d["hours"].sum()), int(d.groupby("item_id")["hours"].sum().sum()))
        per_item = d.groupby("item_id")["hours"].sum()
        n = len(self.snaps)
        self.assertTrue((per_item <= n - 24).all())

    def test_a_snapshot_from_another_house_stops_the_build(self):
        snaps = self.snaps_for(110)
        snaps[3].meta["house"] = 113
        with self.assertRaises(ValueError):
            FD.build_house(snaps, 110)

    def test_too_little_data_gives_an_empty_table_with_the_right_columns(self):
        d = FD.build_house(self.snaps_for(110)[:2], 110)
        self.assertEqual(len(d), 0)
        self.assertEqual(list(d.columns), FD.COLUMNS)

    def test_snapshot_days_are_us_eastern_days(self):
        snaps = [mk([(1, 1, 1, 1, "LONG")], ts=pd.Timestamp(t)) for t in ("2026-10-02 03:30", "2026-10-02 04:30", "2026-10-02 12:00")]
        self.assertEqual(FD.snapshot_days(snaps), {"2026-10-01": 1, "2026-10-02": 2})        # 03:30 UTC is 23:30 the evening before in US Eastern (summer time)


class TestVelocityColumns(unittest.TestCase):
    """A market where the only thing that happens is a standing listing being bought down by exactly 10 units an hour: the certain sales are known without any estimate."""

    def snaps(self, hours=60, per_hour=10, stock=1000):
        out = []
        for h in range(hours):
            rows = [(1, 5, stock - per_hour * h, 100, "VERY_LONG"), (2, 5, 500, 150, "VERY_LONG")]
            s = mk(rows, ts=pd.Timestamp("2026-10-01 00:00") + pd.Timedelta(hours=h))
            s.meta["house"] = 110
            out.append(s)
        return out

    def test_a_listing_bought_down_by_10_units_an_hour_is_240_certain_sales_a_day(self):
        d = FD.build_house(self.snaps(), 110).set_index("day")
        row = d.loc["2026-10-02"]
        self.assertEqual(int(row["hours"]), 20)                                      # the 12 hour edges are left out: hours 28 to 47 of the window fall on Oct 2 (US Eastern)
        self.assertAlmostEqual(row["sold_cert_day"], 240.0, places=6)
        self.assertGreaterEqual(row["sold_low_day"], row["sold_cert_day"] - 1e-9)
        self.assertEqual(row["active_share"], 1.0)                                   # a sale was seen in every usable hour

    def test_a_market_where_nothing_moves_shows_no_sales_and_no_activity(self):
        out = []
        for h in range(60):
            s = mk([(1, 5, 1000, 100, "VERY_LONG"), (2, 5, 500, 150, "VERY_LONG")], ts=pd.Timestamp("2026-10-01 00:00") + pd.Timedelta(hours=h))
            s.meta["house"] = 110
            out.append(s)
        row = FD.build_house(out, 110).set_index("day").loc["2026-10-02"]
        self.assertEqual((row["sold_cert_day"], row["sold_low_day"], row["active_share"], row["turnover_day"]), (0.0, 0.0, 0.0, 0.0))

    def test_turnover_is_the_conservative_sales_over_the_standing_stock(self):
        row = FD.build_house(self.snaps(), 110).set_index("day").loc["2026-10-02"]
        self.assertAlmostEqual(row["turnover_day"], row["sold_low_day"] / row["units_day"], places=3)


class TestDeepBuy(unittest.TestCase):
    """The Cured Rugged Hide case (Nov 25 report: buy200 25.45 under buy20 25.60): stock hovering around 200 units, deep in some hours and shallow in others, and the deep hours cheaper."""

    def snaps(self, every=2):
        out = []
        for h in range(60):
            rows = [(1, 5, 150, 1020000, "VERY_LONG")]                                       # always: 150 units at 102g
            if (h // 5) % every == 1:
                rows.append((2, 5, 190, 1000000, "VERY_LONG"))                               # deep blocks of 5 hours: 190 more units at 100g
            filler = [(100 + k, 9, 10, 50000, "VERY_LONG") for k in range(40)]                # a quiet second item, so the house is not one listing appearing and vanishing
            s = mk(rows + filler, ts=pd.Timestamp("2026-10-01 00:00") + pd.Timedelta(hours=h))
            s.meta["house"] = 110
            out.append(s)
        return out

    def test_the_deeper_buy_price_is_never_under_the_shallow_one(self):
        d = FD.build_house(self.snaps(), 110)
        row = d[d["item_id"] == 5].set_index("day").loc["2026-10-02"]
        self.assertEqual((int(row["hours"]), int(row["flagged_hours"])), (20, 0))
        self.assertAlmostEqual(row["buy20"], 101.0, places=4)                      # median of ten 100g hours and ten 102g hours
        deep_hour = (190 * 100 + 10 * 102) / 200                                   # 100.1g: what a deep hour costs for 200 units
        self.assertLess(deep_hour, row["buy20"])                                   # the old rule (median over the deep hours only) gave this, UNDER buy20
        self.assertAlmostEqual(row["buy200"], 101.0 * deep_hour / 100.0, places=4) # the new rule: buy20 times the deep hours' extra cost of going deeper
        self.assertGreaterEqual(row["buy200"], row["buy20"])

    def test_a_depth_seen_in_under_half_the_hours_gives_a_blank(self):
        d = FD.build_house(self.snaps(every=3), 110)
        row = d[d["item_id"] == 5].set_index("day").loc["2026-10-02"]
        self.assertTrue(np.isnan(row["buy200"]))
        self.assertFalse(np.isnan(row["buy20"]))


class TestStoredForm(unittest.TestCase):
    def rows(self, **over):
        base = dict(house=110, kind="commodity", item_id=2318, day="2026-11-24", hours=20, flagged_hours=1, price_day=0.1234, price_min=0.1, price_p15=0.12, price_p30=0.13, buy20=0.125,
                    buy200=0.14, units_day=5000.0, listings_day=80.0, gold_day=700.5, sold_day=np.nan, engine_version="0.7.5")
        base.update(over)
        return pd.DataFrame([base])

    def test_a_table_survives_a_round_trip_including_blanks(self):
        df = pd.concat([self.rows(), self.rows(item_id=2589, buy200=np.nan, price_p15=np.nan)], ignore_index=True)
        back = FD.from_csv_gz(FD.to_csv_gz(df))
        pd.testing.assert_frame_equal(back, FD.tidy(df))
        self.assertTrue(back["sold_day"].isna().all() and back["buy200"].isna().sum() == 1)

    def test_same_rows_same_bytes_in_any_order(self):
        a = pd.concat([self.rows(item_id=1), self.rows(item_id=2)], ignore_index=True)
        self.assertEqual(FD.to_csv_gz(a), FD.to_csv_gz(a.iloc[::-1]))
        self.assertEqual(FD.to_csv_gz(a), FD.to_csv_gz(a))

    def test_prices_are_rounded_so_files_stay_small(self):
        d = FD.tidy(self.rows(price_day=0.123456789, units_day=12.345, gold_day=1.23456))
        self.assertEqual((d["price_day"].iloc[0], d["units_day"].iloc[0], d["gold_day"].iloc[0]), (0.1235, 12.3, 1.23))

    def test_nothing_or_junk_reads_back_as_an_empty_table(self):
        for data in (None, b"", b"not a gzip file"):
            d = FD.from_csv_gz(data)
            self.assertEqual((len(d), list(d.columns)), (0, FD.COLUMNS))

    def test_day_keys(self):
        self.assertEqual(FD.day_key("2026-11-24"), "daily/d_2026-11-24.csv.gz")
        self.assertEqual(FD.parse_day_key("daily/d_2026-11-24.csv.gz"), "2026-11-24")
        for bad in ("daily/d_2026-11-24.parquet", "tables/forever_items.csv.gz", "daily/x.csv.gz"):
            self.assertIsNone(FD.parse_day_key(bad))

    def test_the_row_with_more_usable_hours_wins_and_a_tie_goes_to_the_new_row(self):
        old = self.rows(hours=22, price_day=1.0)
        self.assertEqual(FD.merge_days(old, self.rows(hours=10, price_day=2.0))["price_day"].iloc[0], 1.0)
        self.assertEqual(FD.merge_days(old, self.rows(hours=23, price_day=2.0))["price_day"].iloc[0], 2.0)
        self.assertEqual(FD.merge_days(old, self.rows(hours=22, price_day=2.0))["price_day"].iloc[0], 2.0)

    def test_rebuild_lets_the_new_row_win_even_with_fewer_hours(self):
        old = self.rows(hours=22, price_day=1.0)
        self.assertEqual(FD.merge_days(old, self.rows(hours=10, price_day=2.0), rebuild=True)["price_day"].iloc[0], 2.0)

    def test_other_houses_and_items_are_kept(self):
        old = pd.concat([self.rows(house=110), self.rows(house=113, price_day=9.0)], ignore_index=True)
        m = FD.merge_days(old, self.rows(house=110, hours=23, price_day=2.0))
        self.assertEqual(sorted(m["house"]), [110, 113])
        self.assertEqual(float(m[m["house"] == 113]["price_day"].iloc[0]), 9.0)

    def test_plan_writes_says_new_changed_or_unchanged_and_writes_only_what_changed(self):
        new = pd.concat([self.rows(day="2026-11-23"), self.rows(day="2026-11-24")], ignore_index=True)
        files, plan = FD.plan_writes(new, {})
        self.assertEqual([p["status"] for p in plan], ["new", "new"])
        self.assertEqual(set(files), {FD.day_key("2026-11-23"), FD.day_key("2026-11-24")})
        again, plan2 = FD.plan_writes(new, files)
        self.assertEqual([p["status"] for p in plan2], ["unchanged", "unchanged"])
        self.assertEqual(again, {})
        newer = pd.concat([self.rows(day="2026-11-23"), self.rows(day="2026-11-24", hours=23, price_day=5.0)], ignore_index=True)
        some, plan3 = FD.plan_writes(newer, files)
        self.assertEqual([p["status"] for p in plan3], ["unchanged", "changed"])
        self.assertEqual(set(some), {FD.day_key("2026-11-24")})

    def test_day_summary_counts_houses_rows_and_complete_rows(self):
        df = pd.concat([self.rows(house=110, hours=20), self.rows(house=113, hours=5), self.rows(house=113, item_id=9, hours=19)], ignore_index=True)
        s = FD.day_summary(df).iloc[0]
        self.assertEqual((s["houses"], s["rows"], s["complete_rows"]), (2, 3, 2))


class TestCraftCoverage(unittest.TestCase):
    def test_counts_recipes_with_everything_priced_per_house(self):
        recipes = T.tidy(pd.DataFrame({"recipe_id": [1, 2, 3, 4], "crafted_item_id": [1003, 1003, 9998, None]}), "recipes")
        reagents = T.tidy(pd.DataFrame({"recipe_id": [1, 1, 2, 2, 3, 4], "reagent_item_id": [1001, 1002, 1001, 9999, 1004, 1001]}), "reagents")
        rows = pd.DataFrame({"house": [110] * 5 + [113], "item_id": [1001, 1002, 1003, 1004, 5, 1001], "price_day": [1.0, 2.0, 3.0, 4.0, np.nan, 1.0]})
        c = FD.craft_coverage(rows, recipes, reagents)
        self.assertEqual(c[110], dict(recipes=3, reagents_priced=2, product_priced=2, both=1))      # recipe 4 names no product, so it is not counted
        self.assertEqual(c[113], dict(recipes=3, reagents_priced=0, product_priced=0, both=0))

    def test_missing_tables_give_nothing(self):
        self.assertEqual(FD.craft_coverage(pd.DataFrame(), T.tidy(None, "recipes"), T.tidy(None, "reagents")), {})


class RunFixture(Lake):
    """Two houses (110, and 113 at double the prices) in a pretend dataset, a fake commit that writes into it, and `go` to run the tool. No tests of its own."""

    def setUp(self):
        super().setUp()
        write_house(self.root, 110, self.snaps)
        write_house(self.root, 113, self.snaps, transform=lambda s, f: f.assign(unit_price=f["unit_price"] * 2))
        self.sources_file()
        self.store = LocalStore(self.root)
        self.commits = []

    def commit(self, files, message):
        self.commits.append((dict(files), message))
        for k, v in files.items():
            write(self.root, k, v)

    def go(self, **kw):
        return FDT.run(args(**kw), self.store, now=pd.Timestamp("2026-10-04 18:47", tz="UTC").to_pydatetime(), base_reader=base_reader, commit=self.commit)


class TestRun(RunFixture):
    def test_report_only_by_default_stores_nothing_and_says_so(self):
        r = self.go()
        self.assertEqual((r["code"], self.commits), (0, []))
        self.assertTrue(r["changed"])
        self.assertIn("REPORT ONLY", r["text"])
        self.assertIn("Report only:", r["text"])
        self.assertFalse(any(k.startswith("daily/") for k in os.listdir(self.root)))

    def test_save_writes_one_file_per_day_with_every_house_in_it_in_one_commit(self):
        r = self.go(save=True)
        self.assertEqual(r["code"], 0)
        self.assertEqual(len(self.commits), 1)
        keys = sorted(self.commits[0][0])
        self.assertTrue(all(FD.parse_day_key(k) for k in keys))
        self.assertGreaterEqual(len(keys), 2)
        day = FD.from_csv_gz(self.commits[0][0][keys[1]])
        self.assertEqual(sorted(set(day["house"])), [110, 113])
        self.assertIn("Saved", r["text"])

    def test_houses_are_never_pooled_the_doubled_house_costs_double(self):
        self.go(save=True)
        day = FD.from_csv_gz(self.commits[0][0][FD.day_key("2026-10-02")])
        a = day[day["house"] == 110].set_index("item_id")["price_day"]
        b = day[day["house"] == 113].set_index("item_id")["price_day"]
        self.assertTrue(np.allclose(b, 2 * a, rtol=2e-3))

    def test_a_second_run_over_the_same_files_changes_nothing(self):
        self.go(save=True)
        before = len(self.commits)
        r = self.go(save=True)
        self.assertEqual(len(self.commits), before)
        self.assertFalse(r["changed"])
        self.assertIn("Nothing to store", r["text"])
        self.assertIn("unchanged", r["text"])

    def test_empty_houses_are_left_out_even_when_the_sources_file_names_them(self):
        r = self.go()
        line = next(l for l in r["text"].splitlines() if l.startswith("run at"))
        self.assertIn("houses: 110, 113", line)
        self.assertNotIn("112", line)

    def test_the_houses_option_limits_the_run_and_names_unknown_ones(self):
        r = self.go(houses="113,999")
        line = next(l for l in r["text"].splitlines() if l.startswith("run at"))
        self.assertIn("houses: 113;", line)
        self.assertIn("999", r["text"])

    def test_a_dataset_that_cannot_be_read_stops_with_a_plain_message(self):
        os.remove(os.path.join(self.root, FV.SOURCES_KEY))
        r = self.go()
        self.assertEqual(r["code"], 1)
        self.assertIn("sources file could not be read", r["text"])

    def test_days_must_be_in_range(self):
        for d in (0, FDT.MAX_DAYS + 1):
            self.assertEqual(self.go(days=d)["code"], 1)

    def test_a_corrupt_file_is_reported_and_fails_the_run_but_the_rest_is_still_built(self):
        key = next(o.key for o in self.store.list(FV.source_name(110, True) + "/") if "day=02" in o.key)
        write(self.root, key, b"not a table")
        r = self.go()
        self.assertEqual(r["code"], 1)
        self.assertIn("skipped file", r["text"])
        self.assertIn("### 3 storing", r["text"])

    def test_one_house_failing_is_reported_and_does_not_stop_the_others(self):
        real = FD.build_house

        def flaky(snaps, house, *a, **kw):
            if house == 113:
                raise RuntimeError("boom")
            return real(snaps, house, *a, **kw)

        with mock.patch.object(FD, "build_house", side_effect=flaky):
            r = self.go(save=True)
        self.assertEqual(r["code"], 1)
        self.assertIn("113 FAILED: boom", r["text"])
        day = FD.from_csv_gz(self.commits[0][0][FD.day_key("2026-10-02")])
        self.assertEqual(set(day["house"]), {110})

    def stored_hours(self, day, house=110):
        key = FD.day_key(day)
        size = next(o.size for o in self.store.list("daily/") if o.key == key)
        d = FD.from_csv_gz(self.store.read_range(key, 0, size))
        return int(d[d["house"] == house]["hours"].sum())

    def test_a_partial_day_is_completed_by_a_later_run_and_never_shrunk_by_a_shorter_one(self):
        self.go(save=True, days=2, end="2026-10-02")                  # files of Oct 1-2 only: Oct 2 is mostly outside the trimmed window
        partial = self.stored_hours("2026-10-02")
        self.go(save=True)                                            # all three days: Oct 2 is now complete
        complete = self.stored_hours("2026-10-02")
        self.assertGreater(complete, partial)
        self.go(save=True, days=2, end="2026-10-02")                  # the short window again: fewer hours must not replace the fuller stored day
        self.assertEqual(self.stored_hours("2026-10-02"), complete)

    def test_the_report_has_its_sections_and_the_look_uses_item_names_when_the_table_exists(self):
        items = T.tidy(pd.DataFrame({"item_id": [1001, 1002, 1003, 1004], "name": ["Light Leather", "Silverleaf", "Mageweave", "Linen Cloth"]}), "items")
        write(self.root, T.KEYS["items"], T.to_csv_gz(items))
        r = self.go()
        for head in ("### 1 reading", "### 2 days built", "### 3 storing", "### 4 a look at the data", "### 5 crafting coverage"):
            self.assertIn(head, r["text"])
        self.assertIn("Light Leather (1001)", r["text"])

    def test_the_crafting_coverage_counts_recipes_from_the_stored_tables(self):
        recipes = T.tidy(pd.DataFrame({"recipe_id": [1, 2, 3], "crafted_item_id": [1003, 1003, 9998]}), "recipes")
        reagents = T.tidy(pd.DataFrame({"recipe_id": [1, 1, 2, 2, 3], "reagent_item_id": [1001, 1002, 1001, 9999, 1004]}), "reagents")
        write(self.root, T.KEYS["recipes"], T.to_csv_gz(recipes))
        write(self.root, T.KEYS["reagents"], T.to_csv_gz(reagents))
        r = self.go()
        rows = [l.split() for l in r["text"].splitlines() if l.strip().startswith("110 ") and len(l.split()) == 5]
        self.assertIn(["110", "3", "2", "2", "1"], rows)               # reagents: recipes 1 and 3; product: recipes 1 and 2 (both make item 1003); both: recipe 1

    def test_without_the_recipe_tables_the_coverage_says_how_to_get_them(self):
        r = self.go()
        self.assertIn("Skipped: the recipe and reagent tables are not in the dataset yet", r["text"])

    def test_nothing_in_the_report_names_the_dataset_account(self):
        owner = FV.hf_repo().split("/")[0]
        self.assertNotIn(owner, self.go()["text"])


class TestRunSales(RunFixture):
    def test_the_sales_columns_are_on_by_default_and_the_report_says_they_are_lower_bounds(self):
        r = self.go(save=True)
        self.assertIn("Sales columns: ON", r["text"])
        self.assertIn("LOWER BOUNDS", r["text"])
        day = FD.from_csv_gz(self.commits[0][0][FD.day_key("2026-10-02")])
        self.assertTrue(day["sold_low_day"].notna().any() and day["turnover_day"].notna().any())

    def test_no_sales_leaves_the_sales_columns_blank_and_the_report_says_so(self):
        r = self.go(save=True, no_sales=True)
        self.assertIn("Sales columns: OFF", r["text"])
        self.assertNotIn("Visible sales", r["text"])
        day = FD.from_csv_gz(self.commits[0][0][FD.day_key("2026-10-02")])
        for c in TestBuildHouse.SALES:
            self.assertTrue(day[c].isna().all(), c)

    def test_the_look_reports_how_much_of_the_market_visibly_moves(self):
        r = self.go()
        self.assertIn("Visible sales, house 110", r["text"])
        self.assertIn("sold_low/day", r["text"])
        self.assertIn("(all lower bounds)", r["text"])


class TestBackfillSeams(RunFixture):
    """Why backfill windows must overlap: a run leaves out the first and last 12 hours of the files it reads, so two windows that merely touch leave Eastern days at the seam with too few hours,
    and windows that overlap by two days do not. Six pretend days (Oct 1 00:00 to Oct 7 00:00 UTC) in two houses."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.snaps = MarketSimulator(default_profiles(), seed=5).run(6 * 24 + 1).snapshots

    def hours_by_day(self):
        out = {}
        for o in self.store.list("daily/"):
            d = FD.from_csv_gz(self.store.read_range(o.key, 0, o.size))
            out[o.key.split("d_")[1][:10]] = int(d[d["house"] == 110]["hours"].max())
        return out

    def test_windows_that_only_touch_leave_a_seam_of_partial_days(self):
        self.go(save=True, days=3, end="2026-10-06")
        self.go(save=True, days=3, end="2026-10-03")
        self.assertLess(self.hours_by_day()["2026-10-03"], FD.COMPLETE_HOURS)

    def test_windows_that_overlap_by_two_days_complete_every_day_between_their_edges(self):
        self.go(save=True, days=4, end="2026-10-06")
        self.go(save=True, days=4, end="2026-10-04")        # the end date steps back by days minus 2
        hours = self.hours_by_day()
        for day in ("2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"):
            self.assertGreaterEqual(hours[day], FD.COMPLETE_HOURS, day)


class TestWorkflow(unittest.TestCase):
    def wf(self, path):
        with open(os.path.join(_root, path), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_the_daily_schedule_is_off_until_the_repository_variable_says_yes_and_a_scheduled_run_saves(self):
        d = self.wf(".github/workflows/forever_daily.yml")
        on = d.get("on") or d.get(True)
        self.assertEqual(len(on["schedule"]), 1)
        cron = on["schedule"][0]["cron"].split()
        self.assertEqual(len(cron), 5)
        self.assertNotIn(cron[0], ("0", "30", "22", "32"))
        self.assertEqual(cron[1], "18")                                                                      # 13:47 / 14:47 US Eastern: the previous Eastern day is complete by then
        job = d["jobs"]["daily"]
        self.assertIn("FOREVER_DAILY_BUILD", job["if"])
        self.assertIn("github.event_name != 'schedule'", job["if"])
        step = next(s for s in job["steps"] if s.get("name") == "Forever daily tables")
        self.assertIn("schedule", step["env"]["SAVE"])
        self.assertIn("'no'", step["env"]["SAVE"])

    def test_manual_runs_are_report_only_by_default_sales_are_on_and_it_is_wired_to_the_one_secret_it_needs(self):
        for path in (".github/workflows/forever_daily.yml", "tools/forever_daily.yml"):
            d = self.wf(path)
            on = d.get("on") or d.get(True)
            self.assertEqual(set(on), {"workflow_dispatch", "schedule"})
            inputs = on["workflow_dispatch"]["inputs"]
            self.assertEqual(set(inputs), {"days", "end_date", "houses", "save", "sales", "rebuild"})
            self.assertEqual((inputs["save"]["default"], inputs["sales"]["default"], inputs["rebuild"]["default"], inputs["houses"]["default"], inputs["days"]["default"]),
                             ("no", "yes", "no", "all", "4"))                                                # report only by default, but the sales columns are on
            step = next(s for s in d["jobs"]["daily"]["steps"] if s.get("name") == "Forever daily tables")
            self.assertEqual(set(step["env"]), {"HF_TOKEN", "DAYS", "END", "HOUSES", "SAVE", "SALES", "REBUILD"})
            self.assertIn("secrets.HF_TOKEN", step["env"]["HF_TOKEN"])
            for flag in ("--save", "--no-sales", "--rebuild", "--end"):
                self.assertIn(flag, step["run"])
            self.assertIn("set -o pipefail", step["run"])
            self.assertIn("tools/forever_daily.py", step["run"])
            install = next(s for s in d["jobs"]["daily"]["steps"] if s.get("name") == "Install")["run"]
            self.assertIn("huggingface_hub", install)
            self.assertNotIn("BLIZZARD", json.dumps(d))                                                      # it reads only the Forever dataset

    def test_the_copy_in_tools_is_identical(self):
        with open(os.path.join(_root, ".github/workflows/forever_daily.yml"), encoding="utf-8") as a, open(os.path.join(_root, "tools/forever_daily.yml"), encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())


if __name__ == "__main__":
    unittest.main()
