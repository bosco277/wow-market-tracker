"""engine/forevertables.py and tools/forever_tables.py on pretend data (nothing leaves the machine): the recipe crawl and its parsing, the stored tables (round trip, same bytes for the same table, blank integers),
the incremental item table (named ids never asked again, unnamed ones retried late, no answer = not recorded, WoW Forever sources only), the house map, the whole tool on a local folder with a fake dataset
commit, and wiring checks on the workflow."""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import date
from unittest import mock

import pandas as pd
import yaml

from helpers import T0  # noqa: F401  (sets sys.path)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "tools"))

from engine import forever as FV  # noqa: E402
from engine import foreveritems as FI  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine import housemap as HM  # noqa: E402
from engine.loader import LocalStore  # noqa: E402
from test_forever_probe import base_reader, frame, item_json, key_for, world, write  # noqa: E402
import forever_tables as FT  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731
FORBIDDEN = ("namespace=static-us&", "namespace=static-classic-us&", "namespace=dynamic-us", "namespace=dynamic-classic")


class FastLimiter:
    def __init__(self, per_second):
        pass

    def wait(self):
        pass


PROFS = {
    197: dict(name="Tailoring", tiers={2948: dict(name="Tailoring", cats={"Bandages": [101, 102], "Bags": [103]})}),
    164: dict(name="Blacksmithing", tiers={2437: dict(name="Weaponsmithing", cats={"Weapons": [201]}), 2438: dict(name="Armorsmithing", cats={"Armor": [202, 101]})}),     # 101 is listed twice
    333: dict(name="Enchanting", tiers={2494: dict(name="Enchanting", cats={"Enchants": [301]})}),
}
RECIPES = {
    101: {"id": 101, "name": "Linen Bandage", "reagents": [{"reagent": {"id": 2000, "name": "Linen Cloth"}, "quantity": 1}], "crafted_item": {"id": 3000, "name": "Linen Bandage"}, "crafted_quantity": {"value": 1}},
    102: {"id": 102, "name": "Heavy Linen Bandage", "rank": 2, "reagents": [{"reagent": {"id": 2001, "name": "A"}, "quantity": 2}, {"reagent": {"id": 2002, "name": "B"}, "quantity": 1}],
          "crafted_item": {"id": 3001, "name": "Heavy"}, "crafted_quantity": {"minimum": 1, "maximum": 2}},
    103: {"id": 103, "name": "Small Bag", "reagents": [{"item": {"id": 2003, "name": "C"}, "quantity": 3}], "crafted_item": {"id": 3002, "name": "Bag"}},
    201: {"id": 201, "name": "Sword", "reagents": [{"reagent": {"id": 2010, "name": "Mystery"}, "quantity": 1}], "crafted_item": {"id": 3000, "name": "Linen Bandage"}, "crafted_quantity": 1},
    202: {"id": 202, "name": "Mail", "rank": {"id": 1}, "reagents": [{"reagent": {"id": 2005, "name": "E"}, "quantity": 4}, {"quantity": 1}], "crafted_item": {"id": 3001, "name": "Heavy"}},
    301: {"id": 301, "name": "Enchant Bracer", "reagents": [{"reagent": {"id": 2006, "name": "F"}, "quantity": 1}]},
}
FOREVER_ITEMS = {2000: "Linen Cloth", 2001: "A", 2002: "B", 2003: "C", 3000: "Linen Bandage", 3001: "Heavy", 3002: "Bag"}
ERA_ITEMS = {2005: "E", 2006: "F"}


class StaticApi:
    """A pretend Blizzard static-data API. `fail` = {exact path: [statuses to answer first, in order]}."""

    def __init__(self, professions=None, recipes=None, items=None, fail=None, version="1.60.1_70399"):
        self.version = version
        self.professions = PROFS if professions is None else professions
        self.recipes = RECIPES if recipes is None else recipes
        self.items = {"static-forever-us": FOREVER_ITEMS, "static-classic1x-us": ERA_ITEMS} if items is None else items
        self.fail = {k: list(v) for k, v in (fail or {}).items()}
        self.calls = []

    def paths(self, prefix):
        return [u.split(".com")[1].split("?")[0] for u in self.calls if u.split(".com")[1].startswith(prefix)]

    def __call__(self, url):
        self.calls.append(url)
        path = url.split(".com")[1].split("?")[0]
        ns = url.split("namespace=")[1].split("&")[0]
        seq = self.fail.get(path)
        if seq:
            return seq.pop(0), None
        if path == "/data/wow/profession/index":
            if ns != "static-forever-us":
                return 404, None
            return 200, {"professions": [{"id": p, "name": d["name"]} for p, d in self.professions.items()]}
        m = re.fullmatch(r"/data/wow/profession/(\d+)", path)
        if m and int(m.group(1)) in self.professions:
            d = self.professions[int(m.group(1))]
            return 200, {"id": int(m.group(1)), "name": d["name"], "type": {"type": "PRIMARY", "name": "Primary"}, "skill_tiers": [{"id": t, "name": v["name"]} for t, v in d["tiers"].items()]}
        m = re.fullmatch(r"/data/wow/profession/(\d+)/skill-tier/(\d+)", path)
        if m and int(m.group(1)) in self.professions and int(m.group(2)) in self.professions[int(m.group(1))]["tiers"]:
            t = self.professions[int(m.group(1))]["tiers"][int(m.group(2))]
            return 200, {"id": int(m.group(2)), "name": t["name"], "categories": [{"name": c, "recipes": [{"id": r, "name": self.recipes[r]["name"]} for r in ids]} for c, ids in t["cats"].items()]}
        m = re.fullmatch(r"/data/wow/recipe/(\d+)", path)
        if m:
            rid = int(m.group(1))
            if rid not in self.recipes:
                return 404, None
            return 200, {**self.recipes[rid], "_links": {"self": {"href": f"https://us.api.blizzard.com/data/wow/recipe/{rid}?namespace=static-{self.version}-forever-us"}}}
        m = re.fullmatch(r"/data/wow/item/(\d+)", path)
        if m:
            if ns not in self.items:
                return 403, None
            i = int(m.group(1))
            return (200, item_json(i, self.items[ns][i])) if i in self.items[ns] else (404, None)
        return 404, None


class Fast(unittest.TestCase):
    def setUp(self):
        for mod in (T, FI):
            p = mock.patch.object(mod, "RateLimiter", FastLimiter)
            p.start()
            self.addCleanup(p.stop)


class TestParsing(Fast):
    def test_tier_shapes(self):
        self.assertEqual(T.parse_tier({"categories": [{"name": "A", "recipes": [{"id": 1, "name": "x"}, {"name": "no id"}]}, {"name": "B", "recipes": [{"id": 2, "name": "y"}]}]}),
                         [("A", 1, "x"), ("B", 2, "y")])
        self.assertEqual(T.parse_tier({"recipes": [{"id": 3, "name": "z"}]}), [("", 3, "z")])
        self.assertEqual(T.parse_tier({}), [])
        self.assertEqual(T.parse_tier({"categories": ["junk", None]}), [])

    def test_crafted_quantity_forms(self):
        self.assertEqual(T.parse_quantity({"minimum": 1, "maximum": 3}), (1, 3))
        self.assertEqual(T.parse_quantity({"value": 5}), (5, 5))
        self.assertEqual(T.parse_quantity(2), (2, 2))
        self.assertEqual(T.parse_quantity({"minimum": 4}), (4, 4))
        for junk in (None, True, "x", {}, {"minimum": "a"}):
            self.assertEqual(T.parse_quantity(junk), (None, None))

    def test_recipe_reagent_and_product_variants(self):
        rec, reag, bad = T.parse_recipe(RECIPES[103])
        self.assertEqual((rec["crafted_item_id"], rec["n_reagents"], rec["crafted_qty_min"]), (3002, 1, None))
        self.assertEqual(reag, [dict(recipe_id=103, reagent_item_id=2003, reagent="C", quantity=3)])
        rec, reag, bad = T.parse_recipe(RECIPES[202])                              # one reagent line has no item at all
        self.assertEqual((rec["n_reagents"], bad), (1, 1))
        rec, reag, bad = T.parse_recipe(RECIPES[301])
        self.assertIsNone(rec["crafted_item_id"])
        self.assertEqual(rec["crafted_item"], "")
        rec, reag, bad = T.parse_recipe({"id": 9, "name": "Bare"})
        self.assertEqual((rec["n_reagents"], reag, bad), (0, [], 0))
        rec, reag, bad = T.parse_recipe({"id": 8, "reagents": [{"reagent": {"id": 5}, "quantity": "many"}]})
        self.assertIsNone(reag[0]["quantity"])
        with self.assertRaises(KeyError):
            T.parse_recipe({"name": "no id"})

    def test_rank_forms_and_the_version_in_a_response_link(self):
        self.assertEqual([T._rank(x) for x in (2, "3", {"id": 4}, {"value": 5}, None, True, "x", {})], [2, 3, 4, 5, None, None, None, None])
        link = {"_links": {"self": {"href": "https://us.api.blizzard.com/data/wow/recipe/1353?namespace=static-1.60.1_70399-forever-us"}}}
        self.assertEqual(T.static_version_from(link), "1.60.1_70399")
        self.assertIsNone(T.static_version_from({"_links": {"self": {"href": "https://x?namespace=static-forever-us"}}}))
        self.assertIsNone(T.static_version_from(None))
        api = StaticApi(version="1.61.0_71000")
        self.assertEqual(T.probe_version(api, 101, NOSLEEP), "1.61.0_71000")
        self.assertIsNone(T.probe_version(StaticApi(fail={"/data/wow/recipe/101": [404]}), 101, NOSLEEP))

    def test_placeholder_professions(self):
        for n in ("Test Profession [DNT]", "test profession", "Foo [DNT]"):
            self.assertTrue(T.is_test_profession(n), n)
        for n in ("Tailoring", "First Aid", ""):
            self.assertFalse(T.is_test_profession(n), n)

    def test_name_can_be_a_locale_dict(self):
        rec, _, _ = T.parse_recipe({"id": 7, "name": {"en_US": "Hello", "de_DE": "Hallo"}})
        self.assertEqual(rec["recipe"], "Hello")


class TestCrawl(Fast):
    def crawl(self, **kw):
        api = StaticApi(**kw)
        return T.crawl_recipes(api, sleep=NOSLEEP), api

    def test_every_profession_tier_and_recipe_is_read_and_a_recipe_listed_twice_is_kept_once(self):
        c, api = self.crawl()
        self.assertEqual(sorted(c.recipes["recipe_id"]), [101, 102, 103, 201, 202, 301])
        self.assertEqual(c.duplicates, 1)
        self.assertEqual(c.recipes.set_index("recipe_id").loc[101, "profession"], "Tailoring")             # the first listing wins
        self.assertEqual(c.recipes.set_index("recipe_id").loc[202, "skill_tier"], "Armorsmithing")
        self.assertEqual(c.recipes.set_index("recipe_id").loc[102, "category"], "Bandages")
        self.assertEqual(c.failed, [])
        self.assertEqual(dict(zip(c.professions["profession"], c.professions["skill_tiers"])), {"Tailoring": 1, "Blacksmithing": 2, "Enchanting": 1})
        self.assertEqual(c.bad_reagents, 1)
        self.assertEqual(len(c.reagents), 7)                                                              # 1 + 2 + 1 + 1 + 1 (the line without an item is skipped) + 1
        self.assertEqual(sorted(api.paths("/data/wow/recipe/")), [f"/data/wow/recipe/{i}" for i in (101, 102, 103, 201, 202, 301)])

    def test_integers_stay_integers_and_missing_ones_are_blank(self):
        c, _ = self.crawl()
        data = T.to_csv_gz(c.recipes)
        text = __import__("gzip").decompress(data).decode()
        self.assertNotIn(".0", text)
        self.assertIn("Enchant Bracer", text)
        back = T.from_csv_gz(data, "recipes")
        self.assertTrue(pd.isna(back.set_index("recipe_id").loc[301, "crafted_item_id"]))
        self.assertEqual(int(back.set_index("recipe_id").loc[102, "crafted_qty_max"]), 2)

    def test_samples_and_keys_describe_the_shape_for_the_report(self):
        c, _ = self.crawl()
        self.assertEqual(len(c.samples), 2)
        self.assertIn("reagents", c.keys)
        self.assertEqual(c.keys["id"], 6)
        self.assertEqual(c.keys["crafted_item"], 5)

    def test_only_forever_static_data_is_asked_about_recipes(self):
        c, api = self.crawl()
        self.assertTrue(all("namespace=static-forever-us&" in u for u in api.calls))

    def test_a_missing_profession_index_stops_the_crawl_with_its_status(self):
        with self.assertRaises(RuntimeError) as cm:
            T.crawl_recipes(StaticApi(fail={"/data/wow/profession/index": [404]}), sleep=NOSLEEP)
        self.assertIn("status 404", str(cm.exception))

    def test_trouble_is_retried_and_a_dead_recipe_is_listed_not_fatal(self):
        c, _ = self.crawl(fail={"/data/wow/recipe/101": [429, 503], "/data/wow/recipe/102": [503, 503, 503], "/data/wow/profession/333": [404]})
        self.assertIn(101, set(c.recipes["recipe_id"]))                                                   # recovered on the third try
        self.assertNotIn(102, set(c.recipes["recipe_id"]))
        self.assertNotIn(301, set(c.recipes["recipe_id"]))                                                # its profession could not be read
        self.assertEqual(sorted(c.failed), [("profession", 333, 404), ("recipe", 102, 503)])

    def test_a_failed_recipe_keeps_its_stored_rows(self):
        full, _ = self.crawl()
        part, _ = self.crawl(fail={"/data/wow/recipe/102": [503, 503, 503]})
        rec, reag, kept = T.keep_previous_for_failures(part, full.recipes, full.reagents)
        self.assertEqual(kept, 1)
        self.assertEqual(sorted(rec["recipe_id"]), [101, 102, 103, 201, 202, 301])
        self.assertEqual(int((reag["recipe_id"] == 102).sum()), 2)
        rec, reag, kept = T.keep_previous_for_failures(part, T.tidy(None, "recipes"), T.tidy(None, "reagents"))
        self.assertEqual(kept, 0)
        self.assertNotIn(102, set(rec["recipe_id"]))

    def test_a_placeholder_profession_is_left_out_and_never_asked_about(self):
        profs = {**PROFS, 2933: dict(name="Test Profession [DNT]", tiers={9: dict(name="Test", cats={"x": [901]})})}
        recipes = {**RECIPES, 901: {"id": 901, "name": "Test Recipe", "reagents": []}}
        api = StaticApi(professions=profs, recipes=recipes)
        c = T.crawl_recipes(api, sleep=NOSLEEP)
        self.assertEqual(c.excluded, ["Test Profession [DNT] (2933)"])
        self.assertNotIn(901, set(c.recipes["recipe_id"]))
        self.assertNotIn("/data/wow/profession/2933", api.paths("/data/wow/profession/"))
        self.assertNotIn("/data/wow/recipe/901", api.paths("/data/wow/recipe/"))
        self.assertEqual(len(c.professions), 3)

    def test_ranks_and_the_version_come_out_of_the_crawl(self):
        c, _ = self.crawl()
        r = c.recipes.set_index("recipe_id")
        self.assertEqual((r.loc[102, "rank"], r.loc[202, "rank"]), (2, 1))
        self.assertTrue(pd.isna(r.loc[101, "rank"]))
        self.assertEqual(c.version, "1.60.1_70399")
        self.assertEqual((c.fetched, c.reused), (6, 0))
        self.assertNotIn(".0", __import__("gzip").decompress(T.to_csv_gz(c.recipes)).decode())

    def test_a_second_crawl_reads_the_lists_but_fetches_only_new_recipes(self):
        first, _ = self.crawl()
        api = StaticApi()
        again = T.crawl_recipes(api, sleep=NOSLEEP, prev_recipes=first.recipes, prev_reagents=first.reagents, refetch=False)
        self.assertEqual((again.fetched, again.reused, api.paths("/data/wow/recipe/")), (0, 6, []))
        self.assertEqual(len(api.calls), 8)                                                               # index, 3 professions, 4 tiers
        pd.testing.assert_frame_equal(again.recipes, first.recipes)
        pd.testing.assert_frame_equal(again.reagents, first.reagents)
        recipes = {**RECIPES, 104: {"id": 104, "name": "New", "reagents": [{"reagent": {"id": 2000, "name": "x"}, "quantity": 5}], "crafted_item": {"id": 3002, "name": "x"}}}
        profs = {**PROFS, 197: dict(PROFS[197], tiers={2948: dict(name="Tailoring", cats={"Bandages": [101, 102, 104], "Bags": [103]})})}
        api = StaticApi(professions=profs, recipes=recipes)
        more = T.crawl_recipes(api, sleep=NOSLEEP, prev_recipes=first.recipes, prev_reagents=first.reagents, refetch=False)
        self.assertEqual((more.fetched, more.reused, api.paths("/data/wow/recipe/")), (1, 6, ["/data/wow/recipe/104"]))
        self.assertEqual(int(more.reagents[more.reagents["recipe_id"] == 104]["quantity"].iloc[0]), 5)

    def test_a_recipe_that_moved_takes_its_new_tier_and_one_no_longer_listed_drops_out(self):
        first, _ = self.crawl()
        profs = {197: dict(name="Tailoring", tiers={2948: dict(name="Tailoring", cats={"Moved": [101, 102]})})}                # 103 and everything else is gone; 101 and 102 changed category
        api = StaticApi(professions=profs)
        again = T.crawl_recipes(api, sleep=NOSLEEP, prev_recipes=first.recipes, prev_reagents=first.reagents, refetch=False)
        self.assertEqual(sorted(again.recipes["recipe_id"]), [101, 102])
        self.assertEqual(set(again.recipes["category"]), {"Moved"})
        self.assertEqual(sorted(set(again.reagents["recipe_id"])), [101, 102])

    def test_refetch_reads_every_recipe_again(self):
        first, _ = self.crawl()
        api = StaticApi()
        again = T.crawl_recipes(api, sleep=NOSLEEP, prev_recipes=first.recipes, prev_reagents=first.reagents, refetch=True)
        self.assertEqual((again.fetched, again.reused, len(api.paths("/data/wow/recipe/"))), (6, 0, 6))

    def test_items_a_recipe_mentions_and_the_summary_table(self):
        c, _ = self.crawl()
        self.assertEqual(T.recipe_item_ids(c.recipes, c.reagents), {2000, 2001, 2002, 2003, 2005, 2006, 2010, 3000, 3001, 3002})
        s = T.recipe_summary(c.recipes, c.reagents)
        self.assertRegex(s, r"Tailoring\s+197\s+3\s+3\s+3")
        self.assertRegex(s, r"Enchanting\s+333\s+1\s+1\s+0")
        self.assertRegex(s, r"all\s+6\s+6\s+5")
        self.assertEqual(T.recipe_summary(T.tidy(None, "recipes"), T.tidy(None, "reagents")), "No recipes.")


class TestStoredTables(unittest.TestCase):
    def table(self):
        return T.tidy(pd.DataFrame([
            dict(item_id=2, name="Beta, the \"second\"", item_class_id=7, item_class="Trade Goods", quality="COMMON", item_level=5, is_stackable=True, max_count=20, sell_price_copper=100, source="forever", checked_at="2026-11-25"),
            dict(item_id=1, name="", source="none", checked_at="2026-11-25", is_stackable=False)]), "items")

    def test_same_table_same_bytes_and_a_round_trip_loses_nothing(self):
        a, b = T.to_csv_gz(self.table()), T.to_csv_gz(self.table())
        self.assertEqual(a, b)
        back = T.from_csv_gz(a, "items")
        pd.testing.assert_frame_equal(back, self.table())
        self.assertEqual(list(back["item_id"]), [1, 2])                                                   # sorted by id
        self.assertEqual(back.loc[1, "name"], 'Beta, the "second"')
        self.assertEqual(back.loc[0, "name"], "")
        self.assertTrue(pd.isna(back.loc[0, "item_level"]))
        self.assertEqual(back["is_stackable"].tolist(), [False, True])

    def test_empty_and_unreadable_inputs_give_an_empty_table_with_the_right_columns(self):
        for data in (None, b"", b"not gzip", T.to_csv_gz(T.tidy(None, "items"))):
            t = T.from_csv_gz(data, "items")
            self.assertEqual(list(t.columns), T.ITEM_COLUMNS)
            self.assertEqual(len(t), 0)

    def test_names_that_look_like_nothing_special_survive(self):
        t = T.tidy(pd.DataFrame([dict(item_id=5, name="NA", source="forever", checked_at="x")]), "items")
        self.assertEqual(T.from_csv_gz(T.to_csv_gz(t), "items").loc[0, "name"], "NA")


class TestUpdateItems(Fast):
    TODAY = date(2026, 11, 25)

    def run_(self, ids, previous=None, api=None, **kw):
        api = api or StaticApi()
        items, st = T.update_items(ids, previous, api, self.TODAY, sleep=NOSLEEP, **kw)
        return items, st, api

    def test_first_build_names_from_forever_then_classic_era_and_records_the_rest_as_unnamed(self):
        items, st, api = self.run_([2000, 2005, 2004, 3000])
        row = items.set_index("item_id")
        self.assertEqual(row.loc[2000, "source"], "forever")
        self.assertEqual(row.loc[2005, "source"], "classic_era")
        self.assertEqual((row.loc[2004, "source"], row.loc[2004, "name"], row.loc[2004, "checked_at"]), ("none", "", "2026-11-25"))
        self.assertEqual(row.loc[2000, "name"], "Linen Cloth")
        self.assertEqual(row.loc[2000, "item_class"], "Trade Goods")
        self.assertEqual((st["asked"], st["named_by"], st["unnamed"], st["no_answer"]), (4, {"forever": 2, "classic_era": 1}, 1, 0))

    def test_retail_and_plain_classic_are_never_asked(self):
        _, _, api = self.run_([2000, 2005, 2004, 999999])
        self.assertFalse(any(any(f in u for f in FORBIDDEN) for u in api.calls), [u for u in api.calls if "static-us" in u or "static-classic-us" in u])
        self.assertEqual({u.split("namespace=")[1].split("&")[0] for u in api.calls}, {"static-forever-us", "static-classic1x-us"})

    def test_named_ids_are_never_asked_again_and_unnamed_ones_wait_for_the_retry_period(self):
        first, _, _ = self.run_([2000, 2005, 2004])
        again, st, api = self.run_([2000, 2005, 2004], first)
        self.assertEqual(api.calls, [])
        self.assertEqual((st["asked"], st["already_named"], st["kept_unnamed"]), (0, 2, 1))
        pd.testing.assert_frame_equal(again, first)
        old = first.copy()
        old.loc[old["item_id"] == 2004, "checked_at"] = "2026-11-21"                                      # four days ago: due
        _, st, api = self.run_([2000, 2005, 2004], old, retry_days=3)
        self.assertEqual(st["asked"], 1)
        self.assertEqual(api.paths("/data/wow/item/"), ["/data/wow/item/2004"] * 2)                       # asked of both sources
        old.loc[old["item_id"] == 2004, "checked_at"] = "2026-11-23"                                      # two days ago: not yet
        _, st, api = self.run_([2004], old, retry_days=3)
        self.assertEqual(st["asked"], 0)

    def test_an_unnamed_id_that_gets_named_later_is_replaced_not_duplicated(self):
        first, _, _ = self.run_([2004])
        first.loc[:, "checked_at"] = "2026-11-01"
        api = StaticApi(items={"static-forever-us": {**FOREVER_ITEMS, 2004: "Later"}, "static-classic1x-us": ERA_ITEMS})
        again, st, _ = self.run_([2004], first, api=api)
        self.assertEqual(len(again), 1)
        self.assertEqual((again.iloc[0]["name"], again.iloc[0]["source"]), ("Later", "forever"))

    def test_a_lookup_with_no_real_answer_is_not_recorded_so_the_next_run_asks_again(self):
        api = StaticApi(fail={"/data/wow/item/2004": [503] * 3})                                           # Forever never answers for 2004; Classic Era says 404
        items, st, _ = self.run_([2004, 2000], api=api)
        self.assertEqual(st["no_answer"], 1)
        self.assertEqual(list(items["item_id"]), [2000])
        _, st2, _ = self.run_([2004, 2000], items)
        self.assertEqual(st2["asked"], 1)                                                                  # asked again (and now nobody names it, so it is recorded)
        self.assertEqual(st2["unnamed"], 1)

    def test_refresh_all_asks_every_wanted_id_again_and_a_changed_name_replaces_the_old_one(self):
        first, _, _ = self.run_([2000, 2005, 2004])
        api = StaticApi(items={"static-forever-us": {**FOREVER_ITEMS, 2000: "Renamed", 2005: "E again"}, "static-classic1x-us": ERA_ITEMS})
        again, st, _ = self.run_([2000, 2005, 2004], first, api=api, refresh_all=True)
        self.assertEqual((st["asked"], st["already_named"], st["kept_unnamed"]), (3, 0, 0))
        row = again.set_index("item_id")
        self.assertEqual((row.loc[2000, "name"], row.loc[2005, "name"], row.loc[2005, "source"]), ("Renamed", "E again", "forever"))
        self.assertEqual(row.loc[2004, "source"], "none")
        self.assertEqual(len(again), 3)

    def test_refresh_all_keeps_an_old_row_when_the_lookup_gets_no_real_answer(self):
        first, _, _ = self.run_([2000, 2005])
        api = StaticApi(fail={"/data/wow/item/2000": [503] * 6, "/data/wow/item/2005": [503] * 3})              # Forever never answers for 2000 and 2005 this time
        again, st, _ = self.run_([2000, 2005], first, api=api, refresh_all=True)
        row = again.set_index("item_id")
        self.assertEqual((row.loc[2000, "name"], row.loc[2000, "source"]), ("Linen Cloth", "forever"))
        self.assertEqual((st["no_answer"], st["named_by"]), (1, {"classic_era": 1}))                           # 2000 got no answer from either source; 2005 was named by Classic Era again
        self.assertEqual(len(again), 2)

    def test_ids_no_longer_wanted_keep_their_rows(self):
        first, _, _ = self.run_([2000, 2005])
        again, _, _ = self.run_([3000], first)
        self.assertEqual(sorted(again["item_id"]), [2000, 2005, 3000])

    def test_nothing_wanted_nothing_asked(self):
        items, st, api = self.run_([])
        self.assertEqual((len(items), st["asked"], api.calls), (0, 0, []))
        self.assertEqual(list(items.columns), T.ITEM_COLUMNS)


class TestRecheck(Fast):
    TODAY = date(2026, 12, 2)

    def table(self):
        t, _ = T.update_items([2000, 2005, 2006, 2004], None, StaticApi(), date(2026, 11, 25), sleep=NOSLEEP)
        return t

    def test_classic_era_rows_that_forever_now_names_switch_over_and_nothing_else_changes(self):
        first = self.table()
        api = StaticApi(items={"static-forever-us": {**FOREVER_ITEMS, 2005: "E in Forever"}, "static-classic1x-us": ERA_ITEMS})
        out, st = T.recheck_forever(first, api, self.TODAY, sleep=NOSLEEP)
        self.assertEqual(st, {"checked": 2, "upgraded": 1})
        row = out.set_index("item_id")
        self.assertEqual((row.loc[2005, "source"], row.loc[2005, "name"], row.loc[2005, "checked_at"]), ("forever", "E in Forever", "2026-12-02"))
        self.assertEqual((row.loc[2006, "source"], row.loc[2006, "checked_at"]), ("classic_era", "2026-11-25"))        # still only Classic Era knows it: untouched
        self.assertEqual(row.loc[2004, "source"], "none")
        self.assertEqual(row.loc[2000, "source"], "forever")
        self.assertEqual(len(out), len(first))
        self.assertEqual({u.split("namespace=")[1].split("&")[0] for u in api.calls}, {"static-forever-us"})          # Forever only

    def test_a_subset_of_ids_limits_the_check_and_no_classic_rows_means_no_calls(self):
        first = self.table()
        api = StaticApi()
        out, st = T.recheck_forever(first, api, self.TODAY, ids=[2006, 2000], sleep=NOSLEEP)
        self.assertEqual((st["checked"], api.paths("/data/wow/item/")), (1, ["/data/wow/item/2006"]))
        api = StaticApi()
        out, st = T.recheck_forever(first[first["source"] != "classic_era"], api, self.TODAY, sleep=NOSLEEP)
        self.assertEqual((st, api.calls), ({"checked": 0, "upgraded": 0}, []))

    def test_forever_gives_no_answer_so_the_rows_stay_as_they_were(self):
        first = self.table()
        api = StaticApi(fail={"/data/wow/item/2005": [503] * 3, "/data/wow/item/2006": [503] * 3})
        out, st = T.recheck_forever(first, api, self.TODAY, sleep=NOSLEEP)
        self.assertEqual(st["upgraded"], 0)
        pd.testing.assert_frame_equal(out, first)


class TestMeta(unittest.TestCase):
    def test_round_trip_and_junk(self):
        m = {"schema": 2, "static_version": "1.60.1_70399", "forever_recheck": "2026-11-25"}
        self.assertEqual(T.parse_meta(T.meta_bytes(m)), m)
        self.assertEqual(T.meta_bytes(m), T.meta_bytes(dict(reversed(list(m.items())))))                       # key order does not change the bytes
        for junk in (None, b"", b"not json", b"[1, 2]", b"\xff\xfe"):
            self.assertEqual(T.parse_meta(junk), {})


class TestHouseMap(unittest.TestCase):
    def test_covers_every_house_the_api_lists_and_only_those(self):
        m = T.house_map()
        self.assertEqual(sorted(map(int, m["houses"])), [h for h in range(110, 126) if h != 122])

    def test_factions_agree_with_the_fingerprint_module_and_neutral_rows_are_marked_assumed(self):
        m = T.house_map()["houses"]
        for fac, hs in HM.FACTION_HOUSES.items():
            for h in hs:
                self.assertEqual(m[str(h)]["faction"], fac, h)
        for h in (116, 117, 121, 125):
            self.assertEqual((m[str(h)]["confidence"], m[str(h)]["faction"]), ("assumed", "neutral"))
        for h in (112, 115, 118):
            self.assertEqual((m[str(h)]["confidence"], m[str(h)]["ruleset"]), ("empty", None))

    def test_rulesets_by_slot_and_the_confirmed_house(self):
        m = T.house_map()["houses"]
        self.assertEqual([(m[str(h)]["ruleset"], m[str(h)]["ruleset_id"]) for h in (110, 113, 116)], [("pvp", 1)] * 3)
        self.assertEqual([m[str(h)]["ruleset"] for h in (111, 114, 117)], ["pve"] * 3)
        self.assertEqual([m[str(h)]["ruleset"] for h in (119, 120, 121)], ["rp"] * 3)
        self.assertEqual([(m[str(h)]["ruleset"], m[str(h)]["ruleset_id"]) for h in (123, 124, 125)], [("rp pvp", 15)] * 3)
        self.assertEqual(m["110"]["confidence"], "confirmed")
        self.assertEqual(m["111"]["confidence"], "likely")
        self.assertIn("not read from the API", m["119"]["id_basis"])
        self.assertIn("API", m["113"]["id_basis"])

    def test_the_stored_bytes_do_not_depend_on_the_day(self):
        self.assertEqual(T.house_map_bytes(), T.house_map_bytes())
        self.assertEqual(json.loads(T.house_map_bytes())["as_of"], "2026-11-25")


def args(**kw):
    d = dict(parts="all", save=False, retry_days=3, max_ids=30000, recheck_days=7, full=False, out="x")
    d.update(kw)
    return type("A", (), d)()


class TestWholeTool(Fast):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "lake")
        self.houses = [110, 111]
        w = world(self.houses, n_items=10)                                                                 # item ids 2000 to 2009 on the houses
        names = {}
        for h in self.houses:
            for kind in (True, False):
                src = FV.source_name(h, kind)
                names[src] = {"house_id": h}
                for s in w[h]:
                    write(self.root, key_for(src, s.ts), frame(s, src, buyout=not kind))
        write(self.root, FV.SOURCES_KEY, json.dumps(names).encode())
        self.now = pd.Timestamp("2026-11-25 12:10").to_pydatetime()
        self.store = LocalStore(self.root)
        self.commits = []

    def commit(self, files, message):
        self.commits.append((dict(files), message))
        for k, v in files.items():                                                                         # what a real commit would do: the files are now in the dataset
            write(self.root, k, v)

    def go(self, api=None, store=None, **kw):
        api = api or StaticApi()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = FT.run(args(**kw), api, store or self.store, now=self.now, base_reader=base_reader, sleep=NOSLEEP, commit=self.commit)
        r["api"] = api
        return r

    def test_report_only_by_default_nothing_is_committed(self):
        r = self.go()
        t = r["text"]
        self.assertEqual((r["code"], self.commits), (0, []))
        self.assertIn("report only (nothing is saved)", t)
        self.assertIn("Nothing stored yet: this is the first build.", t)
        self.assertIn("3 professions, 6 recipes listed (6 read from Blizzard, 0 reused from the stored table; all read because: first build), 7 reagent lines.", t)
        self.assertIn("10 item ids on the auction houses (newest snapshot of every house that has data).", t)
        self.assertIn("10 item ids used or made by recipes (4 of them not on the auction houses)", t)         # 3000 to 3002 and 2010
        self.assertIn("Named by: forever 7, classic_era 2; nobody names 5.", t)
        self.assertIn("Now 9 of the 14 ids are named (64.3%)", t)
        self.assertIn("Unnamed: 5 ids", t)
        self.assertIn("in recipes: 1 (e.g. 2010)", t)                                                      # an unnamed id that a recipe uses is called out
        self.assertIn("Recipes a crafting tool can price today (reagents listed, product named, every item named): 4 of 6.", t)
        self.assertIn("Linen Bandage: 1 x Linen Cloth  ->  1 x Linen Bandage", t)
        self.assertIn("NOT saved (report only)", t)
        self.assertRegex(t, r"tables/forever_items\.csv\.gz: .*, new")
        self.assertRegex(t, r"tables/forever_house_map\.json: .*, new")
        self.assertIn("15 houses:", t)

    def test_save_writes_the_four_files_in_one_commit_and_they_read_back(self):
        r = self.go(save=True)
        self.assertEqual(r["code"], 0)
        self.assertEqual(len(self.commits), 1)
        files, msg = self.commits[0]
        self.assertEqual(sorted(files), sorted(T.KEYS.values()))
        self.assertIn("Saved 5 file(s)", r["text"])
        self.assertNotIn(FV.hf_repo().split("/")[0], msg)
        rec = T.from_csv_gz(files[T.KEYS["recipes"]], "recipes")
        rea = T.from_csv_gz(files[T.KEYS["reagents"]], "reagents")
        itm = T.from_csv_gz(files[T.KEYS["items"]], "items")
        self.assertEqual((len(rec), len(rea), len(itm)), (6, 7, 14))
        self.assertEqual(int((itm["source"] == "none").sum()), 5)
        self.assertEqual(json.loads(files[T.KEYS["house_map"]])["houses"]["110"]["confidence"], "confirmed")
        self.assertEqual(json.loads(files[T.KEYS["meta"]]), {"forever_recheck": "2026-11-25", "schema": T.SCHEMA, "static_version": "1.60.1_70399"})
        self.assertEqual(rec.set_index("recipe_id").loc[102, "rank"], 2)
        self.assertEqual(rec.set_index("recipe_id").loc[202, "rank"], 1)
        self.assertTrue(pd.isna(rec.set_index("recipe_id").loc[101, "rank"]))

    def test_a_second_run_finds_everything_unchanged_and_asks_nothing_about_items(self):
        self.go(save=True)
        self.commits.clear()
        r = self.go(save=True)
        self.assertEqual(self.commits, [])
        self.assertIn("Every table is identical to what is stored: nothing to save.", r["text"])
        self.assertIn("Asked 0 id(s) (9 already named, 5 unnamed and not due for another try).", r["text"])
        self.assertEqual(r["api"].paths("/data/wow/item/"), [])
        self.assertIn("Against the stored table: 0 new recipe(s), 0 no longer listed.", r["text"])
        self.assertIn("(0 read from Blizzard, 6 reused from the stored table)", r["text"])
        self.assertEqual(len(r["api"].calls), 9)                                                           # one version probe, the index, three professions, four tiers: no recipe, no item
        self.assertEqual(r["api"].paths("/data/wow/recipe/"), ["/data/wow/recipe/201"])                     # the probe only
        self.assertIn("Blizzard calls this run: 9.", r["text"])
        self.assertIn("Static data version: 1.60.1_70399 (stored: 1.60.1_70399).", r["text"])
        self.assertIn("Forever re-check of Classic-Era-named rows: not due (last done 2026-11-25; every 7 days).", r["text"])

    def stored_meta(self):
        with open(os.path.join(self.root, T.KEYS["meta"]), "rb") as f:
            return T.parse_meta(f.read())

    def set_meta(self, **kw):
        write(self.root, T.KEYS["meta"], T.meta_bytes({**self.stored_meta(), **kw}))

    def test_a_new_static_data_version_reads_everything_again_and_records_the_new_version(self):
        self.go(save=True)
        self.commits.clear()
        api = StaticApi(version="1.61.0_71000", items={"static-forever-us": {**FOREVER_ITEMS, 2004: "Newly named"}, "static-classic1x-us": ERA_ITEMS})
        r = self.go(api=api, save=True)
        t = r["text"]
        self.assertIn("Static data version: 1.61.0_71000 (stored: 1.60.1_70399) CHANGED: every recipe and item is read again.", t)
        self.assertIn("(6 read from Blizzard, 0 reused from the stored table; all read because: data version changed)", t)
        self.assertIn("Asked 14 id(s) (0 already named, 0 unnamed and not due for another try): all of them, data version changed.", t)
        self.assertEqual(len(api.paths("/data/wow/recipe/")), 7)                                              # six recipes and the version probe
        self.assertEqual(self.stored_meta()["static_version"], "1.61.0_71000")
        self.assertEqual(self.stored_meta()["forever_recheck"], "2026-11-25")
        with open(os.path.join(self.root, T.KEYS["items"]), "rb") as f:
            itm = T.from_csv_gz(f.read(), "items").set_index("item_id")
        self.assertEqual(itm.loc[2004, "name"], "Newly named")
        self.assertNotIn("Forever re-check of", t)                                                            # a full read already asked Forever first

    def test_tables_from_the_first_layout_read_the_recipes_in_full_once_and_do_a_forever_recheck(self):
        self.go(save=True)
        os.remove(os.path.join(self.root, T.KEYS["meta"]))                                                    # what v0.7.45 stored: no meta file
        self.commits.clear()
        api = StaticApi(items={"static-forever-us": {**FOREVER_ITEMS, 2005: "E in Forever"}, "static-classic1x-us": ERA_ITEMS})
        r = self.go(api=api, save=True)
        t = r["text"]
        self.assertIn("(6 read from Blizzard, 0 reused from the stored table; all read because: the stored tables are from an older layout)", t)
        self.assertIn("Forever re-check of 2 Classic-Era-named row(s) (last done: never): 1 are now named by Forever and were switched over.", t)
        self.assertEqual(self.stored_meta(), {"forever_recheck": "2026-11-25", "schema": T.SCHEMA, "static_version": "1.60.1_70399"})
        self.assertIn("Forever re-check of Classic-Era-named rows: not due", self.go(save=True)["text"])        # and now it is quiet again

    def test_the_forever_recheck_comes_due_after_the_chosen_number_of_days(self):
        self.go(save=True)
        self.set_meta(forever_recheck="2026-11-20")                                                           # five days ago
        r = self.go(save=True)
        self.assertIn("not due (last done 2026-11-20; every 7 days)", r["text"])
        r = self.go(save=True, recheck_days=5)
        self.assertIn("Forever re-check of 2 Classic-Era-named row(s) (last done: 2026-11-20)", r["text"])
        self.assertEqual(self.stored_meta()["forever_recheck"], "2026-11-25")

    def test_full_reads_everything_again_without_a_version_change(self):
        self.go(save=True)
        api = StaticApi()
        r = self.go(api=api, full=True)
        self.assertIn("FULL refresh asked for", r["text"])
        self.assertIn("all read because: full refresh asked for", r["text"])
        self.assertEqual(len(api.paths("/data/wow/recipe/")), 7)                                              # six recipes and the version probe
        self.assertIn("Asked 14 id(s)", r["text"])
        self.assertIn("all of them, full refresh asked for", r["text"])

    def test_a_blocked_recipe_save_does_not_move_the_stored_version_or_layout(self):
        self.go(save=True)
        os.remove(os.path.join(self.root, T.KEYS["meta"]))
        fail = {f"/data/wow/recipe/{i}": [503] * 9 for i in (101, 102, 103, 201)}
        self.go(api=StaticApi(fail=fail), save=True)
        meta = self.stored_meta() if os.path.exists(os.path.join(self.root, T.KEYS["meta"])) else {}
        self.assertNotIn("schema", meta)                                                                       # the recipes were not saved, so the layout is still the old one

    def test_a_changed_recipe_list_commits_only_what_changed(self):
        self.go(save=True)
        self.commits.clear()
        recipes = {**RECIPES, 104: {"id": 104, "name": "New", "reagents": [{"reagent": {"id": 2000, "name": "x"}, "quantity": 1}], "crafted_item": {"id": 3002, "name": "x"}}}
        profs = {**PROFS, 197: dict(PROFS[197], tiers={2948: dict(name="Tailoring", cats={"Bandages": [101, 102, 104], "Bags": [103]})})}
        r = self.go(api=StaticApi(professions=profs, recipes=recipes), save=True)
        files, _ = self.commits[0]
        self.assertEqual(sorted(files), sorted([T.KEYS["recipes"], T.KEYS["reagents"]]))
        self.assertIn("1 new recipe(s), 0 no longer listed.", r["text"])
        self.assertIn("(1 read from Blizzard, 6 reused from the stored table)", r["text"])

    def test_too_many_failed_recipe_lookups_block_saving_recipes_but_not_the_rest(self):
        fail = {f"/data/wow/recipe/{i}": [503] * 3 for i in (101, 102, 103, 201)}
        r = self.go(api=StaticApi(fail=fail), save=True)
        self.assertIn("recipes: NOT saved", r["text"])
        self.assertIn("4 lookup(s) failed", r["text"])
        files, _ = self.commits[0]
        self.assertNotIn(T.KEYS["recipes"], files)
        self.assertIn(T.KEYS["items"], files)

    def test_far_fewer_recipes_than_stored_is_not_overwritten(self):
        self.go(save=True)
        self.commits.clear()
        r = self.go(api=StaticApi(professions={197: PROFS[197]}), save=True)
        self.assertIn("far fewer than the 6 stored", r["text"])
        self.assertTrue(all(T.KEYS["recipes"] not in f for f, _ in self.commits))

    def test_only_forever_static_data_and_classic_era_are_ever_asked(self):
        r = self.go(save=True)
        urls = r["api"].calls
        self.assertTrue(urls)
        self.assertFalse([u for u in urls if any(f in u for f in FORBIDDEN)])
        self.assertTrue(all(u.split("namespace=")[1].split("&")[0] in ("static-forever-us", "static-classic1x-us") for u in urls))
        self.assertTrue(all("/data/wow/item/" in u for u in urls if "classic1x" in u))                      # Classic Era is asked about items only, never about professions

    def test_a_failed_commit_is_reported_and_exits_nonzero_without_naming_the_dataset(self):
        def boom(files, message):
            raise RuntimeError(f"403 Forbidden: {FV.hf_repo()} write access denied")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = FT.run(args(save=True), StaticApi(), self.store, now=self.now, base_reader=base_reader, sleep=NOSLEEP, commit=boom)
        self.assertEqual(r["code"], 1)
        self.assertIn("SAVE FAILED: 403 Forbidden: <forever dataset> write access denied", r["text"])
        self.assertNotIn(FV.hf_repo().split("/")[0], r["text"])

    def test_parts_run_alone(self):
        r = self.go(parts="housemap", save=True)
        self.assertEqual(list(self.commits[0][0]), [T.KEYS["house_map"]])
        self.assertNotIn("### 1 recipes", r["text"])
        self.assertEqual(r["api"].calls, [])
        r = self.go(parts="recipes")
        self.assertNotIn("### 2 items", r["text"])
        self.assertEqual(r["api"].paths("/data/wow/item/"), [])

    def test_items_alone_use_the_stored_recipes_for_their_ids(self):
        self.go(parts="recipes", save=True)
        r = self.go(parts="items")
        self.assertIn("10 item ids used or made by recipes", r["text"])

    def test_the_lake_being_unreadable_still_builds_from_recipes(self):
        os.remove(os.path.join(self.root, FV.SOURCES_KEY))
        r = self.go()
        self.assertIn("The lake could not be read", r["text"])
        self.assertIn("10 item ids used or made by recipes", r["text"])
        self.assertEqual(r["code"], 0)

    def test_a_dead_profession_index_is_a_nonzero_exit_but_the_report_still_comes(self):
        r = self.go(api=StaticApi(fail={"/data/wow/profession/index": [404]}))
        self.assertEqual(r["code"], 1)
        self.assertIn("recipes: failed (profession index not served (status 404))", r["text"])

    def test_too_many_ids_stops_the_items_part(self):
        r = self.go(max_ids=5)
        self.assertEqual(r["code"], 1)
        self.assertIn("is more than --max-ids 5", r["text"])

    def test_unknown_part_and_no_account_name_in_the_report(self):
        self.assertEqual(self.go(parts="recipes,prices")["code"], 1)
        t = self.go()["text"]
        self.assertNotIn(FV.hf_repo().split("/")[0], t)
        self.assertFalse(any(line.startswith("=") for line in t.splitlines()))


class TestMain(unittest.TestCase):
    def test_missing_secrets_are_a_clear_stop(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(out):
            for k in ("BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET", "HF_TOKEN"):
                os.environ.pop(k, None)
            self.assertEqual(FT.main(["--out", tempfile.gettempdir()]), 1)
        self.assertIn("BLIZZARD_CLIENT_ID", out.getvalue())

    def test_an_error_carrying_the_dataset_address_is_blanked_before_it_is_printed(self):
        class Boom:
            def __init__(self, *a, **k):
                raise RuntimeError(f"404 https://huggingface.co/api/datasets/{FV.hf_repo()}/tree/main")
        out = io.StringIO()
        env = {"HF_TOKEN": "x", "BLIZZARD_CLIENT_ID": "i", "BLIZZARD_CLIENT_SECRET": "s"}
        with mock.patch.dict(os.environ, env), mock.patch.object(FT.FP, "Blizzard", Boom), contextlib.redirect_stdout(out):
            self.assertEqual(FT.main(["--out", tempfile.gettempdir()]), 1)
        self.assertNotIn(FV.hf_repo().split("/")[0], out.getvalue())

    def test_the_commit_retries_transient_trouble_and_stops_at_once_on_a_refusal(self):
        class Resp:
            def __init__(self, c):
                self.status_code = c

        class Err(Exception):
            def __init__(self, c):
                self.response = Resp(c)

        calls = []

        class Api:
            seq = [503, 429, None]

            def __init__(self, token=None):
                pass

            def create_commit(self, **kw):
                calls.append(sorted(op.path_in_repo for op in kw["operations"]))
                s = self.seq.pop(0)
                if s:
                    raise Err(s)
        hub = type("Hub", (), {"HfApi": Api, "CommitOperationAdd": lambda path_in_repo, path_or_fileobj: type("Op", (), {"path_in_repo": path_in_repo})()})
        sleeps = []
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            FT.hf_commit({"a": b"1", "b": b"2"}, "m", sleep=sleeps.append)
            self.assertEqual((len(calls), sleeps), (3, [15, 30]))
            Api.seq = [403]
            calls.clear()
            with self.assertRaises(Err):
                FT.hf_commit({"a": b"1"}, "m", sleep=sleeps.append)
            self.assertEqual(len(calls), 1)


class TestWorkflow(unittest.TestCase):
    def wf(self, path):
        with open(os.path.join(_root, path), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_the_daily_schedule_is_off_until_the_repository_variable_says_yes_and_a_scheduled_run_saves(self):
        d = self.wf(".github/workflows/forever_tables.yml")
        on = d.get("on") or d.get(True)
        self.assertEqual(len(on["schedule"]), 1)
        cron = on["schedule"][0]["cron"].split()
        self.assertEqual(len(cron), 5)
        self.assertNotIn(cron[0], ("0", "30", "22", "32"))                                                  # not on the hour, half hour, or at the minutes Blizzard and our fetcher use
        job = d["jobs"]["tables"]
        self.assertIn("FOREVER_TABLES_DAILY", job["if"])
        self.assertIn("github.event_name != 'schedule'", job["if"])                                         # manual runs never depend on the variable
        step = next(s for s in job["steps"] if s.get("name") == "Forever tables")
        self.assertIn("schedule", step["env"]["SAVE"])
        self.assertIn("'yes'", step["env"]["SAVE"])
        self.assertIn("'no'", step["env"]["SAVE"])                                                           # a manual run with nothing chosen saves nothing
        self.assertIn("--full", step["run"])

    def test_manual_only_wired_to_the_right_secrets_and_report_only_by_default(self):
        for path in (".github/workflows/forever_tables.yml", "tools/forever_tables.yml"):
            d = self.wf(path)
            on = d.get("on") or d.get(True)
            self.assertEqual(set(on), {"workflow_dispatch", "schedule"})
            inputs = on["workflow_dispatch"]["inputs"]
            self.assertEqual(set(inputs), {"parts", "save", "retry_days", "full"})
            self.assertEqual((inputs["save"]["default"], inputs["parts"]["default"]), ("no", "all"))
            step = next(s for s in d["jobs"]["tables"]["steps"] if s.get("name") == "Forever tables")
            self.assertEqual(set(step["env"]), {"HF_TOKEN", "BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET", "PARTS", "SAVE", "RETRY_DAYS", "FULL"})
            for k in ("HF_TOKEN", "BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET"):
                self.assertIn(f"secrets.{k}", step["env"][k])
            self.assertIn('"$SAVE" = "yes"', step["run"])
            self.assertIn("--save", step["run"])
            self.assertIn("set -o pipefail", step["run"])
            self.assertIn("tools/forever_tables.py", step["run"])
            install = next(s for s in d["jobs"]["tables"]["steps"] if s.get("name") == "Install")["run"]
            self.assertIn("huggingface_hub", install)

    def test_the_copy_in_tools_is_identical(self):
        with open(os.path.join(_root, ".github/workflows/forever_tables.yml"), encoding="utf-8") as a, open(os.path.join(_root, "tools/forever_tables.yml"), encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())


if __name__ == "__main__":
    unittest.main()
