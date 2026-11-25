"""engine/foreveritems.py, engine/housemap.py and tools/forever_probe.py on pretend data (nothing leaves the machine): the item lookup order and its coverage report, the house fingerprint
(a synthetic world where the right house is known), the structure section's reading of the `ruleset` field, the whole tool end to end on a local folder, and wiring checks on the workflow and fixture."""
import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from helpers import T0  # noqa: F401  (sets sys.path)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "tools"))

from engine import forever as FV  # noqa: E402
from engine import foreveritems as FI  # noqa: E402
from engine import housemap as HM  # noqa: E402
from engine.loader import LocalStore  # noqa: E402
from engine.schema import Snapshot  # noqa: E402
import forever_probe as P  # noqa: E402


def base_reader(fileobj, item_ids):
    return pickle.loads(fileobj.read())


def write(root, key, obj):
    path = os.path.join(root, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        if isinstance(obj, (bytes, str)):
            fh.write(obj if isinstance(obj, bytes) else obj.encode())
        else:
            pickle.dump(obj, fh)


def item_json(i, name=None, cls="Trade Goods"):
    return {"id": i, "name": name or f"Item {i}", "item_class": {"id": 7, "name": cls}, "item_subclass": {"id": 9, "name": "Herb"}, "quality": {"type": "COMMON"}, "level": 5,
            "sell_price": 100, "purchase_price": 400, "is_stackable": True, "max_count": 20}


class Api:
    """A pretend Blizzard: answers by (namespace, kind, id). `items` = {namespace: {id: name}}; ids not listed answer 404; a namespace not listed answers 403."""

    def __init__(self, items=None, extra=None, flaky=None):
        self.items, self.extra, self.calls, self.flaky = items or {}, extra or {}, [], flaky or {}

    def __call__(self, url):
        self.calls.append(url)
        for frag, ans in self.extra.items():
            if frag in url:
                return ans(url) if callable(ans) else ans
        ns = url.split("namespace=")[1].split("&")[0]
        if "/data/wow/item/" in url:
            i = int(url.split("/data/wow/item/")[1].split("?")[0])
            if ns not in self.items:
                return 403, None
            if (ns, i) in self.flaky and self.flaky[(ns, i)] > 0:
                self.flaky[(ns, i)] -= 1
                return 429, None
            return (200, item_json(i, self.items[ns][i])) if i in self.items[ns] else (404, None)
        return 404, None


NOSLEEP = lambda s: None  # noqa: E731


class TestItemLookup(unittest.TestCase):
    def test_first_source_that_names_the_item_wins_and_later_ones_are_not_asked(self):
        api = Api({"static-forever-us": {1: "Forever One"}, "static-classic1x-us": {1: "Era One", 2: "Era Two"}, "static-us": {3: "Retail Three"}})
        row, tried = FI.resolve_one(1, api, sleep=NOSLEEP)
        self.assertEqual((row["name"], row["source"], tried), ("Forever One", "forever", {"forever": 200}))
        row, tried = FI.resolve_one(2, api, sleep=NOSLEEP)
        self.assertEqual((row["name"], row["source"], tried), ("Era Two", "classic_era", {"forever": 404, "classic_era": 200}))
        row, tried = FI.resolve_one(3, api, sleep=NOSLEEP)
        self.assertEqual((row["source"], tried), ("retail", {"forever": 404, "classic_era": 404, "classic": 403, "retail": 200}))

    def test_an_item_nobody_names_is_none_with_every_status(self):
        row, tried = FI.resolve_one(9, Api({"static-forever-us": {}, "static-us": {}}), sleep=NOSLEEP)
        self.assertIsNone(row)
        self.assertEqual(tried, {"forever": 404, "classic_era": 403, "classic": 403, "retail": 404})

    def test_trouble_is_retried_and_a_dead_source_falls_through_to_the_next(self):
        api = Api({"static-forever-us": {5: "Five"}}, flaky={("static-forever-us", 5): 2})
        row, _ = FI.resolve_one(5, api, tries=3, sleep=NOSLEEP)
        self.assertEqual(row["source"], "forever")
        dead = Api({"static-classic1x-us": {5: "Era Five"}}, extra={"static-forever-us": lambda u: (_ for _ in ()).throw(OSError("down"))})
        row, tried = FI.resolve_one(5, dead, tries=2, sleep=NOSLEEP)
        self.assertEqual((row["source"], tried["forever"]), ("classic_era", 0))

    def test_resolve_items_returns_named_rows_and_an_attempt_table(self):
        api = Api({"static-forever-us": {1: "A", 2: "B"}, "static-classic1x-us": {3: "C"}})
        items, att = FI.resolve_items([3, 1, 2, 4, 1], api, workers=3, per_second=1000, sleep=NOSLEEP)
        self.assertEqual(sorted(items["item_id"]), [1, 2, 3])
        self.assertEqual(dict(zip(items["item_id"], items["source"])), {1: "forever", 2: "forever", 3: "classic_era"})
        att = att.set_index("item_id")
        self.assertEqual(list(att.index), [1, 2, 3, 4])
        self.assertEqual(att.loc[3, "forever"], 404)
        self.assertTrue(pd.isna(att.loc[1, "classic_era"]))                    # not asked: Forever already named it
        self.assertEqual(list(att.loc[4].dropna()), [404, 404, 403, 403])

    def test_coverage_text_counts_sources_and_ranks_the_unnamed_by_value(self):
        api = Api({"static-forever-us": {i: f"N{i}" for i in (1, 2, 3, 4)}, "static-classic1x-us": {5: "N5", 6: "N6"}})
        ids = list(range(1, 11))
        items, att = FI.resolve_items(ids, api, per_second=1000, sleep=NOSLEEP)
        value = pd.Series({1: 100.0, 7: 5000.0, 8: 300.0, 9: 20.0, 10: 10.0, 5: 50.0})
        t = FI.coverage_text(ids, items, att, value, top=2)
        self.assertIn("10 distinct item ids", t)
        self.assertIn("6 named (60.0%); 4 not named", t)
        lines = t.splitlines()
        self.assertTrue(any(l.startswith("  forever") and "named      4" in l for l in lines))
        self.assertTrue(any(l.startswith("  classic_era") and "named      2" in l for l in lines))
        self.assertIn("Unnamed ids hold", t)
        self.assertLess(t.index("      7 "), t.index("      8 "))                # 5,000g before 300g
        self.assertNotIn("      9 ", t)                                           # top=2 keeps only two

    def test_coverage_text_with_nothing_missing_or_nothing_asked(self):
        self.assertEqual(FI.coverage_text([], pd.DataFrame(), pd.DataFrame()), "No item ids to check.")
        api = Api({"static-forever-us": {1: "A"}})
        items, att = FI.resolve_items([1], api, sleep=NOSLEEP)
        self.assertNotIn("Biggest unnamed", FI.coverage_text([1], items, att))


class TestAveragePrice(unittest.TestCase):
    def test_average_price_walks_up_the_book(self):
        lad = HM.make_ladder(np.array([200, 100]), np.array([10, 5]))          # unsorted on purpose
        self.assertAlmostEqual(HM.avg_price(lad, 5), 100.0)
        self.assertAlmostEqual(HM.avg_price(lad, 10), 150.0)
        self.assertAlmostEqual(HM.avg_price(lad, 15), 2500 / 15)
        self.assertAlmostEqual(HM.avg_price(lad, 1), 100.0)
        self.assertIsNone(HM.avg_price(lad, 16))
        self.assertIsNone(HM.avg_price(lad, 0))
        self.assertIsNone(HM.avg_price(HM.make_ladder(np.array([]), np.array([])), 1))

    def test_name_map_prefers_the_id_with_the_most_units(self):
        items = pd.DataFrame({"item_id": [10, 11, 12], "name": ["Herb", "Herb", "Ore"]})
        self.assertEqual(HM.name_map(items, pd.Series({10: 5.0, 11: 90.0})), {"Herb": 11, "Ore": 12})
        self.assertEqual(HM.name_map(items), {"Herb": 10, "Ore": 12})
        self.assertEqual(HM.name_map(pd.DataFrame()), {})

    def test_page_names(self):
        self.assertEqual(HM.slug_parts("uspvp--alliance"), ("us", "pvp", "alliance"))
        self.assertEqual(HM.slug_parts("usrppvp--horde"), ("us", "rp pvp", "horde"))
        self.assertEqual(HM.slug_label("uspve--horde"), "US PvE (Horde)")
        self.assertEqual(HM.slug_label("usrppvp--alliance"), "US RP PvP (Alliance)")


def world(houses, n_items=10, hours=6, spread=0.5, seed=3, same=()):
    """{house: [commodity Snapshots]}: every house has its own price level per item (independent noise of `spread` in log terms; houses in `same` copy the first one's), three listings per item
    (quantity 100 each, each 3% dearer than the last), prices that drift a little every hour."""
    rng = np.random.default_rng(seed)
    base = 1000.0 * (1 + np.arange(n_items))
    level = {h: np.exp(rng.normal(0, spread, n_items)) for h in houses}
    drifts = {h: np.exp(rng.normal(0, 0.02, (hours, n_items))) for h in houses}
    for h in same:
        level[h], drifts[h] = level[houses[0]], drifts[houses[0]]
    out = {}
    for h in houses:
        snaps = []
        for k in range(hours):
            drift = drifts[h][k]
            rows, aid = [], 1
            for i in range(n_items):
                p0 = base[i] * level[h][i] * drift[i]
                for j in range(3):
                    rows.append((h * 1_000_000 + aid, 2000 + i, f"{2000 + i}", 100, int(round(p0 * (1.03 ** j))), "LONG"))
                    aid += 1
            book = pd.DataFrame(rows, columns=["auction_id", "item_id", "variant_key", "quantity", "unit_price", "time_left"])
            snaps.append(Snapshot(FV.source_name(h, True), pd.Timestamp("2026-11-25 06:25") + pd.Timedelta(hours=k), book, kind="commodity", meta={"house": h}))
        out[h] = snaps
    return out


def page_from(snaps, hour, qty=150, items=range(10), source="ah"):
    """What a site page would show if it scanned `snaps` at snapshot index `hour`: the average price of buying `qty` units of each item."""
    s = snaps[hour]
    rows = []
    for i in items:
        b = s.book[s.book["item_id"] == 2000 + i]
        lad = HM.make_ladder(b["unit_price"].to_numpy(), b["quantity"].to_numpy())
        rows.append(dict(item=f"Item {i}", qty=qty, each_copper=int(round(HM.avg_price(lad, qty))), source=source))
    return {"scan_age": "3 h ago", "price_ladders": 1800, "rows": rows}


NAMES = {f"Item {i}": 2000 + i for i in range(10)}


class TestFingerprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.houses = [110, 111, 113, 114]
        cls.w = world(cls.houses)

    def fixture(self, truth):
        return {"houses": {slug: page_from(self.w[h], hour) for slug, (h, hour) in truth.items()}}

    def test_each_page_lines_up_with_its_own_house_at_the_right_hour(self):
        truth = {"uspvp--alliance": (110, 3), "uspve--alliance": (111, 1), "uspvp--horde": (113, 4), "uspve--horde": (114, 2)}
        long = HM.fingerprint(self.w, self.fixture(truth), NAMES)
        wide = HM.wide_scores(long)
        for slug, (h, hour) in truth.items():
            r = HM.ranked(wide, slug)
            self.assertEqual(r[0][0], h, slug)
            self.assertLess(r[0][1], 0.02)
            self.assertGreater(r[1][1], 5 * max(r[0][1], 0.01))
            row = long[(long["slug"] == slug) & (long["house"] == h)].iloc[0]
            self.assertEqual(row["ts"], self.w[h][hour].ts)
            self.assertEqual(int(row["items"]), 10)

    def test_verdicts_pass_the_method_check_and_call_clear_matches_clear(self):
        truth = {"uspvp--alliance": (110, 3), "uspve--alliance": (111, 1), "uspvp--horde": (113, 4), "uspve--horde": (114, 2)}
        wide = HM.wide_scores(HM.fingerprint(self.w, self.fixture(truth), NAMES))
        groups = {"alliance": (110, 111), "horde": (113, 114), "neutral": ()}
        lines = HM.verdicts(wide, known={"uspvp--alliance": 110}, groups=groups)
        self.assertTrue(lines[0].startswith("[OK] method check"))
        self.assertEqual(sum(1 for l in lines if l.startswith("[OK] US ")), 4)
        self.assertTrue(any(l.startswith("Alliance one-to-one assignment") and "PVP = 110" in l and "PVE = 111" in l for l in lines))
        self.assertTrue(any(l.startswith("Horde one-to-one assignment") and "PVP = 113" in l and "PVE = 114" in l for l in lines))
        self.assertFalse(any("differs from each page" in l for l in lines))

    def test_the_method_check_fails_loudly_when_the_known_house_is_not_the_best(self):
        truth = {"uspvp--alliance": (111, 3), "uspve--alliance": (110, 1)}               # the 'PvP Alliance' page really is house 111's
        wide = HM.wide_scores(HM.fingerprint(self.w, self.fixture(truth), NAMES))
        lines = HM.verdicts(wide, known={"uspvp--alliance": 110}, groups={"alliance": (110, 111)})
        self.assertTrue(lines[0].startswith("[PROBLEM] method check FAILED"))
        self.assertIn("lines up best with house 111", lines[0])

    def test_two_houses_with_the_same_prices_are_called_unclear(self):
        w = world([110, 111, 113], same=(111,))
        fx = {"houses": {"uspvp--alliance": page_from(w[110], 2)}}
        wide = HM.wide_scores(HM.fingerprint(w, fx, NAMES))
        line = [l for l in HM.verdicts(wide, known={}, groups={"alliance": (110, 111)}) if l.startswith("[")][0]
        self.assertTrue(line.startswith("[WATCH]"))
        self.assertIn("UNCLEAR", line)

    def test_a_page_whose_best_house_is_on_the_other_factions_list_is_flagged(self):
        fx = {"houses": {"uspvp--alliance": page_from(self.w[113], 2)}}
        wide = HM.wide_scores(HM.fingerprint(self.w, fx, NAMES))
        line = [l for l in HM.verdicts(wide, known={}, groups={"alliance": (110, 111), "horde": (113, 114)}) if l.startswith("[")][0]
        self.assertIn("house 113 is not on the alliance list", line)

    def test_vendor_rows_and_unknown_names_are_ignored_and_too_few_items_gives_no_score(self):
        page = page_from(self.w[110], 2)
        page["rows"][0]["source"] = "vendor"
        page["rows"].append(dict(item="Unheard Of", qty=10, each_copper=5, source="ah"))
        self.assertEqual(len(HM.page_rows(page, NAMES)), 9)
        few = {"houses": {"uspvp--alliance": page_from(self.w[110], 2, items=range(4))}}
        long = HM.fingerprint(self.w, few, NAMES)
        self.assertTrue(long["score"].isna().all())

    def test_a_quantity_the_book_cannot_fill_is_skipped_not_guessed(self):
        page = page_from(self.w[110], 2)
        for r in page["rows"][:3]:
            r["qty"] = 100_000
        lad = HM.snapshot_ladders(self.w[110][2], NAMES.values())
        sc, n = HM.score_page(lad, HM.page_rows(page, NAMES))
        self.assertEqual(n, 7)
        self.assertFalse(np.isnan(sc))
        for r in page["rows"][3:6]:
            r["qty"] = 100_000
        sc, n = HM.score_page(lad, HM.page_rows(page, NAMES))
        self.assertEqual(n, 4)
        self.assertTrue(np.isnan(sc))

    def test_best_assignment_tries_every_pairing_and_survives_missing_scores(self):
        wide = pd.DataFrame({110: [0.1, 0.5, np.nan], 111: [0.4, 0.1, 0.9], 112: [0.9, 0.9, 0.2]}, index=["a", "b", "c"])
        asg, tot = HM.best_assignment(wide, ["a", "b", "c"], [110, 111, 112])
        self.assertEqual(asg, {"a": 110, "b": 111, "c": 112})
        self.assertAlmostEqual(tot, 0.4)
        asg, tot = HM.best_assignment(wide, ["a", "b", "c"], [110, 111])
        self.assertEqual((asg, tot), ({}, tot))
        self.assertTrue(np.isnan(tot))

    def test_matrix_text_marks_the_best_house(self):
        wide = pd.DataFrame({110: [0.10], 111: [0.40], 112: [np.nan]}, index=["uspvp--alliance"])
        t = HM.matrix_text(wide)
        self.assertIn("0.10*", t)
        self.assertIn("0.40 ", t)
        self.assertIn("-", t.splitlines()[1])


class TestStructure(unittest.TestCase):
    HOUSES = {110: "Alliance Auction House", 111: "Alliance Auction House", 113: "Horde Auction House", 116: "Blackwater Auction House"}
    OWN = {110: 1, 111: 2, 113: 1, 116: 1}

    def api(self, follows_url, show_auctions=True):
        def house_record(url):
            rs = int(url.split("/ruleset/")[1].split("/")[0])
            h = int(url.split("/auctions/")[1].split("?")[0])
            ids = [{"id": h * 1000 + k, "item": {"id": 1}, "quantity": 1, "buyout": 5} for k in range(3)]
            return 200, {"id": h, "name": self.HOUSES[h], "ruleset": {"key": {"href": f"https://us.api.blizzard.com/data/wow/ruleset/{rs if follows_url else self.OWN[h]}?namespace=x"}},
                         "auctions": ids if show_auctions else []}

        def lst(url):
            return 200, {"auctions": [{"id": h, "name": n} for h, n in self.HOUSES.items()]}
        extra = {"/auctions/?": lst, "/data/wow/region/index": (200, {"regions": [{"id": 91}]}), "/data/wow/ruleset/index": (200, {"rulesets": [{"id": 1}, {"id": 2}]}),
                 "/data/wow/ruleset/1?": (200, {"id": 1, "name": "PvP"}), "/data/wow/realm/index": (404, None), "/data/wow/search/realm": (200, {"results": []})}
        for rs in (1, 2, 3, 15):
            for h in self.HOUSES:
                extra[f"/ruleset/{rs}/auctions/{h}?"] = house_record
        return Api(extra=extra)

    def test_a_ruleset_field_that_follows_the_url_is_reported_as_not_a_property_of_the_house(self):
        t = P.structure_section(self.api(True), sleep=NOSLEEP)
        self.assertIn("/data/wow/realm/index", t)
        self.assertRegex(t, r"/data/wow/realm/index\s+404")
        self.assertIn("The house list is the SAME under every ruleset.", t)
        self.assertIn("FOLLOWS THE URL for houses [110, 111, 113, 116]", t)
        self.assertIn("house 110 Alliance Auction House: via 1: field=1 n=3", t)

    def test_a_ruleset_field_that_does_not_follow_the_url_states_each_houses_own_ruleset(self):
        t = P.structure_section(self.api(False), sleep=NOSLEEP)
        self.assertNotIn("FOLLOWS THE URL", t)
        self.assertIn("house 110 = ruleset 1", t)
        self.assertIn("house 111 = ruleset 2", t)
        self.assertIn("(same data under every URL)", t)

    def test_no_auction_contents_reach_the_report(self):
        t = P.structure_section(self.api(True), sleep=NOSLEEP)
        self.assertNotIn("buyout", t)
        self.assertNotIn("110000", t)

    def test_no_served_house_list_is_said_plainly(self):
        t = P.structure_section(Api(), sleep=NOSLEEP)
        self.assertIn("No ruleset served a list of auction houses.", t)

    def test_by_default_each_house_is_read_through_two_rulesets_and_all_four_on_request(self):
        for flag, want in ((False, {1, 2}), (True, {1, 2, 3, 15})):
            api = self.api(True)
            r = P.run(args(sections="structure", all_rulesets=flag), api, None, sleep=NOSLEEP)
            self.assertEqual(r["code"], 0)
            used = {int(u.split("/ruleset/")[1].split("/")[0]) for u in api.calls if "/auctions/110?" in u}
            self.assertEqual(used, want)

    def test_ruleset_field_shapes(self):
        self.assertEqual(P.rs_id(2), 2)
        self.assertEqual(P.rs_id("15"), 15)
        self.assertEqual(P.rs_id({"id": 3}), 3)
        self.assertEqual(P.rs_id({"key": {"href": "https://x/data/wow/ruleset/15?namespace=y"}}), 15)
        for junk in (None, "abc", {"name": "x"}, True, [1]):
            self.assertIsNone(P.rs_id(junk))

    def test_the_id_fingerprint_ignores_order(self):
        a = [{"id": 3}, {"id": 1}, {"id": 2}]
        b = [{"id": 2}, {"id": 3}, {"id": 1}]
        self.assertEqual(P.ids_digest(a), P.ids_digest(b))
        self.assertEqual(P.ids_digest(a)[1], 3)
        self.assertNotEqual(P.ids_digest(a)[0], P.ids_digest(a[:2])[0])


class TestRecipeProbe(unittest.TestCase):
    def test_a_namespace_with_recipes_shows_each_step(self):
        api = Api(extra={"/profession/index": (200, {"professions": [{"id": 171}]}), "/profession/171/skill-tier/2": (200, {"categories": [{"recipes": [{"id": 5}, {"id": 6}]}]}),
                         "/profession/171?": (200, {"skill_tiers": [{"id": 2}]}), "/recipe/5": (200, {"reagents": [{}, {}]})})
        t = P.recipe_probe(api, "forever", "static-forever-us", NOSLEEP)
        self.assertIn("profession index 200 (1 professions)", t)
        self.assertIn("skill tier 2 200 (2 recipes)", t)
        self.assertIn("sample recipe 200 (2 reagents listed)", t)

    def test_a_namespace_without_them_stops_at_the_index(self):
        t = P.recipe_probe(Api(), "retail", "static-us", NOSLEEP)
        self.assertEqual(t.split("; "), ["  retail       profession index 404"])


# ------------------------------------------------------------------------------------------------ the whole tool on a local folder
def args(**kw):
    d = dict(sections="all", hours=40, all_rulesets=False, out="x")
    d.update(kw)
    return type("A", (), d)()


def frame(snap, source, buyout=False):
    b = snap.book
    f = pd.DataFrame({"id": b["auction_id"].to_numpy(), "item": [{"id": int(i)} for i in b["item_id"]], "quantity": b["quantity"].to_numpy(), "time_left": b["time_left"].to_numpy(),
                      "item_id": b["item_id"].to_numpy(), "variant_key": b["variant_key"].to_numpy(), "source": source, "snapshot_ts": snap.ts, "scan_timestamp": snap.ts + pd.Timedelta(minutes=6),
                      "schema_version": 2})
    if buyout:
        f["quantity"] = 1
        f["buyout"] = b["unit_price"].to_numpy()
    else:
        f["unit_price"] = b["unit_price"].to_numpy()
    return f


def key_for(source, ts):
    return f"{source}/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/hour_{ts.hour:02d}_{ts.minute:02d}.parquet"


class TestWholeTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.houses = [110, 111, 113, 116]
        cls.w = world(cls.houses)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "lake")
        names = {}
        for h in self.houses:
            for kind in (True, False):
                src = FV.source_name(h, kind)
                names[src] = {"house_id": h}
                for s in self.w[h]:
                    write(self.root, key_for(src, s.ts), frame(s, src, buyout=not kind))
        write(self.root, FV.SOURCES_KEY, json.dumps(names).encode())
        self.now = pd.Timestamp("2026-11-25 12:10").to_pydatetime()
        self.store = LocalStore(self.root)
        self.api = Api({"static-forever-us": {2000 + i: f"Item {i}" for i in range(8)}, "static-classic1x-us": {2008: "Item 8"}}, extra={"/profession/index": (404, None)})
        truth = {"uspvp--alliance": (110, 3), "uspve--alliance": (111, 1), "uspvp--horde": (113, 4)}
        self.fixture = {"captured": "test", "houses": {slug: page_from(self.w[h], hour, items=range(9)) for slug, (h, hour) in truth.items()}}
        self.fixture["houses"]["uspvp--alliance"]["rows"].append(dict(item="Unheard Of", qty=10, each_copper=5, source="ah"))

    def go(self, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return P.run(args(**kw), self.api, self.store, now=self.now, base_reader=base_reader, sleep=NOSLEEP, fixture=self.fixture)

    def test_items_section_counts_the_ids_in_the_newest_snapshots_and_who_named_them(self):
        r = self.go(sections="items")
        t = r["text"]
        self.assertEqual(r["code"], 0)
        self.assertIn("### 2 items", t)
        self.assertNotIn("### 1", t)
        self.assertIn("10 distinct item ids seen in the Forever lake; 9 named (90.0%); 1 not named by any source.", t)
        self.assertRegex(t, r"classic_era\s+asked\s+2\s+named\s+1")
        self.assertIn("2009", t)                                                    # the one unnamed id is listed
        self.assertIn("profession index 404", t)

    def test_the_newest_file_is_the_one_read(self):
        obj = P.newest_object(self.store, FV.source_name(110, True), self.now)
        self.assertIn("hour_11_25", obj.key)
        self.assertIsNone(P.newest_object(self.store, FV.source_name(999, True), self.now))

    def test_gold_in_play_prices_commodities_by_unit_and_regular_auctions_by_buyout(self):
        ids, value, units, notes = P.newest_summary(self.store, [110], self.now, base_reader)
        self.assertEqual(ids, [2000 + i for i in range(10)])
        b = self.w[110][-1].book
        want_commodity = float((b["quantity"] * b["unit_price"]).sum()) / 10000.0
        want_auction = float(b["unit_price"].sum()) / 10000.0                       # regular auctions here hold one unit per row at the same price
        self.assertAlmostEqual(float(value.sum()), want_commodity + want_auction, places=3)
        self.assertAlmostEqual(float(units.sum()), float(b["quantity"].sum()))
        self.assertEqual(notes, [])

    def test_fingerprint_section_finds_the_houses_and_passes_the_method_check(self):
        r = self.go(sections="fingerprint")
        t = r["text"]
        self.assertIn("### 3 house fingerprint", t)
        self.assertIn("Item names mapped to ids: 9 of 10 (not found: Unheard Of).", t)   # nine real names, one the pretend API has never heard of
        self.assertIn("[OK] method check: US PvP (Alliance) lines up best with house 110", t)
        self.assertIn("US PvE (Alliance): best house 111", t)
        self.assertIn("US PvP (Horde): best house 113", t)
        self.assertIn("Score matrix", t)
        self.assertIn("Best snapshot time per page", t)
        for h in (110, 111, 113, 116):
            self.assertIn(str(h), t)

    def test_fingerprint_with_too_few_named_items_says_so(self):
        self.api = Api({"static-forever-us": {2000: "Item 0", 2001: "Item 1"}})
        t = self.go(sections="fingerprint")["text"]
        self.assertIn("Too few names could be mapped", t)

    def test_all_sections_run_in_order_and_one_failing_section_does_not_end_the_report(self):
        with mock.patch.object(P, "structure_section", side_effect=RuntimeError(f"boom {FV.hf_repo()}")):
            t = self.go()["text"]
        self.assertIn("structure section failed: boom <forever dataset>", t)
        self.assertNotIn(FV.hf_repo().split("/")[0], t)
        self.assertLess(t.index("### 1 structure"), t.index("### 2 items"))
        self.assertLess(t.index("### 2 items"), t.index("### 3 house fingerprint"))

    def test_an_unknown_section_or_an_unreachable_dataset_stops_with_a_clear_message(self):
        self.assertEqual(self.go(sections="items,prices")["code"], 1)
        os.remove(os.path.join(self.root, FV.SOURCES_KEY))
        r = self.go(sections="items")
        self.assertEqual(r["code"], 1)
        self.assertIn("empty or unreachable", r["text"])
        self.assertEqual(self.go(sections="structure")["code"], 0)                   # structure needs no lake

    def test_no_account_name_reaches_the_report(self):
        t = self.go()["text"]
        self.assertNotIn(FV.hf_repo().split("/")[0], t)
        self.assertFalse(any(line.startswith("=") for line in t.splitlines()))


class TestMain(unittest.TestCase):
    def test_missing_secrets_are_a_clear_stop(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(out):
            for k in ("BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET", "HF_TOKEN"):
                os.environ.pop(k, None)
            self.assertEqual(P.main(["--out", tempfile.gettempdir()]), 1)
        self.assertIn("BLIZZARD_CLIENT_ID", out.getvalue())

    def test_an_error_that_carries_the_dataset_address_is_blanked_before_it_is_printed(self):
        class Boom:
            def __init__(self, *a, **k):
                raise RuntimeError(f"404 https://huggingface.co/api/datasets/{FV.hf_repo()}/tree/main")
        out = io.StringIO()
        env = {"HF_TOKEN": "x", "BLIZZARD_CLIENT_ID": "i", "BLIZZARD_CLIENT_SECRET": "s"}
        with mock.patch.dict(os.environ, env), mock.patch.object(P, "Blizzard", Boom), contextlib.redirect_stdout(out):
            self.assertEqual(P.main(["--out", tempfile.gettempdir()]), 1)
        self.assertIn("forever_probe failed", out.getvalue())
        self.assertNotIn(FV.hf_repo().split("/")[0], out.getvalue())
        self.assertNotIn(FV.hf_repo().split("/")[1], out.getvalue())


class TestFixtureAndWorkflow(unittest.TestCase):
    def test_the_fixture_is_the_eight_us_pages_with_priced_rows(self):
        fx = P.load_fixture()
        self.assertEqual(sorted(fx["houses"]), sorted(f"us{rs}--{f}" for rs in ("pve", "pvp", "rp", "rppvp") for f in ("alliance", "horde")))
        for slug, page in fx["houses"].items():
            self.assertGreaterEqual(len(page["rows"]), 14, slug)
            self.assertGreater(page["price_ladders"], 1000)
            for r in page["rows"]:
                self.assertIn(r["source"], ("ah", "vendor"))
                self.assertGreater(r["qty"], 0)
                self.assertIsInstance(r["each_copper"], int)
                self.assertGreater(r["each_copper"], 0)
            self.assertGreaterEqual(sum(1 for r in page["rows"] if r["source"] == "ah"), HM.MIN_ITEMS + 2, slug)
        self.assertIn("approximate", fx["note"])
        spot = {r["item"]: r for r in fx["houses"]["uspvp--alliance"]["rows"]}["Stranglevine"]
        self.assertEqual((spot["qty"], spot["each_copper"]), (368, 2591))              # 25s 91c
        self.assertEqual({r["item"]: r["each_copper"] for r in fx["houses"]["uspvp--alliance"]["rows"]}["Crystal Vial"], 500)

    def test_the_known_house_is_a_page_in_the_fixture(self):
        self.assertTrue(set(HM.KNOWN) <= set(P.load_fixture()["houses"]))

    def wf(self, path):
        with open(os.path.join(_root, path), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_manual_only_read_only_and_wired_to_the_right_secrets(self):
        for path in (".github/workflows/forever_probe.yml", "tools/forever_probe.yml"):
            d = self.wf(path)
            on = d.get("on") or d.get(True)
            self.assertEqual(set(on), {"workflow_dispatch"})
            self.assertEqual(set(on["workflow_dispatch"]["inputs"]), {"sections", "hours", "all_rulesets"})
            text = json.dumps(d)
            self.assertNotIn("upload", text.replace("upload-artifact", ""))
            self.assertNotIn("contents: write", text)
            step = next(s for s in d["jobs"]["probe"]["steps"] if s.get("name") == "Forever probe")
            self.assertEqual(set(step["env"]), {"HF_TOKEN", "BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET", "SECTIONS", "HOURS", "ALL_RULESETS"})
            for k in ("HF_TOKEN", "BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET"):
                self.assertIn(f"secrets.{k}", step["env"][k])
            self.assertIn("set -o pipefail", step["run"])
            self.assertIn("tools/forever_probe.py", step["run"])

    def test_the_copy_in_tools_is_identical_and_the_fixture_ships_with_the_tool(self):
        with open(os.path.join(_root, ".github/workflows/forever_probe.yml"), encoding="utf-8") as a, open(os.path.join(_root, "tools/forever_probe.yml"), encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())
        self.assertTrue(os.path.exists(os.path.join(_root, "tools", "fwc_fingerprint.json")))


if __name__ == "__main__":
    unittest.main()
