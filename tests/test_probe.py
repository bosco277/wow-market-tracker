import contextlib
import io
import os
import sys
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import probe as P  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731


def _known():
    """What the real API does for the four namespaces that exist: retail, Progression Classic, Classic Era, Anniversary."""
    d = {}
    for name in ("", "classic", "classic1x", "classicann"):
        sns = f"static-{name}-us" if name else "static-us"
        dns = f"dynamic-{name}-us" if name else "dynamic-us"
        d[sns] = {"/data/wow/item-class/index": (200, {"item_classes": [{"id": 0, "name": "SECRETVALUE"}]}),
                  "/data/wow/item/2589": (200, {"id": 2589, "name": "SECRETVALUE", "quality": {"type": "COMMON"}})}
        d[dns] = {"/data/wow/connected-realm/index": (200, {"connected_realms": [{"href": "https://us.api.blizzard.com/data/wow/connected-realm/4?namespace=" + dns}]}),
                  "/data/wow/realm/index": (200, {"realms": [{"id": 1, "name": "SECRETVALUE"}]})}
        if name in ("", "classic"):
            d[dns]["/data/wow/connected-realm/4/auctions"] = (200, {"auctions": [{"id": 9, "buyout": 5, "seller": "SECRETVALUE"}]})
        else:
            d[dns]["/data/wow/connected-realm/4/auctions/index"] = (200, {"auctions": [{"id": 2, "name": "SECRETVALUE"}]})
    d["dynamic-us"]["/data/wow/auctions/commodities"] = (200, {"auctions": [{"id": 1, "item": {"id": 2589}, "quantity": 5, "unit_price": 100}]})
    return d


class FakeBlizz:
    def __init__(self, live=None, token_ok=True, boom=()):
        self.table, self.token_ok, self.boom, self.calls = _known(), token_ok, set(boom), []
        self.table.update(live or {})

    def __call__(self, url):
        u = urlparse(url)
        ns = parse_qs(u.query).get("namespace", [""])[0]
        self.calls.append((u.path, ns))
        if not self.token_ok:
            return 401, None
        if ns in self.boom:
            raise OSError("network down")
        t = self.table.get(ns)
        if t is None:
            return 403, None
        return t.get(u.path, (404, None))


class TestDescribe(unittest.TestCase):
    def test_shape_has_names_and_types_but_never_values(self):
        js = {"name": "SECRETVALUE", "n": 3, "x": 1.5, "ok": True, "none": None, "rows": [{"id": 1, "seller": "SECRETVALUE"}], "empty": []}
        s = P.describe(js)
        self.assertNotIn("SECRETVALUE", s)
        for part in ("name: str", "n: int", "x: float", "ok: bool", "none: null", "rows: list[1] of {id: int, seller: str}", "empty: list[0]"):
            self.assertIn(part, s)

    def test_depth_and_key_limits(self):
        deep = {"a": {"b": {"c": {"d": 1}}}}
        self.assertIn("{...}", P.describe(deep, depth=2))
        wide = {f"k{i}": i for i in range(30)}
        self.assertIn("+10 more", P.describe(wide, max_keys=20))

    def test_namespace_names(self):
        self.assertEqual(P.namespace("dynamic", "", "us"), "dynamic-us")
        self.assertEqual(P.namespace("static", "classic1x", "eu"), "static-classic1x-eu")


class TestBlizzardProbe(unittest.TestCase):
    def test_nothing_live_means_every_guess_looks_like_the_made_up_one(self):
        fb = FakeBlizz()
        res = P.blizzard_probe(fb, sleep=NOSLEEP)
        self.assertTrue(res["token_ok"])
        self.assertEqual(res["flagged"], [])
        self.assertEqual(set(res["baseline"].values()), {403})
        for c in ("retail", "classic", "classic1x", "classicann"):
            self.assertEqual(set(res["controls"][c].values()), {200})
        text = P.format_blizzard(res)
        self.assertIn(f"None of the {len(P.CANDIDATES)} guesses", text)
        self.assertIn("camelot", text)
        self.assertNotIn("SECRETVALUE", text)
        # every candidate was asked in both a static and a dynamic namespace
        asked = {ns for _, ns in fb.calls}
        self.assertIn("static-camelot-us", asked)
        self.assertIn("dynamic-camelot-us", asked)

    def test_a_live_namespace_is_flagged_and_dug_into(self):
        live = {"static-camelot-us": {"/data/wow/item-class/index": (200, {"item_classes": [{"id": 0, "name": "SECRETVALUE"}]})},
                "dynamic-camelot-us": {"/data/wow/auctions/commodities": (200, {"auctions": [{"id": 1, "quantity": 5}], "market": "SECRETVALUE"})}}
        res = P.blizzard_probe(FakeBlizz(live), sleep=NOSLEEP)
        self.assertEqual(res["flagged"], ["camelot"])
        self.assertEqual(res["rows"]["camelot"], {"/data/wow/item-class/index": 200, "/data/wow/connected-realm/index": 404})
        deep = dict((p, (s, shape)) for p, s, shape in res["deep"]["camelot"])
        status, shape = deep["dynamic /data/wow/auctions/commodities"]
        self.assertEqual(status, 200)
        self.assertIn("auctions: list[1]", shape)
        text = P.format_blizzard(res)
        self.assertIn("1 of", text)
        self.assertIn("DIFFERENTLY", text)
        self.assertIn("(flagged)", text)
        self.assertNotIn("SECRETVALUE", text)

    def test_known_namespaces_are_dug_into_for_comparison_including_the_first_realms_auction_paths(self):
        res = P.blizzard_probe(FakeBlizz(), sleep=NOSLEEP)
        retail = {p: s for p, s, _ in res["deep"]["retail"]}
        self.assertEqual(retail["dynamic /data/wow/auctions/commodities"], 200)
        self.assertEqual(retail["dynamic /data/wow/connected-realm/{id}/auctions"], 200)
        self.assertEqual(retail["dynamic /data/wow/connected-realm/{id}/auctions/index"], 404)
        era = {p: s for p, s, _ in res["deep"]["classic1x"]}
        self.assertEqual(era["dynamic /data/wow/connected-realm/{id}/auctions/index"], 200)
        self.assertEqual(era["dynamic /data/wow/connected-realm/{id}/auctions"], 404)

    def test_extra_guesses_are_tried_lowercased_and_without_duplicates(self):
        fb = FakeBlizz()
        res = P.blizzard_probe(fb, extra=["Camelot2", " fvus ", "camelot"], sleep=NOSLEEP)
        self.assertEqual(res["tried"][:3], ["camelot2", "fvus", "camelot"])
        self.assertEqual(res["tried"].count("camelot"), 1)
        self.assertIn("dynamic-camelot2-us", {ns for _, ns in fb.calls})

    def test_bad_credentials_are_reported_and_nothing_is_dug_into(self):
        fb = FakeBlizz(token_ok=False)
        res = P.blizzard_probe(fb, sleep=NOSLEEP)
        self.assertFalse(res["token_ok"])
        self.assertEqual(res["deep"], {})
        text = P.format_blizzard(res)
        self.assertIn("nothing below means anything", text)
        self.assertEqual(max(res["baseline"].values()), 401)
        self.assertEqual(res["flagged"], [])

    def test_network_failures_count_as_errors_and_are_not_flagged(self):
        res = P.blizzard_probe(FakeBlizz(boom={"dynamic-camelot-us", "static-camelot-us"}), sleep=NOSLEEP)
        self.assertEqual(res["errors"], 2)
        self.assertNotIn("camelot", res["flagged"])
        self.assertIn("2 requests failed", P.format_blizzard(res))

    def test_a_different_baseline_status_does_not_confuse_it(self):
        # if the unknown namespace answered 404 rather than 403, a guess is flagged only when it differs from THAT
        fb = FakeBlizz()
        orig = fb.__call__

        class Weird(FakeBlizz):
            def __call__(self, url):
                s, js = orig(url)
                return (404, None) if s == 403 else (s, js)
        res = P.blizzard_probe(Weird(), sleep=NOSLEEP)
        self.assertEqual(set(res["baseline"].values()), {404})
        self.assertEqual(res["flagged"], [])

    def test_requests_are_paced_and_the_token_never_appears_in_a_url(self):
        waits = []
        fb = FakeBlizz()
        P.blizzard_probe(fb, sleep=waits.append)
        self.assertTrue(waits and all(w > 0 for w in waits))
        self.assertLess(len(fb.calls), 200)


def ahl_route(status=200, ctype="application/json", body="{}", headers=None):
    return status, ctype, body, headers or {}


class FakeAHL:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url):
        path = url[len(P.AHL_BASE):]
        self.calls.append(path)
        r = self.routes.get(path)
        return r if r is not None else (404, "text/html", "<html>not found</html>", {})


OPENAPI = {"openapi": "3.0.0", "paths": {
    "/v1/markets": {"get": {"summary": "List markets"}},
    "/v1/status": {"get": {}},
    "/v1/markets/{market}/items/{item}/hourly": {"get": {"summary": "hourly rows", "parameters": [
        {"name": "market", "in": "path", "required": True, "schema": {"enum": ["us-normal-alliance", "us-normal-horde"]}},
        {"name": "item", "in": "path", "required": True}]}},
    "/v1/deals": {"get": {"summary": "deals", "parameters": [{"name": "market", "in": "query", "required": True}]}},
    "/v1/export": {"post": {"summary": "bulk export"}},
}}


class TestAhledgerProbe(unittest.TestCase):
    def _openapi_routes(self):
        import json
        return {"/openapi.json": ahl_route(body=json.dumps(OPENAPI)),
                "/v1/markets": ahl_route(body=json.dumps([{"id": "us-normal-alliance", "name": "SECRETVALUE", "scans": 12}])),
                "/v1/status": ahl_route(body=json.dumps({"ok": True, "version": "1"}))}

    def test_openapi_is_read_and_only_parameterless_gets_are_fetched(self):
        waits, fa = [], FakeAHL(self._openapi_routes())
        res = P.ahledger_probe(fa, sleep=waits.append)
        self.assertEqual(fa.calls, ["/openapi.json", "/v1/markets", "/v1/status"])            # markets ranks first; deals needs a parameter; export is a POST
        self.assertEqual(res["requests"], 3)
        self.assertEqual(waits, [1.0, 1.0])                                                     # one request per second, no wait before the first
        eps = {(m, p): params for m, p, params, _ in res["endpoints"]}
        self.assertEqual(len(eps), 5)
        hourly = eps[("GET", "/v1/markets/{market}/items/{item}/hourly")]
        self.assertTrue(any("market(path, required, values: us-normal-alliance/us-normal-horde)" == x for x in hourly))
        text = P.format_ahledger(res)
        self.assertIn("GET    /v1/markets/{market}/items/{item}/hourly", text)
        self.assertIn("POST   /v1/export", text)
        self.assertIn("list[1] of {id: str, name: str, scans: int}", text)
        self.assertNotIn("SECRETVALUE", text)
        self.assertIn("Archiving is not addressed", text)

    def test_without_openapi_paths_are_taken_from_a_docs_page(self):
        html = "<html><title>API</title><pre>GET /v1/markets\nGET /v1/markets/{id}/prices\nGET /v1/deals</pre></html>"
        import json
        fa = FakeAHL({"/": ahl_route(ctype="text/html; charset=utf-8", body=html), "/v1/markets": ahl_route(body=json.dumps({"markets": [{"id": "a"}]})),
                      "/v1/deals": ahl_route(body="not json at all", ctype="text/plain")})
        res = P.ahledger_probe(fa, sleep=NOSLEEP)
        self.assertEqual(res["endpoints"], [])
        self.assertEqual(res["text_paths"], ["/v1/markets", "/v1/markets/{id}/prices", "/v1/deals"])
        probed = {p["path"]: p for p in res["probed"]}
        self.assertEqual(set(probed), {"/v1/markets", "/v1/deals"})                             # the one with {id} is only listed
        self.assertIn("markets: list[1]", probed["/v1/markets"]["shape"])
        self.assertIn("not JSON", probed["/v1/deals"]["shape"])
        self.assertIn("Paths mentioned", P.format_ahledger(res))

    def test_nothing_found_says_to_ask_them(self):
        res = P.ahledger_probe(FakeAHL({}), sleep=NOSLEEP)
        self.assertEqual(res["requests"], len(P.DOC_PATHS))
        self.assertEqual(res["probed"], [])
        self.assertIn("must come from AHLedger", P.format_ahledger(res))

    def test_the_request_cap_is_respected(self):
        fa = FakeAHL({})
        res = P.ahledger_probe(fa, max_requests=3, sleep=NOSLEEP)
        self.assertEqual(len(fa.calls), 3)
        self.assertEqual(res["requests"], 3)

    def test_an_authentication_demand_is_reported(self):
        fa = FakeAHL({"/openapi.json": ahl_route(status=401, ctype="application/json", body="{}", headers={"WWW-Authenticate": "Bearer realm=ahl"})})
        res = P.ahledger_probe(fa, sleep=NOSLEEP)
        self.assertIn("Bearer realm=ahl", P.format_ahledger(res))

    def test_a_failing_fetch_does_not_crash(self):
        def boom(url):
            raise OSError("down")
        res = P.ahledger_probe(boom, sleep=NOSLEEP)
        self.assertTrue(all(d["status"] == 0 for d in res["docs"]))


def forever_world(listings=False, index=False, n=6):
    """What the log of Nov 1 showed for 'forever': a dynamic namespace with a realm index and no connected-realm index, no static namespace, no commodities."""
    realms = [{"id": 5000 + i, "name": f"Market {i}", "slug": f"market-{i}", "key": {"href": "x"}} for i in range(n)]
    dyn = {"/data/wow/realm/index": (200, {"realms": realms})}
    for i, r in enumerate(realms[:3]):
        cid = 9000 + i
        dyn[f"/data/wow/realm/{r['slug']}"] = (200, {"id": r["id"], "name": r["name"], "connected_realm": {"href": f"https://us.api.blizzard.com/data/wow/connected-realm/{cid}?namespace=dynamic-forever-us"}})
        dyn[f"/data/wow/connected-realm/{cid}"] = (200, {"id": cid, "status": {"type": "UP"}})
        if index:
            dyn[f"/data/wow/connected-realm/{cid}/auctions/index"] = (200, {"auctions": [{"id": 2, "name": "SECRETVALUE"}]})
        if listings:
            dyn[f"/data/wow/connected-realm/{cid}/auctions/2"] = (200, {"auctions": [{"id": 1, "buyout": 5, "quantity": 3, "seller": "SECRETVALUE"}]})
            dyn[f"/data/wow/connected-realm/{cid}/auctions"] = (200, {"auctions": [{"id": 1, "buyout": 5}]})
    return {"dynamic-forever-us": dyn, "dynamic-forever-eu": {"/data/wow/realm/index": (200, {"realms": realms[:2]})}}


class TestForeverFollowUp(unittest.TestCase):
    def test_a_namespace_with_only_a_realm_index_is_flagged_listed_and_followed(self):
        fb = FakeBlizz(forever_world())
        res = P.blizzard_probe(fb, sleep=NOSLEEP)
        self.assertEqual(res["flagged"], ["forever"])
        self.assertEqual(res["rows"]["forever"], {"/data/wow/item-class/index": 403, "/data/wow/connected-realm/index": 404})   # exactly the Nov 1 pattern
        self.assertEqual([n for n, _, _ in res["names"]["forever"]], [f"Market {i}" for i in range(6)])
        paths = {path for path, ns in fb.calls if ns == "dynamic-forever-us"}
        for i in range(3):
            self.assertIn(f"/data/wow/realm/market-{i}", paths)                                   # the first three realms are followed
            self.assertIn(f"/data/wow/connected-realm/{9000 + i}/auctions/index", paths)           # via the connected realm id the realm names
        self.assertNotIn("/data/wow/realm/market-3", paths)
        v = res["verdicts"]["forever"]
        self.assertEqual((v["realms"], v["static_items"], v["auction_index"], v["auction_listings"]), (6, 403, False, False))
        self.assertEqual(res["regions"]["forever"], {"eu": (200, 2), "kr": (403, None), "tw": (403, None)})

    def test_the_report_gives_a_plain_verdict_and_the_realm_names(self):
        text = P.format_blizzard(P.blizzard_probe(FakeBlizz(forever_world()), sleep=NOSLEEP))
        for needle in ("VERDICT for 'forever'", "Blizzard knows this namespace: YES", "6 realms listed", "not published yet (answers 403)", "Auction LISTINGS (prices and quantities): not reachable",
                       "Other regions: eu answers (2 realms), kr 403, tw 403", "Realms in dynamic-forever-us (6): Market 0 [5000], Market 1 [5001]"):
            self.assertIn(needle, text)
        self.assertFalse(any(line.startswith("==") for line in text.splitlines()))
        self.assertNotIn("SECRETVALUE", text)

    def test_reachable_auction_index_and_listings_are_told_apart(self):
        v = P.blizzard_probe(FakeBlizz(forever_world(index=True)), sleep=NOSLEEP)["verdicts"]["forever"]
        self.assertEqual((v["auction_index"], v["auction_listings"]), (True, False))            # the houses are named but no prices: like Classic Era
        res = P.blizzard_probe(FakeBlizz(forever_world(index=True, listings=True)), sleep=NOSLEEP)
        v = res["verdicts"]["forever"]
        self.assertEqual((v["auction_index"], v["auction_listings"]), (True, True))
        labels = [lab for lab, st, _ in res["deep"]["forever"] if st == 200]
        self.assertTrue(any("auction house id" in lab for lab in labels))                       # the first house named by the index was asked for its listing
        text = P.format_blizzard(res)
        self.assertIn("REACHABLE - see the shape below", text)
        self.assertNotIn("SECRETVALUE", text)

    def test_known_classic_namespaces_get_the_same_follow_up_for_comparison_but_their_realm_names_are_not_printed(self):
        res = P.blizzard_probe(FakeBlizz(), sleep=NOSLEEP)
        self.assertIn("classic1x", res["deep"])
        self.assertTrue(any(lab.startswith("realm 1:") for lab, _, _ in res["deep"]["classic1x"]))
        self.assertEqual(res["names"], {})
        self.assertEqual(res["verdicts"], {})
        self.assertNotIn("VERDICT", P.format_blizzard(res))

    def test_a_dead_realm_index_does_not_break_the_follow_up(self):
        world = forever_world()
        world["dynamic-forever-us"]["/data/wow/realm/index"] = (200, {"realms": []})
        res = P.blizzard_probe(FakeBlizz(world), sleep=NOSLEEP)
        self.assertEqual(res["verdicts"]["forever"]["realms"], None)
        self.assertIn("not listed", P.format_blizzard(res))


class TestCommands(unittest.TestCase):
    def _args(self, **kw):
        import lake
        base = dict(ids="", out="x", upload=False)
        base.update(kw)
        return lake.argparse.Namespace(**base)

    def test_api_probe_prints_a_report_and_takes_word_guesses_from_item_ids(self):
        import lake
        fb, buf = FakeBlizz(), io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_api_probe(None, self._args(ids="Camelot2, 123 ,fvus"), get=fb, sleep=NOSLEEP)
        text = buf.getvalue()
        self.assertEqual(res["tried"][:2], ["camelot2", "fvus"])
        self.assertNotIn("123", res["tried"])
        self.assertIn("BLIZZARD NAMESPACE PROBE", text)
        self.assertNotIn("SECRETVALUE", text)

    def test_ahledger_check_prints_a_report(self):
        import json
        import lake
        fa, buf = FakeAHL({"/openapi.json": ahl_route(body=json.dumps(OPENAPI)), "/v1/markets": ahl_route(body="[]")}), io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_ahledger_check(None, self._args(), fetch=fa, sleep=NOSLEEP)
        self.assertEqual(res["requests"], 3)                                                   # the OpenAPI file, then /v1/markets and /v1/status
        self.assertIn("AHLEDGER API PROBE", buf.getvalue())
        self.assertIn("/v1/export", buf.getvalue())

    def test_missing_blizzard_credentials_name_the_mode(self):
        import lake
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                lake.run_api_probe(None, self._args())
        self.assertIn("api_probe needs BLIZZARD_CLIENT_ID", str(cm.exception))

    def test_the_command_line_accepts_both_modes_without_opening_a_store(self):
        import lake
        with mock.patch.object(lake, "run_api_probe", return_value=1) as a, mock.patch.object(lake, "run_ahledger_check", return_value=2) as b, \
                mock.patch.object(lake, "make_store", side_effect=AssertionError("no store needed")):
            self.assertEqual(lake.main(["api_probe", "--ids", "camelot2"]), 1)
            self.assertEqual(lake.main(["ahledger_check"]), 2)
        self.assertEqual(a.call_args[0][1].ids, "camelot2")


class TestWorkflowWiring(unittest.TestCase):
    def test_modes_are_wired_in_both_copies_and_save_no_files(self):
        import yaml
        with open(os.path.join(_root, ".github", "workflows", "pick_items.yml")) as f:
            a = f.read()
        with open(os.path.join(_root, "tools", "pick_items.yml")) as f:
            b = f.read()
        self.assertEqual(a, b)
        wf = yaml.safe_load(a)
        inputs = wf[True]["workflow_dispatch"]["inputs"]
        for mode in ("api_probe", "ahledger_check"):
            self.assertIn(mode, inputs["mode"]["options"])
        save = next(s for s in wf["jobs"]["run"]["steps"] if s.get("name") == "Save results")
        for mode in ("api_probe", "ahledger_check"):
            self.assertIn(f"inputs.mode != '{mode}'", save["if"])
        run = next(s for s in wf["jobs"]["run"]["steps"] if s.get("name") == "Run")
        self.assertEqual(run["env"]["BLIZZARD_CLIENT_ID"], "${{ secrets.BLIZZARD_CLIENT_ID }}")
        self.assertLessEqual(len(inputs), 25)


if __name__ == "__main__":
    unittest.main()
