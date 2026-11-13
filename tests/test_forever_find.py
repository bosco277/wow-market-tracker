"""forever_find.py against a pretend Blizzard: nothing leaves the machine."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import requests
import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
import fetch_forever as F  # noqa: E402
import forever_find as FF  # noqa: E402

HOST = "https://us.api.blizzard.com"
LM = "Fri, 13 Nov 2026 13:31:02 GMT"
INDEX = {"auctions": [{"id": 110, "name": {"en_US": "Alliance Auction House"}}, {"id": 113, "name": {"en_US": "Horde Auction House"}}, {"id": 112, "name": {"en_US": "Alliance Auction House"}}]}


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error for url")


class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        return self.routes.get(url, Resp(404))

    def post(self, url, **kw):
        return Resp(200, {"access_token": "tok"})


def routes(h110_com=(), h113_com=(), h110_reg=(), h113_reg=()):
    base = f"{HOST}/data/wow/ruleset/1/auctions"
    ok = {"Last-Modified": LM}
    return {f"{base}/": Resp(200, INDEX),
            f"{base}/110": Resp(200, {"auctions": list(h110_reg)}, ok), f"{base}/110/commodities": Resp(200, {"auctions": list(h110_com)}, ok),
            f"{base}/113": Resp(200, {"auctions": list(h113_reg)}, ok), f"{base}/113/commodities": Resp(200, {"auctions": list(h113_com)}, ok),
            f"{base}/112": Resp(200, {"id": 112}), f"{base}/112/commodities": Resp(200, {"id": 112})}


def com(i, item, qty, price):
    return {"id": i, "item": {"id": item}, "quantity": qty, "unit_price": price, "time_left": "LONG"}


def run(route_map, env=None):
    out = io.StringIO()
    http = FakeHTTP(route_map)
    with mock.patch.object(F, "HTTP", http), mock.patch.object(F.time, "sleep", lambda s: None), \
            mock.patch.dict(os.environ, {"FOREVER_RULESETS": "", "FIND_ITEM_ID": "", "FIND_PRICE": "", "FIND_QUANTITY": "", **(env or {})}), contextlib.redirect_stdout(out):
        code = FF.main()
    return code, out.getvalue(), http


class TestFinder(unittest.TestCase):
    def test_copper_formatting(self):
        self.assertEqual(FF.fmt_copper(54321), "5g 43s 21c")
        self.assertEqual(FF.fmt_copper(125), "0g 1s 25c")
        self.assertEqual(FF.fmt_copper(0), "0g 0s 0c")

    def test_finds_the_commodity_listing_in_one_house(self):
        r = routes(h110_com=[com(1, 2589, 20, 120), com(2, 2589, 1, 54321)], h113_com=[com(3, 2589, 14, 125), com(4, 2592, 1, 54321)])
        code, out, http = run(r)
        self.assertEqual(code, 0)
        self.assertIn("house 110 (Alliance Auction House), commodities: auction id 2, quantity 1, price 54321 copper (5g 43s 21c)", out)
        self.assertIn("Result: the listing is in house 110.", out)
        self.assertIn("house 113 (Horde Auction House): 0 regular and 1 commodity listings of this item, exact matches 0", out)    # same price on a different item is not a match
        self.assertIn(f"data as of {LM}", out)

    def test_regular_auction_matches_on_buyout(self):
        reg = [{"id": 7, "item": {"id": 2589}, "bid": 100, "buyout": 54321, "quantity": 1, "time_left": "SHORT"},
               {"id": 8, "item": {"id": 2589}, "bid": 54321, "quantity": 1, "time_left": "SHORT"}]            # bid-only: no buyout, not a match
        code, out, _ = run(routes(h113_reg=reg))
        self.assertIn("house 113 (Horde Auction House), auctions: auction id 7", out)
        self.assertIn("Result: the listing is in house 113.", out)

    def test_not_found_says_so_and_shows_the_closest_prices(self):
        r = routes(h110_com=[com(1, 2589, 20, 120), com(2, 2589, 5, 5432)], h113_com=[com(3, 2589, 14, 125)])
        code, out, _ = run(r)
        self.assertEqual(code, 0)
        self.assertIn("NOT FOUND in any house yet", out)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("closest:")]
        self.assertEqual(len(lines), 3)
        self.assertIn("price 5432 copper", lines[0])                                                           # closest first

    def test_more_than_one_matching_house_is_flagged(self):
        r = routes(h110_com=[com(1, 2589, 1, 54321)], h113_com=[com(2, 2589, 1, 54321)])
        code, out, _ = run(r)
        self.assertIn("house 110, 113", out)
        self.assertIn("More than one house matched", out)

    def test_settings_and_quantity_filter(self):
        r = routes(h110_com=[com(1, 2592, 3, 777), com(2, 2592, 9, 777)])
        code, out, _ = run(r, env={"FIND_ITEM_ID": "2592", "FIND_PRICE": "777", "FIND_QUANTITY": "9"})
        self.assertIn("Looking for item 2592 at 777 copper (0g 7s 77c), quantity 9.", out)
        self.assertIn("auction id 2, quantity 9", out)
        self.assertNotIn("auction id 1,", out)

    def test_no_house_list_is_an_answer_not_a_failure(self):
        code, out, _ = run({})
        self.assertEqual(code, 0)
        self.assertIn("serves no auction-house list", out)

    def test_a_failing_house_is_reported_and_the_rest_still_searched(self):
        r = routes(h113_com=[com(4, 2589, 1, 54321)])
        r[f"{HOST}/data/wow/ruleset/1/auctions/110"] = Resp(500)
        code, out, _ = run(r)
        self.assertEqual(code, 1)
        self.assertIn("house 110 (Alliance Auction House): FAILED", out)
        self.assertIn("Result: the listing is in house 113.", out)

    def test_nothing_is_written_or_sent(self):
        http = FakeHTTP(routes())
        with mock.patch.object(F, "HTTP", http), mock.patch.object(F.time, "sleep", lambda s: None), contextlib.redirect_stdout(io.StringIO()):
            FF.main()
        self.assertTrue(all(c.startswith(HOST + "/data/wow/ruleset/") for c in http.calls))                    # only read-only Blizzard auction routes
        with open(os.path.join(_root, "forever_find.py"), encoding="utf-8") as f:
            src = f.read()
        for word in ("hf_commit", "notify(", "HF_TOKEN", "DISCORD"):
            self.assertNotIn(word, src)


class TestFindWorkflow(unittest.TestCase):
    def setUp(self):
        path = os.path.join(_root, ".github", "workflows", "forever_find.yml")
        with open(path, encoding="utf-8") as f:
            self.text = f.read()
        self.wf = yaml.safe_load(self.text)
        self.trig = self.wf.get("on") or self.wf.get(True)

    def test_hand_started_only_with_defaults_for_the_linen_cloth_test(self):
        self.assertEqual(list(self.trig), ["workflow_dispatch"])
        ins = self.trig["workflow_dispatch"]["inputs"]
        self.assertEqual((ins["item_id"]["default"], ins["price_copper"]["default"], ins["quantity"]["default"]), ("2589", "54321", ""))

    def test_only_blizzard_secrets_and_the_script(self):
        step = self.wf["jobs"]["find"]["steps"][-1]
        self.assertEqual(step["run"], "python forever_find.py")
        self.assertEqual(sorted(k for k in step["env"] if "secrets." in step["env"][k]), ["BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET"])
        self.assertNotIn("HF_TOKEN", self.text)
        self.assertNotIn("DISCORD", self.text)

    def test_tools_copy_is_identical(self):
        with open(os.path.join(_root, ".github", "workflows", "forever_find.yml"), encoding="utf-8") as f:
            a = f.read()
        with open(os.path.join(_root, "tools", "forever_find.yml"), encoding="utf-8") as f:
            b = f.read()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
