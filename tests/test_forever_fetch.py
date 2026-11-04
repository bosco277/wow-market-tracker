"""fetch_forever.py against a pretend Blizzard and a pretend Hugging Face: nothing leaves the machine. Covers: quiet before launch, the first live hour (alert + one commit), unchanged hours,
per-realm isolation of errors, a failed commit advancing nothing, and that no account name reaches the output. Plus wiring checks on the workflow file."""
import contextlib
import io
import json
import os
import re
import sys
import unittest
from unittest import mock

import pandas as pd
import requests
import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
import fetch_forever as F  # noqa: E402

HOST = "https://us.api.blizzard.com"
INDEX = {"realms": [
    {"id": 6798, "name": "CAMWOW US1 Realm 2 PvP", "slug": "camwow-us1-realm-2-pvp"},
    {"id": 6800, "name": "CAMWOW US2 Realm 2 PvP", "slug": "camwow-us2-realm-2-pvp"},
    {"id": 6811, "name": "CAMWOW US2 Account", "slug": "camwow-us2-account"}]}
RECORDS = {"camwow-us1-realm-2-pvp": 7001, "camwow-us2-realm-2-pvp": 7002, "camwow-us2-account": 7003}
AUCTIONS = [{"id": 1, "item": {"id": 2589, "context": 0}, "bid": 0, "buyout": 150, "quantity": 20, "time_left": "SHORT"},
            {"id": 2, "item": {"id": 2592, "bonus_lists": [5, 3]}, "bid": 0, "buyout": 90, "quantity": 1, "time_left": "LONG"}]
LM = "Tue, 04 Nov 2026 15:31:02 GMT"


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error for url")


class FakeHTTP:
    """routes: {url: Resp or [Resp, ...] (used in order, the last one repeats)}. Unknown URLs answer 404."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        r = self.routes.get(url, Resp(404))
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        return r

    def post(self, url, **kw):
        if "oauth" in url:
            return Resp(200, {"access_token": "tok"})
        return Resp(204)


def routes(index=Resp(200, INDEX), auctions=None):
    r = {f"{HOST}/data/wow/realm/index": index}
    for slug, cid in RECORDS.items():
        r[f"{HOST}/data/wow/realm/{slug}"] = Resp(200, {"connected_realm": {"href": f"{HOST}/data/wow/connected-realm/{cid}?namespace=dynamic-forever-us"}})
    for cid in RECORDS.values():
        r[f"{HOST}/data/wow/connected-realm/{cid}/auctions"] = (auctions or {}).get(cid, Resp(404))
    return r


def run_main(route_map, store=None, commit_error=None, realm_ids="6798,6800", webhook=True):
    """Runs main() with everything outside faked. Returns (exit code, printed text, commits, notices, store, http)."""
    store = {} if store is None else store
    commits, notices = [], []
    http = FakeHTTP(route_map)

    def fake_commit(files, message):
        if commit_error:
            raise commit_error
        commits.append((dict(files), message))
        store.update(files)

    out = io.StringIO()
    with mock.patch.object(F, "HTTP", http), mock.patch.object(F, "hf_get", lambda k: store.get(k)), mock.patch.object(F, "hf_commit", fake_commit), \
            mock.patch.object(F, "ensure_repo", lambda: None), mock.patch.object(F, "notify", notices.append), mock.patch.object(F.time, "sleep", lambda s: None), \
            mock.patch.object(F, "to_parquet_bytes", lambda df: df.to_csv(index=False).encode("utf-8")), \
            mock.patch.dict(os.environ, {"FOREVER_REALM_IDS": realm_ids}), contextlib.redirect_stdout(out):
        code = F.main()
    return code, out.getvalue(), commits, notices, store, http


def live(cids=(7001, 7002), header=LM):
    return {c: Resp(200, {"auctions": AUCTIONS}, {"Last-Modified": header}) for c in cids}


class TestBeforeLaunch(unittest.TestCase):
    def test_everything_dark_is_quiet_and_green(self):
        code, out, commits, notices, store, http = run_main(routes())
        self.assertEqual(code, 0)
        self.assertEqual((commits, notices, store), ([], [], {}))
        self.assertIn("not served yet 2", out)
        self.assertIn("2 realm(s) resolved of 2 wanted", out)

    def test_realm_list_not_served_stops_early(self):
        code, out, commits, notices, store, http = run_main(routes(index=Resp(403)))
        self.assertEqual((code, commits, notices), (0, [], []))
        self.assertIn("not live yet", out)
        self.assertFalse([c for c in http.calls if "auctions" in c[0]])               # no pointless auction calls

    def test_unknown_realm_id_is_logged_not_a_failure(self):
        code, out, commits, notices, *_ = run_main(routes(), realm_ids="6798,424242")
        self.assertEqual((code, notices), (0, []))
        self.assertIn("424242", out)


class TestRealmListChanged(unittest.TestCase):
    def test_live_list_with_none_of_our_realms_alerts_and_shows_the_list(self):
        new_list = {"realms": [{"id": 9001, "name": "Camelot PvP 1", "slug": "camelot-pvp-1"}, {"id": 9002, "name": "Camelot PvE 1", "slug": "camelot-pve-1"}]}
        code, out, commits, notices, store, http = run_main(routes(index=Resp(200, new_list)))
        self.assertEqual((code, commits), (0, []))
        self.assertEqual(len(notices), 1)
        for text in (out, notices[0]):
            self.assertIn("9001 Camelot PvP 1", text)
            self.assertIn("9002 Camelot PvE 1", text)
            self.assertIn("FOREVER_REALM_IDS", text)
        self.assertFalse([c for c in http.calls if "/auctions" in c[0]])

    def test_a_partly_matching_list_does_not_alert(self):
        code, out, commits, notices, *_ = run_main(routes(), realm_ids="6798,424242")
        self.assertEqual((code, notices), (0, []))

    def test_long_list_is_cut_at_25(self):
        big = {"realms": [{"id": 9000 + i, "name": f"R{i}", "slug": f"r{i}"} for i in range(40)]}
        code, out, commits, notices, *_ = run_main(routes(index=Resp(200, big)))
        self.assertIn("first 25 of 40", notices[0])
        self.assertNotIn("9039 R39", notices[0])

    def test_served_but_empty_list_is_quiet(self):
        code, out, commits, notices, *_ = run_main(routes(index=Resp(200, {"realms": []})))
        self.assertEqual((code, commits, notices), (0, [], []))
        self.assertIn("served but empty", out)


class TestGoingLive(unittest.TestCase):
    def test_first_live_hour_writes_one_commit_and_one_alert(self):
        code, out, commits, notices, store, http = run_main(routes(auctions=live()))
        self.assertEqual(code, 0)
        self.assertEqual(len(commits), 1)
        files, message = commits[0]
        data_keys = [k for k in files if k.startswith("forever_")]
        self.assertEqual(len(data_keys), 2)
        for k in data_keys:
            self.assertRegex(k, r"^forever_700[12]/year=\d{4}/month=\d{2}/day=\d{2}/hour_\d{2}_\d{2}\.parquet$")
        self.assertEqual(json.loads(files[F.MARKERS_KEY]), {"forever_7001": LM, "forever_7002": LM})
        sources = json.loads(files[F.SOURCES_KEY])
        self.assertEqual(sources["forever_7001"], {"realm_id": 6798, "realm_name": "CAMWOW US1 Realm 2 PvP", "connected_realm_id": 7001})
        self.assertEqual(len(notices), 1)
        self.assertIn("LIVE", notices[0])
        self.assertIn("2 realm(s)", notices[0])
        first_calls = [c for c in http.calls if c[0].endswith("/auctions")]
        self.assertTrue(all("If-Modified-Since" not in c[2] for c in first_calls))        # nothing stored yet, so nothing to be modified since
        self.assertTrue(all(c[1]["namespace"] == "dynamic-forever-us" for c in http.calls))

    def test_unchanged_hour_costs_nothing_and_stays_quiet(self):
        _, _, _, _, store, _ = run_main(routes(auctions=live()))
        code, out, commits, notices, _, http = run_main(routes(auctions={7001: Resp(304), 7002: Resp(304)}), store=store)
        self.assertEqual((code, commits, notices), (0, [], []))
        self.assertIn("unchanged 2", out)
        sent = [c[2].get("If-Modified-Since") for c in http.calls if c[0].endswith("/auctions")]
        self.assertEqual(sent, [LM, LM])

    def test_later_snapshots_are_stored_without_a_second_alert(self):
        _, _, _, _, store, _ = run_main(routes(auctions=live()))
        later = "Tue, 04 Nov 2026 16:31:05 GMT"
        code, out, commits, notices, store, _ = run_main(routes(auctions=live(header=later)), store=store)
        self.assertEqual((code, len(commits), notices), (0, 1, []))
        self.assertEqual(json.loads(store[F.MARKERS_KEY])["forever_7001"], later)

    def test_one_realm_going_live_alerts_once_for_that_realm(self):
        code, out, commits, notices, *_ = run_main(routes(auctions=live(cids=(7002,))))
        self.assertEqual((code, len(commits), len(notices)), (0, 1, 1))
        self.assertIn("1 realm(s)", notices[0])
        self.assertIn("not served yet 1", out)

    def test_served_but_empty_is_kept_out_and_marker_not_advanced(self):
        code, out, commits, notices, *_ = run_main(routes(auctions={7001: Resp(200, {"auctions": []}, {"Last-Modified": LM}), 7002: Resp(404)}))
        self.assertEqual((code, commits, notices), (0, [], []))
        self.assertIn("served but empty 1", out)

    def test_frame_has_the_retail_columns_and_values(self):
        df = F.build_frame(AUCTIONS, "forever_7001", F.parse_last_modified(LM), pd.Timestamp("2026-11-04 15:40"))
        for col in ("id", "item", "item_id", "variant_key", "source", "snapshot_ts", "scan_timestamp", "schema_version", "buyout", "quantity", "time_left"):
            self.assertIn(col, df.columns)
        self.assertEqual(list(df["item_id"]), [2589, 2592])
        self.assertEqual(df["variant_key"].iloc[0], "2589:" + df["variant_key"].iloc[0].split(":")[1])           # context is in the item struct, so it gets a hash
        self.assertEqual(df["snapshot_ts"].iloc[0], pd.Timestamp("2026-11-04 15:31:02"))
        self.assertEqual(F.variant_key({"id": 5}), "5")
        self.assertEqual(F.variant_key({"id": 7, "bonus_lists": [3, 1]}), F.variant_key({"id": 7, "bonus_lists": [1, 3]}))
        with self.assertRaises(ValueError):
            F.build_frame([], "x", None, pd.Timestamp("2026-11-04"))

    def test_same_columns_as_the_retail_extractor_when_it_can_be_imported(self):
        try:
            import fetch
        except Exception:
            self.skipTest("fetch.py needs boto3 (not installed here)")
        now = pd.Timestamp("2026-11-04 15:40")
        mine = F.build_frame(AUCTIONS, "s", F.parse_last_modified(LM), now)
        theirs = fetch.build_frame(AUCTIONS, "s", fetch.parse_last_modified(LM), now)
        pd.testing.assert_frame_equal(mine, theirs)

    def test_real_parquet_round_trip(self):
        try:
            import pyarrow  # noqa: F401
        except Exception:
            self.skipTest("pyarrow not installed here")
        df = F.build_frame(AUCTIONS, "forever_7001", F.parse_last_modified(LM), pd.Timestamp("2026-11-04 15:40"))
        back = pd.read_parquet(io.BytesIO(F.to_parquet_bytes(df)))
        self.assertEqual(list(back["item_id"]), [2589, 2592])


class TestWhenThingsGoWrong(unittest.TestCase):
    def test_a_failing_realm_does_not_stop_the_other(self):
        code, out, commits, notices, *_ = run_main(routes(auctions={7001: Resp(500), 7002: Resp(200, {"auctions": AUCTIONS}, {"Last-Modified": LM})}))
        self.assertEqual(code, 1)
        self.assertEqual(len(commits), 1)                                    # realm 2's snapshot still saved
        self.assertEqual([k for k in commits[0][0] if k.startswith("forever_")][0][:12], "forever_7002")
        self.assertTrue(any("problems" in n and "forever_7001" in n for n in notices))
        self.assertTrue(any("LIVE" in n for n in notices))

    def test_server_trouble_is_retried_then_succeeds(self):
        code, out, commits, notices, *_ = run_main(routes(auctions={7001: [Resp(503), Resp(200, {"auctions": AUCTIONS}, {"Last-Modified": LM})], 7002: Resp(404)}))
        self.assertEqual((code, len(commits)), (0, 1))
        self.assertIn("1 realm(s)", notices[0])

    def test_a_failed_commit_advances_nothing_and_says_so(self):
        code, out, commits, notices, store, _ = run_main(routes(auctions=live()), commit_error=RuntimeError("boom"))
        self.assertEqual(code, 1)
        self.assertEqual((commits, store), ([], {}))                         # markers live in the commit, so they did not move
        self.assertFalse(any("LIVE" in n for n in notices))                  # no "live" alert for data that was not saved
        self.assertTrue(any("commit failed" in n for n in notices))

    def test_no_dataset_or_account_name_in_output_or_alerts(self):
        err = RuntimeError(f"404 Client Error for url: https://huggingface.co/api/datasets/{F.HF_REPO_ID}/commit/main")
        code, out, commits, notices, *_ = run_main(routes(auctions=live()), commit_error=err)
        text = out + "\n".join(notices)
        self.assertNotIn(F.HF_REPO_ID, text)
        self.assertNotIn(F.HF_REPO_ID.split("/")[0], text)
        self.assertIn("<forever dataset>", text)

    def test_token_failure_is_an_error(self):
        http = FakeHTTP(routes())
        http.post = lambda url, **kw: Resp(401)
        notices = []
        with mock.patch.object(F, "HTTP", http), mock.patch.object(F, "notify", notices.append), mock.patch.object(F.time, "sleep", lambda s: None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(F.main(), 1)
        self.assertTrue(notices and "token" in notices[0])

    def test_not_found_is_an_answer_but_other_client_errors_raise(self):
        with mock.patch.object(F, "HTTP", FakeHTTP({"u1": Resp(404), "u2": Resp(401)})), mock.patch.object(F.time, "sleep", lambda s: None):
            self.assertEqual(F.api_get("u1", "t")[0], 404)
            with self.assertRaises(requests.HTTPError):
                F.api_get("u2", "t")

    def test_realm_ids_setting(self):
        with mock.patch.dict(os.environ, {"FOREVER_REALM_IDS": ""}):
            self.assertEqual(F.wanted_realm_ids(), F.DEFAULT_REALM_IDS)
        self.assertEqual(len(F.DEFAULT_REALM_IDS), 8)
        with mock.patch.dict(os.environ, {"FOREVER_REALM_IDS": "6798, 6800 6922 junk"}):
            self.assertEqual(F.wanted_realm_ids(), [6798, 6800, 6922])


class TestWorkflowFile(unittest.TestCase):
    path = os.path.join(_root, ".github", "workflows", "forever_fetch.yml")
    copy = os.path.join(_root, "tools", "forever_fetch.yml")

    def _text(self, p):
        with open(p, encoding="utf-8") as f:
            return f.read()

    def test_copy_in_tools_is_identical(self):
        self.assertEqual(self._text(self.path), self._text(self.copy))

    def test_triggers_gate_secrets_and_no_backblaze(self):
        text = self._text(self.path)
        doc = yaml.safe_load(text)
        trig = doc.get("on", doc.get(True))
        self.assertEqual(sorted(trig), ["workflow_dispatch", "workflow_run"])
        self.assertEqual(trig["workflow_run"]["workflows"], ["Hourly Auction Extractor"])
        job = doc["jobs"]["extract-data"]
        self.assertIn("vars.FOREVER_ENABLED == 'true'", job["if"])
        self.assertIn("workflow_dispatch", job["if"])                                    # a hand-started run always goes
        self.assertEqual(doc["concurrency"]["group"], "forever-extract")
        self.assertEqual(set(re.findall(r"secrets\.([A-Z0-9_]+)", text)), {"BLIZZARD_CLIENT_ID", "BLIZZARD_CLIENT_SECRET", "HF_TOKEN", "DISCORD_WEBHOOK_URL"})
        run = " ".join(s.get("run", "") for s in job["steps"])
        self.assertIn("python fetch_forever.py", run)
        self.assertNotIn("requirements.txt", run)                                        # never depends on (or replaces) the repo's requirements file
        self.assertNotIn("B2_", text)
        env = [s for s in job["steps"] if s.get("env")][-1]["env"]
        self.assertEqual(env["FOREVER_REALM_IDS"], "${{ inputs.realm_ids || vars.FOREVER_REALM_IDS }}")            # hand-typed ids win, then the repository variable

    def test_listens_to_the_retail_workflow_by_its_exact_name(self):
        p = os.path.join(_root, ".github", "workflows", "extract.yml")
        if not os.path.exists(p):
            self.skipTest("extract.yml is not in this folder")
        self.assertEqual(yaml.safe_load(self._text(p))["name"], "Hourly Auction Extractor")

    def test_no_account_names_in_workflow_or_script_output_strings(self):
        text = self._text(self.path).lower()
        for word in ("bosco", "kohler"):
            self.assertNotIn(word, text)

    def test_retail_fetcher_is_not_imported_or_changed_by_this_script(self):
        with open(os.path.join(_root, "fetch_forever.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotRegex(src, r"(?m)^\s*(import fetch\b|from fetch\b)")


if __name__ == "__main__":
    unittest.main()
