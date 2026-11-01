"""Tests for compact.py (the daily compaction script) against fake Backblaze and Hugging Face clients."""
import importlib.util
import io
import os
import sys
import types
import unittest
from unittest import mock

import pandas as pd

from helpers import T0  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = "commodities/year=2026/month=10/day=31/"


class FakeS3:
    class NoSuchKey(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "NoSuchKey"}}

    def __init__(self):
        self.objects, self.puts, self.fail_get_daily, self.truncate_uploads = {}, 0, None, False

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 3]                     # tiny pages: exercises pagination
        out = {"Contents": [{"Key": k} for k in page], "IsTruncated": start + 3 < len(keys)}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + 3)
        return out

    def get_object(self, Bucket, Key):
        if self.fail_get_daily and Key.endswith("daily.parquet"):
            raise self.fail_get_daily
        if Key not in self.objects:
            raise self.NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_fileobj(self, f, Bucket, Key):
        data = f.read()
        self.objects[Key] = data[:-1] if self.truncate_uploads else data

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)


class FakeHf:
    def __init__(self):
        self.uploads, self.commits, self.deleted, self.fail_upload = [], [], [], False

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
        if self.fail_upload:
            raise RuntimeError("hf down")
        self.uploads.append(path_in_repo)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append([o.path_in_repo for o in operations])

    def delete_file(self, path_in_repo, repo_id, repo_type):
        self.deleted.append(path_in_repo)


def load_compact(s3, hf):
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *a, **k: s3
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda token=None: hf
    hub.CommitOperationDelete = type("CommitOperationDelete", (), {"__init__": lambda self, path_in_repo: setattr(self, "path_in_repo", path_in_repo)})
    with mock.patch.dict(sys.modules, {"boto3": boto3, "huggingface_hub": hub, "duckdb": types.ModuleType("duckdb")}):
        spec = importlib.util.spec_from_file_location("compact_under_test", os.path.join(ROOT, "compact.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def hourly(h, n=2):
    return pd.DataFrame({"item": [{"id": 100 + i} for i in range(n)], "quantity": [10] * n, "unit_price": [1000 + h] * n, "fetched": [f"h{h:02d}"] * n})


def put(s3, key, df):
    b = io.BytesIO()
    df.to_pickle(b)
    s3.objects[key] = b.getvalue()


def read(s3, key):
    return pd.read_pickle(io.BytesIO(s3.objects[key]))


class TestCompact(unittest.TestCase):
    def setUp(self):
        self.s3, self.hf = FakeS3(), FakeHf()
        self.c = load_compact(self.s3, self.hf)
        patches = [mock.patch.object(pd.DataFrame, "to_parquet", lambda df, path, **kw: df.to_pickle(path)),
                   mock.patch.object(pd, "read_parquet", lambda path, **kw: pd.read_pickle(path))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def hours(self, hs):
        for h in hs:
            put(self.s3, f"{PREFIX}hour_{h:02d}_33.parquet", hourly(h))

    def test_fresh_compaction_writes_daily_then_deletes_hourly_in_one_hf_commit(self):
        self.hours(range(5))
        self.c.compact_folder(PREFIX)
        daily = read(self.s3, PREFIX + "daily.parquet")
        self.assertEqual(len(daily), 10)
        self.assertEqual(set(daily["fetched"]), {f"h{h:02d}" for h in range(5)})
        self.assertFalse([k for k in self.s3.objects if "hour_" in k])
        self.assertEqual(len(self.hf.commits), 1)
        self.assertEqual(len(self.hf.commits[0]), 5)
        self.assertEqual(self.hf.uploads, [PREFIX + "daily.parquet"])

    def test_rerun_after_a_cut_short_run_merges_instead_of_overwriting(self):
        self.hours(range(3))
        self.c.compact_folder(PREFIX)                     # first run finished hours 0-2
        self.hours([3, 4])                                # hours that were still waiting, plus
        put(self.s3, f"{PREFIX}hour_01_33.parquet", hourly(1))   # one leftover that the cut-short run never deleted
        self.c.compact_folder(PREFIX)
        daily = read(self.s3, PREFIX + "daily.parquet")
        self.assertEqual(set(daily["fetched"]), {f"h{h:02d}" for h in range(5)})
        self.assertEqual(len(daily), 10)                  # the duplicate hour 1 was not counted twice

    def test_nothing_is_deleted_if_the_daily_file_arrives_damaged(self):
        self.hours(range(3))
        self.s3.truncate_uploads = True
        with self.assertRaises(RuntimeError):
            self.c.compact_folder(PREFIX)
        self.assertEqual(len([k for k in self.s3.objects if "hour_" in k]), 3)
        self.assertEqual(self.hf.commits, [])

    def test_hf_failure_keeps_the_hf_hourly_files_as_backup(self):
        self.hours(range(3))
        self.hf.fail_upload = True
        self.c.compact_folder(PREFIX)
        self.assertEqual(self.hf.commits, [])
        self.assertEqual(self.hf.deleted, [])
        self.assertFalse([k for k in self.s3.objects if "hour_" in k])

    def test_unexpected_error_reading_the_existing_daily_stops_before_any_write(self):
        self.hours(range(3))
        put(self.s3, PREFIX + "daily.parquet", hourly(9))
        before = self.s3.objects[PREFIX + "daily.parquet"]
        self.s3.fail_get_daily = ConnectionError("network")
        with self.assertRaises(ConnectionError):
            self.c.compact_folder(PREFIX)
        self.assertEqual(self.s3.objects[PREFIX + "daily.parquet"], before)
        self.assertEqual(len([k for k in self.s3.objects if "hour_" in k]), 3)

    def test_pagination_handles_more_files_than_one_page(self):
        self.hours(range(8))                              # pages of 3
        self.c.compact_folder(PREFIX)
        self.assertEqual(len(read(self.s3, PREFIX + "daily.parquet")), 16)

    def test_main_continues_after_one_folder_fails_and_then_exits_with_an_error(self):
        calls = []

        def fake_compact(prefix):
            calls.append(prefix)
            if prefix.startswith("realm_96"):
                raise RuntimeError("boom")
        self.c.compact_folder = fake_compact
        self.c.generate_dashboard_payloads = lambda: calls.append("payloads")
        with self.assertRaises(SystemExit) as cm:
            self.c.main()
        self.assertEqual(len([c for c in calls if c != "payloads"]), 1 + len(self.c.REALM_IDS))
        self.assertNotIn("payloads", calls)               # skipped by default: the new app does not use them
        self.assertIn("realm_96", str(cm.exception))

    def test_dashboard_payloads_only_run_when_asked(self):
        calls = []
        self.c.compact_folder = lambda prefix: None
        self.c.generate_dashboard_payloads = lambda: calls.append("payloads")
        with mock.patch.dict(os.environ, {"BUILD_DASHBOARD_PAYLOADS": "1"}):
            self.c.main()
        self.assertEqual(calls, ["payloads"])


if __name__ == "__main__":
    unittest.main()
