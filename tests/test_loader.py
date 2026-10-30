import os
import pickle
import tempfile
import unittest
from datetime import date

import pandas as pd

from helpers import T0  # noqa: F401  (sets sys.path)
from engine import classify_series
from engine.loader import (
    Cache,
    B2Store, BudgetExceeded, ByteBudget, HFStore, LoaderError, LocalStore, ObjectInfo, RangeFile,
    choose_files, coverage_report, load_snapshots, plan_load,
)
from engine.simulator import MarketSimulator, default_profiles
from test_pipeline import as_fetch_v2_frame

SRC = "commodities"


def fake_reader(fileobj, item_ids):
    """Stands in for pyarrow: the 'parquet' files in these tests are pickled DataFrames."""
    return pickle.loads(fileobj.read())


def key_for(ts, kind="hour"):
    base = f"{SRC}/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/"
    return base + (f"hour_{ts.hour:02d}_{ts.minute:02d}.parquet" if kind == "hour" else "daily.parquet")


def write(root, key, frame):
    path = os.path.join(root, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(frame, fh)


class LakeCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = MarketSimulator(default_profiles(), seed=5).run(30)
        cls.snaps = cls.res.snapshots

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lake = os.path.join(self.tmp.name, "lake")
        self.cache = os.path.join(self.tmp.name, "cache")
        self.addCleanup(self.tmp.cleanup)

    def write_hourly(self, snaps=None, transform=None):
        for s in snaps or self.snaps:
            f = as_fetch_v2_frame([s])
            write(self.lake, key_for(s.ts), transform(f) if transform else f)

    @property
    def span(self):
        return self.snaps[0].ts.date(), self.snaps[-1].ts.date()


class TestRangeFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = bytes(range(256)) * 40  # 10240 bytes
        with open(os.path.join(self.tmp.name, "f.bin"), "wb") as fh:
            fh.write(self.data)
        self.store = LocalStore(self.tmp.name)

    def test_reads_match_the_file_across_block_boundaries(self):
        f = RangeFile(self.store, "f.bin", len(self.data), block_size=1000)
        f.seek(995)
        self.assertEqual(f.read(10), self.data[995:1005])
        f.seek(-7, 2)
        self.assertEqual(f.read(), self.data[-7:])
        f.seek(0)
        self.assertEqual(f.read(), self.data)
        self.assertEqual(f.read(5), b"")

    def test_only_the_blocks_touched_are_downloaded_and_counted(self):
        budget = ByteBudget("b2", None)
        f = RangeFile(self.store, "f.bin", len(self.data), budget, block_size=1000)
        f.seek(-100, 2)
        f.read()
        self.assertEqual(budget.session_bytes, 240)  # the last block is only 240 bytes long
        f.seek(5000)
        f.read(10)
        f.seek(5000)
        f.read(10)  # same block again: cached, not charged twice
        self.assertEqual(budget.session_bytes, 240 + 1000)
        self.assertEqual(budget.requests, 2)


class TestBudget(unittest.TestCase):
    def test_stops_before_going_over_and_remembers_between_runs(self):
        with tempfile.TemporaryDirectory() as d:
            b = ByteBudget("b2", 1000, d)
            b.charge(600)
            with self.assertRaises(BudgetExceeded):
                b.charge(500)
            self.assertEqual(b.used_today(), 600)  # the refused download was not counted
            again = ByteBudget("b2", 1000, d)       # a fresh run still sees today's usage
            self.assertEqual(again.used_today(), 600)
            again.charge(400)
            with self.assertRaises(BudgetExceeded):
                again.charge(1)
            self.assertEqual(ByteBudget("hf", 1000, d).used_today(), 0)  # budgets are per store


class TestPlanning(unittest.TestCase):
    def test_hourly_preferred_when_nearly_complete_daily_added_when_hours_are_missing(self):
        full = [f"commodities/year=2026/month=10/day=29/hour_{h:02d}_33.parquet" for h in range(24)]
        objs = [ObjectInfo(k, 10) for k in [
            "commodities/year=2026/month=10/day=27/daily.parquet",
            "commodities/year=2026/month=10/day=28/hour_00_05.parquet",
            "commodities/year=2026/month=10/day=28/hour_01_05.parquet",
            "commodities/year=2026/month=10/day=28/daily.parquet",
            "commodities/year=2026/month=10/day=29/daily.parquet",
            "commodities/year=2026/month=10/day=30/hour_00_05.parquet",
            "realm_73/year=2026/month=10/day=28/daily.parquet",
            "commodities/not_a_snapshot.txt",
        ] + full]
        got = [o.key for o in choose_files(objs, "commodities", date(2026, 10, 27), date(2026, 10, 29))]
        self.assertEqual(got[:4], [
            "commodities/year=2026/month=10/day=27/daily.parquet",
            "commodities/year=2026/month=10/day=28/daily.parquet",       # only 2 hourly hours: daily is needed too
            "commodities/year=2026/month=10/day=28/hour_00_05.parquet",
            "commodities/year=2026/month=10/day=28/hour_01_05.parquet",
        ])
        self.assertEqual(got[4:], sorted(full))                          # a complete day of hourly files: daily not downloaded
        self.assertNotIn("commodities/year=2026/month=10/day=29/daily.parquet", got)


class TestLoading(LakeCase):
    def test_round_trip_matches_the_original_snapshots(self):
        self.write_hourly()
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(len(r.snapshots), len(self.snaps))
        for a, b in zip(self.snaps, r.snapshots):
            self.assertEqual(a.ts, b.ts)
            x = a.book.sort_values("auction_id").reset_index(drop=True)
            y = b.book.sort_values("auction_id").reset_index(drop=True)
            for c in ("variant_key", "time_left"):
                x[c], y[c] = x[c].astype(object), y[c].astype(object)
            pd.testing.assert_frame_equal(x, y)

    def test_plan_reports_size_cache_and_missing_days(self):
        self.write_hourly()
        lo, hi = self.span
        store = LocalStore(self.lake)
        p = plan_load(store, SRC, lo, hi + pd.Timedelta(days=3), cache_dir=self.cache)
        self.assertEqual(len(p.files), len(self.snaps))
        self.assertEqual(len(p.days_missing), 3)
        self.assertEqual(p.cached, [])
        load_snapshots(store, SRC, lo, hi, cache_dir=self.cache, reader=fake_reader)
        p2 = plan_load(store, SRC, lo, hi, cache_dir=self.cache)
        self.assertEqual(len(p2.cached), len(self.snaps))
        self.assertEqual(p2.bytes_to_download_upper_bound, 0)

    def test_second_load_downloads_nothing(self):
        self.write_hourly()
        store = LocalStore(self.lake)
        first = load_snapshots(store, SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        second = load_snapshots(store, SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertGreater(first.downloaded_bytes, 0)
        self.assertEqual(second.downloaded_bytes, 0)
        self.assertEqual(len(second.snapshots), len(first.snapshots))

    def test_budget_stops_the_load_instead_of_leaving_silent_holes(self):
        self.write_hourly()
        one_file = os.path.getsize(os.path.join(self.lake, key_for(self.snaps[0].ts)))
        with self.assertRaises(BudgetExceeded):
            load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader,
                           max_bytes=int(one_file * 3.5))
        # what was fetched before the stop is cached, so tomorrow's run only needs the rest
        p = plan_load(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache)
        self.assertEqual(len(p.cached), 3)

    def test_item_filter_gives_the_same_answers_for_those_items(self):
        self.write_hourly()
        store = LocalStore(self.lake)
        full = load_snapshots(store, SRC, *self.span, cache_dir=os.path.join(self.cache, "a"), reader=fake_reader)
        part = load_snapshots(store, SRC, *self.span, item_ids=[1002, 1003], cache_dir=os.path.join(self.cache, "b"),
                              reader=fake_reader)
        self.assertEqual(set(part.snapshots[0].book["item_id"]), {1002, 1003})
        self.assertEqual(part.snapshots[0].covered_items, frozenset({1002, 1003}))
        cols = ["item_id", "t0", "t1", "sold_cert", "sold_low", "sold_est", "sold_high"]
        a = classify_series(full.snapshots)
        a = a[a["item_id"].isin([1002, 1003])][cols].sort_values(["item_id", "t1"]).reset_index(drop=True)
        b = classify_series(part.snapshots)[cols].sort_values(["item_id", "t1"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_filtered_request_is_served_from_a_full_cache(self):
        self.write_hourly()
        store = LocalStore(self.lake)
        load_snapshots(store, SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        r = load_snapshots(store, SRC, *self.span, item_ids=[1001], cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(r.downloaded_bytes, 0)
        self.assertEqual(set(r.snapshots[3].book["item_id"]), {1001})

    def test_duplicate_snapshots_are_dropped(self):
        self.write_hourly()
        s = self.snaps[4]
        write(self.lake, key_for(s.ts).replace(f"hour_{s.ts.hour:02d}", "hour_23"), as_fetch_v2_frame([s]))
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(len(r.snapshots), len(self.snaps))
        self.assertEqual(r.duplicates_dropped, 1)

    def test_hours_missing_from_the_hourly_files_are_recovered_from_the_daily_file(self):
        """The real failure: some hourly files of a day deleted, the daily file (all hours) was ignored."""
        for day in sorted({x.ts.date() for x in self.snaps}):
            day_snaps = [x for x in self.snaps if x.ts.date() == day]
            write(self.lake, key_for(day_snaps[0].ts, "daily"), as_fetch_v2_frame(day_snaps))
        self.write_hourly(snaps=self.snaps[::7])          # only a few hourly files survive
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual([x.ts for x in r.snapshots], [x.ts for x in self.snaps])
        self.assertGreater(r.duplicates_dropped, 0)

    def test_the_same_snapshot_a_few_minutes_apart_is_still_one_snapshot(self):
        s0, s1 = self.snaps[3], self.snaps[4]
        write(self.lake, key_for(s0.ts), as_fetch_v2_frame([s0]))
        shifted = as_fetch_v2_frame([s0])
        shifted["snapshot_ts"] = shifted["snapshot_ts"] + pd.Timedelta(minutes=4)
        write(self.lake, key_for(s0.ts).replace(f"hour_{s0.ts.hour:02d}", "hour_23"), shifted)
        write(self.lake, key_for(s1.ts), as_fetch_v2_frame([s1]))
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(len(r.snapshots), 2)
        self.assertEqual(r.duplicates_dropped, 1)

    def _mixed_day(self, day_snaps):
        """Shaped like the real 2026-10-29 daily file: older hours have item_id <NA>, an empty source / variant_key /
        snapshot_ts and only clean_item_id + scan_timestamp; newer hours have everything."""
        half = len(day_snaps) // 2
        full = as_fetch_v2_frame(day_snaps)
        full["clean_item_id"] = full["item"].map(lambda x: x["id"]).astype("int64")
        full["item_id"] = full["item_id"].astype("Int64")
        full = full.drop(columns=["item"])
        scan = pd.concat([pd.Series([x.ts] * len(x.book)) for x in day_snaps], ignore_index=True)
        n_old = sum(len(x.book) for x in day_snaps[:half])
        full.loc[:n_old - 1, "item_id"] = pd.NA
        full.loc[:n_old - 1, "variant_key"] = None
        full.loc[:n_old - 1, "source"] = None
        full.loc[:n_old - 1, "snapshot_ts"] = pd.NaT
        full.loc[:n_old - 1, "scan_timestamp"] = scan[:n_old].to_numpy()
        return full

    def test_mixed_layout_day_keeps_the_older_rows(self):
        """The suspected real failure: item_id present for new rows made the loader ignore clean_item_id for old rows."""
        day = self.snaps[:12]
        write(self.lake, key_for(day[0].ts, "daily"), self._mixed_day(day))
        r = load_snapshots(LocalStore(self.lake), SRC, day[0].ts.date(), day[-1].ts.date(), cache_dir=self.cache, reader=fake_reader)
        got = [x.ts for x in r.snapshots if x.ts.date() == day[0].ts.date()]
        want = [x.ts for x in day if x.ts.date() == day[0].ts.date()]
        self.assertEqual(got, want)
        first = r.snapshots[0]
        self.assertEqual(len(first.book), len(day[0].book))

    def test_a_snapshot_with_only_a_fraction_of_the_usual_listings_is_treated_as_missing(self):
        snaps = list(self.snaps[:12])
        part = snaps[5]
        snaps[5] = type(part)(part.source, part.ts, part.book.iloc[: max(1, len(part.book) // 10)].reset_index(drop=True),
                              kind=part.kind, meta=dict(part.meta))
        self.write_hourly(snaps=snaps)
        r = load_snapshots(LocalStore(self.lake), SRC, snaps[0].ts.date(), snaps[-1].ts.date(), cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(len(r.snapshots), 11)
        self.assertEqual([t for t, _, _ in r.incomplete_dropped], [part.ts])
        self.assertNotIn(part.ts, [x.ts for x in r.snapshots])

    def test_a_cache_written_by_the_old_loader_is_not_reused(self):
        c = Cache(self.cache, "local")
        self.assertTrue(c._path("a/b.parquet", "ALL").endswith(".v2.pkl"))

    def test_daily_file_with_many_snapshots(self):
        for day in sorted({s.ts.date() for s in self.snaps}):
            day_snaps = [s for s in self.snaps if s.ts.date() == day]
            write(self.lake, key_for(day_snaps[0].ts, "daily"), as_fetch_v2_frame(day_snaps))
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual([s.ts for s in r.snapshots], [s.ts for s in self.snaps])

    def test_old_v1_files_load_with_a_fetch_time_warning(self):
        def to_v1(f):
            f = f.drop(columns=["item_id", "variant_key", "snapshot_ts", "schema_version"])
            return f
        self.write_hourly(transform=to_v1)
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(len(r.snapshots), len(self.snaps))
        self.assertTrue(all(s.meta.get("ts_source") == "scan_timestamp" for s in r.snapshots))
        self.assertEqual(coverage_report(r.snapshots)["snapshots_with_fetch_time_only"], len(self.snaps))

    def test_compacted_daily_layout_without_item_struct(self):
        def to_compacted(f):
            f = f.copy()
            f["clean_item_id"] = f["item_id"]
            f["item_json"] = f["item"].map(lambda x: str(x))
            return f.drop(columns=["item", "item_id", "variant_key"])
        for day in sorted({s.ts.date() for s in self.snaps}):
            day_snaps = [s for s in self.snaps if s.ts.date() == day]
            write(self.lake, key_for(day_snaps[0].ts, "daily"), to_compacted(as_fetch_v2_frame(day_snaps)))
        r = load_snapshots(LocalStore(self.lake), SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        self.assertEqual(len(r.snapshots), len(self.snaps))
        self.assertEqual(set(r.snapshots[0].book["item_id"]), {1001, 1002, 1003, 1004})

    def test_bad_file_raises_by_default_and_is_reported_when_skipping(self):
        self.write_hourly()
        bad = key_for(self.snaps[2].ts)
        with open(os.path.join(self.lake, bad), "wb") as fh:
            fh.write(b"not a file")
        store = LocalStore(self.lake)
        with self.assertRaises(LoaderError):
            load_snapshots(store, SRC, *self.span, cache_dir=self.cache, reader=fake_reader)
        r = load_snapshots(store, SRC, *self.span, cache_dir=self.cache, reader=fake_reader, on_error="skip")
        self.assertEqual(len(r.snapshots), len(self.snaps) - 1)
        self.assertEqual([k for k, _ in r.skipped], [bad])

    def test_file_with_wrong_columns_is_rejected_not_guessed(self):
        write(self.lake, key_for(self.snaps[0].ts), pd.DataFrame({"foo": [1, 2]}))
        with self.assertRaises(LoaderError):
            load_snapshots(LocalStore(self.lake), SRC, *self.span, reader=fake_reader)

    def test_snapshots_outside_the_date_range_are_excluded(self):
        self.write_hourly()
        day = self.snaps[0].ts.date()
        r = load_snapshots(LocalStore(self.lake), SRC, day, day, cache_dir=self.cache, reader=fake_reader)
        self.assertTrue(all(s.ts.date() == day for s in r.snapshots))
        self.assertLess(len(r.snapshots), len(self.snaps))


class TestCoverage(LakeCase):
    def test_flags_a_gap(self):
        ok = coverage_report(self.snaps)
        self.assertTrue(ok["ok"])
        holed = self.snaps[:10] + self.snaps[16:]
        rep = coverage_report(holed)
        self.assertGreaterEqual(rep["max_gap_min"], 6 * 55)
        self.assertEqual(rep["gaps_over_4h"], 1)


class FakeResponse:
    def __init__(self, status=200, json_data=None, content=b"", links=None):
        self.status_code, self._json, self.content, self.links = status, json_data, content, links or {}

    def json(self): return self._json
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        if "/api/datasets/" in url and "page2" not in url:
            return FakeResponse(json_data=[
                {"type": "file", "path": "commodities/year=2026/month=10/day=28/hour_00_05.parquet", "size": 100},
                {"type": "directory", "path": "commodities/year=2026"},
            ], links={"next": {"url": url + "&page2"}})
        if "/api/datasets/" in url:
            return FakeResponse(json_data=[{"type": "file", "path": "commodities/year=2026/month=10/day=28/hour_01_05.parquet", "size": 200}])
        data = bytes(range(100))
        rng = headers["Range"].split("=")[1]
        a, b = (int(x) for x in rng.split("-"))
        if headers.get("X-Ignore-Range"):
            return FakeResponse(200, content=data)
        return FakeResponse(206, content=data[a:b + 1])


class TestHFStore(unittest.TestCase):
    def test_listing_follows_pagination_and_ignores_folders(self):
        s = HFStore("me/repo", "tok", FakeSession())
        got = s.list("commodities/")
        self.assertEqual([(o.key, o.size) for o in got], [
            ("commodities/year=2026/month=10/day=28/hour_00_05.parquet", 100),
            ("commodities/year=2026/month=10/day=28/hour_01_05.parquet", 200),
        ])

    def test_range_read_sends_the_right_header_and_token(self):
        sess = FakeSession()
        s = HFStore("me/repo", "tok", sess)
        self.assertEqual(s.read_range("a/b.parquet", 10, 20), bytes(range(10, 20)))
        url, headers = sess.calls[-1]
        self.assertEqual(headers["Range"], "bytes=10-19")
        self.assertEqual(headers["Authorization"], "Bearer tok")


if __name__ == "__main__":
    unittest.main()
