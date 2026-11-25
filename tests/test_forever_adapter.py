"""engine/forever.py and tools/forever_adapter.py on a pretend Forever dataset (a local folder of pickled frames; nothing leaves the machine): names, the kind that comes from the source name
(and the tripwire that the retail reader alone gets it wrong), repeated auction ids (dropped, counted, copies that disagree), houses never mixed, the retrying store, the house-keyed hourly table
(same numbers as the retail history), the check tool's report and its safety stops, and wiring checks on the workflow."""
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

from engine import daily as D  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine.classifier import classify_series  # noqa: E402
from engine.loader import LoaderError, LocalStore, ObjectInfo, Store, load_snapshots  # noqa: E402
from engine.schema import SchemaError, snapshots_from_blizzard_v2  # noqa: E402
from engine.simulator import MarketSimulator, default_profiles  # noqa: E402
from test_pipeline import as_fetch_v2_frame  # noqa: E402
import forever_adapter as A  # noqa: E402


def base_reader(fileobj, item_ids):
    """Stands in for pyarrow: the 'parquet' files in these tests are pickled DataFrames."""
    return pickle.loads(fileobj.read())


def key_for(source, ts):
    return f"{source}/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/hour_{ts.hour:02d}_{ts.minute:02d}.parquet"


def write(root, key, obj):
    path = os.path.join(root, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        if isinstance(obj, (bytes, str)):
            fh.write(obj if isinstance(obj, bytes) else obj.encode())
        else:
            pickle.dump(obj, fh)


def commodity_frame(snap, source):
    f = as_fetch_v2_frame([snap])
    f["source"] = source
    return f


def regular_frame(snap, source, n=40, bid_only=3):
    """What Forever's regular auctions look like: one unit per row, a buyout (no unit price), and a few bid-only rows."""
    f = as_fetch_v2_frame([snap]).head(n).copy()
    f["source"] = source
    f["quantity"] = 1
    f["buyout"] = f["unit_price"].astype("float64")
    f["bid"] = 100
    f = f.drop(columns=["unit_price"])
    f.loc[f.index[:bid_only], "buyout"] = np.nan
    return f


def write_house(root, house, snaps, commodity=True, transform=None):
    source = FV.source_name(house, commodity)
    for s in snaps:
        f = commodity_frame(s, source) if commodity else regular_frame(s, source)
        write(root, key_for(source, s.ts), transform(s, f) if transform else f)


class Lake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snaps = MarketSimulator(default_profiles(), seed=5).run(60).snapshots

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "lake")
        self.cache = os.path.join(self.tmp.name, "cache")

    @property
    def span(self):
        return self.snaps[0].ts.date(), self.snaps[-1].ts.date()

    def sources_file(self, houses=(110, 113, 116, 112)):
        names = {}
        for h in houses:
            names[FV.source_name(h)] = dict(house_id=h)
            names[FV.source_name(h, True)] = dict(house_id=h)
        write(self.root, FV.SOURCES_KEY, json.dumps(names).encode())


class TestNames(unittest.TestCase):
    def test_parse_source(self):
        self.assertEqual(FV.parse_source("forever_us_r1_h110_commodities"), dict(region="us", ruleset=1, house=110, commodity=True))
        self.assertEqual(FV.parse_source("forever_us_r1_h113"), dict(region="us", ruleset=1, house=113, commodity=False))
        for bad in ("commodities", "us_r1", "forever_us_r1_h110_commodities_x", "forever_us_r1_hx", "forever_us_r1_h110/", ""):
            self.assertIsNone(FV.parse_source(bad))

    def test_source_name_round_trips(self):
        for house in (110, 125):
            for commodity in (False, True):
                p = FV.parse_source(FV.source_name(house, commodity))
                self.assertEqual((p["house"], p["commodity"]), (house, commodity))
        self.assertTrue(FV.is_commodity_source("forever_us_r1_h110_commodities"))
        self.assertFalse(FV.is_commodity_source("forever_us_r1_h110"))

    def test_parse_houses(self):
        self.assertEqual(FV.parse_houses(""), [110, 113, 116])
        self.assertEqual(FV.parse_houses(None), [110, 113, 116])
        self.assertEqual(FV.parse_houses("113, 110;110"), [110, 113])
        self.assertEqual(FV.parse_houses("all", [125, 110, 110]), [110, 125])
        with self.assertRaises(ValueError):
            FV.parse_houses("all", [])
        with self.assertRaises(ValueError):
            FV.parse_houses("110,abc")

    def test_scrub_blanks_the_whole_dataset_name_and_the_owner(self):
        t = FV.scrub("404 https://huggingface.co/datasets/bosco277/wow-forever-data-lake/resolve/main/x for bosco277")
        self.assertNotIn("bosco277", t)
        self.assertNotIn("wow-forever-data-lake", t)
        with mock.patch.dict(os.environ, {"FOREVER_HF_REPO": "someone/other-data"}):
            t2 = FV.scrub("failed someone/other-data and someone")
            self.assertNotIn("someone", t2)
            self.assertNotIn("other-data", t2)


class TestDuplicateIds(unittest.TestCase):
    def frame(self):
        ts = pd.Timestamp("2026-11-25 14:22")
        return pd.DataFrame({"id": [1, 2, 3, 4], "item_id": [10, 10, 11, 12], "unit_price": [100, 200, 300, 400], "quantity": [5, 5, 5, 5], "time_left": ["LONG"] * 4,
                             "source": "forever_us_r1_h110_commodities", "snapshot_ts": ts, "scan_timestamp": ts + pd.Timedelta(minutes=3)})

    def test_a_clean_frame_comes_back_untouched(self):
        f = self.frame()
        out, dropped, conflicting = FV.drop_duplicate_ids(f)
        self.assertEqual((dropped, conflicting), (0, 0))
        pd.testing.assert_frame_equal(out, f)

    def test_an_exact_copy_is_dropped_counted_and_not_called_a_disagreement(self):
        f = pd.concat([self.frame(), self.frame().iloc[[1]]], ignore_index=True)
        out, dropped, conflicting = FV.drop_duplicate_ids(f)
        self.assertEqual((len(out), dropped, conflicting), (4, 1, 0))
        self.assertEqual(out["id"].tolist(), [1, 2, 3, 4])

    def test_copies_that_differ_are_reported_and_the_first_stays(self):
        dup = self.frame().iloc[[1]].copy()
        dup["unit_price"] = 999
        out, dropped, conflicting = FV.drop_duplicate_ids(pd.concat([self.frame(), dup], ignore_index=True))
        self.assertEqual((dropped, conflicting), (1, 1))
        self.assertEqual(int(out.loc[out["id"] == 2, "unit_price"].iloc[0]), 200)

    def test_the_same_id_in_another_snapshot_or_another_source_is_not_a_repeat(self):
        a, b, c = self.frame(), self.frame(), self.frame()
        b["snapshot_ts"] = b["snapshot_ts"] + pd.Timedelta(hours=1)
        c["source"] = "forever_us_r1_h113_commodities"
        out, dropped, _ = FV.drop_duplicate_ids(pd.concat([a, b, c], ignore_index=True))
        self.assertEqual((len(out), dropped), (12, 0))

    def test_a_missing_snapshot_time_falls_back_to_the_fetch_time(self):
        f = self.frame()
        f["snapshot_ts"] = pd.NaT
        out, dropped, _ = FV.drop_duplicate_ids(pd.concat([f, f.iloc[[0]]], ignore_index=True))
        self.assertEqual((len(out), dropped), (4, 1))

    def test_empty_and_odd_frames_pass_through(self):
        e = pd.DataFrame(columns=["id", "source"])
        self.assertEqual(FV.drop_duplicate_ids(e)[1:], (0, 0))
        odd = pd.DataFrame({"x": [1, 1]})
        out, dropped, conflicting = FV.drop_duplicate_ids(odd)
        self.assertEqual((len(out), dropped, conflicting), (2, 0, 0))

    def test_the_engine_rejects_a_repeated_id_without_this_and_accepts_it_with(self):
        f = pd.concat([self.frame(), self.frame().iloc[[2]]], ignore_index=True)
        f["variant_key"] = f["item_id"].astype(str)
        with self.assertRaisesRegex(SchemaError, "duplicate auction_id"):
            snapshots_from_blizzard_v2(f)
        clean, _, _ = FV.drop_duplicate_ids(f)
        self.assertEqual(len(snapshots_from_blizzard_v2(clean)), 1)


class TestLoading(Lake):
    def test_the_commodity_kind_comes_from_the_name_and_the_retail_reader_alone_gets_it_wrong(self):
        write_house(self.root, 110, self.snaps)
        raw = load_snapshots(LocalStore(self.root), FV.source_name(110, True), *self.span, reader=base_reader)
        self.assertEqual({s.kind for s in raw.snapshots}, {"auction"})            # TRIPWIRE: if the retail reader ever learns this, the adapter's kind step can go
        sl = FV.load_source(LocalStore(self.root), 110, True, *self.span, base_reader=base_reader)
        self.assertEqual({s.kind for s in sl.snapshots}, {"commodity"})
        self.assertEqual(len(sl.snapshots), len(self.snaps))
        self.assertTrue(all(s.meta["house"] == 110 and s.meta["market"] == "h110" for s in sl.snapshots))
        self.assertEqual((sl.kind, sl.house, sl.source, sl.files), ("commodity", 110, "forever_us_r1_h110_commodities", len(self.snaps)))

    def test_regular_auctions_are_priced_by_buyout_and_unpriced_rows_are_counted(self):
        write_house(self.root, 113, self.snaps, commodity=False)
        sl = FV.load_source(LocalStore(self.root), 113, False, *self.span, base_reader=base_reader)
        self.assertEqual({s.kind for s in sl.snapshots}, {"auction"})
        self.assertEqual(sl.unpriced_rows, 3 * len(self.snaps))
        s0 = sl.snapshots[0]
        self.assertEqual(len(s0.book), 37)
        self.assertEqual(set(s0.book["quantity"]), {1})
        src = as_fetch_v2_frame([self.snaps[0]]).head(40)
        want = sorted(src["unit_price"].iloc[3:].astype(int))
        self.assertEqual(sorted(s0.book["unit_price"]), want)

    def test_a_repeated_auction_id_no_longer_rejects_the_snapshot_and_is_counted(self):
        bad_at = self.snaps[10].ts

        def put_dup(s, f):
            return pd.concat([f, f.iloc[[5]]], ignore_index=True) if s.ts == bad_at else f

        write_house(self.root, 114, self.snaps, transform=put_dup)
        with self.assertRaisesRegex(SchemaError, "duplicate auction_id"):
            load_snapshots(LocalStore(self.root), FV.source_name(114, True), *self.span, reader=base_reader)
        sl = FV.load_source(LocalStore(self.root), 114, True, *self.span, base_reader=base_reader)
        self.assertEqual(len(sl.snapshots), len(self.snaps))
        self.assertEqual((sl.dups.duplicate_rows, sl.dups.conflicting_ids), (1, 0))
        self.assertEqual(len(sl.dups.files), 1)
        self.assertIn("h114_commodities", sl.dups.files[0][0])
        self.assertIn(f"hour_{bad_at.hour:02d}_{bad_at.minute:02d}", sl.dups.files[0][0])

    def test_copies_that_disagree_are_counted_separately(self):
        def put_dup(s, f):
            if s.ts != self.snaps[4].ts:
                return f
            d = f.iloc[[2]].copy()
            d["unit_price"] = d["unit_price"] + 7
            return pd.concat([f, d], ignore_index=True)

        write_house(self.root, 110, self.snaps, transform=put_dup)
        sl = FV.load_source(LocalStore(self.root), 110, True, *self.span, base_reader=base_reader)
        self.assertEqual((sl.dups.duplicate_rows, sl.dups.conflicting_ids), (1, 1))

    def test_files_that_came_from_the_cache_are_not_recounted_and_say_so(self):
        bad_at = self.snaps[3].ts
        write_house(self.root, 110, self.snaps, transform=lambda s, f: pd.concat([f, f.iloc[[0]]], ignore_index=True) if s.ts == bad_at else f)
        first = FV.load_source(LocalStore(self.root), 110, True, *self.span, cache_dir=self.cache, base_reader=base_reader)
        self.assertEqual((first.dups.duplicate_rows, first.files_cached), (1, 0))
        again = FV.load_source(LocalStore(self.root), 110, True, *self.span, cache_dir=self.cache, base_reader=base_reader)
        self.assertEqual((again.dups.duplicate_rows, again.files_cached, again.files), (0, len(self.snaps), len(self.snaps)))
        self.assertEqual(len(again.snapshots), len(self.snaps))

    def test_a_file_in_one_houses_folder_that_belongs_to_another_house_stops_the_load(self):
        source = FV.source_name(110, True)
        for s in self.snaps:
            f = commodity_frame(s, source)
            if s.ts == self.snaps[2].ts:
                f["source"] = FV.source_name(113, True)
            write(self.root, key_for(source, s.ts), f)
        with self.assertRaisesRegex(LoaderError, "never mixed"):
            FV.load_source(LocalStore(self.root), 110, True, *self.span, base_reader=base_reader)

    def test_a_commodity_folder_whose_rows_say_regular_auctions_stops_the_load(self):
        source = FV.source_name(110, True)
        for s in self.snaps:
            f = commodity_frame(s, source)
            f["source"] = FV.source_name(110, False)
            write(self.root, key_for(source, s.ts), f)
        with self.assertRaisesRegex(LoaderError, "never mixed"):
            FV.load_source(LocalStore(self.root), 110, True, *self.span, base_reader=base_reader)

    def test_houses_are_loaded_one_by_one_and_never_pooled(self):
        write_house(self.root, 113, self.snaps[:20])
        write_house(self.root, 110, self.snaps[:30])
        write_house(self.root, 110, self.snaps[:30], commodity=False)
        m = FV.load_houses(LocalStore(self.root), [113, 110, 110], *self.span, base_reader=base_reader)
        self.assertEqual(list(m), [110, 113])
        self.assertEqual((len(m[110].snapshots("commodity")), len(m[110].snapshots("auction"))), (30, 30))
        self.assertEqual((len(m[113].snapshots("commodity")), len(m[113].snapshots("auction"))), (20, 0))
        self.assertTrue(all(s.meta["house"] == 113 for s in m[113].snapshots("commodity")))

    def test_kinds_can_be_limited_and_bad_ones_are_refused(self):
        write_house(self.root, 110, self.snaps[:10])
        h = FV.load_house(LocalStore(self.root), 110, *self.span, kinds=("commodity",), base_reader=base_reader)
        self.assertIsNotNone(h.commodity)
        self.assertIsNone(h.auction)
        self.assertEqual(h.snapshots("auction"), [])
        with self.assertRaises(ValueError):
            FV.load_house(LocalStore(self.root), 110, *self.span, kinds=("bids",), base_reader=base_reader)
        with self.assertRaises(ValueError):
            h.snapshots("bids")

    def test_plan_counts_files_before_anything_is_read(self):
        write_house(self.root, 110, self.snaps[:12])
        plans = FV.plan_houses(LocalStore(self.root), [110, 113], *self.span)
        got = {(h, k): len(p.files) for h, k, p in plans}
        self.assertEqual(got, {(110, "commodity"): 12, (110, "auction"): 0, (113, "commodity"): 0, (113, "auction"): 0})

    def test_houses_in_the_sources_file(self):
        self.sources_file(houses=(116, 110, 112))
        self.assertEqual(FV.houses_in_store(LocalStore(self.root)), [110, 112, 116])

    def test_a_missing_or_broken_sources_file_gives_no_houses(self):
        self.assertEqual(FV.houses_in_store(LocalStore(self.root)), [])
        write(self.root, FV.SOURCES_KEY, b"{not json")
        self.assertEqual(FV.houses_in_store(LocalStore(self.root)), [])


class Flaky(Store):
    name = "flaky"

    def __init__(self, fail, exc):
        self.fail, self.exc, self.calls = fail, exc, 0

    def list(self, prefix):
        self.calls += 1
        if self.calls <= self.fail:
            raise self.exc
        return [ObjectInfo("a", 1)]

    def read_range(self, key, start, end):
        self.calls += 1
        if self.calls <= self.fail:
            raise self.exc
        return b"x"


class Http(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.response = type("R", (), {"status_code": code})()


class TestRetryStore(unittest.TestCase):
    def test_transient_trouble_is_retried_with_growing_pauses(self):
        pauses = []
        for exc in (ConnectionError("reset"), Http(429), Http(503)):
            pauses.clear()
            st = FV.RetryStore(Flaky(2, exc), attempts=4, delay=3.0, sleep=pauses.append)
            self.assertEqual(st.read_range("k", 0, 1), b"x")
            self.assertEqual(pauses, [3.0, 6.0])

    def test_a_real_error_is_not_retried(self):
        for exc in (ValueError("bug"), Http(403), Http(401)):
            inner = Flaky(5, exc)
            st = FV.RetryStore(inner, sleep=lambda s: self.fail("should not wait"))
            with self.assertRaises(type(exc)):
                st.read_range("k", 0, 1)
            self.assertEqual(inner.calls, 1)

    def test_gives_up_after_the_last_attempt(self):
        inner = Flaky(99, Http(503))
        st = FV.RetryStore(inner, attempts=3, sleep=lambda s: None)
        with self.assertRaises(Http):
            st.list("p")
        self.assertEqual(inner.calls, 3)

    def test_a_folder_that_does_not_exist_lists_as_empty_but_a_missing_file_is_an_error(self):
        st = FV.RetryStore(Flaky(9, Http(404)), sleep=lambda s: None)
        self.assertEqual(st.list("forever_us_r1_h112/"), [])
        with self.assertRaises(Http):
            st.read_range("k", 0, 1)

    def test_it_keeps_the_inner_stores_name(self):
        self.assertEqual(FV.RetryStore(Flaky(0, ValueError())).name, "flaky")


class TestHourly(Lake):
    def markets(self, houses=(110, 113)):
        for h in houses:
            write_house(self.root, h, self.snaps)
        return FV.load_houses(LocalStore(self.root), houses, *self.span, kinds=("commodity",), base_reader=base_reader)

    def test_same_numbers_as_the_retail_history_with_the_house_in_front(self):
        m = self.markets((110,))
        hl = FV.hourly_for_house(m[110].snapshots("commodity"), 110)
        ids = {int(i) for i in self.snaps[0].book["item_id"].unique()}
        for s in self.snaps:
            ids.update(int(i) for i in s.book["item_id"].unique())
        snaps = sorted(self.snaps, key=lambda s: s.ts)
        want = D.trim_edges(D.hourly_long(D.build_hourly(snaps, classify_series(snaps), ids), snaps, ids), snaps[0].ts, snaps[-1].ts)
        self.assertEqual(list(hl.columns), ["house"] + D.HOURLY_COLUMNS)
        self.assertEqual(set(hl["house"]), {110})
        self.assertGreater(len(hl), 0)
        pd.testing.assert_frame_equal(hl.drop(columns="house").reset_index(drop=True), want.reset_index(drop=True), check_dtype=False)

    def test_edges_are_trimmed_like_the_retail_history(self):
        m = self.markets((110,))
        hl = FV.hourly_for_house(m[110].snapshots("commodity"), 110)
        self.assertGreaterEqual(hl["ts"].min(), self.snaps[0].ts + pd.Timedelta(hours=12))
        self.assertLessEqual(hl["ts"].max(), self.snaps[-1].ts - pd.Timedelta(hours=12))

    def test_every_row_of_a_many_house_table_carries_its_house(self):
        m = self.markets((110, 113))
        hl = FV.hourly_by_house(m)
        self.assertEqual(set(hl["house"]), {110, 113})
        self.assertEqual(len(hl[hl["house"] == 110]), len(hl[hl["house"] == 113]))
        self.assertFalse(hl.duplicated(["house", "item_id", "ts"]).any())

    def test_skipping_the_classifier_keeps_the_floors_and_leaves_the_two_flags_false(self):
        m = self.markets((110,))
        slow = FV.hourly_for_house(m[110].snapshots("commodity"), 110)
        fast = FV.hourly_for_house(m[110].snapshots("commodity"), 110, classify=False)
        self.assertEqual(len(slow), len(fast))
        np.testing.assert_allclose(slow["floor_gold"].to_numpy(), fast["floor_gold"].to_numpy())
        self.assertFalse(fast["flag_wipe"].any() or fast["flag_long_gap"].any())

    def test_item_list_limits_the_table(self):
        m = self.markets((110,))
        item = int(self.snaps[0].book["item_id"].iloc[0])
        hl = FV.hourly_for_house(m[110].snapshots("commodity"), 110, ids=[item])
        self.assertEqual(set(hl["item_id"]), {item})

    def test_too_little_data_gives_an_empty_table_with_the_right_columns(self):
        write_house(self.root, 110, self.snaps[:2])
        m = FV.load_houses(LocalStore(self.root), [110], *self.span, kinds=("commodity",), base_reader=base_reader)
        hl = FV.hourly_by_house(m)
        self.assertEqual((len(hl), list(hl.columns)), (0, ["house"] + D.HOURLY_COLUMNS))

    def test_a_snapshot_of_another_house_is_refused(self):
        m = self.markets((113,))
        with self.assertRaisesRegex(ValueError, "never mixed"):
            FV.hourly_for_house(m[113].snapshots("commodity"), 110)

    def test_regular_auction_snapshots_are_not_used_for_the_table(self):
        write_house(self.root, 110, self.snaps, commodity=False)
        m = FV.load_houses(LocalStore(self.root), [110], *self.span, kinds=("auction",), base_reader=base_reader)
        self.assertEqual(len(FV.hourly_by_house(m)), 0)

    def test_gaps_are_found_from_snapshot_times(self):
        snaps = [s for i, s in enumerate(self.snaps[:20]) if i not in (7, 8)]
        g = FV.snapshot_gaps(snaps)
        self.assertEqual(len(g), 1)
        self.assertAlmostEqual(g[0][2], 3.0, delta=0.1)                          # the simulator's snapshots are about 59 minutes apart, not exactly 60
        self.assertEqual(FV.snapshot_gaps(self.snaps[:20]), [])


def args(**kw):
    d = dict(houses="", days=3, end="", kinds="both", tables=False, fast=False, max_files=1500, max_mb=800, cache=None, out="x")
    d.update(kw)
    return type("A", (), d)()


class TestCheckTool(Lake):
    def setUp(self):
        super().setUp()
        self.sources_file()
        self.now = self.snaps[-1].ts + pd.Timedelta(minutes=30)
        for h in (110, 113):
            write_house(self.root, h, self.snaps)
            write_house(self.root, h, self.snaps, commodity=False)

    def go(self, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return A.run(args(**kw), LocalStore(self.root), now=self.now, base_reader=base_reader)

    def test_a_normal_run_reports_every_section_and_stays_quiet_about_the_account(self):
        r = self.go(houses="110,113", days=3)
        self.assertEqual(r["code"], 0)
        t = r["text"]
        for head in ("### forever adapter check", "### 1 what was read", "### 2 kind check", "### 3 repeated auction ids", "### 4 gaps"):
            self.assertIn(head, t)
        self.assertNotIn("### 5", t)
        self.assertNotIn("bosco277", t)
        self.assertFalse(any(line.startswith("=") for line in t.splitlines()))
        self.assertIn(f"commodity snapshots: {2 * len(self.snaps)}", t)
        self.assertIn("Snapshots whose kind or house disagreed with what was asked for: 0", t)
        self.assertIn("None in this window.", t)

    def test_the_default_is_the_pvp_set_and_houses_without_files_are_named(self):
        r = self.go(houses="")
        self.assertEqual(r["code"], 0)
        self.assertIn("houses: 110, 113, 116", r["text"])
        self.assertIn("houses with no files in this window: 116", r["text"])
        self.assertEqual(sorted(r["markets"]), [110, 113])

    def test_all_means_every_house_in_the_sources_file(self):
        r = self.go(houses="all")
        self.assertIn("houses: 110, 112, 113, 116", r["text"])
        self.assertIn("houses with no files in this window: 112, 116", r["text"])

    def test_repeated_ids_show_up_with_their_files(self):
        at = self.snaps[6].ts
        write_house(self.root, 110, self.snaps, transform=lambda s, f: pd.concat([f, f.iloc[[1]]], ignore_index=True) if s.ts == at else f)
        t = self.go(houses="110")["text"]
        self.assertIn("Repeated auction ids dropped: 1 rows", t)
        self.assertIn("Every repeated id was an exact copy", t)
        self.assertIn(f"forever_us_r1_h110_commodities/year={at.year}", t)

    def test_gaps_in_the_window_are_listed(self):
        gone = {self.snaps[9].ts, self.snaps[10].ts}
        write_house(self.root, 110, [s for s in self.snaps if s.ts not in gone])
        for s in gone:
            os.remove(os.path.join(self.root, key_for(FV.source_name(110, True), s)))
        t = self.go(houses="110", kinds="commodity")["text"]
        self.assertNotIn("No stretch of more than 90 minutes", t)
        self.assertIn("3.0", t)

    def test_it_stops_before_downloading_when_the_request_is_too_big(self):
        def boom(fileobj, ids):
            raise AssertionError("nothing should be downloaded")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = A.run(args(houses="110,113", max_files=10), LocalStore(self.root), now=self.now, base_reader=boom)
        self.assertEqual(r["code"], 2)
        self.assertIn("STOPPED before downloading", r["text"])
        self.assertNotIn("### 1 what was read", r["text"])
        with contextlib.redirect_stdout(out):
            r = A.run(args(houses="110,113", max_mb=0), LocalStore(self.root), now=self.now, base_reader=boom)
        self.assertEqual(r["code"], 2)

    def test_an_unreachable_dataset_is_said_plainly(self):
        os.remove(os.path.join(self.root, FV.SOURCES_KEY))
        r = self.go()
        self.assertEqual(r["code"], 1)
        self.assertIn("empty or unreachable", r["text"])

    def test_an_empty_window_is_not_a_success(self):
        r = self.go(houses="110", end="2020-01-02", days=1)
        self.assertEqual(r["code"], 1)
        self.assertIn("Nothing to read", r["text"])

    def test_the_hourly_table_is_described_only_when_asked(self):
        r = self.go(houses="110,113", kinds="commodity", tables=True, days=3)
        self.assertIn("### 5 hourly table", r["text"])
        self.assertEqual(set(r["hourly"]["house"]), {110, 113})
        t = r["text"].split("### 5 hourly table")[1]
        self.assertIn("110", t)
        self.assertIn("113", t)

    def test_unreadable_files_are_reported_and_do_not_end_the_check(self):
        bad_at = self.snaps[5].ts
        write(self.root, key_for(FV.source_name(110, True), bad_at), b"not a pickle")
        r = self.go(houses="110", kinds="commodity")
        self.assertEqual(r["code"], 0)
        self.assertIn("unreadable: forever_us_r1_h110_commodities/year=", r["text"])

    def test_window_is_the_last_n_utc_days_ending_at_the_end_date(self):
        s, e = A.window(args(days=3, end="2026-11-25"), pd.Timestamp("2026-12-01"))
        self.assertEqual((str(s), str(e)), ("2026-11-23", "2026-11-25"))
        s, e = A.window(args(days=2), pd.Timestamp("2026-11-25 14:00"))
        self.assertEqual((str(s), str(e)), ("2026-11-24", "2026-11-25"))


class TestMain(unittest.TestCase):
    def test_no_token_is_a_clear_stop(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(out):
            os.environ.pop("HF_TOKEN", None)
            self.assertEqual(A.main(["--out", tempfile.gettempdir()]), 1)
        self.assertIn("HF_TOKEN", out.getvalue())

    def test_an_error_that_carries_the_dataset_address_is_blanked_before_it_is_printed(self):
        class Boom:
            def __init__(self, *a, **k):
                pass

            def list(self, prefix):
                raise RuntimeError("404 Client Error: https://huggingface.co/api/datasets/bosco277/wow-forever-data-lake/tree/main/system")

        out = io.StringIO()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "x"}), mock.patch.object(A, "HFStore", Boom), contextlib.redirect_stdout(out):
            self.assertEqual(A.main(["--out", tempfile.gettempdir()]), 1)
        self.assertIn("forever_adapter failed", out.getvalue())
        self.assertNotIn("bosco277", out.getvalue())
        self.assertNotIn("wow-forever-data-lake", out.getvalue())


class TestWorkflow(unittest.TestCase):
    def wf(self, path):
        with open(os.path.join(_root, path), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_manual_only_and_read_only(self):
        for path in (".github/workflows/forever_adapter.yml", "tools/forever_adapter.yml"):
            d = self.wf(path)
            on = d.get("on") or d.get(True)
            self.assertEqual(set(on), {"workflow_dispatch"})
            self.assertEqual(set(on["workflow_dispatch"]["inputs"]), {"houses", "days", "tables"})
            text = json.dumps(d)
            self.assertNotIn("upload", text.replace("upload-artifact", ""))
            self.assertNotIn("contents: write", text)
            step = next(s for s in d["jobs"]["adapter"]["steps"] if s.get("name") == "Forever adapter check")
            self.assertEqual(set(step["env"]), {"HF_TOKEN", "HOUSES", "DAYS", "TABLES"})
            self.assertIn("secrets.HF_TOKEN", step["env"]["HF_TOKEN"])
            self.assertIn("set -o pipefail", step["run"])
            self.assertIn("tools/forever_adapter.py", step["run"])

    def test_the_copy_in_tools_is_identical(self):
        with open(os.path.join(_root, ".github/workflows/forever_adapter.yml"), encoding="utf-8") as a, open(os.path.join(_root, "tools/forever_adapter.yml"), encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())

    def test_it_installs_what_the_reader_needs(self):
        d = self.wf(".github/workflows/forever_adapter.yml")
        install = next(s for s in d["jobs"]["adapter"]["steps"] if s.get("name") == "Install")["run"]
        for pkg in ("pandas", "numpy", "pyarrow", "requests"):
            self.assertIn(pkg, install)


if __name__ == "__main__":
    unittest.main()
