"""tools/forever_health.py on a pretend Forever dataset (nothing leaves the machine): file-name parsing, gaps and complete days, growth, the engine-reader test (including the known catch
that a '_commodities' source is read as 'auction'), shared auction ids, price comparison between houses, and that no account name reaches the report. Plus wiring checks on the workflow."""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
import fetch_forever as F  # noqa: E402
import forever_health as H  # noqa: E402

START = pd.Timestamp("2026-11-13 04:25")           # UTC; 11:25 PM ET on Nov 12


def key(source, ts):
    return f"{source}/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/hour_{ts.hour:02d}_{ts.minute:02d}.parquet"


def commodity_rows(rng, items=40, units=200, scale=1.0, base=None, noise=0.02, ids_from=0):
    base = base if base is not None else 5000.0 + 1000.0 * np.arange(items)
    rows = []
    for i in range(items):
        for j in range(3):                                           # three listings per item; the cheapest sets the floor
            rows.append({"id": ids_from + i * 10 + j, "item": {"id": 2000 + i}, "quantity": units, "unit_price": int(base[i] * scale * (1 + 0.02 * j) * np.exp(rng.normal(0, noise))), "time_left": "LONG"})
    return rows


def regular_rows(rng, n=30, ids_from=500000, bid_only=2):
    rows = [{"id": ids_from + i, "item": {"id": 3000 + i % 7, "bonus_lists": [1, 2]} if i % 3 else {"id": 3000 + i % 7}, "quantity": 1, "buyout": int(rng.integers(10000, 90000)), "bid": 100,
             "time_left": "MEDIUM"} for i in range(n)]
    for k in range(bid_only):
        rows[k].pop("buyout")                                         # bid-only auctions carry no buyout at all
    return rows


class World:
    """An in-memory Forever dataset. `separate` houses move on their own; the others copy house 0's prices (a shared market)."""

    def __init__(self, days=3, houses=(110, 113, 116), skip=(), separate=(113,), copy_ids=(), scale=None, seed=1):
        self.rng = np.random.default_rng(seed)
        self.frames, self.entries, self.sizes = {}, [], {}
        self.houses = houses
        runs = [START + pd.Timedelta(hours=h) for h in range(days * 24) if h not in set(skip)]
        self.runs = runs
        base = 5000.0 + 1000.0 * np.arange(40)
        hourly = {t: np.exp(self.rng.normal(0, 0.03, 40)) for t in runs}                              # the shared market's hourly moves
        for hid in houses:
            for kind in ("auctions", "commodities"):
                source = f"forever_us_r1_h{hid}" + ("_commodities" if kind == "commodities" else "")
                for t in runs:
                    if kind == "commodities":
                        move = np.exp(self.rng.normal(0, 0.03, 40)) if hid in separate else hourly[t]
                        rows = commodity_rows(self.rng, base=base * move, scale=(scale or {}).get(hid, 1.0), noise=0.0, ids_from=1000000 * (hid if hid not in copy_ids else 110))
                    else:
                        rows = regular_rows(self.rng, ids_from=500000 * hid)
                    snap = t - pd.Timedelta(minutes=3)
                    df = F.build_frame(rows, source, snap.to_pydatetime(), t.to_pydatetime())
                    k = key(source, t)
                    self.frames[k] = df
                    self.entries.append((k, 4000 + len(df) * 50))
        self.entries.append(("system/forever_markers.json", 300))
        self.entries.append((".gitattributes", 100))
        self.json = {H.SOURCES_KEY: {s: dict(region="us", ruleset=1, house_id=h, house_name=f"House {h}", kind=("commodities" if s.endswith("_commodities") else "auctions"))
                                     for h in houses for s in (f"forever_us_r1_h{h}", f"forever_us_r1_h{h}_commodities")},
                     H.MARKERS_KEY: {"_ruleset_check": {"house": 110, "same": [2, 3], "differ": [], "unserved": [15]}}}
        self.fail = set()

    def list_entries(self):
        return list(self.entries)

    def read_json(self, k):
        return self.json.get(k, {})

    def read_frame(self, k, columns=None):
        if k in self.fail:
            raise OSError("boom bosco277/wow-forever-data-lake")
        df = self.frames[k]
        return df[columns] if columns else df


def args(**kw):
    d = dict(days=0, compare_hours=24, max_mb=800, out="x")
    d.update(kw)
    return type("A", (), d)()


class TestInventory(unittest.TestCase):
    def test_parse_key(self):
        self.assertEqual(H.parse_key("forever_us_r1_h110_commodities/year=2026/month=11/day=25/hour_14_25.parquet"), ("forever_us_r1_h110_commodities", pd.Timestamp("2026-11-25 14:25")))
        for bad in ("system/forever_markers.json", "forever_us_r1_h110/year=2026/month=11/day=25/", "x/year=2026/month=11/day=25/hour_14_25.csv", ".gitattributes"):
            self.assertIsNone(H.parse_key(bad))

    def test_kind_and_house(self):
        self.assertEqual(H.kind_of("forever_us_r1_h110_commodities"), "commodities")
        self.assertEqual(H.kind_of("forever_us_r1_h110"), "auctions")
        self.assertEqual(H.house_of("forever_us_r1_h110_commodities"), 110)
        self.assertIsNone(H.house_of("commodities"))

    def test_inventory_splits_snapshot_files_from_the_rest(self):
        w = World(days=1)
        inv, other = H.inventory(w.entries)
        self.assertEqual(len(inv), 24 * 6)
        self.assertEqual(sorted(p for p, _ in other), [".gitattributes", "system/forever_markers.json"])
        self.assertTrue(inv["ts"].is_monotonic_increasing)


class TestGapsAndDays(unittest.TestCase):
    def test_a_missing_hour_is_a_gap_and_counted(self):
        t = [START + pd.Timedelta(hours=h) for h in range(10) if h not in (4,)]
        g = H.gaps_in(t)
        self.assertEqual(len(g), 1)
        self.assertAlmostEqual(g[0][2], 2.0)
        self.assertEqual(H.missed_hours(g), 1)
        t2 = [START + pd.Timedelta(hours=h) for h in range(12) if h not in (4, 5, 6)]
        self.assertEqual(H.missed_hours(H.gaps_in(t2)), 3)                       # a 4-hour gap hides three hourly refreshes
        self.assertEqual(H.gaps_in([START + pd.Timedelta(minutes=70 * i) for i in range(5)]), [])    # 70 minutes between files is normal jitter, not a gap

    def test_coverage_per_source(self):
        w = World(days=2, skip=(10, 11, 30))
        inv, _ = H.inventory(w.entries)
        cov = H.coverage(inv, START + pd.Timedelta(days=2, hours=3)).set_index("source")
        x = cov.loc["forever_us_r1_h110_commodities"]
        self.assertEqual(int(x["files"]), 48 - 3)
        self.assertEqual(int(x["missed"]), 3)                                    # a 3-hour gap (two missing runs counted as 2) and a single miss (1)
        self.assertAlmostEqual(x["coverage"], 45 / 48)
        self.assertAlmostEqual(x["longest_gap_h"], 3.0)
        self.assertAlmostEqual(x["age_h"], (START + pd.Timedelta(days=2, hours=3) - (START + pd.Timedelta(hours=47))).total_seconds() / 3600)

    def test_days_are_complete_only_without_gaps_and_partial_ends_are_marked(self):
        w = World(days=4, skip=(40,))                                           # START is Nov 13 04:25 UTC; hour 40 falls on Nov 14 evening ET
        inv, _ = H.inventory(w.entries)
        days = H.day_table(inv["ts"], START + pd.Timedelta(days=5)).set_index("day")
        self.assertEqual(days.loc[pd.Timestamp("2026-11-12").date(), "note"], "data starts during this day")
        self.assertFalse(bool(days.loc[pd.Timestamp("2026-11-14").date(), "complete"]))
        self.assertGreater(days.loc[pd.Timestamp("2026-11-14").date(), "longest_gap_h"], 1.5)
        self.assertTrue(bool(days.loc[pd.Timestamp("2026-11-15").date(), "complete"]))
        self.assertEqual(int(days.loc[pd.Timestamp("2026-11-15").date(), "runs"]), 24)
        self.assertNotEqual(days["note"].iloc[-1], "")                           # the last day is not counted: the data stops inside it
        self.assertEqual(H.day_table(inv["ts"], START + pd.Timedelta(days=3, hours=20))["note"].iloc[-1], "day not over yet")
        self.assertEqual(days["note"].iloc[-1], "data ends during this day")

    def test_a_day_with_a_dst_change_still_has_24_or_25_hours_of_files(self):
        t = [pd.Timestamp("2026-10-31 04:25") + pd.Timedelta(hours=h) for h in range(24 * 5)]
        d = H.day_table(t, pd.Timestamp("2026-11-10")).set_index("day")
        self.assertTrue(bool(d.loc[pd.Timestamp("2026-11-01").date(), "complete"]))      # Nov 1 has 25 hours in New York
        self.assertEqual(int(d.loc[pd.Timestamp("2026-11-01").date(), "runs"]), 25)


class TestGrowth(unittest.TestCase):
    def test_rates_and_days_left(self):
        rows = [("forever_us_r1_h110", START + pd.Timedelta(hours=h), 1_000_000, "k") for h in range(48)]
        rows += [("forever_us_r1_h999", START + pd.Timedelta(hours=h), 3_000_000, "k") for h in range(48)]
        inv = pd.DataFrame(rows, columns=["source", "ts", "size", "key"])
        g = H.growth(inv, START + pd.Timedelta(hours=48))
        a = g["all"]
        self.assertEqual(a["total_files"], 96)
        self.assertAlmostEqual(a["total_mb"], 48 * 4.0)
        self.assertAlmostEqual(a["files_per_day"], 96 / (47 / 24), places=3)
        self.assertAlmostEqual(a["days_to_files"], (100_000 - 96) / a["files_per_day"], places=3)
        self.assertAlmostEqual(a["days_to_storage"], (100_000 - 192) / a["mb_per_day"], places=3)
        self.assertEqual(g["pvp"]["total_files"], 48)                            # only house 110 is in the PvP set
        self.assertEqual(list(g["per_source"].index), ["forever_us_r1_h999", "forever_us_r1_h110"])    # biggest first


class TestSchema(unittest.TestCase):
    def frame(self, source, rows):
        return F.build_frame(rows, source, pd.Timestamp("2026-11-25 14:22").to_pydatetime(), pd.Timestamp("2026-11-25 14:25").to_pydatetime())

    def test_a_commodity_file_reads_but_the_engine_calls_it_an_auction(self):
        # THE KNOWN CATCH (docs 2u): engine/schema.py marks a snapshot 'commodity' only when the source is named exactly 'commodities'. When the adapter fixes that, update this test.
        rng = np.random.default_rng(1)
        s = H.schema_report(self.frame("forever_us_r1_h110_commodities", commodity_rows(rng)), "forever_us_r1_h110_commodities")
        self.assertEqual(s["kind"], "commodities")
        self.assertEqual((s["missing"], s["extra"]), ([], []))
        self.assertTrue(s["engine_ok"])
        self.assertEqual(s["engine_kind"], "auction")
        self.assertFalse(s["kind_ok"])
        self.assertEqual(s["engine_dropped"], 0)
        self.assertEqual(s["schema_versions"], [2])
        self.assertAlmostEqual(s["lag_min"], 3.0)

    def test_a_plain_commodities_source_would_be_fine(self):
        rng = np.random.default_rng(1)
        s = H.schema_report(self.frame("commodities", commodity_rows(rng)), "commodities")
        self.assertEqual(s["kind"], "auctions")                                  # by the Forever naming rule, the plain name is not a commodities source: kind_of looks at the suffix
        self.assertTrue(s["engine_ok"])

    def test_regular_auctions_bid_only_rows_are_counted_and_dropped_by_the_engine(self):
        rng = np.random.default_rng(1)
        s = H.schema_report(self.frame("forever_us_r1_h110", regular_rows(rng, n=30, bid_only=3)), "forever_us_r1_h110")
        self.assertEqual(s["kind"], "auctions")
        self.assertAlmostEqual(s["bid_only"], 3 / 30)
        self.assertEqual(s["engine_dropped"], 3)
        self.assertTrue(s["engine_ok"] and s["kind_ok"])
        self.assertAlmostEqual(s["null"]["buyout"], 0.1)
        self.assertGreater(s["variant_share"], 0.3)                              # gear-style variants have a hash in variant_key
        self.assertEqual(s["time_left"], {"MEDIUM": 30})
        self.assertEqual(s["dup_ids"], 0)

    def test_a_file_missing_columns_is_reported_and_the_engine_error_is_kept(self):
        rng = np.random.default_rng(1)
        df = self.frame("forever_us_r1_h110_commodities", commodity_rows(rng)).drop(columns=["time_left", "unit_price"])
        s = H.schema_report(df, "forever_us_r1_h110_commodities")
        self.assertEqual(sorted(s["missing"]), ["time_left", "unit_price"])
        self.assertFalse(s["engine_ok"])
        self.assertTrue(s["engine_error"])

    def test_extra_columns_are_listed_and_repeated_ids_counted(self):
        rng = np.random.default_rng(1)
        df = self.frame("forever_us_r1_h110", regular_rows(rng, n=10))
        df["mystery"] = 1
        df.loc[1, "id"] = df.loc[0, "id"]
        s = H.schema_report(df, "forever_us_r1_h110")
        self.assertEqual(s["extra"], ["mystery"])
        self.assertEqual(s["dup_ids"], 1)


class TestMarkets(unittest.TestCase):
    def test_id_overlap(self):
        o = H.id_overlap({"a": {1, 2, 3, 4}, "b": {1, 2, 3, 4}, "c": {9, 10}})
        x = o.set_index(["a", "b"])
        self.assertEqual(x.loc[("a", "b"), "share_of_smaller"], 1.0)
        self.assertEqual(x.loc[("a", "c"), "share_of_smaller"], 0.0)
        self.assertEqual(H.id_overlap({"a": set(), "b": {1}}).iloc[0]["shared"], 0)

    def pan(self, **kw):
        w = World(days=2, **kw)
        inv, _ = H.inventory(w.entries)
        frames = {}
        for _, x in inv[inv["source"].str.endswith("_commodities")].iterrows():
            frames.setdefault(x["source"], []).append((x["ts"], w.read_frame(x["key"], H.COMPARE_COLS)))
        return H.panels(frames)

    def test_floors_and_units_per_item(self):
        rng = np.random.default_rng(1)
        df = F.build_frame(commodity_rows(rng, items=3, units=50, noise=0.0), "forever_us_r1_h110_commodities", None, pd.Timestamp("2026-11-25 14:25").to_pydatetime())
        f, u = H.floors_units(df)
        self.assertAlmostEqual(f[2000], 0.5)                                       # 5000 copper cheapest of three listings
        self.assertEqual(u[2000], 150)                                              # 3 listings x 50 units

    def test_a_shared_market_shows_ratio_one_and_high_correlation_a_separate_one_does_not(self):
        pan = self.pan(separate=(113,))
        pc = H.price_compare(pan).set_index(["a", "b"])
        shared = pc.loc[("forever_us_r1_h110_commodities", "forever_us_r1_h116_commodities")]
        apart = pc.loc[("forever_us_r1_h110_commodities", "forever_us_r1_h113_commodities")]
        self.assertGreater(shared["items"], 30)
        self.assertAlmostEqual(shared["ratio_med"], 1.0, places=3)
        self.assertGreater(shared["identical"], 0.95)
        self.assertGreater(shared["corr_level"], 0.99)
        self.assertLess(abs(apart["corr_change"]), 0.4)                             # independent hourly noise
        self.assertLess(apart["identical"], 0.1)

    def test_a_dearer_house_shows_up_as_a_price_ratio(self):
        pan = self.pan(separate=(), scale={116: 2.0})
        pc = H.price_compare(pan).set_index(["a", "b"])
        x = pc.loc[("forever_us_r1_h110_commodities", "forever_us_r1_h116_commodities")]
        self.assertAlmostEqual(x["ratio_med"], 2.0, places=2)
        self.assertLess(x["ratio_hi"] - x["ratio_lo"], 0.05)
        self.assertAlmostEqual(x["corr_level"], 1.0, places=2)

    def test_thin_items_are_left_out(self):
        pan = self.pan(separate=())
        thin = {s: (f, u * 0.0 + 5) for s, (f, u) in pan.items()}
        self.assertEqual(int(H.price_compare(thin).iloc[0]["items"]), 0)             # below the 20-unit bar in both houses

    def test_item_prices_and_timing(self):
        w = World(days=1)
        inv, _ = H.inventory(w.entries)
        frames = {}
        for _, x in inv[inv["source"].str.endswith("_commodities")].iterrows():
            frames.setdefault(x["source"], []).append((x["ts"], w.read_frame(x["key"], H.COMPARE_COLS)))
        ip = H.item_prices(H.panels(frames), top=5)
        self.assertEqual(len(ip), 5)
        self.assertIn("forever_us_r1_h113_commodities|price", ip.columns)
        tm = H.timing_stats(frames).set_index("source")
        self.assertEqual(int(tm.loc["forever_us_r1_h110_commodities", "files"]), 24)
        self.assertEqual(int(tm.loc["forever_us_r1_h110_commodities", "distinct"]), 24)
        self.assertAlmostEqual(tm.loc["forever_us_r1_h110_commodities", "lag_med"], 3.0)


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World(days=4, skip=(30, 31, 32))
        cls.r = H.run(args(compare_hours=48), cls.w, now=START + pd.Timedelta(days=4, hours=1), names={2000: "Linen Cloth"})
        cls.text = cls.r["text"]

    def test_report_has_every_section_and_the_readout(self):
        for part in ("### FOREVER HEALTH CHECK", "### 1. WHAT IS BEING COLLECTED", "### 2. DO WE HAVE EVERY HOUR?", "### 3. SIZE AND GROWTH", "### 4. HOW THE DATA IS STORED",
                     "### 5. ARE THE HOUSES SEPARATE MARKETS?", "### 6. SNAPSHOT TIMING", "[OK", "LIFT AND SHIFT"):
            self.assertIn(part, self.text)
        self.assertFalse(any(line.startswith("=") for line in self.text.splitlines()))     # a spreadsheet would read a leading = as a formula

    def test_the_three_missing_runs_show_up_as_a_gap_and_a_broken_day(self):
        self.assertEqual(self.r["union"]["n_gaps"], 1)
        self.assertEqual(self.r["union"]["missed"], 3)
        self.assertIn("gap   4.0 h", self.text)
        self.assertEqual(int((~self.r["days"]["complete"]).sum()), len(self.r["days"]) - int(self.r["days"]["complete"].sum()))
        self.assertTrue(any("WATCH" in line and "hourly refreshes" in line for line in self.text.splitlines()) or any("OK" in line and "hourly refreshes" in line for line in self.text.splitlines()))

    def test_sources_are_listed_by_house_with_counts(self):
        st = self.r["sources_table"].set_index("house")
        self.assertEqual(list(st.index), [110, 113, 116])
        self.assertEqual(int(st.loc[110, "listings_commodity"]), 120)
        self.assertEqual(int(st.loc[110, "listings_regular"]), 30)
        self.assertIn("House 110", self.text)
        self.assertIn('"same": [2, 3]', self.text)                                   # the stored ruleset check, as the fetcher wrote it

    def test_engine_test_finds_the_commodity_naming_catch_in_every_commodity_source(self):
        self.assertEqual(sum(1 for s in self.r["schema"] if s["kind"] == "commodities" and not s["kind_ok"]), 3)
        self.assertTrue(all(s["engine_ok"] for s in self.r["schema"]))
        self.assertIn("3 of 3 commodity sources would be called 'auction'", self.text)

    def test_markets_section_has_overlap_ratios_and_names(self):
        self.assertEqual(len(self.r["overlap"]), 3)
        self.assertTrue((self.r["overlap"]["share_of_smaller"] == 0.0).all())          # the pretend houses have different auction ids
        self.assertIn("Linen Cloth", self.text)
        self.assertIn("price ratio B/A", self.text)
        pc = self.r["compare"].set_index(["a", "b"])
        self.assertGreater(pc.loc[("forever_us_r1_h110_commodities", "forever_us_r1_h116_commodities"), "corr_level"], 0.99)

    def test_growth_and_pvp_set_are_in_the_report(self):
        self.assertIn("Only the PvP set (houses 110, 113, 116)", self.text)
        self.assertIn("100,000-file guideline", self.text)

    def test_no_account_name_reaches_the_report_even_when_a_read_fails(self):
        w = World(days=2)
        bad = next(k for k in w.frames if "h116_commodities" in k and "hour_10_25" in k)
        w.fail = {bad}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = H.run(args(compare_hours=48), w, now=START + pd.Timedelta(days=2, hours=1))
        self.assertNotIn("bosco277", r["text"] + buf.getvalue())
        self.assertNotIn("wow-forever-data-lake", r["text"] + buf.getvalue())
        self.assertIn("<forever dataset>", buf.getvalue())

    def test_an_unreadable_newest_file_is_a_finding_not_a_crash(self):
        w = World(days=1)
        newest = max((k for k in w.frames if "forever_us_r1_h113/" in k), key=lambda k: k)
        w.fail = {newest}
        r = H.run(args(compare_hours=0), w, now=START + pd.Timedelta(days=1, hours=1))
        s = next(x for x in r["schema"] if x["source"] == "forever_us_r1_h113")
        self.assertFalse(s["engine_ok"])
        self.assertIn("<forever dataset>", s["engine_error"])
        self.assertIn("PROBLEM", r["text"])

    def test_compare_hours_zero_skips_the_download_and_max_mb_shortens_it(self):
        w = World(days=2)
        reads = []
        orig = w.read_frame
        w.read_frame = lambda k, columns=None: (reads.append(k), orig(k, columns))[1]
        r0 = H.run(args(compare_hours=0), w, now=START + pd.Timedelta(days=2, hours=1))
        self.assertEqual(len(reads), 6)                                                  # only each source's newest file
        self.assertTrue(r0["compare"].empty)
        reads.clear()
        r1 = H.run(args(compare_hours=48, max_mb=0.05), w, now=START + pd.Timedelta(days=2, hours=1))
        self.assertLess(len(reads), 6 + 3 * 48)                                          # the window was cut to fit the cap
        self.assertIn("MB read", r1["compare_note"])

    def test_empty_dataset(self):
        class Empty:
            def list_entries(self):
                return [("system/forever_markers.json", 10)]
        r = H.run(args(), Empty(), now=START)
        self.assertIn("No snapshot files found", r["text"])

    def test_a_stopped_collector_is_flagged(self):
        w = World(days=2)
        r = H.run(args(compare_hours=0), w, now=START + pd.Timedelta(days=2, hours=9))
        self.assertIn("PROBLEM", [line.split("]")[0].strip("[ ") for line in r["text"].splitlines() if "Newest file is" in line][0])
        self.assertIn("collector may have stopped", r["text"])


class TestScrub(unittest.TestCase):
    def test_dataset_and_owner_are_blanked(self):
        t = H.scrub(f"HTTP 401 for https://huggingface.co/datasets/{H.HF_REPO}/resolve/main/x ({H.HF_REPO.split('/')[0]})")
        self.assertNotIn(H.HF_REPO.split("/")[0], t)
        self.assertIn("<forever dataset>", t)


class TestWorkflow(unittest.TestCase):
    def load(self, path):
        with open(os.path.join(_root, path), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_manual_only_read_only_with_the_secret_and_an_artifact(self):
        for path in (".github/workflows/forever_health.yml", "tools/forever_health.yml"):
            doc = self.load(path)
            on = doc.get("on", doc.get(True))
            self.assertEqual(set(on), {"workflow_dispatch"})
            steps = doc["jobs"]["health"]["steps"]
            run = next(s for s in steps if "forever_health.py" in s.get("run", ""))
            self.assertEqual(run["env"]["HF_TOKEN"], "${{ secrets.HF_TOKEN }}")
            self.assertIn("set -o pipefail", run["run"])
            up = next(s for s in steps if "upload-artifact" in s.get("uses", ""))
            self.assertEqual(up["with"]["path"], "out/forever_health.txt")
            self.assertEqual(up["if"], "${{ always() }}")
            self.assertIn("huggingface_hub", next(s for s in steps if "pip install" in s.get("run", ""))["run"])

    def test_the_two_copies_match(self):
        with open(os.path.join(_root, ".github/workflows/forever_health.yml"), encoding="utf-8") as a, open(os.path.join(_root, "tools/forever_health.yml"), encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())

    def test_nothing_is_written_to_any_dataset(self):
        with open(os.path.join(_root, "tools/forever_health.py"), encoding="utf-8") as f:
            src = f.read()
        for word in ("create_commit", "upload_file", "upload_folder", "create_repo", "CommitOperation"):
            self.assertNotIn(word, src)


if __name__ == "__main__":
    unittest.main()
