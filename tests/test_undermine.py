import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import undermine as U  # noqa: E402

KEY = "sekrit-key-123"


class Resp:
    def __init__(self, status=200, js=None):
        self.status_code, self._js = status, js

    def json(self):
        if self._js is None:
            raise ValueError("no json")
        return self._js


class FakeSession:
    """get(url, headers, timeout) -> scripted response. `script` is a function(url) -> Resp (or an exception to raise)."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        out = self.script(url)
        if isinstance(out, Exception):
            raise out
        return out


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def daily_js(iid, days=5, start="2022-09-04"):
    d = pd.date_range(start, periods=days)
    return {"result": {"daily": [{"day": x.strftime("%Y-%m-%d"), "price": 1000 + int(iid) % 7, "quantity": 500 + k} for k, x in enumerate(d)]}}


def hourly_js(iid, hours=48, start="2026-10-30 00:00", price=250000, qty0=10000, drops=None):
    ts = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    q, rows = qty0, []
    for k, t in enumerate(ts):
        q = q - (drops[k] if drops is not None else 0)
        rows.append({"snapshot": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": price, "quantity": int(q)})
    return {"result": {"hourly": rows}}


def make_client(script, **kw):
    clock = Clock()
    c = U.Client(KEY, session=FakeSession(script), sleep=clock.sleep, now=clock.now, pause=0, **kw)
    return c, clock


class FakeClient:
    """Stands in for U.Client in run_* tests: answers from a dict {item_id: json}, 404 for others, can run out of budget after N calls."""

    def __init__(self, daily=None, hourly=None, budget_after=None, reject=False):
        self.daily, self.hourly, self.budget_after, self.reject = daily or {}, hourly or {}, budget_after, reject
        self.points, self.n, self.paths = 0, 0, []

    def get(self, path):
        self.paths.append(path)
        if self.reject:
            raise U.KeyRejected("HTTP 401: the key was rejected, or this endpoint needs a paid membership.")
        if self.budget_after is not None and self.n >= self.budget_after:
            raise U.BudgetSpent("HTTP 429")
        self.n += 1
        self.points += 3
        iid = int(path.split("/commodities/")[1].split("/")[0])
        src = self.daily if path.endswith("daily.json") else self.hourly
        return (200, src[iid]) if iid in src else (404, None)


def item_groups(per=40):
    """Two big groups across all four eras and two qualities, one small group."""
    rows = []
    for gname, base in (("Alpha", 1000), ("Beta", 2000)):
        for k in range(per):
            era_id = (50000, 150000, 220000, 250000)[k % 4]
            rows.append(dict(item_id=era_id + base + k, group=gname, quality="COMMON" if k % 8 < 4 else "RARE"))
    for k in range(5):
        rows.append(dict(item_id=900000 + k, group="Tiny", quality="COMMON"))
    return pd.DataFrame(rows)


class TestClient(unittest.TestCase):
    def test_sends_the_key_and_gzip_and_the_key_never_appears_in_errors(self):
        s = FakeSession(lambda url: Resp(200, {"ok": 1}))
        c = U.Client(KEY, session=s, pause=0, sleep=lambda x: None)
        self.assertEqual(c.get("/v1/region/us/commodities/1/daily.json"), (200, {"ok": 1}))
        url, h = s.calls[0]
        self.assertTrue(url.startswith("https://api.undermine.exchange/v1/region/us/"))
        self.assertEqual(h["Authorization"], f"ApiKey {KEY}")
        self.assertEqual(h["Accept-Encoding"], "gzip")
        self.assertEqual(c.points, 3)
        for status, exc in ((401, U.KeyRejected), (403, U.KeyRejected), (429, U.BudgetSpent)):
            c2 = U.Client(KEY, session=FakeSession(lambda url, st=status: Resp(st)), pause=0, sleep=lambda x: None)
            with self.assertRaises(exc) as cm:
                c2.get("/x")
            self.assertNotIn(KEY, str(cm.exception))
        with self.assertRaises(U.KeyRejected):
            U.Client("", session=s)

    def test_404_is_reported_and_network_trouble_retries_then_gives_up(self):
        c, clock = make_client(lambda url: Resp(404))
        self.assertEqual(c.get("/x"), (404, None))
        c, clock = make_client(lambda url: ConnectionError("down"))
        self.assertEqual(c.get("/x"), (0, None))
        self.assertEqual(len(c._s.calls), 3)
        self.assertTrue(clock.slept)                                                    # it backed off between tries
        c, clock = make_client(lambda url: Resp(200, None))                              # 200 with a broken body
        self.assertEqual(c.get("/x"), (0, None))

    def test_local_budget_waits_for_the_hour_to_roll_instead_of_hitting_429(self):
        c, clock = make_client(lambda url: Resp(200, {}), points_per_hour=9)
        for _ in range(3):
            c.get("/x")
        self.assertEqual([x for x in clock.slept if x > 0], [])
        c.get("/x")                                                                     # the 4th would be 12 points inside an hour
        self.assertGreaterEqual(sum(clock.slept), 3600 - 1)
        self.assertEqual(len(c._s.calls), 4)


class TestParsers(unittest.TestCase):
    def test_parse_daily(self):
        js = daily_js(5, days=3)
        js["result"]["daily"].append({"day": "2022-09-06", "price": 7, "quantity": 9})   # duplicate day: the last one wins
        js["result"]["daily"].append({"day": "2022-09-07", "price": None, "quantity": 9})  # no price: dropped
        d = U.parse_daily(js, 5)
        self.assertEqual(list(d.columns), U.HISTORY_COLUMNS)
        self.assertEqual(list(d["day"]), ["2022-09-04", "2022-09-05", "2022-09-06"])
        self.assertEqual(d["price"].iloc[-1], 7)
        self.assertTrue(U.parse_daily({}, 5).empty and U.parse_daily({"result": {"daily": []}}, 5).empty)

    def test_parse_hourly(self):
        h = U.parse_hourly(hourly_js(5, hours=4), 5)
        self.assertEqual(len(h), 4)
        self.assertTrue(h["ts"].is_monotonic_increasing)
        self.assertEqual(str(h["ts"].dt.tz), "UTC")
        self.assertTrue(U.parse_hourly(None, 5).empty)


class TestSampling(unittest.TestCase):
    def test_caps_each_group_and_spreads_over_era_and_quality(self):
        g = item_groups()
        ids = U.sample_items(g, per_group=12)
        s = g.set_index("item_id").loc[ids].reset_index()
        counts = s["group"].value_counts()
        self.assertEqual(counts["Alpha"], 12)
        self.assertEqual(counts["Beta"], 12)
        self.assertEqual(counts["Tiny"], 5)                                             # a small group is taken whole
        a = s[s["group"] == "Alpha"]
        self.assertEqual(a["item_id"].map(lambda i: 0 if i < 120000 else 1 if i < 200000 else 2 if i < 240000 else 3).nunique(), 4)   # every era is in
        self.assertEqual(a["quality"].nunique(), 2)
        self.assertEqual(len(ids), len(set(ids)))

    def test_any_prefix_is_balanced_across_groups(self):
        ids = U.sample_items(item_groups(), per_group=30)
        gmap = item_groups().set_index("item_id")["group"]
        first = pd.Series([gmap[i] for i in ids[:30]]).value_counts()
        self.assertEqual(set(first.index), {"Alpha", "Beta", "Tiny"})
        self.assertLessEqual(first.max() - first.min(), 10)                              # Tiny runs out after 5, the big two stay level
        self.assertLessEqual(abs(first["Alpha"] - first["Beta"]), 1)

    def test_exclude_and_empty(self):
        g = item_groups()
        first = U.sample_items(g, per_group=3)
        again = U.sample_items(g, per_group=3, exclude=first)
        self.assertFalse(set(first) & set(again))
        self.assertEqual(U.sample_items(pd.DataFrame(), 5), [])
        self.assertEqual(U.sample_items(None, 5), [])

    def test_check_sample_uses_only_well_measured_busy_items_that_we_have_hours_for(self):
        items = pd.DataFrame({"item_id": range(1, 41), "family": ["A", "B"] * 20, "daily_est": [100.0] * 30 + [1.0] * 10,
                              "confidence": ["High", "Low"] * 20})
        hours = pd.DataFrame({"item_id": list(range(1, 37))})
        ids = U.sample_for_check(items, hours, n=8)
        self.assertEqual(len(ids), 8)
        got = items.set_index("item_id").loc[ids]
        self.assertTrue((got["confidence"] == "High").all() and (got["daily_est"] >= 20).all())
        self.assertTrue(all(i <= 36 for i in ids))
        self.assertEqual(got["family"].nunique(), 1)                                    # only family A has High-confidence busy items here
        self.assertEqual(U.sample_for_check(items.iloc[0:0], hours), [])


class TestPullAndMerge(unittest.TestCase):
    def test_pulls_up_to_the_batch_and_records_missing_items(self):
        fc = FakeClient(daily={i: daily_js(i, days=5) for i in (1, 2, 4, 5)})            # 3 is not tracked
        r = U.pull_daily([1, 2, 3, 4, 5], fc, batch=4, progress=lambda *a: None)
        self.assertEqual(r["stopped"], "")
        self.assertEqual(sorted(r["data"]["item_id"].unique()), [1, 2, 4])               # the 5th item is beyond the batch
        m = r["meta"].set_index("item_id")
        self.assertEqual(m.loc[3, "status"], "not_found")
        self.assertEqual((m.loc[1, "status"], m.loc[1, "n_days"], m.loc[1, "first_day"]), ("ok", 5, "2022-09-04"))

    def test_budget_stop_keeps_what_was_fetched(self):
        fc = FakeClient(daily={i: daily_js(i) for i in range(1, 11)}, budget_after=4)
        r = U.pull_daily(list(range(1, 11)), fc, batch=10, progress=lambda *a: None)
        self.assertEqual(r["stopped"], "budget")
        self.assertEqual(sorted(r["data"]["item_id"].unique()), [1, 2, 3, 4])
        self.assertEqual(len(r["meta"]), 4)

    def test_rejected_key_stops_with_a_reason(self):
        out = []
        r = U.pull_daily([1, 2], FakeClient(reject=True), progress=out.append)
        self.assertEqual(r["stopped"], "key")
        self.assertTrue(r["data"].empty and r["meta"].empty)

    def test_network_failures_are_not_recorded_so_the_next_run_retries_them(self):
        class Flaky(FakeClient):
            def get(self, path):
                return (0, None) if "/1/" in path else super().get(path)
        r = U.pull_daily([1, 2], Flaky(daily={2: daily_js(2)}), progress=lambda *a: None)
        self.assertEqual(list(r["meta"]["item_id"]), [2])

    def test_an_empty_answer_is_recorded_as_empty(self):
        r = U.pull_daily([1], FakeClient(daily={1: {"result": {"daily": []}}}), progress=lambda *a: None)
        self.assertEqual(r["meta"].iloc[0]["status"], "empty")

    def test_merge_history_and_meta(self):
        a = U.parse_daily(daily_js(1, days=3), 1)
        b = U.parse_daily(daily_js(1, days=5), 1).assign(price=999)                      # overlaps and extends; newer values win
        m = U.merge_history(a, pd.concat([b, U.parse_daily(daily_js(2, days=2), 2)], ignore_index=True))
        self.assertEqual(len(m), 5 + 2)
        self.assertTrue((m[m["item_id"] == 1]["price"] == 999).all())
        self.assertTrue(U.merge_history(None, pd.DataFrame(columns=U.HISTORY_COLUMNS)).empty)
        m1 = pd.DataFrame([dict(item_id=1, status="empty", fetched_at="a", first_day="", last_day="", n_days=0)])
        m2 = pd.DataFrame([dict(item_id=1, status="ok", fetched_at="b", first_day="x", last_day="y", n_days=3)])
        mm = U.merge_meta(m1, m2)
        self.assertEqual((len(mm), mm.iloc[0]["status"]), (1, "ok"))


def ours_and_under(items=(7, 8), hours=48, drops_scale=1.0, gold=True, seed=1):
    """Our hourly rows and Undermine's for the same hours: stock falls by the units sold (plus noise), price is the same."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-10-30 00:00", periods=hours, freq="h", tz="UTC")
    o, u = [], []
    for it in items:
        sold = rng.integers(0, 60, hours).astype(float)
        sold[0] = np.nan
        units = 50000 - np.cumsum(np.nan_to_num(sold)) * drops_scale + 9000 * (np.arange(hours) % 5 == 0)   # restocks make the totals noisy
        o.append(pd.DataFrame({"item_id": it, "ts": ts, "floor_gold": 25.0, "units_listed": units, "sold_est": sold}))
        u.append(pd.DataFrame({"item_id": it, "ts": ts + pd.Timedelta(minutes=7), "price": 250000.0 if gold else 25.0, "quantity": units}))
    return pd.concat(o, ignore_index=True), pd.concat(u, ignore_index=True)


class TestCalibration(unittest.TestCase):
    def test_same_feed_in_copper_gives_ratio_one_and_exact_matches(self):
        ours, under = ours_and_under()
        m = U.align_hourly(under, ours)
        self.assertEqual(len(m), 96)
        rep = U.calibration_report(m)
        self.assertAlmostEqual(rep["price_ratio"], 1.0)
        self.assertAlmostEqual(rep["price_exact"], 1.0)
        self.assertAlmostEqual(rep["qty_ratio"], 1.0)
        self.assertEqual(rep["items"], 2)

    def test_gold_instead_of_copper_shows_up_as_a_ratio_of_one_ten_thousandth(self):
        ours, under = ours_and_under(gold=False)
        rep = U.calibration_report(U.align_hourly(under, ours))
        self.assertAlmostEqual(rep["price_ratio"], 1e-4, places=6)
        self.assertEqual(rep["price_exact"], 0.0)

    def test_stock_drops_track_sales_when_restocks_are_modest(self):
        ours, under = ours_and_under(items=(7, 8, 9), hours=80)
        ours["units_listed"] = ours.groupby("item_id")["sold_est"].transform(lambda s: 50000 - s.fillna(0).cumsum())
        under["quantity"] = ours["units_listed"].to_numpy()
        rep = U.calibration_report(U.align_hourly(under, ours))
        self.assertGreater(rep["spearman_median"], 0.95)
        self.assertEqual(len(rep["per_item"]), 3)
        self.assertAlmostEqual(rep["per_item"]["proxy_over_ours"].median(), 1.0, places=1)

    def test_unrelated_stock_changes_track_weakly(self):
        ours, under = ours_and_under(items=(7, 8, 9), hours=120)
        rng = np.random.default_rng(5)
        under["quantity"] = 50000 + rng.integers(-3000, 3000, len(under))
        rep = U.calibration_report(U.align_hourly(under, ours))
        self.assertLess(abs(rep["spearman_median"]), 0.3)

    def test_nothing_lines_up_outside_the_tolerance_and_empty_inputs(self):
        ours, under = ours_and_under()
        far = under.assign(ts=under["ts"] + pd.Timedelta(hours=40))
        self.assertTrue(U.align_hourly(far.assign(ts=far["ts"] + pd.Timedelta(days=30)), ours).empty)
        self.assertEqual(U.calibration_report(pd.DataFrame()), {})
        self.assertTrue(U.align_hourly(pd.DataFrame(), ours).empty)

    def test_depth_buckets(self):
        per = pd.DataFrame({"item_id": range(6), "spearman": [0.9, 0.8, 0.5, 0.4, 0.2, 0.1], "median_units": [100, 200, 3000, 4000, 90000, 100000],
                            "our_sold_per_hour": [50.0] * 6})
        b = U._depth_buckets(per)
        self.assertEqual(list(b["bucket"]), ["shallow stock", "medium", "deep stock"])
        self.assertTrue(b["spearman_median"].is_monotonic_decreasing)


class TestCommands(unittest.TestCase):
    def _args(self, **kw):
        import lake
        base = dict(ids="", out="x", upload=False, batch=800)
        base.update(kw)
        return lake.argparse.Namespace(**base)

    def _run(self, fn, args, tables, client):
        import lake
        buf, uploaded, written = io.StringIO(), [], {}
        store = dict(tables)

        def fake_to_parquet(self_, path, **kw):
            written[os.path.basename(path)] = self_.copy()
            open(path, "wb").close()

        with mock.patch.object(pd.DataFrame, "to_parquet", fake_to_parquet), \
                mock.patch.object(lake, "_upload_folder_to_engine", side_effect=lambda folder, msg: uploaded.append(sorted(os.listdir(folder)))), \
                contextlib.redirect_stdout(buf):
            out = getattr(lake, fn)(None, args, client=client, read_table=lambda n: store.get(n, pd.DataFrame()))
        return out, buf.getvalue(), uploaded, written

    def test_pull_is_resumable_and_stores_progress_privately(self):
        g = item_groups(per=12)
        ids = U.sample_items(g, per_group=30)
        fc = FakeClient(daily={i: daily_js(i, days=6) for i in ids})
        out, text, up, wrote = self._run("run_undermine", self._args(batch=10, upload=True, out="x"), {"item_groups": g}, fc)
        self.assertEqual(len(wrote["market_history_meta.parquet"]), 10)
        self.assertEqual(up, [["market_history.parquet", "market_history_meta.parquet"]])
        self.assertIn("still to fetch", text)
        self.assertNotIn(KEY, text)
        self.assertNotIn("2022-09-05", text)                                              # only summary numbers are printed, never the day rows
        # second run: starts from what the first stored, does not fetch those items again
        tables = {"item_groups": g, "market_history": wrote["market_history.parquet"], "market_history_meta": wrote["market_history_meta.parquet"]}
        fc2 = FakeClient(daily={i: daily_js(i, days=6) for i in ids})
        out2, text2, up2, wrote2 = self._run("run_undermine", self._args(batch=1000, upload=True), tables, fc2)
        fetched1 = set(wrote["market_history_meta.parquet"]["item_id"])
        fetched2 = {int(p.split("/commodities/")[1].split("/")[0]) for p in fc2.paths}
        self.assertFalse(fetched1 & fetched2)
        self.assertEqual(len(wrote2["market_history_meta.parquet"]), len(ids))
        self.assertIn("All sampled items are done.", text2)
        # third run has nothing to do and spends nothing
        fc3 = FakeClient(daily={})
        out3, text3, up3, _ = self._run("run_undermine", self._args(upload=True), {"item_groups": g, **{k: wrote2[k + ".parquet"] for k in ("market_history", "market_history_meta")}}, fc3)
        self.assertEqual((fc3.paths, up3), ([], []))
        self.assertIn("Nothing to fetch", text3)

    def test_pull_that_runs_out_of_budget_saves_what_it_has(self):
        g = item_groups(per=12)
        ids = U.sample_items(g, per_group=30)
        fc = FakeClient(daily={i: daily_js(i) for i in ids}, budget_after=6)
        out, text, up, wrote = self._run("run_undermine", self._args(upload=True), {"item_groups": g}, fc)
        self.assertIn("STOPPED: the hourly point budget is used up", text)
        self.assertEqual(len(wrote["market_history_meta.parquet"]), 6)
        self.assertEqual(len(up), 1)

    def test_without_upload_it_says_nothing_was_saved(self):
        g = item_groups(per=5)
        ids = U.sample_items(g, per_group=30)
        out, text, up, wrote = self._run("run_undermine", self._args(), {"item_groups": g}, FakeClient(daily={i: daily_js(i) for i in ids}))
        self.assertIn("NOT SAVED", text)
        self.assertEqual((up, wrote), ([], {}))

    def test_rejected_key_is_reported_and_nothing_is_uploaded(self):
        g = item_groups(per=5)
        out, text, up, wrote = self._run("run_undermine", self._args(upload=True), {"item_groups": g}, FakeClient(reject=True))
        self.assertIn("STOPPED: Undermine rejected the key", text)
        self.assertEqual(up, [])

    def test_pull_without_groups_says_what_to_run_first(self):
        out, text, up, _ = self._run("run_undermine", self._args(), {}, FakeClient())
        self.assertIn("run mode groups", text)

    def test_missing_secret_stops_without_printing_anything_secret(self):
        import lake
        with mock.patch.dict(os.environ, {"UNDERMINE_API_KEY": ""}):
            with self.assertRaises(SystemExit) as cm:
                lake._undermine_client(None)
        self.assertIn("UNDERMINE_API_KEY", str(cm.exception))
        with mock.patch.dict(os.environ, {"UNDERMINE_API_KEY": KEY}):
            self.assertIsInstance(lake._undermine_client(None), U.Client)

    def test_check_command_reports_units_feed_and_tracking_without_raw_rows(self):
        ours, under = ours_and_under(items=(7, 8, 9), hours=120)
        ours["units_listed"] = ours.groupby("item_id")["sold_est"].transform(lambda s: 50000 - s.fillna(0).cumsum())
        under["quantity"] = ours["units_listed"].to_numpy()
        items = pd.DataFrame({"item_id": [7, 8, 9], "family": "A", "daily_est": 100.0, "confidence": "High"})
        hourly = {i: {"result": {"hourly": [{"snapshot": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": float(p), "quantity": float(q)}
                                            for t, p, q in zip(g["ts"], g["price"], g["quantity"])]}}
                  for i, g in under.groupby("item_id")}
        fc = FakeClient(hourly=hourly)
        out, text, up, _ = self._run("run_undermine_check", self._args(), {"items": items, "item_hourly": ours}, fc)
        self.assertIn("UNITS CHECK", text)
        self.assertIn("1.0000", text)
        self.assertIn("tracks well", text)
        self.assertIn("FEED CHECK", text)
        self.assertEqual(up, [])
        self.assertTrue(all(p.endswith("hourly.json") for p in fc.paths))
        self.assertNotIn("250000", text)
        self.assertNotIn(KEY, text)

    def test_check_command_needs_derive_tables(self):
        out, text, up, _ = self._run("run_undermine_check", self._args(), {}, FakeClient())
        self.assertIn("run derive with upload first", text)


class TestWorkflowWiring(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(_root, ".github", "workflows", "pick_items.yml")) as f:
            self.a = f.read()
        with open(os.path.join(_root, "tools", "pick_items.yml")) as f:
            self.b = f.read()

    def test_the_two_copies_are_identical(self):
        self.assertEqual(self.a, self.b)

    def test_secret_batch_and_modes_are_wired_and_no_artifact_for_undermine(self):
        import yaml
        wf = yaml.safe_load(self.a)
        inputs = wf[True]["workflow_dispatch"]["inputs"]
        self.assertIn("undermine", inputs["mode"]["options"])
        self.assertIn("undermine_check", inputs["mode"]["options"])
        self.assertIn("undermine_test", inputs["mode"]["options"])
        self.assertEqual(inputs["batch"]["default"], "800")
        steps = wf["jobs"]["run"]["steps"]
        run = next(s for s in steps if s.get("name") == "Run")
        self.assertEqual(run["env"]["UNDERMINE_API_KEY"], "${{ secrets.UNDERMINE_API_KEY }}")
        self.assertIn('--batch "$BATCH"', run["run"])
        self.assertNotIn("UNDERMINE_API_KEY", run["run"])                                 # the key is only ever an env var
        save = next(s for s in steps if s.get("name") == "Save results")
        for mode in ("ledger", "undermine", "undermine_check", "undermine_test"):
            self.assertIn(f"inputs.mode != '{mode}'", save["if"])
        self.assertLessEqual(len(inputs), 25)

    def test_the_command_line_accepts_the_new_modes(self):
        import lake
        with mock.patch.object(lake, "run_undermine", return_value=1) as a, mock.patch.object(lake, "run_undermine_check", return_value=2) as b, \
                mock.patch.object(lake, "make_store", side_effect=AssertionError("no store needed")):
            self.assertEqual(lake.main(["undermine", "--batch", "50"]), 1)
            self.assertEqual(lake.main(["undermine_check"]), 2)
            with mock.patch.object(lake, "run_undermine_test", return_value=3):
                self.assertEqual(lake.main(["undermine_test"]), 3)
        self.assertEqual(a.call_args[0][1].batch, 50)


if __name__ == "__main__":
    unittest.main()
