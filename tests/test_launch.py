import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import history as HI  # noqa: E402
from engine import launch as LA  # noqa: E402

START = "2022-09-04"


def launch_world(seed=3, days=520, launches=((90, 70), (210, 70), (330, 70)), split_second=False, noise=0.03):
    """Three launches of 70 items (35 of FamA with a big first-day overshoot, 35 of FamB with a small one), each item settling at its own level with its own speed, plus
    10 steady old items and one flat single new item per week (the trickle that makes a launch week stand out). Returns (history, groups, true settled level per item)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days)
    rows, groups, truth = [], [], {}
    iid = [1000]

    def add(start_idx, settled, m0, tau, fam, stock_tau=8.0):
        iid[0] += 1
        age = np.clip(np.arange(days) - start_idx, 0, None)
        mult = 1 + (m0 - 1) * np.exp(-age / tau)
        mult_s = 1 - 0.8 * np.exp(-age / stock_tau)
        price = settled * mult * 10 ** rng.normal(0, noise, days)
        qty = 5000 * mult_s * 10 ** rng.normal(0, noise, days)
        d = dates[start_idx:]
        rows.append(pd.DataFrame({"item_id": iid[0], "day": d.strftime("%Y-%m-%d"), "price": price[start_idx:] * 10000.0, "quantity": qty[start_idx:]}))
        groups.append(dict(item_id=iid[0], group=fam + ": x", family=fam))
        truth[iid[0]] = settled

    for _ in range(10):
        add(0, 10 ** rng.uniform(0, 2), 1.0, 5.0, "FamOld")
    busy = [s + extra for s, _ in launches for extra in (0, 7)]
    for wk in range(5, 62, 1):
        if any(wk * 7 + 1 <= b <= wk * 7 + 7 for b in busy):                            # keep the singles out of the launch weeks so each launch has exactly its own items
            continue
        add(wk * 7 + 1, 10 ** rng.uniform(0, 2), 1.0, 5.0, "FamSingle")
    for start, n in launches:
        for j in range(n):
            fam = "FamA" if j % 2 == 0 else "FamB"
            m0 = np.exp(rng.normal(np.log(6.0 if fam == "FamA" else 2.0), 0.25))
            tau = np.exp(rng.normal(np.log(6.0), 0.4))
            s = start + (7 if (split_second and j % 2 == 1) else 0)
            add(s, 10 ** rng.uniform(0, 2.5), m0, tau, fam)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(groups), truth


class TestLaunches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hist, cls.groups, cls.truth = launch_world()
        cls.fm = HI.fresh_matrices(HI.clean(cls.hist))
        cls.details = LA.backtest(cls.fm, cls.groups)
        cls.summ = LA.summarize(cls.details)

    def test_three_launches_are_found_with_their_items(self):
        lab = LA.launch_labels(self.fm)
        sizes = lab.value_counts().sort_index()
        self.assertEqual(list(sizes.values), [70, 70, 70])
        self.assertEqual(len(sizes), 3)

    def test_two_adjacent_burst_weeks_are_one_launch(self):
        h, g, _ = launch_world(split_second=True)
        fm = HI.fresh_matrices(HI.clean(h))
        self.assertGreaterEqual(len(fm["burst_weeks"]), 6)                            # every launch spans two weeks
        sizes = LA.launch_labels(fm).value_counts()
        self.assertEqual(sorted(sizes.values), [70, 70, 70])

    def test_the_prior_beats_the_naive_guess_and_family_beats_the_plain_prior(self):
        s = self.summ[(self.summ["metric"] == "price") & (self.summ["age"] == 7)].iloc[0]
        self.assertGreater(s["miss_naive"], 1.3)                                       # "settled = today's price" is off by a lot on day 7
        self.assertLess(s["miss_age"], s["miss_naive"] * 0.85)
        self.assertLess(s["miss_family"], s["miss_age"] * 0.95)                        # FamA and FamB overshoot differently, so the family guess is tighter
        self.assertLess(s["miss_family"], 1.2)
        s1 = self.summ[(self.summ["metric"] == "price") & (self.summ["age"] == 1)].iloc[0]
        self.assertGreater(s1["miss_naive"], s["miss_naive"])                          # the earlier the day, the worse the naive guess

    def test_the_ranges_are_calibrated_on_launches_the_model_has_not_seen(self):
        t = self.summ[self.summ["metric"] == "price"]
        for m in ("age", "family"):
            self.assertTrue(t[f"hit50_{m}"].between(0.38, 0.62).all(), t[f"hit50_{m}"].tolist())
            self.assertTrue(t[f"hit80_{m}"].between(0.68, 0.92).all(), t[f"hit80_{m}"].tolist())
        self.assertTrue((t["width80_family"] <= t["width80_age"] + 1e-9).all())

    def test_every_hidden_launch_is_tested_and_labelled(self):
        self.assertEqual(self.details["launch"].nunique(), 3)
        self.assertEqual(set(self.details["metric"]), {"price", "stock"})
        self.assertEqual(sorted(self.details["age"].unique()), list(LA.AGES_USED))
        one = self.details[(self.details["metric"] == "price") & (self.details["age"] == 7)]
        self.assertEqual(len(one), 210)

    def test_best_models_pick_the_smallest_miss_and_default_to_age(self):
        best = LA.best_models(self.summ)
        self.assertIn(best[("price", 7)], ("family", "path"))
        tie = pd.DataFrame([{"metric": "price", "age": 7, "miss_age": 1.50, "miss_family": 1.49, "miss_path": 1.55}])
        self.assertEqual(LA.best_models(tie)[("price", 7)], "age")                       # under 3% better is not worth the extra moving part

    def test_a_launch_too_small_to_test_is_skipped(self):
        h, g, _ = launch_world(launches=((90, 70), (210, 70), (330, 8)))
        fm = HI.fresh_matrices(HI.clean(h))
        d = LA.backtest(fm, g)
        self.assertEqual(d["launch"].nunique(), 2)

    def test_no_launches_means_an_empty_backtest(self):
        h, g, _ = launch_world(launches=())
        fm = HI.fresh_matrices(HI.clean(h))
        self.assertTrue(LA.backtest(fm, g).empty)
        self.assertTrue(LA.build_prior(fm, g).empty)


class TestPriorAndEstimates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h, g, _ = launch_world()
        cls.fm = HI.fresh_matrices(HI.clean(h))
        cls.best = LA.best_models(LA.summarize(LA.backtest(cls.fm, g)))
        cls.prior = LA.build_prior(cls.fm, g, cls.best)
        cls.groups = g

    def test_prior_table_has_the_scopes_models_and_ages(self):
        p = self.prior
        self.assertEqual(set(p["scope"]), {"all", "FamA", "FamB"})                      # families under 15 items get no scope of their own
        self.assertEqual(set(p["metric"]), {"price", "stock"})
        self.assertEqual(set(p["model"]), {"age", "path"})
        self.assertEqual(set(p[p["model"] == "age"]["age"]), set(LA.PRIOR_AGES))
        self.assertEqual(set(p[p["model"] == "path"]["age"]), set(LA.AGES_USED))        # a path needs a day 0 to start from
        self.assertIn("best_model", p.columns)
        r = p[(p["scope"] == "FamA") & (p["metric"] == "price") & (p["age"] == 0) & (p["model"] == "age")].iloc[0]
        self.assertAlmostEqual(10 ** r["l50"], 6.0, delta=1.0)                           # FamA's median day-0 multiple was built to be about 6
        r = p[(p["scope"] == "FamB") & (p["metric"] == "price") & (p["age"] == 0) & (p["model"] == "age")].iloc[0]
        self.assertAlmostEqual(10 ** r["l50"], 2.0, delta=0.5)
        self.assertTrue((p["l10"] <= p["l25"]).all() and (p["l25"] <= p["l50"]).all() and (p["l75"] <= p["l90"]).all())

    def test_an_estimate_from_a_new_launch_holds_the_truth_about_as_often_as_promised(self):
        h2, g2, truth = launch_world(seed=11)
        d = HI.clean(h2)
        first = d.groupby("item_id")["day"].min()
        inside, n = 0, 0
        for iid, s in truth.items():
            if iid not in set(g2[g2["family"].isin(["FamA", "FamB"])]["item_id"]):
                continue
            it = d[d["item_id"] == iid].set_index("day")["price"]
            f0 = first[iid]
            p0, p7 = it.get(f0), it.get(f0 + pd.Timedelta(days=7))
            if p0 is None or p7 is None:
                continue
            fam = g2.set_index("item_id").loc[iid, "family"]
            e = LA.settled_estimate(self.prior, fam, "price", 7, p7, p0)
            n += 1
            inside += e["lo80"] <= s <= e["hi80"]
            self.assertLessEqual(e["lo80"], e["lo50"])
            self.assertLessEqual(e["lo50"], e["mid"])
            self.assertLessEqual(e["mid"], e["hi50"])
            self.assertLessEqual(e["hi50"], e["hi80"])
        self.assertEqual(n, 210)
        self.assertTrue(0.70 <= inside / n <= 0.92, inside / n)

    def test_estimate_edge_cases(self):
        p = self.prior
        self.assertIsNone(LA.settled_estimate(p, "FamA", "price", 30, 10.0, 20.0))        # past the window the prior can speak to
        self.assertIsNone(LA.settled_estimate(p, "FamA", "price", 7, 0.0, 20.0))
        self.assertIsNone(LA.settled_estimate(pd.DataFrame(), "FamA", "price", 7, 10.0, 20.0))
        self.assertIsNone(LA.settled_estimate(p, "FamA", "price", 7, None, 20.0))
        e = LA.settled_estimate(p, "Nope", "price", 7, 10.0, 20.0)                        # an unknown family falls back to all items
        self.assertEqual(e["scope"], "all")
        e4 = LA.settled_estimate(p, "FamA", "price", 4, 10.0, 20.0)                       # age 4 uses the nearest age the prior has
        self.assertIn(e4["age"], (3, 5))
        e0 = LA.settled_estimate(p, "FamA", "price", 0, 10.0, None, model="path")         # no path on day 0: falls back to a plain guess, still a range
        self.assertIn(e0["model"], ("age", "family"))
        self.assertGreater(e0["hi80"], e0["lo80"])
        a = LA.settled_estimate(p, "FamA", "price", 7, 10.0, 20.0, model="age")
        self.assertEqual((a["scope"], a["model"]), ("all", "age"))
        f = LA.settled_estimate(p, "FamA", "price", 7, 10.0, 20.0, model="family")
        self.assertEqual((f["scope"], f["model"]), ("FamA", "family"))
        pa = LA.settled_estimate(p, "FamA", "price", 7, 10.0, 20.0, model="path")
        self.assertEqual(pa["model"], "path")

    def test_launch_report_covers_only_young_items_seen_on_the_latest_day(self):
        days = pd.date_range("2026-11-04", periods=30)
        rows = []
        for iid, first_idx, fam in ((1, 0, "FamA"), (2, 12, "FamB"), (3, 25, "FamA"), (4, 0, "FamA")):
            for i in range(first_idx, 30):
                if iid == 4 and i > 20:
                    continue                                                                # item 4 vanished a week ago
                rows.append({"item_id": iid, "day": days[i], "price": 100.0 / (1 + 0.1 * (i - first_idx)), "quantity": 500.0 * (1 + 0.2 * (i - first_idx))})
        g = pd.DataFrame({"item_id": [1, 2, 3, 4], "family": ["FamA", "FamB", "FamA", "FamA"], "group": "x"})
        rep = LA.launch_report(pd.DataFrame(rows), self.prior, g, max_age=21)
        self.assertEqual(sorted(rep["item_id"]), [2, 3])                                    # item 1 is 29 days old, item 4 is not on the latest day
        for c in ("price_now", "settled_mid", "settled_lo80", "settled_hi80", "price_now_vs_settled", "settled_units_mid"):
            self.assertTrue(rep[c].notna().all(), c)
        self.assertTrue((rep["settled_lo80"] <= rep["settled_mid"]).all() and (rep["settled_mid"] <= rep["settled_hi80"]).all())
        self.assertTrue(LA.launch_report(pd.DataFrame(), self.prior, g).empty)
        self.assertTrue(LA.launch_report(pd.DataFrame(rows), pd.DataFrame(), g).empty)


class TestCommands(unittest.TestCase):
    def _args(self, **kw):
        import lake
        base = dict(ids="", out="x", upload=False)
        base.update(kw)
        return lake.argparse.Namespace(**base)

    def test_backtest_prints_the_tables_aggregates_only_and_stores_the_prior_only_when_asked(self):
        import lake
        h, g, _ = launch_world()
        tables = {"market_history": h, "item_groups": g}
        for upload in (False, True):
            buf, stored = io.StringIO(), []
            with mock.patch.object(lake, "_store_private", side_effect=lambda t, m, u: (stored.append(sorted(t)) or bool(u))), contextlib.redirect_stdout(buf):
                res = lake.run_launch_backtest(None, self._args(upload=upload), read_table=lambda n: tables.get(n, pd.DataFrame()))
            text = buf.getvalue()
            self.assertEqual(stored, [["launch_prior"]])
            self.assertIn("LAUNCH FORECASTER BACKTEST: 210 items in 3 launches", text)
            for needle in ("SETTLED PRICE: typical miss", "SETTLED STOCK", "do the ranges hold", "DOES IT TRANSFER BETWEEN LAUNCHES", "BEST GUESS PER DAY"):
                self.assertIn(needle, text)
            self.assertIn("Saved engine/launch_prior.parquet" if upload else "Not saved: tick upload", text)
            self.assertEqual(len(res["launches"]), 3)
            self.assertNotIn("1001", text)                                                  # no item ids
            self.assertFalse(any(line.startswith("==") for line in text.splitlines()))     # a spreadsheet would turn those lines into errors

    def test_backtest_says_what_to_run_first(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertFalse(lake.run_launch_backtest(None, self._args(), read_table=lambda n: pd.DataFrame()))
        self.assertIn("run mode undermine", buf.getvalue())
        h, g, _ = launch_world(launches=())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lake.run_launch_backtest(None, self._args(), read_table=lambda n: {"market_history": h, "item_groups": g}.get(n, pd.DataFrame()))
        self.assertIn("Nothing to test", buf.getvalue())

    def test_launch_now_reads_our_daily_table_and_prints_ranges(self):
        import lake
        h, g, _ = launch_world()
        fm = HI.fresh_matrices(HI.clean(h))
        prior = LA.build_prior(fm, g, LA.best_models(LA.summarize(LA.backtest(fm, g))))
        days = pd.date_range("2026-11-04", periods=12)
        rows = [{"item_id": 500 + i, "day": d, "price_day": 80.0 / (1 + 0.15 * k), "units_day": 300.0 * (1 + 0.3 * k)} for i in range(6) for k, d in enumerate(days)]
        groups = pd.DataFrame({"item_id": [500 + i for i in range(6)], "family": ["FamA", "FamB"] * 3, "group": "x"})
        tables = {"launch_prior": prior, "item_daily": pd.DataFrame(rows), "item_groups": groups}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rep = lake.run_launch_now(None, self._args(), read_table=lambda n: tables.get(n, pd.DataFrame()))
        text = buf.getvalue()
        self.assertEqual(len(rep), 6)
        for needle in ("LAUNCH NOW: 6 young items", "BY FAMILY", "THE 15 MOST-LISTED YOUNG ITEMS", "FamA", "settled_lo80"):
            self.assertIn(needle, text)

    def test_launch_now_says_what_is_missing(self):
        import lake
        for tables, needle in (({}, "run mode launch_backtest"), ({"launch_prior": pd.DataFrame({"x": [1]})}, "run mode daily")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lake.run_launch_now(None, self._args(), read_table=lambda n, t=tables: t.get(n, pd.DataFrame()))
            self.assertIn(needle, buf.getvalue())

    def test_command_line_and_workflow_know_the_new_modes(self):
        import yaml
        import lake
        with mock.patch.object(lake, "run_launch_backtest", return_value=1), mock.patch.object(lake, "run_launch_now", return_value=2), \
                mock.patch.object(lake, "make_store", side_effect=AssertionError("no store needed")):
            self.assertEqual(lake.main(["launch_backtest"]), 1)
            self.assertEqual(lake.main(["launch_now"]), 2)
        with open(os.path.join(_root, ".github", "workflows", "pick_items.yml")) as f:
            wf = yaml.safe_load(f)
        with open(os.path.join(_root, "tools", "pick_items.yml")) as f:
            self.assertEqual(yaml.safe_load(f), wf)
        inputs = wf[True]["workflow_dispatch"]["inputs"]
        save = next(s for s in wf["jobs"]["run"]["steps"] if s.get("name") == "Save results")
        for mode in ("launch_backtest", "launch_now"):
            self.assertIn(mode, inputs["mode"]["options"])
            self.assertIn(f"inputs.mode != '{mode}'", save["if"])


if __name__ == "__main__":
    unittest.main()
