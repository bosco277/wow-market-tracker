import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))
from engine import history as H  # noqa: E402

START = "2022-09-04"


def world(days=700, seed=4, market_sd=0.03):
    """Three groups of 30 items over `days` days.
    Alpha: its own weekly pattern (Saturday dearer, Tuesday cheaper, stock the other way), its own shared price shocks and 12 shared double-price days.
    Beta / Gamma: no weekly pattern, no own shocks. Every item also feels one small market-wide shock.
    Bg0..Bg9: ten small background groups of 8 items with a small shock of their own each, so that "the rest of the market" is a fair control.
    Fresh: 50 items that first appear together on day 400 (price 3x and stock 0.2x of settled, settling over about two weeks) and 9 single new items that start flat."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days)
    dow = dates.dayofweek.to_numpy()
    market = rng.normal(0, market_sd, days)
    shock_a = rng.normal(0, 0.06, days)
    shock_a[rng.choice(days, 12, replace=False)] += 0.45
    weekly = np.zeros(7)
    weekly[5], weekly[1] = 0.04, -0.04
    rows, groups = [], []
    iid = 1000

    def add(item, start_idx, logp, stock, group):
        d = dates[start_idx:]
        rows.append(pd.DataFrame({"item_id": item, "day": d.strftime("%Y-%m-%d"), "price": 10 ** logp[start_idx:] * 10000.0, "quantity": stock[start_idx:]}))
        groups.append(dict(item_id=item, group=group, family="FamA" if group in ("Alpha", "Beta") else "FamC" if group.startswith("Bg") else "FamB"))

    for gname, n, w, sh in (("Alpha", 30, weekly, shock_a), ("Beta", 30, np.zeros(7), 0.0), ("Gamma", 30, np.zeros(7), 0.0)):
        for _ in range(n):
            iid += 1
            logp = np.log10(10 ** rng.uniform(0, 3)) + w[dow] + market + sh + rng.normal(0, 0.015, days)
            stock = 10 ** (4 - w[dow] + rng.normal(0, 0.03, days))
            add(iid, 0, logp, stock, gname)
    for b in range(10):
        shock_b = rng.normal(0, 0.04, days)
        for _ in range(8):
            iid += 1
            logp = np.log10(10 ** rng.uniform(0, 3)) + market + shock_b + rng.normal(0, 0.015, days)
            add(iid, 0, logp, 10 ** (4 + rng.normal(0, 0.03, days)), f"Bg{b}")
    for _ in range(50):                                          # the launch burst
        iid += 1
        age = np.arange(days) - 400
        mult_p = 1 + 2 * np.exp(-np.clip(age, 0, None) / 5.0)
        mult_s = 1 - 0.8 * np.exp(-np.clip(age, 0, None) / 7.0)
        add(iid, 400, np.log10(20.0 * mult_p) + rng.normal(0, 0.02, days), 5000 * mult_s * 10 ** rng.normal(0, 0.03, days), "Fresh")
    for s in (150, 210, 270, 330, 450, 500, 520, 560, 580):      # single new items
        iid += 1
        add(iid, s, np.log10(20.0) + rng.normal(0, 0.02, days), 5000 * 10 ** rng.normal(0, 0.03, days), "Fresh")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(groups)


class TestParts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hist, cls.groups = world()
        cls.res = H.history_report(cls.groups, cls.hist)
        h2, g2 = world(market_sd=0.05)                           # a louder market as well, to see the verdicts hold
        cls.res2 = H.history_report(g2, h2)

    def test_clean_converts_copper_to_gold_and_drops_bad_rows(self):
        h = pd.DataFrame({"item_id": [1, 1, 1, 2], "day": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], "price": [10000, 20000, 0, 5000.0],
                          "quantity": [5, 6, 7, 8]})
        c = H.clean(h)
        self.assertEqual(len(c), 2)                                              # duplicate day collapsed to the last, zero price dropped
        self.assertAlmostEqual(c[c["item_id"] == 1]["price"].iloc[0], 2.0)
        self.assertTrue(H.clean(pd.DataFrame()).empty and H.clean(None).empty)

    def test_detrended_lift_is_zero_for_a_normal_day_and_log_ratio_for_a_spike(self):
        d = pd.date_range("2024-01-01", periods=60)
        M = pd.DataFrame([np.full(60, 10.0)], columns=d, index=[1])
        M.iloc[0, 30] = 20.0
        L = H.detrended(M)
        self.assertAlmostEqual(L.iloc[0, 30], np.log10(2.0), places=6)
        self.assertAlmostEqual(L.iloc[0, 10], 0.0, places=6)

    def test_weekday_rhythm_finds_the_groups_own_pattern_and_says_no_for_the_others(self):
        g = self.res["rhythm"]["price"]["groups"].set_index("group")
        a = g.loc["Alpha"]
        self.assertTrue(a["verdict"].startswith("group-specific rhythm"), a["verdict"])
        self.assertEqual((a["peak"], a["low"]), ("Sat", "Tue"))
        self.assertAlmostEqual(a["swing_pct"], (10 ** 0.08 - 1) * 100, delta=6.0)   # +0.04 vs -0.04 in log10, give or take the sampling noise of the shared shocks
        self.assertGreater(a["stable_r"], 0.9)                                        # the same in the early and late halves
        for name in ("Beta", "Gamma"):
            self.assertTrue(g.loc[name, "verdict"] == "no better than a flat day" or g.loc[name, "verdict"].startswith("too small to use"), g.loc[name, "verdict"])
            self.assertLess(g.loc[name, "swing_pct"], 4.0)
        stock = self.res["rhythm"]["stock"]["groups"].set_index("group")
        self.assertEqual((stock.loc["Alpha", "peak"], stock.loc["Alpha", "low"]), ("Tue", "Sat"))   # stock was built the other way round
        self.assertTrue(stock.loc["Alpha", "verdict"].startswith("group-specific rhythm"))

    def test_a_pattern_that_changes_between_early_and_late_years_is_flagged_unstable(self):
        hist, groups = world(seed=9)
        h = hist.copy()
        h["day"] = pd.to_datetime(h["day"])
        late = (h["day"] >= pd.Timestamp(START) + pd.Timedelta(days=350)) & h["item_id"].between(1001, 1030)
        sat_shift = ((h["day"].dt.dayofweek == 5) | (h["day"].dt.dayofweek == 1)) & late
        h.loc[sat_shift, "price"] *= np.where(h.loc[sat_shift, "day"].dt.dayofweek == 5, 10 ** -0.08, 10 ** 0.08)   # late years: Saturday cheaper, Tuesday dearer
        r = H.history_report(groups, h.assign(day=h["day"].dt.strftime("%Y-%m-%d")))
        a = r["rhythm"]["price"]["groups"].set_index("group").loc["Alpha"]
        self.assertLess(a["stable_r"], H.STABLE_R)
        self.assertIn("not stable across years", a["verdict"])

    def test_moving_together_beyond_the_market_and_spikes_together(self):
        for res in (self.res, self.res2):
            g = res["comove"]["groups"].set_index("group")
            a, b = g.loc["Alpha"], g.loc["Beta"]
            self.assertEqual(a["price_verdict"], "moves together beyond the market")
            self.assertGreater(a["price_extra"], 0.2)
            self.assertEqual(b["price_verdict"], "moves with the whole market")     # Alpha's loud shocks do not make Beta look like it moves on its own
            self.assertLess(abs(b["price_extra"]), 0.1)
            self.assertEqual(g.loc["Bg3", "price_verdict"], "moves together beyond the market")   # a small group with a shock of its own is still found
        self.assertEqual(a["price_spike_verdict"], "spikes together beyond the market")
        self.assertGreater(a["price_events"], 100)
        self.assertEqual(self.res2["comove"]["overall"].iloc[0]["price_verdict"], "items share market-wide moves")
        self.assertIn("families", self.res2["comove"])

    def test_fresh_items_are_found_in_the_burst_and_the_curves_have_the_built_in_shape(self):
        f = self.res["fresh"]
        self.assertEqual(f["new_items"], 59)
        weeks = dict(f["burst_weeks"])
        wk = pd.Timestamp(START) + pd.Timedelta(days=400)
        wk = wk - pd.Timedelta(days=wk.dayofweek)
        self.assertEqual(weeks, {wk: 50})
        burst = f["cohorts"]["launch burst"]
        self.assertEqual(burst["items"], 50)
        t = burst["table"].set_index("age_days")
        self.assertAlmostEqual(t.loc[0, "price_x"], 3.0, delta=0.3)
        self.assertAlmostEqual(t.loc[0, "stock_x"], 0.2, delta=0.05)
        self.assertAlmostEqual(t.loc[56, "price_x"], 1.0, delta=0.05)
        self.assertLess(t.loc[0, "price_low"], t.loc[0, "price_x"] + 1e-9)
        self.assertGreaterEqual(t.loc[0, "price_high"], t.loc[0, "price_x"] - 1e-9)
        for k in ("settle_price", "settle_stock"):
            self.assertTrue(8 <= burst[k] <= 20, (k, burst[k]))
        trickle = f["cohorts"]["trickle (single new items)"]
        self.assertEqual(trickle["items"], 9)
        self.assertAlmostEqual(trickle["table"].set_index("age_days").loc[0, "price_x"], 1.0, delta=0.1)
        fam = f["by_family"].set_index("family")
        self.assertEqual(list(fam.index), ["FamB"])
        self.assertEqual(fam.loc["FamB", "items"], 50)
        self.assertAlmostEqual(fam.loc["FamB", "price_day0_x"], 3.0, delta=0.3)

    def test_items_are_counted_and_short_histories_are_left_out_of_rhythm(self):
        self.assertEqual(self.res["all_items"], 229)
        self.assertEqual(self.res["items"], 174)                                       # 90 old items + the 4 single new items that have a year of days; the burst items do not

    def test_nothing_to_report_without_history_or_without_enough_new_items(self):
        self.assertEqual(H.history_report(self.groups, pd.DataFrame()), {})
        short = self.hist[self.hist["item_id"] <= 1030]                               # Alpha only: nothing is new
        r = H.history_report(self.groups, short)
        self.assertEqual(r["fresh"]["new_items"], 0)
        self.assertNotIn("cohorts", r["fresh"])


def sticky_world(days=520, seed=5):
    """Sticky: 12 items whose price sits EXACTLY at its normal level except on 40% of Saturdays (the same Saturdays for all of them), when it is 20-30% higher: a weekday
    rhythm that a median-day test cannot see. Weekly: 12 items with loud day-to-day price noise and a small shock shared by the whole calendar week (invisible day by day,
    visible in weekly averages). Bg0..Bg7: 8 items each with the loud noise only."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days)
    dow = dates.dayofweek.to_numpy()
    sat = np.flatnonzero(dow == 5)
    bump = np.zeros(days, bool)
    bump[rng.choice(sat, size=int(0.4 * len(sat)), replace=False)] = True
    week_id = ((dates - pd.to_timedelta(dow, unit="D")).normalize()).factorize()[0]
    shock_w = rng.normal(0, 0.05, week_id.max() + 1)[week_id]
    rows, groups = [], []
    iid = 2000

    def add(price, stock, group, family):
        nonlocal iid
        iid += 1
        rows.append(pd.DataFrame({"item_id": iid, "day": dates.strftime("%Y-%m-%d"), "price": price * 10000.0, "quantity": stock}))
        groups.append(dict(item_id=iid, group=group, family=family))

    for _ in range(12):
        add(20.0 * (1 + rng.uniform(0.2, 0.3) * bump), np.full(days, 5000.0), "Sticky", "FamS")
    for _ in range(12):
        add(10 ** (1.3 + shock_w + rng.normal(0, 0.2, days)), 5000 * 10 ** rng.normal(0, 0.05, days), "Weekly", "FamW")
    for b in range(8):
        for _ in range(8):
            add(10 ** (1.3 + rng.normal(0, 0.2, days)), 5000 * 10 ** rng.normal(0, 0.05, days), f"Bg{b}", "FamB")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(groups)


class TestPartFourFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hist, cls.groups = sticky_world()
        cls.res = H.history_report(cls.groups, cls.hist)

    def test_a_price_that_is_flat_most_days_hides_its_rhythm_from_the_median_day_but_not_from_the_average(self):
        med = self.res["rhythm"]["price"]["groups"].set_index("group").loc["Sticky"]
        self.assertLess(med["swing_pct"], 0.5)                                         # median day: exactly normal on every weekday, nothing to measure
        self.assertNotIn("group-specific", med["verdict"])
        avg = self.res["rhythm"]["price_mean"]["groups"].set_index("group").loc["Sticky"]
        self.assertEqual((avg["peak"],), ("Sat",))
        self.assertGreater(avg["swing_pct"], 5.0)
        self.assertTrue(avg["verdict"].startswith("group-specific rhythm"), avg["verdict"])
        self.assertGreater(avg["stable_r"], 0.9)
        self.assertEqual(set(self.res["rhythm"]), {"price", "price_mean", "stock"})

    def test_flat_day_share_says_how_sticky_each_family_is(self):
        f = self.res["flat"]["price"].set_index("family")
        self.assertEqual(f.index[0], "ALL ITEMS")
        self.assertGreater(f.loc["FamS", "flat_median"], 0.9)
        self.assertLess(f.loc["FamB", "flat_median"], 0.05)
        self.assertLessEqual(f.loc["FamS", "flat_low"], f.loc["FamS", "flat_median"])
        self.assertLessEqual(f.loc["FamS", "flat_median"], f.loc["FamS", "flat_high"])
        self.assertIn("stock", self.res["flat"])

    def test_weekly_averages_need_four_days_and_drop_empty_weeks(self):
        d = pd.date_range("2024-01-01", periods=10)                                      # Mon Jan 1 .. Wed Jan 10: a full week and a 3-day week
        L = pd.DataFrame([np.arange(10.0)], columns=d, index=[1])
        W = H.weekly(L)
        self.assertEqual(W.shape, (1, 1))
        self.assertAlmostEqual(W.iloc[0, 0], 3.0)                                        # mean of 0..6
        self.assertEqual(W.columns[0], pd.Timestamp("2024-01-01"))
        L.iloc[0, 3] = np.nan                                                            # 6 days seen still counts; with only 3 seen it would not
        self.assertEqual(H.weekly(L).shape, (1, 1))
        L.iloc[0, [1, 2, 4]] = np.nan
        self.assertEqual(H.weekly(L).shape[1], 0)

    def test_a_move_that_lasts_a_week_shows_in_weekly_averages_but_not_day_by_day(self):
        day = self.res["comove"]["groups"].set_index("group").loc["Weekly"]
        wk = self.res["comove_weekly"]["groups"].set_index("group").loc["Weekly"]
        self.assertLess(day["price_corr_group"], 0.25)
        self.assertEqual(day["price_verdict"], "no shared movement")
        self.assertGreater(wk["price_corr_group"], 0.3)
        self.assertGreater(wk["price_corr_group"], 2 * day["price_corr_group"])
        self.assertEqual(wk["price_verdict"], "moves together beyond the market")
        self.assertLess(wk["price_corr_market"], 0.15)
        self.assertNotIn("price_events", self.res["comove_weekly"]["groups"].columns)   # spikes are a day-level idea; the weekly test leaves them out

    def test_fresh_curves_are_followed_to_112_days(self):
        hist, groups = world()
        t = H.history_report(groups, hist)["fresh"]["cohorts"]["launch burst"]["table"].set_index("age_days")
        self.assertEqual(max(t.index), 112)
        self.assertAlmostEqual(t.loc[112, "price_x"], 1.0, delta=0.05)
        self.assertGreaterEqual(t.loc[84, "price_x"], 0.95)

    def test_the_report_prints_the_new_sections_and_no_line_a_spreadsheet_would_break(self):
        import lake
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            lake.run_undermine_test(None, lake.argparse.Namespace(out=d, ids=""), read_table=lambda n: {"market_history": self.hist, "item_groups": self.groups}.get(n, pd.DataFrame()))
        text = buf.getvalue()
        for needle in ("PRICE (median day)", "PRICE (average of all days)", "WITHIN 0.5% OF ITS OWN NORMAL LEVEL", "PART 4B-WEEKLY", "-- WEEKLY: BY GROUP --", "Sticky"):
            self.assertIn(needle, text)
        self.assertFalse(any(line.startswith("==") for line in text.splitlines()))


class TestMarketControl(unittest.TestCase):
    def test_median_of_the_other_groups_ignores_a_loud_group_and_falls_back_with_few_groups(self):
        from engine import grouptest as GT
        days = 20
        rows, lab = [], []
        for g, val in (("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 100.0), ("e", 5.0)):    # five groups of 4 items, constant lifts
            for _ in range(4):
                rows.append(np.full(days, val))
                lab.append(g)
        A = np.array(rows)
        V = ~np.isnan(A)
        frame = pd.DataFrame({"group": lab})
        out = GT._group_median_control(A, V, frame, "group")
        self.assertTrue(np.allclose(out[np.array(lab) == "a"], np.median([2.0, 3.0, 100.0, 5.0])))   # median of the others, not their (huge) average
        self.assertTrue(np.allclose(out[np.array(lab) == "d"], np.median([1.0, 2.0, 3.0, 5.0])))
        few = GT._group_median_control(A[:12], V[:12], frame.iloc[:12], "group")                    # three groups: only two others
        self.assertTrue(np.isnan(few).all())
        small = GT._group_median_control(A, V, frame.assign(group=[f"g{i}" for i in range(20)]), "group")   # groups of one item: none qualifies
        self.assertTrue(np.isnan(small).all())


class TestCommand(unittest.TestCase):
    def _run(self, tables):
        import lake
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            args = lake.argparse.Namespace(out=d, ids="")
            res = lake.run_undermine_test(None, args, read_table=lambda n: tables.get(n, pd.DataFrame()))
            written = os.listdir(d)
        return res, buf.getvalue(), written

    def test_prints_the_three_parts_aggregates_only_and_writes_no_files(self):
        hist, groups = world()
        res, text, written = self._run({"market_history": hist, "item_groups": groups})
        self.assertTrue(res)
        for needle in ("4A", "4B", "4C", "group-specific rhythm", "moves together beyond the market", "launch burst", "survivor"):
            self.assertIn(needle, text)
        self.assertEqual(written, [])                                                   # nothing is written next to the (downloadable) results
        self.assertNotIn("1001", text.split("4A")[1])                                   # item ids never appear: group-level numbers only

    def test_says_what_to_run_first_when_tables_are_missing(self):
        res, text, _ = self._run({})
        self.assertFalse(res)
        self.assertIn("run mode undermine", text)
        hist, groups = world(days=400)
        res, text, _ = self._run({"market_history": hist})
        self.assertIn("item_groups", text)


if __name__ == "__main__":
    unittest.main()
