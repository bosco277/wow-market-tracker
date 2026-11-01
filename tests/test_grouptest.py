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
from engine import daily as D  # noqa: E402
from engine import grouptest as GT  # noqa: E402


def fake_tables(seed=3):
    """Three groups of 30: Tight (alike), Noise (items unrelated to each other), Mixed (two id bands that behave very differently)."""
    rng = np.random.default_rng(seed)
    rows = []
    iid = 1000

    def add(group, family, sales, price, n, id0, conf="High", name=None, tier=0, sib=1, quality="COMMON", use=""):
        nonlocal iid
        for k in range(n):
            iid += 1
            rows.append(dict(item_id=id0 + k, name=name or f"{group} {iid}", group=group, family=family, use=use, quality=quality, siblings=sib,
                             tier_guess=tier, daily_est=float(sales(k)), floor_gold=float(price(k)), confidence=conf if k % 5 else "Low"))

    add("Consumables: Tight", "Consumables", lambda k: 5000 * 10 ** rng.normal(0, 0.05), lambda k: 200 * 10 ** rng.normal(0, 0.05), 30, 300000)
    add("Consumables: Noise", "Consumables", lambda k: 10 ** rng.uniform(0, 4), lambda k: 10 ** rng.uniform(0, 3), 30, 310000)
    add("Crafting materials: Mixed", "Crafting materials", lambda k: 1000 * 10 ** rng.normal(0, 0.05) if k < 20 else 10 * 10 ** rng.normal(0, 0.05),
        lambda k: 5.0, 40, 10000)                                   # item ids are rewritten below so the busy half sits in a different id band
    items = pd.DataFrame(rows)
    # make the Mixed group split cleanly by id band: first 20 in band "0k", last 20 in band "40k"
    mixed = items["group"] == "Crafting materials: Mixed"
    ids = np.where(np.arange(mixed.sum()) < 20, 1000 + np.arange(mixed.sum()), 40000 + np.arange(mixed.sum()))
    items.loc[mixed, "item_id"] = ids
    # three-tier names: tier 2 costs double and sells half of tier 1
    t = []
    for n in range(8):
        for tier in (1, 2, 3):
            t.append(dict(item_id=500000 + n * 10 + tier, name=f"Flask {n}", group="Consumables: Flasks", family="Consumables", use="", quality="COMMON",
                          siblings=3, tier_guess=tier, daily_est=100.0 / tier, floor_gold=10.0 * tier, confidence="High"))
    items = pd.concat([items, pd.DataFrame(t)], ignore_index=True)
    groups = items[["item_id", "name", "group", "family", "use", "quality", "siblings", "tier_guess"]].copy()
    return groups, items[["item_id", "floor_gold", "daily_est", "confidence"]].copy()


class TestLooMedian(unittest.TestCase):
    def test_never_uses_the_item_itself(self):
        df = pd.DataFrame({"k": ["a"] * 5, "v": [1.0, 2.0, 3.0, 4.0, 100.0]})
        out = GT._loo_median(df, ["k"], "v")
        self.assertAlmostEqual(out[4], 2.5)            # the 100 is left out of its own guess
        self.assertAlmostEqual(out[0], 3.5)            # median of 2,3,4,100

    def test_too_few_peers_gives_nothing(self):
        df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
        self.assertTrue(np.isnan(GT._loo_median(df, ["k"], "v")).all())
        self.assertEqual(len(GT._loo_median(pd.DataFrame({"k": [], "v": []}), ["k"], "v")), 0)


class TestGroupTest(unittest.TestCase):
    def setUp(self):
        self.groups, self.items = fake_tables()
        self.res = GT.grouptest(self.groups, self.items)

    def test_a_tight_group_beats_guessing_and_noise_does_not(self):
        g = self.res["groups"].set_index("group")
        self.assertLess(g.loc["Consumables: Tight", "sales_off_group"], 1.2)
        self.assertGreater(g.loc["Consumables: Tight", "sales_gain"], 3)
        self.assertTrue(g.loc["Consumables: Tight", "verdict"].startswith(("adds", "helps")))
        self.assertGreater(g.loc["Consumables: Noise", "sales_off_group"], 2)           # unrelated items: the group says little
        self.assertNotIn("adds to its family", g.loc["Consumables: Noise", "verdict"])

    def test_small_groups_get_no_verdict(self):
        groups = pd.concat([self.groups, pd.DataFrame([dict(item_id=900001, name="Lonely", group="Key: Key", family="Other", use="", quality="COMMON", siblings=1, tier_guess=0)])])
        items = pd.concat([self.items, pd.DataFrame([dict(item_id=900001, floor_gold=5.0, daily_est=3.0, confidence="High")])])
        g = GT.grouptest(groups, items)["groups"].set_index("group")
        self.assertEqual(g.loc["Key: Key", "verdict"], "too few to judge")
        self.assertEqual(g.loc["Key: Key", "scored"], 0)

    def test_low_confidence_numbers_are_flagged_when_reliable_items_are_too_few(self):
        items = self.items.copy()
        keep = items["item_id"].between(310000, 310999)
        items.loc[keep, "confidence"] = "Low"
        g = GT.grouptest(self.groups, items)["groups"].set_index("group")
        self.assertIn("Low-confidence", g.loc["Consumables: Noise", "verdict"])

    def test_coarse_group_is_split_by_id_band(self):
        c = self.res["coarse"].set_index("group")
        self.assertIn("Crafting materials: Mixed", c.index)
        row = c.loc["Crafting materials: Mixed"]
        self.assertEqual(row["best_split"], "id_band")
        self.assertGreater(row["gain_id_band"], 3)
        self.assertIn("0k", row["slices"])
        self.assertIn("g (n=", row["slices"])                                          # each slice also shows its median price
        self.assertNotIn("Consumables: Tight", c.index)                                # alike items are not "coarse"

    def test_tiers(self):
        r = self.res["tiers_ratios"].set_index("tier_guess")
        self.assertAlmostEqual(r.loc[2, "sales_vs_lowest_tier"], 0.5)
        self.assertAlmostEqual(r.loc[3, "price_vs_lowest_tier"], 3.0)
        guess = self.res["tiers_guess"].set_index("what")
        self.assertIn("sales per day", guess.index)
        self.assertGreaterEqual(guess.loc["sales per day", "items"], 24)

    def test_empty_and_missing_inputs_do_not_crash(self):
        self.assertEqual(GT.grouptest(pd.DataFrame(), self.items), {})
        self.assertEqual(GT.grouptest(self.groups, pd.DataFrame({"item_id": [1]})), {})
        bad = self.items.assign(daily_est=0.0)
        self.assertEqual(GT.grouptest(self.groups, bad), {})


def fake_hourly(groups, days=14):
    """Hourly rows for the Tight group (sells in the evening, hour 20 Eastern) and the Noise group (sells at 8 in the morning)."""
    ts = pd.date_range("2026-10-05", periods=days * 24, freq="h", tz="America/New_York").tz_convert("UTC")
    hour = ts.tz_convert("America/New_York").hour
    rng = np.random.default_rng(5)
    parts = []
    for g, peak in (("Consumables: Tight", 20), ("Consumables: Noise", 8)):
        for iid in groups.loc[groups["group"] == g, "item_id"]:
            sold = 2.0 + 20.0 * (hour == peak) + rng.uniform(0, 1, len(ts))
            parts.append(pd.DataFrame({"item_id": iid, "ts": ts, "floor_gold": 10.0, "units_listed": 100, "sold_est": sold,
                                       "spike": False, "blip": False}))
    return pd.concat(parts, ignore_index=True)


class TestEra(unittest.TestCase):
    def test_era_follows_item_id(self):
        groups, items = fake_tables()
        m = GT.prepare(groups, items).set_index("item_id")
        self.assertEqual(m.loc[1005, "era"], "old (<120k)")
        self.assertEqual(m.loc[300001, "era"], "current (240k+)")
        self.assertEqual(m.loc[500011, "era"], "current (240k+)")
        edges = pd.DataFrame({"item_id": [119999, 120000, 199999, 200000, 239999, 240000], "name": "x", "group": "g", "family": "f"})
        e = GT.prepare(edges, pd.DataFrame({"item_id": edges["item_id"], "floor_gold": 1.0, "daily_est": 1.0, "confidence": "High"}))
        self.assertEqual(list(e["era"]), [GT.ERA_LABELS[i] for i in (0, 1, 1, 2, 2, 3)])


class TestRhythm(unittest.TestCase):
    def test_groups_with_their_own_daily_pattern_beat_a_flat_day_and_the_all_items_average(self):
        groups, items = fake_tables()
        rh = GT.grouptest(groups, items, fake_hourly(groups))["rhythm"]
        g = rh["groups"].set_index("group")
        self.assertGreater(g.loc["Consumables: Tight", "hour_vs_flat"], 3)
        self.assertGreater(g.loc["Consumables: Tight", "hour_vs_all"], 3)             # the two groups peak at different hours
        self.assertEqual(g.loc["Consumables: Tight", "hour_verdict"], "group-specific rhythm")
        self.assertEqual(rh["items"], 60)
        self.assertIn("overall", rh)

    def test_everyone_sharing_one_rhythm_is_called_what_it_is(self):
        groups, items = fake_tables()
        h = fake_hourly(groups)
        noise = groups.loc[groups["group"] == "Consumables: Noise", "item_id"]
        h = h[~h["item_id"].isin(noise)]                                              # only the evening group is left
        rh = GT.grouptest(groups, items, h)["rhythm"]
        g = rh["groups"].set_index("group")
        self.assertEqual(g.loc["Consumables: Tight", "hour_verdict"], "same rhythm as every item")

    def test_flagged_hours_and_short_histories_are_left_out(self):
        groups, items = fake_tables()
        h = fake_hourly(groups, days=14)
        short = h[h["item_id"] == h["item_id"].iloc[0]].head(60)                        # 60 readings: under the 96 needed
        prof = GT.hourly_profiles(short)
        self.assertEqual(prof, {})
        h["spike"] = True
        self.assertEqual(GT.hourly_profiles(h), {})                                      # every hour flagged: nothing usable
        self.assertEqual(GT.hourly_profiles(pd.DataFrame()), {})
        self.assertEqual(GT.hourly_profiles(pd.DataFrame({"item_id": [1]})), {})

    def test_profiles_are_shares(self):
        groups, items = fake_tables()
        prof = GT.hourly_profiles(fake_hourly(groups))
        self.assertTrue(np.allclose(prof["sold_hour"].sum(axis=1), 1.0))
        self.assertNotIn("sold_dow", prof)
        tight = groups.loc[groups["group"] == "Consumables: Tight", "item_id"].iloc[0]
        self.assertEqual(int(prof["sold_hour"].loc[tight].idxmax()), 20)


def lift_world(market_wide=False, days=30, seed=11):
    """36 items in three groups of 12 with very different sizes (10 to 1000 units a day). Cloth and Herb each have their own good and bad days
    (a shared factor; Cloth also has two big spike days); Solo items move independently. With market_wide, every item shares one factor instead."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-10-01", periods=days * 24, freq="h", tz="America/New_York").tz_convert("UTC")
    day = np.repeat(np.arange(days), 24)
    shared = {g: 10 ** (0.12 * rng.standard_normal(days)) for g in ("Cloth", "Herb")}
    shared["Cloth"][[d for d in (5, 17) if d < days]] = 3.0
    market = 10 ** (0.15 * rng.standard_normal(days))
    rows, grp = [], []
    iid = 5000
    for g in ("Cloth", "Herb", "Solo"):
        for k in range(12):
            iid += 1
            level = 10 ** rng.uniform(1.0, 3.0)
            f = market if market_wide else (shared[g] if g in shared else 10 ** (0.12 * rng.standard_normal(days)))
            sold = level / 24 * f[day] * 10 ** (0.1 * rng.standard_normal(len(ts)))
            price = 10.0 * (f[day] if g != "Solo" or market_wide else 1.0)
            rows.append(pd.DataFrame({"item_id": iid, "ts": ts, "floor_gold": price, "units_listed": 100, "sold_est": sold, "spike": False, "blip": False}))
            grp.append(dict(item_id=iid, name=f"{g} {k}", group=f"Crafting: {g}", family="Crafting"))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(grp)


class TestLiftingTogether(unittest.TestCase):
    def test_daily_series_drops_quantities_and_thin_items(self):
        h, groups = lift_world()
        ds = GT.daily_series(D.daily_table(h))
        self.assertEqual(ds["items"], 36)
        self.assertEqual(ds["days"], 30)
        self.assertLess(abs(ds["sales"].median(axis=1)).max(), 0.05)                  # each item's lift is measured against its own normal day
        quiet = h[h["item_id"] == 5001].assign(sold_est=0.01)
        h2 = pd.concat([h[h["item_id"] != 5001], quiet])
        self.assertNotIn(5001, GT.daily_series(D.daily_table(h2))["sales"].index)       # under 3 units a day: left out
        few = h[(h["item_id"] != 5002) | (h["ts"] < h["ts"].min() + pd.Timedelta(days=4))]
        self.assertNotIn(5002, GT.daily_series(D.daily_table(few))["sales"].index)      # under 6 days: left out
        gappy = h[~((h["item_id"] == 5003) & (pd.to_datetime(h["ts"]).dt.hour % 2 == 0))]
        self.assertNotIn(5003, GT.daily_series(D.daily_table(gappy))["sales"].index)    # every day missing half its hours: no usable day
        self.assertEqual(GT.daily_series(pd.DataFrame()), {})
        self.assertEqual(GT.daily_series(h), {})                                       # the hourly table is not the daily one: no crash, nothing scored

    def test_group_with_its_own_good_days_beats_the_market_and_independent_items_do_not(self):
        h, groups = lift_world()
        cm = GT.comove_report(groups, D.daily_table(h))
        g = cm["groups"].set_index("group")
        self.assertGreater(g.loc["Crafting: Cloth", "sales_corr_group"], 0.7)
        self.assertGreater(g.loc["Crafting: Cloth", "sales_extra"], 0.3)
        self.assertEqual(g.loc["Crafting: Cloth", "sales_verdict"], "moves together beyond the market")
        self.assertEqual(g.loc["Crafting: Solo", "sales_verdict"], "no shared movement")
        self.assertGreater(g.loc["Crafting: Cloth", "price_extra"], 0.3)

    def test_spikes_on_the_same_days_are_found(self):
        h, groups = lift_world()
        g = GT.comove_report(groups, D.daily_table(h))["groups"].set_index("group")
        self.assertGreaterEqual(g.loc["Crafting: Cloth", "spike_events"], 20)           # 12 items x 2 spike days
        self.assertGreater(g.loc["Crafting: Cloth", "spike_group"], 5)
        self.assertGreater(g.loc["Crafting: Cloth", "spike_group"], 1.15 * g.loc["Crafting: Cloth", "spike_market"])
        self.assertEqual(g.loc["Crafting: Cloth", "spike_verdict"], "spikes together beyond the market")
        self.assertEqual(g.loc["Crafting: Solo", "spike_verdict"], "too few spike days")

    def test_a_market_wide_factor_is_not_credited_to_the_group(self):
        h, groups = lift_world(market_wide=True)
        cm = GT.comove_report(groups, D.daily_table(h))
        g = cm["groups"].set_index("group")
        self.assertEqual(g.loc["Crafting: Cloth", "sales_verdict"], "moves with the whole market")
        self.assertLess(g.loc["Crafting: Cloth", "sales_extra"], GT.EXTRA_CORR)
        self.assertEqual(cm["overall"].loc[0, "sales_verdict"], "items share market-wide lifts")

    def test_nothing_to_score_gives_empty(self):
        h, groups = lift_world(days=5)
        self.assertEqual(GT.comove_report(groups, D.daily_table(h)), {})               # 5 days: under the 6 needed

    def test_a_short_history_is_labelled_provisional(self):
        import lake
        h, groups = lift_world(days=8)
        items = pd.DataFrame({"item_id": groups["item_id"], "floor_gold": 5.0, "daily_est": 100.0, "confidence": "High"})
        tables = {"items": items, "item_groups": groups.assign(use="", quality="COMMON", siblings=1, tier_guess=0), "item_daily": D.daily_table(h)}
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            res = lake.run_grouptest(None, lake.argparse.Namespace(out=d), read_table=lambda n: tables.get(n, pd.DataFrame()))
        self.assertEqual(res["comove"]["days"], 8)
        self.assertIn("PROVISIONAL: the history holds only 8 days", buf.getvalue())


class TestDayOfWeek(unittest.TestCase):
    def test_weekday_pattern_from_the_daily_history(self):
        days = pd.date_range("2026-10-05", periods=28, freq="D")                       # four full weeks, Monday first
        rows = []
        for iid in range(1, 25):
            g = "A" if iid <= 12 else "B"
            for d in days:
                big = (d.dayofweek == 5) if g == "A" else (d.dayofweek == 1)           # A sells on Saturdays, B on Tuesdays
                rows.append(dict(item_id=iid, day=d.strftime("%Y-%m-%d"), hours=24, sold_day=10.0 * (1 + 4 * big) * iid, price_day=5.0, units_day=100, flagged_hours=0))
        daily = pd.DataFrame(rows)
        groups = pd.DataFrame({"item_id": range(1, 25), "name": "x", "group": ["Crafting: A"] * 12 + ["Crafting: B"] * 12, "family": "Crafting"})
        prof = GT.dow_profile(daily)
        self.assertEqual(len(prof), 24)
        self.assertEqual(int(prof.loc[1].idxmax()), 5)
        self.assertEqual(int(prof.loc[20].idxmax()), 1)
        rh = GT.shape_report(groups, None, daily)
        g = rh["groups"].set_index("group")
        self.assertGreater(g.loc["Crafting: A", "dow_vs_flat"], 2)
        self.assertGreater(g.loc["Crafting: A", "dow_vs_all"], 2)
        self.assertEqual(g.loc["Crafting: A", "dow_verdict"], "group-specific rhythm")
        short = daily[daily["day"] < "2026-10-15"]                                      # 10 days: under the 14 needed
        self.assertTrue(GT.dow_profile(short).empty)


class TestCommand(unittest.TestCase):
    def test_prints_and_writes_files(self):
        import lake
        groups, items = fake_tables()
        lh, lg = lift_world()
        tables = {"items": items, "item_groups": groups, "item_hourly": fake_hourly(groups), "item_daily": D.daily_table(lh)}
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            args = lake.argparse.Namespace(out=d)
            with contextlib.redirect_stdout(buf):
                res = lake.run_grouptest(None, args, read_table=lambda n: tables.get(n, pd.DataFrame()))
            self.assertTrue(res)
            for f in ("grouptest_groups.csv", "grouptest_families.csv", "grouptest_coarse.csv", "grouptest_tiers.csv", "grouptest_rhythm_groups.csv",
                      "grouptest_rhythm_overall.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
        text = buf.getvalue()
        for s in ("PART 1", "PART 2", "GROUP TEST", "BY GROUP", "Consumables: Tight", "SAME-NAME QUALITY TIERS", "WHAT SPLITS THEM", "by id_band", "DAILY RHYTHM", "ALL ITEMS",
                  "group-specific rhythm", "items share a common daily pattern"):
            self.assertIn(s, text)

    def test_runs_without_an_hourly_table_and_says_so(self):
        import lake
        groups, items = fake_tables()
        tables = {"items": items, "item_groups": groups}
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            res = lake.run_grouptest(None, lake.argparse.Namespace(out=d), read_table=lambda n: tables.get(n, pd.DataFrame()))
        self.assertTrue(res)
        self.assertIn("Lifting-together test skipped: engine/item_daily.parquet does not exist yet", buf.getvalue())
        self.assertIn("Daily-rhythm test skipped", buf.getvalue())

    def test_missing_tables_say_what_to_run(self):
        import lake
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = lake.run_grouptest(None, lake.argparse.Namespace(out="x"), read_table=lambda n: pd.DataFrame())
        self.assertEqual(res, {})
        self.assertIn("groups with upload ticked", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
