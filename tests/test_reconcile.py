import unittest

import numpy as np
import pandas as pd

from helpers import mk  # noqa: F401
from engine.reconcile import read_log, reconcile, summarize, trace_listing, verdict_text

ITEM = 7


def snap(ts, tiers):
    """tiers: {price_copper: units}; one listing per tier (the item the test watches)."""
    rows = [(i + 1, ITEM, q, p, "LONG") for i, (p, q) in enumerate(sorted(tiers.items())) if q > 0]
    return mk(rows, ts=pd.Timestamp(ts))


def log_df(rows):
    cols = ["date", "time_posted", "tz", "item_id", "lane", "list_price", "quantity", "units_sold", "units_expired", "note"]
    return pd.DataFrame(rows, columns=cols)


class TestReadLog(unittest.TestCase):
    def test_times_are_converted_to_utc_and_example_rows_dropped(self):
        df = read_log(log_df([
            ["2026-10-30", "21:05", "EDT", "7", "UNDER", "0.99", "45", "45", "0", "example row: ignore"],
            ["2026-10-30", "21:05", "EDT", "7", "UNDER", "0.99", "45", "", "", ""],
            ["2026-11-02", "09:00", "EST", "7", "SAME", "1.00", "45", "30", "15", "abnormal market"],
        ]))
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "posted_utc"], pd.Timestamp("2026-10-31 01:05"))
        self.assertEqual(df.loc[1, "posted_utc"], pd.Timestamp("2026-11-02 14:00"))
        self.assertEqual(int(df.loc[0, "price_copper"]), 9900)
        self.assertTrue(np.isnan(df.loc[0, "units_sold"]))
        self.assertFalse(bool(df.loc[0, "done"]))
        self.assertTrue(bool(df.loc[1, "abnormal"]))

    def test_unknown_time_zone_is_an_error_not_a_guess(self):
        with self.assertRaises(ValueError):
            read_log(log_df([["2026-10-30", "21:05", "PDT", "7", "UNDER", "0.99", "45", "", "", ""]]))


class TestReconcile(unittest.TestCase):
    def hourly(self, fills):
        """fills: list of tier dicts, one per snapshot at 00:33, 01:33, ... UTC on 2026-10-31."""
        return [snap(pd.Timestamp("2026-10-31 00:33") + pd.Timedelta(hours=k), t) for k, t in enumerate(fills)]

    def test_a_visible_listing_appears_and_clears_and_an_invisible_one_is_counted(self):
        base = {10000: 100, 11000: 500}
        fills = [base,                                   # 00:33 before the post
                 {**base, 9900: 45},                     # 01:33 our 45 at 0.99 are there
                 {**base, 9900: 20},                     # 02:33
                 base,                                   # 03:33 gone
                 base, base, base]
        snaps = self.hourly(fills)
        log = read_log(log_df([["2026-10-30", "21:05", "EDT", str(ITEM), "UNDER", "0.99", "45", "45", "0", ""]]))
        per, s = reconcile(snaps, log)
        r = per.iloc[0]
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["appeared_share"], 1.0)
        self.assertFalse(bool(r["invisible"]))
        self.assertAlmostEqual(r["observed_hours_to_clear"], 3 + 33 / 60 - 1 - 5 / 60, places=2)   # 01:05 -> 03:33
        # same units, but they sold before the 01:33 snapshot: the snapshot after posting shows nothing at 0.99
        fills2 = [base] * 7
        per2, s2 = reconcile(self.hourly(fills2), log)
        self.assertTrue(bool(per2.iloc[0]["invisible"]))
        self.assertEqual(s2["invisible_count"], 1)
        self.assertEqual(s2["invisible_share"], 1.0)

    def test_a_listing_with_no_snapshot_nearby_is_marked_unusable_not_guessed(self):
        snaps = self.hourly([{10000: 100}] * 3)
        log = read_log(log_df([["2026-10-30", "21:05", "EDT", str(ITEM), "UNDER", "0.99", "45", "45", "0", ""]]))
        log["posted_utc"] = pd.Timestamp("2026-10-31 20:05")
        per, s = reconcile(snaps, log)
        self.assertNotEqual(per.iloc[0]["status"], "ok")
        self.assertEqual(s["usable"], 0)


class TestRateCheckAndFirstInLine(unittest.TestCase):
    def row(self, price, sold=40, q=40, lane="UNDER", done=True):
        return pd.Series(dict(price_copper=price, quantity=float(q), posted_utc=pd.Timestamp("2026-10-31 01:05"), lane=lane,
                              units_sold=float(sold), units_expired=0.0, abnormal=False, done=done, item_id=ITEM))

    def books(self, mine_until_h, floor=10000, floor_units=200, under_price=9900, extra=None):
        """Snapshots 00:33, 01:33, ...: a floor tier of others; our 40 units at `under_price` stay until `mine_until_h`
        hours after the 00:33 snapshot. `extra(k)` may add other (price, units) tiers at snapshot k."""
        out = []
        for k in range(0, 12):
            t = pd.Timestamp("2026-10-31 00:33") + pd.Timedelta(hours=k)
            tiers = {floor: floor_units}
            if 1 <= k < mine_until_h:
                tiers[under_price] = tiers.get(under_price, 0) + 40
            if extra:
                for pr, u in extra(k):
                    tiers[pr] = tiers.get(pr, 0) + u
            prices = np.array(sorted(tiers))
            out.append((t, prices, np.array([tiers[x] for x in prices])))
        return out

    def test_a_first_in_line_listing_gets_a_clearing_window_and_a_prediction_to_compare(self):
        vel = dict(low=20.0, est=40.0, high=80.0)
        t = trace_listing(self.row(9900), self.books(4), partner_qty=0, vel=vel)
        self.assertTrue(t["cheapest_at_posting"] and t["alone"])
        self.assertAlmostEqual(t["appeared_share"], 1.0)
        self.assertAlmostEqual(t["pred_lifo_h_est"], 1.0)                       # 40 units at 40 an hour
        self.assertAlmostEqual(t["still_there_h"], 2 + 28 / 60, places=2)             # last seen at 03:33, posted 01:05
        self.assertAlmostEqual(t["observed_hours_to_clear"], 3 + 28 / 60, places=2)   # gone at 04:33
        self.assertGreater(t["front_share"], 0.99)

    def test_a_listing_that_shares_its_price_with_older_units_is_not_used_for_timing_or_detection(self):
        vel = dict(low=40.0, est=40.0, high=40.0)
        t = trace_listing(self.row(10000, lane="SAME"), self.books(4), partner_qty=0, vel=vel)
        self.assertFalse(t["alone"])
        self.assertTrue(np.isnan(t["appeared_share"]))
        self.assertTrue(np.isnan(t["observed_hours_to_clear"]))
        self.assertFalse(t["invisible"])

    def test_front_share_falls_when_cheaper_listings_arrive_after_you(self):
        vel = dict(low=40.0, est=40.0, high=40.0)
        pushed = lambda k: [(9800, 30)] if k >= 3 else []                       # an undercutter from the 03:33 snapshot on
        t = trace_listing(self.row(9900, sold=0), self.books(99, extra=pushed), partner_qty=0, vel=vel)
        self.assertLess(t["front_share"], 0.5)
        self.assertTrue(np.isnan(t["observed_hours_to_clear"]))

    def row_set(self, specs):
        vel = dict(low=40.0, est=40.0, high=40.0)
        return pd.DataFrame([trace_listing(self.row(9900, sold=sold), self.books(k, extra=ex), 0, vel) for sold, k, ex in specs])

    def test_summary_counts_rate_check_and_unsold_causes_separately(self):
        none = None
        pushed = lambda k: [(9800, 30)] if k >= 3 else []
        per = self.row_set([
            (40, 4, none),           # sold; real clearing 3.x h against a predicted 1 h  -> predicted too FAST
            (40, 2, none),           # sold; cleared by the 02:33 snapshot (1.4 h window 0-1.4 h) vs predicted 1 h -> inside
            (0, 99, none),           # unsold, at the front all the time, predicted to sell inside 12 h -> overstated rate
            (0, 99, pushed),         # unsold, pushed back
        ])
        s = summarize(per)
        self.assertEqual(s["rate_check_n"], 2)
        self.assertEqual(s["rate_pred_too_fast"], 1)
        self.assertEqual(s["rate_inside"], 1)
        self.assertEqual(s["unsold_front_n"], 2)
        self.assertEqual(s["unsold_front_stayed"], 1)
        self.assertEqual(s["unsold_front_pushed_back"], 1)
        self.assertEqual(s["unsold_front_stayed_but_predicted_to_sell"], 1)
        text = " ".join(verdict_text(s))
        self.assertIn("RATE CHECK", text)
        self.assertIn("pushed back", text)
        self.assertIn("QUEUE ORDER: not read from this log", text)


PASTED = "\n".join([
    "date\ttime_posted\ttz\tItem\titem_id\tlane\tlist_price\tquantity\tunits_sold\tunits_expired\tNote 1",
    "2026-10-30\t8:22 AM\tEDT\tIceweb Spider Silk\t42253\tUNDER\t0.99\t45\t45\t\tsold before 11:AM",
    "2026-10-30\t8:22 AM\tEDT\tIceweb Spider Silk\t42253\tSAME\t1\t45\t\t45\t",
    "2026-10-30\t8:25 AM\tEDT\tMystic Sapphire\t192841\tbelow\t13.56\t40\t\t40\t",
    "2026-10-30\t8:23 AM\tEDT\tMystic Sapphire\t192841\tabove\t13.57\t40\t\t40\t",
    "2026-10-30\t11:11 AM\tEDT\tSwiftness Potion\t2459\tUNDER\t5.44\t30\t7\t\t7 as of 3 PM",
    "2026-10-30\t11:11 AM\tEDT\tSwiftness Potion\t2459\tSAME\t5.45\t30\t\t\t",
    "2026-10-30\t11:13 AM\tEDT\tAlgari Mana Oil\t224107\tUNDER\t2.75\t75\t75\t\t75 as of 3 PM",
    "2026-10-30\t11:18 AM\tEDT\tProfaned Tinderbox\t221758\tunder\t2.99\t35\t\t\t",
])


class TestPastedLog(unittest.TestCase):
    def setUp(self):
        import io
        self.df = read_log(io.StringIO(PASTED))

    def test_the_log_as_pasted_from_a_sheet_is_read(self):
        self.assertEqual(len(self.df), 8)
        iceweb = self.df[self.df["item_id"] == 42253].sort_values("price_copper")
        self.assertEqual(list(iceweb["units_sold"]), [45.0, 0.0])            # only "expired 45" filled in: nothing sold
        self.assertEqual(list(iceweb["units_expired"]), [0.0, 45.0])
        self.assertEqual(list(self.df[self.df["item_id"] == 192841].sort_values("price_copper")["lane"]), ["UNDER", "SAME"])
        first = self.df.iloc[0]
        self.assertEqual(first["posted_utc"], pd.Timestamp("2026-10-30 12:22"))   # 8:22 AM EDT

    def test_partial_counts_marked_as_of_are_in_progress_but_a_full_count_is_a_result(self):
        sw = self.df[(self.df["item_id"] == 2459) & (self.df["lane"] == "UNDER")].iloc[0]
        self.assertTrue(bool(sw["in_progress"]))
        self.assertFalse(bool(sw["done"]))
        oil = self.df[self.df["item_id"] == 224107].iloc[0]
        self.assertFalse(bool(oil["in_progress"]))
        self.assertTrue(bool(oil["done"]))
        self.assertEqual(float(oil["units_sold"]), 75.0)
        self.assertEqual(float(oil["units_expired"]), 0.0)
        open_row = self.df[self.df["item_id"] == 221758].iloc[0]
        self.assertFalse(bool(open_row["done"]))


class TestRealSheetQuirks(unittest.TestCase):
    """Quirks of the first real export: item_id came out as #N/A, and 'as of' notes stayed on finished rows."""
    def test_na_item_ids_fall_back_to_the_item_name(self):
        import io
        csv = "\n".join([
            "date,time_posted,tz,Item,item_id,lane,list_price,quantity,units_sold,units_expired,Note 1",
            "2026-10-30,8:22 AM,EDT,Iceweb Spider Silk,#N/A,UNDER,0.99,45,45,,",
            "2026-10-30,3:26 PM,EDT,Eternal Crystal,172232,UNDER,2.11,306,306,,flip",
        ])
        df = read_log(io.StringIO(csv))
        self.assertEqual(sorted(df["item_id"]), [42253, 172232])
        bad = csv.replace("Iceweb Spider Silk,#N/A", "Mystery Herb,#N/A")
        with self.assertRaises(ValueError):
            read_log(io.StringIO(bad))

    def test_as_of_note_with_sold_plus_expired_equal_to_quantity_is_a_final_result(self):
        import io
        csv = "\n".join([
            "date,time_posted,tz,Item,item_id,lane,list_price,quantity,units_sold,units_expired,Note 1",
            "2026-10-30,11:11 AM,EDT,Swiftness Potion,2459,UNDER,5.44,30,7,23,7 as of 3 PM",
            "2026-10-30,11:11 AM,EDT,Swiftness Potion,2459,SAME,5.45,30,7,,7 as of 3 PM",
        ])
        df = read_log(io.StringIO(csv)).sort_values("price_copper")
        self.assertTrue(bool(df.iloc[0]["done"]))
        self.assertEqual(float(df.iloc[0]["units_expired"]), 23.0)
        self.assertTrue(bool(df.iloc[1]["in_progress"]))                       # 7 of 30 and no expired count: still running

    def test_a_row_with_a_wrong_time_is_called_out(self):
        import io
        csv = "\n".join([
            "date,time_posted,tz,Item,item_id,lane,list_price,quantity,units_sold,units_expired,Note 1",
            "2026-10-30,11:16 AM,EDT,Sorrowmoss,13466,UNDER,5.84,35,35,,",
            "2026-10-30,11:16 AM,EDT,Sorrowmoss,13466,SAME,5.85,35,35,,",
            "2026-10-30,11:16 AM,EDT,Sorrowmoss,13466,SAME,8.99,40,,40,",
            "2026-10-30,10:54 PM,EDT,Sorrowmoss,13466,UNDER,8.98,35,15,20,",
        ])
        df = read_log(io.StringIO(csv))
        self.assertEqual(len(df.attrs["warnings"]), 1)
        self.assertIn("13466", df.attrs["warnings"][0])
        ok = read_log(io.StringIO(csv.replace("11:16 AM,EDT,Sorrowmoss,13466,SAME,8.99", "10:54 PM,EDT,Sorrowmoss,13466,SAME,8.99")))
        self.assertEqual(ok.attrs["warnings"], [])


if __name__ == "__main__":
    unittest.main()
