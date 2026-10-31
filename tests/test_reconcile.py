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


class TestQueueOrderTest(unittest.TestCase):
    def row(self, lane, price, sold=40, q=40):
        return pd.Series(dict(price_copper=price, quantity=float(q), posted_utc=pd.Timestamp("2026-10-31 01:05"), lane=lane,
                              units_sold=float(sold), units_expired=0.0, abnormal=False, done=True, item_id=ITEM))

    def books(self, clear_after_h):
        """old tier of 200 units at the floor 1.00; our UNDER 40 at 0.99 and SAME 40 at 1.00 join; everything of ours clears
        after `clear_after_h` hours (our units, not the old 200, disappear)."""
        out = [(pd.Timestamp("2026-10-31 00:33"), np.array([10000]), np.array([200]))]
        for k in range(1, 12):
            t = pd.Timestamp("2026-10-31 00:33") + pd.Timedelta(hours=k)
            mine = k < clear_after_h
            prices = np.array([9900, 10000]) if mine else np.array([10000])
            qty = np.array([40, 240]) if mine else np.array([200])
            out.append((t, prices, qty))
        return out

    def test_same_lane_clearing_in_two_hours_fits_lifo_not_fifo(self):
        vel = dict(low=40.0, est=40.0, high=40.0)
        t = trace_listing(self.row("SAME", 10000), self.books(3), partner_qty=40, vel=vel)
        self.assertEqual(t["status"], "ok")
        self.assertAlmostEqual(t["pred_lifo_h_est"], 2.0)       # (40 of mine ahead + 40 own) / 40 per hour
        self.assertAlmostEqual(t["pred_fifo_h_est"], 7.0)       # + the 200 old units at the same price
        self.assertLess(abs(np.log(t["pred_lifo_h_est"] / t["observed_hours_to_clear"])),
                        abs(np.log(t["pred_fifo_h_est"] / t["observed_hours_to_clear"])))

    def test_verdict_needs_five_comparisons(self):
        vel = dict(low=40.0, est=40.0, high=40.0)
        rows = [trace_listing(self.row("SAME", 10000), self.books(3), 40, vel) for _ in range(4)]
        s = summarize(pd.DataFrame(rows))
        self.assertTrue(any("need 5" in line for line in verdict_text(s)))
        rows.append(trace_listing(self.row("SAME", 10000), self.books(3), 40, vel))
        s5 = summarize(pd.DataFrame(rows))
        self.assertEqual(s5["order_test_n"], 5)
        self.assertEqual(s5["lifo_closer"], 5)
        self.assertTrue(any("NEWEST-FIRST" in line for line in verdict_text(s5)))

    def test_under_lane_does_not_vote_on_order(self):
        vel = dict(low=40.0, est=40.0, high=40.0)
        rows = [trace_listing(self.row("UNDER", 9900), self.books(3), 0, vel) for _ in range(6)]
        s = summarize(pd.DataFrame(rows))
        self.assertEqual(s["same_lane_for_order_test"], 0)


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


if __name__ == "__main__":
    unittest.main()
