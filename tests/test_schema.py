import unittest

import numpy as np
import pandas as pd

from helpers import T0, mk
from engine.schema import SchemaError, Snapshot, TimeLeftBuckets, coerce_book, snapshots_from_blizzard_v2


class TestBuckets(unittest.TestCase):
    def test_expiry_feasible_fraction(self):
        b = TimeLeftBuckets()
        s = pd.Series(["SHORT", "MEDIUM", "LONG", "VERY_LONG"])
        f60 = b.expiry_feasible_fraction(s, 60)
        self.assertAlmostEqual(f60[0], 1.0)
        self.assertAlmostEqual(f60[1], (60 - 30) / 90)
        self.assertEqual(f60[2], 0.0)
        self.assertEqual(f60[3], 0.0)
        f1000 = b.expiry_feasible_fraction(s, 1000)
        self.assertEqual(f1000[2], 1.0)
        self.assertAlmostEqual(f1000[3], (1000 - 720) / 2160)

    def test_bucket_for_remaining_boundaries(self):
        b = TimeLeftBuckets()
        got = b.bucket_for_remaining(np.array([10, 30, 31, 120, 121, 720, 721, 2000]))
        self.assertEqual(list(got), ["SHORT", "SHORT", "MEDIUM", "MEDIUM", "LONG", "LONG", "VERY_LONG", "VERY_LONG"])


class TestSnapshot(unittest.TestCase):
    def test_valid(self):
        s = mk([(1, 5, 10, 100, "LONG")])
        self.assertEqual(s.n_auctions, 1)
        self.assertEqual(str(s.book["unit_price"].dtype), "int64")

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(SchemaError):
            mk([(1, 5, 10, 100, "LONG"), (1, 5, 10, 101, "LONG")])

    def test_rejects_bad_values(self):
        with self.assertRaises(SchemaError):
            mk([(1, 5, 0, 100, "LONG")])
        with self.assertRaises(SchemaError):
            mk([(1, 5, 10, 0, "LONG")])
        with self.assertRaises(SchemaError):
            mk([(1, 5, 10, 100, "FOREVER")])

    def test_rejects_missing_columns_and_tz(self):
        with self.assertRaises(SchemaError):
            coerce_book(pd.DataFrame({"auction_id": [1]}))
        with self.assertRaises(SchemaError):
            Snapshot("x", pd.Timestamp("2026-01-01", tz="UTC"), mk([(1, 5, 10, 100, "LONG")]).book)


class TestAdapter(unittest.TestCase):
    def _frame(self, with_snapshot_ts=True):
        rows = [
            # commodities
            dict(id=1, item_id=10, variant_key="10", quantity=100, unit_price=500, time_left="LONG", source="commodities"),
            dict(id=2, item_id=10, variant_key="10", quantity=50, unit_price=510, time_left="SHORT", source="commodities"),
            # realm auctions (total buyout, bid-only row has no buyout)
            dict(id=3, item_id=99, variant_key="99:abc", quantity=1, buyout=9000, time_left="LONG", source="realm_73"),
            dict(id=4, item_id=98, variant_key="98", quantity=1, buyout=np.nan, time_left="LONG", source="realm_73"),
        ]
        df = pd.DataFrame(rows)
        df["scan_timestamp"] = T0 + pd.Timedelta(minutes=7)
        if with_snapshot_ts:
            df["snapshot_ts"] = T0
        return df

    def test_splits_by_source_and_prices_realm_rows(self):
        snaps = snapshots_from_blizzard_v2(self._frame())
        by = {s.source: s for s in snaps}
        self.assertEqual(set(by), {"commodities", "realm_73"})
        self.assertEqual(by["commodities"].kind, "commodity")
        self.assertEqual(by["realm_73"].kind, "auction")
        self.assertEqual(by["commodities"].ts, T0)
        self.assertEqual(by["realm_73"].n_auctions, 1)  # bid-only row dropped
        self.assertEqual(by["realm_73"].meta["dropped_unpriced"], 1)
        self.assertEqual(int(by["realm_73"].book["unit_price"].iloc[0]), 9000)

    def test_falls_back_to_scan_timestamp(self):
        snaps = snapshots_from_blizzard_v2(self._frame(with_snapshot_ts=False))
        self.assertTrue(all(s.meta.get("ts_source") == "scan_timestamp" for s in snaps))
        self.assertTrue(all(s.ts == T0 + pd.Timedelta(minutes=7) for s in snaps))

    def test_rejects_non_v2_frame(self):
        with self.assertRaises(SchemaError):
            snapshots_from_blizzard_v2(pd.DataFrame({"id": [1]}))


if __name__ == "__main__":
    unittest.main()
