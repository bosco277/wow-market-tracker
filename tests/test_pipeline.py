"""End to end: simulator -> frame shaped exactly like fetch.py schema v2 -> adapter -> classifier -> rates."""
import unittest

import pandas as pd

from helpers import T0  # noqa: F401  (sets sys.path)
from engine import aggregate_rates, classify_series, snapshots_from_blizzard_v2
from engine.simulator import MarketSimulator, default_profiles


def as_fetch_v2_frame(snapshots):
    """Rebuilds what fetch.py writes for commodities: id, item(dict), quantity, unit_price, time_left, + v2 columns."""
    parts = []
    for s in snapshots:
        b = s.book.rename(columns={"auction_id": "id"}).copy()
        b["item"] = [{"id": int(i)} for i in b["item_id"]]
        b["source"] = "commodities"
        b["snapshot_ts"] = s.ts
        b["scan_timestamp"] = s.ts + pd.Timedelta(minutes=6)  # fetched a few minutes after Blizzard published it
        b["schema_version"] = 2
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


class TestPipeline(unittest.TestCase):
    def test_v2_frame_round_trips_through_the_adapter(self):
        res = MarketSimulator(default_profiles(), seed=7).run(12)
        frame = as_fetch_v2_frame(res.snapshots)
        back = snapshots_from_blizzard_v2(frame)
        self.assertEqual(len(back), len(res.snapshots))
        for original, restored in zip(res.snapshots, back):
            self.assertEqual(original.ts, restored.ts)
            self.assertEqual(restored.kind, "commodity")
            # string vs object dtype for variant_key is a pandas storage detail, not a data difference
            a = original.book.sort_values("auction_id").reset_index(drop=True)
            b = restored.book.sort_values("auction_id").reset_index(drop=True)
            for col in ("variant_key", "time_left"):
                a[col] = a[col].astype(object)
                b[col] = b[col].astype(object)
            pd.testing.assert_frame_equal(a, b)

    def test_classification_is_identical_before_and_after_the_round_trip(self):
        res = MarketSimulator(default_profiles(), seed=7).run(12)
        direct = classify_series(res.snapshots)
        via_frame = classify_series(snapshots_from_blizzard_v2(as_fetch_v2_frame(res.snapshots)))
        cols = ["item_id", "t0", "t1", "sold_cert", "sold_low", "sold_est", "sold_high"]
        pd.testing.assert_frame_equal(direct[cols].reset_index(drop=True), via_frame[cols].reset_index(drop=True))

    def test_rates_come_out_of_the_whole_chain(self):
        res = MarketSimulator(default_profiles(), seed=7).run(30)
        snaps = snapshots_from_blizzard_v2(as_fetch_v2_frame(res.snapshots))
        intervals = classify_series(snaps)
        rates = aggregate_rates(intervals, snaps[-1].ts, window_hours=24)
        self.assertEqual(set(rates["item_id"]), {1001, 1002, 1003, 1004})
        self.assertTrue((rates["daily_low"] <= rates["daily_est"] + 1e-9).all())
        self.assertTrue((rates["daily_est"] <= rates["daily_high"] + 1e-9).all())
        self.assertTrue((rates["coverage"] > 0.9).all())


if __name__ == "__main__":
    unittest.main()
