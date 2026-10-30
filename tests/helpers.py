import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from engine.schema import Snapshot

T0 = pd.Timestamp("2026-10-29 12:00:00")


def mk(rows, ts=T0, source="commodities", **kw):
    """rows: (auction_id, item_id, quantity, unit_price, time_left)"""
    df = pd.DataFrame(rows, columns=["auction_id", "item_id", "quantity", "unit_price", "time_left"])
    df["variant_key"] = df["item_id"].astype(str)
    return Snapshot(source, ts, df, **kw)


def minutes(m):
    return T0 + pd.Timedelta(minutes=m)
