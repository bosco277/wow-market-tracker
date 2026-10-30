"""Which listing at the SAME price is bought first: the oldest (FIFO) or the newest (LIFO)?

The test uses listings that were PARTLY bought between two snapshots (quantity went down, still listed). A partial
fill is a certain sale, and it marks the exact point where buying stopped inside that price tier:
- LIFO (newest first): every NEWER listing at that price is gone, every OLDER one is untouched.
- FIFO (oldest first): every OLDER listing at that price is gone, every NEWER one is untouched.
- Neither pattern: order inside a tier is mixed (or cancels / expiries are muddying it).
"Newer" means a higher auction id; if ids were NOT handed out in time order the result would simply look mixed.

Listings that could have expired in the gap (SHORT / MEDIUM time left) are ignored, because expiry takes the OLDEST
listings and would fake a FIFO pattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .schema import Snapshot

CANNOT_EXPIRE_SOON = {"LONG", "VERY_LONG"}


@dataclass
class OrderEvidence:
    tiers_examined: int
    lifo_consistent: int
    fifo_consistent: int
    mixed: int

    @property
    def verdict(self) -> str:
        n = self.lifo_consistent + self.fifo_consistent + self.mixed
        if n < 30:
            return f"not enough evidence yet ({n} usable cases; want at least 30)"
        lifo, fifo = self.lifo_consistent / n, self.fifo_consistent / n
        if lifo >= 0.7 and fifo <= 0.1:
            return "LIFO: newest listing at a price is bought first"
        if fifo >= 0.7 and lifo <= 0.1:
            return "FIFO: oldest listing at a price is bought first"
        return "MIXED / unclear"

    def describe(self) -> str:
        return (f"{self.tiers_examined} price tiers had a partly-bought listing and other listings to compare; "
                f"LIFO-pattern {self.lifo_consistent}, FIFO-pattern {self.fifo_consistent}, mixed {self.mixed}. "
                f"Verdict: {self.verdict}")


def tier_order_evidence(snapshots: Sequence[Snapshot], max_gap_minutes: float = 75.0) -> OrderEvidence:
    snaps = sorted(snapshots, key=lambda s: s.ts)
    lifo = fifo = mixed = examined = 0
    for a, b in zip(snaps[:-1], snaps[1:]):
        if (b.ts - a.ts).total_seconds() / 60.0 > max_gap_minutes or a.kind != "commodity":
            continue
        prev = a.book[["auction_id", "item_id", "unit_price", "quantity", "time_left"]]
        cur = b.book.set_index("auction_id")["quantity"]
        p = prev.assign(curr_qty=prev["auction_id"].map(cur))
        partial = p[(p["curr_qty"].notna()) & (p["curr_qty"] < p["quantity"])]
        if partial.empty:
            continue
        tier_sizes = p.groupby(["item_id", "unit_price"]).size()
        partial_per_tier = partial.groupby(["item_id", "unit_price"]).size()
        for (item, price), n_part in partial_per_tier.items():
            if n_part != 1 or tier_sizes.loc[(item, price)] < 2:
                continue  # need exactly one partial fill and at least one other listing to compare
            tier = p[(p["item_id"] == item) & (p["unit_price"] == price)]
            part_id = int(partial[(partial["item_id"] == item) & (partial["unit_price"] == price)]["auction_id"].iloc[0])
            peers = tier[(tier["auction_id"] != part_id) & tier["time_left"].isin(CANNOT_EXPIRE_SOON)]
            if peers.empty:
                continue
            newer, older = peers[peers["auction_id"] > part_id], peers[peers["auction_id"] < part_id]
            newer_gone, older_gone = newer["curr_qty"].isna().all(), older["curr_qty"].isna().all()
            newer_kept, older_kept = newer["curr_qty"].notna().all(), older["curr_qty"].notna().all()
            examined += 1
            is_lifo = newer_gone and older_kept
            is_fifo = older_gone and newer_kept
            if is_lifo and is_fifo:  # no real contrast (e.g. only one side exists and it behaves both ways)
                examined -= 1
                continue
            if is_lifo:
                lifo += 1
            elif is_fifo:
                fifo += 1
            else:
                mixed += 1
    return OrderEvidence(examined, lifo, fifo, mixed)
