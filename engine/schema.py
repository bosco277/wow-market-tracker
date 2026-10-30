"""Canonical market-snapshot contract.

Everything downstream (classifier, metrics, backtest, alerts) consumes `Snapshot` objects and nothing else.
Data sources (Blizzard retail API today, whatever Forever offers at launch, an addon scan file later) are
adapters that produce Snapshots. Swapping a source never touches the analytics.

Conventions
- Prices are integer COPPER per unit (1 gold = 10,000 copper).
- Times are naive UTC `pd.Timestamp`. `Snapshot.ts` is the authoritative market time (Blizzard's Last-Modified),
  NOT the time we fetched it.
- A snapshot is an ask-side book: one row per live auction (commodity auctions carry a quantity).
- `covered_items` says which items the snapshot is authoritative for. None = the whole market. Partial scans
  (addon scans, truncated fetches) must set it, because an auction missing from a partial scan is NOT evidence
  that it disappeared from the market.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

COPPER_PER_GOLD = 10_000

TIME_LEFT_VALUES: Tuple[str, ...] = ("SHORT", "MEDIUM", "LONG", "VERY_LONG")

# column -> numpy dtype kind we coerce to
CANONICAL_COLUMNS: Dict[str, str] = {
    "auction_id": "int64",
    "item_id": "int64",
    "variant_key": "object",
    "quantity": "int64",
    "unit_price": "int64",
    "time_left": "object",
}


class SchemaError(ValueError):
    """Raised when a frame cannot be turned into a valid canonical snapshot."""


@dataclass(frozen=True)
class TimeLeftBuckets:
    """Remaining-lifetime range (minutes) each Blizzard `time_left` bucket stands for.

    DEFAULTS ARE ASSUMPTIONS taken from Blizzard's documented bucket meanings (SHORT < 30 min, MEDIUM 30 min-2 h,
    LONG 2-12 h, VERY_LONG 12-48 h). Re-verify against the current API docs; the classifier's expiry logic is only
    as good as these numbers, which is why they are a parameter and not a constant.
    """

    bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "SHORT": (0.0, 30.0),
            "MEDIUM": (30.0, 120.0),
            "LONG": (120.0, 720.0),
            "VERY_LONG": (720.0, 2880.0),
        }
    )

    def bucket_for_remaining(self, remaining_minutes: np.ndarray) -> np.ndarray:
        """Maps true remaining minutes to a bucket label (used by the simulator)."""
        r = np.asarray(remaining_minutes, dtype=float)
        out = np.full(r.shape, "VERY_LONG", dtype=object)
        for name in ("LONG", "MEDIUM", "SHORT"):  # narrow to wide so the smallest matching bucket wins
            lo, hi = self.bounds[name]
            out = np.where(r <= hi, name, out)
        return out

    def expiry_feasible_fraction(self, buckets: pd.Series, gap_minutes: float) -> np.ndarray:
        """P(listing's expiry falls inside the next `gap_minutes`), assuming remaining time is uniform in its bucket.

        0.0  -> expiry is impossible within the gap (a disappearance cannot be an expiry)
        1.0  -> every listing in that bucket must be gone by the end of the gap (a disappearance says nothing)
        """
        lo = buckets.map({k: v[0] for k, v in self.bounds.items()}).astype(float).to_numpy()
        hi = buckets.map({k: v[1] for k, v in self.bounds.items()}).astype(float).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (gap_minutes - lo) / (hi - lo)
        frac = np.where(np.isnan(frac), 1.0, frac)  # unknown bucket -> assume expiry possible (conservative)
        return np.clip(frac, 0.0, 1.0)


@dataclass
class Snapshot:
    source: str
    ts: pd.Timestamp
    book: pd.DataFrame
    kind: str = "commodity"  # "commodity" (fungible, per-unit price) | "auction" (per-listing items)
    covered_items: Optional[FrozenSet[int]] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ts = pd.Timestamp(self.ts)
        if self.ts.tzinfo is not None:
            raise SchemaError("Snapshot.ts must be naive UTC (got a tz-aware timestamp)")
        if self.kind not in ("commodity", "auction"):
            raise SchemaError(f"unknown snapshot kind {self.kind!r}")
        self.book = coerce_book(self.book)
        if self.covered_items is not None:
            self.covered_items = frozenset(int(i) for i in self.covered_items)

    @property
    def n_auctions(self) -> int:
        return len(self.book)

    def item_book(self, item_id: int) -> pd.DataFrame:
        return self.book[self.book["item_id"] == item_id]


def coerce_book(book: pd.DataFrame) -> pd.DataFrame:
    """Validates and normalises a book frame to the canonical columns/dtypes. Returns a new frame."""
    missing = [c for c in CANONICAL_COLUMNS if c not in book.columns]
    if missing:
        raise SchemaError(f"book is missing columns: {missing}")
    out = book[list(CANONICAL_COLUMNS)].copy()
    problems: List[str] = []
    for col, dtype in CANONICAL_COLUMNS.items():
        if dtype == "int64":
            if out[col].isna().any():
                problems.append(f"{col} contains nulls")
                continue
            out[col] = out[col].astype("int64")
    if problems:
        raise SchemaError("; ".join(problems))
    if out["auction_id"].duplicated().any():
        raise SchemaError("duplicate auction_id within one snapshot")
    if (out["quantity"] <= 0).any():
        raise SchemaError("quantity must be positive")
    if (out["unit_price"] <= 0).any():
        raise SchemaError("unit_price must be positive")
    bad = set(out["time_left"].unique()) - set(TIME_LEFT_VALUES)
    if bad:
        raise SchemaError(f"unknown time_left values: {sorted(map(str, bad))}")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# Adapter: fetch.py schema v2 (Blizzard) -> Snapshots
# --------------------------------------------------------------------------------------------------------------------
def snapshots_from_blizzard_v2(df: pd.DataFrame) -> List[Snapshot]:
    """Turns rows written by fetch.py (schema_version 2) into Snapshots, one per (source, snapshot_ts).

    Commodity rows already carry `unit_price`. Realm auction rows carry a total `buyout`, so unit_price is
    buyout // quantity; bid-only auctions (no buyout) cannot be priced and are dropped (counted in meta).
    Rows without `snapshot_ts` fall back to `scan_timestamp` (fetch time) and the snapshot is marked
    meta['ts_source'] = 'scan_timestamp' so downstream code can distrust the gap.
    """
    needed = {"id", "item_id", "variant_key", "quantity", "time_left", "source"}
    missing = needed - set(df.columns)
    if missing:
        raise SchemaError(f"frame is not fetch schema v2, missing {sorted(missing)}")

    work = df.copy()
    if "unit_price" not in work.columns:
        work["unit_price"] = np.nan
    if "buyout" in work.columns:
        per_unit = work["buyout"] / work["quantity"]
        work["unit_price"] = work["unit_price"].fillna(per_unit)

    ts_col = "snapshot_ts" if "snapshot_ts" in work.columns else "scan_timestamp"
    if "snapshot_ts" in work.columns and "scan_timestamp" in work.columns:
        work["_ts"] = work["snapshot_ts"].fillna(work["scan_timestamp"])
        ts_fallback = work["snapshot_ts"].isna()
    else:
        work["_ts"] = work[ts_col]
        ts_fallback = pd.Series(ts_col != "snapshot_ts", index=work.index)

    out: List[Snapshot] = []
    for (source, ts), grp in work.groupby(["source", "_ts"], sort=True):
        priced = grp[grp["unit_price"].notna() & (grp["unit_price"] > 0)]
        dropped = len(grp) - len(priced)
        book = pd.DataFrame(
            {
                "auction_id": priced["id"].astype("int64"),
                "item_id": priced["item_id"].astype("int64"),
                "variant_key": priced["variant_key"].astype(object),
                "quantity": priced["quantity"].astype("int64"),
                "unit_price": priced["unit_price"].round().astype("int64"),
                "time_left": priced["time_left"].astype(object),
            }
        )
        meta = {"dropped_unpriced": int(dropped)}
        if bool(ts_fallback.loc[grp.index].any()):
            meta["ts_source"] = "scan_timestamp"
        out.append(
            Snapshot(
                source=str(source),
                ts=pd.Timestamp(ts),
                book=book,
                kind="commodity" if str(source) == "commodities" else "auction",
                meta=meta,
            )
        )
    return out
