"""Forever adapter: the stored Forever files -> Snapshots, one market per auction house.

Why a separate adapter and not just the retail loader
  Forever has no realms. Every auction HOUSE is its own market (health check, Nov 25: no two houses share an auction id, and the houses price the same item differently),
  so nothing here is ever pooled across houses, and everything this module returns says which house it belongs to.
  The retail loader and reader (engine/loader.py, engine/schema.py) are not changed. They are used as they are, through the hooks they already have, and three things are fixed around them:

  1 KIND FROM THE NAME. schema.snapshots_from_blizzard_v2 calls a source a commodity only when it is named exactly "commodities". Forever's commodity sources are named
    forever_us_r1_h<house>_commodities, so the reader calls all of them "auction". Here the kind is set from the name ending, after loading.
  2 DUPLICATE AUCTION IDS. One repeated auction id in a book of 131,365 rows (house 114, Nov 25) makes Snapshot validation reject the WHOLE snapshot. The reader handed to the loader
    drops repeated ids first (the first copy stays) and counts what it dropped, and how many of those ids had copies that DISAGREED (different price, quantity or time left).
  3 NEVER MIX MARKETS. After loading, every snapshot must say it belongs to the house and kind that was asked for; if not, the load stops instead of silently mixing markets.

The result is `HouseData` per house: the commodity snapshots (prices per unit) and the regular-auction snapshots (buyout // quantity), plus what was dropped and why. `hourly_for_house`
turns a house's commodity snapshots into the same hourly table the retail history uses (engine/daily.py hourly_long) with a `house` column in front, so every Forever table carries its market.

Nothing in this module reads or prints an account name: error text goes through `scrub`, which blanks the dataset's name.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import daily as D
from .classifier import classify_series
from .loader import LoadPlan, LoadResult, LoaderError, ObjectInfo, Store, load_snapshots, plan_load, read_parquet_file
from .schema import Snapshot

REGION = "us"
RULESET = 1                      # one ruleset is enough: houses return the same data through rulesets 1, 2, 3 and 15 (ruleset check)
PVP_HOUSES = (110, 113, 116)     # the PvP set (Alliance, Horde, Blackwater): where the planned character plays
SOURCES_KEY = "system/forever_sources.json"
SOURCE_RE = re.compile(r"^forever_(?P<region>[a-z]+)_r(?P<ruleset>\d+)_h(?P<house>\d+)(?P<commodities>_commodities)?$")
KINDS = ("commodity", "auction")
COMPARE_COLUMNS = ("item_id", "buyout", "bid", "unit_price", "quantity", "time_left")     # what two copies of the same auction id are compared on
GAP_MINUTES = 90.0


def hf_repo() -> str:
    return os.environ.get("FOREVER_HF_REPO", "bosco277/wow-forever-data-lake")


def scrub(text) -> str:
    """Error text for logs and reports: the dataset name (and so the account name in it) is blanked out. The whole name goes first, because replacing only the owner would leave the rest."""
    t = str(text)
    repo = hf_repo()
    for part in (repo, repo.split("/")[0]):
        if part:
            t = t.replace(part, "<forever dataset>")
    return t[:300]


# ---------------------------------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------------------------------
def parse_source(source) -> Optional[dict]:
    """{region, ruleset, house, commodity} for a Forever source name, or None for anything else (the retail source 'commodities' included)."""
    m = SOURCE_RE.match(str(source))
    if not m:
        return None
    return dict(region=m["region"], ruleset=int(m["ruleset"]), house=int(m["house"]), commodity=bool(m["commodities"]))


def source_name(house: int, commodity: bool = False, region: str = REGION, ruleset: int = RULESET) -> str:
    return f"forever_{region}_r{int(ruleset)}_h{int(house)}" + ("_commodities" if commodity else "")


def is_commodity_source(source) -> bool:
    return str(source).endswith("_commodities")


def parse_houses(text, available: Optional[Iterable[int]] = None) -> List[int]:
    """'110,113' -> [110, 113]; blank -> the PvP set; 'all' -> every house in `available`."""
    t = str(text or "").strip().lower()
    if not t:
        return list(PVP_HOUSES)
    if t == "all":
        if not available:
            raise ValueError("'all' needs the list of houses in the dataset, and none could be found")
        return sorted({int(h) for h in available})
    out = []
    for tok in re.split(r"[,\s;]+", t):
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"not an auction house number: {tok!r}")
        out.append(int(tok))
    if not out:
        return list(PVP_HOUSES)
    return sorted(set(out))


def houses_in_store(store: Store) -> List[int]:
    """The houses named in the sources file the fetcher keeps in the dataset ([] when the file cannot be found: wrong dataset name, or a token without access)."""
    entry = next((o for o in store.list("system/") if o.key == SOURCES_KEY), None)
    if entry is None or entry.size <= 0:
        return []
    try:
        data = json.loads(store.read_range(SOURCES_KEY, 0, entry.size).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return []
    found = {p["house"] for p in (parse_source(s) for s in data) if p}
    return sorted(found)


# ---------------------------------------------------------------------------------------------------------------------
# Duplicate auction ids
# ---------------------------------------------------------------------------------------------------------------------
def drop_duplicate_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """(frame without repeated auction ids, rows dropped, ids whose copies disagreed).

    An id counts as repeated only within one snapshot (same source and snapshot time). The FIRST copy stays. A pair of copies 'disagree' when they differ in item, price, quantity or time left,
    which would mean the choice of copy changes a number; identical copies (the harmless case) are dropped and counted but not reported as a disagreement."""
    if df is None or len(df) == 0 or "id" not in df.columns:
        return df, 0, 0
    key = pd.DataFrame({"id": df["id"].to_numpy()})
    key["src"] = df["source"].to_numpy() if "source" in df.columns else ""
    if "snapshot_ts" in df.columns and "scan_timestamp" in df.columns:
        key["t"] = df["snapshot_ts"].fillna(df["scan_timestamp"]).to_numpy()
    elif "snapshot_ts" in df.columns:
        key["t"] = df["snapshot_ts"].to_numpy()
    elif "scan_timestamp" in df.columns:
        key["t"] = df["scan_timestamp"].to_numpy()
    else:
        key["t"] = 0
    later = key.duplicated(keep="first").to_numpy()
    dropped = int(later.sum())
    if dropped == 0:
        return df, 0, 0
    involved = key.duplicated(keep=False).to_numpy()
    cmp_cols = [c for c in COMPARE_COLUMNS if c in df.columns]
    conflicting = 0
    if cmp_cols:
        w = key[involved].copy()
        for c in cmp_cols:
            w[c] = df[c].to_numpy()[involved]
        nun = w.groupby(["src", "t", "id"], dropna=False)[cmp_cols].nunique(dropna=False)
        conflicting = int((nun.max(axis=1) > 1).sum())
    return df[~later].reset_index(drop=True), dropped, conflicting


@dataclass
class DupStats:
    """What `make_reader` dropped. Files served from the loader's cache are not read again, so they are not counted (see SourceLoad.files_cached)."""
    duplicate_rows: int = 0
    conflicting_ids: int = 0
    files: List[Tuple[str, int, int]] = field(default_factory=list)      # (file, rows dropped, ids whose copies disagreed)

    def add(self, key: str, dropped: int, conflicting: int) -> None:
        self.duplicate_rows += dropped
        self.conflicting_ids += conflicting
        self.files.append((key, dropped, conflicting))


def make_reader(stats: DupStats, base: Optional[Callable] = None) -> Callable:
    """A `reader` for engine.loader.load_snapshots: reads a file like the retail reader does, then drops repeated auction ids (counted in `stats`)."""
    base = base or read_parquet_file

    def reader(fileobj, item_ids):
        raw = base(fileobj, item_ids)
        clean, dropped, conflicting = drop_duplicate_ids(raw)
        if dropped:
            stats.add(getattr(fileobj, "key", "?"), dropped, conflicting)
        return clean

    return reader


# ---------------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class SourceLoad:
    """One source (a house's commodities, or its regular auctions) over one window."""
    house: int
    commodity: bool
    source: str
    result: LoadResult
    dups: DupStats

    @property
    def kind(self) -> str:
        return "commodity" if self.commodity else "auction"

    @property
    def snapshots(self) -> List[Snapshot]:
        return self.result.snapshots

    @property
    def files(self) -> int:
        return len(self.result.plan.files)

    @property
    def files_cached(self) -> int:
        return len(self.result.plan.cached)

    @property
    def unpriced_rows(self) -> int:
        return int(sum(s.meta.get("dropped_unpriced", 0) for s in self.snapshots))


@dataclass
class HouseData:
    house: int
    commodity: Optional[SourceLoad] = None
    auction: Optional[SourceLoad] = None

    def loads(self) -> List[SourceLoad]:
        return [x for x in (self.commodity, self.auction) if x is not None]

    def snapshots(self, kind: str) -> List[Snapshot]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        sl = self.commodity if kind == "commodity" else self.auction
        return list(sl.snapshots) if sl is not None else []


def load_source(store: Store, house: int, commodity: bool, start: date, end: date, item_ids: Optional[Iterable[int]] = None, cache_dir: Optional[str] = None,
                on_error: str = "raise", progress: Optional[Callable[[str], None]] = None, base_reader: Optional[Callable] = None,
                min_listing_fraction: float = 0.5, region: str = REGION, ruleset: int = RULESET) -> SourceLoad:
    """Snapshots of ONE house's commodities or regular auctions between two UTC dates (inclusive), through the retail loader. The kind is set from the source name; every snapshot is tagged
    meta['house'] and meta['market'] ('h110'); repeated auction ids are dropped and counted."""
    source = source_name(house, commodity, region, ruleset)
    dups = DupStats()
    res = load_snapshots(store, source, start, end, item_ids=item_ids, cache_dir=cache_dir, on_error=on_error, reader=make_reader(dups, base_reader),
                         progress=progress, min_listing_fraction=min_listing_fraction)
    for s in res.snapshots:
        p = parse_source(s.source)
        if p is None or p["house"] != int(house) or p["commodity"] != bool(commodity):
            raise LoaderError(f"{source}: a snapshot in these files says it is {s.source!r}. Markets are never mixed, so the load stops here.")
        s.kind = "commodity" if p["commodity"] else "auction"
        s.meta["house"] = int(house)
        s.meta["market"] = f"h{int(house)}"
    return SourceLoad(int(house), bool(commodity), source, res, dups)


def load_house(store: Store, house: int, start: date, end: date, kinds: Sequence[str] = KINDS, **kw) -> HouseData:
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise ValueError(f"unknown kind {bad}; use {KINDS}")
    out = HouseData(int(house))
    if "commodity" in kinds:
        out.commodity = load_source(store, house, True, start, end, **kw)
    if "auction" in kinds:
        out.auction = load_source(store, house, False, start, end, **kw)
    return out


def load_houses(store: Store, houses: Iterable[int], start: date, end: date, kinds: Sequence[str] = KINDS, **kw) -> Dict[int, HouseData]:
    """{house: HouseData}. One house at a time, never pooled."""
    return {int(h): load_house(store, h, start, end, kinds, **kw) for h in sorted({int(x) for x in houses})}


def plan_houses(store: Store, houses: Iterable[int], start: date, end: date, kinds: Sequence[str] = KINDS, cache_dir: Optional[str] = None,
                region: str = REGION, ruleset: int = RULESET) -> List[Tuple[int, str, LoadPlan]]:
    """What a load would touch (files, bytes) before anything is downloaded: [(house, kind, plan)]."""
    out = []
    for h in sorted({int(x) for x in houses}):
        for kind in kinds:
            out.append((h, kind, plan_load(store, source_name(h, kind == "commodity", region, ruleset), start, end, None, cache_dir)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# A store that retries
# ---------------------------------------------------------------------------------------------------------------------
class RetryStore(Store):
    """Wraps a store: transient failures (connection trouble, HTTP 429, HTTP 5xx) are retried with a growing pause; a folder that does not exist (HTTP 404 on a listing) is an empty list.
    A wrong dataset name or a token without access also answers 404, so callers check that the dataset is readable first (`houses_in_store`)."""

    def __init__(self, inner: Store, attempts: int = 4, delay: float = 3.0, sleep: Callable[[float], None] = time.sleep):
        self.inner, self.attempts, self.delay, self.sleep = inner, attempts, delay, sleep
        self.name = inner.name

    @staticmethod
    def _status(e):
        return getattr(getattr(e, "response", None), "status_code", None)

    def _try(self, fn, on_404=None):
        for i in range(self.attempts):
            try:
                return fn()
            except Exception as e:
                code = self._status(e)
                if code == 404 and on_404 is not None:
                    return on_404
                transient = (code is None and isinstance(e, OSError)) or (code is not None and (code >= 500 or code == 429))
                if i == self.attempts - 1 or not transient:
                    raise
                self.sleep(self.delay * (2 ** i))

    def list(self, prefix: str) -> List[ObjectInfo]:
        return self._try(lambda: self.inner.list(prefix), on_404=[])

    def read_range(self, key: str, start: int, end: int) -> bytes:
        return self._try(lambda: self.inner.read_range(key, start, end))


# ---------------------------------------------------------------------------------------------------------------------
# House-keyed hourly table
# ---------------------------------------------------------------------------------------------------------------------
HOURLY_COLUMNS = ["house"] + D.HOURLY_COLUMNS


def _no_intervals() -> pd.DataFrame:
    return pd.DataFrame({"item_id": pd.Series([], dtype="int64"), "t1": pd.Series([], dtype="datetime64[ns]"), "sold_est": pd.Series([], dtype="float64"),
                         "flag_wipe": pd.Series([], dtype="bool"), "flag_long_gap": pd.Series([], dtype="bool")})


def hourly_for_house(snaps: Sequence[Snapshot], house: int, ids: Optional[Iterable[int]] = None, classify: bool = True) -> pd.DataFrame:
    """The permanent-history hourly table (same columns and meaning as engine/daily.py hourly_long, edges trimmed the same way) for ONE house's commodity snapshots, with `house` in front.

    ids: items to cover (default: every item that appears). classify=True runs the sales classifier so flag_wipe / flag_long_gap are real, exactly as the retail history does;
    classify=False skips it (much faster) and leaves those two flags False. Fewer than 3 commodity snapshots give an empty table."""
    snaps = sorted((s for s in snaps if s.kind == "commodity"), key=lambda s: s.ts)
    empty = pd.DataFrame(columns=HOURLY_COLUMNS)
    if len(snaps) < 3:
        return empty
    for s in snaps:
        p = s.meta.get("house")
        if p is not None and int(p) != int(house):
            raise ValueError(f"snapshot {s.source!r} belongs to house {p}, not {house}: markets are never mixed")
    if ids is None:
        ids = set()
        for s in snaps:
            ids.update(int(i) for i in s.book["item_id"].unique())
    ids = {int(i) for i in ids}
    if not ids:
        return empty
    intervals = classify_series(snaps) if classify else _no_intervals()
    h = D.build_hourly(snaps, intervals, ids)
    hl = D.trim_edges(D.hourly_long(h, snaps, ids), snaps[0].ts, snaps[-1].ts)
    if hl is None or hl.empty:
        return empty
    hl = hl.copy()
    hl.insert(0, "house", int(house))
    return hl[HOURLY_COLUMNS].reset_index(drop=True)


def hourly_by_house(markets: Dict[int, HouseData], ids: Optional[Iterable[int]] = None, classify: bool = True) -> pd.DataFrame:
    """One table for several houses: every row carries its house. Houses are computed one at a time (never pooled)."""
    parts = [hourly_for_house(m.snapshots("commodity"), h, ids, classify) for h, m in sorted(markets.items())]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def snapshot_gaps(snaps: Sequence[Snapshot], gap_minutes: float = GAP_MINUTES) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    """[(last snapshot before, first after, hours between)] for consecutive snapshots further apart than `gap_minutes`."""
    t = sorted(s.ts for s in snaps)
    return [(a, b, (b - a).total_seconds() / 3600.0) for a, b in zip(t[:-1], t[1:]) if (b - a).total_seconds() / 60.0 > gap_minutes]
