"""Loader: reads your stored auction files (Hugging Face mirror, Backblaze B2, or a local folder) into Snapshots.

Design goals, in plain words
- SPARE THE BACKBLAZE CAP. Hugging Face holds a copy of every hourly file and has no daily download limit, so it is the
  default source. B2 is supported but every byte is counted against a budget (default 0.8 GB per UTC day, remembered
  between runs in the cache folder) and the loader stops BEFORE it would go over.
- NEVER DOWNLOAD TWICE. Every file that is read is saved (already trimmed) in a local cache folder.
- ASK FIRST. `plan_load` tells you how many files / bytes a request would touch before anything is downloaded.
- NEVER GUESS. Files that cannot be read raise an error, or are listed in `LoadResult.skipped` when
  `on_error="skip"`; they are never silently turned into empty data.

Storage layout this reads (written by fetch.py / compact.py):
    <source>/year=YYYY/month=MM/day=DD/hour_HH_MM.parquet   one snapshot per file (fetch.py v2 or v1)
    <source>/year=YYYY/month=MM/day=DD/daily.parquet        many snapshots merged by compact.py
Hourly files are preferred when both exist for a day (no duplication, row groups sorted by item_id).
Files from fetch v1 (no snapshot_ts / item_id columns) are upgraded on load; their snapshot time falls back to the
fetch time and the snapshot is marked meta["ts_source"] = "scan_timestamp".

Parquet reading needs `pyarrow`. It is imported only when a parquet file is actually read, so everything else
(and the tests) work without it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .schema import Snapshot, snapshots_from_blizzard_v2

KEY_RE = re.compile(r"^(?P<source>[^/]+)/year=(?P<y>\d{4})/month=(?P<m>\d{2})/day=(?P<d>\d{2})/(?P<file>hour_\d{2}_\d{2}|daily)\.parquet$")
NEEDED_COLUMNS = [
    "id", "item_id", "clean_item_id", "item", "variant_key", "quantity", "unit_price", "buyout", "bid",
    "time_left", "source", "snapshot_ts", "scan_timestamp",
]
MB = 1024 * 1024


class LoaderError(RuntimeError):
    pass


class BudgetExceeded(LoaderError):
    pass


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int


# ---------------------------------------------------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------------------------------------------------
class Store:
    """Something that can list files under a prefix and return a byte range of one file."""
    name = "store"

    def list(self, prefix: str) -> List[ObjectInfo]:
        raise NotImplementedError

    def read_range(self, key: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of the file. `end` may be past the end of the file."""
        raise NotImplementedError


class LocalStore(Store):
    """A folder laid out like the bucket. Used for tests and for working offline from files you already downloaded."""
    name = "local"

    def __init__(self, root: str):
        self.root = root

    def list(self, prefix: str) -> List[ObjectInfo]:
        out = []
        base = os.path.join(self.root, prefix)
        for dirpath, _, files in os.walk(self.root if not os.path.isdir(base) else base):
            for f in files:
                full = os.path.join(dirpath, f)
                key = os.path.relpath(full, self.root).replace(os.sep, "/")
                if key.startswith(prefix):
                    out.append(ObjectInfo(key, os.path.getsize(full)))
        return sorted(out, key=lambda o: o.key)

    def read_range(self, key: str, start: int, end: int) -> bytes:
        with open(os.path.join(self.root, key), "rb") as fh:
            fh.seek(start)
            return fh.read(max(0, end - start))


class HFStore(Store):
    """Hugging Face dataset repo, over plain HTTPS (no extra libraries). Needs HF_TOKEN if the repo is private."""
    name = "hf"

    def __init__(self, repo_id: str, token: Optional[str] = None, session=None):
        import requests
        self.repo_id, self.token = repo_id, token
        self.http = session or requests.Session()

    def _headers(self, extra=None):
        h = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        h.update(extra or {})
        return h

    def list(self, prefix: str) -> List[ObjectInfo]:
        # The tree endpoint lists one folder; walk the year/month/day structure under the source folder.
        out: List[ObjectInfo] = []
        url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{prefix.rstrip('/')}?recursive=true&expand=false"
        while url:
            r = self.http.get(url, headers=self._headers(), timeout=60)
            r.raise_for_status()
            for e in r.json():
                if e.get("type") == "file":
                    out.append(ObjectInfo(e["path"], int(e.get("size", 0))))
            nxt = r.links.get("next", {}).get("url")
            url = nxt
        return sorted(out, key=lambda o: o.key)

    def read_range(self, key: str, start: int, end: int) -> bytes:
        url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{key}"
        r = self.http.get(url, headers=self._headers({"Range": f"bytes={start}-{end - 1}"}), timeout=(10, 180))
        if r.status_code not in (200, 206):
            r.raise_for_status()
        if r.status_code == 200:  # server ignored the range and sent the whole file
            return r.content[start:end]
        return r.content


class B2Store(Store):
    """Backblaze B2 through its S3 API. Every request here counts against your daily download cap."""
    name = "b2"

    def __init__(self, bucket: str, key_id: str, app_key: str, endpoint: str = "https://s3.us-east-005.backblazeb2.com"):
        import boto3
        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key_id, aws_secret_access_key=app_key)

    def list(self, prefix: str) -> List[ObjectInfo]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                out.append(ObjectInfo(o["Key"], int(o["Size"])))
        return sorted(out, key=lambda o: o.key)

    def read_range(self, key: str, start: int, end: int) -> bytes:
        r = self.s3.get_object(Bucket=self.bucket, Key=key, Range=f"bytes={start}-{end - 1}")
        return r["Body"].read()


# ---------------------------------------------------------------------------------------------------------------------
# Byte budget (counts what is actually downloaded, remembered per UTC day)
# ---------------------------------------------------------------------------------------------------------------------
class ByteBudget:
    def __init__(self, store_name: str, max_bytes: Optional[int], state_dir: Optional[str] = None):
        self.store_name, self.max_bytes = store_name, max_bytes
        self.path = os.path.join(state_dir, "usage.json") if state_dir else None
        self.session_bytes = 0
        self.requests = 0

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _read(self) -> dict:
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path) as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def used_today(self) -> int:
        return int(self._read().get(self._today(), {}).get(self.store_name, 0))

    def charge(self, nbytes: int) -> None:
        """Call BEFORE downloading nbytes. Raises BudgetExceeded instead of going over."""
        if self.max_bytes is not None and self.used_today() + nbytes > self.max_bytes:
            raise BudgetExceeded(
                f"{self.store_name}: this download ({nbytes / MB:.1f} MB) would take today's usage "
                f"({self.used_today() / MB:.1f} MB) over the budget of {self.max_bytes / MB:.0f} MB. Stopping."
            )
        self.session_bytes += nbytes
        self.requests += 1
        if self.path:
            state = self._read()
            today = self._today()
            state = {today: state.get(today, {})}  # keep only today
            state[today][self.store_name] = int(state[today].get(self.store_name, 0)) + nbytes
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(state, fh)
            os.replace(tmp, self.path)


class RangeFile(io.RawIOBase):
    """A read-only, seekable file backed by range requests, with a block cache. This is what lets a parquet reader
    fetch just the footer and the columns it wants instead of the whole file. Larger blocks mean fewer requests
    (B2 also counts requests against a daily allowance)."""

    def __init__(self, store: Store, key: str, size: int, budget: Optional[ByteBudget] = None, block_size: int = 4 * MB):
        self.store, self.key, self.size, self.budget, self.block = store, key, size, budget, block_size
        self.pos = 0
        self.blocks: Dict[int, bytes] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        self.pos = max(0, self.pos)
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.blocks:
            start = i * self.block
            end = min(start + self.block, self.size)
            if self.budget:
                self.budget.charge(end - start)
            data = self.store.read_range(self.key, start, end)
            if len(data) != end - start:
                raise LoaderError(f"{self.key}: expected {end - start} bytes at {start}, got {len(data)}")
            self.blocks[i] = data
        return self.blocks[i]

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        chunks, p = [], self.pos
        while p < end:
            i = p // self.block
            blk = self._block(i)
            lo = p - i * self.block
            take = min(len(blk) - lo, end - p)
            chunks.append(blk[lo:lo + take])
            p += take
        self.pos = end
        return b"".join(chunks)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# ---------------------------------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class LoadPlan:
    source: str
    start: date
    end: date
    files: List[ObjectInfo]
    cached: List[str]
    days_missing: List[date]
    total_bytes: int
    bytes_to_download_upper_bound: int

    def describe(self) -> str:
        return (
            f"{self.source}: {self.start} to {self.end}: {len(self.files)} files, "
            f"{self.total_bytes / MB:.0f} MB in total, {len(self.cached)} already cached; "
            f"at most {self.bytes_to_download_upper_bound / MB:.0f} MB left to download"
            + (f"; NO FILES for {len(self.days_missing)} day(s): {', '.join(str(d) for d in self.days_missing)}" if self.days_missing else "")
        )


def _parse_key(key: str):
    m = KEY_RE.match(key)
    if not m:
        return None
    return m["source"], date(int(m["y"]), int(m["m"]), int(m["d"])), m["file"]


def _hours_present(files: List[ObjectInfo]) -> Set[int]:
    out = set()
    for o in files:
        m = re.search(r"hour_(\d{2})_\d{2}\.parquet$", o.key)
        if m:
            out.add(int(m.group(1)))
    return out


def day_inventory(objects: Iterable[ObjectInfo], source: str, start: date, end: date) -> Dict[date, dict]:
    """Per UTC day: which hours have an hourly file, and whether a daily file exists (used by `choose_files` and `lake.py files`)."""
    by_day: Dict[date, Dict[str, List[ObjectInfo]]] = {}
    for o in objects:
        p = _parse_key(o.key)
        if not p or p[0] != source or not (start <= p[1] <= end):
            continue
        by_day.setdefault(p[1], {}).setdefault("daily" if p[2] == "daily" else "hourly", []).append(o)
    return {d: dict(hourly=sorted(v.get("hourly", []), key=lambda o: o.key), daily=sorted(v.get("daily", []), key=lambda o: o.key),
                    hours=sorted(_hours_present(v.get("hourly", [])))) for d, v in by_day.items()}


def choose_files(objects: Iterable[ObjectInfo], source: str, start: date, end: date, min_hourly_hours: int = 22) -> List[ObjectInfo]:
    """Keeps files of `source` within [start, end] (inclusive UTC dates).

    Per day: the hourly files; PLUS the day's daily file when it exists and the hourly files cover fewer than
    `min_hourly_hours` of the 24 hours. (v0.4.1 ignored the daily file whenever ANY hourly file existed, so a day whose
    hourly files had been partly deleted lost the hours that only the daily file still held.) Snapshots stored in both
    are de-duplicated when loading.
    """
    chosen: List[ObjectInfo] = []
    inv = day_inventory(objects, source, start, end)
    for day in sorted(inv):
        d = inv[day]
        files = list(d["hourly"])
        if d["daily"] and len(d["hours"]) < min_hourly_hours:
            files += d["daily"]
        chosen.extend(sorted(files, key=lambda o: o.key))
    return chosen


def _items_sig(item_ids: Optional[Set[int]]) -> str:
    if item_ids is None:
        return "ALL"
    return "i" + hashlib.sha1(",".join(map(str, sorted(item_ids))).encode()).hexdigest()[:12]


class Cache:
    """Trimmed per-file frames saved on disk. Disposable: delete the folder any time to force a fresh download."""

    def __init__(self, root: Optional[str], store_name: str):
        self.root, self.store_name = root, store_name

    def _path(self, key: str, sig: str) -> str:
        return os.path.join(self.root, self.store_name, key.replace("/", "__") + f".{sig}.v2.pkl")

    def get(self, key: str, item_ids: Optional[Set[int]]):
        """Returns the cached frame or None. An ALL entry can serve any item request (the caller filters)."""
        if not self.root:
            return None
        for sig in ("ALL", _items_sig(item_ids)):
            p = self._path(key, sig)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        return pickle.load(fh)
                except Exception:
                    os.remove(p)  # unreadable cache entry: ignore it and refetch
        return None

    def has(self, key: str, item_ids: Optional[Set[int]]) -> bool:
        return bool(self.root) and any(os.path.exists(self._path(key, s)) for s in ("ALL", _items_sig(item_ids)))

    def put(self, key: str, item_ids: Optional[Set[int]], frame: pd.DataFrame) -> None:
        if not self.root:
            return
        p = self._path(key, _items_sig(item_ids))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(frame, fh)
        os.replace(tmp, p)


def plan_load(store: Store, source: str, start: date, end: date, item_ids: Optional[Iterable[int]] = None,
              cache_dir: Optional[str] = None) -> LoadPlan:
    ids = set(int(i) for i in item_ids) if item_ids is not None else None
    files = choose_files(store.list(f"{source}/"), source, start, end)
    cache = Cache(cache_dir, store.name)
    cached = [o.key for o in files if cache.has(o.key, ids)]
    days_have = {_parse_key(o.key)[1] for o in files}
    d, missing = start, []
    while d <= end:
        if d not in days_have:
            missing.append(d)
        d += timedelta(days=1)
    todo = sum(o.size for o in files if o.key not in set(cached))
    return LoadPlan(source, start, end, files, cached, missing, sum(o.size for o in files), todo)


# ---------------------------------------------------------------------------------------------------------------------
# Reading and normalising
# ---------------------------------------------------------------------------------------------------------------------
def read_parquet_file(fileobj, item_ids: Optional[Set[int]]) -> pd.DataFrame:
    """Reads only the needed columns, and skips row groups that cannot contain the wanted items."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(fileobj)
    names = pf.schema_arrow.names
    cols = [c for c in NEEDED_COLUMNS if c in names]
    id_col = next((c for c in ("item_id", "clean_item_id") if c in names), None)
    groups = list(range(pf.num_row_groups))
    if item_ids and id_col:  # row-group pruning from min/max statistics
        keep = []
        ci = names.index(id_col)
        lo_ids, hi_ids = min(item_ids), max(item_ids)
        for g in groups:
            st = pf.metadata.row_group(g).column(ci).statistics
            if st is None or not st.has_min_max or not (st.max < lo_ids or st.min > hi_ids):
                keep.append(g)
        groups = keep
    if not groups:
        return pd.DataFrame(columns=cols)
    return pf.read_row_groups(groups, columns=cols).to_pandas()


def normalize_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Upgrades any stored layout (fetch v1 hourly, v2 hourly, compacted daily) to what snapshots_from_blizzard_v2 needs."""
    out = df.copy()
    # A compacted day can mix layouts (older hours: clean_item_id / item struct; newer hours: item_id), so the item id is
    # taken from EVERY column that has one, row by row. v0.4.2 and earlier only looked at clean_item_id when there was
    # no item_id column at all, which silently dropped the older rows of a mixed day.
    ids = pd.Series(np.nan, index=out.index, dtype="float64")
    found = False
    for c in ("item_id", "clean_item_id"):
        if c in out.columns:
            ids = ids.fillna(pd.to_numeric(out[c], errors="coerce"))
            found = True
    if "item" in out.columns:
        ids = ids.fillna(pd.to_numeric(out["item"].map(lambda x: x.get("id") if isinstance(x, dict) else None), errors="coerce"))
        found = True
    if not found:
        raise LoaderError("file has no item id column (item_id / clean_item_id / item)")
    out["item_id"] = ids
    out = out[out["item_id"].notna()].copy()
    out["item_id"] = out["item_id"].astype("int64")
    # Older rows of a mixed day have EMPTY variant_key / source cells. Fill them: a later groupby on (source, time)
    # silently drops rows whose key is empty, which is how a whole stretch of a day can vanish.
    if "variant_key" not in out.columns:
        out["variant_key"] = out["item_id"].astype(str)
    else:
        out["variant_key"] = out["variant_key"].astype(object).where(out["variant_key"].notna(), out["item_id"].astype(str))
    if "source" not in out.columns:
        out["source"] = source
    else:
        out["source"] = out["source"].astype(object).where(out["source"].notna(), source)
    if "unit_price" not in out.columns and "buyout" not in out.columns:
        raise LoaderError("file has neither unit_price nor buyout")
    if "snapshot_ts" not in out.columns and "scan_timestamp" not in out.columns:
        raise LoaderError("file has neither snapshot_ts nor scan_timestamp")
    for c in ("snapshot_ts", "scan_timestamp"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    keep = [c for c in ("id", "item_id", "variant_key", "quantity", "unit_price", "buyout", "time_left", "source",
                        "snapshot_ts", "scan_timestamp") if c in out.columns]
    return out[keep].reset_index(drop=True)


@dataclass
class LoadResult:
    snapshots: List[Snapshot]
    plan: LoadPlan
    downloaded_bytes: int
    requests: int
    skipped: List[tuple] = field(default_factory=list)  # (key, reason)
    duplicates_dropped: int = 0
    incomplete_dropped: List[tuple] = field(default_factory=list)  # (timestamp, listings, typical listings)


def load_snapshots(
    store: Store, source: str, start: date, end: date, item_ids: Optional[Iterable[int]] = None,
    cache_dir: Optional[str] = None, max_bytes: Optional[int] = None, on_error: str = "raise",
    block_size: int = 4 * MB, reader: Optional[Callable] = None, progress: Optional[Callable[[str], None]] = None,
    min_listing_fraction: float = 0.5,
) -> LoadResult:
    """Loads snapshots for `source` between two UTC dates (inclusive).

    item_ids: keep only these items (much smaller; snapshots are then marked as covering only those items).
    max_bytes: stop before downloading more than this today (per store). Default None = unlimited (fine for
               Hugging Face); use ~0.8e9 for B2.
    reader: how to turn a file object into a DataFrame (default: parquet via pyarrow). Replaceable for tests.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    ids = set(int(i) for i in item_ids) if item_ids is not None else None
    reader = reader or read_parquet_file
    say = progress or (lambda m: None)
    plan = plan_load(store, source, start, end, ids, cache_dir)
    cache = Cache(cache_dir, store.name)
    budget = ByteBudget(store.name, max_bytes, os.path.join(cache_dir, "_state") if cache_dir else None)

    frames: List[pd.DataFrame] = []
    skipped: List[tuple] = []
    for n, obj in enumerate(plan.files, 1):
        try:
            hit = cache.get(obj.key, ids)
            if hit is not None:
                frame = hit
            else:
                say(f"[{n}/{len(plan.files)}] downloading {obj.key} ({obj.size / MB:.1f} MB)")
                raw = reader(RangeFile(store, obj.key, obj.size, budget, block_size), ids)
                frame = normalize_frame(raw, source)
                if len(frame) < len(raw):
                    say(f"  WARNING {obj.key}: {len(raw) - len(frame):,} of {len(raw):,} rows had no usable item id and were dropped")
                if ids is not None:
                    frame = frame[frame["item_id"].isin(ids)]
                cache.put(obj.key, ids, frame)
            if ids is not None:
                frame = frame[frame["item_id"].isin(ids)]
            frames.append(frame)
        except BudgetExceeded:
            raise  # never skip past a budget stop: the result would silently have holes
        except Exception as e:
            if on_error == "raise":
                raise LoaderError(f"{obj.key}: {e}") from e
            skipped.append((obj.key, str(e)))

    snaps: List[Snapshot] = []
    for frame in frames:
        if len(frame):
            snaps.extend(snapshots_from_blizzard_v2(frame))
    # the same Blizzard snapshot can be stored twice (an hourly file and a daily file), possibly with timestamps a few
    # minutes apart (fetch time vs publish time); keep one. Real snapshots are about an hour apart, so 10 minutes is safe.
    unique, dups, last = [], 0, {}
    for s in sorted(snaps, key=lambda s: (s.source, s.ts)):
        prev = last.get(s.source)
        if prev is not None and (s.ts - prev).total_seconds() < 600:
            dups += 1
            continue
        last[s.source] = s.ts
        unique.append(s)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    unique = [s for s in unique if lo <= s.ts < hi]
    # A fetch that captured only part of the book (the first real files hold 15-30k listings where a full book has
    # ~265k) would look like thousands of sales and new listings. Treat such a snapshot as MISSING instead.
    incomplete: List[tuple] = []
    if ids is None and len(unique) >= 5 and min_listing_fraction > 0:
        sizes = pd.Series([len(x.book) for x in unique])
        typical = float(sizes.median())
        keep = []
        for x, n in zip(unique, sizes):
            if n < min_listing_fraction * typical:
                incomplete.append((x.ts, int(n), int(typical)))
            else:
                keep.append(x)
        unique = keep
        for ts, n, t in incomplete:
            say(f"  WARNING snapshot {ts} has only {n:,} listings (typical {t:,}): treated as missing")
    if ids is not None:
        for s in unique:
            s.covered_items = frozenset(ids)
    return LoadResult(unique, plan, budget.session_bytes, budget.requests, skipped, dups, incomplete)


# ---------------------------------------------------------------------------------------------------------------------
# Is the data usable? (a quick health check before trusting any numbers)
# ---------------------------------------------------------------------------------------------------------------------
def coverage_report(snapshots: Sequence[Snapshot], expected_gap_minutes: float = 60.0) -> dict:
    """How continuous is the data? Big gaps matter because sales inside a gap cannot be told apart."""
    if len(snapshots) < 2:
        return dict(n_snapshots=len(snapshots), ok=False, note="need at least 2 snapshots")
    ts = pd.Series([s.ts for s in snapshots]).sort_values().reset_index(drop=True)
    gaps = ts.diff().dropna().dt.total_seconds() / 60.0
    fallback = sum(1 for s in snapshots if s.meta.get("ts_source") == "scan_timestamp")
    return dict(
        n_snapshots=len(snapshots), first=ts.iloc[0], last=ts.iloc[-1],
        median_gap_min=float(gaps.median()), max_gap_min=float(gaps.max()),
        gaps_over_2x_expected=int((gaps > 2 * expected_gap_minutes).sum()),
        gaps_over_4h=int((gaps > 240).sum()),
        snapshots_with_fetch_time_only=fallback,
        ok=bool(gaps.median() <= 1.5 * expected_gap_minutes),
    )
