"""Ways to pack one house-day of WoW Forever hourly files WITHOUT losing anything, and the proof that each is lossless (used by tools/forever_pack_probe.py;
nothing here deletes or stores anything).

The raw data: one file per house per hour, one row per listing up at that hour, every column the fetcher wrote (auction id, item, quantity, price, time
left, and per-file values such as the snapshot time). A listing up for 24 hours is in about 24 files with the same values, so most rows repeat.

Three packed forms of one day (files k = 0..n-1 in time order):
  concat   every row of every file, in one table, with `_file` = k (the retail compaction: fixes the file count)
  sorted   the same rows sorted by auction id, then `_file`, so a listing's repeated rows sit together and compress well
  ledger   one row per RUN of a listing: an auction id with the same values in consecutive files k_first..k_last. A new run starts when any value
           changes (a partial sale lowers the quantity, time left drops a bucket) or the listing skips a file. Per-file values (columns with one value in
           each file, e.g. the snapshot time) are kept once per file in a separate small table.
Rebuilding file k from the ledger = every run with k_first <= k <= k_last, plus that file's per-file values. LOSSLESS means every rebuilt file equals the
original exactly: same rows, same values in every column (rows compared in auction id order; the order of rows inside a file carries no information).
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ID = "id"


def _canon(v):
    """A hashable, exact stand-in for a value (dicts / lists / arrays as sorted JSON), used to compare and group."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, sort_keys=True, default=str)
    if isinstance(v, float) and np.isnan(v):
        return "\x00nan"
    if v is None or v is pd.NA or v is pd.NaT:
        return "\x00none"
    return v


def _canon_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(_canon)
    return out


def file_constants(frames: Sequence[pd.DataFrame]) -> List[str]:
    """Columns that hold a single value inside every file (snapshot time, scan time, source, schema version...)."""
    cols = [c for c in frames[0].columns if c != ID] if frames else []
    keep = []
    for c in cols:
        ok = True
        for f in frames:
            if c not in f.columns or (len(f) and _canon_frame(f[[c]])[c].nunique(dropna=False) > 1):
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def to_concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    parts = [f.assign(_file=k) for k, f in enumerate(frames)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def to_sorted(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    c = to_concat(frames)
    return c.sort_values([ID, "_file"], kind="mergesort").reset_index(drop=True) if len(c) else c


def from_rows(df: pd.DataFrame, n: int, columns: Sequence[str]) -> List[pd.DataFrame]:
    """The files back from `concat` or `sorted`."""
    return [df[df["_file"] == k][list(columns)].reset_index(drop=True) for k in range(n)]


def to_ledger(frames: Sequence[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """(runs, files, constant columns). runs: one row per run with the listing's own columns, `_dup` (a listing id seen twice in one file: its order there),
    `k_first`, `k_last`. files: one row per file with the per-file columns and the file's row count."""
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), []
    const = file_constants(frames)
    # every column of every file, in order of first appearance: a regular-auction file has no 'bid' column in an hour without bids, and the column order can
    # differ between hours (v0.7.56). Each file's own column list is kept in meta, so a column a file did not have is not added back to it.
    all_cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    lcols = [c for c in all_cols if c not in const]                           # includes ID
    files = pd.DataFrame([{**{c: (f[c].iloc[0] if len(f) else None) for c in const}, "_rows": len(f)} for f in frames])
    long = to_concat([f[[c for c in lcols if c in f.columns]] for f in frames])
    long = long.reindex(columns=lcols + ["_file"])
    long["_dup"] = long.groupby(["_file", ID]).cumcount()
    canon = _canon_frame(long[[c for c in lcols if c != ID]])
    long["_sig"] = pd.util.hash_pandas_object(canon, index=False).to_numpy()
    long = long.sort_values([ID, "_dup", "_file"], kind="mergesort").reset_index(drop=True)
    same_key = (long[ID].to_numpy()[1:] == long[ID].to_numpy()[:-1]) & (long["_dup"].to_numpy()[1:] == long["_dup"].to_numpy()[:-1])
    next_file = long["_file"].to_numpy()[1:] == long["_file"].to_numpy()[:-1] + 1
    same_val = long["_sig"].to_numpy()[1:] == long["_sig"].to_numpy()[:-1]
    # (two different rows with the same hash would merge into one run; the round-trip check in check() would then fail, so it cannot pass silently)
    new_run = np.concatenate([[True], ~(same_key & next_file & same_val)]) if len(long) else np.array([], dtype=bool)
    run_id = np.cumsum(new_run) - 1
    long["_run"] = run_id
    first = long[new_run].copy()
    last_file = long.groupby("_run")["_file"].max()
    first["k_first"] = first["_file"]
    first["k_last"] = last_file.loc[first["_run"]].to_numpy()
    runs = first[lcols + ["_dup", "k_first", "k_last"]].reset_index(drop=True)
    return runs, files, const


def from_ledger(runs: pd.DataFrame, files: pd.DataFrame, const: Sequence[str], columns: Sequence[str], dtypes=None) -> List[pd.DataFrame]:
    """The files back from the ledger: file k = every run covering k, plus file k's per-file values, columns in the original order and, with `dtypes`
    (the original column types, as a stored packed file records them), the original types (an hour with no listings would otherwise turn a whole-number
    per-file column into decimals)."""
    if columns and isinstance(columns[0], (list, tuple)):     # one column list per file (v0.7.56)
        per_cols = [list(c) for c in columns]
    else:
        per_cols = [list(columns)] * len(files)
    if isinstance(dtypes, (list, tuple)):          # one set of types per file (v0.7.54: a column can be whole numbers in one hour and decimals in another)
        per_file = [dict(d) for d in dtypes]
    else:
        per_file = [dict(dtypes) if dtypes is not None else {}] * len(files)
    out = []
    for k in range(len(files)):
        dtypes = per_file[k]
        r = runs[(runs["k_first"] <= k) & (runs["k_last"] >= k)].drop(columns=["k_first", "k_last"])
        r = r.sort_values([ID, "_dup"], kind="mergesort").drop(columns=["_dup"]).copy()
        for c in const:
            v = files.at[k, c]
            r[c] = [v] * len(r)
        for c, t in dtypes.items():
            if c in r.columns and str(r[c].dtype) != str(t):
                try:
                    r[c] = r[c].astype(t)
                except (TypeError, ValueError):
                    pass
        out.append(r[per_cols[k]].reset_index(drop=True))
    return out


def same_file(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[bool, str]:
    """Exact equality of two versions of one file: the same columns, and the same rows with the same values (compared in auction id order)."""
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return False, f"{len(a)} rows vs {len(b)}"
    ca = _canon_frame(a).sort_values(list(a.columns), kind="mergesort").reset_index(drop=True)
    cb = _canon_frame(b).sort_values(list(b.columns), kind="mergesort").reset_index(drop=True)
    for c in a.columns:
        x, y = ca[c], cb[c]
        eq = (x == y) | (x.isna() & y.isna())
        if not bool(eq.all()):
            i = int(np.where(~eq.to_numpy())[0][0])
            return False, f"column {c}: {x.iloc[i]!r} vs {y.iloc[i]!r}"
        if str(a[c].dtype) != str(b[c].dtype) and not (a[c].dtype == object or b[c].dtype == object):
            return False, f"column {c}: type {a[c].dtype} vs {b[c].dtype}"
    return True, ""


def check(frames: Sequence[pd.DataFrame], rebuilt: Sequence[pd.DataFrame]) -> Tuple[bool, str]:
    if len(frames) != len(rebuilt):
        return False, f"{len(frames)} files vs {len(rebuilt)}"
    for k, (a, b) in enumerate(zip(frames, rebuilt)):
        ok, why = same_file(a, b)
        if not ok:
            return False, f"file {k}: {why}"
    return True, ""


# ------------------------------------------------------------------------------------------------ the stored packed file (one per source per UTC day)
PACK_VERSION = 3          # 2: column types kept per hourly file; 3: column list kept per hourly file (packs of versions 1 and 2 still unpack)
PACK_PREFIX = "packed"


def pack_key(source: str, day) -> str:
    """packed/<source>/year=YYYY/month=MM/day=DD/ledger.parquet: outside every raw source folder, so no loader takes it for raw or retail-compacted data."""
    d = pd.Timestamp(day)
    return f"{PACK_PREFIX}/{source}/year={d.year:04d}/month={d.month:02d}/day={d.day:02d}/ledger.parquet"


def pack(frames: Sequence[pd.DataFrame], keys: Sequence[str]) -> Tuple[pd.DataFrame, dict]:
    """(runs, meta) for one day: meta holds what rebuilding needs (per-file table, per-file columns, column order and types, the hourly file names)."""
    runs, files, const = to_ledger(frames)
    cols = [list(f.columns) for f in frames]                 # each hour's own columns, in its own order (v0.7.56)
    # each hour's own column types: pandas reads a whole-number column with a missing value (a bid-only auction has no buyout) as decimals, so the same column
    # can be int64 in one hour and float64 in the next (found on house 125, regular auctions, Nov 13: "column buyout: type int64 vs float64")
    meta = dict(version=PACK_VERSION, keys=list(keys), columns=cols, const=list(const),
                dtypes=[{c: str(t) for c, t in f.dtypes.items()} for f in frames], files=files)
    return runs, meta


def unpack(runs: pd.DataFrame, meta: dict) -> List[pd.DataFrame]:
    return from_ledger(runs, meta["files"], meta["const"], meta["columns"], meta["dtypes"])


def parquet_writer(runs: pd.DataFrame, meta: dict) -> bytes:
    """One parquet file (zstd): the runs as the table; the per-file table (itself parquet, base64) and the rest of meta in the file's key-value metadata."""
    import base64
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    fb = io.BytesIO()
    meta["files"].to_parquet(fb, index=False)
    m = {k: v for k, v in meta.items() if k != "files"}
    m["files_parquet_b64"] = base64.b64encode(fb.getvalue()).decode("ascii")
    table = pa.Table.from_pandas(runs, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"forever_pack": json.dumps(m).encode("utf-8")})
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def parquet_reader(data: bytes) -> Tuple[pd.DataFrame, dict]:
    import base64
    import io
    import pyarrow.parquet as pq
    table = pq.read_table(io.BytesIO(data))
    m = json.loads(table.schema.metadata[b"forever_pack"].decode("utf-8"))
    m["files"] = pd.read_parquet(io.BytesIO(base64.b64decode(m.pop("files_parquet_b64"))))
    return table.to_pandas(), m
