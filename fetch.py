"""Hourly auction extractor (schema v2).

What changed versus v1
- Every row now carries `snapshot_ts` (Blizzard's Last-Modified, naive UTC) so consecutive snapshots can be
  diffed on the real time gap. `scan_timestamp` is unchanged (fetch time) so compact.py keeps working.
- Flat helper columns `source`, `item_id`, `variant_key`, `schema_version`. The nested `item` struct is kept
  untouched (the lossless record); `variant_key` is only a convenience and can always be recomputed from it.
- Per-source isolation: one failing endpoint no longer aborts the others. Timeouts and retries everywhere.
- ONE Hugging Face commit per run (data files + markers together) instead of one commit per file.
- The "last update" marker only advances when the snapshot is durably stored on at least one sink.
  If a snapshot reaches only one sink, it is queued in a pending manifest and copied to the other sink at the
  start of a later run (reconcile). If it reaches neither, the marker does NOT advance and the run fails.
- Optional Discord alert on failure (set DISCORD_WEBHOOK_URL; unset = no-op).
"""
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import boto3
import botocore.exceptions
import pandas as pd
import requests
from huggingface_hub import CommitOperationAdd, HfApi

# --- CONFIGURATION ---------------------------------------------------------------------------------
REALM_IDS = [73, 96, 118]
SCHEMA_VERSION = 2

BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

B2_BUCKET_NAME = "wow-market-data-lake"
HF_REPO_ID = "bosco277/wow-market-data-lake"

PENDING_HF = "system/pending_hf.json"  # stored on B2: snapshot keys B2 has but Hugging Face lacks
PENDING_B2 = "system/pending_b2.json"  # stored on HF: snapshot keys HF has but B2 lacks
MAX_RECONCILE_PER_RUN = 12

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.us-east-005.backblazeb2.com",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY,
)
hf_api = HfApi(token=HF_TOKEN)
HTTP = requests.Session()


def log(msg):
    print(msg, flush=True)


def with_retries(fn, what, attempts=3, base_delay=2.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            delay = base_delay * (2 ** i)
            log(f"{what} failed ({e}); retrying in {delay:.0f}s")
            time.sleep(delay)


def notify(message):
    """Posts to Discord if a webhook is configured. Never raises."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        HTTP.post(DISCORD_WEBHOOK_URL, json={"content": message[:1900]}, timeout=10)
    except Exception as e:
        log(f"Discord notification failed: {e}")


# --- Storage helpers (B2 = primary, Hugging Face = mirror) -----------------------------------------
def b2_get(key):
    """Returns the object's bytes, or None if it does not exist. Other errors propagate."""
    try:
        return s3.get_object(Bucket=B2_BUCKET_NAME, Key=key)["Body"].read()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def b2_put(key, data):
    with_retries(lambda: s3.upload_fileobj(io.BytesIO(data), B2_BUCKET_NAME, key), f"B2 upload {key}")


def hf_get(key):
    """Returns the file's bytes from the HF dataset repo, or None if it does not exist."""
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{key}"
    r = HTTP.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def hf_commit(files, message):
    """Writes several files to the HF dataset repo in a single commit."""
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()]
    with_retries(
        lambda: hf_api.create_commit(repo_id=HF_REPO_ID, repo_type="dataset", operations=ops, commit_message=message),
        "HF commit",
    )


# --- Markers (Blizzard Last-Modified per source) ---------------------------------------------------
def marker_key(source):
    return f"system/last_update_{source}.txt"


def get_last_timestamp(source):
    """Last Last-Modified value we stored for this source. B2 first, HF as fallback if B2 errors."""
    key = marker_key(source)
    try:
        data = b2_get(key)
        return data.decode("utf-8").strip() if data else None
    except Exception as e:
        log(f"B2 marker read error for {source} ({e}); falling back to Hugging Face...")
    try:
        data = hf_get(key)
        return data.decode("utf-8").strip() if data else None
    except Exception as e:
        log(f"HF marker read failed for {source}: {e}")
        return None


# --- Pending manifests / reconcile -----------------------------------------------------------------
def read_manifest(getter, manifest_key):
    data = getter(manifest_key)
    return json.loads(data.decode("utf-8")) if data else []


def add_pending(manifest_key, keys, getter, putter):
    try:
        existing = read_manifest(getter, manifest_key)
        merged = existing + [k for k in keys if k not in existing]
        putter(manifest_key, json.dumps(merged).encode("utf-8"))
    except Exception as e:
        log(f"WARNING: could not record pending copies in {manifest_key}: {e}")


def reconcile():
    """Copies snapshots that previously reached only one sink to the other sink (bounded per run)."""
    # B2 -> HF
    try:
        pending = read_manifest(b2_get, PENDING_HF)
        if pending:
            batch = pending[:MAX_RECONCILE_PER_RUN]
            files = {}
            for key in batch:
                data = b2_get(key)
                if data is None:
                    log(f"Reconcile: {key} no longer exists on B2; dropping it from the queue.")
                else:
                    files[key] = data
            if files:
                hf_commit(files, f"Reconcile {len(files)} snapshot(s) from B2")
            b2_put(PENDING_HF, json.dumps(pending[len(batch):]).encode("utf-8"))
            log(f"Reconciled {len(files)} snapshot(s) B2 -> HF.")
    except Exception as e:
        log(f"Reconcile B2 -> HF skipped: {e}")

    # HF -> B2
    try:
        pending = read_manifest(hf_get, PENDING_B2)
        if pending:
            batch = pending[:MAX_RECONCILE_PER_RUN]
            copied = 0
            for key in batch:
                data = hf_get(key)
                if data is None:
                    log(f"Reconcile: {key} no longer exists on HF; dropping it from the queue.")
                    continue
                b2_put(key, data)
                copied += 1
            hf_commit(
                {PENDING_B2: json.dumps(pending[len(batch):]).encode("utf-8")},
                "Update pending B2 reconcile queue",
            )
            log(f"Reconciled {copied} snapshot(s) HF -> B2.")
    except Exception as e:
        log(f"Reconcile HF -> B2 skipped: {e}")


# --- Blizzard API ----------------------------------------------------------------------------------
def get_blizzard_token():
    def call():
        r = HTTP.post(
            "https://oauth.battle.net/token",
            auth=(BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["access_token"]

    return with_retries(call, "Blizzard token request")


def fetch_blizzard_data(url, token, last_modified):
    """Returns (auctions, last_modified_header), or (None, None) if Blizzard has nothing newer (304)."""
    params = {"namespace": "dynamic-us", "locale": "en_US"}
    headers = {"Authorization": f"Bearer {token}"}
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    last_problem = None
    for attempt in range(3):
        try:
            r = HTTP.get(url, params=params, headers=headers, timeout=(10, 180))
        except requests.RequestException as e:
            last_problem = e
        else:
            if r.status_code == 304:
                return None, None
            if r.status_code in (429, 500, 502, 503, 504):
                last_problem = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()  # other 4xx are not retried
                return r.json().get("auctions", []), r.headers.get("Last-Modified")
        if attempt < 2:
            time.sleep(2 * (2 ** attempt))
    raise RuntimeError(f"Blizzard request failed after 3 attempts: {last_problem}")


# --- Transformation --------------------------------------------------------------------------------
def variant_key(item):
    """Stable identity for an item variant. Plain items (commodities) -> '123'. Gear/pets -> '123:<hash>'.

    Built from everything in the item struct except the id (bonus_lists sorted, modifiers sorted). `context`
    is included, so identical gear from different sources can get different keys; coarsen it downstream if
    needed, because the full struct is preserved in the `item` column.
    """
    if not isinstance(item, dict):
        return None
    item_id = item.get("id")
    base = {k: v for k, v in item.items() if k != "id"}
    if isinstance(base.get("bonus_lists"), list):
        base["bonus_lists"] = sorted(base["bonus_lists"])
    if isinstance(base.get("modifiers"), list):
        base["modifiers"] = sorted(
            base["modifiers"],
            key=lambda m: (m.get("type", 0), m.get("value", 0)) if isinstance(m, dict) else (0, 0),
        )
    if not base:
        return str(item_id)
    digest = hashlib.sha1(json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]
    return f"{item_id}:{digest}"


def parse_last_modified(header):
    if not header:
        return None
    return parsedate_to_datetime(header).astimezone(timezone.utc).replace(tzinfo=None)


def build_frame(auctions, source, snapshot_dt, fetched_at):
    df = pd.DataFrame(auctions)
    if df.empty or "item" not in df.columns:
        raise ValueError("empty or malformed auctions payload")
    df["item_id"] = df["item"].map(lambda x: x.get("id") if isinstance(x, dict) else None).astype("Int64")
    df["variant_key"] = df["item"].map(variant_key)
    df["source"] = source
    df["snapshot_ts"] = pd.Timestamp(snapshot_dt) if snapshot_dt is not None else pd.NaT  # Blizzard Last-Modified, naive UTC
    df["scan_timestamp"] = pd.Timestamp(fetched_at)  # fetch time, naive UTC (unchanged from v1)
    df["schema_version"] = SCHEMA_VERSION
    # Sorting by item_id makes the files compress better and lets DuckDB skip row groups when filtering by item.
    return df.sort_values("item_id", kind="stable").reset_index(drop=True)


# --- One cycle -------------------------------------------------------------------------------------
@dataclass
class Snapshot:
    source: str
    key: str
    data: bytes
    marker: str  # Last-Modified header value, or None
    b2_ok: bool


def collect_snapshot(source, url, token):
    """Fetches one source and writes it to B2. Returns None if there is nothing new."""
    auctions, header = fetch_blizzard_data(url, token, get_last_timestamp(source))
    if auctions is None:
        log(f"{source}: no new data.")
        return None
    log(f"{source}: new data, {len(auctions)} auctions.")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df = build_frame(auctions, source, parse_last_modified(header), now)
    key = f"{source}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}_{now.minute:02d}.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    data = buf.getvalue()

    b2_ok = True
    try:
        b2_put(key, data)
        log(f"{source}: uploaded to B2 as {key}")
    except Exception as e:
        b2_ok = False
        log(f"{source}: B2 upload FAILED: {e}")
    return Snapshot(source, key, data, header, b2_ok)


def publish(snaps, failures):
    """Mirrors snapshots to HF in one commit, then advances markers / records pending copies."""
    if not snaps:
        return
    files = {s.key: s.data for s in snaps}
    for s in snaps:
        if s.marker:
            files[marker_key(s.source)] = s.marker.encode("utf-8")

    hf_ok = False
    try:
        hf_commit(files, f"Add {len(snaps)} snapshot(s)")
        hf_ok = True
        log(f"Hugging Face: committed {len(snaps)} snapshot(s).")
    except Exception as e:
        log(f"Hugging Face commit FAILED: {e}")

    missing_hf, missing_b2 = [], []
    for s in snaps:
        if not (s.b2_ok or hf_ok):
            failures.append(f"{s.source}: snapshot stored on neither B2 nor HF; marker NOT advanced")
            continue
        if s.marker:
            try:
                b2_put(marker_key(s.source), s.marker.encode("utf-8"))
            except Exception as e:
                log(f"{s.source}: B2 marker write failed: {e}")
        if not s.b2_ok:
            missing_b2.append(s.key)
        if not hf_ok:
            missing_hf.append(s.key)

    if missing_hf:
        add_pending(PENDING_HF, missing_hf, b2_get, b2_put)
    if missing_b2:
        add_pending(PENDING_B2, missing_b2, hf_get, lambda k, d: hf_commit({k: d}, "Update pending B2 reconcile queue"))


def main():
    log("Starting auction extraction cycle...")
    failures = []
    try:
        token = get_blizzard_token()
    except Exception as e:
        msg = f"Blizzard token request failed: {e}"
        log(msg)
        notify(f"fetch.py: {msg}")
        return 1

    reconcile()

    sources = [("commodities", "https://us.api.blizzard.com/data/wow/auctions/commodities")]
    sources += [(f"realm_{r}", f"https://us.api.blizzard.com/data/wow/connected-realm/{r}/auctions") for r in REALM_IDS]

    snaps = []
    for source, url in sources:
        try:
            snap = collect_snapshot(source, url, token)
        except Exception as e:
            failures.append(f"{source}: {e}")
            log(f"{source}: FAILED: {e}")
            continue
        if snap:
            snaps.append(snap)

    publish(snaps, failures)

    if failures:
        notify("fetch.py problems:\n" + "\n".join(failures))
        log("Cycle finished WITH FAILURES:\n  " + "\n  ".join(failures))
        return 1
    log("Cycle complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
