"""Forever auction extractor. A separate script: the retail extractor (fetch.py) is not touched and does not import this.

What it does each run (about once an hour, started by the workflow forever_fetch.yml after the retail extractor):
  1. asks Blizzard for the Forever realm list (namespace dynamic-forever-us) and finds, for each wanted realm, its connected realm (the connected realm is what owns an auction house);
  2. asks for that connected realm's auctions, with If-Modified-Since so an unchanged hour costs one tiny answer;
  3. writes every NEW snapshot, untouched, to a separate private Hugging Face dataset in the same layout as the retail lake:
        forever_<connected realm id>/year=YYYY/month=MM/day=DD/hour_HH_MM.parquet
     together with a small markers file (last Last-Modified per source) and a sources file (which realm each source is) in ONE commit. Because the markers travel in the commit with the data,
     a failed commit advances nothing and the next run simply fetches the hour again.
Storage is Hugging Face only (no Backblaze): the Forever record starts as a single copy, which also keeps it clear of the Backblaze free cap.

Before Blizzard serves Forever listings the answers are 403 / 404. That is NOT a failure: the run says "not live yet" and exits 0, so nothing goes red and no alert fires every hour. The first time a source
answers with auctions, one Discord message says so (if the DISCORD_WEBHOOK_URL secret is set). Real problems (token, 5xx after retries, a failed commit) exit 1 and alert.

Settings (environment variables; the defaults are right for launch):
  FOREVER_REALM_IDS     comma separated realm ids to collect (default: the eight US PvP game realms from the Nov 3 probe: US1 Realm 2/4/6/8 and US2 Realm 2/4/6/8)
  FOREVER_NAMESPACE     default dynamic-forever-us
  FOREVER_API_HOST      default https://us.api.blizzard.com
  FOREVER_AUCTION_URL   default <host>/data/wow/connected-realm/{cid}/auctions   (change this if Blizzard serves the listings at another path)
  FOREVER_HF_REPO       the private dataset that holds the record (default set below)
Nothing in the output names the dataset or an account: error text has the dataset name blanked out."""
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

SCHEMA_VERSION = 2
NAMESPACE = os.environ.get("FOREVER_NAMESPACE", "dynamic-forever-us")
API_HOST = os.environ.get("FOREVER_API_HOST", "https://us.api.blizzard.com").rstrip("/")
AUCTION_URL = os.environ.get("FOREVER_AUCTION_URL", API_HOST + "/data/wow/connected-realm/{cid}/auctions")
HF_REPO_ID = os.environ.get("FOREVER_HF_REPO", "bosco277/wow-forever-data-lake")
DEFAULT_REALM_IDS = [6798, 6922, 6924, 6926, 6800, 6930, 6932, 6934]     # CAMWOW US1 Realm 2/4/6/8 PvP, US2 Realm 2/4/6/8 PvP
MARKERS_KEY = "system/forever_markers.json"
SOURCES_KEY = "system/forever_sources.json"
NOT_LIVE = (403, 404)            # "Blizzard does not serve this (yet)": quiet, not an error

BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HTTP = requests.Session()


def log(msg):
    print(msg, flush=True)


def clean(text):
    """Error text for logs and alerts: the dataset name is blanked out so no account name ends up in a job log."""
    return str(text).replace(HF_REPO_ID, "<forever dataset>")[:300]


def with_retries(fn, what, attempts=3, base_delay=2.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            delay = base_delay * (2 ** i)
            log(f"{what} failed ({clean(e)}); retrying in {delay:.0f}s")
            time.sleep(delay)


def notify(message):
    """Posts to Discord if a webhook is configured. Never raises."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        HTTP.post(DISCORD_WEBHOOK_URL, json={"content": message[:1900]}, timeout=10)
    except Exception as e:
        log(f"Discord notification failed: {clean(e)}")


# --- Hugging Face (the only store) -----------------------------------------------------------------
def hf_get(key):
    """The file's bytes from the Forever dataset, or None if it (or the dataset) does not exist yet."""
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{key}"
    r = HTTP.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def ensure_repo():
    """Creates the private dataset if it does not exist. Harmless if it does; if the token may not create repositories the later commit shows the real problem."""
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        log(f"Could not confirm the Forever dataset exists ({clean(e)}); trying the commit anyway.")


def hf_commit(files, message):
    """Writes several files in a single commit."""
    from huggingface_hub import CommitOperationAdd, HfApi
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()]
    api = HfApi(token=HF_TOKEN)
    with_retries(lambda: api.create_commit(repo_id=HF_REPO_ID, repo_type="dataset", operations=ops, commit_message=message), "HF commit")


def load_json(key, default):
    data = hf_get(key)
    if not data:
        return default
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError:
        log(f"{key} is not valid JSON; starting it again.")
        return default


# --- Blizzard API ----------------------------------------------------------------------------------
def get_blizzard_token():
    def call():
        r = HTTP.post("https://oauth.battle.net/token", auth=(BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET), data={"grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        return r.json()["access_token"]
    return with_retries(call, "Blizzard token request")


def api_get(url, token, last_modified=None):
    """(status, parsed json or None, Last-Modified header). 200, 304 and 403 / 404 are answers; 429 / 5xx / network trouble are retried three times, then raise; other 4xx raise at once."""
    params = {"namespace": NAMESPACE, "locale": "en_US"}
    headers = {"Authorization": f"Bearer {token}"}
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    problem = None
    for attempt in range(3):
        try:
            r = HTTP.get(url, params=params, headers=headers, timeout=(10, 180))
        except requests.RequestException as e:
            problem = e
        else:
            if r.status_code == 304 or r.status_code in NOT_LIVE:
                return r.status_code, None, None
            if r.status_code in (429, 500, 502, 503, 504):
                problem = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r.status_code, r.json(), r.headers.get("Last-Modified")
        if attempt < 2:
            time.sleep(2 * (2 ** attempt))
    raise RuntimeError(f"Blizzard request failed after 3 attempts: {clean(problem)}")


def realm_index(token):
    """{realm id: realm entry} from the realm list, or None when Blizzard does not serve the Forever realm list."""
    status, body, _ = api_get(f"{API_HOST}/data/wow/realm/index", token)
    if body is None:
        return None
    return {int(r["id"]): r for r in body.get("realms", []) if "id" in r}


def connected_realm_id(realm, token):
    """The connected realm (auction house) a realm belongs to, read from its realm record; None when the record is not served or has no link."""
    status, body, _ = api_get(f"{API_HOST}/data/wow/realm/{realm['slug']}", token)
    href = ((body or {}).get("connected_realm") or {}).get("href", "")
    m = re.search(r"/connected-realm/(\d+)", href)
    return int(m.group(1)) if m else None


# --- Transformation (the same columns the retail extractor writes, so the lake reads alike) --------
def variant_key(item):
    """Stable identity for an item variant: plain items -> '123'; gear / pets -> '123:<hash of everything but the id>'."""
    if not isinstance(item, dict):
        return None
    item_id = item.get("id")
    base = {k: v for k, v in item.items() if k != "id"}
    if isinstance(base.get("bonus_lists"), list):
        base["bonus_lists"] = sorted(base["bonus_lists"])
    if isinstance(base.get("modifiers"), list):
        base["modifiers"] = sorted(base["modifiers"], key=lambda m: (m.get("type", 0), m.get("value", 0)) if isinstance(m, dict) else (0, 0))
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
    df["snapshot_ts"] = pd.Timestamp(snapshot_dt) if snapshot_dt is not None else pd.NaT      # Blizzard Last-Modified, naive UTC
    df["scan_timestamp"] = pd.Timestamp(fetched_at)                                           # fetch time, naive UTC
    df["schema_version"] = SCHEMA_VERSION
    return df.sort_values("item_id", kind="stable").reset_index(drop=True)


def to_parquet_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    return buf.getvalue()


# --- One cycle -------------------------------------------------------------------------------------
def wanted_realm_ids():
    raw = os.environ.get("FOREVER_REALM_IDS", "").strip()
    if not raw:
        return list(DEFAULT_REALM_IDS)
    return [int(x) for x in re.split(r"[,\s]+", raw) if x.isdigit()]


def collect(source, cid, token, markers, now):
    """('new', (key, bytes, Last-Modified header)) for a new snapshot; ('unchanged'|'dark'|'empty', None) otherwise."""
    status, body, header = api_get(AUCTION_URL.format(cid=cid), token, markers.get(source))
    if status == 304:
        return "unchanged", None
    if body is None:
        return "dark", None
    auctions = body.get("auctions", [])
    if not auctions:
        return "empty", None                  # served, but no auctions in it (possible on day one): nothing to keep, marker not advanced
    log(f"{source}: new data, {len(auctions)} auctions.")
    df = build_frame(auctions, source, parse_last_modified(header), now)
    key = f"{source}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}_{now.minute:02d}.parquet"
    return "new", (key, to_parquet_bytes(df), header)


def main():
    log("Starting Forever extraction cycle...")
    try:
        token = get_blizzard_token()
    except Exception as e:
        msg = f"Blizzard token request failed: {clean(e)}"
        log(msg)
        notify(f"fetch_forever.py: {msg}")
        return 1

    failures = []
    try:
        index = realm_index(token)
    except Exception as e:
        log(f"Realm list failed: {clean(e)}")
        notify(f"fetch_forever.py: realm list failed: {clean(e)}")
        return 1
    if index is None:
        log("Forever: Blizzard does not serve the Forever realm list (not live yet). Nothing to do.")
        return 0

    wanted = wanted_realm_ids()
    absent = [r for r in wanted if r not in index]
    if absent:
        log(f"Forever: {len(absent)} wanted realm id(s) are not in the realm list: {absent}")
    sources = {}
    for rid in wanted:
        if rid not in index:
            continue
        try:
            cid = connected_realm_id(index[rid], token)
        except Exception as e:
            failures.append(f"realm {rid}: {clean(e)}")
            log(f"realm {rid}: FAILED to read its realm record: {clean(e)}")
            continue
        if cid is None:
            log(f"realm {rid}: no connected realm served yet.")
            continue
        sources[f"forever_{cid}"] = dict(realm_id=rid, realm_name=index[rid].get("name"), connected_realm_id=cid)

    try:
        markers = load_json(MARKERS_KEY, {})
    except Exception as e:
        log(f"Could not read the markers file: {clean(e)}")
        notify(f"fetch_forever.py: could not read markers: {clean(e)}")
        return 1

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    new, counts = {}, {"unchanged": 0, "dark": 0, "empty": 0}
    for source, info in sources.items():
        try:
            state, got = collect(source, info["connected_realm_id"], token, markers, now)
        except Exception as e:
            failures.append(f"{source}: {clean(e)}")
            log(f"{source}: FAILED: {clean(e)}")
            continue
        if state == "new":
            new[source] = got
        else:
            counts[state] += 1

    first_live = [s for s in new if s not in markers]
    if new:
        files = {key: data for key, data, _ in new.values()}
        for source, (_, _, header) in new.items():
            if header:
                markers[source] = header
        files[MARKERS_KEY] = json.dumps(markers, sort_keys=True).encode("utf-8")
        try:
            known = load_json(SOURCES_KEY, {})
            known.update(sources)
            files[SOURCES_KEY] = json.dumps(known, sort_keys=True).encode("utf-8")
        except Exception as e:
            log(f"Could not read the sources file ({clean(e)}); writing only this run's sources.")
            files[SOURCES_KEY] = json.dumps(sources, sort_keys=True).encode("utf-8")
        try:
            ensure_repo()
            hf_commit(files, f"Add {len(new)} Forever snapshot(s)")
            log(f"Hugging Face: committed {len(new)} snapshot(s).")
        except Exception as e:
            failures.append(f"commit failed, nothing advanced: {clean(e)}")
            log(f"Hugging Face commit FAILED: {clean(e)}")
            first_live = []
        if first_live:
            names = ", ".join(str(sources[s].get("realm_name")) for s in first_live[:5])
            notify(f"Forever auction data is LIVE: first snapshot(s) for {len(first_live)} realm(s) ({names}). Collection is running.")

    log(f"Forever summary: {len(sources)} realm(s) resolved of {len(wanted)} wanted; new snapshots {len(new)}, unchanged {counts['unchanged']}, "
        f"not served yet {counts['dark']}, served but empty {counts['empty']}.")
    if failures:
        notify("fetch_forever.py problems:\n" + "\n".join(failures))
        log("Cycle finished WITH FAILURES:\n  " + "\n  ".join(failures))
        return 1
    log("Cycle complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
