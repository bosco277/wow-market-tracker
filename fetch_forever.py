"""Forever auction extractor. A separate script: the retail extractor (fetch.py) is not touched and does not import this.

Forever has no realms in its auction API. An unofficial community post (Nov 12; this script's own first run confirms or refutes it) says listings sit under RULESETS and AUCTION HOUSES:
    /data/wow/ruleset/{ruleset}/auctions/   (or .../auctions/index)   the auction houses (Alliance / Horde / Blackwater, five of each; both spellings are tried)
    /data/wow/ruleset/{ruleset}/auctions/{house}               that house's regular auctions (price field: buyout)
    /data/wow/ruleset/{ruleset}/auctions/{house}/commodities   that house's commodity auctions (price field: unit_price)
all with namespace dynamic-forever-us (rulesets: 1 PvP, 2 Normal, 3 Roleplay, 15 Roleplay PvP).

What it does each run (about once an hour, started by the workflow forever_fetch.yml after the retail extractor):
  1. asks for the auction-house list of each wanted ruleset (default: ruleset 1, PvP);
  2. for every house that has listings (or only the houses in FOREVER_HOUSE_IDS) reads the regular auctions and the commodities, each with If-Modified-Since so an unchanged hour costs one tiny answer;
  3. writes every NEW snapshot, untouched, to a separate private Hugging Face dataset in the same layout as the retail lake:
        forever_<region>_r<ruleset>_h<house>[_commodities]/year=YYYY/month=MM/day=DD/hour_HH_MM.parquet
     together with a small markers file (last Last-Modified per source) and a sources file (what each source is) in ONE commit. Because the markers travel in the commit with the data,
     a failed commit advances nothing and the next run simply fetches the hour again;
  4. once, checks whether the ruleset in the URL changes the data: it re-reads the biggest house through the other rulesets and compares. A Blizzard-forum post (Nov 6) says every ruleset returns the SAME houses with
     IDENTICAL auctions (the houses, not the rulesets, are the markets, and totals must not be summed across rulesets), so one ruleset is enough and this check should say "same". The answer goes to the log
     (and to Discord if the data DIFFERS, because then every ruleset is its own market and FOREVER_RULESETS must list them all). The log also shows, every run, how many listings each house has and how many MB the run wrote.
If Blizzard serves no ruleset list at all the script falls back to the first design (realm list -> connected realm -> /connected-realm/{id}/auctions), kept in case Forever also gets that shape.
Storage is Hugging Face only (no Backblaze): the Forever record starts as a single copy, which also keeps it clear of the Backblaze free cap.

Before Blizzard serves anything the answers are 403 / 404. That is NOT a failure: the run says "not live yet" and exits 0, so nothing goes red and no alert fires every hour. The first time a source
answers with auctions, one Discord message says so (if the DISCORD_WEBHOOK_URL secret is set). Real problems (token, 5xx after retries, a failed commit) exit 1 and alert.

Settings (environment variables; the defaults are right for now):
  FOREVER_RULESETS      comma separated ruleset ids to read the house lists from (default 1). In GitHub: workflow input `rulesets`, else the repository variable FOREVER_RULESETS.
  FOREVER_HOUSE_IDS     comma separated auction-house ids to collect (default: every house that has listings). Workflow input `house_ids`, else the repository variable FOREVER_HOUSE_IDS.
  FOREVER_REALM_IDS     only for the fallback design: comma separated realm ids (default: the eight US PvP game realms from the Nov 3 probe). Workflow input `realm_ids`, else the variable FOREVER_REALM_IDS.
  FOREVER_NAMESPACE     default dynamic-forever-us (the region in source names is taken from its last word)
  FOREVER_API_HOST      default https://us.api.blizzard.com
  FOREVER_AUCTION_URL   fallback design only: default <host>/data/wow/connected-realm/{cid}/auctions
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
REGION = NAMESPACE.rsplit("-", 1)[-1]                 # "us" for dynamic-forever-us: goes into source names so a later EU record cannot collide
DEFAULT_RULESETS = [1]                                # 1 = PvP, the ruleset of the planned character; the other rulesets are read once to check whether the URL's ruleset matters
KNOWN_RULESETS = (1, 2, 3, 15)                        # 1 PvP, 2 Normal, 3 Roleplay, 15 Roleplay PvP (community post, Nov 12)
CHECK_KEY = "_ruleset_check"                          # entry in the markers file: the answer to "does the ruleset in the URL change the data?"

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


def house_entries(body):
    """{house id: entry} from an auction-house index answer. The list is expected under "auctions"; any other list of entries with an id is accepted in case Blizzard names it differently."""
    if not isinstance(body, dict):
        return {}
    rows = None
    for key in ("auctions", "houses", "auction_houses"):
        if isinstance(body.get(key), list):
            rows = body[key]
            break
    if rows is None:
        rows = next((v for v in body.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]), [])
    return {int(h["id"]): h for h in rows if isinstance(h, dict) and str(h.get("id", "")).isdigit()}


def house_name(entry, house_id):
    name = entry.get("name")
    if isinstance(name, dict):
        name = name.get("en_US") or next(iter(name.values()), None)
    return str(name) if name else f"house {house_id}"


def house_index(token, ruleset):
    """{house id: entry} for a ruleset, or None when Blizzard serves no auction-house list for it under either spelling of the route.
    The route has changed once already: on launch day (Nov 5) a community post found `/auctions/index` working and `/auctions` a 404; the next morning (Nov 6) a Blizzard-forum post found the trailing-slash form
    `/auctions/` working. Both are tried, the slash form first, and the first one that lists houses is used."""
    served = False
    for tail in ("/", "/index"):
        status, body, _ = api_get(f"{API_HOST}/data/wow/ruleset/{ruleset}/auctions{tail}", token)
        if body is None:
            continue
        served = True
        entries = house_entries(body)
        if entries:
            return entries
    return {} if served else None


def digest_of(auctions):
    return hashlib.sha1(json.dumps(auctions, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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


def int_list(name):
    """The integers in an environment variable (commas or spaces between them), or None when it is blank."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [int(x) for x in re.split(r"[,\s]+", raw) if x.isdigit()]


def wanted_rulesets():
    return int_list("FOREVER_RULESETS") or list(DEFAULT_RULESETS)


def wanted_house_ids():
    """None = every house that has listings."""
    return int_list("FOREVER_HOUSE_IDS")


def collect_url(source, url, token, markers, now):
    """('new', {key, data, header, rows, digest}) for a new snapshot; ('unchanged'|'dark'|'empty', None) otherwise."""
    marker = markers.get(source)
    marker = marker if isinstance(marker, str) else None
    status, body, header = api_get(url, token, marker)
    if status == 304:
        return "unchanged", None
    if body is None:
        return "dark", None
    auctions = body.get("auctions", [])
    if not auctions:
        return "empty", None                  # served, but no auctions in it (possible on day one): nothing to keep, marker not advanced
    if header and header == marker:
        return "unchanged", None              # the server ignored If-Modified-Since but says this is the snapshot we already have
    log(f"{source}: new data, {len(auctions)} auctions.")
    df = build_frame(auctions, source, parse_last_modified(header), now)
    key = f"{source}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}_{now.minute:02d}.parquet"
    return "new", dict(key=key, data=to_parquet_bytes(df), header=header, rows=len(auctions), digest=digest_of(auctions))


def collect(source, cid, token, markers, now):
    """The first design (one connected realm = one source): ('new', (key, bytes, Last-Modified header)) or ('unchanged'|'dark'|'empty', None)."""
    state, got = collect_url(source, AUCTION_URL.format(cid=cid), token, markers, now)
    return state, ((got["key"], got["data"], got["header"]) if got else None)


def ruleset_check(token, new, plan):
    """Does the ruleset in the URL change the data? Re-reads the biggest regular-auctions snapshot of this run through the other known rulesets and compares the content.
    Returns {"house", "same", "differ", "unserved"} or None when it cannot say yet (no regular snapshot this run, or a snapshot changed between the two reads, which would look like a difference)."""
    candidates = [(got["rows"], s) for s, got in new.items() if plan[s]["kind"] == "auctions"]
    if not candidates:
        return None
    _, source = max(candidates)
    info, mine = plan[source], new[source]
    result = dict(house=info["house_id"], same=[], differ=[], unserved=[])
    for rs in KNOWN_RULESETS:
        if rs == info["ruleset"]:
            continue
        status, body, header = api_get(f"{API_HOST}/data/wow/ruleset/{rs}/auctions/{info['house_id']}", token)
        if body is None or not body.get("auctions"):
            result["unserved"].append(rs)
        elif digest_of(body["auctions"]) == mine["digest"]:
            result["same"].append(rs)
        elif header and header == mine["header"]:
            result["differ"].append(rs)                          # same snapshot time, different content: a real difference
        else:
            return None                                          # a new snapshot landed between the two reads: inconclusive, try again next hour
    return result


def describe_check(r):
    if r["differ"]:
        return (f"house {r['house']} returns DIFFERENT data through ruleset(s) {r['differ']}: the ruleset in the URL matters, so each ruleset is its own market. "
                f"Set the variable FOREVER_RULESETS to every ruleset you want (e.g. 1,2,3,15) and tell Claude")
    if r["same"]:
        extra = f" ({r['unserved']} not served)" if r["unserved"] else ""
        return f"house {r['house']} returns the SAME data through rulesets {r['same']}{extra}: the ruleset in the URL does not change the data, one ruleset covers every house"
    return f"house {r['house']}: the other rulesets {r['unserved']} are not served, so there was nothing to compare"


def run_houses(token, served):
    """The ruleset / auction-house design. `served` = {ruleset: {house id: entry}} for the rulesets whose house list Blizzard serves."""
    wanted_houses = wanted_house_ids()
    plan = {}
    listed = 0
    for rs in sorted(served):
        index = served[rs]
        listed += len(index)
        log(f"Forever: ruleset {rs} lists {len(index)} auction house(s): " + "; ".join(f"{h} {house_name(index[h], h)}" for h in sorted(index)))
        for hid in sorted(index):
            if wanted_houses is not None and hid not in wanted_houses:
                continue
            for kind, suffix in (("auctions", ""), ("commodities", "_commodities")):
                plan[f"forever_{REGION}_r{rs}_h{hid}{suffix}"] = dict(region=REGION, ruleset=rs, house_id=hid, house_name=house_name(index[hid], hid), kind=kind)
    if not listed:
        log("Forever: the auction-house list is served but empty. Nothing to do.")
        return 0
    if not plan:
        msg = (f"The Forever auction-house list is live ({listed} house entries) but none of the wanted house ids {wanted_houses} are in it, so nothing is being collected. "
               "Clear or correct the repository variable FOREVER_HOUSE_IDS.")
        log("Forever: " + msg)
        notify("fetch_forever.py: " + msg)
        return 0

    try:
        markers = load_json(MARKERS_KEY, {})
    except Exception as e:
        log(f"Could not read the markers file: {clean(e)}")
        notify(f"fetch_forever.py: could not read markers: {clean(e)}")
        return 1

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    failures, new = [], {}
    states = {}
    for source, info in plan.items():
        suffix = "/commodities" if info["kind"] == "commodities" else ""
        url = f"{API_HOST}/data/wow/ruleset/{info['ruleset']}/auctions/{info['house_id']}{suffix}"
        try:
            state, got = collect_url(source, url, token, markers, now)
        except Exception as e:
            failures.append(f"{source}: {clean(e)}")
            log(f"{source}: FAILED: {clean(e)}")
            continue
        states[source] = state
        if state == "new":
            new[source] = got
    counts = {k: sum(1 for v in states.values() if v == k) for k in ("unchanged", "dark", "empty")}

    # which houses have what (the survey a person wants to read), and any houses that came back with identical content
    for rs in sorted(served):
        for hid in sorted(served[rs]):
            reg, com = f"forever_{REGION}_r{rs}_h{hid}", f"forever_{REGION}_r{rs}_h{hid}_commodities"
            if reg not in plan:
                continue
            def show(src):
                return f"{new[src]['rows']} listings" if src in new else {"unchanged": "unchanged", "dark": "not served", "empty": "empty"}.get(states.get(src), "failed")
            log(f"  ruleset {rs} house {hid} ({plan[reg]['house_name']}): auctions {show(reg)}, commodities {show(com)}")
    by_digest = {}
    for source, got in new.items():
        by_digest.setdefault((plan[source]["kind"], got["digest"]), []).append(source)
    for (kind, _), group in by_digest.items():
        if len(group) > 1:
            log(f"Forever: these {kind} sources returned IDENTICAL content (possible duplicate markets): {', '.join(group)}")

    check = None
    if new and CHECK_KEY not in markers:
        try:
            check = ruleset_check(token, new, plan)
        except Exception as e:
            log(f"Ruleset check failed ({clean(e)}); it will be tried again next hour.")
    if check:
        log("Forever ruleset check: " + describe_check(check))
        if check["differ"]:
            notify("Forever: " + describe_check(check))

    first_live = [s for s in new if s not in markers]
    total = sum(len(got["data"]) for got in new.values())
    if new:
        files = {got["key"]: got["data"] for got in new.values()}
        for source, got in new.items():
            if got["header"]:
                markers[source] = got["header"]
        if check:
            markers[CHECK_KEY] = check
        files[MARKERS_KEY] = json.dumps(markers, sort_keys=True).encode("utf-8")
        try:
            known = load_json(SOURCES_KEY, {})
            known.update(plan)
            files[SOURCES_KEY] = json.dumps(known, sort_keys=True).encode("utf-8")
        except Exception as e:
            log(f"Could not read the sources file ({clean(e)}); writing only this run's sources.")
            files[SOURCES_KEY] = json.dumps(plan, sort_keys=True).encode("utf-8")
        try:
            ensure_repo()
            hf_commit(files, f"Add {len(new)} Forever snapshot(s)")
            log(f"Hugging Face: committed {len(new)} snapshot(s), {total / 1e6:.1f} MB of parquet.")
        except Exception as e:
            failures.append(f"commit failed, nothing advanced: {clean(e)}")
            log(f"Hugging Face commit FAILED: {clean(e)}")
            first_live = []
        if first_live:
            houses = sorted({(plan[s]["house_id"], plan[s]["house_name"]) for s in first_live})
            names = ", ".join(f"{h} {n}" for h, n in houses[:6]) + (" ..." if len(houses) > 6 else "")
            note = f" Ruleset check: {describe_check(check)}." if check else ""
            notify(f"Forever auction data is LIVE: first snapshot(s) for {len(first_live)} source(s) across {len(houses)} auction house(s) ({names}). "
                   f"This run wrote {total / 1e6:.1f} MB of parquet.{note}")

    log(f"Forever summary: {len(served)} ruleset(s) read, {len(plan) // 2} house(s) wanted of {listed} listed; new snapshots {len(new)}, unchanged {counts['unchanged']}, "
        f"not served yet {counts['dark']}, served but empty {counts['empty']}.")
    if failures:
        notify("fetch_forever.py problems:\n" + "\n".join(failures))
        log("Cycle finished WITH FAILURES:\n  " + "\n  ".join(failures))
        return 1
    log("Cycle complete!")
    return 0


def main():
    log("Starting Forever extraction cycle...")
    try:
        token = get_blizzard_token()
    except Exception as e:
        msg = f"Blizzard token request failed: {clean(e)}"
        log(msg)
        notify(f"fetch_forever.py: {msg}")
        return 1

    served = {}
    try:
        for rs in wanted_rulesets():
            index = house_index(token, rs)
            if index is not None:
                served[rs] = index
    except Exception as e:
        log(f"Auction-house list failed: {clean(e)}")
        notify(f"fetch_forever.py: auction-house list failed: {clean(e)}")
        return 1
    if served:
        return run_houses(token, served)
    return run_realms(token)


def run_realms(token):
    """The first design: realm list -> connected realm -> /connected-realm/{id}/auctions. Used only when Blizzard serves no ruleset list."""
    failures = []
    try:
        index = realm_index(token)
    except Exception as e:
        log(f"Realm list failed: {clean(e)}")
        notify(f"fetch_forever.py: realm list failed: {clean(e)}")
        return 1
    if index is None:
        log("Forever: Blizzard does not serve the Forever auction-house or realm lists (not live yet). Nothing to do.")
        return 0

    wanted = wanted_realm_ids()
    absent = [r for r in wanted if r not in index]
    if absent:
        log(f"Forever: {len(absent)} wanted realm id(s) are not in the realm list: {absent}")
    if not index:
        log("Forever: the realm list is served but empty. Nothing to do.")
        return 0
    if wanted and len(absent) == len(wanted):
        shown = "; ".join(f"{i} {index[i].get('name')}" for i in sorted(index)[:25])
        more = f" (first 25 of {len(index)} shown)" if len(index) > 25 else ""
        msg = (f"The Forever realm list is live ({len(index)} realms) but none of the wanted realm ids are in it, so nothing is being collected. "
               f"Set the repository variable FOREVER_REALM_IDS to the ids you want (comma separated). Realms{more}: {shown}")
        log("Forever: " + msg)
        notify("fetch_forever.py: " + msg)
        return 0
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
