"""Forever probe: three read-only questions in one run. Nothing is written to any dataset.

    python tools/forever_probe.py                       # all three sections
    python tools/forever_probe.py --sections items      # one of: structure, items, fingerprint (comma separated)
    python tools/forever_probe.py --all-rulesets        # structure: read every house through all four rulesets (default: rulesets 1 and 2, half the downloads)

  1 STRUCTURE    What does Blizzard's Forever API say about how markets are laid out? The region, ruleset and auction-house answers, the status of the realm and connected-realm routes, and, for
                 every house, the `ruleset` field of its own record as read through each ruleset's URL (a house that says "ruleset 2" whichever URL it is read through would state its own ruleset)
                 together with a fingerprint of its auction ids (the same number everywhere = the same data).
  2 ITEMS        Every item id seen in the newest snapshot of every house, looked up in Forever's static data first and then Classic Era, Classic and retail: how many ids each source names, how
                 many nobody names, and the biggest unnamed ids by gold in play. Also whether each source serves professions and recipes (a crafting tool needs reagent lists).
  3 FINGERPRINT  Eight pages of foreverwowcraft.com (US PvE / PvP / RP / RP PvP x Alliance / Horde) list about 15 items each with a quantity and the average price of buying that quantity. The same
                 purchases are priced from each of our houses over the last --hours hours; the house whose prices line up with a page is that page's house. A method check comes first: the US PvP
                 Alliance page must line up with house 110, the one house a listing test has confirmed.

Writes ONE plain-text report, out/forever_probe.txt (a .txt on purpose: lines starting with "=" turn into #ERROR! when a spreadsheet opens them).
Environment: HF_TOKEN (read access to the Forever dataset); BLIZZARD_CLIENT_ID and BLIZZARD_CLIENT_SECRET (the same secrets the extractors use); FOREVER_HF_REPO to name another dataset.
The dataset's name is blanked out of everything printed, so no account name reaches a job log. The fixture of site prices is tools/fwc_fingerprint.json (a point-in-time capture; see its `note`).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import forever as FV  # noqa: E402
from engine import foreveritems as FI  # noqa: E402
from engine import housemap as HM  # noqa: E402
from engine.loader import HFStore, RangeFile, normalize_frame, read_parquet_file  # noqa: E402
from engine.probe import describe  # noqa: E402

API = "https://us.api.blizzard.com"
DYNAMIC = "dynamic-forever-us"
RULESETS = (1, 2, 3, 15)                      # 1 PvP, 2 Normal (PvE), 3 Roleplay, 15 Roleplay PvP (community post, Nov 12)
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fwc_fingerprint.json")
SECTIONS = ("structure", "items", "fingerprint")
COPPER_PER_GOLD = 10_000.0


def url_for(path, namespace):
    return f"{API}{path}{'&' if '?' in path else '?'}namespace={namespace}&locale=en_US"


# ------------------------------------------------------------------------------------------------------------ Blizzard client
class Blizzard:
    """get(url) -> (status, json or None). 429 / 5xx / network trouble are retried three times; everything else is an answer."""

    def __init__(self, client_id, secret, session=None, sleep=time.sleep):
        import requests
        self.http, self.sleep = session or requests.Session(), sleep
        r = self.http.post("https://oauth.battle.net/token", auth=(client_id, secret), data={"grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        self.headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def get(self, url):
        status = 0
        for k in range(3):
            try:
                r = self.http.get(url, headers=self.headers, timeout=(10, 180))
                status = r.status_code
                if status == 200:
                    try:
                        return 200, r.json()
                    except ValueError:
                        return 200, None
                if status not in (429, 500, 502, 503, 504):
                    return status, None
            except Exception:
                status = 0
            if k < 2:
                self.sleep(2 * (2 ** k))
        return status, None


def _try(get, url):
    try:
        code, js = get(url)
    except Exception:
        return 0, None
    return int(code), js


# ------------------------------------------------------------------------------------------------------------ 1 structure
def rs_id(value):
    """The ruleset number out of whatever shape the `ruleset` field has (a number, {id}, or {key: {href: .../ruleset/2}})."""
    if isinstance(value, dict):
        if str(value.get("id", "")).isdigit():
            return int(value["id"])
        href = ((value.get("key") or {}).get("href")) or value.get("href") or ""
        digits = "".join(ch if ch.isdigit() else " " for ch in href.split("/ruleset/")[-1]).split()
        return int(digits[0]) if "/ruleset/" in href and digits else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def house_entries(body):
    if not isinstance(body, dict):
        return {}
    rows = next((body[k] for k in ("auctions", "houses", "auction_houses") if isinstance(body.get(k), list)), None)
    if rows is None:
        rows = next((v for v in body.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]), [])
    return {int(h["id"]): h for h in rows if isinstance(h, dict) and str(h.get("id", "")).isdigit()}


def house_label(entry, house):
    name = entry.get("name") if isinstance(entry, dict) else None
    if isinstance(name, dict):
        name = name.get("en_US") or next(iter(name.values()), None)
    return str(name) if name else f"house {house}"


def ids_digest(auctions):
    ids = sorted(int(a["id"]) for a in auctions if isinstance(a, dict) and str(a.get("id", "")).isdigit())
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:8], len(ids)


def structure_section(get, rulesets=RULESETS, sleep=time.sleep):
    out = []

    def ep(path, ns=DYNAMIC, depth=2):
        sleep(0.1)
        code, js = _try(get, url_for(path, ns))
        shape = f"  {describe(js, depth=depth, max_keys=12)}" if js is not None else ""
        out.append(f"{path:<46} {code}{shape}"[:420])
        return code, js

    out.append("Routes (namespace dynamic-forever-us). 200 = served; 404 = the namespace exists and has nothing there; 403 = Blizzard does not know the namespace.")
    for path in ("/data/wow/region/index", "/data/wow/ruleset/index", "/data/wow/realm/index", "/data/wow/connected-realm/index", "/data/wow/search/realm?orderby=id",
                 "/data/wow/search/connected-realm?orderby=id"):
        ep(path)
    out.append("")
    out.append("Ruleset records:")
    for rs in rulesets:
        sleep(0.1)
        code, js = _try(get, url_for(f"/data/wow/ruleset/{rs}", DYNAMIC))
        text = json.dumps(js, separators=(",", ":"))[:360] if js is not None else ""
        out.append(f"  ruleset {rs}: {code} {text}")
    out.append("")
    lists = {}
    for rs in rulesets:
        for tail in ("/", "/index"):
            sleep(0.1)
            code, js = _try(get, url_for(f"/data/wow/ruleset/{rs}/auctions{tail}", DYNAMIC))
            ents = house_entries(js)
            if ents:
                lists[rs] = ents
                break
    for rs, ents in lists.items():
        out.append(f"Houses listed under ruleset {rs}: {len(ents)}: " + "; ".join(f"{h} {house_label(e, h)}" for h, e in sorted(ents.items())))
    if not lists:
        out.append("No ruleset served a list of auction houses.")
        return "\n".join(out)
    sets = {rs: tuple(sorted(e)) for rs, e in lists.items()}
    out.append("The house list is " + ("the SAME under every ruleset." if len(set(sets.values())) == 1 else f"DIFFERENT between rulesets: {sets}"))
    out.append("")
    first = lists[min(lists)]
    out.append("House records read through each ruleset's URL (the `ruleset` field of the record, auctions in it, and a fingerprint of the auction ids):")
    seen = {}
    for house in sorted(first):
        cells = []
        for rs in rulesets:
            sleep(0.2)
            code, js = _try(get, url_for(f"/data/wow/ruleset/{rs}/auctions/{house}", DYNAMIC))
            if js is None:
                cells.append((rs, code, None, None, None))
                continue
            dg, n = ids_digest(js.get("auctions") or [])
            cells.append((rs, code, rs_id(js.get("ruleset")), n, dg))
        seen[house] = cells
        out.append(f"  house {house} {house_label(first[house], house)}: " + " | ".join(
            f"via {rs}: " + (f"{code}" if n is None else f"field={fld if fld is not None else '?'} n={n} #{dg}") for rs, code, fld, n, dg in cells))
    own = {}
    for house, cells in seen.items():
        fields = {fld for _, _, fld, n, _ in cells if n is not None and fld is not None}
        same = len({dg for _, _, _, n, dg in cells if n is not None}) <= 1
        own[house] = (fields, same)
    out.append("")
    out.append("Reading: " + "; ".join(f"house {h}: ruleset field {sorted(f) if f else 'absent'}{' (same data under every URL)' if s else ' (data differs by URL)'}" for h, (f, s) in sorted(own.items())))
    varied = [h for h, (f, _) in own.items() if len(f) > 1]
    stated = [h for h, (f, _) in own.items() if len(f) == 1]
    if varied:
        out.append(f"The `ruleset` field FOLLOWS THE URL for houses {varied}: it names the ruleset that was asked for, not a property of the house.")
    if stated and not varied:
        out.append("The `ruleset` field is the same whichever URL is used: " + ", ".join(f"house {h} = ruleset {next(iter(own[h][0]))}" for h in sorted(stated)) + ". If those differ between houses, they ARE each house's own ruleset.")
    if not stated and not varied:
        out.append("No house record carries a `ruleset` field we could read: this route cannot map houses to rulesets.")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------ the newest snapshot of every source
def newest_object(store, source, now):
    for back in range(3):
        d = (now - timedelta(days=back)).date()
        objs = [o for o in store.list(f"{source}/year={d.year}/month={d.month:02d}/day={d.day:02d}/") if o.key.endswith(".parquet")]
        if objs:
            return max(objs, key=lambda o: o.key)
    return None


def newest_book(store, house, commodity, now, base_reader=None):
    """DataFrame item_id, units, price (copper per unit), or None when the source has no recent file."""
    source = FV.source_name(house, commodity)
    obj = newest_object(store, source, now)
    if obj is None:
        return None
    raw = (base_reader or read_parquet_file)(RangeFile(store, obj.key, obj.size), None)
    raw, _, _ = FV.drop_duplicate_ids(raw)
    f = normalize_frame(raw, source)
    price = f["unit_price"] if "unit_price" in f.columns else pd.Series(np.nan, index=f.index)
    if "buyout" in f.columns:
        price = price.fillna(f["buyout"] / f["quantity"])
    out = pd.DataFrame({"item_id": f["item_id"].astype("int64"), "units": f["quantity"].astype("float64"), "price": price.astype("float64")}).dropna()
    return out[out["price"] > 0]


def newest_summary(store, houses, now, base_reader=None, progress=None):
    """(ids, gold in play per id, commodity units per id, notes) over the newest snapshot of each house and kind."""
    value, units, notes = {}, {}, []
    for h in houses:
        for commodity in (True, False):
            try:
                b = newest_book(store, h, commodity, now, base_reader)
            except Exception as e:
                notes.append(f"house {h} {'commodity' if commodity else 'auction'}: not read ({FV.scrub(e)})")
                continue
            if b is None:
                continue
            g = (b["units"] * b["price"]).groupby(b["item_id"]).sum() / COPPER_PER_GOLD
            for i, v in g.items():
                value[int(i)] = value.get(int(i), 0.0) + float(v)
            if commodity:
                for i, u in b.groupby("item_id")["units"].sum().items():
                    units[int(i)] = units.get(int(i), 0.0) + float(u)
            if progress:
                progress(f"  house {h} {'commodity' if commodity else 'auction'}: {len(b):,} listings")
    return sorted(value), pd.Series(value, dtype="float64"), pd.Series(units, dtype="float64"), notes


# ------------------------------------------------------------------------------------------------------------ 2 items
def recipe_probe(get, label, namespace, sleep=time.sleep):
    """Status line for professions / recipes in one namespace: index, first profession, its first skill tier, a sample recipe."""
    def call(path):
        sleep(0.1)
        return _try(get, url_for(path, namespace))

    parts = []
    code, js = call("/data/wow/profession/index")
    profs = (js or {}).get("professions") or []
    parts.append(f"profession index {code}" + (f" ({len(profs)} professions)" if code == 200 else ""))
    if not profs:
        return f"  {label:<12} " + "; ".join(parts)
    pid = profs[0].get("id")
    code, js = call(f"/data/wow/profession/{pid}")
    tiers = (js or {}).get("skill_tiers") or []
    parts.append(f"profession {pid} {code}" + (f" ({len(tiers)} skill tiers)" if code == 200 else ""))
    if not tiers:
        return f"  {label:<12} " + "; ".join(parts)
    tid = tiers[0].get("id")
    code, js = call(f"/data/wow/profession/{pid}/skill-tier/{tid}")
    recipes = [r for c in ((js or {}).get("categories") or []) for r in (c.get("recipes") or [])]
    parts.append(f"skill tier {tid} {code}" + (f" ({len(recipes)} recipes)" if code == 200 else ""))
    if not recipes:
        return f"  {label:<12} " + "; ".join(parts)
    code, js = call(f"/data/wow/recipe/{recipes[0].get('id')}")
    reag = (js or {}).get("reagents")
    parts.append(f"sample recipe {code}" + (f" ({len(reag)} reagents listed)" if isinstance(reag, list) else (" (no reagent list)" if code == 200 else "")))
    return f"  {label:<12} " + "; ".join(parts)


def items_section(get, ids, value, sleep=time.sleep, workers=8, per_second=20.0):
    """(text, items DataFrame). Resolves `ids` through FI.NAMESPACES and prints the coverage and the recipe probe."""
    items, att = FI.resolve_items(ids, get, workers=workers, per_second=per_second, sleep=sleep)
    out = [FI.coverage_text(ids, items, att, value), "", "Professions and recipes (a crafting tool needs reagent lists):"]
    for label, ns in FI.NAMESPACES:
        out.append(recipe_probe(get, label, ns, sleep))
    return "\n".join(out), items


# ------------------------------------------------------------------------------------------------------------ 3 fingerprint
def load_fixture(path=FIXTURE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def snapshots_for(store, houses, ids, now, hours, base_reader=None):
    """{house: commodity snapshots of the last `hours` hours}, loaded one house at a time with the item filter."""
    start = (now - timedelta(hours=hours)).date()
    out, notes = {}, []
    for h in houses:
        try:
            sl = FV.load_source(store, h, True, start, now.date(), item_ids=ids, on_error="skip", base_reader=base_reader)
        except Exception as e:
            notes.append(f"house {h}: not loaded ({FV.scrub(e)})")
            continue
        lo = pd.Timestamp(now) - pd.Timedelta(hours=hours)
        snaps = [s for s in sl.snapshots if s.ts >= lo]
        if snaps:
            out[h] = snaps
        if sl.result.skipped:
            notes.append(f"house {h}: {len(sl.result.skipped)} unreadable file(s) skipped")
    return out, notes


def fingerprint_section(store, items, units, houses, now, hours, base_reader=None, fixture=None):
    fx = fixture or load_fixture()
    names = {r["item"] for p in fx["houses"].values() for r in p["rows"] if r.get("source") == "ah"}
    sub = items[items["name"].isin(names)] if len(items) else items
    ids_by_name = HM.name_map(sub, units)
    out = [f"Fixture: {len(fx['houses'])} pages, captured {fx.get('captured', '?')}. Item names mapped to ids: {len(ids_by_name)} of {len(names)}"
           + (f" (not found: {', '.join(sorted(names - set(ids_by_name)))})" if names - set(ids_by_name) else "") + "."]
    if len(ids_by_name) < HM.MIN_ITEMS:
        out.append("Too few names could be mapped to item ids to compare anything.")
        return "\n".join(out)
    wanted_ids = set(ids_by_name.values())
    snaps, notes = snapshots_for(store, houses, wanted_ids, now, hours, base_reader)
    out += notes
    if len(snaps) < 2:
        out.append("Fewer than two houses could be loaded: nothing to compare.")
        return "\n".join(out)
    long = HM.fingerprint(snaps, fx, ids_by_name)
    wide = HM.wide_scores(long)
    out.append(f"Houses compared: {', '.join(map(str, sorted(snaps)))}; snapshots per house: {min(len(v) for v in snaps.values())} to {max(len(v) for v in snaps.values())} (the last {hours} hours).")
    out.append("")
    out += HM.verdicts(wide)
    out += ["", "Score matrix (median |ln(our price / page price)|; lower is closer):", HM.matrix_text(wide), ""]
    out.append("Best snapshot time per page (UTC; the page's own build time is in the fixture for three pages):")
    for slug in wide.index:
        r = HM.ranked(wide, slug)
        if not r:
            continue
        h = r[0][0]
        row = long[(long["slug"] == slug) & (long["house"] == h)].iloc[0]
        hint = fx["houses"][slug].get("hints", {}).get("page_modified")
        out.append(f"  {slug:<18} house {h}: {pd.Timestamp(row['ts']).strftime('%m-%d %H:%M')} ({int(row['items'])} items)" + (f"   page built {hint}" if hint else ""))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------ run
def run(args, get, store, now=None, base_reader=None, sleep=time.sleep, fixture=None):
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    want = SECTIONS if args.sections == "all" else tuple(s.strip() for s in args.sections.split(",") if s.strip())
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        return dict(text=f"### forever probe\nUnknown section(s): {bad}. Use {SECTIONS}.", code=1)
    available = FV.houses_in_store(store) if ("items" in want or "fingerprint" in want) else []
    if ("items" in want or "fingerprint" in want) and not available:
        return dict(text="### forever probe\nThe Forever dataset looks empty or unreachable: its sources file was not found. Check HF_TOKEN (read access) and FOREVER_HF_REPO.", code=1)
    out = ["### forever probe", f"run at {pd.Timestamp(now).strftime('%Y-%m-%d %H:%M')} UTC; sections: {', '.join(want)}"]
    if "structure" in want:
        out += ["", "### 1 structure"]
        try:
            out.append(structure_section(get, RULESETS if args.all_rulesets else RULESETS[:2], sleep))
        except Exception as e:
            out.append(f"structure section failed: {FV.scrub(e)}")
    items, units = pd.DataFrame(columns=["item_id", "name", "source"]), pd.Series(dtype="float64")
    if "items" in want or "fingerprint" in want:
        try:
            ids, value, units, notes = newest_summary(store, available, now, base_reader)
            text, items = items_section(get, ids, value, sleep) if ids else ("No item ids found in the newest snapshots.", items)
            if "items" in want:
                out += ["", "### 2 items"] + notes + [text]
        except Exception as e:
            out += ["", f"items section failed: {FV.scrub(e)}"]
    if "fingerprint" in want:
        out += ["", "### 3 house fingerprint"]
        try:
            houses = [h for h in available if any(h in v for v in HM.FACTION_HOUSES.values())]
            out.append(fingerprint_section(store, items, units, houses, now, args.hours, base_reader, fixture))
        except Exception as e:
            out.append(f"fingerprint section failed: {FV.scrub(e)}")
    return dict(text="\n".join(out), code=0, items=items)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="all", help="all, or a comma separated list of: structure, items, fingerprint")
    ap.add_argument("--hours", type=int, default=40, help="fingerprint: how many hours of snapshots to search (default 40)")
    ap.add_argument("--all-rulesets", action="store_true", dest="all_rulesets", help="structure: read each house through all four rulesets (default: 1 and 2, enough to tell whether the record's ruleset follows the URL)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    cid, sec, tok = os.environ.get("BLIZZARD_CLIENT_ID"), os.environ.get("BLIZZARD_CLIENT_SECRET"), os.environ.get("HF_TOKEN")
    if not (cid and sec and tok):
        print("forever_probe needs BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET and HF_TOKEN.")
        return 1
    try:
        r = run(args, Blizzard(cid, sec).get, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except Exception as e:
        print(f"forever_probe failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_probe.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
