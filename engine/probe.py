"""Launch-readiness probes. Neither one stores anything: they print status codes and the SHAPE of answers (field names and counts, never values).

blizzard probe  Is there a Game Data API namespace for WoW Forever yet? Known namespaces (retail, Classic Era, Season of Discovery, Anniversary)
                answer 200; a namespace Blizzard does not know answers 403. The probe tries a list of guesses (camelot, forever, classicplus, ...),
                compares each with a gibberish namespace, and only digs deeper (realms, auctions) into the ones that answer differently.
ahledger probe  What does AHLedger's open API offer? It looks for the API's own description (OpenAPI file or docs page), lists the endpoints
                and their parameters, then fetches a few that need no parameters and prints only the shape of what comes back.
"""
import re
import time

CONTROLS = ("", "classic", "classic1x", "classicann")   # retail, Progression Classic, Classic Era, Anniversary: all known to answer
UNKNOWN = "zzqxv"                                       # gibberish: what a namespace Blizzard does not know looks like
CANDIDATES = (
    "camelot", "camelot1x", "classiccamelot", "classic-camelot", "camelotclassic",
    "forever", "forever1x", "wowforever", "wow-forever", "classicforever", "classic-forever", "foreverclassic", "forever-classic",
    "classicplus", "classic-plus", "classic_plus", "classicplus1x", "classic2", "classic2x", "classic3x", "classic4x", "classic5x", "classic6x",
    "classicfv", "fv", "wowfv", "era", "wowc", "classicfe", "forevereu",
)
STAGE1 = (("static", "/data/wow/item-class/index"), ("dynamic", "/data/wow/connected-realm/index"))
STAGE2 = (
    ("dynamic", "/data/wow/auctions/commodities"),
    ("dynamic", "/data/wow/auctions/index"),
    ("dynamic", "/data/wow/realm/index"),
    ("dynamic", "/data/wow/market/index"),
    ("static", "/data/wow/realm/index"),
    ("static", "/data/wow/item/2589"),
)
REALM_PATHS = ("/data/wow/connected-realm/{id}/auctions/index", "/data/wow/connected-realm/{id}/auctions")
DIVE_CONTROLS = ("classic", "classic1x")      # known namespaces that get the same realm-by-realm follow-up as a flagged one, for comparison
MAX_DIVE_REALMS = 3
OTHER_REGIONS = ("eu", "kr", "tw")

USER_AGENT = "wow-market-tracker/0.7 research probe (personal, non-commercial; about 1 request per second)"
AHL_BASE = "https://api.ahledger.com"
DOC_PATHS = ("/openapi.json", "/swagger.json", "/docs", "/", "/v1", "/api", "/api/v1", "/health")
PREFER = ("market", "price", "scan", "deal", "export", "item", "history", "hour", "day")


def describe(js, depth=3, max_keys=20):
    """The shape of a JSON value: field names and types and list lengths, never the values themselves."""
    if isinstance(js, dict):
        if depth <= 0:
            return "{...}"
        keys = list(js)[:max_keys]
        inner = ", ".join(f"{k}: {describe(js[k], depth - 1, max_keys)}" for k in keys)
        more = f", +{len(js) - max_keys} more" if len(js) > max_keys else ""
        return "{" + inner + more + "}"
    if isinstance(js, list):
        return "list[0]" if not js else f"list[{len(js)}] of {describe(js[0], depth - 1, max_keys)}"
    if isinstance(js, bool):
        return "bool"
    if isinstance(js, int):
        return "int"
    if isinstance(js, float):
        return "float"
    if js is None:
        return "null"
    if isinstance(js, str):
        return "str"
    return type(js).__name__


def namespace(kind, name, region="us"):
    return f"{kind}-{name}-{region}" if name else f"{kind}-{region}"


def _call(get, region, kind, name, path, sleep):
    sleep(0.05)
    url = f"https://{region}.api.blizzard.com{path}?namespace={namespace(kind, name, region)}&locale=en_US"
    try:
        status, js = get(url)
    except Exception:
        return 0, None
    return int(status), js


def _stage1(get, region, name, sleep):
    return {path: _call(get, region, kind, name, path, sleep) for kind, path in STAGE1}


def _deep(get, region, name, sleep, first_js=None):
    """Status and shape for the auction-related paths of one namespace, plus the first connected realm's auction paths when there is one.
    Returns (entries, realm_index_json_or_None)."""
    out, realm_js = [], None
    for kind, path in STAGE2:
        status, js = _call(get, region, kind, name, path, sleep)
        out.append((f"{kind} {path}", status, describe(js) if status == 200 else ""))
        if kind == "dynamic" and path == "/data/wow/realm/index" and status == 200:
            realm_js = js
    realm_id = None
    if isinstance(first_js, dict):
        for r in first_js.get("connected_realms") or []:
            m = re.search(r"/connected-realm/(\d+)", str((r or {}).get("href", "")))
            if m:
                realm_id = m.group(1)
                break
    if realm_id:
        for tpl in REALM_PATHS:
            path = tpl.format(id=realm_id)
            status, js = _call(get, region, "dynamic", name, path, sleep)
            out.append((f"dynamic {tpl}", status, describe(js, depth=2) if status == 200 else ""))
    return out, realm_js


def _realm_list(js):
    """[(name, slug, id)] from a realm index answer."""
    out = []
    if isinstance(js, dict):
        for r in js.get("realms") or []:
            if isinstance(r, dict) and r.get("id") is not None:
                out.append((str(r.get("name", "")), str(r.get("slug", "")), r.get("id")))
    return out


def _dive(get, region, name, sleep, realms):
    """Follow the first few realms of a realm index: the realm itself, its connected realm, the auction-house index, the listing path, and the first auction
    house's listing when the index names one. This is the step that says whether any auction data is reachable at all."""
    out = []
    for rname, slug, rid in realms[:MAX_DIVE_REALMS]:
        status, js = _call(get, region, "dynamic", name, f"/data/wow/realm/{slug}", sleep)
        cid = rid
        if status == 200 and isinstance(js, dict):
            m = re.search(r"/connected-realm/(\d+)", str((js.get("connected_realm") or {}).get("href", "")))
            if m:
                cid = m.group(1)
        out.append((f"realm {rid}: /data/wow/realm/{{slug}}", status, describe(js, depth=2) if status == 200 else ""))
        ah_id = None
        for label, path in (("/data/wow/connected-realm/{id}", f"/data/wow/connected-realm/{cid}"),
                            ("/data/wow/connected-realm/{id}/auctions/index", f"/data/wow/connected-realm/{cid}/auctions/index"),
                            ("/data/wow/connected-realm/{id}/auctions", f"/data/wow/connected-realm/{cid}/auctions")):
            st, j = _call(get, region, "dynamic", name, path, sleep)
            out.append((f"realm {rid}: {label}", st, describe(j, depth=2) if st == 200 else ""))
            if label.endswith("auctions/index") and st == 200 and isinstance(j, dict):
                first = (j.get("auctions") or [None])[0]
                if isinstance(first, dict) and first.get("id") is not None:
                    ah_id = first["id"]
        if ah_id is not None:
            st, j = _call(get, region, "dynamic", name, f"/data/wow/connected-realm/{cid}/auctions/{ah_id}", sleep)
            out.append((f"realm {rid}: /data/wow/connected-realm/{{id}}/auctions/{{auction house id}}", st, describe(j, depth=2) if st == 200 else ""))
    return out


def _realm_map(get, region, name, sleep, realms, limit=80):
    """Read every realm record (up to `limit`) and group the realms by the connected realm each one names. In Classic a connected realm is what shares one auction
    house, so this shows how many separate markets the realm list really holds, before any auction data is served. Game data only: ids, ruleset, timezone."""
    ok, failed, groups, types = 0, 0, {}, {}
    for rname, slug, rid in realms[:limit]:
        st, js = _call(get, region, "dynamic", name, f"/data/wow/realm/{slug}", sleep)
        if st != 200 or not isinstance(js, dict):
            failed += 1
            continue
        ok += 1
        m = re.search(r"/connected-realm/(\d+)", str((js.get("connected_realm") or {}).get("href", "")))
        t = js.get("type")
        tname = str(t.get("name") or t.get("type") or "") if isinstance(t, dict) else str(t or "")
        g = groups.setdefault(m.group(1) if m else None, {"cid": m.group(1) if m else None, "realms": 0, "types": {}, "timezones": set(), "categories": set()})
        g["realms"] += 1
        g["types"][tname] = g["types"].get(tname, 0) + 1
        g["timezones"].add(str(js.get("timezone", "")))
        g["categories"].add(str(js.get("category", "")))
        types[tname] = types.get(tname, 0) + 1
    ordered = sorted(groups.values(), key=lambda g: (-g["realms"], str(g["cid"])))
    return {"read": ok, "failed": failed, "groups": ordered, "types": types}


def _auction_evidence(items):
    """(index_reachable, listings_reachable) from deep entries: a 200 whose shape holds an `auctions` list. The auction-house INDEX only names the houses;
    a listing is any other auctions path (commodities, a realm's auctions, one house's auctions)."""
    index = listings = False
    for label, status, shape in items:
        if status != 200 or "auctions: list[" not in shape:
            continue
        if label.endswith("auctions/index"):
            index = True
        elif "auctions" in label:
            listings = True
    return index, listings


def blizzard_probe(get, region="us", extra=(), sleep=time.sleep):
    """get(url) -> (status, json_or_None). Returns {controls, baseline, rows, flagged, deep, errors, token_ok}."""
    paths = [p for _, p in STAGE1]
    controls = {c or "retail": _stage1(get, region, c, sleep) for c in CONTROLS}
    baseline = _stage1(get, region, UNKNOWN, sleep)
    names = list(dict.fromkeys([str(x).strip().lower() for x in extra if str(x).strip()] + list(CANDIDATES)))
    rows, flagged, errors = {}, [], 0
    for name in names:
        res = _stage1(get, region, name, sleep)
        rows[name] = {p: res[p][0] for p in paths}
        errors += sum(1 for p in paths if res[p][0] == 0)
        if any(res[p][0] not in (0, baseline[p][0]) for p in paths):
            flagged.append(name)
    token_ok = controls["retail"][paths[1]][0] == 200
    deep, names_found, regions, verdicts, realm_maps, region_names = {}, {}, {}, {}, {}, {}
    if token_ok:
        for c in CONTROLS:
            key = c or "retail"
            deep[key], rjs = _deep(get, region, c, sleep, controls[key][paths[1]][1])
            if c in DIVE_CONTROLS:
                deep[key] += _dive(get, region, c, sleep, _realm_list(rjs))
        for name in flagged:
            res = _stage1(get, region, name, sleep)
            deep[name], rjs = _deep(get, region, name, sleep, res[paths[1]][1])
            realms = _realm_list(rjs)
            names_found[name] = realms
            deep[name] += _dive(get, region, name, sleep, realms)
            realm_maps[name] = _realm_map(get, region, name, sleep, realms)
            regions[name], region_names[name] = {}, {}
            for rg in OTHER_REGIONS:
                st, j = _call(get, rg, "dynamic", name, "/data/wow/realm/index", sleep)
                regions[name][rg] = (st, len(_realm_list(j)) if st == 200 else None)
                region_names[name][rg] = [n for n, _, _ in _realm_list(j)]
            index, listings = _auction_evidence(deep[name])
            verdicts[name] = {"realms": len(realms) if realms else None, "static_items": rows[name][paths[0]], "auction_index": index, "auction_listings": listings}
    return {"region": region, "paths": paths, "controls": {k: {p: v[p][0] for p in paths} for k, v in controls.items()},
            "baseline": {p: baseline[p][0] for p in paths}, "rows": rows, "flagged": flagged, "deep": deep, "errors": errors,
            "token_ok": token_ok, "tried": names, "names": names_found, "regions": regions, "verdicts": verdicts,
            "realm_maps": realm_maps, "region_names": region_names}


def format_blizzard(res):
    paths, region = res["paths"], res["region"]
    short = {paths[0]: "item classes", paths[1]: "connected realms"}
    out = [f"BLIZZARD NAMESPACE PROBE ({region})"]
    if not res["token_ok"]:
        out.append("The retail control did not answer 200, so the credentials or the network are the problem; nothing below means anything.")
    out.append("Known namespaces (these should all answer 200):")
    for k, v in res["controls"].items():
        out.append(f"  {k:<10} " + " | ".join(f"{short[p]} {v[p]}" for p in paths))
    out.append(f"A made-up namespace ('{UNKNOWN}') answers: " + " | ".join(f"{short[p]} {res['baseline'][p]}" for p in paths)
               + "   <- this is what 'Blizzard does not know that namespace' looks like")
    n = len(res["tried"])
    if res["errors"]:
        out.append(f"{res['errors']} requests failed outright (network); those guesses were not judged.")
    if res["flagged"]:
        out.append(f"\n{len(res['flagged'])} of {n} guesses answer DIFFERENTLY from the made-up one:")
        for name in res["flagged"]:
            out.append(f"  {name:<18} " + " | ".join(f"{short[p]} {res['rows'][name][p]}" for p in paths))
    else:
        out.append(f"\nNone of the {n} guesses answers differently from the made-up one: no Forever namespace among them yet.")
        out.append("Tried: " + ", ".join(res["tried"]))
    for name in res["flagged"]:
        v = res.get("verdicts", {}).get(name)
        if not v:
            continue
        rg = res.get("regions", {}).get(name, {})
        out.append(f"\nVERDICT for '{name}' (namespace dynamic-{name}-{region}):")
        out.append(f"  Blizzard knows this namespace: YES. Realm index: " + (f"{v['realms']} realms listed" if v["realms"] else "not listed") + ".")
        out.append(f"  Static data (items, item classes): " + ("answers" if v["static_items"] == 200 else f"not published yet (answers {v['static_items']})") + ".")
        out.append("  Auction-house index (names the houses): " + ("reachable" if v["auction_index"] else "not reachable") + ".")
        out.append("  Auction LISTINGS (prices and quantities): " + ("REACHABLE - see the shape below." if v["auction_listings"] else "not reachable at any path tried."))
        if rg:
            out.append("  Other regions: " + ", ".join(f"{r} " + (f"answers ({n} realms)" if st == 200 else f"{st}") for r, (st, n) in rg.items()) + ".")
        names = res.get("names", {}).get(name) or []
        if names:
            out.append(f"  Realms in dynamic-{name}-{region} ({len(names)}): " + ", ".join(f"{n} [{i}]" for n, _, i in names[:80]) + (" ..." if len(names) > 80 else ""))
        rm = res.get("realm_maps", {}).get(name)
        if rm and rm["read"]:
            gs = rm["groups"]
            out.append(f"  Realm map: {rm['read']} realm records read" + (f" ({rm['failed']} could not be read)" if rm["failed"] else "") + f", pointing to {len(gs)} different connected realms"
                       " (in Classic a connected realm is one shared auction house). Realm types: " + ", ".join(f"{t or '?'} {n}" for t, n in sorted(rm["types"].items(), key=lambda kv: -kv[1])) + ".")
            if all(g["realms"] == 1 for g in gs):
                out.append("    Every realm points to its own connected realm: no realms share an auction house in this list.")
            else:
                for g in gs[:25]:
                    out.append(f"    connected realm {g['cid'] or '(none named)'}: {g['realms']} realms; types " + ", ".join(f"{t or '?'} {n}" for t, n in sorted(g["types"].items(), key=lambda kv: -kv[1]))
                               + "; timezone " + "/".join(sorted(g["timezones"])) + "; category " + "/".join(sorted(g["categories"])))
                if len(gs) > 25:
                    out.append(f"    ... and {len(gs) - 25} more connected realms")
        for rg, rnames in (res.get("region_names", {}).get(name) or {}).items():
            if rnames:
                out.append(f"  Realms in dynamic-{name}-{rg} ({len(rnames)}): " + ", ".join(rnames[:60]) + (" ..." if len(rnames) > 60 else ""))
    for name, items in res["deep"].items():
        tag = "(known, for comparison)" if name in [c or "retail" for c in CONTROLS] else "(flagged)"
        out.append(f"\n-- {name} {tag} --")
        for path, status, shape in items:
            out.append(f"  {status}  {path}" + (f"\n       shape: {shape[:300]}" if shape else ""))
    return "\n".join(out)


# ---------------------------------------------------------------- AHLedger

def _openapi_endpoints(doc):
    """[(method, path, [param strings], summary)] from an OpenAPI/Swagger document; parameters are listed by name, location, required, allowed values."""
    eps = []
    for path, item in (doc.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        shared = item.get("parameters") or []
        for method, op in item.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch") or not isinstance(op, dict):
                continue
            params = []
            for p in list(shared) + list(op.get("parameters") or []):
                if not isinstance(p, dict):
                    continue
                enum = (p.get("schema") or {}).get("enum") or p.get("enum") or []
                txt = f"{p.get('name', '?')}({p.get('in', '?')}{', required' if p.get('required') else ''}"
                if enum:
                    txt += ", values: " + "/".join(str(e) for e in enum[:12]) + (f"/+{len(enum) - 12}" if len(enum) > 12 else "")
                params.append(txt + ")")
            eps.append((method.upper(), path, params, str(op.get("summary") or op.get("description") or "")[:100]))
    return eps


def _rank(path):
    p = path.lower()
    return -sum(1 for w in PREFER if w in p)


def _text_paths(text):
    found = re.findall(r"(?<![\w.])(/(?:v\d+|api)(?:/[A-Za-z0-9_\-{}.:]+)*)", text or "")
    return list(dict.fromkeys(found))[:80]


def ahledger_probe(fetch, base=AHL_BASE, max_requests=30, pause=1.0, sleep=time.sleep):
    """fetch(url) -> (status, content_type, body_text, headers). One request per `pause` seconds, at most `max_requests` in all."""
    import json
    used = {"n": 0}

    def get(path):
        if used["n"] >= max_requests:
            return None
        if used["n"]:
            sleep(pause)
        used["n"] += 1
        try:
            return fetch(base + path)
        except Exception:
            return 0, "", "", {}

    docs, endpoints, text_paths, auth = [], [], [], ""
    for path in DOC_PATHS:
        r = get(path)
        if r is None:
            break
        status, ctype, body, headers = r
        auth = auth or str((headers or {}).get("www-authenticate", (headers or {}).get("WWW-Authenticate", "")))
        kind, js = "", None
        if status == 200 and "json" in (ctype or "").lower():
            try:
                js = json.loads(body)
                kind = "json"
            except ValueError:
                kind = "bad json"
        elif status == 200:
            kind = "html" if "html" in (ctype or "").lower() else (ctype or "text").split(";")[0]
        docs.append({"path": path, "status": status, "type": (ctype or "").split(";")[0], "bytes": len(body or ""), "kind": kind,
                     "shape": describe(js, depth=2) if js is not None else ""})
        if isinstance(js, dict) and isinstance(js.get("paths"), dict):
            endpoints = _openapi_endpoints(js)
            break
        if status == 200 and kind != "json":
            for p in _text_paths(body):
                if p not in text_paths:
                    text_paths.append(p)
    candidates = [p for (m, p, params, _) in endpoints if m == "GET" and "{" not in p and not any("required" in x for x in params)]
    if not candidates:
        candidates = [p for p in text_paths if "{" not in p]
    probed = []
    for path in sorted(dict.fromkeys(candidates), key=_rank)[:10]:
        r = get(path)
        if r is None:
            break
        status, ctype, body, _ = r
        shape = ""
        if status == 200:
            try:
                shape = describe(json.loads(body), depth=3)
            except ValueError:
                shape = f"not JSON ({(ctype or '?').split(';')[0]}, {len(body or '')} bytes)"
        probed.append({"path": path, "status": status, "shape": shape})
    return {"requests": used["n"], "docs": docs, "endpoints": endpoints, "text_paths": text_paths, "probed": probed, "auth": auth}


def format_ahledger(res, base=AHL_BASE):
    out = [f"AHLEDGER API PROBE ({base})", f"{res['requests']} requests, about 1 per second."]
    out.append("\nWhere the API describes itself:")
    for d in res["docs"]:
        out.append(f"  {d['status']:>3}  {d['path']:<14} {d['type'] or '-'}  {d['bytes']:,} bytes" + (f"  {d['kind']}" if d["kind"] else ""))
        if d["shape"]:
            out.append(f"        shape: {d['shape'][:300]}")
    if res["auth"]:
        out.append(f"\nA server asked for authentication: {res['auth'][:200]}")
    if res["endpoints"]:
        out.append(f"\nENDPOINTS ({len(res['endpoints'])})")
        for m, p, params, summary in res["endpoints"][:80]:
            out.append(f"  {m:<6} {p}" + (f"   [{'; '.join(params)}]" if params else "") + (f"   - {summary}" if summary else ""))
    elif res["text_paths"]:
        out.append("\nNo OpenAPI file. Paths mentioned in the pages above:")
        out.extend(f"  {p}" for p in res["text_paths"][:60])
    else:
        out.append("\nNo OpenAPI file and no paths found in the pages above: the API's address or documentation must come from AHLedger.")
    if res["probed"]:
        out.append("\nENDPOINTS THAT NEED NO PARAMETERS (shape only, no values)")
        for pr in res["probed"]:
            out.append(f"  {pr['status']}  {pr['path']}" + (f"\n       shape: {pr['shape'][:400]}" if pr["shape"] else ""))
    out.append("\nTheir terms: credit AHLedger with a link; do not present their data as our own scan or resell it as a feed. "
               "Archiving is not addressed, so ask them before keeping history.")
    return "\n".join(out)
