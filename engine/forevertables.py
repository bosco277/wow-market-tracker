"""Forever tables: recipes, reagents, item master data and the house map. WoW Forever only: nothing here reads retail data or retail item tables.

Where the data comes from
  * Recipes: Forever's own static data (`static-forever-us`): profession index -> each profession -> its skill tiers -> each tier's recipes -> each recipe (reagents, the item it makes).
    Classic Era's static data does not serve professions at all (Nov 25 probe), so there is no fallback.
  * Item names and details: Forever's own static data first, then Classic Era (`static-classic1x-us`) for the ids Forever does not name (Forever names about 58% of the ids on the auction houses, Classic Era most of
    the rest). Retail and Classic named none of the remainder in the Nov 25 probe, so they are not asked. Every row records which source named it.
  * House map: written down by hand from evidence (price fingerprint against foreveritems' pages, a listing test, the API's own lists), each row saying how sure we are.

The item table only ever grows: a named id is never fetched again; an id nobody names is tried again after `retry_days` (Forever's static data may be filled in later); an id whose lookups got no answer at all
(network trouble, 429, 5xx) is NOT recorded as unnamed, so the next run tries it again.
Tables are stored as gzip CSV (small, readable by anything, and verifiable without extra libraries); integer columns keep their blanks (Int64) so ids never turn into 12345.0.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import gzip
import io
import json
import time
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import foreveritems as FI
from .groups import RateLimiter

NAMESPACE = "static-forever-us"
ITEM_SOURCES: Tuple[Tuple[str, str], ...] = FI.NAMESPACES[:2]          # forever, classic_era (retail and Classic are not asked: they named nothing)
API = "https://us.api.blizzard.com"
KEYS = {
    "items": "tables/forever_items.csv.gz",
    "recipes": "tables/forever_recipes.csv.gz",
    "reagents": "tables/forever_reagents.csv.gz",
    "house_map": "tables/forever_house_map.json",
}
PARTS = ("recipes", "items", "housemap")

RECIPE_COLUMNS = ["recipe_id", "recipe", "profession_id", "profession", "skill_tier_id", "skill_tier", "category", "crafted_item_id", "crafted_item", "crafted_qty_min", "crafted_qty_max", "n_reagents"]
REAGENT_COLUMNS = ["recipe_id", "reagent_item_id", "reagent", "quantity"]
ITEM_COLUMNS = ["item_id", "name", "item_class_id", "item_class", "item_subclass_id", "item_subclass", "quality", "item_level", "required_level", "inventory_type", "is_stackable", "max_count",
                "sell_price_copper", "purchase_price_copper", "source", "checked_at"]
INT_COLUMNS = {
    "recipes": ["recipe_id", "profession_id", "skill_tier_id", "crafted_item_id", "crafted_qty_min", "crafted_qty_max", "n_reagents"],
    "reagents": ["recipe_id", "reagent_item_id", "quantity"],
    "items": ["item_id", "item_class_id", "item_subclass_id", "item_level", "required_level", "max_count", "sell_price_copper", "purchase_price_copper"],
}
COLUMNS = {"recipes": RECIPE_COLUMNS, "reagents": REAGENT_COLUMNS, "items": ITEM_COLUMNS}


# ------------------------------------------------------------------------------------------------------------ stored tables
def _ints(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = df.copy()
    for c in INT_COLUMNS[kind]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    return out


def tidy(df: Optional[pd.DataFrame], kind: str) -> pd.DataFrame:
    """The table with exactly its columns, in order, integer columns nullable-int, text columns without NaN."""
    cols = COLUMNS[kind]
    if df is None or len(df) == 0:
        return _ints(pd.DataFrame({c: pd.Series(dtype="object") for c in cols}), kind)
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = _ints(out[cols], kind)
    for c in cols:
        if c not in INT_COLUMNS[kind] and c != "is_stackable":
            out[c] = out[c].astype(object).where(out[c].notna(), "").astype(str)
    if "is_stackable" in out.columns:
        out["is_stackable"] = out["is_stackable"].fillna(False).astype(bool)
    sort = {"items": ["item_id"], "recipes": ["profession_id", "skill_tier_id", "recipe_id"], "reagents": ["recipe_id", "reagent_item_id"]}[kind]
    return out.sort_values(sort, kind="mergesort").reset_index(drop=True)


def to_csv_gz(df: pd.DataFrame) -> bytes:
    """Same table -> same bytes (no timestamp in the gzip header), so an unchanged table is recognised as unchanged and not committed again."""
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as f:
        f.write(raw)
    return buf.getvalue()


def from_csv_gz(data: Optional[bytes], kind: str) -> pd.DataFrame:
    """A stored table back as a tidy frame; empty frame when there is nothing (or the bytes are not a table)."""
    if not data:
        return tidy(None, kind)
    try:
        df = pd.read_csv(io.BytesIO(data), compression="gzip", keep_default_na=False, na_values=[""], dtype={c: "object" for c in COLUMNS[kind] if c not in INT_COLUMNS[kind] and c != "is_stackable"})
    except Exception:
        return tidy(None, kind)
    if "is_stackable" in df.columns:
        df["is_stackable"] = df["is_stackable"].map(lambda v: str(v).strip().lower() in ("true", "1"))
    return tidy(df, kind)


# ------------------------------------------------------------------------------------------------------------ fetching
def api_url(path: str, namespace: str = NAMESPACE) -> str:
    return f"{API}{path}?namespace={namespace}&locale=en_US"


def fetch(get: Callable, path: str, limiter: Optional[RateLimiter] = None, tries: int = 3, sleep: Callable[[float], None] = time.sleep) -> Tuple[int, Optional[dict]]:
    """(status, json or None). 200 -> the json; 403 / 404 -> None at once (an answer); 429 / 5xx / network trouble are retried, then (last status or 0, None)."""
    code = 0
    for k in range(tries):
        if limiter:
            limiter.wait()
        try:
            code, js = get(api_url(path))
        except Exception:
            code, js = 0, None
        code = int(code)
        if code == 200 and isinstance(js, dict):
            return 200, js
        if code in FI.NOT_SERVED:
            return code, None
        sleep(min(2 ** k, 8) * 0.5)
    return code, None


def _id(x) -> Optional[int]:
    try:
        return int(x["id"]) if isinstance(x, dict) and x.get("id") is not None else None
    except (TypeError, ValueError):
        return None


def _name(x) -> str:
    n = x.get("name") if isinstance(x, dict) else None
    if isinstance(n, dict):
        n = n.get("en_US") or next(iter(n.values()), "")
    return str(n) if n else ""


def parse_tier(js: dict) -> List[Tuple[str, int, str]]:
    """[(category, recipe id, recipe name)] from a skill tier record: recipes sit under `categories` (the usual shape) or directly under `recipes`."""
    out: List[Tuple[str, int, str]] = []
    for cat in js.get("categories") or []:
        for r in (cat.get("recipes") or []) if isinstance(cat, dict) else []:
            if _id(r) is not None:
                out.append((_name(cat), _id(r), _name(r)))
    for r in js.get("recipes") or []:
        if _id(r) is not None:
            out.append(("", _id(r), _name(r)))
    return out


def parse_quantity(q) -> Tuple[Optional[int], Optional[int]]:
    """(min, max) crafted quantity from {minimum, maximum}, {value}, or a bare number; (None, None) when absent."""
    if isinstance(q, bool):
        return None, None
    if isinstance(q, (int, float)):
        return int(q), int(q)
    if isinstance(q, dict):
        lo = q.get("minimum", q.get("value"))
        hi = q.get("maximum", q.get("value"))
        lo = lo if lo is not None else hi
        hi = hi if hi is not None else lo
        try:
            return (int(lo), int(hi)) if lo is not None else (None, None)
        except (TypeError, ValueError):
            return None, None
    return None, None


def parse_recipe(js: dict) -> Tuple[dict, List[dict], int]:
    """(recipe fields, reagent rows, reagents without a usable item id). Every field is optional in the API, so each is read defensively."""
    rid = int(js["id"])
    ci = js.get("crafted_item") if isinstance(js.get("crafted_item"), dict) else {}
    lo, hi = parse_quantity(js.get("crafted_quantity"))
    reagents, bad = [], 0
    for r in js.get("reagents") or []:
        if not isinstance(r, dict):
            bad += 1
            continue
        item = r.get("reagent") if isinstance(r.get("reagent"), dict) else (r.get("item") if isinstance(r.get("item"), dict) else {})
        iid = _id(item)
        if iid is None:
            bad += 1
            continue
        q = r.get("quantity")
        reagents.append(dict(recipe_id=rid, reagent_item_id=iid, reagent=_name(item), quantity=int(q) if isinstance(q, (int, float)) and not isinstance(q, bool) else None))
    rec = dict(recipe_id=rid, recipe=_name(js), crafted_item_id=_id(ci), crafted_item=_name(ci), crafted_qty_min=lo, crafted_qty_max=hi, n_reagents=len(reagents))
    return rec, reagents, bad


@dataclass
class Crawl:
    professions: pd.DataFrame
    recipes: pd.DataFrame
    reagents: pd.DataFrame
    failed: List[Tuple[str, int, int]] = field(default_factory=list)       # (what, id, last status)
    keys: Counter = field(default_factory=Counter)                          # top-level keys of recipe responses
    samples: List[dict] = field(default_factory=list)
    duplicates: int = 0                                                     # recipe ids listed under more than one tier
    bad_reagents: int = 0
    notes: List[str] = field(default_factory=list)


def crawl_recipes(get: Callable, workers: int = 8, per_second: float = 20.0, tries: int = 3, sleep: Callable[[float], None] = time.sleep, sample: int = 2) -> Crawl:
    """Everything the profession routes of Forever's static data say about recipes. Raises RuntimeError when the profession index itself cannot be read (nothing to build on);
    a single profession, tier or recipe that cannot be read is listed in `failed` and the rest goes on."""
    limiter = RateLimiter(per_second)
    code, js = fetch(get, "/data/wow/profession/index", limiter, tries, sleep)
    profs = [(_id(p), _name(p)) for p in ((js or {}).get("professions") or []) if _id(p) is not None]
    if not profs:
        raise RuntimeError(f"profession index not served (status {code})")
    out = Crawl(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    listed: Dict[int, dict] = {}
    prof_rows = []
    for pid, pname in profs:
        code, pj = fetch(get, f"/data/wow/profession/{pid}", limiter, tries, sleep)
        if pj is None:
            out.failed.append(("profession", pid, code))
            continue
        tiers = [(_id(t), _name(t)) for t in (pj.get("skill_tiers") or []) if _id(t) is not None]
        ptype = _name(pj.get("type")) if isinstance(pj.get("type"), dict) else ""
        prof_rows.append(dict(profession_id=pid, profession=_name(pj) or pname, profession_type=ptype, skill_tiers=len(tiers)))
        for tid, tname in tiers:
            code, tj = fetch(get, f"/data/wow/profession/{pid}/skill-tier/{tid}", limiter, tries, sleep)
            if tj is None:
                out.failed.append(("skill tier", tid, code))
                continue
            for cat, rid, rname in parse_tier(tj):
                if rid in listed:
                    out.duplicates += 1
                    continue
                listed[rid] = dict(recipe_id=rid, recipe=rname, profession_id=pid, profession=_name(pj) or pname, skill_tier_id=tid, skill_tier=_name(tj) or tname, category=cat)
    out.professions = pd.DataFrame(prof_rows)

    def one(rid: int):
        return rid, fetch(get, f"/data/wow/recipe/{rid}", limiter, tries, sleep)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, sorted(listed)))
    rec_rows, reag_rows = [], []
    for rid, (code, rj) in results:
        base = listed[rid]
        if rj is None:
            out.failed.append(("recipe", rid, code))
            continue
        out.keys.update(rj.keys())
        if len(out.samples) < sample and rj.get("reagents"):
            out.samples.append(rj)
        try:
            rec, reagents, bad = parse_recipe(rj)
        except (KeyError, TypeError, ValueError):
            out.failed.append(("recipe (unreadable)", rid, code))
            continue
        out.bad_reagents += bad
        base = {**base, **{k: v for k, v in rec.items() if k != "recipe" or v}}
        rec_rows.append(base)
        reag_rows.extend(reagents)
    out.recipes = tidy(pd.DataFrame(rec_rows), "recipes")
    out.reagents = tidy(pd.DataFrame(reag_rows), "reagents")
    return out


def keep_previous_for_failures(crawl: Crawl, prev_recipes: pd.DataFrame, prev_reagents: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Recipes whose lookup failed this time keep their rows from the stored tables (when there are any). Returns (recipes, reagents, how many kept)."""
    ids = {rid for what, rid, _ in crawl.failed if what.startswith("recipe")}
    if not ids or len(prev_recipes) == 0:
        return crawl.recipes, crawl.reagents, 0
    have = ids & set(prev_recipes["recipe_id"].dropna().astype(int))
    if not have:
        return crawl.recipes, crawl.reagents, 0
    rec = pd.concat([crawl.recipes, prev_recipes[prev_recipes["recipe_id"].isin(have)]], ignore_index=True)
    rea = pd.concat([crawl.reagents, prev_reagents[prev_reagents["recipe_id"].isin(have)]], ignore_index=True)
    return tidy(rec, "recipes"), tidy(rea, "reagents"), len(have)


def recipe_item_ids(recipes: pd.DataFrame, reagents: pd.DataFrame) -> set:
    """Every item id a recipe mentions: what it makes and what it uses."""
    ids = set()
    if len(recipes):
        ids |= set(recipes["crafted_item_id"].dropna().astype(int))
    if len(reagents):
        ids |= set(reagents["reagent_item_id"].dropna().astype(int))
    return ids


def recipe_summary(crawl_recipes_: pd.DataFrame, reagents: pd.DataFrame) -> str:
    """Plain text: recipes per profession, how many list reagents, how many name what they make."""
    r = crawl_recipes_
    if len(r) == 0:
        return "No recipes."
    have_reag = set(reagents["recipe_id"].dropna().astype(int)) if len(reagents) else set()
    rows = []
    for (pid, pname), g in r.groupby(["profession_id", "profession"], sort=True):
        ids = set(g["recipe_id"].astype(int))
        rows.append((pname, int(pid), len(g), len(ids & have_reag), int(g["crafted_item_id"].notna().sum())))
    head = f"  {'profession':<22}{'id':>5}{'recipes':>9}{'with reagents':>15}{'name their item':>17}"
    lines = [head] + [f"  {n:<22}{p:>5}{k:>9,}{a:>15,}{b:>17,}" for n, p, k, a, b in rows]
    lines.append(f"  {'all':<22}{'':>5}{len(r):>9,}{len(set(r['recipe_id'].astype(int)) & have_reag):>15,}{int(r['crafted_item_id'].notna().sum()):>17,}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------ item master
def update_items(ids: Iterable[int], previous: Optional[pd.DataFrame], get: Callable, today: date, retry_days: int = 3, workers: int = 8, per_second: float = 20.0,
                 sleep: Callable[[float], None] = time.sleep) -> Tuple[pd.DataFrame, dict]:
    """(new item table, stats). Only ids that are new, or were unnamed `retry_days` or more ago, are asked; named ids are never fetched again."""
    prev = tidy(previous, "items")
    want = sorted({int(i) for i in ids})
    named = set(prev.loc[prev["source"] != "none", "item_id"].astype(int))
    cutoff = (today - timedelta(days=retry_days)).isoformat()
    seen = prev.set_index("item_id")
    stale = {int(i) for i in prev.loc[(prev["source"] == "none") & (prev["checked_at"] <= cutoff), "item_id"]}
    known = {int(i) for i in seen.index}
    todo = [i for i in want if i not in named and (i not in known or i in stale)]
    stats = dict(wanted=len(want), already_named=len([i for i in want if i in named]), asked=len(todo), named_by={}, unnamed=0, no_answer=0, kept_unnamed=len([i for i in want if i in known and i not in named and i not in stale]))
    if not todo:
        return prev, stats
    items, att = FI.resolve_items(todo, get, namespaces=ITEM_SOURCES, workers=workers, per_second=per_second, sleep=sleep)
    stamp = today.isoformat()
    fresh = items.copy() if len(items) else pd.DataFrame(columns=["item_id"])
    if len(fresh):
        fresh["checked_at"] = stamp
        stats["named_by"] = {str(k): int(v) for k, v in fresh["source"].value_counts().items()}
    named_now = set(fresh["item_id"].astype(int)) if len(fresh) else set()
    labels = [l for l, _ in ITEM_SOURCES]
    misses = []
    for _, row in att.iterrows():
        i = int(row["item_id"])
        if i in named_now:
            continue
        statuses = [row.get(l) for l in labels]
        if all(pd.notna(s) and int(s) in FI.NOT_SERVED for s in statuses):
            misses.append(dict(item_id=i, name="", source="none", checked_at=stamp, is_stackable=False))
        else:
            stats["no_answer"] += 1                     # not recorded: a source gave no real answer, so the next run asks again
    stats["unnamed"] = len(misses)
    replaced = set(todo)
    old = prev[~prev["item_id"].astype(int).isin(replaced)]
    new = pd.concat([old, tidy(fresh, "items") if len(fresh) else None, pd.DataFrame(misses) if misses else None], ignore_index=True)
    # an id asked this time but not answered keeps its old row, if it had one
    lost = [i for i in todo if i not in named_now and i not in {m["item_id"] for m in misses} and i in known]
    if lost:
        new = pd.concat([new, prev[prev["item_id"].astype(int).isin(lost)]], ignore_index=True)
    return tidy(new, "items"), stats


# ------------------------------------------------------------------------------------------------------------ house map
# Evidence behind each row (Nov 25, 2026):
#   structure run   the API lists 15 houses (110 to 125 without 122); 112, 115 and 118 hold no auctions; the `ruleset` field of a house record just repeats the ruleset asked for, so it says nothing about the house.
#   price match     foreverwowcraft.com's eight US pages priced against every house's commodity book over 40 hours: each page lined up with exactly one house (scores 0.00 to 0.02; the PvE Alliance match
#                   is "unclear" only because the Horde PvE house scores almost as well, but that house is taken by the Horde PvE page).
#   listing test    house 110 is the player's Alliance house on the PvP ruleset (a Linen Cloth listing on Nov 13).
#   neutral houses  not on the site; matched to a ruleset only by slot order (116, 117, 121, 125 fill the same slots as the faction houses) and by size. NOT verified.
RULESET_INFO = {
    "pvp": dict(ruleset_id=1, id_basis="API: ruleset 1 = PvP"),
    "pve": dict(ruleset_id=2, id_basis="API: ruleset 2 = Normal"),
    "rp": dict(ruleset_id=3, id_basis="community post (Nov 12); not read from the API"),
    "rp pvp": dict(ruleset_id=15, id_basis="community post (Nov 12); not read from the API"),
}
SLOTS: Tuple[Tuple[str, int, int, int], ...] = (("pvp", 110, 113, 116), ("pve", 111, 114, 117), ("rp", 119, 120, 121), ("rp pvp", 123, 124, 125))
EMPTY_HOUSES = (112, 115, 118)
MATCH_SCORE = {110: 0.00, 111: 0.02, 113: 0.00, 114: 0.02, 119: 0.02, 120: 0.00, 123: 0.01, 124: 0.01}


AS_OF = "2026-11-25"          # the date of the evidence, NOT the run date: the stored bytes only change when the map itself changes


def house_map() -> dict:
    houses: Dict[str, dict] = {}
    for ruleset, a, h, n in SLOTS:
        info = RULESET_INFO[ruleset]
        for house, faction in ((a, "alliance"), (h, "horde"), (n, "neutral")):
            row = dict(region="us", ruleset=ruleset, **info, faction=faction)
            if faction == "neutral":
                row.update(confidence="assumed", basis="slot order and size only (neutral houses are not on foreverwowcraft.com); not verified")
            elif house == 110:
                row.update(confidence="confirmed", basis="listing test (Nov 13) and price match (Nov 25, score 0.00)")
            elif house == 111:
                row.update(confidence="likely", basis=f"price match (Nov 25, score {MATCH_SCORE[house]:.2f}); the runner-up is the Horde PvE house, which another page claims")
            else:
                row.update(confidence="matched", basis=f"price match (Nov 25, score {MATCH_SCORE[house]:.2f}), clear against the next house")
            houses[str(house)] = row
    for house in EMPTY_HOUSES:
        houses[str(house)] = dict(region="us", faction={112: "alliance", 115: "horde", 118: "neutral"}[house], ruleset=None, confidence="empty", basis="listed by the API, zero auctions on Nov 25; has never returned data")
    return dict(
        as_of=AS_OF,
        notes=["House 122 is not listed by the API. The `ruleset` field in a house record repeats the ruleset in the URL and says nothing about the house.",
               "Neutral rows (Blackwater) are the weakest: confidence 'assumed' means a guess from slot order and size until a listing test confirms it."],
        houses=dict(sorted(houses.items(), key=lambda kv: int(kv[0]))),
    )


def house_map_bytes() -> bytes:
    return (json.dumps(house_map(), indent=1, sort_keys=False) + "\n").encode("utf-8")
