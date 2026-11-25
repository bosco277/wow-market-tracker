"""Forever tables: recipes, reagents, item names and the house map, built from Blizzard's Forever data and saved in the Forever dataset (WoW Forever only; retail is not touched).

    python tools/forever_tables.py                     # build everything and REPORT; saves nothing
    python tools/forever_tables.py --save              # the same, then store the tables that changed
    python tools/forever_tables.py --parts items       # one of: recipes, items, housemap (comma separated)

  recipes    every profession of Forever's static data -> its skill tiers -> their recipes -> each recipe (what it uses, what it makes). Forever only: Classic Era does not serve professions.
  items      a name and details for every item id on the auction houses (newest snapshot of every house) and in a recipe (reagents and products). Forever's own data first, Classic Era for the ids Forever does
             not name; each row says which. Only new ids are asked on later runs; ids nobody names are retried after --retry-days.
  housemap   the house -> faction / ruleset table, with how sure we are of each row (written from evidence, not fetched).

Stored in the dataset (gzip CSV; a table is only written when its content changed):
    tables/forever_recipes.csv.gz  tables/forever_reagents.csv.gz  tables/forever_items.csv.gz  tables/forever_house_map.json

Writes ONE plain-text report, out/forever_tables.txt (a .txt on purpose).
Environment: BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET; HF_TOKEN (write access to the Forever dataset to save, read access otherwise); FOREVER_HF_REPO to name another dataset.
The dataset's name is blanked out of everything printed, so no account name reaches a job log.
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from engine import forever as FV  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.loader import HFStore  # noqa: E402
import forever_probe as FP  # noqa: E402

COMMIT_TRIES = 5


# ------------------------------------------------------------------------------------------------------------ the dataset
def read_key(store, key):
    """The stored file's bytes, or None when it does not exist (or the dataset cannot be listed)."""
    prefix = key.rsplit("/", 1)[0] + "/"
    entry = next((o for o in store.list(prefix) if o.key == key), None)
    if entry is None or entry.size <= 0:
        return None
    return store.read_range(key, 0, entry.size)


def hf_commit(files, message, sleep=time.sleep):
    """Writes several files in one commit; transient Hugging Face trouble is retried (15s, 30s, 60s, 120s), a wrong token or missing permission is raised at once."""
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for i in range(COMMIT_TRIES):
        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=io.BytesIO(v)) for k, v in files.items()]
        try:
            api.create_commit(repo_id=FV.hf_repo(), repo_type="dataset", operations=ops, commit_message=message)
            return
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if i == COMMIT_TRIES - 1 or not (code is None or code >= 500 or code == 429):
                raise
            sleep(15 * 2 ** i)


# ------------------------------------------------------------------------------------------------------------ report pieces
def kb(n):
    return f"{n / 1024:,.0f} KB" if n >= 1024 else f"{n} B"


def sample_text(js, limit=700):
    return json.dumps(js, separators=(",", ":"), ensure_ascii=False)[:limit]


def readable_recipes(recipes, reagents, items, per_profession=1, top=6):
    """A few recipes as sentences ('2 x Linen Cloth -> Linen Bandage'), one per profession first, so the data can be eyeballed."""
    if len(recipes) == 0 or len(reagents) == 0:
        return []
    names = dict(zip(items["item_id"].astype(int), items["name"])) if len(items) else {}
    named = {i for i, n in names.items() if n}
    by = {rid: g for rid, g in reagents.groupby("recipe_id")}
    out, seen = [], {}
    for r in recipes.itertuples():
        if pd.isna(r.crafted_item_id) or int(r.crafted_item_id) not in named or r.recipe_id not in by:
            continue
        g = by[r.recipe_id]
        if not all(int(i) in named for i in g["reagent_item_id"].dropna()):
            continue
        if seen.get(r.profession_id, 0) >= per_profession:
            continue
        seen[r.profession_id] = seen.get(r.profession_id, 0) + 1
        q = f"{int(r.crafted_qty_min)}" if pd.notna(r.crafted_qty_min) else "?"
        uses = " + ".join(f"{int(x.quantity) if pd.notna(x.quantity) else '?'} x {names[int(x.reagent_item_id)]}" for x in g.itertuples())
        out.append(f"  {r.profession}: {r.recipe}: {uses}  ->  {q} x {names[int(r.crafted_item_id)]}")
        if len(out) >= top:
            break
    return out


def usable_count(recipes, reagents, items):
    """Recipes that list reagents, name what they make, and have every item named: the ones a crafting tool can price today."""
    if len(recipes) == 0:
        return 0, 0
    named = set(items.loc[items["name"].astype(str) != "", "item_id"].astype(int)) if len(items) else set()
    bad_reag = set(reagents.loc[~reagents["reagent_item_id"].isin(named), "recipe_id"].astype(int)) if len(reagents) else set()
    with_reag = set(reagents["recipe_id"].astype(int)) if len(reagents) else set()
    ok = recipes[recipes["crafted_item_id"].notna() & recipes["crafted_item_id"].isin(named) & recipes["recipe_id"].isin(with_reag) & ~recipes["recipe_id"].isin(bad_reag)]
    return len(ok), len(recipes)


# ------------------------------------------------------------------------------------------------------------ run
def run(args, get, store, now=None, base_reader=None, sleep=time.sleep, commit=hf_commit):
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    today = now.date()
    want = T.PARTS if args.parts == "all" else tuple(s.strip() for s in args.parts.split(",") if s.strip())
    bad = [s for s in want if s not in T.PARTS]
    if bad:
        return dict(text=f"### forever tables\nUnknown part(s): {bad}. Use {T.PARTS}.", code=1)
    out = ["### forever tables", f"run at {pd.Timestamp(now).strftime('%Y-%m-%d %H:%M')} UTC; parts: {', '.join(want)}; {'SAVE' if args.save else 'report only (nothing is saved)'}"]
    code = 0
    files, info = {}, {}
    stored = {k: read_key(store, T.KEYS[k]) for k in ("recipes", "reagents", "items", "house_map")}
    prev = {k: T.from_csv_gz(stored[k], k) for k in ("recipes", "reagents", "items")}
    if any(stored[k] for k in stored):
        out.append("Stored before this run: " + ", ".join(f"{k} {len(prev[k]):,} rows" for k in prev if stored[k]) + (", house map" if stored["house_map"] else "") + ".")
    else:
        out.append("Nothing stored yet: this is the first build.")

    recipes, reagents = prev["recipes"], prev["reagents"]
    crawl = None
    if "recipes" in want:
        out += ["", "### 1 recipes (Forever's own data)"]
        try:
            crawl = T.crawl_recipes(get, sleep=sleep)
        except Exception as e:
            out.append(f"recipes: failed ({FV.scrub(e)})")
            code = 1
        if crawl is not None:
            recipes, reagents, kept = T.keep_previous_for_failures(crawl, prev["recipes"], prev["reagents"])
            out.append(f"{len(crawl.professions)} professions, {len(crawl.recipes):,} recipes read, {len(crawl.reagents):,} reagent lines.")
            out.append(T.recipe_summary(recipes, reagents))
            if len(crawl.professions):
                out.append("Professions: " + ", ".join(f"{r.profession} ({r.skill_tiers} tier{'s' if r.skill_tiers != 1 else ''})" for r in crawl.professions.itertuples()))
            out.append(f"Recipe keys in the answers: " + ", ".join(f"{k} {v:,}" for k, v in sorted(crawl.keys.items(), key=lambda kv: (-kv[1], kv[0]))))
            if crawl.duplicates:
                out.append(f"{crawl.duplicates:,} recipe id(s) were listed under more than one tier; the first listing was kept.")
            if crawl.bad_reagents:
                out.append(f"{crawl.bad_reagents:,} reagent line(s) had no item id and were skipped.")
            no_product = int(recipes["crafted_item_id"].isna().sum())
            out.append(f"{no_product:,} recipe(s) name no item they make (usual for enchants and similar).")
            if crawl.failed:
                out.append(f"{len(crawl.failed)} lookup(s) failed: " + ", ".join(f"{w} {i} (status {c})" for w, i, c in crawl.failed[:12]) + (" ..." if len(crawl.failed) > 12 else "")
                           + (f"; {kept} recipe(s) kept their stored rows." if kept else "."))
            for s in crawl.samples:
                out.append("Sample recipe record: " + sample_text(s))
            limit = max(3, int(0.01 * (len(crawl.recipes) + len(crawl.failed))))
            if len(crawl.failed) > limit:
                info["recipes"] = f"NOT saved: {len(crawl.failed)} lookups failed (more than {limit} allowed); run again"
            elif len(prev["recipes"]) and len(recipes) < 0.8 * len(prev["recipes"]):
                info["recipes"] = f"NOT saved: {len(recipes):,} recipes is far fewer than the {len(prev['recipes']):,} stored; check before overwriting"
            elif len(recipes) == 0:
                info["recipes"] = "NOT saved: no recipes read"
            else:
                files[T.KEYS["recipes"]] = T.to_csv_gz(recipes)
                files[T.KEYS["reagents"]] = T.to_csv_gz(reagents)
            if len(prev["recipes"]):
                have, now_ids = set(prev["recipes"]["recipe_id"].astype(int)), set(recipes["recipe_id"].astype(int))
                out.append(f"Against the stored table: {len(now_ids - have):,} new recipe(s), {len(have - now_ids):,} no longer listed.")
    items = prev["items"]
    if "items" in want:
        out += ["", "### 2 items (Forever first, Classic Era for the rest)"]
        try:
            ids, value = set(), pd.Series(dtype="float64")
            houses = FV.houses_in_store(store)
            if houses:
                ah_ids, value, _, notes = FP.newest_summary(store, houses, now, base_reader)
                out += notes
                ids |= set(ah_ids)
                out.append(f"{len(ah_ids):,} item ids on the auction houses (newest snapshot of each of {len(houses)} houses).")
            else:
                out.append("The lake could not be read (no sources file): no auction-house ids this run.")
            rids = T.recipe_item_ids(recipes, reagents)
            out.append(f"{len(rids):,} item ids used or made by recipes ({len(rids - ids):,} of them not on the auction houses).")
            ids |= rids
            if not ids:
                raise RuntimeError("no item ids to look up")
            if len(ids) > args.max_ids:
                raise RuntimeError(f"{len(ids):,} ids is more than --max-ids {args.max_ids:,}")
            items, st = T.update_items(ids, prev["items"], get, today, retry_days=args.retry_days, sleep=sleep)
            out.append(f"Asked {st['asked']:,} id(s) ({st['already_named']:,} already named, {st['kept_unnamed']:,} unnamed and not due for another try).")
            if st["asked"]:
                out.append("Named by: " + (", ".join(f"{k} {v:,}" for k, v in st["named_by"].items()) or "none") + f"; nobody names {st['unnamed']:,}" + (f"; {st['no_answer']:,} got no answer and will be asked again" if st["no_answer"] else "") + ".")
            wanted = items[items["item_id"].astype(int).isin(ids)]
            named_all = wanted[wanted["source"] != "none"]
            out.append(f"Now {len(named_all):,} of the {len(ids):,} ids are named ({len(named_all) / len(ids):.1%}); by source: " + ", ".join(f"{k} {v:,}" for k, v in named_all["source"].value_counts().items()) + ".")
            missing = sorted(ids - set(named_all["item_id"].astype(int)))
            if missing:
                v = value.reindex(missing).fillna(0.0).sort_values(ascending=False) if len(value) else pd.Series(0.0, index=missing)
                share = (float(v.sum()) / float(value.sum())) if len(value) and float(value.sum()) > 0 else 0.0
                in_recipes = [i for i in missing if i in rids]
                out.append(f"Unnamed: {len(missing):,} ids holding {share:.1%} of the gold in play; in recipes: {len(in_recipes):,}" + (f" (e.g. {', '.join(map(str, in_recipes[:8]))})" if in_recipes else "") + ".")
                out.append("Biggest unnamed ids (item id, gold in play): " + ", ".join(f"{int(i)} ({g:,.0f}g)" for i, g in v.head(8).items()))
            files[T.KEYS["items"]] = T.to_csv_gz(items)
        except Exception as e:
            out.append(f"items: failed ({FV.scrub(e)})")
            code = 1
    if "recipes" in want or "items" in want:
        ok, total = usable_count(recipes, reagents, items)
        out += ["", f"Recipes a crafting tool can price today (reagents listed, product named, every item named): {ok:,} of {total:,}."] + readable_recipes(recipes, reagents, items)
    if "housemap" in want:
        out += ["", "### 3 house map", "15 houses: " + ", ".join(f"{h} {r['faction'][:1].upper()}{r['faction'][1:]} {r.get('ruleset') or 'empty'} ({r['confidence']})" for h, r in T.house_map()["houses"].items())]
        files[T.KEYS["house_map"]] = T.house_map_bytes()

    out += ["", "### Storing"]
    changed = {}
    for k, v in files.items():
        old = stored[next(n for n, key in T.KEYS.items() if key == k)]
        state = "new" if old is None else ("unchanged" if old == v else "changed")
        out.append(f"  {k}: {kb(len(v))}, {state}")
        if state != "unchanged":
            changed[k] = v
    for k, why in info.items():
        out.append(f"  {k}: {why}")
    if not files:
        out.append("  nothing to store")
    elif not changed:
        out.append("Every table is identical to what is stored: nothing to save.")
    elif not args.save:
        out.append("NOT saved (report only). Run again with save = yes to store the tables marked new or changed.")
    else:
        try:
            commit(changed, f"Forever tables: {', '.join(sorted(os.path.basename(k).split('.')[0] for k in changed))}")
            out.append(f"Saved {len(changed)} file(s) to the Forever dataset in one commit.")
        except Exception as e:
            out.append(f"SAVE FAILED: {FV.scrub(e)}")
            code = 1
    return dict(text="\n".join(out), code=code, files=files, changed=changed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="all", help="all, or a comma separated list of: recipes, items, housemap")
    ap.add_argument("--save", action="store_true", help="store the tables that are new or changed (default: report only)")
    ap.add_argument("--retry-days", type=int, default=3, dest="retry_days", help="ask again about an unnamed id after this many days (default 3)")
    ap.add_argument("--max-ids", type=int, default=30000, dest="max_ids", help="stop if more item ids than this are wanted (default 30000)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    cid, sec, tok = os.environ.get("BLIZZARD_CLIENT_ID"), os.environ.get("BLIZZARD_CLIENT_SECRET"), os.environ.get("HF_TOKEN")
    if not (cid and sec and tok):
        print("forever_tables needs BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET and HF_TOKEN.")
        return 1
    try:
        r = run(args, FP.Blizzard(cid, sec).get, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except Exception as e:
        print(f"forever_tables failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_tables.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
