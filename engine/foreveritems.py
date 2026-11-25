"""Forever item master data, step 1: which item ids can Blizzard's static data name, and where does it fall short?

Forever's own static data (`static-forever-us`) has gaps: on Nov 6 a community developer found 43% of the item ids in one EU house returned 404 there. Retail's item table (engine/groups.py) cannot fill them blindly:
retail and Forever are different games with different item sets. So every id is tried in an ORDER, and the row records which source named it:

    forever      static-forever-us   the game itself (preferred)
    classic_era  static-classic1x-us Classic Era: most Forever items look like vanilla items (every item on foreverwowcraft.com's alchemy pages is a vanilla item)
    classic      static-classic-us   Classic progression
    retail       static-us           last resort; the same item id can mean a different item here

`resolve_items` does the lookups (paced, threaded, retried), `coverage_text` says how well it went, ranked by what the missing items are worth, so the question "do the gaps matter?" gets a number.
Nothing here writes anywhere; the items TABLE (stored daily, only new ids fetched) is the next step and builds on this.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .groups import RateLimiter, parse_item

NAMESPACES: Tuple[Tuple[str, str], ...] = (
    ("forever", "static-forever-us"),
    ("classic_era", "static-classic1x-us"),
    ("classic", "static-classic-us"),
    ("retail", "static-us"),
)
ITEM_URL = "https://us.api.blizzard.com/data/wow/item/{id}?namespace={ns}&locale=en_US"
NOT_SERVED = (403, 404)          # "this namespace does not know the item (or the namespace)": an answer, not an error


def item_url(item_id: int, namespace: str) -> str:
    return ITEM_URL.format(id=int(item_id), ns=namespace)


def _one(item_id: int, namespace: str, get: Callable, limiter: Optional[RateLimiter], tries: int, sleep: Callable[[float], None]) -> Tuple[Optional[dict], int]:
    """(json or None, last status). 200 -> json; 403 / 404 -> None at once; 429 / 5xx / network trouble are retried, then None with the last status (0 = no answer)."""
    code = 0
    for k in range(tries):
        if limiter:
            limiter.wait()
        try:
            code, js = get(item_url(item_id, namespace))
        except Exception:
            code, js = 0, None
        code = int(code)
        if code == 200 and isinstance(js, dict) and "id" in js:
            return js, 200
        if code in NOT_SERVED:
            return None, code
        sleep(min(2 ** k, 8) * 0.5)
    return None, code


def resolve_one(item_id: int, get: Callable, namespaces: Sequence[Tuple[str, str]] = NAMESPACES, limiter: Optional[RateLimiter] = None, tries: int = 3,
                sleep: Callable[[float], None] = time.sleep) -> Tuple[Optional[dict], Dict[str, int]]:
    """(item row or None, {source label: status for each source tried}). The first source that names the item wins and later ones are not asked."""
    tried: Dict[str, int] = {}
    for label, ns in namespaces:
        js, code = _one(item_id, ns, get, limiter, tries, sleep)
        tried[label] = code
        if js is not None:
            row = parse_item(js)
            row["item_id"] = int(item_id)
            row["source"] = label
            return row, tried
    return None, tried


def resolve_items(item_ids: Iterable[int], get: Callable, namespaces: Sequence[Tuple[str, str]] = NAMESPACES, workers: int = 8, per_second: float = 20.0,
                  tries: int = 3, sleep: Callable[[float], None] = time.sleep) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(items, attempts). items: one row per NAMED id (parse_item's columns plus `source`). attempts: one row per id asked, with the status of each source tried
    (blank = not asked because an earlier source answered), so the report can show what each source knew."""
    ids = sorted({int(i) for i in item_ids})
    limiter = RateLimiter(per_second)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda i: resolve_one(i, get, namespaces, limiter, tries, sleep), ids))
    rows = [r for r, _ in res if r]
    labels = [l for l, _ in namespaces]
    att = pd.DataFrame([{"item_id": i, **{l: t.get(l) for l in labels}} for i, (_, t) in zip(ids, res)], columns=["item_id"] + labels)
    items = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["item_id", "name", "source"])
    return items, att


def coverage_text(ids: Sequence[int], items: pd.DataFrame, attempts: pd.DataFrame, value: Optional[pd.Series] = None, labels: Sequence[str] = tuple(l for l, _ in NAMESPACES), top: int = 15,
                  names_hint: Optional[Dict[int, str]] = None) -> str:
    """Plain-text coverage: how many ids each source named (first source that answered), how many nobody named, and the biggest misses by value.
    value: Series indexed by item id (gold in play: units x price, summed over what we saw)."""
    n = len(set(int(i) for i in ids))
    if n == 0:
        return "No item ids to check."
    named = items.set_index("item_id")["source"] if len(items) else pd.Series(dtype=object)
    lines = [f"{n:,} distinct item ids seen in the Forever lake; {len(named):,} named ({len(named) / n:.1%}); {n - len(named):,} not named by any source."]
    for lab in labels:
        k = int((named == lab).sum())
        asked = int(attempts[lab].notna().sum()) if lab in attempts else 0
        ok = int((attempts[lab] == 200).sum()) if lab in attempts else 0
        miss = int(attempts[lab].isin(NOT_SERVED).sum()) if lab in attempts else 0
        other = asked - ok - miss
        lines.append(f"  {lab:<12} asked {asked:>6,}   named {k:>6,}   404/403 {miss:>6,}" + (f"   no answer {other:,}" if other else ""))
    missing = sorted(set(int(i) for i in ids) - set(int(i) for i in named.index))
    if missing:
        v = (value.reindex(missing).fillna(0.0) if value is not None else pd.Series(0.0, index=missing)).sort_values(ascending=False)
        tot = float(value.sum()) if value is not None and len(value) else 0.0
        miss_val = float(v.sum())
        if tot > 0:
            lines.append(f"Unnamed ids hold {miss_val / tot:.1%} of the gold in play across the houses (units x price at the newest snapshot).")
        lines.append("Biggest unnamed ids (item id, gold in play):")
        for i, g in v.head(top).items():
            lines.append(f"  {int(i):>7}  {g:>14,.0f}g" + (f"   {names_hint.get(int(i), '')}" if names_hint else ""))
    return "\n".join(lines)
