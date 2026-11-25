"""House fingerprint: which of OUR auction houses is each page of foreverwowcraft.com about?

Why: Blizzard's API gives house numbers (110 ... 125) but not which ruleset or faction each belongs to, and foreverwowcraft.com gives ruleset and faction (US PvE / PvP / RP / RP PvP, Alliance / Horde)
but no house numbers. Both describe the same auction houses, so the same item bought in the same quantity should cost about the same in both. For each page the fixture (tools/fwc_fingerprint.json) holds about
15 items with a quantity and the average price per unit of buying that quantity from the site's own scan. Here the same quantity is priced from each of our houses' books, hour by hour, and the house
whose prices line up with the page is the page's house.

What a score is: for one page and one house at one snapshot, the MEDIAN over items of |ln(our price / the page's price)|. 0.05 means the typical item is within about 5%; 0.7 means a factor of two.
Each page/house pair gets its best score over the snapshots searched (the pages say 'scanned N hours ago' relative to when they were built, and they are rebuilt now and then, so the right hour is not known
exactly: every snapshot in the window is tried, for every house alike).

What this can and cannot say: it works only if the page's scan is recent enough to sit inside the window, and only if the price LEVELS of the houses differ (they do: the health check found 1.0x to 4x).
A house that matches clearly is a strong hint, not proof; a page with no clear match is reported as unclear. It says nothing about neutral houses (the site does not record them).
Prices are average prices for the quantity (walking up the book from the cheapest listing), never the floor, because that is what the page shows.
"""
from __future__ import annotations

import itertools
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import Snapshot

# The community post's table of houses (Nov 12; unverified except where KNOWN says otherwise): five of each, the same ids in every ruleset's index.
FACTION_HOUSES = {"alliance": (110, 111, 112, 119, 123), "horde": (113, 114, 115, 120, 124), "neutral": (116, 117, 118, 121, 125)}
KNOWN = {"uspvp--alliance": 110}        # confirmed by the Nov 13 listing test (a Linen Cloth listing on the player's Alliance house of the US PvP ruleset showed up in house 110 only)
MIN_ITEMS = 6                           # a pair is scored only when at least this many items can be priced from our book
CLEAR_SCORE = 0.35                      # a best score above this is not a clear match, however far ahead of the others
CLEAR_RATIO = 1.5                       # the runner-up must score at least this many times the winner to call the match clear
CLEAR_GAP = 0.05                        # the runner-up must also be at least this far away in absolute terms (two near-zero scores are never 'clear')
RULESET_LABEL = {"pve": "PvE", "pvp": "PvP", "rp": "RP", "rp pvp": "RP PvP"}
NO_SCORE = 9.9                          # stands in for "could not be scored" when assignments are compared


def slug_parts(slug: str) -> Tuple[str, str, str]:
    """('us', 'pvp', 'alliance') for 'uspvp--alliance'; ('us', 'rp pvp', 'horde') for 'usrppvp--horde'."""
    realm, _, faction = str(slug).partition("--")
    m = re.match(r"^(us|eu)(.*)$", realm)
    region, rs = (m.group(1), m.group(2)) if m else ("", realm)
    return region, {"rppvp": "rp pvp"}.get(rs, rs), faction


def slug_label(slug: str) -> str:
    region, rs, faction = slug_parts(slug)
    return f"{region.upper()} {RULESET_LABEL.get(rs, rs.upper())} ({faction.title()})"


# ------------------------------------------------------------------------------------------------------------ prices from a book
Ladder = Tuple[np.ndarray, np.ndarray, np.ndarray]


def make_ladder(unit_price: np.ndarray, quantity: np.ndarray) -> Ladder:
    """(prices sorted up, cumulative quantity, cumulative cost) for one item's listings."""
    order = np.argsort(unit_price, kind="mergesort")
    p = np.asarray(unit_price, dtype="float64")[order]
    q = np.asarray(quantity, dtype="float64")[order]
    return p, np.cumsum(q), np.cumsum(q * p)


def avg_price(ladder: Ladder, qty: float) -> Optional[float]:
    """Average price per unit of buying `qty` units from the cheapest listing up; None when the book holds fewer than `qty` units."""
    p, cq, cc = ladder
    if qty <= 0 or len(p) == 0 or cq[-1] < qty:
        return None
    i = int(np.searchsorted(cq, qty, side="left"))
    prev_q = float(cq[i - 1]) if i > 0 else 0.0
    prev_c = float(cc[i - 1]) if i > 0 else 0.0
    return (prev_c + (qty - prev_q) * float(p[i])) / qty


def snapshot_ladders(snap: Snapshot, ids: Iterable[int]) -> Dict[int, Ladder]:
    want = {int(i) for i in ids}
    b = snap.book[snap.book["item_id"].isin(want)]
    out: Dict[int, Ladder] = {}
    for item, g in b.groupby("item_id"):
        out[int(item)] = make_ladder(g["unit_price"].to_numpy(), g["quantity"].to_numpy())
    return out


def name_map(items: pd.DataFrame, weight: Optional[pd.Series] = None) -> Dict[str, int]:
    """{item name: item id}. Where several ids share a name, the id with the most weight (units on sale in our houses) wins; without weights, the lowest id."""
    if items is None or items.empty:
        return {}
    t = items[["item_id", "name"]].dropna().copy()
    t["item_id"] = t["item_id"].astype(int)
    t["w"] = t["item_id"].map(weight).fillna(0.0) if weight is not None else 0.0
    t = t.sort_values(["name", "w", "item_id"], ascending=[True, False, True])
    return {str(n): int(g["item_id"].iloc[0]) for n, g in t.groupby("name")}


# ------------------------------------------------------------------------------------------------------------ scoring
def page_rows(page: dict, ids_by_name: Dict[str, int]) -> List[Tuple[int, float, float]]:
    """[(item id, qty, site price per unit in copper)] for the page's auction-house rows whose name we can map to an id."""
    out = []
    for r in page.get("rows", []):
        if r.get("source") != "ah":
            continue
        i = ids_by_name.get(r["item"])
        if i is not None and r["qty"] > 0 and r["each_copper"] > 0:
            out.append((int(i), float(r["qty"]), float(r["each_copper"])))
    return out


def score_page(ladders: Dict[int, Ladder], rows: Sequence[Tuple[int, float, float]], min_items: int = MIN_ITEMS) -> Tuple[float, int]:
    """(median |ln(ours / page)|, items priced); (nan, n) when fewer than min_items could be priced."""
    errs = []
    for item, qty, each in rows:
        lad = ladders.get(item)
        ours = avg_price(lad, qty) if lad is not None else None
        if ours and ours > 0:
            errs.append(abs(float(np.log(ours / each))))
    if len(errs) < min_items:
        return float("nan"), len(errs)
    return float(np.median(errs)), len(errs)


def fingerprint(snaps_by_house: Dict[int, Sequence[Snapshot]], fixture: dict, ids_by_name: Dict[str, int], min_items: int = MIN_ITEMS) -> pd.DataFrame:
    """Long table: slug, house, score (best over snapshots), items (priced at that snapshot), ts (the snapshot), snapshots_scored."""
    pages = fixture["houses"] if "houses" in fixture else fixture
    rows_by_slug = {slug: page_rows(p, ids_by_name) for slug, p in pages.items()}
    wanted = {i for rr in rows_by_slug.values() for i, _, _ in rr}
    out = []
    for house, snaps in sorted(snaps_by_house.items()):
        ladders = [(s.ts, snapshot_ladders(s, wanted)) for s in sorted(snaps, key=lambda s: s.ts) if s.kind == "commodity"]
        for slug, rr in rows_by_slug.items():
            best, best_n, best_ts, scored = float("nan"), 0, None, 0
            for ts, lad in ladders:
                sc, n = score_page(lad, rr, min_items)
                if np.isnan(sc):
                    continue
                scored += 1
                if np.isnan(best) or sc < best:
                    best, best_n, best_ts = sc, n, ts
            out.append(dict(slug=slug, house=int(house), score=best, items=best_n, ts=best_ts, snapshots_scored=scored))
    return pd.DataFrame(out, columns=["slug", "house", "score", "items", "ts", "snapshots_scored"])


def wide_scores(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot(index="slug", columns="house", values="score")


def best_assignment(wide: pd.DataFrame, slugs: Sequence[str], houses: Sequence[int]) -> Tuple[Dict[str, int], float]:
    """The one-house-per-page assignment with the lowest total score (every way of choosing is tried: at most 5 pages). Unscorable pairs count as NO_SCORE."""
    slugs, houses = [s for s in slugs if s in wide.index], [h for h in houses if h in wide.columns]
    if not slugs or len(houses) < len(slugs):
        return {}, float("nan")
    best, best_total = {}, float("inf")
    for perm in itertools.permutations(houses, len(slugs)):
        total = sum(NO_SCORE if np.isnan(wide.loc[s, h]) else float(wide.loc[s, h]) for s, h in zip(slugs, perm))
        if total < best_total:
            best, best_total = dict(zip(slugs, perm)), total
    return best, best_total


def ranked(wide: pd.DataFrame, slug: str) -> List[Tuple[int, float]]:
    row = wide.loc[slug].dropna().sort_values()
    return [(int(h), float(v)) for h, v in row.items()]


def verdicts(wide: pd.DataFrame, known: Optional[Dict[str, int]] = None, groups: Optional[Dict[str, Sequence[int]]] = None) -> List[str]:
    """Plain-language lines: the method check against what is known, then each page's best house (clear or unclear), then the best one-to-one assignment within each faction."""
    known = KNOWN if known is None else known
    groups = FACTION_HOUSES if groups is None else groups
    lines: List[str] = []
    for slug, house in known.items():
        if slug not in wide.index:
            continue
        r = ranked(wide, slug)
        if r and r[0][0] == house:
            lines.append(f"[OK] method check: {slug_label(slug)} lines up best with house {house}, the one house we know for it (score {r[0][1]:.2f}).")
        elif r:
            lines.append(f"[PROBLEM] method check FAILED: {slug_label(slug)} should be house {house} but lines up best with house {r[0][0]} ({r[0][1]:.2f}; house {house} scores "
                         f"{wide.loc[slug].get(house, float('nan')):.2f}). Do not trust the other lines below.")
        else:
            lines.append(f"[WATCH] method check: {slug_label(slug)} could not be scored.")
    for slug in wide.index:
        r = ranked(wide, slug)
        if len(r) < 2:
            lines.append(f"[WATCH] {slug_label(slug)}: too few houses could be scored ({len(r)}).")
            continue
        (h1, s1), (h2, s2) = r[0], r[1]
        ratio = s2 / s1 if s1 > 0 else (1.0 if s2 <= 0 else float("inf"))
        tag = "OK" if (s1 <= CLEAR_SCORE and ratio >= CLEAR_RATIO and s2 - s1 >= CLEAR_GAP) else "WATCH"
        word = "clear" if tag == "OK" else "UNCLEAR"
        fac = slug_parts(slug)[2]
        side = "" if h1 in groups.get(fac, ()) else f" (house {h1} is not on the {fac} list in the community table)"
        lines.append(f"[{tag}] {slug_label(slug)}: best house {h1} (score {s1:.2f}), next {h2} ({s2:.2f}), {ratio:.1f}x apart: {word}{side}.")
    for fac in ("alliance", "horde"):
        sl = [s for s in wide.index if slug_parts(s)[2] == fac]
        hs = [h for h in groups.get(fac, ()) if h in wide.columns]
        if len(sl) >= 2 and len(hs) >= len(sl):
            asg, tot = best_assignment(wide, sl, hs)
            indep = {s: ranked(wide, s)[0][0] for s in sl if ranked(wide, s)}
            agree = all(indep.get(s) == h for s, h in asg.items())
            lines.append(f"{fac.title()} one-to-one assignment (lowest total score {tot:.2f}): " + ", ".join(f"{slug_parts(s)[1].upper()} = {h}" for s, h in sorted(asg.items())) +
                         ("" if agree else "   (differs from each page's own best house: treat as unclear)"))
    return lines


def matrix_text(wide: pd.DataFrame) -> str:
    cols = sorted(wide.columns)
    head = ["page".ljust(18)] + [str(c).rjust(6) for c in cols]
    rows = []
    for slug in wide.index:
        cells = [slug.ljust(18)]
        best = wide.loc[slug].min()
        for c in cols:
            v = wide.loc[slug, c]
            txt = "-" if np.isnan(v) else f"{v:.2f}"
            cells.append((txt + ("*" if (not np.isnan(v) and v == best) else " ")).rjust(6))
        rows.append("".join(cells))
    return "".join(head) + "\n" + "\n".join(rows) + "\n(* = the page's best house; lower is closer; - = could not be scored)"
