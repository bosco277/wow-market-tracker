"""Data helpers for the WoW Forever view of the app. No Streamlit or Plotly in here, so they can be tested without a browser.

The Forever view reads three things from the Forever dataset on Hugging Face (never the retail one):
  daily/d_YYYY-MM-DD.csv.gz        one row per auction house, item and US Eastern day (written by the "Forever daily tables" workflow)
  tables/forever_items.csv.gz      item names and classes (written by the "Forever tables" workflow)
  tables/forever_house_map.json    which house is which ruleset and faction

Every house is shown on its own: Forever has no realms, each auction house is its own market, and houses are never added together.
The sales columns are LOWER BOUNDS: a listing posted and sold inside one hour is in no hourly snapshot.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_REPO = "bosco277/wow-forever-data-lake"
DAILY_DIR = "daily"
DAILY_RE = re.compile(r"(?:^|/)d_(\d{4}-\d{2}-\d{2})\.csv\.gz$")
ITEMS_KEY = "tables/forever_items.csv.gz"
HOUSE_MAP_KEY = "tables/forever_house_map.json"
COMPLETE_HOURS = 18             # a day with 18+ usable hours is complete (the same rule as the daily tables)
MIN_COMPLETE_SHARE = 0.5        # the default day is the newest one where at least half the rows are complete
STALE_DAYS = 2                  # the daily job runs once a day; a newest stored day older than this means it missed runs
DEFAULT_HOUSE = 110
TZ = "America/New_York"

RULESET_NAME = {"pvp": "PvP", "pve": "PvE", "rp": "RP", "rp pvp": "RP-PvP"}
FACTION_NAME = {"alliance": "Alliance", "horde": "Horde", "neutral": "Neutral"}
FACTION_ORDER = {"alliance": 0, "horde": 1, "neutral": 2}
# The slots as the fetcher found them (Nov 25): used when the stored house map is missing.
FALLBACK_SLOTS = (("pvp", 110, 113, 116), ("pve", 111, 114, 117), ("rp", 119, 120, 121), ("rp pvp", 123, 124, 125))

NUM_COLUMNS = ["house", "item_id", "hours", "flagged_hours", "price_day", "price_min", "price_p15", "price_p30", "buy20", "buy200", "units_day", "listings_day",
               "gold_day", "sold_cert_day", "sold_low_day", "sold_day", "sold_high_day", "turnover_day", "active_share"]


# ------------------------------------------------------------------------------------------------ reading
def scrub(text, repo: str = DEFAULT_REPO) -> str:
    """An error message with the dataset's name and owner blanked out (nothing on screen names an account)."""
    t = str(text)
    repo = repo or DEFAULT_REPO
    owner = repo.split("/")[0]
    for part in sorted({repo, owner, repo.split("/")[-1]}, key=len, reverse=True):
        if part:
            t = t.replace(part, "<dataset>")
    return t


def day_of_key(key: str) -> Optional[str]:
    m = DAILY_RE.search(str(key))
    return m.group(1) if m else None


def read_daily(blobs: Iterable[bytes]) -> pd.DataFrame:
    """The stored day files (gzip CSV bytes) as one table. Unreadable files are skipped; the numbers come back as numbers."""
    parts = []
    for b in blobs:
        try:
            parts.append(pd.read_csv(io.BytesIO(b), compression="gzip", dtype={"kind": "object", "day": "object", "engine_version": "object"}))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(columns=["house", "kind", "item_id", "day"] + NUM_COLUMNS[2:])
    d = pd.concat(parts, ignore_index=True)
    for c in NUM_COLUMNS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan
    d = d.dropna(subset=["house", "item_id", "day"])
    d["house"] = d["house"].astype(int)
    d["item_id"] = d["item_id"].astype(int)
    d["day"] = d["day"].astype(str)
    # a day stored twice (should not happen) keeps the row with more hours
    d = d.sort_values(["house", "item_id", "day", "hours"]).drop_duplicates(["house", "item_id", "day"], keep="last")
    return d.reset_index(drop=True)


def read_items(data: Optional[bytes]) -> pd.DataFrame:
    """item_id, name, item_class from the stored item table (empty table if it is missing or unreadable)."""
    try:
        t = pd.read_csv(io.BytesIO(data), compression="gzip", keep_default_na=False, na_values=[""], dtype={"name": "object", "item_class": "object"})
        t = t[[c for c in ("item_id", "name", "item_class") if c in t.columns]]
        t["item_id"] = pd.to_numeric(t["item_id"], errors="coerce")
        t = t.dropna(subset=["item_id"])
        t["item_id"] = t["item_id"].astype(int)
        for c in ("name", "item_class"):
            if c not in t.columns:
                t[c] = ""
            t[c] = t[c].fillna("").astype(str)
        return t.drop_duplicates("item_id").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["item_id", "name", "item_class"])


def parse_house_map(data: Optional[bytes]) -> Dict[int, dict]:
    """{house: {ruleset, faction, confidence, basis}} from the stored house map; the Nov 25 slot order if it is missing."""
    try:
        m = json.loads(data.decode("utf-8")) if data else {}
        houses = {int(k): dict(v) for k, v in (m.get("houses") or {}).items()}
    except Exception:
        houses = {}
    if not houses:
        for ruleset, a, h, n in FALLBACK_SLOTS:
            for house, faction in ((a, "alliance"), (h, "horde"), (n, "neutral")):
                houses[house] = dict(ruleset=ruleset, faction=faction, confidence="unknown", basis="house map not stored yet: slot order only")
    return houses


# ------------------------------------------------------------------------------------------------ houses
def house_label(house: int, hmap: Dict[int, dict]) -> str:
    """'110 · PvP · Alliance', with '(unverified)' when the house map only assumes it."""
    r = hmap.get(int(house)) or {}
    ruleset = RULESET_NAME.get(r.get("ruleset"), str(r.get("ruleset") or "?"))
    faction = FACTION_NAME.get(r.get("faction"), str(r.get("faction") or "?"))
    tail = " (unverified)" if r.get("confidence") in ("assumed", "unknown") else ""
    return f"{int(house)} · {ruleset} · {faction}{tail}"


def house_order(houses: Iterable[int], hmap: Dict[int, dict]) -> List[int]:
    """Houses grouped by ruleset in the map's slot order (PvP, PvE, RP, RP-PvP), Alliance / Horde / Neutral inside each."""
    rs = [r for r, *_ in FALLBACK_SLOTS]
    def key(h):
        r = hmap.get(int(h)) or {}
        return (rs.index(r["ruleset"]) if r.get("ruleset") in rs else 99, FACTION_ORDER.get(r.get("faction"), 9), int(h))
    return sorted({int(h) for h in houses}, key=key)


def same_ruleset(house: int, hmap: Dict[int, dict], available: Iterable[int]) -> List[int]:
    """The houses of the same ruleset (the faction houses and the neutral one), in Alliance / Horde / Neutral order, that have data."""
    avail = {int(h) for h in available}
    rs = (hmap.get(int(house)) or {}).get("ruleset")
    if not rs:
        return [int(house)] if int(house) in avail else []
    return house_order([h for h, r in hmap.items() if r.get("ruleset") == rs and h in avail], hmap)


# ------------------------------------------------------------------------------------------------ days
def day_summary(daily: pd.DataFrame, house: int) -> pd.DataFrame:
    """Per stored day for one house: rows, complete rows and the complete share."""
    d = daily[daily["house"] == int(house)]
    if d.empty:
        return pd.DataFrame(columns=["day", "rows", "complete", "share"])
    g = d.groupby("day")
    s = pd.DataFrame({"rows": g.size(), "complete": g["hours"].apply(lambda x: int((x >= COMPLETE_HOURS).sum()))}).reset_index()
    s["share"] = s["complete"] / s["rows"]
    return s.sort_values("day").reset_index(drop=True)


def default_day(summ: pd.DataFrame) -> Optional[str]:
    """The newest day where at least half the rows are complete (the newest stored day is usually still filling in); else the newest day."""
    if summ is None or summ.empty:
        return None
    ok = summ[summ["share"] >= MIN_COMPLETE_SHARE]
    return str((ok if len(ok) else summ)["day"].iloc[-1])


def day_label(day: str, summ: pd.DataFrame) -> str:
    try:
        r = summ.set_index("day").loc[day]
        dt = datetime.strptime(day, "%Y-%m-%d")
        name = f"{dt:%a %b} {dt.day}"
        return name if r["share"] >= MIN_COMPLETE_SHARE else f"{name} (partial: {r['share']:.0%} of items complete)"
    except Exception:
        return str(day)


def freshness(daily: pd.DataFrame, now: Optional[datetime] = None) -> Tuple[str, bool]:
    """A one-line caption on how current the stored days are, and whether it looks like the daily job stopped."""
    if daily is None or daily.empty:
        return "", False
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now).tz_convert(TZ).date() if pd.Timestamp(now).tzinfo else pd.Timestamp(now).date()
    days = sorted(daily["day"].unique())
    newest = datetime.strptime(days[-1], "%Y-%m-%d").date()
    lag = (today - newest).days
    stale = lag > STALE_DAYS
    text = (f"Daily tables: {len(days)} days stored, {days[0]} to {days[-1]} (US Eastern days). The newest day fills in as the day goes on; "
            f"the job runs once a day at about 1:47 PM ET.")
    if stale:
        text = (f"The newest stored day is {days[-1]}, {lag} days ago: the 'Forever daily tables' job seems to have stopped. Check the Actions tab on GitHub "
                f"(and that the repository variable FOREVER_DAILY_BUILD is yes). {len(days)} days stored.")
    return text, stale


# ------------------------------------------------------------------------------------------------ names
def item_names(items: pd.DataFrame) -> Dict[int, str]:
    if items is None or items.empty:
        return {}
    return {int(i): n for i, n in zip(items["item_id"], items["name"]) if isinstance(n, str) and n.strip()}


def item_classes(items: pd.DataFrame) -> Dict[int, str]:
    if items is None or items.empty or "item_class" not in items.columns:
        return {}
    return {int(i): c for i, c in zip(items["item_id"], items["item_class"]) if isinstance(c, str) and c.strip()}


def name_of(item_id: int, names: Dict[int, str]) -> str:
    return names.get(int(item_id)) or f"Item {int(item_id)}"


def item_labels(ids: Iterable[int], names: Dict[int, str]) -> Dict[int, str]:
    """{item_id: 'Name (id)'} for the search box."""
    return {int(i): f"{name_of(i, names)} ({int(i)})" for i in ids}


# ------------------------------------------------------------------------------------------------ tables
MARKET_COLUMNS = ["Item", "Class", "Item ID", "Price (g)", "Lowest (g)", "Cheapest 30% (g)", "Buy 20 avg (g)", "Buy 200 avg (g)", "Units listed",
                  "Gold in play (g)", "Seen sold/day (units)", "Certainly sold/day (units)", "Seen traded (g/day)", "Turnover (%)", "Hours selling (%)", "Hours"]


def market_view(daily: pd.DataFrame, house: int, day: str, names: Dict[int, str], classes: Optional[Dict[int, str]] = None, min_gold: float = 0.0,
                complete_only: bool = True, query: str = "", class_filter: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """The Market table for one house and day, ranked by gold seen traded per day (seen sold x price). `_id` holds the item id for clicks."""
    classes = classes or {}
    d = daily[(daily["house"] == int(house)) & (daily["day"] == str(day))].copy()
    if complete_only:
        d = d[d["hours"] >= COMPLETE_HOURS]
    if min_gold:
        d = d[d["gold_day"].fillna(0) >= float(min_gold)]
    d["Item"] = [name_of(i, names) for i in d["item_id"]]
    d["Class"] = [classes.get(int(i), "") for i in d["item_id"]]
    if class_filter is not None:
        keep = set(class_filter)
        d = d[d["Class"].isin(keep)]
    q = str(query or "").strip().lower()
    if q:
        d = d[d["Item"].str.lower().str.contains(q, regex=False) | d["item_id"].astype(str).str.contains(q, regex=False)]
    out = pd.DataFrame({
        "Item": d["Item"], "Class": d["Class"], "Item ID": d["item_id"].astype(int),
        "Price (g)": d["price_day"], "Lowest (g)": d["price_min"], "Cheapest 30% (g)": d["price_p30"], "Buy 20 avg (g)": d["buy20"], "Buy 200 avg (g)": d["buy200"],
        "Units listed": d["units_day"], "Gold in play (g)": d["gold_day"], "Seen sold/day (units)": d["sold_low_day"], "Certainly sold/day (units)": d["sold_cert_day"],
        "Seen traded (g/day)": d["sold_low_day"] * d["price_day"], "Turnover (%)": d["turnover_day"] * 100.0, "Hours selling (%)": d["active_share"] * 100.0,
        "Hours": d["hours"], "_id": d["item_id"].astype(int)})
    out = out.sort_values(["Seen traded (g/day)", "Gold in play (g)"], ascending=False, na_position="last").reset_index(drop=True)
    return out


def class_options(daily: pd.DataFrame, house: int, day: str, classes: Dict[int, str]) -> List[str]:
    d = daily[(daily["house"] == int(house)) & (daily["day"] == str(day))]
    return sorted({classes.get(int(i), "") for i in d["item_id"]} - {""})


def item_history(daily: pd.DataFrame, house: int, item_id: int) -> pd.DataFrame:
    """Every stored day of one item in one house, oldest first, with a complete flag."""
    h = daily[(daily["house"] == int(house)) & (daily["item_id"] == int(item_id))].sort_values("day").copy()
    h["complete"] = h["hours"] >= COMPLETE_HOURS
    return h.reset_index(drop=True)


def history_table(h: pd.DataFrame) -> pd.DataFrame:
    """The item's days as a readable table, newest first."""
    if h is None or h.empty:
        return pd.DataFrame()
    t = pd.DataFrame({"Day": h["day"], "Price (g)": h["price_day"], "Lowest (g)": h["price_min"], "Buy 20 avg (g)": h["buy20"], "Units listed": h["units_day"],
                      "Seen sold/day (units)": h["sold_low_day"], "Certainly sold/day (units)": h["sold_cert_day"], "Turnover (%)": h["turnover_day"] * 100.0,
                      "Hours": h["hours"], "Complete": h["complete"]})
    return t.iloc[::-1].reset_index(drop=True)


def item_across_houses(daily: pd.DataFrame, day: str, item_id: int, hmap: Dict[int, dict]) -> pd.DataFrame:
    """One item on one day in every house that has it, in house-map order (never added together)."""
    d = daily[(daily["day"] == str(day)) & (daily["item_id"] == int(item_id))]
    if d.empty:
        return pd.DataFrame()
    d = d.set_index("house").loc[house_order(d["house"], hmap)].reset_index()
    return pd.DataFrame({"House": [house_label(h, hmap) for h in d["house"]], "Price (g)": d["price_day"].to_numpy(), "Buy 20 avg (g)": d["buy20"].to_numpy(),
                         "Units listed": d["units_day"].to_numpy(), "Seen sold/day (units)": d["sold_low_day"].to_numpy(), "Hours": d["hours"].to_numpy()})


def compare_view(daily: pd.DataFrame, day: str, houses: List[int], hmap: Dict[int, dict], names: Dict[int, str], min_gold: float = 0.0,
                 complete_only: bool = True, min_units: float = 20.0) -> pd.DataFrame:
    """The houses of one ruleset side by side on one day: price and units in each, where an item is cheapest and dearest, and the gap between them.
    An item needs a (complete, if asked) row with at least `min_units` units in at least two of the houses; `min_gold` is gold in play in the first house given."""
    houses = [int(h) for h in houses]
    d = daily[(daily["day"] == str(day)) & daily["house"].isin(houses)].copy()
    if complete_only:
        d = d[d["hours"] >= COMPLETE_HOURS]
    d = d[d["units_day"].fillna(0) >= float(min_units)]
    if d.empty or len(houses) < 2:
        return pd.DataFrame()
    price = d.pivot_table(index="item_id", columns="house", values="price_day", aggfunc="first")
    units = d.pivot_table(index="item_id", columns="house", values="units_day", aggfunc="first")
    gold = d.pivot_table(index="item_id", columns="house", values="gold_day", aggfunc="first")
    price = price.reindex(columns=houses)
    units = units.reindex(index=price.index, columns=houses)
    n = price.notna().sum(axis=1)
    keep = n >= 2
    if min_gold:
        g0 = gold.reindex(index=price.index).get(houses[0])
        keep &= (g0.fillna(0) >= float(min_gold)) if g0 is not None else False
    price, units = price[keep], units[keep]
    if price.empty:
        return pd.DataFrame()
    lo, hi = price.min(axis=1), price.max(axis=1)
    out = pd.DataFrame({"Item": [name_of(i, names) for i in price.index], "Item ID": price.index.astype(int)})
    for h in houses:
        short = house_label(h, hmap).split(" · ")[-1].replace(" (unverified)", "")
        out[f"{short} {h} (g)"] = price[h].to_numpy()
        out[f"{short} {h} units"] = units[h].to_numpy()
    out["Cheapest at"] = [int(h) for h in price.idxmin(axis=1)]
    out["Dearest at"] = [int(h) for h in price.idxmax(axis=1)]
    out["Gap (g)"] = (hi - lo).to_numpy()
    out["Gap (%)"] = ((hi / lo - 1.0) * 100.0).to_numpy()
    out["_id"] = price.index.astype(int)
    return out.sort_values("Gap (%)", ascending=False).reset_index(drop=True)


def market_headline(daily: pd.DataFrame, house: int, day: str) -> dict:
    """The few numbers above the Market table."""
    d = daily[(daily["house"] == int(house)) & (daily["day"] == str(day))]
    full = d[d["hours"] >= COMPLETE_HOURS]
    active = (full["sold_low_day"].fillna(0) > 0)
    return dict(items=int(len(d)), complete=int(len(full)), gold=float(full["gold_day"].fillna(0).sum()),
                traded=float((full["sold_low_day"].fillna(0) * full["price_day"].fillna(0)).sum()),
                selling_share=float(active.mean()) if len(full) else float("nan"))
