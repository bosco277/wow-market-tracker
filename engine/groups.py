"""Item groups: which kind of thing each item is, so what we learn about one item can inform the others in its group.

Source: Blizzard's own item data (the Game Data API `item` endpoint, namespace static-us). It gives every item a class and a
subclass (for example class Tradeskill / subclass Cloth, or class Consumable / subclass Potion), a quality, an item level and the
vendor sell price. Every addon's built-in categories (TSM's default groups, the auction house category list) come from the same
class and subclass numbers, so this is the most complete grouping that exists. Community groups (TSM public groups) are made by
individual players and cover only what they chose, so they are not used as a base.

What this module adds on top:
  group      "Crafting materials: Cloth", "Consumables: Potion", ... (class and subclass in plain words)
  family     the coarse parent: Consumables, Crafting materials, Gems, Recipes, Gear, Enhancements, Pets and mounts, Other
  use        a finer label from name rules where Blizzard lumps things together ("Feast", "Healing potion", "Augment rune" ...);
             these are name-based guesses, shown separately so they can be checked
  name_family / siblings / tier_guess  items that share a name are quality tiers of one item (Tier 1 / 2 / 3); tier_guess ranks them
             by item id (lowest id first), which is how they have been numbered so far but is a guess
  vendor_sell_gold  what a vendor pays; the auction price cannot sensibly sit below it
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

COPPER_PER_GOLD = 10_000.0
ITEM_URL = "https://us.api.blizzard.com/data/wow/item/{id}?namespace=static-us&locale=en_US"

CLASS_LABEL = {
    "Consumable": "Consumables", "Tradeskill": "Crafting materials", "Trade Goods": "Crafting materials", "Gem": "Gems", "Recipe": "Recipes",
    "Weapon": "Weapons", "Armor": "Armor", "Item Enhancement": "Enhancements", "Glyph": "Glyphs", "Battle Pet": "Battle pets",
    "Container": "Containers", "Miscellaneous": "Miscellaneous", "Profession": "Profession tools", "Quest": "Quest items",
    "Reagent": "Reagents", "Housing": "Housing decor",
}
FAMILY = {
    "Consumable": "Consumables", "Tradeskill": "Crafting materials", "Trade Goods": "Crafting materials", "Gem": "Gems", "Recipe": "Recipes",
    "Weapon": "Gear", "Armor": "Gear", "Item Enhancement": "Enhancements", "Glyph": "Enhancements", "Battle Pet": "Pets and mounts",
    "Container": "Other", "Miscellaneous": "Other", "Profession": "Other", "Quest": "Other", "Reagent": "Other", "Housing": "Other",
}

# Name rules for finer uses. Order matters: the first match wins. These are guesses from the name only.
USE_RULES = [
    (r"\bfeast\b|\bcauldron\b|\bbanquet\b", "Feast / cauldron"),
    (r"\baugment rune\b|\bvantus rune\b", "Rune"),
    (r"\bphial\b|\bflask\b", "Flask / phial"),
    (r"healing potion|health potion|healthstone", "Healing potion"),
    (r"mana potion|\bmana\b.*potion", "Mana potion"),
    (r"\bpotion\b", "Combat potion"),
    (r"\belixir\b", "Elixir"),
    (r"whetstone|weightstone|sharpening|oil\b", "Weapon buff"),
    (r"\bbandage\b", "Bandage"),
    (r"\bore\b|\bbar\b|\bingot\b", "Ore or bar"),
    (r"\bleaf\b|\bbloom\b|\bpetal\b|\bweed\b|\broot\b|\blotus\b", "Herb"),
    (r"\bbolt\b|\bcloth\b|\bsilk\b|\bweave\b", "Cloth"),
    (r"\bhide\b|\bleather\b|\bscale\b|\bscales\b", "Leather or scales"),
    (r"\bshard\b|\bcrystal\b|\bdust\b|\bessence\b", "Enchanting material"),
    (r"\btopaz\b|\bsapphire\b|\bemerald\b|\bruby\b|\bonyx\b|\bdiamond\b", "Gem material"),
]


# ------------------------------------------------------------------ fetching
class RateLimiter:
    """At most `per_second` calls a second across all threads (Blizzard allows 100 a second; we stay far below)."""

    def __init__(self, per_second: float):
        self.gap, self.next, self.lock = 1.0 / per_second, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def fetch_item_json(item_id: int, get: Callable, limiter: Optional[RateLimiter] = None, tries: int = 4) -> Optional[dict]:
    """One item from the API. `get(url)` returns (status_code, json_or_None). None when the item does not exist (404) or kept failing."""
    for k in range(tries):
        if limiter:
            limiter.wait()
        try:
            code, js = get(ITEM_URL.format(id=int(item_id)))
        except Exception:
            code, js = 599, None
        if code == 200 and isinstance(js, dict):
            return js
        if code == 404:
            return None
        time.sleep(min(2 ** k, 8) * 0.5)                  # 429 / 5xx / network trouble: back off and retry
    return None


def parse_item(js: dict) -> dict:
    """The fields we keep from one item response. Every field is optional in the API, so each is read defensively."""
    ic, sc = js.get("item_class") or {}, js.get("item_subclass") or {}
    q = js.get("quality") or {}
    it = js.get("inventory_type") or {}
    return dict(
        item_id=int(js["id"]), name=js.get("name", ""), item_class_id=ic.get("id"), item_class=ic.get("name", ""),
        item_subclass_id=sc.get("id"), item_subclass=sc.get("name", ""), quality=q.get("type", ""), item_level=js.get("level"),
        required_level=js.get("required_level"), inventory_type=it.get("type", ""), is_stackable=bool(js.get("is_stackable", False)),
        max_count=js.get("max_count"), sell_price_copper=js.get("sell_price"), purchase_price_copper=js.get("purchase_price"))


def fetch_items(item_ids: Iterable[int], get: Callable, workers: int = 8, per_second: float = 15.0) -> pd.DataFrame:
    ids = sorted({int(i) for i in item_ids})
    limiter = RateLimiter(per_second)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda i: fetch_item_json(i, get, limiter), ids))
    rows = [parse_item(js) for js in res if js]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ grouping
def _use(name: str) -> str:
    n = str(name).lower()
    for pat, label in USE_RULES:
        if re.search(pat, n):
            return label
    return ""


def tag(df: pd.DataFrame) -> pd.DataFrame:
    """Add group, family, use, name_family, siblings, tier_guess and vendor_sell_gold to the parsed items."""
    if df.empty:
        return df
    t = df.copy()
    cls = t["item_class"].fillna("").astype(str)
    sub = t["item_subclass"].fillna("").astype(str)
    t["family"] = cls.map(lambda c: FAMILY.get(c, "Other"))
    t["group"] = [f"{CLASS_LABEL.get(c, c or 'Unknown')}: {s}" if s else CLASS_LABEL.get(c, c or "Unknown") for c, s in zip(cls, sub)]
    t["use"] = t["name"].map(_use)
    t.loc[~cls.isin(["Consumable", "Tradeskill", "Trade Goods", "Gem", "Item Enhancement"]), "use"] = ""   # name rules only where they make sense
    t["name_family"] = t["name"]
    t["siblings"] = t.groupby("name")["item_id"].transform("count")
    t["tier_guess"] = t.groupby("name")["item_id"].rank(method="first").astype(int)
    t.loc[t["siblings"] == 1, "tier_guess"] = 0                                              # 0 = no other item with this name
    t["vendor_sell_gold"] = pd.to_numeric(t["sell_price_copper"], errors="coerce") / COPPER_PER_GOLD
    t["vendor_buy_gold"] = pd.to_numeric(t["purchase_price_copper"], errors="coerce") / COPPER_PER_GOLD
    return t


def refresh(existing: Optional[pd.DataFrame], wanted: Iterable[int], get: Callable, **kw) -> pd.DataFrame:
    """Keep what we already know; ask the API only for item ids not in `existing` (so later runs are short). Items the API
    does not know are simply left out, and asked about again next time."""
    wanted = {int(i) for i in wanted}
    have = set(existing["item_id"].astype(int)) if existing is not None and len(existing) else set()
    new = fetch_items(wanted - have, get, **kw) if wanted - have else pd.DataFrame()
    base = existing[existing["item_id"].isin(wanted)] if existing is not None and len(existing) else pd.DataFrame()
    raw_cols = ["item_id", "name", "item_class_id", "item_class", "item_subclass_id", "item_subclass", "quality", "item_level", "required_level",
                "inventory_type", "is_stackable", "max_count", "sell_price_copper", "purchase_price_copper"]
    parts = [x[[c for c in raw_cols if c in x.columns]] for x in (base, new) if len(x)]
    if not parts:
        return pd.DataFrame()
    return tag(pd.concat(parts, ignore_index=True).drop_duplicates("item_id", keep="last"))


def group_summary(groups: pd.DataFrame, items: pd.DataFrame, by: str = "group") -> pd.DataFrame:
    """One row per group: how many items, typical price and typical sales per day (from the items table the engine builds).
    Medians, so one wild item does not move a group; `spread` is the middle half of sales per day (p25 - p75) so you can see how
    alike the items in a group are: a group is only useful if its items behave alike."""
    if groups.empty or items.empty:
        return pd.DataFrame()
    m = groups.merge(items[["item_id", "floor_gold", "daily_est", "confidence"]], on="item_id", how="inner")
    m["gold_flow"] = m["daily_est"] * m["floor_gold"]
    g = m.groupby(by)
    out = g.agg(items=("item_id", "count"), price_median=("floor_gold", "median"), sold_per_day_median=("daily_est", "median"),
                sold_p25=("daily_est", lambda x: x.quantile(0.25)), sold_p75=("daily_est", lambda x: x.quantile(0.75)),
                gold_flow_median=("gold_flow", "median"), share_low_confidence=("confidence", lambda x: float((x == "Low").mean()))).reset_index()
    out["spread"] = out["sold_p75"] / out["sold_p25"].replace(0, np.nan)
    return out.sort_values("items", ascending=False).reset_index(drop=True)
