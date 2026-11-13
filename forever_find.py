"""Forever listing finder. Answers: WHICH Forever auction house holds the one listing I just posted in game?

Why: Forever has no realms. The API shows 12 auction houses with listings (five of each: Alliance, Horde, Blackwater/neutral, one set empty) and we do not know which set is the player's market.
Post ONE distinctive listing in game (for example 1 Linen Cloth for 5g 43s 21c = 54321 copper), wait for Blizzard's data to refresh (up to about an hour, not confirmed for Forever), then run this.

It reads Blizzard's LIVE data directly (no Hugging Face, no Discord, nothing is written anywhere) and prints, for every auction house, how many listings of that item exist and which listing matches
the price exactly: regular auctions are compared on `buyout`, commodities on `unit_price` (both in copper; 10,000 copper = 1 gold). If nothing matches it says when each house's data was last updated
(compare with the minute you posted) and shows the closest prices, which also exposes a units mistake.

Settings (environment variables; the workflow fills them from its boxes):
  FIND_ITEM_ID   item id (default 2589 = Linen Cloth)
  FIND_PRICE     price in copper (default 54321)
  FIND_QUANTITY  optional: only count listings with this quantity
Nothing in the output names an account or a character: the auction data carries no owner field and none is requested.
Exit code 0 even when nothing is found (that is an answer, not a failure); 1 only for real trouble (token, repeated 5xx)."""
import os
import sys

import fetch_forever as F

DEFAULT_ITEM = 2589          # Linen Cloth
DEFAULT_PRICE = 54321        # 5g 43s 21c


def fmt_copper(c):
    c = int(c)
    return f"{c // 10000}g {c // 100 % 100}s {c % 100}c"


def int_setting(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else default


def scan_house(token, ruleset, hid, item_id, price, quantity):
    """Returns (rows, matches, last_modified) for one house: rows = [(kind, auction id, quantity, price, time_left)] of the item, matches = the exact-price subset."""
    rows, last_modified = [], {}
    for kind, tail, field in (("auctions", "", "buyout"), ("commodities", "/commodities", "unit_price")):
        status, body, header = F.api_get(f"{F.API_HOST}/data/wow/ruleset/{ruleset}/auctions/{hid}{tail}", token)
        last_modified[kind] = header
        for a in (body or {}).get("auctions", []) or []:
            if (a.get("item") or {}).get("id") != item_id or a.get(field) is None:
                continue
            if quantity is not None and a.get("quantity") != quantity:
                continue
            rows.append((kind, a.get("id"), a.get("quantity"), int(a[field]), a.get("time_left")))
    return rows, [r for r in rows if r[3] == price], last_modified


def main():
    item_id = int_setting("FIND_ITEM_ID", DEFAULT_ITEM)
    price = int_setting("FIND_PRICE", DEFAULT_PRICE)
    quantity = int_setting("FIND_QUANTITY", None)
    F.log(f"Looking for item {item_id} at {price} copper ({fmt_copper(price)})" + (f", quantity {quantity}" if quantity is not None else "") + ".")
    try:
        token = F.get_blizzard_token()
        ruleset = F.wanted_rulesets()[0]
        index = F.house_index(token, ruleset)
    except Exception as e:
        F.log(f"Could not start: {F.clean(e)}")
        return 1
    if not index:
        F.log("Blizzard serves no auction-house list right now, so there is nothing to search.")
        return 0

    found, closest, failures = [], [], []
    for hid in sorted(index):
        name = F.house_name(index[hid], hid)
        try:
            rows, matches, lm = scan_house(token, ruleset, hid, item_id, price, quantity)
        except Exception as e:
            failures.append(f"house {hid}: {F.clean(e)}")
            F.log(f"  house {hid} ({name}): FAILED: {F.clean(e)}")
            continue
        reg = sum(1 for r in rows if r[0] == "auctions")
        com = sum(1 for r in rows if r[0] == "commodities")
        stamp = lm.get("commodities") or lm.get("auctions") or "no data (empty house)"
        F.log(f"  house {hid} ({name}): {reg} regular and {com} commodity listings of this item, exact matches {len(matches)}; data as of {stamp}")
        for kind, aid, qty, p, left in matches:
            found.append((hid, name, kind, aid, qty, p, left))
        closest.extend((abs(p - price), hid, name, kind, qty, p) for kind, aid, qty, p, left in rows)

    if found:
        F.log("FOUND:")
        for hid, name, kind, aid, qty, p, left in found:
            F.log(f"  house {hid} ({name}), {kind}: auction id {aid}, quantity {qty}, price {p} copper ({fmt_copper(p)}), time left {left}")
        houses = sorted({f[0] for f in found})
        F.log(f"Result: the listing is in house {', '.join(map(str, houses))}." + ("" if len(houses) == 1 else " More than one house matched: another seller may share this price, or houses overlap; post a second listing at a different price."))
    else:
        F.log("NOT FOUND in any house yet. Either Blizzard's data has not refreshed since you posted (compare the 'data as of' times with your posting time, then run this again later), "
              "or the price / item / quantity differs from what is stored.")
        for diff, hid, name, kind, qty, p in sorted(closest)[:5]:
            F.log(f"  closest: house {hid} ({name}), {kind}: quantity {qty}, price {p} copper ({fmt_copper(p)})")
    if failures:
        F.log("Some houses could not be read:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
