"""House flips: buy on one auction house of a ruleset and resell on another (WoW Forever). No Streamlit in here, so it can be tested on its own.

A character reaches its own faction's auction house and the neutral one (Blackwater). Two directions:
  Neutral -> faction (the main one): buy what is listed on the neutral house, resell on your Alliance or Horde house.
  Faction -> Neutral (the reverse): buy on your house, resell on the neutral house. The neutral house is natively slower, so an item needs a bigger margin,
                                    or must be one we have seen selling in most hours on the neutral house.

How a row is worked out (all prices per unit, gold):
  resell price   the LOWER of the resell house's lowest price now and its typical price (median of the last 72 hours' lowest prices): you have to match the
                 current lowest listing to sell, and a price that is above its usual level is not counted on.
  after the cut  resell price x (1 - that house's auction house cut). The cuts are the ASSUMPTIONS below, subtracted from every profit here.
  pay at most    after-the-cut price / (1 + required margin)
  units          every unit listed on the buy house now at or under 'pay at most', cheapest first, but no more than the resell house sells in `days_to_sell`
                 days (its best estimate of sales per day, which is itself a lower bound), so a row never assumes you can sell more than the market takes.
  profit         (after-the-cut price x units) - what those units cost. margin = after-the-cut price / average price paid - 1.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---- ASSUMPTIONS (Classic-era values; NOT yet checked on WoW Forever). Subtracted from every profit figure; listed in the footnote under the page.
FEES = dict(faction_cut=0.05, neutral_cut=0.15)
DEFAULTS = dict(min_margin=0.15, min_margin_reverse=0.30, fast_active_share=0.75, days_to_sell=2.0)

COLUMNS = ["Item", "Item ID", "Lowest there now (g)", "Pay at most (g)", "Units at or under (units)", "Buy (units)", "Cost (g)", "Average paid (g)",
           "Resell at (g)", "Profit after cut (g)", "Margin after cut (%)", "Resell house sells per day (units)", "Days to sell", "Seen selling there (%)",
           "Confidence", "Why it qualifies"]


def resell_price(sell_items: pd.DataFrame) -> pd.Series:
    """Per item: the lower of the lowest price now and the typical (p50) price on the resell house."""
    s = sell_items.set_index("item_id")
    now = pd.to_numeric(s.get("floor_gold"), errors="coerce")
    typ = pd.to_numeric(s.get("band_p50"), errors="coerce") if "band_p50" in s.columns else now
    return pd.concat([now, typ], axis=1).min(axis=1, skipna=True)


def flip_table(buy_book: pd.DataFrame, sell_items: pd.DataFrame, sell_cut: float, min_margin: float, days_to_sell: float = DEFAULTS["days_to_sell"],
               names: Optional[Dict[int, str]] = None, fast_share: Optional[float] = None, fast_margin: Optional[float] = None) -> pd.DataFrame:
    """Items worth buying from `buy_book` (the buy house's order book: item_id, price_gold, quantity) and reselling on the house whose derived items table is
    `sell_items`. Each item needs `min_margin` after the cut; with `fast_share` set, an item seen selling in at least that share of the resell house's hours needs
    only `fast_margin` (the reverse direction's 'very high velocity' rule)."""
    names = names or {}
    if buy_book is None or buy_book.empty or sell_items is None or sell_items.empty:
        return pd.DataFrame(columns=COLUMNS + ["_id"])
    s = sell_items.drop_duplicates("item_id").set_index("item_id")
    price = resell_price(sell_items)
    net = price * (1.0 - float(sell_cut))
    active = pd.to_numeric(s.get("active_share", pd.Series(np.nan, index=s.index)), errors="coerce")
    need = pd.Series(float(min_margin), index=s.index)
    why = pd.Series(f"margin {min_margin:.0%}+", index=s.index)
    if fast_share is not None and fast_margin is not None:
        fast = active >= float(fast_share)
        need = need.where(~fast, float(fast_margin))
        why = why.where(~fast, f"fast seller there (seen selling in {fast_share:.0%}+ of hours), margin {fast_margin:.0%}+")
    cap_per_day = pd.to_numeric(s.get("daily_est", pd.Series(np.nan, index=s.index)), errors="coerce").fillna(0.0)
    max_buy = net / (1.0 + need)
    rows = []
    for item, lv in buy_book.groupby("item_id"):
        item = int(item)
        if item not in s.index or not np.isfinite(max_buy.get(item, np.nan)) or not np.isfinite(net.get(item, np.nan)):
            continue
        lv = lv.sort_values("price_gold")
        ok = lv[lv["price_gold"] <= max_buy[item]]
        avail = float(ok["quantity"].sum())
        cap = cap_per_day[item] * float(days_to_sell)
        if avail <= 0 or cap < 1:
            continue
        take, cost, left = 0.0, 0.0, np.floor(cap)
        for p, q in zip(ok["price_gold"], ok["quantity"]):
            n = min(float(q), left)
            take += n
            cost += n * float(p)
            left -= n
            if left <= 0:
                break
        if take <= 0:
            continue
        avg = cost / take
        profit = net[item] * take - cost
        if profit <= 0:
            continue
        rows.append({
            "Item": names.get(item) or (str(s.at[item, "name"]) if "name" in s.columns and isinstance(s.at[item, "name"], str) and s.at[item, "name"] else f"Item {item}"),
            "Item ID": item, "Lowest there now (g)": float(lv["price_gold"].iloc[0]), "Pay at most (g)": float(max_buy[item]),
            "Units at or under (units)": avail, "Buy (units)": take, "Cost (g)": cost, "Average paid (g)": avg, "Resell at (g)": float(price[item]),
            "Profit after cut (g)": profit, "Margin after cut (%)": (net[item] / avg - 1.0) * 100.0,
            "Resell house sells per day (units)": float(cap_per_day[item]), "Days to sell": take / cap_per_day[item] if cap_per_day[item] > 0 else np.nan,
            "Seen selling there (%)": float(active.get(item, np.nan)) * 100.0, "Confidence": str(s.at[item, "confidence"]) if "confidence" in s.columns else "",
            "Why it qualifies": why[item], "_id": item})
    if not rows:
        return pd.DataFrame(columns=COLUMNS + ["_id"])
    return pd.DataFrame(rows).sort_values("Profit after cut (g)", ascending=False).reset_index(drop=True)


def assumptions(fees: Dict[str, float] = FEES) -> list:
    """The footnote lines: what is assumed and subtracted."""
    return [
        f"Auction house cut on the Alliance and Horde houses: {fees['faction_cut']:.0%} of the sale price. On the neutral (Blackwater) house: {fees['neutral_cut']:.0%}. "
        "These are the Classic-era values, not yet checked on WoW Forever; every profit and margin on the House flips tab already has the cut of the house you resell on subtracted.",
        "Deposits are not subtracted: on Classic-era rules the deposit is returned when the item sells and lost only if the listing expires.",
        "Resell price is the lower of the resell house's lowest price now and its typical price over the last 72 hours. Units are capped at what that house sells in the "
        "chosen number of days (best estimate, a lower bound: listings posted and sold inside one hour are in no snapshot).",
        "Prices are from the latest hourly snapshot; listings may be gone by the time you look. Check in game before you buy.",
    ]
