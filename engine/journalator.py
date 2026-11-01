"""Reads the Journalator add-on's saved file (WTF/Account/<ACCOUNT>/SavedVariables/Journalator.lua) and turns the auction
logs into plain tables. No copy-and-paste export needed.

How the file is built (checked against a real file, 46 invoices / 37 postings / 11 failures, which matched the CSV exports):
  JOURNALATOR_ARCHIVE.stores.<kind>["Logs-<time>"].data is a string made by three layers:
    1. LibSerialize (a compact binary form of a Lua table, with back-references for repeated strings and tables)
    2. LibDeflate raw DEFLATE
    3. LibDeflate "EncodeForPrint" (6 bits per character, a-z A-Z 0-9 ( ) , lowest bits first)
  JOURNALATOR_LOGS holds recent entries that have not been archived yet; the add-on archives them at /reload or logout.

Every time is a Unix timestamp (UTC). For a SALE it is when you collected the mail, so it is an upper bound on when the sale
happened. For a purchase, a posting and an expiry notice it is when the add-on saw the event.
"""
from __future__ import annotations

import re
import struct
import zlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

COPPER_PER_GOLD = 10_000.0
_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()"
_VAL = {c: i for i, c in enumerate(_ALPHABET)}
AH_CUT = 0.05


# ------------------------------------------------------------------ decoding
def decode_print(s: str) -> bytes:
    """LibDeflate EncodeForPrint -> bytes."""
    bits = n = 0
    out = bytearray()
    for ch in s:
        bits |= _VAL[ch] << n
        n += 6
        while n >= 8:
            out.append(bits & 255)
            bits >>= 8
            n -= 8
    return bytes(out)


class _Reader:
    """LibSerialize reader. Type byte: xxxxxxx1 = 7 bit int; CCCC TT10 = embedded count (TT: 0 string, 1 map, 2 array, 3 mixed);
    NNNN S100 = 12 bit int (S = negative, N = the low 4 bits, one more byte holds the high 8); otherwise bits 3-7 pick the type (see `_decode_typed`)."""

    def __init__(self, b: bytes):
        if not b or b[0] != 1:
            raise ValueError("not a LibSerialize payload (version byte is not 1)")
        self.b, self.i = b, 1
        self.strs: List[str] = []
        self.tabs: list = []

    def byte(self) -> int:
        v = self.b[self.i]
        self.i += 1
        return v

    def num(self, k: int) -> int:
        v = int.from_bytes(self.b[self.i:self.i + k], "big")
        self.i += k
        return v

    def value(self):
        b = self.byte()
        if b & 1:
            return b >> 1
        if b & 3 == 2:
            c, tt = b >> 4, (b >> 2) & 3
            if tt == 0:
                return self.string(c)
            if tt == 1:
                return self.map(c)
            if tt == 2:
                return self.array(c)
            return self.mixed(c, self.byte())
        if b & 7 == 4:
            v = (b >> 4) | (self.byte() << 4)              # low 4 bits sit in the type byte, the other 8 bits follow
            return -v if (b >> 3) & 1 else v
        return self._decode_typed(b >> 3, b)

    def _decode_typed(self, idx: int, b: int):
        if idx == 0:
            return None
        if 1 <= idx <= 8:                                  # INT16/24/32/64, positive then negative
            size = {1: 2, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 8, 8: 8}[idx]
            v = self.num(size)
            return v if idx % 2 == 1 else -v
        if idx == 9:
            v = struct.unpack(">d", self.b[self.i:self.i + 8])[0]
            self.i += 8
            return v
        if idx in (10, 11):                                # float kept as text
            n = self.byte()
            v = float(self.b[self.i:self.i + n].decode())
            self.i += n
            return v if idx == 10 else -v
        if idx == 12:
            return True
        if idx == 13:
            return False
        if 14 <= idx <= 16:
            return self.string(self.num(idx - 13))
        if 17 <= idx <= 19:
            return self.map(self.num(idx - 16))
        if 20 <= idx <= 22:
            return self.array(self.num(idx - 19))
        if 23 <= idx <= 25:
            k = idx - 22
            a = self.num(k)
            return self.mixed(a, self.num(k))
        if 26 <= idx <= 28:
            return self.strs[self.num(idx - 25) - 1]
        if 29 <= idx <= 31:
            return self.tabs[self.num(idx - 28) - 1]
        raise ValueError(f"unknown LibSerialize type byte {b:#04x} at offset {self.i - 1}")

    def string(self, n: int) -> str:
        t = self.b[self.i:self.i + n].decode("utf-8", "replace")
        self.i += n
        if n > 2:                                          # only strings over 2 characters can be referred back to
            self.strs.append(t)
        return t

    def map(self, c: int) -> dict:
        t: dict = {}
        self.tabs.append(t)
        for _ in range(c):
            k = self.value()
            t[k] = self.value()
        return t

    def array(self, c: int) -> list:
        t: list = []
        self.tabs.append(t)
        for _ in range(c):
            t.append(self.value())
        return t

    def mixed(self, a: int, m: int) -> dict:
        t: dict = {}
        self.tabs.append(t)
        for i in range(a):
            t[i + 1] = self.value()
        for _ in range(m):
            k = self.value()
            t[k] = self.value()
        return t


def decode_payload(data: str):
    """One `data` string from the saved file -> the Lua table inside it (dicts and lists)."""
    raw = zlib.decompressobj(-15).decompress(decode_print(data))
    r = _Reader(raw)
    out = r.value()
    if r.i != len(raw):
        raise ValueError(f"decoded {r.i} of {len(raw)} bytes; the file may use a newer format")
    return out


_STORE = re.compile(r'\["Logs-(\d+)"\]\s*=\s*\{\s*\["timestamp"\]\s*=\s*(\d+),\s*\["version"\]\s*=\s*(\d+),\s*\["data"\]\s*=\s*"([^"]*)"', re.S)


def read_archives(text: str) -> List[dict]:
    """Every archived log block in the file: [{'archived': unix time, 'logs': {...}}, ...]."""
    out = []
    for m in _STORE.finditer(text):
        out.append({"archived": int(m.group(2)), "logs": decode_payload(m.group(4))})
    return out


def read_savedvars(path: str) -> dict:
    """Read the file. Returns {'Invoices': [...], 'Posting': [...], 'Failures': [...], ...} merged over all archive blocks
    (a record that appears in two blocks is kept once), plus the warnings to show the user in `warnings`."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    merged: Dict[str, list] = {}
    seen = set()
    for blk in read_archives(text):
        for kind, recs in blk["logs"].items():
            if not isinstance(recs, list):
                continue
            for rec in recs:
                key = (kind, repr(sorted(rec.items(), key=lambda kv: kv[0])) if isinstance(rec, dict) else repr(rec))
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(kind, []).append(rec)
    warnings = []
    if not merged:
        warnings.append("no archived logs found in the file. Type /reload in game (the add-on archives its logs then) and send the file again.")
    if re.search(r"JOURNALATOR_LOGS\s*=\s*\{", text):
        warnings.append("the file still holds recent logs that are not archived yet; they were NOT read. Type /reload in game and save the file again.")
    merged["warnings"] = warnings
    return merged


# ------------------------------------------------------------------ tables
_ITEM_ID = re.compile(r"Hitem:(\d+)")


def _item_id(link) -> float:
    m = _ITEM_ID.search(str(link or ""))
    return float(m.group(1)) if m else np.nan


def _to_time(unix) -> pd.Series:
    return pd.to_datetime(pd.Series(unix, dtype="float64"), unit="s", utc=True)


def tables(logs: dict) -> Dict[str, pd.DataFrame]:
    """Clean tables with item ids and gold. Times are UTC (`*_utc`).

    purchases  what you bought: item_id, name, units, unit_gold, total_gold
    sales      what sold: units, unit_gold (the price you listed at), gross_gold, cut_gold (the 5%), deposit_back_gold, net_gold
               (= gross - cut; the deposit is returned separately), collected_utc (when you opened the mail)
    postings   what you listed: units, unit_gold, deposit_gold
    failures   expired / cancelled listings
    """
    inv = pd.DataFrame(logs.get("Invoices", []))
    post = pd.DataFrame(logs.get("Posting", []))
    fail = pd.DataFrame(logs.get("Failures", []))
    out: Dict[str, pd.DataFrame] = {}
    if inv.empty:
        out["purchases"] = out["sales"] = pd.DataFrame()
    else:
        for c in ("playerName", "consignment", "deposit"):
            if c not in inv:
                inv[c] = "" if c == "playerName" else 0
        inv["item_id"] = inv["itemLink"].map(_item_id)
        inv["time_utc"] = _to_time(inv["time"])
        buy = inv[inv["invoiceType"] == "buyer"]
        sell = inv[inv["invoiceType"] == "seller"]
        out["purchases"] = pd.DataFrame({
            "time_utc": buy["time_utc"], "item_id": buy["item_id"], "name": buy["itemName"], "units": buy["count"],
            "unit_gold": buy["value"] / buy["count"] / COPPER_PER_GOLD, "total_gold": buy["value"] / COPPER_PER_GOLD,
            "seller": buy["playerName"]}).reset_index(drop=True)
        out["sales"] = pd.DataFrame({
            "collected_utc": sell["time_utc"], "item_id": sell["item_id"], "name": sell["itemName"], "units": sell["count"],
            "unit_gold": sell["value"] / sell["count"] / COPPER_PER_GOLD, "gross_gold": sell["value"] / COPPER_PER_GOLD,
            "cut_gold": sell["consignment"] / COPPER_PER_GOLD, "deposit_back_gold": sell["deposit"] / COPPER_PER_GOLD,
            "net_gold": (sell["value"] - sell["consignment"]) / COPPER_PER_GOLD, "buyer": sell["playerName"]}).reset_index(drop=True)
    if post.empty:
        out["postings"] = pd.DataFrame()
    else:
        out["postings"] = pd.DataFrame({
            "time_utc": _to_time(post["time"]), "item_id": post["itemLink"].map(_item_id), "name": post["itemName"], "units": post["count"],
            "unit_gold": post["buyout"] / COPPER_PER_GOLD, "deposit_gold": post["deposit"] / COPPER_PER_GOLD})
    if fail.empty:
        out["failures"] = pd.DataFrame()
    else:
        out["failures"] = pd.DataFrame({
            "time_utc": _to_time(fail["time"]), "item_id": fail["itemLink"].map(_item_id), "name": fail["itemName"], "units": fail["count"],
            "kind": fail["failedType"]})
    # Sale invoices carry no item link, only a name. Fill in the item id where the name belongs to exactly one item.
    ids = pd.concat([out[k][["name", "item_id"]] for k in ("purchases", "postings", "failures") if len(out.get(k, []))], ignore_index=True).dropna()
    one = ids.groupby("name")["item_id"].nunique()
    name_to_id = ids[ids["name"].isin(one[one == 1].index)].drop_duplicates("name").set_index("name")["item_id"]
    if len(out.get("sales", [])):
        out["sales"]["item_id"] = out["sales"]["name"].map(name_to_id)
    for k, v in out.items():
        if len(v):
            tcol = "collected_utc" if k == "sales" else "time_utc"
            out[k] = v.sort_values(tcol).reset_index(drop=True)
    return out


def match_listings(t: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per posting: how many units sold, expired or are still open, and what it cost.

    A sale (which carries only the item NAME, not its id) belongs to the oldest posting with that name at the same price that still
    has units left. An expiry belongs to the
    oldest posting of that item with units left. Sales are dated by when you collected the mail, so `hours_to_collect` is an upper
    bound on the real time to sell."""
    p = t.get("postings", pd.DataFrame())
    if p.empty:
        return pd.DataFrame()
    p = p.sort_values("time_utc").reset_index(drop=True).copy()
    p["units_sold"], p["units_expired"] = 0, 0
    p["net_gold"], p["deposit_back_gold"] = 0.0, 0.0
    first = [None] * len(p)
    left = p["units"].astype(int).tolist()
    s = t.get("sales", pd.DataFrame())
    for _, r in s.iterrows():
        need = int(r["units"])
        per_unit_net = r["net_gold"] / r["units"]
        per_unit_dep = r["deposit_back_gold"] / r["units"]
        for i in p.index[(p["name"] == r["name"]) & (p["time_utc"] <= r["collected_utc"]) & ((p["unit_gold"] - r["unit_gold"]).abs() < 5e-5)]:
            if need <= 0:
                break
            take = min(need, left[i])
            if take <= 0:
                continue
            left[i] -= take
            need -= take
            p.at[i, "units_sold"] += take
            p.at[i, "net_gold"] += take * per_unit_net
            p.at[i, "deposit_back_gold"] += take * per_unit_dep
            if first[i] is None:
                first[i] = r["collected_utc"]
    f = t.get("failures", pd.DataFrame())
    for _, r in f.iterrows():
        if r.get("kind") not in ("expired", "cancelled", "canceled"):
            continue
        need = int(r["units"])
        for i in p.index[(p["item_id"] == r["item_id"]) & (p["time_utc"] <= r["time_utc"])]:
            if need <= 0:
                break
            take = min(need, left[i])
            if take <= 0:
                continue
            left[i] -= take
            need -= take
            p.at[i, "units_expired"] += take
    p["units_open"] = left
    p["first_collected_utc"] = pd.to_datetime(pd.Series(first, index=p.index, dtype=object), utc=True)
    p["deposit_lost_gold"] = p["deposit_gold"] * p["units_expired"] / p["units"].clip(lower=1)
    p["hours_to_collect"] = (p["first_collected_utc"] - p["time_utc"]).dt.total_seconds() / 3600.0
    p["status"] = np.where(p["units_open"] == 0, np.where(p["units_sold"] > 0, np.where(p["units_expired"] > 0, "part sold, rest expired", "sold"), "expired"),
                           np.where(p["units_sold"] > 0, "part sold, rest open", "open"))
    return p


def flip_ledger(t: Dict[str, pd.DataFrame], listings: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per item you bought: what the units that SOLD returned after the 5% cut against what they cost (sales are matched to your listings made after the first purchase), what is still
    unsold (at cost), and the deposits. Only listings made after your first purchase of the item count, so units you already had and
    sold earlier are not mixed in."""
    b = t.get("purchases", pd.DataFrame())
    if b.empty:
        return pd.DataFrame()
    lst = listings if listings is not None else match_listings(t)
    rows = []
    for item_id, g in b.groupby("item_id"):
        t0 = g["time_utc"].min()
        bought, cost = float(g["units"].sum()), float(g["total_gold"].sum())
        avg = cost / bought
        ll = lst[(lst["item_id"] == item_id) & (lst["time_utc"] >= t0)] if len(lst) else lst
        sold = float(ll["units_sold"].sum()) if len(ll) else 0.0
        net = float(ll["net_gold"].sum()) if len(ll) else 0.0
        sold_cost = avg * sold
        rows.append(dict(
            item_id=int(item_id), name=g["name"].iloc[0], first_buy_utc=t0, bought=bought, avg_cost_gold=avg,
            sold=sold, unsold=bought - sold, listed_at_gold=(float(ll["unit_gold"].iloc[-1]) if len(ll) else np.nan),
            net_return_gold=net, cost_of_sold_gold=sold_cost, profit_on_sold_gold=net - sold_cost,
            margin_on_sold=(net / sold_cost - 1.0) if sold_cost > 0 else np.nan, unsold_at_cost_gold=cost - sold_cost,
            deposit_paid_gold=float(ll["deposit_gold"].sum()) if len(ll) else 0.0,
            deposit_back_gold=float(ll["deposit_back_gold"].sum()) if len(ll) else 0.0,
            deposit_lost_gold=float(ll["deposit_lost_gold"].sum()) if len(ll) else 0.0,
            hours_to_collect=float(ll["hours_to_collect"].min()) if len(ll) and ll["hours_to_collect"].notna().any() else np.nan))
    return pd.DataFrame(rows).sort_values("first_buy_utc").reset_index(drop=True)
