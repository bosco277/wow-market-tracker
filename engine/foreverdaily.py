"""Forever daily tables: ONE ROW PER AUCTION HOUSE, ITEM AND US EASTERN DAY, built from the hourly commodity files.

Why
  The hourly files are a raw record (thousands of listings per house per hour). Nothing can read them quickly, and they will have to be compacted before the free storage runs out. This
  module boils each day down to one small row per house and item. It is the table the app's Forever view, the buy-below prices and the crafting margins read.

What a row says (all prices in gold per unit; a "usable" hour is one with a good price reading: no price spike, no listing blip, not a wipe or a long gap)
  house, kind      the market (110 = Alliance PvP, 116 = Neutral PvP, see tables/forever_house_map.json); kind is 'commodity' (regular auctions are a later step)
  item_id, day     the item and the US Eastern day (YYYY-MM-DD)
  hours            usable hours that day. A day with fewer than 18 is stored but should not be trusted as a full day.
  flagged_hours    hours left out because they were flagged
  price_day        the day's typical lowest price (median of the usable hours' lowest price)
  price_min        the lowest price seen in any usable hour
  price_p15/p30    typical price at which the cheapest 15% / 30% of the units for sale have all been reached (one cheap listing cannot set it); the retail table has these too
  buy20, buy200    typical AVERAGE price per unit to buy 20 / 200 units, walking up the listings from the cheapest. buy20 is the median over the usable hours that had 20 units for sale.
                   buy200 is buy20 times the median, over the hours that had 200 units, of that hour's buy200 / buy20 (the extra cost of going deeper), so it can never be lower than buy20
                   (v0.7.48: a plain median over only the deep hours could be, because those hours were the deeper, cheaper ones). Blank when fewer than half the usable hours had that many units.
  units_day        typical units for sale
  listings_day     typical number of listings
  gold_day         typical gold in play: price x quantity summed over all listings
  sold_cert_day    units per 24 hours that CERTAINLY sold: a listing that is still standing with fewer units (only a purchase does that)
  sold_low_day     conservative: the certain sales plus disappearances that could not have been expiries, less an allowance for cancel-and-relist
  sold_day         the point estimate (the classifier's cancel share is an uncalibrated 10%)
  sold_high_day    optimistic: every disappearance consistent with a purchase
  turnover_day     sold_low_day / units_day: the share of the standing stock that was seen selling per day (a lower bound)
  active_share     the share of usable hours in which at least some sale was seen (sold_low above 0): how often the market visibly moves
  engine_version   the engine that computed the row

THE SALES COLUMNS ARE LOWER BOUNDS ON TRUE VOLUME. Blizzard refreshes every house once an hour (at about :22) and nothing faster is published, so a listing posted and sold inside one hour is never in any
snapshot. The player reports that most Forever sales happen within minutes of posting; if that holds, most of the volume is invisible (to be measured, see below). What the snapshots do show is a sample of it (a listing posted shortly before a snapshot and bought
shortly after it, and everything that stands for a while), so the columns are good for COMPARING items and days and for spotting a market that is speeding up or slowing down, but not for the true number.
Scaling them to true volume needs ground truth (your own sales and posting times); that check is a separate step. The classifier is the retail one: its expiry logic uses Classic's time-left buckets
(which fit what we saw on Nov 13), but its cancel share and blind-spot allowance are not calibrated on Forever.

Same shape as the retail `item_daily` table (engine/daily.py) with the house in front and the extra columns after it, so retail tools can read one house's slice unchanged.

Stored in the Forever dataset as ONE FILE PER DAY, all houses in it: daily/d_YYYY-MM-DD.csv.gz (gzip CSV with no timestamp in the header, so an unchanged day is recognised as unchanged and
not committed again). A day is merged with what is stored: for the same house, item and day the row with MORE usable hours wins (a partial day never shrinks a fuller one; a tie goes to the
new row). `rebuild` makes the new row win outright (after a change to how rows are computed).

Each house is computed on its own, never pooled. The first and last 12 hours of the loaded window are left out of every row (the spike and blip flags look 12 hours either side of an hour), so a
run that loads days D-3 .. D completes day D-3 .. D-1 and leaves the newest hours to the next run.
"""
from __future__ import annotations

import gzip
import io
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import __version__
from . import daily as D
from . import forever as FV
from .classifier import classify_series
from .schema import COPPER_PER_GOLD, Snapshot

KIND = "commodity"
DAILY_DIR = "daily"
DAY_KEY_RE = re.compile(r"^daily/d_(\d{4}-\d{2}-\d{2})\.csv\.gz$")
TZ = "America/New_York"
BUY_SIZES = (20, 200)
COMPLETE_HOURS = 18                      # same meaning as the retail history: fewer usable hours is a partial day
KEY_COLUMNS = ["house", "kind", "item_id", "day"]
COLUMNS = ["house", "kind", "item_id", "day", "hours", "flagged_hours", "price_day", "price_min", "price_p15", "price_p30", "buy20", "buy200",
           "units_day", "listings_day", "gold_day", "sold_cert_day", "sold_low_day", "sold_day", "sold_high_day", "turnover_day", "active_share", "engine_version"]
INT_COLUMNS = ("house", "item_id", "hours", "flagged_hours")
TEXT_COLUMNS = ("kind", "day", "engine_version")
ROUND = {"price_day": 4, "price_min": 4, "price_p15": 4, "price_p30": 4, "buy20": 4, "buy200": 4, "units_day": 1, "listings_day": 1, "gold_day": 2,
         "sold_cert_day": 2, "sold_low_day": 2, "sold_day": 2, "sold_high_day": 2, "turnover_day": 4, "active_share": 3}
EXTRA = {"price_min": ("floor_gold", "min"), "listings_day": ("n_listings", "median"), "gold_day": ("gold", "median"),
         "buy20": ("buy20", "median"),
         "sold_cert_day": ("sold_cert", "mean"), "sold_low_day": ("sold_low", "mean"), "sold_high_day": ("sold_high", "mean"), "active_share": ("active", "mean")}
DEEP_MIN_SHARE = 0.5          # a deeper buy price needs that depth in at least half of the usable hours


def _deep_extra() -> dict:
    """Hourly helper columns for the deeper buy sizes: the depth ratio (median) and whether the depth was there (mean = share of hours)."""
    out = {}
    for n in BUY_SIZES[1:]:
        out[f"_ratio{n}"] = (f"ratio{n}", "median")
        out[f"_has{n}"] = (f"has{n}", "mean")
    return out


PER_DAY = ("sold_cert_day", "sold_low_day", "sold_high_day")        # a mean per hourly interval, scaled to 24 hours like the retail sold_day


# ---------------------------------------------------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------------------------------------------------
def day_key(day: str) -> str:
    return f"{DAILY_DIR}/d_{day}.csv.gz"


def parse_day_key(key: str) -> Optional[str]:
    m = DAY_KEY_RE.match(str(key))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------------------------------------------------
# One house: snapshots -> rows
# ---------------------------------------------------------------------------------------------------------------------
def book_stats(snaps: Sequence[Snapshot], ids: Iterable[int], sizes: Sequence[int] = BUY_SIZES) -> pd.DataFrame:
    """Per item per snapshot: item_id, ts, gold (price x quantity summed over every listing, in gold) and buy<N> for each size (the average price per unit, in gold, of the cheapest N units:
    walk up the listings from the cheapest; NaN when fewer than N units are for sale)."""
    ids = {int(i) for i in ids}
    cols = ["item_id", "ts", "gold"] + [f"buy{n}" for n in sizes]
    parts = []
    for s in snaps:
        b = s.book[["item_id", "unit_price", "quantity"]]
        b = b[b["item_id"].isin(ids) & (b["quantity"] > 0)]
        if b.empty:
            continue
        b = b.assign(unit_price=b["unit_price"].astype("float64"), quantity=b["quantity"].astype("float64"))
        b = b.sort_values(["item_id", "unit_price"], kind="mergesort").reset_index(drop=True)
        b["cost"] = b["unit_price"] * b["quantity"]
        grp = b.groupby("item_id", sort=True)
        b["cum_q"] = grp["quantity"].cumsum()
        b["cum_c"] = grp["cost"].cumsum()
        out = pd.DataFrame({"gold": grp["cost"].sum() / COPPER_PER_GOLD})
        for n in sizes:
            first = b[b["cum_q"] >= n - 1e-9].groupby("item_id", sort=True).head(1).set_index("item_id")
            before_q = first["cum_q"] - first["quantity"]
            before_c = first["cum_c"] - first["cost"]
            out[f"buy{n}"] = ((before_c + (n - before_q) * first["unit_price"]) / n) / COPPER_PER_GOLD
        out = out.reset_index()
        out["ts"] = s.ts
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


def _commodity_snaps(snaps: Sequence[Snapshot], house: int) -> List[Snapshot]:
    snaps = sorted((s for s in snaps if s.kind == "commodity"), key=lambda s: s.ts)
    for s in snaps:
        p = s.meta.get("house")
        if p is not None and int(p) != int(house):
            raise ValueError(f"snapshot {s.source!r} belongs to house {p}, not {house}: markets are never mixed")
    return snaps


def build_house(snaps: Sequence[Snapshot], house: int, ids: Optional[Iterable[int]] = None, sales: bool = True) -> pd.DataFrame:
    """The daily rows for ONE house's commodity snapshots (every day the window touches; the first and last 12 hours of the window are left out).

    sales=True (default) runs the sales classifier and fills the sales columns (sold_cert_day, sold_low_day, sold_day, sold_high_day, turnover_day, active_share). They are LOWER BOUNDS on true volume
    (see the module note). sales=False skips the classifier: prices and depth are complete, the sales columns are blank, and every hour with a price reading counts as usable.
    Fewer than 3 commodity snapshots give an empty table."""
    snaps = _commodity_snaps(snaps, house)
    if len(snaps) < 3:
        return tidy(None)
    if ids is None:
        ids = set()
        for s in snaps:
            ids.update(int(i) for i in s.book["item_id"].unique())
    ids = {int(i) for i in ids}
    if not ids:
        return tidy(None)
    intervals = classify_series(snaps) if sales else FV._no_intervals()
    h = D.build_hourly(snaps, intervals, ids)
    if h is None or h.empty:
        return tidy(None)
    if sales and not intervals.empty:
        iv = intervals[["item_id", "t1", "sold_cert", "sold_low", "sold_high"]].rename(columns={"t1": "ts"})
        h = h.merge(iv, on=["item_id", "ts"], how="left")
        h["active"] = np.where(h["sold_low"].notna(), (h["sold_low"] > 0).astype(float), np.nan)
    stats = book_stats(snaps, ids)
    if not stats.empty:
        h = h.merge(stats, on=["item_id", "ts"], how="left")
    base = f"buy{BUY_SIZES[0]}"
    for n in BUY_SIZES[1:]:
        col = f"buy{n}"
        if col in h.columns and base in h.columns:
            h[f"ratio{n}"] = h[col] / h[base]
            h[f"has{n}"] = h[col].notna().astype(float)
    if not sales:
        h["sold_est"] = 0.0                      # every hour with a price counts; the sales columns are blanked below
    h = D.trim_edges(h, snaps[0].ts, snaps[-1].ts)
    if h.empty:
        return tidy(None)
    d = D.daily_table(h, TZ, extra={**EXTRA, **_deep_extra()})
    if d.empty:
        return tidy(None)
    for n in BUY_SIZES[1:]:
        ok = d[f"_has{n}"].fillna(0) >= DEEP_MIN_SHARE
        d[f"buy{n}"] = np.where(ok, d[f"buy{BUY_SIZES[0]}"] * d[f"_ratio{n}"], np.nan)
        d = d.drop(columns=[f"_ratio{n}", f"_has{n}"])
    if sales:
        for c in PER_DAY:
            d[c] = d[c] * 24.0
        d["turnover_day"] = np.where(d["units_day"] > 0, d["sold_low_day"] / d["units_day"].where(d["units_day"] > 0), np.nan)
    else:
        for c in ("sold_day", "turnover_day") + PER_DAY + ("active_share",):
            d[c] = np.nan
    d.insert(0, "kind", KIND)
    d.insert(0, "house", int(house))
    return tidy(d)


def snapshot_days(snaps: Sequence[Snapshot]) -> Dict[str, int]:
    """{US Eastern day: how many snapshots fall on it} (for the report: a half-empty day shows here)."""
    out: Dict[str, int] = {}
    for s in snaps:
        t = pd.Timestamp(s.ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        day = t.tz_convert(TZ).strftime("%Y-%m-%d")
        out[day] = out.get(day, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------------------------------------------------
# Stored form
# ---------------------------------------------------------------------------------------------------------------------
def tidy(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Exactly COLUMNS in order, integer columns as int, prices rounded, sorted by house, kind, item, day."""
    if df is None or len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c in INT_COLUMNS else "object" if c in TEXT_COLUMNS else "float64") for c in COLUMNS})
    out = df.reindex(columns=COLUMNS).copy()
    out["kind"] = out["kind"].fillna(KIND).astype(str)
    out["engine_version"] = out["engine_version"].fillna("").astype(str)
    out["day"] = out["day"].astype(str)
    for c in INT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).round().astype("int64")
    for c, n in ROUND.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64").round(n)
    return out.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def to_csv_gz(df: pd.DataFrame) -> bytes:
    """Same rows -> same bytes (no timestamp in the gzip header), so an unchanged day is recognised as unchanged."""
    raw = tidy(df).to_csv(index=False, lineterminator="\n").encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as f:
        f.write(raw)
    return buf.getvalue()


def from_csv_gz(data: Optional[bytes]) -> pd.DataFrame:
    """A stored day back as a tidy frame; empty when there is nothing (or the bytes are not a table)."""
    if not data:
        return tidy(None)
    try:
        df = pd.read_csv(io.BytesIO(data), compression="gzip", dtype={"kind": "object", "day": "object", "engine_version": "object"})
    except Exception:
        return tidy(None)
    return tidy(df)


def split_days(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """{day: that day's rows}."""
    if df is None or df.empty:
        return {}
    return {d: g.reset_index(drop=True) for d, g in df.groupby("day", sort=True)}


def merge_days(old: Optional[pd.DataFrame], new: Optional[pd.DataFrame], rebuild: bool = False) -> pd.DataFrame:
    """Stored rows plus new ones. Same house, kind, item and day: the row with MORE usable hours wins and a tie goes to the new row; rebuild=True lets the new row win outright.
    Rows only one side has are kept (a run for one house leaves the other houses' rows alone)."""
    parts = []
    if old is not None and len(old):
        parts.append(tidy(old).assign(_src=0))
    if new is not None and len(new):
        parts.append(tidy(new).assign(_src=1))
    if not parts:
        return tidy(None)
    m = pd.concat(parts, ignore_index=True)
    order = KEY_COLUMNS + ["_src"] if rebuild else KEY_COLUMNS + ["hours", "_src"]
    m = m.sort_values(order, kind="mergesort").drop_duplicates(KEY_COLUMNS, keep="last")
    return tidy(m.drop(columns="_src"))


def plan_writes(new: pd.DataFrame, existing: Dict[str, bytes], rebuild: bool = False) -> Tuple[Dict[str, bytes], List[dict]]:
    """For every day in `new`: merge it with the stored day (existing = {key: bytes} of the stored files) and say what happens.
    Returns ({key: bytes} of the files that are new or changed, one dict per day: day, key, status new / changed / unchanged, rows, rows_before, bytes)."""
    files: Dict[str, bytes] = {}
    plan: List[dict] = []
    for day, rows in split_days(new).items():
        key = day_key(day)
        before = existing.get(key)
        old = from_csv_gz(before)
        merged = merge_days(old, rows, rebuild)
        data = to_csv_gz(merged)
        if before is None:
            status = "new"
        elif data == before:
            status = "unchanged"
        else:
            status = "changed"
        if status != "unchanged":
            files[key] = data
        plan.append(dict(day=day, key=key, status=status, rows=len(merged), rows_before=len(old), bytes=len(data)))
    return files, plan


# ---------------------------------------------------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------------------------------------------------
def day_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per day: houses, item-day rows, rows with a complete day (COMPLETE_HOURS or more usable hours)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["day", "houses", "rows", "complete_rows"])
    g = df.groupby("day", sort=True)
    return pd.DataFrame({"houses": g["house"].nunique(), "rows": g.size(), "complete_rows": g["hours"].apply(lambda x: int((x >= COMPLETE_HOURS).sum()))}).reset_index()


def craft_coverage(rows: pd.DataFrame, recipes: pd.DataFrame, reagents: pd.DataFrame) -> Dict[int, dict]:
    """Per house, for one day's rows: of the recipes that name a product and list reagents, how many have every reagent priced, the product priced, and both.
    A reagent that only a vendor sells has no auction price, so 'reagents priced' is a floor."""
    out: Dict[int, dict] = {}
    if rows is None or rows.empty or recipes is None or recipes.empty or reagents is None or reagents.empty:
        return out
    prod = recipes[recipes["crafted_item_id"].notna()][["recipe_id", "crafted_item_id"]]
    rg = reagents[["recipe_id", "reagent_item_id"]]
    base = set(prod["recipe_id"].astype(int)) & set(rg["recipe_id"].astype(int))
    for house, hr in rows.groupby("house", sort=True):
        priced = {int(i) for i in hr.loc[hr["price_day"].notna() & (hr["price_day"] > 0), "item_id"]}
        ok = rg.assign(ok=rg["reagent_item_id"].isin(priced).astype(int)).groupby("recipe_id")["ok"].min()
        reag_ok = {int(r) for r in ok.index[ok == 1]} & base
        prod_ok = {int(r) for r in prod.loc[prod["crafted_item_id"].isin(priced), "recipe_id"]} & base
        out[int(house)] = dict(recipes=len(base), reagents_priced=len(reag_ok), product_priced=len(prod_ok), both=len(reag_ok & prod_ok))
    return out
