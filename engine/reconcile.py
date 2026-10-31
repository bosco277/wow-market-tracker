"""Check the buy-and-list experiment log against the hourly snapshots.

The log (one row per listing: when you posted it, price, quantity, units sold, units expired) is the one place we know
what really sold. For each listing this module asks what the snapshots show for the same price, and answers three questions:

1. DETECTION   Did the snapshot right after posting show about `quantity` new units at that price? If you sold the whole
               listing before that snapshot it shows nothing: that is the blind spot ("invisible"), counted directly.
2. TIMING      For a listing that was the CHEAPEST in the book with nothing else at its price, how many hours until
               nothing at or below that price was left, against the hours the classifier's recent sales rate predicts?
               The snapshots are hourly, so the observed time is a window: still there at one snapshot, gone at the next.
               Listings that share a price with older units are NOT used here: other sellers' units arriving and leaving
               at the same price swamp the signal (the first real run showed negative "seen" shares and clearing times of
               20 minutes for listings that sold nothing). Queue order (newest-first or not) cannot be read from this log;
               the `order` check on the book history does that.
3. TOTALS      Units posted, sold, expired. Listings where the log and the snapshots disagree are listed one by one.

Honest limits: other sellers also post and cancel at your price while you wait, which blurs the "units fell back" test, so
an observed time is approximate (the report says which snapshot it came from). Nothing here changes the classifier; it only
produces the numbers we use to decide whether to.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .market import item_series
from .schema import COPPER_PER_GOLD, Snapshot

SILVER = COPPER_PER_GOLD // 100
UTC_OFFSET_HOURS = {"EDT": -4, "EST": -5, "UTC": 0, "GMT": 0}
LOG_COLUMNS = ["date", "time_posted", "tz", "item_id", "lane", "list_price", "quantity", "units_sold", "units_expired", "note"]


ITEM_IDS_BY_NAME = {
    "iceweb spider silk": 42253, "mystic sapphire": 192841, "profaned tinderbox": 221758, "oil of dawn": 243735,
    "denshroom deep roast": 260274, "laribole": 154124, "refulgent crystal": 219952, "eternal earth": 35624,
    "algari healing potion": 211880, "swiftness potion": 2459, "invigorating healing potion": 244838,
    "primordial saronite": 49908, "algari mana oil": 224107, "dawnlight": 130180, "sorrowmoss": 13466,
    "eternal crystal": 172232, "linen cloth": 2589,
}
COLUMN_ALIASES = {"note 1": "note", "notes": "note", "note1": "note", "comment": "note", "comments": "note"}
LANE_WORDS = {"UNDER": "UNDER", "BELOW": "UNDER", "SAME": "SAME", "ABOVE": "SAME"}


def read_log(source) -> pd.DataFrame:
    """Read the log (a path, comma or tab separated, or a DataFrame). Adds posted_utc (naive UTC, like the snapshots),
    price_copper, abnormal, done. Tolerates what a pasted sheet looks like: a "Note 1" header, an extra item-name column,
    times like "8:22 AM", lane words (below = UNDER, above = SAME; anything else = OTHER).

    Results: if only units_expired is filled in, units_sold = quantity - units_expired; if units_sold equals the quantity,
    units_expired = 0. A partial count with "as of" in the note (7 of 30 "as of 3 PM") is IN PROGRESS, not a result.
    Example rows (note starts with 'example row') are dropped, and so are rows without a quantity or price."""
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, sep=None, engine="python")
    raw.columns = [COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in raw.columns]
    if "note" not in raw.columns:
        raw["note"] = ""
    missing = [c for c in LOG_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"log is missing columns: {missing}")
    df = raw[LOG_COLUMNS].copy()
    df["item_name"] = raw["item"] if "item" in raw.columns else ""
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df[~df["note"].str.lower().str.startswith("example row")]
    df = df[(df["quantity"] != "") & (df["list_price"] != "")].copy()
    # item_id can arrive as "#N/A" from a failed sheet lookup: fall back to the item name
    ids = pd.to_numeric(df["item_id"], errors="coerce")
    by_name = df["item_name"].str.lower().map(ITEM_IDS_BY_NAME)
    ids = ids.fillna(by_name)
    if ids.isna().any():
        bad = sorted(set(df.loc[ids.isna(), "item_name"].replace("", "(no name)")) | set(df.loc[ids.isna(), "item_id"]))
        raise ValueError(f"cannot work out the item id for: {bad}; fill in item_id")
    df["item_id"] = ids.astype("int64")
    df["quantity"] = pd.to_numeric(df["quantity"])
    df["list_price"] = pd.to_numeric(df["list_price"])
    for c in ("units_sold", "units_expired"):
        df[c] = pd.to_numeric(df[c].replace("", np.nan))
    off = df["tz"].str.upper().map(UTC_OFFSET_HOURS)
    if off.isna().any():
        bad = sorted(df.loc[off.isna(), "tz"].unique())
        raise ValueError(f"unknown time zone label(s) in the log: {bad}; use EDT or EST")
    local = pd.to_datetime(df["date"] + " " + df["time_posted"])
    df["posted_utc"] = local - pd.to_timedelta(off.astype(float), unit="h")
    df["price_copper"] = (df["list_price"] * COPPER_PER_GOLD).round().astype("int64")
    df["lane"] = df["lane"].str.upper().map(LANE_WORDS).fillna("OTHER")
    df["abnormal"] = df["note"].str.lower().str.contains("abnormal")
    as_of = df["note"].str.lower().str.contains("as of")
    sold, exp, q = df["units_sold"], df["units_expired"], df["quantity"]
    df["units_sold"] = sold.where(sold.notna(), (q - exp).where(exp.notna()))       # only expired filled in: the rest sold
    df["units_expired"] = exp.where(exp.notna(), pd.Series(0.0, index=df.index).where(df["units_sold"] >= q))
    complete = (df["units_sold"] + df["units_expired"]) >= q - 0.5          # sold + expired accounts for every unit: final
    in_progress = as_of & (df["units_sold"] < q) & ~complete
    df.loc[in_progress, ["units_sold", "units_expired"]] = np.nan
    df["done"] = df["units_sold"].notna()
    df["in_progress"] = in_progress
    df = df.sort_values("posted_utc").reset_index(drop=True)
    df.attrs["warnings"] = check_log(df)
    return df


def check_log(df: pd.DataFrame, cluster_minutes: float = 30.0, max_spread: float = 0.10) -> List[str]:
    """Things in the log that look like typos. Listings of one item posted within `cluster_minutes` of each other are one
    batch (the UNDER and SAME rows); their prices should be within `max_spread` of each other."""
    out = []
    for item, g in df.groupby("item_id"):
        g = g.sort_values("posted_utc")
        cluster = (g["posted_utc"].diff() > pd.Timedelta(minutes=cluster_minutes)).cumsum()
        for _, c in g.groupby(cluster):
            if len(c) > 1 and c["price_copper"].max() > c["price_copper"].min() * (1 + max_spread):
                when = c["posted_utc"].min()
                out.append(f"item {int(item)}: listings posted together around {when:%Y-%m-%d %H:%M} UTC have prices "
                           f"{c['list_price'].min():g} to {c['list_price'].max():g}; check the time or price of one of them")
    return out


def _item_books(snapshots: Sequence[Snapshot], item_ids) -> Dict[int, List[tuple]]:
    """item -> sorted list of (ts, prices array, quantities array)."""
    out: Dict[int, List[tuple]] = {int(i): [] for i in item_ids}
    for s in sorted(snapshots, key=lambda x: x.ts):
        if s.kind != "commodity":
            continue
        b = s.book[s.book["item_id"].isin(list(out))]
        for i, g in b.groupby("item_id"):
            out[int(i)].append((s.ts, g["unit_price"].to_numpy(dtype="int64"), g["quantity"].to_numpy(dtype="int64")))
    return out


def _units(prices: np.ndarray, qty: np.ndarray, p: int) -> tuple:
    """(units strictly below p, units exactly at p)"""
    return int(qty[prices < p].sum()), int(qty[prices == p].sum())


def _velocity(series: Optional[pd.DataFrame], t0: pd.Timestamp, hours: float = 24.0, min_rows: int = 6) -> dict:
    """Mean units per hour the classifier credits over the `hours` before t0 (low / est / high), flagged intervals left out."""
    nan = dict(low=np.nan, est=np.nan, high=np.nan)
    if series is None or series.empty or "sold_est" not in series.columns:
        return nan
    s = series[(series["ts"] <= t0) & (series["ts"] > t0 - pd.Timedelta(hours=hours))]
    for c in ("flag_wipe", "flag_long_gap"):
        if c in s.columns:
            s = s[~s[c].fillna(False).astype(bool)]
    s = s.dropna(subset=["sold_est"])
    if len(s) < min_rows:
        return nan
    gaps = s["gap_minutes"] / 60.0 if "gap_minutes" in s.columns else pd.Series(1.0, index=s.index)
    span = float(gaps.sum()) or float(len(s))
    return dict(low=float(s["sold_low"].sum() / span), est=float(s["sold_est"].sum() / span), high=float(s["sold_high"].sum() / span))


def _hours_to_sell(ahead: float, own: float, per_hour: float) -> float:
    return float((ahead + own) / per_hour) if per_hour and per_hour > 0 and per_hour == per_hour else float("nan")


def trace_listing(row: pd.Series, books: List[tuple], partner_qty: float, vel: dict, duration_hours: float = 12.0,
                  gone_share: float = 0.10, max_gap_hours: float = 1.5) -> dict:
    """What the snapshots show for ONE log row. `partner_qty` is your own cheaper listings posted with it (they are ahead of
    you in the queue and also counted in the 'at or below' total)."""
    p, q, t0 = int(row["price_copper"]), float(row["quantity"]), pd.Timestamp(row["posted_utc"])
    out = dict(item_id=int(row["item_id"]), lane=row["lane"], posted_utc=t0, price_gold=p / COPPER_PER_GOLD, quantity=q,
               units_sold=row["units_sold"], units_expired=row["units_expired"], abnormal=bool(row["abnormal"]), done=bool(row["done"]))
    before = [b for b in books if b[0] <= t0]
    after = [b for b in books if t0 < b[0] <= t0 + pd.Timedelta(hours=duration_hours + 1.0)]
    if not before or not after:
        out["status"] = "no snapshots around the posting time"
        return out
    b0 = before[-1]
    a1 = after[0]
    out["gap_before_h"] = (t0 - b0[0]).total_seconds() / 3600.0
    out["gap_after_h"] = (a1[0] - t0).total_seconds() / 3600.0
    if out["gap_before_h"] > max_gap_hours or out["gap_after_h"] > max_gap_hours:
        out["status"] = "snapshot too far from the posting time (a hole)"
        return out
    out["vs_floor_silver"] = (p - int(b0[1].min())) / SILVER      # price against the lowest price in the snapshot before posting
    below0, at0 = _units(b0[1], b0[2], p)
    below1, at1 = _units(a1[1], a1[2], p)
    alone = at0 == 0                      # nothing else was listed at your price: then what appears there is yours
    cheapest = alone and below0 == 0      # ...and nothing was cheaper: you were first in line
    out["tier_before"], out["tier_after_first"] = at0, at1
    out["alone"], out["cheapest_at_posting"] = bool(alone), bool(cheapest)
    out["appeared_share"] = (at1 / q) if (alone and q) else np.nan
    sold = row["units_sold"]
    out["invisible"] = bool(alone and row["done"] and sold >= 0.9 * q and out["appeared_share"] < 0.2)
    target = q + float(partner_qty)
    t_gone, still_there, n_obs, front_hits = np.nan, 0.0, 0, 0
    for ts, pr, qt in after:
        if ts > t0 + pd.Timedelta(hours=duration_hours):
            break
        n_obs += 1
        b, a = _units(pr, qt, p)
        if b == 0:
            front_hits += 1
        if cheapest:
            remaining = min(float(b + a), target)
            if remaining <= gone_share * target:
                t_gone = (ts - t0).total_seconds() / 3600.0
                break
            still_there = (ts - t0).total_seconds() / 3600.0
    out["observed_hours_to_clear"] = t_gone
    out["still_there_h"] = still_there if cheapest else np.nan
    out["front_share"] = front_hits / n_obs if n_obs else np.nan   # share of the next snapshots in which nothing was cheaper than you
    out["snapshots_watched"] = n_obs
    ahead_lifo = float(below0) + float(partner_qty)
    ahead_fifo = ahead_lifo + float(at0)
    for k in ("low", "est", "high"):
        out[f"pred_lifo_h_{k}"] = _hours_to_sell(ahead_lifo, q, vel[k])
        out[f"pred_fifo_h_{k}"] = _hours_to_sell(ahead_fifo, q, vel[k])
    out["status"] = "ok"
    return out


def reconcile(snapshots: Sequence[Snapshot], log: pd.DataFrame, duration_hours: float = 12.0, partner_window_min: float = 30.0):
    """Returns (per_listing DataFrame, summary dict)."""
    if log.empty:
        return pd.DataFrame(), {"listings": 0}
    ids = sorted(log["item_id"].unique())
    books = _item_books(snapshots, ids)
    series = item_series(snapshots, ids)
    rows = []
    for idx, r in log.iterrows():
        same = log[(log["item_id"] == r["item_id"]) & (log.index != idx)
                   & ((log["posted_utc"] - r["posted_utc"]).abs() <= pd.Timedelta(minutes=partner_window_min))
                   & (log["price_copper"] < r["price_copper"])]
        partner = float(same["quantity"].sum())
        vel = _velocity(series.get(int(r["item_id"])), pd.Timestamp(r["posted_utc"]))
        rows.append(trace_listing(r, books[int(r["item_id"])], partner, vel, duration_hours))
    per = pd.DataFrame(rows)
    return per, summarize(per)


def summarize(per: pd.DataFrame) -> dict:
    s: dict = {"listings": int(len(per))}
    if per.empty:
        return s
    ok = per[per["status"] == "ok"]
    s["usable"] = int(len(ok))
    s["not_usable"] = int(len(per) - len(ok))
    done = per[per["units_sold"].notna()]
    if ok.empty:
        s["units_posted"] = float(done["quantity"].sum())
        s["units_sold"] = float(done["units_sold"].sum())
        s["units_expired"] = float(done["units_expired"].fillna(0).sum())
        return s
    s["units_posted"] = float(done["quantity"].sum())
    s["units_sold"] = float(done["units_sold"].sum())
    s["units_expired"] = float(done["units_expired"].fillna(0).sum())
    if len(done):
        s["share_fully_sold"] = float((done["units_sold"] >= done["quantity"] - 0.5).mean())
    sold_ok = ok[ok["units_sold"].fillna(0) >= 0.9 * ok["quantity"]]
    s["fully_sold_usable"] = int(len(sold_ok))
    alone_sold = sold_ok[sold_ok["alone"].fillna(False).astype(bool)]
    s["detectable_sold_out"] = int(len(alone_sold))
    if len(alone_sold):
        s["invisible_count"] = int(alone_sold["invisible"].sum())
        s["invisible_share"] = float(alone_sold["invisible"].mean())
    seen = ok[ok["alone"].fillna(False).astype(bool) & ~ok["invisible"].fillna(False).astype(bool)]
    s["detection_n"] = int(len(seen))
    if len(seen):
        s["median_appeared_share"] = float(seen["appeared_share"].median())

    front = ok[ok["cheapest_at_posting"].fillna(False).astype(bool) & ok["done"].fillna(False).astype(bool)]
    # RATE CHECK: listings first in line that sold out, with a measured clearing window
    rc = front[(front["units_sold"] >= 0.9 * front["quantity"]) & front["observed_hours_to_clear"].notna() & ~front["abnormal"]]
    s["rate_check_n"] = int(len(rc))
    if len(rc):
        pred = rc["pred_lifo_h_est"]
        valid = pred.notna()
        rc = rc[valid]
        pred = pred[valid]
        s["rate_check_n"] = int(len(rc))
        s["rate_inside"] = int(((pred >= rc["still_there_h"]) & (pred <= rc["observed_hours_to_clear"])).sum())
        s["rate_pred_too_fast"] = int((pred < rc["still_there_h"]).sum())      # real sales were slower than predicted
        s["rate_pred_too_slow"] = int((pred > rc["observed_hours_to_clear"]).sum())   # real sales were faster
    # UNSOLD while first in line: pushed back by cheaper listings, or really slow?
    un = front[(front["units_sold"] == 0) & ~front["abnormal"]]
    s["unsold_front_n"] = int(len(un))
    if len(un):
        stayed = un[un["front_share"] >= 0.8]
        s["unsold_front_stayed"] = int(len(stayed))
        s["unsold_front_pushed_back"] = int((un["front_share"] < 0.8).sum())
        s["unsold_front_stayed_but_predicted_to_sell"] = int((stayed["pred_lifo_h_est"] <= 12.0).sum())
    return s


def verdict_text(s: dict) -> List[str]:
    """Plain-language lines for the summary. No verdict is claimed from fewer than 5 usable comparisons."""
    out = []
    if not s.get("listings"):
        return ["No listings in the log yet."]
    out.append(f"{s['listings']} listings in the log; {s.get('usable', 0)} have usable snapshots around them"
               + (f" ({s['not_usable']} do not: see the status column)." if s.get("not_usable") else "."))
    if s.get("units_posted"):
        out.append(f"Results filled in for {s['units_posted']:,.0f} units: {s['units_sold']:,.0f} sold, {s['units_expired']:,.0f} came back."
                   + (f" {s['share_fully_sold']:.0%} of listings sold out completely." if "share_fully_sold" in s else ""))
    if "invisible_share" in s:
        out.append(f"BLIND SPOT: of {s['detectable_sold_out']} sold-out listings with no other units at their price, {s['invisible_count']} "
                   f"({s['invisible_share']:.0%}) were gone before the next snapshot and never appeared in the data."
                   + (" Fewer than 5: rough." if s["detectable_sold_out"] < 5 else ""))
    if "median_appeared_share" in s:
        out.append(f"DETECTION: for {s['detection_n']} listings with no other units at their price, the first snapshot after posting showed "
                   f"{s['median_appeared_share']:.0%} of the units you posted (median; 100% means we see every unit you list).")
    if s.get("rate_check_n"):
        out.append(f"RATE CHECK on {s['rate_check_n']} listings that were first in line and sold out: our predicted clearing time fell inside the "
                   f"observed window in {s['rate_inside']}, was too FAST (real sales slower) in {s['rate_pred_too_fast']}, too SLOW (real sales "
                   f"faster) in {s['rate_pred_too_slow']}." + (" Fewer than 5: rough." if s["rate_check_n"] < 5 else ""))
    if s.get("unsold_front_n"):
        line = (f"UNSOLD while first in line: {s['unsold_front_n']} listings sold nothing. {s['unsold_front_pushed_back']} were pushed back by cheaper "
                f"listings that appeared afterwards (the price moved, not necessarily a slow item); {s['unsold_front_stayed']} stayed at the front all "
                f"12 hours")
        if s["unsold_front_stayed"]:
            line += f", and for {s['unsold_front_stayed_but_predicted_to_sell']} of those we had predicted a sale inside 12 hours (a rate we overstated)."
        else:
            line += "."
        out.append(line)
    out.append("QUEUE ORDER: not read from this log. SAME listings share a price with older units, which the snapshots cannot tell apart; "
               "use the `order` check on the book history.")
    return out


def item_evidence(per: pd.DataFrame, rate_check_max_listings: int = 0) -> pd.DataFrame:
    """Per item, what the experiment showed about our sales rate (first-in-line listings only, abnormal rows left out):
    inside / too_fast (real sales slower than predicted) / too_slow (real sales faster), and unsold_stayed (sold nothing
    while first in line for at least 80% of the snapshots, although a sale inside 12 hours was predicted) / unsold_pushed
    (sold nothing but was pushed back by cheaper listings, which says nothing about the rate)."""
    cols = ["item_id", "checks", "inside", "too_fast", "too_slow", "unsold_stayed", "unsold_pushed"]
    if per.empty or "cheapest_at_posting" not in per.columns:
        return pd.DataFrame(columns=cols)
    ok = per[(per["status"] == "ok") & per["cheapest_at_posting"].fillna(False).astype(bool)
             & per["done"].fillna(False).astype(bool) & ~per["abnormal"].fillna(False).astype(bool)]
    rows = []
    for item, g in ok.groupby("item_id"):
        sold = g[(g["units_sold"] >= 0.9 * g["quantity"]) & g["observed_hours_to_clear"].notna() & g["pred_lifo_h_est"].notna()]
        pred = sold["pred_lifo_h_est"]
        un = g[g["units_sold"] == 0]
        stayed = un[(un["front_share"] >= 0.8) & (un["pred_lifo_h_est"] <= 12.0)]
        rows.append(dict(
            item_id=int(item), checks=int(len(sold)),
            inside=int(((pred >= sold["still_there_h"]) & (pred <= sold["observed_hours_to_clear"])).sum()),
            too_fast=int((pred < sold["still_there_h"]).sum()), too_slow=int((pred > sold["observed_hours_to_clear"]).sum()),
            unsold_stayed=int(len(stayed)), unsold_pushed=int((un["front_share"] < 0.8).sum())))
    return pd.DataFrame(rows, columns=cols)
