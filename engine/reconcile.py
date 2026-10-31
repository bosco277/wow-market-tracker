"""Check the buy-and-list experiment log against the hourly snapshots.

The log (one row per listing: when you posted it, price, quantity, units sold, units expired) is the one place we know
what really sold. For each listing this module asks what the snapshots show for the same price, and answers three questions:

1. DETECTION   Did the snapshot right after posting show about `quantity` new units at that price? If you sold the whole
               listing before that snapshot it shows nothing: that is the blind spot ("invisible"), counted directly.
2. TIMING      How many hours until the units at or below the listing's price fell back to where they were before
               you posted, versus how long a newest-first (LIFO) queue and an oldest-first (FIFO) queue each predict?
               Only the SAME lane tells the two apart, because an UNDER listing is first in line under both rules.
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
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df[~df["note"].str.lower().str.startswith("example row")]
    df = df[(df["item_id"] != "") & (df["quantity"] != "") & (df["list_price"] != "")].copy()
    df["item_id"] = pd.to_numeric(df["item_id"]).astype("int64")
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
    in_progress = as_of & (df["units_sold"] < q)
    df.loc[in_progress, ["units_sold", "units_expired"]] = np.nan
    df["done"] = df["units_sold"].notna()
    df["in_progress"] = in_progress
    return df.sort_values("posted_utc").reset_index(drop=True)


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
               units_sold=row["units_sold"], units_expired=row["units_expired"], abnormal=bool(row["abnormal"]))
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
    out["tier_before"], out["tier_after_first"] = at0, at1
    out["appeared_share"] = (at1 - at0) / q if q else np.nan
    sold = row["units_sold"]
    out["invisible"] = bool(row["done"] and sold >= 0.9 * q and out["appeared_share"] < 0.2)
    target = q + float(partner_qty)
    base = below0 + at0
    t_gone, n_obs = np.nan, 0
    for ts, pr, qt in after:
        if ts > t0 + pd.Timedelta(hours=duration_hours):
            break
        n_obs += 1
        b, a = _units(pr, qt, p)
        remaining = min(max(b + a - base, 0.0), target)
        if remaining <= gone_share * target:
            t_gone = (ts - t0).total_seconds() / 3600.0
            break
    out["observed_hours_to_clear"] = t_gone
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
    if len(sold_ok):
        s["invisible_share"] = float(sold_ok["invisible"].mean())
        s["invisible_count"] = int(sold_ok["invisible"].sum())
    seen = ok[~ok["invisible"].fillna(False).astype(bool)]
    if len(seen):
        s["median_appeared_share"] = float(seen["appeared_share"].median())
    # SAME means "at the lowest price when you posted": judged from the book, not from the label you typed
    at_floor = (sold_ok["vs_floor_silver"] == 0) if "vs_floor_silver" in sold_ok.columns else (sold_ok["lane"] == "SAME")
    normal_same = sold_ok[at_floor & (~sold_ok["abnormal"]) & sold_ok["observed_hours_to_clear"].notna()]
    s["same_lane_for_order_test"] = int(len(normal_same))
    if len(normal_same):
        def err(col):
            pred = normal_same[col].clip(lower=0.25)
            obs = normal_same["observed_hours_to_clear"].clip(lower=0.25)
            return np.abs(np.log(pred / obs))
        e_l, e_f = err("pred_lifo_h_est"), err("pred_fifo_h_est")
        valid = e_l.notna() & e_f.notna()
        s["order_test_n"] = int(valid.sum())
        if valid.any():
            s["lifo_closer"] = int((e_l[valid] < e_f[valid]).sum())
            s["fifo_closer"] = int((e_f[valid] < e_l[valid]).sum())
            s["lifo_median_log_error"] = float(e_l[valid].median())
            s["fifo_median_log_error"] = float(e_f[valid].median())
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
        out.append(f"BLIND SPOT: {s['invisible_count']} of {s['fully_sold_usable']} listings that sold out ({s['invisible_share']:.0%}) were gone before the "
                   f"next snapshot, so they never appeared in the data." + (" Fewer than 5: rough." if s["fully_sold_usable"] < 5 else ""))
    if "median_appeared_share" in s:
        out.append(f"DETECTION: in the first snapshot after posting, the tier at your price grew by {s['median_appeared_share']:.0%} of the units you "
                   f"posted (median; 100% means we see every unit you list).")
    if s.get("order_test_n", 0) >= 5:
        winner = "NEWEST-FIRST (LIFO)" if s["lifo_closer"] > s["fifo_closer"] else ("OLDEST-FIRST (FIFO)" if s["fifo_closer"] > s["lifo_closer"] else "neither")
        out.append(f"QUEUE ORDER from {s['order_test_n']} SAME-lane listings: LIFO predicted the clearing time better in {s['lifo_closer']}, "
                   f"FIFO in {s['fifo_closer']}; leans {winner}. (median log error: LIFO {s['lifo_median_log_error']:.2f}, FIFO {s['fifo_median_log_error']:.2f})")
    elif s.get("same_lane_for_order_test", 0):
        out.append(f"QUEUE ORDER: only {s.get('order_test_n', 0)} comparable SAME-lane listings so far; need 5 before saying anything.")
    return out
