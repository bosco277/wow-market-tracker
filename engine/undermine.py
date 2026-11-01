"""Undermine Exchange: long price and stock history for commodities (https://undermine.exchange/api.html).

What the API gives, per commodity and per region (US): DAILY rows (day, price, quantity: the price and units listed at the snapshot with the most
units that day, every day since tracking began) and HOURLY rows (snapshot time, lowest price, total units listed) for about the last 14 days.
It does NOT give sales or the listing-by-listing book, so it cannot replace our own hourly collection. What it is good for here:
  * years of daily price and stock behaviour per item, so groups can be fingerprinted on far more than a week of our own data;
  * a calibration check: how well do totals-only hourly rows track the sales we infer from the full book?

Budget: 3,000 points per sliding hour, 3 points per history request when gzip is sent. The client below keeps a local count under
`points_per_hour`, stops cleanly on HTTP 429 (the budget the server tracks may include earlier runs), and never prints the key."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASE = "https://api.undermine.exchange"
HISTORY_COLUMNS = ["item_id", "day", "price", "quantity"]
META_COLUMNS = ["item_id", "status", "fetched_at", "first_day", "last_day", "n_days"]


class BudgetSpent(Exception):
    """HTTP 429: the hourly point budget is used up. Progress so far is kept; run again in about an hour."""


class KeyRejected(Exception):
    """HTTP 401 / 403: the key is missing, wrong or the endpoint needs a paid membership."""


class Client:
    """Minimal paced client. `session` needs .get(url, headers=, timeout=) like requests.Session."""

    def __init__(self, key: str, session=None, points_per_hour: int = 2700, cost: int = 3, pause: float = 0.25,
                 sleep: Callable = time.sleep, now: Callable = time.time):
        if not key:
            raise KeyRejected("No API key. Set the UNDERMINE_API_KEY secret.")
        self._headers = {"Authorization": f"ApiKey {key}", "Accept-Encoding": "gzip", "User-Agent": "wow-market-tracker"}
        if session is None:
            import requests
            session = requests.Session()
        self._s, self._cost, self._budget, self._pause = session, cost, points_per_hour, pause
        self._sleep, self._now = sleep, now
        self._spent = deque()                      # (time, points)
        self.points = 0

    def _wait_for_budget(self):
        while True:
            now = self._now()
            while self._spent and now - self._spent[0][0] >= 3600:
                self._spent.popleft()
            if sum(p for _, p in self._spent) + self._cost <= self._budget:
                return
            self._sleep(max(1.0, 3600 - (now - self._spent[0][0]) + 1))

    def get(self, path: str) -> Tuple[int, Optional[dict]]:
        """(status, json). 404 gives (404, None). 401/403 raise KeyRejected, 429 raises BudgetSpent; network trouble retries, then gives (0, None)."""
        for attempt in range(3):
            self._wait_for_budget()
            try:
                r = self._s.get(BASE + path, headers=self._headers, timeout=60)
            except Exception:
                self._sleep(2 ** attempt)
                continue
            self._spent.append((self._now(), self._cost))
            self.points += self._cost
            if r.status_code == 200:
                try:
                    out = (200, r.json())
                except ValueError:
                    out = (0, None)
                self._sleep(self._pause)
                return out
            if r.status_code in (401, 403):
                raise KeyRejected(f"HTTP {r.status_code}: the key was rejected, or this endpoint needs a paid membership.")
            if r.status_code == 429:
                raise BudgetSpent("HTTP 429: the hourly point budget is used up.")
            if r.status_code == 404:
                return 404, None
            self._sleep(2 ** attempt)
        return 0, None


def parse_daily(js: dict, item_id: int) -> pd.DataFrame:
    rows = ((js or {}).get("result") or {}).get("daily") or []
    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    d = pd.DataFrame(rows)
    out = pd.DataFrame({"item_id": int(item_id), "day": d["day"].astype(str), "price": pd.to_numeric(d["price"], errors="coerce"),
                        "quantity": pd.to_numeric(d["quantity"], errors="coerce")})
    return out.dropna(subset=["price", "quantity"]).drop_duplicates("day", keep="last").reset_index(drop=True)


def parse_hourly(js: dict, item_id: int) -> pd.DataFrame:
    rows = ((js or {}).get("result") or {}).get("hourly") or []
    if not rows:
        return pd.DataFrame(columns=["item_id", "ts", "price", "quantity"])
    d = pd.DataFrame(rows)
    return pd.DataFrame({"item_id": int(item_id), "ts": pd.to_datetime(d["snapshot"], utc=True), "price": pd.to_numeric(d["price"], errors="coerce"),
                         "quantity": pd.to_numeric(d["quantity"], errors="coerce")}).dropna().sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------ which items
def sample_items(groups: pd.DataFrame, per_group: int = 30, exclude: Iterable[int] = ()) -> List[int]:
    """A balanced list of item ids: at most `per_group` per group, spread across era and quality inside each group (round-robin over the
    strata), and ordered so that ANY prefix is balanced too (first one item from every group, then the second, ...). So a run that stops
    early still leaves useful coverage. Uses the item_groups table (item_id, group, quality)."""
    if groups is None or groups.empty:
        return []
    from .grouptest import ERA_EDGES, ERA_LABELS
    g = groups.drop_duplicates("item_id").copy()
    g = g[~g["item_id"].isin(set(int(x) for x in exclude))]
    g["_era"] = pd.Series(ERA_LABELS)[np.searchsorted(ERA_EDGES, g["item_id"].to_numpy(), side="right") - 1].to_numpy()
    g["_q"] = g["quality"].fillna("").astype(str) if "quality" in g.columns else ""
    ranked = []
    for name, part in g.groupby("group"):
        strata = [list(p.sort_values("item_id")["item_id"]) for _, p in part.groupby(["_era", "_q"])]
        strata.sort(key=lambda s: (-len(s), s[0]))
        order, k = [], 0
        while len(order) < min(per_group, len(part)):
            for s in strata:
                if k < len(s) and len(order) < per_group:
                    order.append(int(s[k]))
            k += 1
        ranked += [(i, name, int(iid)) for i, iid in enumerate(order)]
    ranked.sort()
    return [iid for _, _, iid in ranked]


# ------------------------------------------------------------------ daily pull
def merge_history(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    parts = [p for p in (old, new) if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    m = pd.concat([p[HISTORY_COLUMNS] for p in parts], ignore_index=True)
    return m.drop_duplicates(["item_id", "day"], keep="last").sort_values(["item_id", "day"]).reset_index(drop=True)


def pull_daily(item_ids: Sequence[int], client, region: str = "us", batch: int = 800, progress: Callable = print) -> dict:
    """Fetch the daily history of up to `batch` items. Returns {"data", "meta", "stopped"}: stopped is "" or why it ended early
    ("budget" or "key"); whatever was fetched before that is returned."""
    frames, meta, stopped = [], [], ""
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    for n, iid in enumerate(item_ids[:batch], 1):
        try:
            status, js = client.get(f"/v1/region/{region}/commodities/{int(iid)}/daily.json")
        except BudgetSpent:
            stopped = "budget"
            break
        except KeyRejected as e:
            stopped = "key"
            progress(f"  {e}")
            break
        if status == 404:
            meta.append(dict(item_id=int(iid), status="not_found", fetched_at=stamp, first_day="", last_day="", n_days=0))
            continue
        if status != 200:
            continue                                                     # network trouble: not recorded, so the next run tries again
        d = parse_daily(js, iid)
        if d.empty:
            meta.append(dict(item_id=int(iid), status="empty", fetched_at=stamp, first_day="", last_day="", n_days=0))
            continue
        frames.append(d)
        meta.append(dict(item_id=int(iid), status="ok", fetched_at=stamp, first_day=d["day"].min(), last_day=d["day"].max(), n_days=len(d)))
        if n % 100 == 0:
            progress(f"  {n} items fetched ({client.points} points used)")
    return {"data": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HISTORY_COLUMNS),
            "meta": pd.DataFrame(meta, columns=META_COLUMNS), "stopped": stopped}


def merge_meta(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    parts = [p for p in (old, new) if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=META_COLUMNS)
    return pd.concat([p[META_COLUMNS] for p in parts], ignore_index=True).drop_duplicates("item_id", keep="last").sort_values("item_id").reset_index(drop=True)


# ------------------------------------------------------------------ hourly calibration
def align_hourly(under: pd.DataFrame, ours: pd.DataFrame, tolerance_minutes: int = 30) -> pd.DataFrame:
    """Match each Undermine hourly row to our nearest snapshot row of the same item (within the tolerance). `ours`: item_id, ts, floor_gold,
    units_listed, sold_est. Returns one row per matched pair, in time order, with Undermine's price/quantity next to ours."""
    if under.empty or ours.empty:
        return pd.DataFrame()
    o = ours[["item_id", "ts", "floor_gold", "units_listed", "sold_est"]].copy()
    o["ts"] = pd.to_datetime(o["ts"], utc=True).astype("datetime64[ns, UTC]")          # same resolution on both sides, or the merge refuses
    o = o.sort_values("ts")
    u = under.rename(columns={"ts": "ts_u", "price": "price_u", "quantity": "quantity_u"}).copy()
    u["ts_u"] = pd.to_datetime(u["ts_u"], utc=True).astype("datetime64[ns, UTC]")
    u["ts"] = u["ts_u"]
    m = pd.merge_asof(u.sort_values("ts"), o, on="ts", by="item_id", direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes))
    m = m.dropna(subset=["floor_gold"]).sort_values(["item_id", "ts_u"]).reset_index(drop=True)
    return m


def calibration_report(m: pd.DataFrame, min_pairs: int = 30) -> dict:
    """How well do Undermine's totals-only hourly rows match ours and track our sales estimate?
    price_ratio / qty_ratio: Undermine / ours (1.0 = identical units and feed; 0.0001 would mean gold vs copper);
    exact shares; and per item the rank correlation between 'units that disappeared from the total this hour' and our sold_est."""
    if m is None or m.empty:
        return {}
    ours_copper = m["floor_gold"] * 10000.0
    out = {"pairs": int(len(m)), "items": int(m["item_id"].nunique()),
           "price_ratio": float((m["price_u"] / ours_copper).median()), "price_exact": float((m["price_u"].round() == ours_copper.round()).mean()),
           "qty_ratio": float((m["quantity_u"] / m["units_listed"].replace(0, np.nan)).median()), "qty_exact": float((m["quantity_u"] == m["units_listed"]).mean())}
    rows = []
    for iid, g in m.groupby("item_id"):
        g = g.sort_values("ts_u")
        drop = (-g["quantity_u"].diff()).clip(lower=0)
        ok = drop.notna() & g["sold_est"].notna()
        if ok.sum() < min_pairs or g.loc[ok, "sold_est"].nunique() < 3 or drop[ok].nunique() < 3:
            continue
        rho = float(drop[ok].rank().corr(g.loc[ok, "sold_est"].rank()))
        tot_s = float(g.loc[ok, "sold_est"].sum())
        rows.append(dict(item_id=int(iid), hours=int(ok.sum()), spearman=rho, proxy_over_ours=float(drop[ok].sum() / tot_s) if tot_s > 0 else np.nan,
                         median_units=float(g["quantity_u"].median()), our_sold_per_hour=float(g.loc[ok, "sold_est"].mean())))
    per = pd.DataFrame(rows)
    out["per_item"] = per
    if len(per):
        out["spearman_median"] = float(per["spearman"].median())
        out["depth_buckets"] = _depth_buckets(per)
    return out


def _depth_buckets(per: pd.DataFrame) -> pd.DataFrame:
    """The proxy should work worse when the stock is huge compared with hourly sales. Bucket by units listed per hourly sale."""
    p = per.copy()
    p["stock_per_sale"] = p["median_units"] / p["our_sold_per_hour"].replace(0, np.nan)
    p = p.dropna(subset=["stock_per_sale", "spearman"])
    if len(p) < 3:
        return pd.DataFrame()
    p["bucket"] = pd.qcut(p["stock_per_sale"].rank(method="first"), q=min(3, len(p)), labels=["shallow stock", "medium", "deep stock"][:min(3, len(p))])
    return p.groupby("bucket", observed=True).agg(items=("item_id", "count"), spearman_median=("spearman", "median"),
                                                  stock_per_hourly_sale=("stock_per_sale", "median")).reset_index()


def sample_for_check(items: pd.DataFrame, hourly: pd.DataFrame, n: int = 40, seed: int = 7) -> List[int]:
    """Items for the hourly calibration: well-measured (Medium/High confidence), selling at least 20 a day, present in our hourly table, spread over
    the families (round-robin)."""
    if items is None or items.empty or hourly is None or hourly.empty:
        return []
    it = items[items["item_id"].isin(set(hourly["item_id"].unique()))]
    if "confidence" in it.columns:
        it = it[it["confidence"].isin(("Medium", "High"))]
    if "daily_est" in it.columns:
        it = it[pd.to_numeric(it["daily_est"], errors="coerce") >= 20]
    if it.empty:
        return []
    rng = np.random.default_rng(seed)
    fam = "family" if "family" in it.columns else None
    lists = []
    for _, part in (it.groupby(fam) if fam else [("all", it)]):
        ids = list(part["item_id"].astype(int))
        rng.shuffle(ids)
        lists.append(ids)
    out, k = [], 0
    while len(out) < n and any(k < len(x) for x in lists):
        for x in lists:
            if k < len(x) and len(out) < n:
                out.append(x[k])
        k += 1
    return out
