"""Commodity auction-house simulator with a ground-truth event ledger.

Purpose: give the classifier something to be wrong against. The simulator knows exactly which units were bought,
cancelled and expired, so we can measure how often [sold_low, sold_high] brackets the truth, how biased sold_est
is, and what happens when the real market breaks our assumptions (stress parameters below).

IMPORTANT: every behavioural parameter here (cancel rates, undercut habits, buyer tolerance) is an invented
stand-in for real player behaviour. Validation against the simulator tests the LOGIC (given these dynamics, do
the bounds hold?) and quantifies sensitivity. It cannot calibrate the parameters. That needs real events from
your own experiment log.

Mechanics
- Sellers list new stacks (Poisson). A new listing undercuts the floor, joins an existing price tier ("wall"), or
  posts near fair value. Sellers refuse to undercut below 60% of fair value.
- Buyers arrive (Poisson), want a quantity, and sweep the book cheapest-first (earliest listing first within a
  tier, unless `fifo_within_tier=False`) up to a personal limit price.
- Sellers cancel listings that sit above the floor; most of those are re-posted as an undercut (churn).
- Listings expire after 12/24/48 hours.
- Snapshots are taken on a jittered schedule and can be missed entirely.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import Snapshot, TimeLeftBuckets


@dataclass(frozen=True)
class ItemProfile:
    item_id: int
    fair_price: int  # copper per unit
    list_rate_per_hr: float
    list_qty_mean: float
    buy_rate_per_hr: float
    buy_qty_mean: float
    p_undercut: float = 0.55
    p_wall: float = 0.15
    cancel_rate_per_hr: float = 0.04  # hazard per listing that is NOT at the floor
    p_relist: float = 0.7
    random_cancel_per_hr: float = 0.004  # hazard per listing, any position, never relisted
    buyer_tolerance: float = 0.08  # mean premium over fair value a buyer will pay
    duration_hours: Tuple[int, ...] = (12, 24, 48)
    duration_probs: Tuple[float, ...] = (0.2, 0.3, 0.5)


def default_profiles() -> List[ItemProfile]:
    return [
        ItemProfile(1001, 10_000, 14.0, 300, 11.0, 280),    # staple: ~1g, deep and busy
        ItemProfile(1002, 250_000, 5.0, 18, 3.5, 14),       # mid: ~25g
        ItemProfile(1003, 3_000_000, 0.7, 3, 0.3, 2),       # thin: ~300g, few listings
        ItemProfile(1004, 3_000, 30.0, 600, 24.0, 560),     # very liquid, very cheap
    ]


@dataclass
class SimulationResult:
    snapshots: List[Snapshot]
    ledger: pd.DataFrame  # ts, item_id, auction_id, type (LIST|SALE|CANCEL|EXPIRE), qty, price, relist
    start: pd.Timestamp

    def truth_between(self, prev: Snapshot, curr: Snapshot) -> pd.DataFrame:
        """Ground truth for the interval (prev.ts, curr.ts].

        sold_observable  units sold from listings that were already in `prev` (what a classifier can possibly see)
        sold_total       all units sold, including listings created AND sold inside the interval (blind spot)
        """
        led = self.ledger
        w = led[(led["ts"] > prev.ts) & (led["ts"] <= curr.ts)]
        prev_ids = set(prev.book["auction_id"].tolist())
        sales = w[w["type"] == "SALE"]
        obs = sales[sales["auction_id"].isin(prev_ids)].groupby("item_id")["qty"].sum()
        tot = sales.groupby("item_id")["qty"].sum()
        canc = w[(w["type"] == "CANCEL") & w["auction_id"].isin(prev_ids)].groupby("item_id")["qty"].sum()
        expd = w[(w["type"] == "EXPIRE") & w["auction_id"].isin(prev_ids)].groupby("item_id")["qty"].sum()
        idx = sorted(set(prev.book["item_id"].unique()))
        out = pd.DataFrame(index=pd.Index(idx, name="item_id"))
        out["sold_observable"] = obs.reindex(out.index).fillna(0.0)
        out["sold_total"] = tot.reindex(out.index).fillna(0.0)
        out["cancelled_units"] = canc.reindex(out.index).fillna(0.0)
        out["expired_units"] = expd.reindex(out.index).fillna(0.0)
        return out.reset_index()


class MarketSimulator:
    def __init__(
        self,
        profiles: Sequence[ItemProfile],
        start: pd.Timestamp = pd.Timestamp("2026-10-01 00:00:00"),
        seed: int = 0,
        buckets: Optional[TimeLeftBuckets] = None,
        fifo_within_tier: bool = True,
        diurnal_amplitude: float = 0.4,
        drift_sigma_per_hr: float = 0.004,
        mean_revert_per_hr: float = 0.02,
        demand_mult: Optional[Callable[[int], float]] = None,
        supply_mult: Optional[Callable[[int], float]] = None,
        source: str = "sim",
    ):
        self.profiles = list(profiles)
        self.start = pd.Timestamp(start)
        self.rng = np.random.default_rng(seed)
        self.buckets = buckets or TimeLeftBuckets()
        self.fifo = fifo_within_tier
        self.diurnal = diurnal_amplitude
        self.drift_sigma = drift_sigma_per_hr
        self.mean_revert = mean_revert_per_hr
        self.demand_mult = demand_mult or (lambda minute: 1.0)
        self.supply_mult = supply_mult or (lambda minute: 1.0)
        self.source = source

        self._next_id = 1
        self._books: Dict[int, Dict[int, list]] = {p.item_id: {} for p in self.profiles}  # aid -> [qty, price, expire_min]
        self._heaps: Dict[int, list] = {p.item_id: [] for p in self.profiles}
        self._fair_mult: Dict[int, float] = {p.item_id: 1.0 for p in self.profiles}
        self._log: List[tuple] = []

    # ------------------------------------------------------------------------------------------------ helpers
    def _fair(self, prof: ItemProfile) -> float:
        return prof.fair_price * self._fair_mult[prof.item_id]

    def _diurnal(self, minute: int) -> float:
        hour = (minute / 60.0) % 24.0
        return max(0.1, 1.0 + self.diurnal * np.sin(2 * np.pi * (hour - 14.0) / 24.0))

    def _log_event(self, minute: int, item: int, aid: int, typ: str, qty: int, price: int, relist: bool = False):
        self._log.append((minute, item, aid, typ, int(qty), int(price), relist))

    def _qty(self, mean: float) -> int:
        sigma = 0.6
        mu = np.log(max(mean, 1.0)) - sigma ** 2 / 2
        return max(1, int(round(self.rng.lognormal(mu, sigma))))

    def _price_for_new_listing(self, prof: ItemProfile, book: dict) -> int:
        fair = self._fair(prof)
        if not book:
            return max(1, int(round(fair * (1 + abs(self.rng.normal(0, 0.04))))))
        floor = min(a[1] for a in book.values())
        u = self.rng.random()
        if u < prof.p_undercut:
            cand = floor - max(1, int(round(floor * self.rng.uniform(0, 0.003))))
            if cand >= 0.6 * fair:
                return max(1, cand)
        elif u < prof.p_undercut + prof.p_wall:
            prices = sorted({a[1] for a in book.values()})[:8]
            return int(prices[self.rng.integers(len(prices))])
        return max(1, int(round(fair * (1 + abs(self.rng.normal(0, 0.06))))))

    def _list(self, prof: ItemProfile, minute: int, qty: int, relist: bool = False) -> None:
        book = self._books[prof.item_id]
        price = self._price_for_new_listing(prof, book)
        dur = int(self.rng.choice(prof.duration_hours, p=prof.duration_probs)) * 60
        aid = self._next_id
        self._next_id += 1
        book[aid] = [qty, price, minute + dur]
        heapq.heappush(self._heaps[prof.item_id], (minute + dur, aid))
        self._log_event(minute, prof.item_id, aid, "LIST", qty, price, relist)

    def _buy(self, prof: ItemProfile, minute: int, want: int, limit: float) -> None:
        book = self._books[prof.item_id]
        if not book:
            return
        if self.fifo:
            key = lambda kv: (kv[1][1], kv[0])
        else:
            key = lambda kv: (kv[1][1], self.rng.random())
        cands = sorted((kv for kv in book.items() if kv[1][1] <= limit), key=key)
        for aid, a in cands:
            if want <= 0:
                break
            take = min(want, a[0])
            a[0] -= take
            want -= take
            self._log_event(minute, prof.item_id, aid, "SALE", take, a[1])
            if a[0] == 0:
                del book[aid]

    # ------------------------------------------------------------------------------------------------ one minute
    def _step(self, minute: int) -> None:
        dm = self._diurnal(minute) * self.demand_mult(minute)
        sm = self.supply_mult(minute)
        if minute % 60 == 0:  # hourly fair-value drift (log Ornstein-Uhlenbeck)
            for p in self.profiles:
                lm = np.log(self._fair_mult[p.item_id])
                lm += -self.mean_revert * lm + self.drift_sigma * self.rng.normal()
                self._fair_mult[p.item_id] = float(np.exp(lm))

        for prof in self.profiles:
            book = self._books[prof.item_id]

            # cancels (and most get re-posted as undercuts)
            if book:
                floor = min(a[1] for a in book.values())
                elig = [aid for aid, a in book.items() if a[1] > floor]
                n = self.rng.poisson(len(elig) * prof.cancel_rate_per_hr / 60.0)
                if n and elig:
                    for aid in self.rng.choice(elig, size=min(n, len(elig)), replace=False):
                        aid = int(aid)
                        if aid not in book:
                            continue
                        qty, price, _ = book.pop(aid)
                        self._log_event(minute, prof.item_id, aid, "CANCEL", qty, price)
                        if self.rng.random() < prof.p_relist:
                            self._list(prof, minute, qty, relist=True)
                n2 = self.rng.poisson(len(book) * prof.random_cancel_per_hr / 60.0)
                if n2 and book:
                    keys = list(book.keys())
                    for aid in self.rng.choice(keys, size=min(n2, len(keys)), replace=False):
                        aid = int(aid)
                        if aid in book:
                            qty, price, _ = book.pop(aid)
                            self._log_event(minute, prof.item_id, aid, "CANCEL", qty, price)

            # new listings
            for _ in range(self.rng.poisson(prof.list_rate_per_hr * sm / 60.0)):
                self._list(prof, minute, self._qty(prof.list_qty_mean))

            # buyers
            for _ in range(self.rng.poisson(prof.buy_rate_per_hr * dm / 60.0)):
                limit = self._fair(prof) * (1.0 + self.rng.exponential(prof.buyer_tolerance))
                self._buy(prof, minute, self._qty(prof.buy_qty_mean), limit)

            # expiries
            heap = self._heaps[prof.item_id]
            while heap and heap[0][0] <= minute:
                _, aid = heapq.heappop(heap)
                if aid in book:
                    qty, price, _ = book.pop(aid)
                    self._log_event(minute, prof.item_id, aid, "EXPIRE", qty, price)

    # ------------------------------------------------------------------------------------------------ snapshots
    def _snapshot(self, minute: int) -> Snapshot:
        rows = []
        for item, book in self._books.items():
            for aid, (qty, price, exp) in book.items():
                rows.append((aid, item, str(item), qty, price, exp - minute))
        df = pd.DataFrame(rows, columns=["auction_id", "item_id", "variant_key", "quantity", "unit_price", "remaining"])
        if df.empty:
            df = pd.DataFrame(
                {c: pd.Series(dtype=t) for c, t in
                 [("auction_id", "int64"), ("item_id", "int64"), ("variant_key", "object"), ("quantity", "int64"),
                  ("unit_price", "int64"), ("remaining", "float64")]}
            )
        df["time_left"] = self.buckets.bucket_for_remaining(df["remaining"].to_numpy())
        df = df.drop(columns=["remaining"])
        return Snapshot(self.source, self.start + pd.Timedelta(minutes=minute), df)

    def run(
        self,
        hours: int,
        snapshot_interval_min: int = 60,
        jitter_min: int = 4,
        miss_prob: float = 0.0,
        burn_in_hours: int = 72,
    ) -> SimulationResult:
        total_min = int(hours * 60)
        # snapshot schedule (jittered, strictly increasing, some may be missed)
        sched: List[int] = []
        k = 0
        while k * snapshot_interval_min <= total_min:
            t = k * snapshot_interval_min + (int(self.rng.integers(-jitter_min, jitter_min + 1)) if jitter_min and k else 0)
            t = max(0, min(total_min, t))
            if (k == 0 or self.rng.random() >= miss_prob) and (not sched or t > sched[-1]):
                sched.append(t)
            k += 1
        sched_set = set(sched)

        snaps: List[Snapshot] = []
        for minute in range(-int(burn_in_hours * 60), total_min + 1):
            self._step(minute)
            if minute in sched_set:
                snaps.append(self._snapshot(minute))

        led = pd.DataFrame(self._log, columns=["minute", "item_id", "auction_id", "type", "qty", "price", "relist"])
        led["ts"] = self.start + pd.to_timedelta(led["minute"], unit="m")
        led = led[["ts", "item_id", "auction_id", "type", "qty", "price", "relist"]]
        return SimulationResult(snaps, led, self.start)
