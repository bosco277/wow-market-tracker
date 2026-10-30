"""Replay harness: would this strategy have made money, under honest assumptions about liquidity?

Design rules
- NO LOOK-AHEAD. At snapshot k the strategy sees only snapshots[0..k] and classifier intervals that ended by then.
  tests/test_backtest.py proves it: perturbing future snapshots cannot change earlier decisions.
- ENTRY is a sweep of the observed ask book at the decision snapshot (you pay the real price tiers, not a mid or
  a floor). This is optimistic about latency: in reality someone else may buy first before you act.
- EXIT lists the position at a later snapshot at the realistic queue-position price (metrics.sell_clearing_price)
  given how many units the market is expected to absorb, net of the AH cut. Units that would not sell within the
  exit horizon are valued at a haircut to the floor.
- The unknown is how much the market really absorbs, so every backtest can be run three times (assumption = low /
  est / high of the classifier's sales bounds) to give a P&L RANGE instead of a single hopeful number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from .classifier import ClassifierParams, aggregate_rates, classify_transitions
from .metrics import AH_CUT, depth_vwap_table, expected_sales, sell_clearing_price, target_depth
from .schema import Snapshot


@dataclass(frozen=True)
class Order:
    item_id: int
    units: int
    max_unit_price: Optional[int] = None  # do not pay above this per unit (None = no cap)


@dataclass
class Context:
    now: pd.Timestamp
    snapshot: Snapshot
    history: List[Snapshot]          # snapshots up to and including `now`
    intervals: pd.DataFrame          # classifier item rows whose interval ended at or before `now`
    rates: pd.DataFrame              # trailing-24h daily low/est/high per item (observed time only)
    metrics_history: pd.DataFrame    # per past snapshot x item: ts, item_id, floor, depth_vwap, units
    free_cash: float                 # copper
    open_items: frozenset            # items with an open position (one position per item)


class Strategy(Protocol):
    def decide(self, ctx: Context) -> List[Order]: ...


@dataclass(frozen=True)
class BacktestConfig:
    bankroll: float = 100_000_000.0      # copper (10,000 gold)
    hold_hours: float = 6.0
    exit_horizon_hours: float = 12.0     # how long you are willing to wait for the exit to sell
    sales_assumption: str = "est"        # "low" | "est" | "high": how much the market absorbs at exit
    ah_cut: float = AH_CUT
    unfilled_haircut: float = 0.5        # unsold units valued at this fraction of (floor x (1 - cut))
    warmup_hours: float = 24.0           # no trading until this much history exists
    buy_depth_assumption: str = "high"   # depth used for the reference buy VWAP in metrics_history
    blind_spot_correction: float = 1.0
    classifier: ClassifierParams = field(default_factory=ClassifierParams)


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    summary: dict
    orders_log: List[tuple]  # (snapshot index, item_id, units) for auditing / look-ahead tests


TRADE_COLUMNS = [
    "item_id", "entry_ts", "exit_ts", "units", "cost", "entry_vwap", "list_price", "filled_units",
    "proceeds", "pnl", "roi", "assumption",
]


def _sweep(item_book: pd.DataFrame, units: int, max_price: Optional[int], budget: float):
    """Buys up to `units` cheapest-first, respecting a price cap and a cash budget. Returns (units, cost)."""
    b = item_book.sort_values(["unit_price", "auction_id"])
    got, cost = 0, 0.0
    for price, qty in zip(b["unit_price"].to_numpy(), b["quantity"].to_numpy()):
        if got >= units or (max_price is not None and price > max_price):
            break
        take = int(min(qty, units - got))
        affordable = int((budget - cost) // price)
        take = min(take, affordable)
        if take <= 0:
            break
        got += take
        cost += take * float(price)
        if take < qty:
            break
    return got, cost


def run_backtest(snapshots: List[Snapshot], strategy: Strategy, config: Optional[BacktestConfig] = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    if len(snapshots) < 3:
        raise ValueError("need at least 3 snapshots")
    if len({s.source for s in snapshots}) != 1:
        raise ValueError("backtest one source at a time")

    transitions = [classify_transitions(a, b, cfg.classifier).items for a, b in zip(snapshots[:-1], snapshots[1:])]
    all_intervals = pd.concat(transitions, ignore_index=True)

    open_pos: List[dict] = []
    trades: List[dict] = []
    orders_log: List[tuple] = []
    cash = float(cfg.bankroll)
    metric_rows: List[pd.DataFrame] = []
    start_ts = snapshots[0].ts

    def close_position(pos: dict, snap: Snapshot, intervals_now: pd.DataFrame, forced: bool = False) -> None:
        nonlocal cash
        item_book = snap.item_book(pos["item_id"])
        exp = expected_sales(intervals_now, snap.ts, cfg.exit_horizon_hours, cfg.sales_assumption,
                             blind_spot_correction=cfg.blind_spot_correction)
        e = exp[exp["item_id"] == pos["item_id"]]
        expected_units = float(e["expected_units"].iloc[0]) if len(e) else 0.0
        floor = float(item_book["unit_price"].min()) if len(item_book) else 0.0
        if len(item_book):
            q = sell_clearing_price(item_book, expected_units, pos["units"])
            list_price, filled = q.list_price, min(q.expected_filled, pos["units"])
        else:
            list_price, filled = float("nan"), 0.0
        revenue = (filled * list_price * (1 - cfg.ah_cut)) if filled > 0 else 0.0
        leftover = pos["units"] - filled
        residual = leftover * floor * (1 - cfg.ah_cut) * cfg.unfilled_haircut
        proceeds = revenue + residual
        cash += proceeds
        trades.append(dict(
            item_id=pos["item_id"], entry_ts=pos["entry_ts"], exit_ts=snap.ts, units=pos["units"], cost=pos["cost"],
            entry_vwap=pos["cost"] / pos["units"], list_price=list_price, filled_units=filled, proceeds=proceeds,
            pnl=proceeds - pos["cost"], roi=(proceeds - pos["cost"]) / pos["cost"], assumption=cfg.sales_assumption,
        ))

    for k, snap in enumerate(snapshots):
        now = snap.ts
        intervals_now = all_intervals[all_intervals["t1"] <= now]  # intervals ending at or before now: no future data

        # reference metrics for this snapshot (buy-side depth from the HIGH end of expected sales)
        exp_hi = expected_sales(intervals_now, now, 24.0, cfg.buy_depth_assumption, blind_spot_correction=cfg.blind_spot_correction)
        if len(exp_hi):
            depth = pd.Series(
                [target_depth(u) for u in exp_hi["expected_units"]], index=exp_hi["item_id"].to_numpy()
            )
            m = depth_vwap_table(snap.book, depth)
        else:
            m = pd.DataFrame(columns=["item_id", "units_filled", "vwap", "marginal_price", "floor"])
        floors = snap.book.groupby("item_id").agg(floor=("unit_price", "min"), units=("quantity", "sum")).reset_index()
        mh = floors.merge(m[["item_id", "vwap"]].rename(columns={"vwap": "depth_vwap"}), on="item_id", how="left")
        mh.insert(0, "ts", now)
        metric_rows.append(mh)

        # exits due
        still_open = []
        for pos in open_pos:
            if (now - pos["entry_ts"]) >= pd.Timedelta(hours=cfg.hold_hours):
                close_position(pos, snap, intervals_now)
            else:
                still_open.append(pos)
        open_pos = still_open

        # entries
        if (now - start_ts) >= pd.Timedelta(hours=cfg.warmup_hours) and k < len(snapshots) - 1:
            ctx = Context(
                now=now, snapshot=snap, history=snapshots[: k + 1], intervals=intervals_now,
                rates=aggregate_rates(intervals_now, now, 24.0),
                metrics_history=pd.concat(metric_rows, ignore_index=True), free_cash=cash,
                open_items=frozenset(p["item_id"] for p in open_pos),
            )
            for order in strategy.decide(ctx):
                orders_log.append((k, int(order.item_id), int(order.units)))
                if order.item_id in {p["item_id"] for p in open_pos} or order.units <= 0:
                    continue
                units, cost = _sweep(snap.item_book(order.item_id), order.units, order.max_unit_price, cash)
                if units <= 0:
                    continue
                cash -= cost
                open_pos.append(dict(item_id=order.item_id, units=units, cost=cost, entry_ts=now))

    # positions still open at the end are closed against the final book (marked, not hidden)
    for pos in open_pos:
        close_position(pos, snapshots[-1], all_intervals, forced=True)

    df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(cfg, df, summarize_trades(df, cfg), orders_log)


def summarize_trades(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return dict(n_trades=0, total_cost=0.0, total_pnl=0.0, roi_on_cost=float("nan"), hit_rate=float("nan"),
                    mean_roi=float("nan"), fill_ratio=float("nan"), max_drawdown=0.0, max_drawdown_pct_of_bankroll=0.0)
    t = trades.sort_values("exit_ts")
    cum = t["pnl"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.maximum(cum, 0.0))
    dd = float((peak - cum).max())
    return dict(
        n_trades=int(len(t)),
        total_cost=float(t["cost"].sum()),
        total_pnl=float(t["pnl"].sum()),
        roi_on_cost=float(t["pnl"].sum() / t["cost"].sum()),
        hit_rate=float((t["pnl"] > 0).mean()),
        mean_roi=float(t["roi"].mean()),
        fill_ratio=float(t["filled_units"].sum() / t["units"].sum()),
        max_drawdown=dd,
        max_drawdown_pct_of_bankroll=dd / cfg.bankroll,
    )


def run_backtest_range(
    snapshots: List[Snapshot], strategy_factory: Callable[[], Strategy], config: Optional[BacktestConfig] = None
) -> Dict[str, BacktestResult]:
    """Runs the same strategy under low / est / high market-absorption assumptions at exit. P&L is a RANGE."""
    cfg = config or BacktestConfig()
    out = {}
    for a in ("low", "est", "high"):
        out[a] = run_backtest(snapshots, strategy_factory(), BacktestConfig(**{**cfg.__dict__, "sales_assumption": a}))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Baseline strategy (a reference to beat, not a recommendation)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class DipBuyer:
    """Buy when the floor drops well below its recent median; size by a fraction of expected daily sales."""
    dip: float = 0.92                 # buy when floor <= dip x trailing median floor
    ref_hours: float = 24.0
    daily_fraction: float = 0.10      # position = this share of expected daily sales (low end: sell-side liquidity)
    max_units: int = 100_000
    min_units: int = 1
    price_cap_vs_ref: float = 0.97    # never pay above this x reference per unit

    def decide(self, ctx: Context) -> List[Order]:
        mh = ctx.metrics_history
        cutoff = ctx.now - pd.Timedelta(hours=self.ref_hours)
        past = mh[(mh["ts"] > cutoff) & (mh["ts"] < ctx.now)]
        if past.empty or ctx.rates.empty:
            return []
        ref = past.groupby("item_id")["floor"].median()
        cur = ctx.snapshot.book.groupby("item_id")["unit_price"].min()
        orders = []
        for _, r in ctx.rates.iterrows():
            item = int(r["item_id"])
            if item in ctx.open_items or item not in ref.index or item not in cur.index:
                continue
            if cur[item] <= self.dip * ref[item]:
                units = int(min(self.max_units, self.daily_fraction * r["daily_low"]))
                if units >= self.min_units:
                    orders.append(Order(item, units, int(self.price_cap_vs_ref * ref[item])))
        return orders
