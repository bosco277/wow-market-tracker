import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from engine.backtest import BacktestConfig, Order, run_backtest
from engine.simulator import MarketSimulator, default_profiles

res = MarketSimulator(default_profiles(), seed=11).run(72)

class Greedy:
    def decide(self, ctx):
        return [Order(1001, 10_000_000)]

cfg = BacktestConfig(bankroll=2_000_000.0, warmup_hours=2, hold_hours=3)
r = run_backtest(res.snapshots, Greedy(), cfg)
t = r.trades.sort_values("entry_ts").reset_index(drop=True)

# replay cash exactly: events are entries (cash -= cost) and exits (cash += proceeds), exits before entries at equal ts
events = []
for _, row in t.iterrows():
    events.append((row["entry_ts"], 1, -row["cost"], row["cost"]))
    events.append((row["exit_ts"], 0, row["proceeds"], 0.0))
events.sort(key=lambda e: (e[0], e[1]))
cash = cfg.bankroll
min_cash = cash
for ts, kind, delta, cost in events:
    if kind == 1:
        assert cost <= cash + 1e-6, f"entry at {ts} costs {cost} but cash was only {cash}"
    cash += delta
    min_cash = min(min_cash, cash)
print(t[["entry_ts", "exit_ts", "cost", "proceeds", "pnl"]].round(0).head(8).to_string())
print("max single-trade cost:", t["cost"].max(), "bankroll:", cfg.bankroll)
print("min cash over replay:", round(min_cash, 2), " final cash:", round(cash, 2))
