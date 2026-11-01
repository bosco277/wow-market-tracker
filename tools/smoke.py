import sys, time
sys.path.insert(0, "/home/claude/wow-engine")
import pandas as pd
from engine.simulator import MarketSimulator, default_profiles
from engine.classifier import classify_transitions

t = time.time()
sim = MarketSimulator(default_profiles(), seed=1)
res = sim.run(hours=48)
print("sim runtime %.1fs, snapshots=%d, ledger rows=%d" % (time.time() - t, len(res.snapshots), len(res.ledger)))
print(res.ledger["type"].value_counts().to_dict())
s0, s1 = res.snapshots[10], res.snapshots[11]
print("gap min:", (s1.ts - s0.ts).total_seconds() / 60)
print(s0.book.groupby("item_id").agg(auctions=("auction_id", "size"), units=("quantity", "sum"), floor=("unit_price", "min")))
r = classify_transitions(s0, s1)
truth = res.truth_between(s0, s1)
m = r.items.merge(truth, on="item_id")
print(m[["item_id", "prev_units", "sold_cert", "sold_low", "sold_est", "sold_high", "sold_observable", "sold_total"]].round(1).to_string())
print(r.events["event"].value_counts().to_dict())
