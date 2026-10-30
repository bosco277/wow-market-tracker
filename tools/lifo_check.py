"""How much does knowing the order inside a price tier help, and what does a wrong assumption cost?

    python tools/lifo_check.py
Compares classifier settings on simulated markets whose true rule is newest-first (LIFO), oldest-first or random.
Columns: coverage = share of intervals whose true observable sales fall inside [low, high] (higher is better);
high_viol = true sales ABOVE the high bound (should be ~0); low_viol = true sales BELOW the low bound;
low/est/high ratio = bound as a multiple of the true observable sales.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from engine.classifier import ClassifierParams
from engine.simulator import MarketSimulator, default_profiles
from engine.validation import evaluate_run, summarize

MARKETS = {"true rule = LIFO": dict(lifo_within_tier=True), "true rule = FIFO": dict(fifo_within_tier=True),
           "true rule = random": dict(fifo_within_tier=False)}
SETTINGS = {"unknown": "unknown", "assume lifo": "lifo", "assume fifo": "fifo"}

rows = []
for mname, kw in MARKETS.items():
    runs = [MarketSimulator(default_profiles(), seed=s, **kw).run(96) for s in (1, 2, 3)]
    for sname, order in SETTINGS.items():
        params = ClassifierParams(tier_order=order)
        s = summarize(pd.concat([evaluate_run(run, params) for run in runs], ignore_index=True))
        agg = s[["coverage", "low_viol", "high_viol", "low_ratio", "est_ratio", "high_ratio"]].mean().round(3)
        rows.append({"market": mname, "classifier": sname, **agg.to_dict()})
print(pd.DataFrame(rows).to_string(index=False))
