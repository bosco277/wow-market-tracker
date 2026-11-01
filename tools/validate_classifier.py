"""Runs the classifier against simulated markets, including stress cases where the market breaks our assumptions.

    python tools/validate_classifier.py            # prints the scenario table
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from engine.classifier import ClassifierParams
from engine.schema import TimeLeftBuckets
from engine.simulator import ItemProfile, MarketSimulator, default_profiles
from engine.validation import evaluate_run, summarize

HOURS = 96
SEEDS = (1, 2, 3)


def scale_cancels(profiles, factor, relist=None):
    out = []
    for p in profiles:
        kw = dict(cancel_rate_per_hr=p.cancel_rate_per_hr * factor)
        if relist is not None:
            kw["p_relist"] = relist
        out.append(dataclasses.replace(p, **kw))
    return out


SHIFTED = TimeLeftBuckets(bounds={"SHORT": (0.0, 60.0), "MEDIUM": (60.0, 240.0), "LONG": (240.0, 900.0), "VERY_LONG": (900.0, 2880.0)})

# name -> (profiles, simulator kwargs, run kwargs, classifier params)
SCENARIOS = {
    "baseline": (default_profiles(), {}, {}, ClassifierParams()),
    "heavy cancel churn (x6, 90% relist)": (scale_cancels(default_profiles(), 6, 0.9), {}, {}, ClassifierParams()),
    "random order within tier, classifier assumes FIFO": (default_profiles(), {"fifo_within_tier": False}, {}, ClassifierParams(fifo_within_tier=True)),
    "random order within tier, classifier tier-aware (default)": (default_profiles(), {"fifo_within_tier": False}, {}, ClassifierParams()),
    "30% of snapshots missed": (default_profiles(), {}, {"miss_prob": 0.3}, ClassifierParams()),
    "true time_left buckets differ from assumed": (default_profiles(), {"buckets": SHIFTED}, {}, ClassifierParams()),
    "demand x5 for 12h (reset-day style spike)": (
        default_profiles(),
        {"demand_mult": lambda m: 5.0 if 1440 <= m < 1440 + 720 else 1.0},
        {},
        ClassifierParams(),
    ),
}


def run_scenario(name):
    profiles, sim_kw, run_kw, params = SCENARIOS[name]
    frames = []
    for seed in SEEDS:
        res = MarketSimulator(profiles, seed=seed, **sim_kw).run(HOURS, **run_kw)
        rows = evaluate_run(res, params)
        rows["seed"] = seed
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name in SCENARIOS:
        rows = run_scenario(name)
        s = summarize(rows)
        print(f"\n=== {name} ===")
        print(s[["item_id", "intervals", "coverage", "low_viol", "high_viol", "obs_units", "low_ratio", "est_ratio", "high_ratio", "blind_spot_ratio"]]
              .round(3).to_string(index=False))
