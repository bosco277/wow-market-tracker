import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from engine.classifier import ClassifierParams
from engine.simulator import MarketSimulator, default_profiles
from engine.validation import evaluate_run, summarize

for sim_fifo in (True, False):
    for cls_fifo in (True, False):
        frames = []
        for seed in (1, 2):
            res = MarketSimulator(default_profiles(), seed=seed, fifo_within_tier=sim_fifo).run(72)
            frames.append(evaluate_run(res, ClassifierParams(fifo_within_tier=cls_fifo)))
        s = summarize(pd.concat(frames, ignore_index=True))
        tot = s[["obs_units", "low_units", "est_units", "high_units"]].sum()
        print(f"sim_fifo={sim_fifo!s:5} classifier_fifo={cls_fifo!s:5} | coverage={s['coverage'].mean():.3f} "
              f"low_viol={s['low_viol'].mean():.3f} high_viol={s['high_viol'].mean():.3f} | "
              f"low/obs={tot.low_units/tot.obs_units:.3f} est/obs={tot.est_units/tot.obs_units:.3f} high/obs={tot.high_units/tot.obs_units:.3f}")
