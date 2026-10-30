"""Scores classifier output against simulator ground truth (or, later, against your experiment log).

Per item-interval we compare the classifier's [sold_low, sold_high] with `sold_observable`: units actually sold
from listings that were already in the previous snapshot. That is the only quantity a snapshot diff could ever
recover. Units sold from listings created and sold inside the interval (`sold_total - sold_observable`) are the
blind spot and are reported separately.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .classifier import ClassifierParams, classify_transitions
from .simulator import SimulationResult

TOL = 0.5  # units


def evaluate_run(result: SimulationResult, params: Optional[ClassifierParams] = None) -> pd.DataFrame:
    """One row per (interval, item) with classifier bounds next to ground truth."""
    params = params or ClassifierParams()
    frames: List[pd.DataFrame] = []
    snaps = result.snapshots
    for prev, curr in zip(snaps[:-1], snaps[1:]):
        r = classify_transitions(prev, curr, params)
        truth = result.truth_between(prev, curr)
        frames.append(r.items.merge(truth, on="item_id", how="left"))
    return pd.concat(frames, ignore_index=True)


def summarize(rows: pd.DataFrame, by: str = "item_id") -> pd.DataFrame:
    """Bracketing and bias statistics. Flagged intervals (wipe/long gap) are excluded, like production rates."""
    use = rows[~(rows["flag_wipe"] | rows["flag_long_gap"])].copy()
    use["covered"] = (use["sold_low"] <= use["sold_observable"] + TOL) & (use["sold_observable"] <= use["sold_high"] + TOL)
    use["low_viol"] = use["sold_observable"] < use["sold_low"] - TOL
    use["high_viol"] = use["sold_observable"] > use["sold_high"] + TOL
    g = use.groupby(by)
    out = pd.DataFrame(
        {
            "intervals": g.size(),
            "coverage": g["covered"].mean(),
            "low_viol": g["low_viol"].mean(),
            "high_viol": g["high_viol"].mean(),
            "obs_units": g["sold_observable"].sum(),
            "low_units": g["sold_low"].sum(),
            "est_units": g["sold_est"].sum(),
            "high_units": g["sold_high"].sum(),
            "total_units": g["sold_total"].sum(),
        }
    )
    out["low_ratio"] = out["low_units"] / out["obs_units"].where(out["obs_units"] > 0)
    out["est_ratio"] = out["est_units"] / out["obs_units"].where(out["obs_units"] > 0)
    out["high_ratio"] = out["high_units"] / out["obs_units"].where(out["obs_units"] > 0)
    out["blind_spot_ratio"] = out["total_units"] / out["obs_units"].where(out["obs_units"] > 0)
    return out.reset_index()
