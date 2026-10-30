"""WoW market engine: canonical snapshots, sales inference with bounds, simulator, metrics, backtest."""
from .schema import (  # noqa: F401
    COPPER_PER_GOLD,
    SchemaError,
    Snapshot,
    TimeLeftBuckets,
    snapshots_from_blizzard_v2,
)
from .classifier import (  # noqa: F401
    ClassifierParams,
    TransitionResult,
    aggregate_rates,
    classify_series,
    classify_transitions,
)

__version__ = "0.4.3"
