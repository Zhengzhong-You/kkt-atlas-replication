"""Benchmark methods for contextual stochastic optimization."""

from kkt_atlas.baselines.contextual_empirical import KNearestNeighborSAA, fit_knn_saa
from kkt_atlas.baselines.iceo import (
    FittedICEO,
    FullSupportICEOModel,
    ICEOModelConfig,
    ICEOSnapshot,
    ICEOTrainingConfig,
    ICEOTrainingResult,
    fit_iceo,
    select_iceo_checkpoint,
    train_iceo_model,
)
from kkt_atlas.baselines.matched_fixed_support import (
    MatchedObservedSupportResult,
    ObservedSupportInitialization,
    ObservedSupportLocalModel,
    fit_matched_observed_support_atlas,
    snap_centroids_to_observed_scenarios,
)
from kkt_atlas.baselines.saa import LocalSAA, PooledSAA
from kkt_atlas.baselines.sof import FittedSOF, SOFConfig, fit_sof

__all__ = [
    "FittedICEO",
    "FittedSOF",
    "FullSupportICEOModel",
    "ICEOModelConfig",
    "ICEOSnapshot",
    "ICEOTrainingConfig",
    "ICEOTrainingResult",
    "KNearestNeighborSAA",
    "LocalSAA",
    "MatchedObservedSupportResult",
    "ObservedSupportInitialization",
    "ObservedSupportLocalModel",
    "PooledSAA",
    "SOFConfig",
    "fit_iceo",
    "fit_knn_saa",
    "fit_matched_observed_support_atlas",
    "fit_sof",
    "select_iceo_checkpoint",
    "snap_centroids_to_observed_scenarios",
    "train_iceo_model",
]
