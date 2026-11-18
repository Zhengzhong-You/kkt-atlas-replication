"""Finite-catalog data for contextual multicommodity network reservation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from kkt_atlas.confirmation_data_v1 import (
    ConditionalTestData,
    FittingCatalogData,
    array_sha256,
    combine_fitting_and_test_data,
)
from kkt_atlas.data import FiniteCatalogData, ObservedSplit

_DGP_VERSION = "multicommodity-network-reservation-dgp-v1"
_MASTER_SCENARIO_DIM = 128
_REGIME_MATRIX = np.asarray(
    (
        (1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class NetworkReservationDataConfig:
    """Frozen controls for one nested NET finite-catalog instance."""

    support_size: int
    scenario_dim: int
    training_size: int
    validation_size: int = 500
    test_size: int = 1000
    context_dim: int = 8
    latent_rank: int = 6
    seed: int = 0
    master_support_size: int = 10_000
    master_scenario_dim: int = _MASTER_SCENARIO_DIM
    context_slope: float = 3.0
    crash_probability: float = 0.12
    factor_scale: float = 0.35
    group_factor_scale: float = 0.40
    idiosyncratic_scale: float = 0.20
    shock_scale: float = 0.90
    baseline_demand_minimum: float = 0.04
    baseline_demand_maximum: float = 0.06

    def __post_init__(self) -> None:
        if (
            self.support_size < 2
            or self.support_size % 2
            or self.support_size > self.master_support_size
        ):
            raise ValueError(
                "support_size must be an even prefix of the master catalog"
            )
        if self.master_support_size < 2 or self.master_support_size % 2:
            raise ValueError("master_support_size must be an even integer")
        if not 1 <= self.scenario_dim <= self.master_scenario_dim:
            raise ValueError("scenario_dim is outside the master catalog")
        if self.master_scenario_dim != _MASTER_SCENARIO_DIM:
            raise ValueError("the NET master catalog must have 128 coordinates")
        if min(self.training_size, self.validation_size, self.test_size) < 1:
            raise ValueError("all sample sizes must be positive")
        if self.context_dim != 8:
            raise ValueError("the NET context dimension must equal eight")
        if self.latent_rank != 6:
            raise ValueError("the NET latent rank must equal six")
        if not 0.0 <= self.crash_probability <= 1.0:
            raise ValueError("crash_probability must lie in [0,1]")
        if min(
            self.context_slope,
            self.factor_scale,
            self.group_factor_scale,
            self.idiosyncratic_scale,
            self.shock_scale,
        ) <= 0.0:
            raise ValueError("all NET DGP scales must be positive")
        if not 0.0 < self.baseline_demand_minimum < self.baseline_demand_maximum:
            raise ValueError("NET baseline demand bounds must be positive and ordered")


def generate_network_reservation_data_v1(
    config: NetworkReservationDataConfig,
) -> FiniteCatalogData:
    """Generate all splits after verifying their shared catalog identity."""

    return combine_fitting_and_test_data(
        generate_network_reservation_fitting_data_v1(config),
        generate_network_reservation_test_data_v1(config),
    )


def generate_network_reservation_fitting_data_v1(
    config: NetworkReservationDataConfig,
) -> FittingCatalogData:
    """Generate the catalog and observed training and validation splits."""

    catalog, scores = _catalog_and_scores(config)
    training_contexts = _sample_contexts(config, "training", config.training_size)
    validation_contexts = _sample_contexts(
        config,
        "validation",
        config.validation_size,
    )
    training_probabilities = _probabilities(config, training_contexts, scores)
    validation_probabilities = _probabilities(config, validation_contexts, scores)
    training_indices = _sample_indices(
        training_probabilities,
        _rng(config.seed, "training-outcomes"),
    )
    validation_indices = _sample_indices(
        validation_probabilities,
        _rng(config.seed, "validation-outcomes"),
    )
    return FittingCatalogData(
        catalog=catalog,
        training=ObservedSplit(training_contexts, training_indices),
        validation=ObservedSplit(validation_contexts, validation_indices),
        metadata=_metadata(config),
    )


def generate_network_reservation_test_data_v1(
    config: NetworkReservationDataConfig,
) -> ConditionalTestData:
    """Generate evaluator-only contexts and exact conditional probabilities."""

    catalog, scores = _catalog_and_scores(config)
    test_contexts = _sample_contexts(config, "test", config.test_size)
    return ConditionalTestData(
        contexts=test_contexts,
        probabilities=_probabilities(config, test_contexts, scores),
        catalog_hash=array_sha256(catalog),
        metadata=_metadata(config),
    )


def network_probabilities_v1(
    config: NetworkReservationDataConfig,
    contexts: np.ndarray,
) -> np.ndarray:
    """Evaluate the exact NET finite-catalog probabilities at contexts."""

    _, scores = _catalog_and_scores(config)
    return _probabilities(config, contexts, scores)


def _catalog_and_scores(
    config: NetworkReservationDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    factors = _rng(config.seed, "catalog-factors").normal(
        size=(config.master_support_size, config.latent_rank)
    )
    idiosyncratic = _rng(config.seed, "catalog-idiosyncratic").normal(
        size=(config.master_support_size, config.master_scenario_dim)
    )
    shock = (
        _rng(config.seed, "catalog-shock").random(config.master_support_size)
        < config.crash_probability
    ).astype(np.float64)
    shock_type = _rng(config.seed, "catalog-shock-type").integers(
        0,
        4,
        size=config.master_support_size,
    )
    coordinate = np.arange(config.master_scenario_dim, dtype=np.float64)
    baseline = config.baseline_demand_minimum + coordinate * (
        config.baseline_demand_maximum - config.baseline_demand_minimum
    ) / 127.0
    group = np.arange(config.master_scenario_dim, dtype=np.int64) % 4
    log_multiplier = (
        config.factor_scale * factors[:, 0, None]
        + config.group_factor_scale * factors[:, 1 + group]
        + config.idiosyncratic_scale * idiosyncratic
        + config.shock_scale
        * shock[:, None]
        * _REGIME_MATRIX[shock_type][:, group]
        - 0.5
        * (
            config.factor_scale**2
            + config.group_factor_scale**2
            + config.idiosyncratic_scale**2
        )
    )
    master = baseline[None, :] * np.exp(log_multiplier)
    scores = np.mean(master / baseline[None, :], axis=1)
    return (
        np.ascontiguousarray(
            master[: config.support_size, : config.scenario_dim]
        ),
        np.ascontiguousarray(scores[: config.support_size]),
    )


def _probabilities(
    config: NetworkReservationDataConfig,
    contexts: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != config.context_dim:
        raise ValueError("contexts have the wrong shape")
    if np.any(~np.isfinite(values)):
        raise ValueError("contexts must be finite")
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_values.shape != (config.support_size,) or np.any(
        ~np.isfinite(score_values)
    ):
        raise ValueError("scenario scores do not match the support")
    order = np.lexsort((np.arange(config.support_size), score_values))
    low = order[: config.support_size // 2]
    high = order[config.support_size // 2 :]
    index = values.sum(axis=1) / np.sqrt(float(config.context_dim))
    high_mass = _sigmoid(config.context_slope * index)
    probabilities = np.empty(
        (len(values), config.support_size),
        dtype=np.float64,
    )
    probabilities[:, low] = (
        2.0 * (1.0 - high_mass) / config.support_size
    )[:, None]
    probabilities[:, high] = (
        2.0 * high_mass / config.support_size
    )[:, None]
    return np.ascontiguousarray(probabilities)


def _sample_contexts(
    config: NetworkReservationDataConfig,
    split: str,
    sample_size: int,
) -> np.ndarray:
    return np.asarray(
        _rng(config.seed, f"{split}-contexts").normal(
            size=(sample_size, config.context_dim)
        ),
        dtype=np.float64,
    )


def _sample_indices(
    probabilities: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    uniforms = generator.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int64)


def _metadata(config: NetworkReservationDataConfig) -> dict[str, object]:
    return {
        "schema": _DGP_VERSION,
        "support_size": config.support_size,
        "scenario_dim": config.scenario_dim,
        "training_size": config.training_size,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "context_dim": config.context_dim,
        "seed": config.seed,
        "master_support_size": config.master_support_size,
        "master_scenario_dim": config.master_scenario_dim,
        "baseline_demand_range": [
            config.baseline_demand_minimum,
            config.baseline_demand_maximum,
        ],
        "latent_rank": config.latent_rank,
        "shared_factor_scale": config.factor_scale,
        "group_factor_scale": config.group_factor_scale,
        "idiosyncratic_scale": config.idiosyncratic_scale,
        "shock_probability": config.crash_probability,
        "shock_scale": config.shock_scale,
        "regime_logit_scale": config.context_slope,
        "context_distribution": "standard_gaussian",
        "conditional_map": "two_regime_single_index_v1",
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponential = np.exp(array[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _rng(seed: int, token: str) -> np.random.Generator:
    payload = f"{_DGP_VERSION}|{int(seed)}|{token}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)
