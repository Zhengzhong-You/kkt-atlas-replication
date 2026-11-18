"""Continuous-context finite-catalog data for smooth capacity allocation."""

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

_DGP_VERSION = "smooth-capacity-continuous-context-dgp-v2"


@dataclass(frozen=True)
class ContinuousCapacityDataConfig:
    """Configuration for the version-2 smooth-capacity data generator."""

    support_size: int
    scenario_dim: int
    training_size: int
    validation_size: int = 500
    test_size: int = 500
    context_dim: int = 8
    latent_rank: int = 6
    seed: int = 0
    master_support_size: int = 5000
    master_scenario_dim: int = 128
    base_demand: float = 5.0
    probability_bandwidth: float = 1.4
    nonlinear_scale: float = 0.45
    latent_scale: float = 0.22
    idiosyncratic_scale: float = 0.035
    context_scale: float = 1.0
    catalog_bias_scale: float = 0.08

    def __post_init__(self) -> None:
        if self.support_size < 2 or self.support_size > self.master_support_size:
            raise ValueError("support_size is outside the master catalog")
        if self.scenario_dim < 2 or self.scenario_dim > self.master_scenario_dim:
            raise ValueError("scenario_dim is outside the master catalog")
        if min(self.training_size, self.validation_size, self.test_size) < 1:
            raise ValueError("all sample sizes must be positive")
        if self.context_dim < 1 or self.latent_rank < 1:
            raise ValueError("context_dim and latent_rank must be positive")
        if self.base_demand <= 0.0 or self.probability_bandwidth <= 0.0:
            raise ValueError("base_demand and probability_bandwidth must be positive")
        if self.context_scale <= 0.0:
            raise ValueError("context_scale must be positive")
        if (
            min(
                self.nonlinear_scale,
                self.latent_scale,
                self.idiosyncratic_scale,
                self.catalog_bias_scale,
            )
            < 0.0
        ):
            raise ValueError("all variation scales must be nonnegative")


def generate_continuous_capacity_data_v2(
    config: ContinuousCapacityDataConfig,
) -> FiniteCatalogData:
    """Generate a nested catalog with a smooth conditional probability law."""

    return combine_fitting_and_test_data(
        generate_continuous_capacity_fitting_data_v2(config),
        generate_continuous_capacity_test_data_v2(config),
    )


def generate_continuous_capacity_fitting_data_v2(
    config: ContinuousCapacityDataConfig,
) -> FittingCatalogData:
    """Generate only the catalog and observed training and validation splits."""

    catalog = _catalog(config)
    training_contexts = _sample_contexts(config, "training", config.training_size)
    validation_contexts = _sample_contexts(config, "validation", config.validation_size)

    training_probabilities = capacity_probabilities_v2(config, training_contexts)
    validation_probabilities = capacity_probabilities_v2(config, validation_contexts)
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


def generate_continuous_capacity_test_data_v2(
    config: ContinuousCapacityDataConfig,
) -> ConditionalTestData:
    """Generate only evaluator-side contexts and exact conditional laws."""

    catalog = _catalog(config)
    test_contexts = _sample_contexts(config, "test", config.test_size)
    return ConditionalTestData(
        contexts=test_contexts,
        probabilities=capacity_probabilities_v2(config, test_contexts),
        catalog_hash=array_sha256(catalog),
        metadata=_metadata(config),
    )


def capacity_probabilities_v2(
    config: ContinuousCapacityDataConfig,
    contexts: np.ndarray,
) -> np.ndarray:
    """Evaluate the exact finite-catalog probabilities at given contexts."""

    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != config.context_dim:
        raise ValueError("contexts have the wrong shape")
    if np.any(~np.isfinite(values)):
        raise ValueError("contexts must be finite")

    targets = _smooth_latent_targets(config, values)
    catalog_latent = _rng(config.seed, "catalog-latent").normal(size=(config.master_support_size, config.latent_rank))[
        : config.support_size
    ]
    catalog_bias = _rng(config.seed, "catalog-bias").normal(
        scale=config.catalog_bias_scale,
        size=config.master_support_size,
    )[: config.support_size]
    squared_distances = np.sum(
        (targets[:, None, :] - catalog_latent[None, :, :]) ** 2,
        axis=2,
    )
    logits = -0.5 * squared_distances / (config.probability_bandwidth**2)
    logits = logits + catalog_bias[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(probabilities, dtype=np.float64)


def _metadata(config: ContinuousCapacityDataConfig) -> dict[str, object]:
    return {
        "schema": _DGP_VERSION,
        "support_size": config.support_size,
        "scenario_dim": config.scenario_dim,
        "training_size": config.training_size,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "context_dim": config.context_dim,
        "latent_rank": config.latent_rank,
        "seed": config.seed,
        "master_support_size": config.master_support_size,
        "master_scenario_dim": config.master_scenario_dim,
        "base_demand": config.base_demand,
        "probability_bandwidth": config.probability_bandwidth,
        "nonlinear_scale": config.nonlinear_scale,
        "latent_scale": config.latent_scale,
        "idiosyncratic_scale": config.idiosyncratic_scale,
        "context_scale": config.context_scale,
        "catalog_bias_scale": config.catalog_bias_scale,
        "context_distribution": "continuous_gaussian",
        "conditional_map": "smooth_nonlinear_latent_v1",
    }


def _catalog(config: ContinuousCapacityDataConfig) -> np.ndarray:
    latent = _rng(config.seed, "catalog-latent").normal(size=(config.master_support_size, config.latent_rank))
    idiosyncratic = _rng(config.seed, "catalog-idiosyncratic").normal(
        size=(config.master_support_size, config.master_scenario_dim)
    )
    loadings = _rng(config.seed, "demand-loadings").normal(size=(config.master_scenario_dim, config.latent_rank))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    coordinate = np.arange(config.master_scenario_dim, dtype=np.float64)
    offsets = 0.06 * np.sin(2.0 * np.pi * coordinate / config.master_scenario_dim)
    log_catalog = np.log(config.base_demand) + offsets[None, :]
    log_catalog = log_catalog + config.latent_scale * (latent @ loadings.T)
    log_catalog = log_catalog + config.idiosyncratic_scale * idiosyncratic
    return np.exp(log_catalog[: config.support_size, : config.scenario_dim]).copy()


def _sample_contexts(
    config: ContinuousCapacityDataConfig,
    split: str,
    sample_size: int,
) -> np.ndarray:
    contexts = _rng(config.seed, f"{split}-contexts").normal(
        scale=config.context_scale,
        size=(sample_size, config.context_dim),
    )
    return np.asarray(contexts, dtype=np.float64)


def _smooth_latent_targets(
    config: ContinuousCapacityDataConfig,
    contexts: np.ndarray,
) -> np.ndarray:
    normalizer = np.sqrt(float(config.context_dim))
    linear_map = _rng(config.seed, "context-linear-map").normal(size=(config.context_dim, config.latent_rank))
    periodic_map = _rng(config.seed, "context-periodic-map").normal(size=(config.context_dim, config.latent_rank))
    curvature_map = _rng(config.seed, "context-curvature-map").normal(size=(config.context_dim, config.latent_rank))
    phase = _rng(config.seed, "context-periodic-phase").uniform(
        -np.pi,
        np.pi,
        size=config.latent_rank,
    )
    linear = contexts @ linear_map / normalizer
    periodic = np.sin(contexts @ periodic_map / normalizer + phase[None, :])
    centered_square = contexts**2 - config.context_scale**2
    curvature = np.tanh(centered_square @ curvature_map / normalizer)
    nonlinear = 0.65 * periodic + 0.35 * curvature
    return np.tanh(linear + config.nonlinear_scale * nonlinear)


def _sample_indices(
    probabilities: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    uniforms = generator.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int64)


def _rng(seed: int, token: str) -> np.random.Generator:
    payload = f"{_DGP_VERSION}|{int(seed)}|{token}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)
