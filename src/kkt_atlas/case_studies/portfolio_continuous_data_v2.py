"""Continuous-context finite-catalog data for smoothed-CVaR portfolios."""

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

_DGP_VERSION = "smoothed-cvar-portfolio-continuous-context-dgp-v2"


@dataclass(frozen=True)
class ContinuousPortfolioDataConfig:
    """Configuration for the continuous-context portfolio generator."""

    support_size: int
    asset_count: int
    training_size: int
    validation_size: int = 300
    test_size: int = 300
    context_dim: int = 8
    factor_count: int = 6
    seed: int = 0
    master_support_size: int = 5000
    master_asset_count: int = 128
    return_bound: float = 0.25
    probability_bandwidth: float = 1.35
    factor_scale: float = 0.032
    idiosyncratic_scale: float = 0.008
    crash_probability: float = 0.12
    crash_scale: float = 0.075
    nonlinear_scale: float = 0.45
    context_scale: float = 1.0
    catalog_bias_scale: float = 0.10
    conditional_map: str = "smooth_nonlinear_factor_target_v1"

    def __post_init__(self) -> None:
        if self.support_size < 2 or self.support_size > self.master_support_size:
            raise ValueError("support_size is outside the master catalog")
        if self.asset_count < 2 or self.asset_count > self.master_asset_count:
            raise ValueError("asset_count is outside the master catalog")
        if min(self.training_size, self.validation_size, self.test_size) < 1:
            raise ValueError("all sample sizes must be positive")
        if self.context_dim < 1 or self.factor_count < 1:
            raise ValueError("context_dim and factor_count must be positive")
        if not 0.0 < self.return_bound < 1.0:
            raise ValueError("return_bound must lie in (0,1)")
        if self.probability_bandwidth <= 0.0 or self.context_scale <= 0.0:
            raise ValueError("bandwidth and context scale must be positive")
        if min(
            self.factor_scale,
            self.idiosyncratic_scale,
            self.crash_scale,
            self.nonlinear_scale,
            self.catalog_bias_scale,
        ) < 0.0:
            raise ValueError("all variation scales must be nonnegative")
        if not 0.0 <= self.crash_probability <= 1.0:
            raise ValueError("crash_probability must lie in [0,1]")
        if self.conditional_map not in {
            "smooth_nonlinear_factor_target_v1",
            "affine_softmax_v1",
        }:
            raise ValueError("unknown portfolio conditional map")


def generate_continuous_portfolio_data_v2(
    config: ContinuousPortfolioDataConfig,
) -> FiniteCatalogData:
    """Generate nested returns with a smooth conditional probability law."""

    return combine_fitting_and_test_data(
        generate_continuous_portfolio_fitting_data_v2(config),
        generate_continuous_portfolio_test_data_v2(config),
    )


def generate_continuous_portfolio_fitting_data_v2(
    config: ContinuousPortfolioDataConfig,
) -> FittingCatalogData:
    """Generate only the catalog and observed training and validation splits."""

    catalog, scenario_states = _catalog_and_states(config)
    training_contexts = _sample_contexts(config, "training", config.training_size)
    validation_contexts = _sample_contexts(config, "validation", config.validation_size)
    training_probabilities = _conditional_probabilities(
        config,
        training_contexts,
        scenario_states,
    )
    validation_probabilities = _conditional_probabilities(
        config,
        validation_contexts,
        scenario_states,
    )
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
        metadata=_metadata(config, scenario_states),
    )


def generate_continuous_portfolio_test_data_v2(
    config: ContinuousPortfolioDataConfig,
) -> ConditionalTestData:
    """Generate only evaluator-side contexts and exact conditional laws."""

    catalog, scenario_states = _catalog_and_states(config)
    test_contexts = _sample_contexts(config, "test", config.test_size)
    return ConditionalTestData(
        contexts=test_contexts,
        probabilities=_conditional_probabilities(config, test_contexts, scenario_states),
        catalog_hash=array_sha256(catalog),
        metadata=_metadata(config, scenario_states),
    )


def portfolio_probabilities_v2(
    config: ContinuousPortfolioDataConfig,
    contexts: np.ndarray,
) -> np.ndarray:
    """Evaluate exact finite-catalog probabilities at supplied contexts."""

    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != config.context_dim:
        raise ValueError("contexts have the wrong shape")
    if np.any(~np.isfinite(values)):
        raise ValueError("contexts must be finite")
    scenario_states = _scenario_states(config)
    return _conditional_probabilities(config, values, scenario_states)


def _metadata(
    config: ContinuousPortfolioDataConfig,
    scenario_states: np.ndarray,
) -> dict[str, object]:
    return {
        "schema": _DGP_VERSION,
        "support_size": config.support_size,
        "asset_count": config.asset_count,
        "training_size": config.training_size,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "context_dim": config.context_dim,
        "factor_count": config.factor_count,
        "seed": config.seed,
        "master_support_size": config.master_support_size,
        "master_asset_count": config.master_asset_count,
        "return_bound": config.return_bound,
        "probability_bandwidth": config.probability_bandwidth,
        "factor_scale": config.factor_scale,
        "idiosyncratic_scale": config.idiosyncratic_scale,
        "crash_probability": config.crash_probability,
        "crash_scale": config.crash_scale,
        "nonlinear_scale": config.nonlinear_scale,
        "context_scale": config.context_scale,
        "catalog_bias_scale": config.catalog_bias_scale,
        "context_distribution": "continuous_gaussian",
        "conditional_map": config.conditional_map,
        "context_regions": None,
        "scenario_state_hash": _array_hash(scenario_states),
    }


def _conditional_probabilities(
    config: ContinuousPortfolioDataConfig,
    contexts: np.ndarray,
    scenario_states: np.ndarray,
) -> np.ndarray:
    targets = _smooth_state_targets(config, contexts)
    catalog_bias = _rng(config.seed, "catalog-bias").normal(
        scale=config.catalog_bias_scale,
        size=config.master_support_size,
    )[: config.support_size]
    distances = np.sum(
        (targets[:, None, :] - scenario_states[None, :, :]) ** 2,
        axis=2,
    )
    logits = -0.5 * distances / (config.probability_bandwidth**2)
    logits += catalog_bias[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(probabilities, dtype=np.float64)


def _catalog_and_states(
    config: ContinuousPortfolioDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    scenario_states = _scenario_states(config, full=True)
    factors = scenario_states[:, :-1]
    crash = scenario_states[:, -1]
    idiosyncratic = _rng(config.seed, "catalog-idiosyncratic").normal(
        size=(config.master_support_size, config.master_asset_count)
    )
    loadings = _rng(config.seed, "asset-loadings").normal(
        size=(config.master_asset_count, config.factor_count)
    )
    loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    crash_exposures = 0.5 + _rng(config.seed, "crash-exposures").random(
        config.master_asset_count
    )
    asset_alpha = _rng(config.seed, "asset-alpha").normal(
        loc=0.004,
        scale=0.0015,
        size=config.master_asset_count,
    )
    raw_returns = asset_alpha[None, :]
    raw_returns = raw_returns + config.factor_scale * (factors @ loadings.T)
    raw_returns = raw_returns + config.idiosyncratic_scale * idiosyncratic
    raw_returns = raw_returns - (
        config.crash_scale * crash[:, None] * crash_exposures[None, :]
    )
    master_catalog = 0.98 * config.return_bound * np.tanh(
        raw_returns / config.return_bound
    )
    catalog = master_catalog[: config.support_size, : config.asset_count].copy()
    return catalog, scenario_states[: config.support_size].copy()


def _scenario_states(
    config: ContinuousPortfolioDataConfig,
    *,
    full: bool = False,
) -> np.ndarray:
    factors = _rng(config.seed, "catalog-factors").normal(
        size=(config.master_support_size, config.factor_count)
    )
    crash = (
        _rng(config.seed, "catalog-crash").random(config.master_support_size)
        < config.crash_probability
    ).astype(np.float64)
    states = np.column_stack((factors, crash))
    return states if full else states[: config.support_size].copy()


def _sample_contexts(
    config: ContinuousPortfolioDataConfig,
    split: str,
    sample_size: int,
) -> np.ndarray:
    return np.asarray(
        _rng(config.seed, f"{split}-contexts").normal(
            scale=config.context_scale,
            size=(sample_size, config.context_dim),
        ),
        dtype=np.float64,
    )


def _smooth_state_targets(
    config: ContinuousPortfolioDataConfig,
    contexts: np.ndarray,
) -> np.ndarray:
    state_dim = config.factor_count + 1
    normalizer = np.sqrt(float(config.context_dim))
    linear_map = _rng(config.seed, "context-linear-map").normal(
        size=(config.context_dim, state_dim)
    )
    linear = contexts @ linear_map / normalizer
    if config.conditional_map == "affine_softmax_v1":
        return linear
    periodic_map = _rng(config.seed, "context-periodic-map").normal(
        size=(config.context_dim, state_dim)
    )
    curvature_map = _rng(config.seed, "context-curvature-map").normal(
        size=(config.context_dim, state_dim)
    )
    phase = _rng(config.seed, "context-periodic-phase").uniform(
        -np.pi,
        np.pi,
        size=state_dim,
    )
    periodic = np.sin(contexts @ periodic_map / normalizer + phase[None, :])
    centered_square = contexts**2 - config.context_scale**2
    curvature = np.tanh(centered_square @ curvature_map / normalizer)
    nonlinear = 0.65 * periodic + 0.35 * curvature
    targets = np.tanh(linear + config.nonlinear_scale * nonlinear)
    # The last scenario-state coordinate is a binary crash indicator. Mapping
    # the smooth target to (0,1) puts it on the same scale without discretizing
    # the contexts themselves.
    targets[:, -1] = 0.5 * (targets[:, -1] + 1.0)
    return targets


def _sample_indices(
    probabilities: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    uniforms = generator.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int64)


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rng(seed: int, token: str) -> np.random.Generator:
    payload = f"{_DGP_VERSION}|{int(seed)}|{token}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)
