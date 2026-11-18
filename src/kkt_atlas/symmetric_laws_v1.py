"""Load fitted campaign methods and expose their predicted finite laws."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from kkt_atlas.baselines.confirmation_baseline_serialization import (
    load_confirmation_baseline_bundle,
    load_matched_observed_support_bundle,
)
from kkt_atlas.baselines.iceo_serialization import load_fitted_iceo
from kkt_atlas.baselines.sof import load_fitted_sof
from kkt_atlas.confirmation_data_v1 import FittingCatalogData
from kkt_atlas.problem import Array, DifferentiableStochasticProblem, matrix, probability_vector
from kkt_atlas.serialization import load_fitted_atlas

_ATLAS_METHODS = {
    "kkt_atlas",
    "fixed_b1",
    "matched_observed_support",
    "cross",
    "pca",
}
_METHODS = _ATLAS_METHODS | {"iceo_a", "knn", "sof"}


@dataclass(frozen=True)
class FiniteLaw:
    scenarios: Array
    probabilities: Array
    support_indices: Array
    support_kind: Literal["local", "catalog"]

    def __post_init__(self) -> None:
        scenarios = np.asarray(self.scenarios, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64).reshape(-1)
        indices = np.asarray(self.support_indices, dtype=np.int64).reshape(-1)
        if (
            scenarios.ndim != 2
            or len(scenarios) == 0
            or len(scenarios) != len(probabilities)
            or indices.shape != probabilities.shape
            or np.any(~np.isfinite(scenarios))
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
        ):
            raise ValueError("finite law arrays are empty, misaligned, or invalid")
        if self.support_kind not in {"local", "catalog"}:
            raise ValueError("support_kind must be local or catalog")
        if np.any(indices < 0) or (len(indices) > 1 and np.any(np.diff(indices) <= 0)):
            raise ValueError("support_indices must be nonnegative, unique, and increasing")
        total = float(probabilities.sum())
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("finite-law probabilities must sum to one")
        probabilities = np.ascontiguousarray(probability_vector(probabilities, len(probabilities)))
        scenarios = np.ascontiguousarray(scenarios)
        indices = np.ascontiguousarray(indices)
        for values in (scenarios, probabilities, indices):
            values.setflags(write=False)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "support_indices", indices)


def load_method_model(
    method: str,
    path: str | Path,
    problem: DifferentiableStochasticProblem,
    fitting: FittingCatalogData,
) -> Any:
    """Load one fitted method without evaluating its decision mapping."""

    if method not in _METHODS:
        raise ValueError(f"unsupported campaign method: {method}")
    if method in {"kkt_atlas", "fixed_b1", "cross", "pca"}:
        return load_fitted_atlas(path, problem).prepare()
    if method == "matched_observed_support":
        return load_matched_observed_support_bundle(path, problem).atlas
    if method == "iceo_a":
        return load_fitted_iceo(path, problem)
    if method == "knn":
        return load_confirmation_baseline_bundle(path, problem).knn_saa
    return load_fitted_sof(
        path,
        problem,
        fitting.training.contexts,
        fitting.training.outcomes(fitting.catalog),
    )


def predict_method_laws(
    model: Any,
    method: str,
    contexts: Array,
    fitting: FittingCatalogData,
) -> tuple[FiniteLaw, ...]:
    """Return one canonically ordered finite law per query context."""

    if method not in _METHODS:
        raise ValueError(f"unsupported campaign method: {method}")
    problem = _problem(model, method)
    queries = matrix(contexts, columns=problem.context_dim, name="contexts")
    if method in _ATLAS_METHODS:
        return _atlas_laws(model, queries)
    if method == "iceo_a":
        return _iceo_laws(model, queries, fitting)
    if method == "knn":
        return _knn_laws(model, queries, fitting)
    return _sof_laws(model, queries, fitting)


def _problem(model: Any, method: str) -> DifferentiableStochasticProblem:
    if method == "iceo_a":
        return model.model.problem
    return model.problem


def _atlas_laws(model: Any, contexts: Array) -> tuple[FiniteLaw, ...]:
    assignments = model.charts.assign(contexts)
    result: list[FiniteLaw | None] = [None] * len(contexts)
    for chart, local_model in enumerate(model.models):
        rows = np.flatnonzero(assignments == chart)
        if len(rows) == 0:
            continue
        device = local_model.context_mean.device
        dtype = local_model.context_mean.dtype
        with torch.no_grad():
            scenarios = local_model.scenarios().detach().cpu().numpy().copy()
            probabilities = (
                local_model.probabilities(torch.as_tensor(contexts[rows], dtype=dtype, device=device))
                .detach()
                .cpu()
                .numpy()
            )
        indices = np.arange(len(scenarios), dtype=np.int64)
        for position, row in enumerate(rows):
            result[int(row)] = FiniteLaw(scenarios, probabilities[position], indices, "local")
    if any(law is None for law in result):
        raise RuntimeError("atlas did not produce a law for every context")
    return tuple(law for law in result if law is not None)


def _iceo_laws(model: Any, contexts: Array, fitting: FittingCatalogData) -> tuple[FiniteLaw, ...]:
    catalog = model.model.catalog.detach().cpu().numpy()
    _require_catalog(catalog, fitting.catalog, "ICEO-A")
    probabilities = model.probabilities(contexts)
    indices = np.arange(len(catalog), dtype=np.int64)
    return tuple(
        FiniteLaw(catalog, probabilities[row], indices, "catalog")
        for row in range(len(contexts))
    )


def _knn_laws(model: Any, contexts: Array, fitting: FittingCatalogData) -> tuple[FiniteLaw, ...]:
    _require_empirical_training(model, fitting, "k-NN")
    neighbors = model.neighbor_indices(contexts)
    laws = []
    for row in neighbors:
        catalog_indices, counts = np.unique(
            model.training_outcome_indices[row],
            return_counts=True,
        )
        laws.append(
            FiniteLaw(
                fitting.catalog[catalog_indices],
                counts.astype(np.float64) / len(row),
                catalog_indices,
                "catalog",
            )
        )
    return tuple(laws)


def _sof_laws(model: Any, contexts: Array, fitting: FittingCatalogData) -> tuple[FiniteLaw, ...]:
    _require_empirical_training(model, fitting, "SOF")
    observation_weights = model.weights(contexts)
    laws = []
    for row in observation_weights:
        catalog_mass = np.bincount(
            fitting.training.outcome_indices,
            weights=row,
            minlength=len(fitting.catalog),
        )
        catalog_indices = np.flatnonzero(catalog_mass > 0.0)
        laws.append(
            FiniteLaw(
                fitting.catalog[catalog_indices],
                catalog_mass[catalog_indices],
                catalog_indices,
                "catalog",
            )
        )
    return tuple(laws)


def _require_empirical_training(model: Any, fitting: FittingCatalogData, label: str) -> None:
    model_catalog = getattr(model, "catalog", fitting.catalog)
    _require_catalog(model_catalog, fitting.catalog, label)
    model_contexts = getattr(model, "training_contexts", getattr(model, "contexts", None))
    if model_contexts is None or not np.array_equal(model_contexts, fitting.training.contexts):
        raise ValueError(f"{label} model does not match fitting contexts")
    model_indices = getattr(model, "training_outcome_indices", None)
    if model_indices is None:
        model_outcomes = getattr(model, "outcomes", None)
        expected = fitting.training.outcomes(fitting.catalog)
        if model_outcomes is None or not np.array_equal(model_outcomes, expected):
            raise ValueError(f"{label} model does not match fitting outcomes")
    elif not np.array_equal(model_indices, fitting.training.outcome_indices):
        raise ValueError(f"{label} model does not match fitting outcome indices")


def _require_catalog(observed: Array, expected: Array, label: str) -> None:
    if not np.array_equal(np.asarray(observed), np.asarray(expected)):
        raise ValueError(f"{label} model does not match the fitting catalog")
