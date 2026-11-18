"""Contextual empirical baseline based on nearest-neighbor prescriptions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch

from kkt_atlas.data import FiniteCatalogData, ObservedSplit
from kkt_atlas.problem import Array, DifferentiableStochasticProblem, matrix


@dataclass(frozen=True)
class KNearestNeighborSAA:
    """Validation-selected nearest-neighbor sample-average approximation."""

    problem: DifferentiableStochasticProblem
    catalog: Array
    training_contexts: Array
    training_outcome_indices: Array
    context_mean: Array
    context_scale: Array
    neighbor_count: int
    ridge: float
    validation_loss: float

    def __post_init__(self) -> None:
        catalog = matrix(
            self.catalog,
            columns=self.problem.scenario_dim,
            name="catalog",
        )
        contexts = matrix(
            self.training_contexts,
            columns=self.problem.context_dim,
            name="training_contexts",
        )
        indices = np.asarray(self.training_outcome_indices, dtype=np.int64).reshape(-1)
        mean = np.asarray(self.context_mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.context_scale, dtype=np.float64).reshape(-1)
        if indices.shape != (len(contexts),):
            raise ValueError("training outcome indices must align with contexts")
        if np.any(indices < 0) or np.any(indices >= len(catalog)):
            raise ValueError("training outcome index is outside the catalog")
        if mean.shape != (self.problem.context_dim,) or scale.shape != mean.shape:
            raise ValueError("context standardization has the wrong shape")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)):
            raise ValueError("context standardization must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("context scales must be positive")
        if not 1 <= self.neighbor_count <= len(contexts):
            raise ValueError("neighbor_count must lie between one and the training size")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        if not math.isfinite(self.validation_loss):
            raise ValueError("validation_loss must be finite")
        object.__setattr__(self, "catalog", _immutable(catalog))
        object.__setattr__(self, "training_contexts", _immutable(contexts))
        object.__setattr__(self, "training_outcome_indices", _immutable_int(indices))
        object.__setattr__(self, "context_mean", _immutable(mean))
        object.__setattr__(self, "context_scale", _immutable(scale))

    def neighbor_indices(self, contexts: Array) -> Array:
        """Return deterministic nearest-neighbor training indices per context."""

        query = matrix(
            contexts,
            columns=self.problem.context_dim,
            name="contexts",
        )
        training = (self.training_contexts - self.context_mean) / self.context_scale
        standardized = (query - self.context_mean) / self.context_scale
        result: Array = np.empty(
            (len(query), self.neighbor_count),
            dtype=np.int64,
        )
        observation_indices: Array = np.arange(
            len(training),
            dtype=np.int64,
        )
        for row, value in enumerate(standardized):
            distances = np.sum((training - value) ** 2, axis=1)
            order = np.lexsort((observation_indices, distances))
            result[row] = order[: self.neighbor_count]
        return result

    def decisions(self, contexts: Array) -> Array:
        """Solve one neighbor-weighted stochastic program for each context."""

        query = matrix(
            contexts,
            columns=self.problem.context_dim,
            name="contexts",
        )
        neighbors = self.neighbor_indices(query)
        parallel = getattr(self.problem,"solve_laws_with_audit",None)
        if parallel is not None:
            from types import SimpleNamespace
            laws=[]
            for training_indices in neighbors:
                unique,counts=np.unique(self.training_outcome_indices[training_indices],return_counts=True)
                laws.append(SimpleNamespace(scenarios=self.catalog[unique],
                    probabilities=counts.astype(np.float64)/self.neighbor_count))
            decisions,_=parallel(query,laws,ridge=self.ridge)
            if np.any(~np.isfinite(decisions)):
                raise FloatingPointError("KNN-SAA produced a nonfinite decision")
            return decisions
        decisions = np.empty(
            (len(query), self.problem.decision_dim),
            dtype=np.float64,
        )
        for row, training_indices in enumerate(neighbors):
            outcome_indices = self.training_outcome_indices[training_indices]
            unique, counts = np.unique(outcome_indices, return_counts=True)
            probabilities = counts.astype(np.float64) / self.neighbor_count
            decisions[row] = self.problem.solve_numpy(
                query[row],
                self.catalog[unique],
                probabilities,
                ridge=self.ridge,
            )
        if np.any(~np.isfinite(decisions)):
            raise FloatingPointError("KNN-SAA produced a nonfinite decision")
        return decisions


def fit_knn_saa(
    problem: DifferentiableStochasticProblem,
    data: FiniteCatalogData,
    *,
    neighbor_counts: tuple[int, ...],
    ridges: tuple[float, ...],
    validation_problem: DifferentiableStochasticProblem | None = None,
    candidate_callback: Callable[[KNearestNeighborSAA, Array, Array], None] | None = None,
) -> KNearestNeighborSAA:
    """Select the neighbor number and ridge by observed validation loss.

    Context standardization and all neighbor searches use the training split.
    Validation realizations are used only to select one candidate. Test
    contexts and test probabilities are never accessed.
    """

    scoring_problem = problem if validation_problem is None else validation_problem
    _validate_problem_data(problem, data)
    _validate_problem_data(scoring_problem, data)
    neighbor_grid = tuple(sorted(set(int(value) for value in neighbor_counts)))
    ridge_grid = tuple(sorted(set(float(value) for value in ridges)))
    if not neighbor_grid or neighbor_grid[0] < 1:
        raise ValueError("neighbor_counts must contain positive integers")
    if neighbor_grid[-1] > len(data.training.contexts):
        raise ValueError("neighbor count exceeds the training size")
    if not ridge_grid or any(
        not math.isfinite(value) or value < 0.0 for value in ridge_grid
    ):
        raise ValueError("ridges must contain nonnegative finite values")

    mean = data.training.contexts.mean(axis=0)
    scale = data.training.contexts.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    candidates: list[KNearestNeighborSAA] = []
    for neighbor_count in neighbor_grid:
        for ridge in ridge_grid:
            candidate = KNearestNeighborSAA(
                problem=scoring_problem,
                catalog=data.catalog,
                training_contexts=data.training.contexts,
                training_outcome_indices=data.training.outcome_indices,
                context_mean=mean,
                context_scale=scale,
                neighbor_count=neighbor_count,
                ridge=ridge,
                validation_loss=0.0,
            )
            decisions = candidate.decisions(data.validation.contexts)
            validation_loss, validation_losses = _observed_loss_values(
                scoring_problem,
                decisions,
                data.validation,
                data.catalog,
            )
            candidates.append(
                KNearestNeighborSAA(
                    problem=scoring_problem,
                    catalog=data.catalog,
                    training_contexts=data.training.contexts,
                    training_outcome_indices=data.training.outcome_indices,
                    context_mean=mean,
                    context_scale=scale,
                    neighbor_count=neighbor_count,
                    ridge=ridge,
                    validation_loss=validation_loss,
                )
            )
            if candidate_callback is not None:
                candidate_callback(candidates[-1], decisions.copy(), validation_losses)
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.validation_loss,
            candidate.neighbor_count,
            candidate.ridge,
        ),
    )
    return replace(selected, problem=problem)


@torch.no_grad()
def _observed_loss_values(
    problem: DifferentiableStochasticProblem,
    decisions: Array,
    split: ObservedSplit,
    catalog: Array,
) -> tuple[float, Array]:
    losses = problem.realized_loss_torch(
        torch.tensor(decisions, dtype=torch.float64),
        torch.tensor(split.outcomes(catalog), dtype=torch.float64),
        torch.tensor(split.contexts, dtype=torch.float64),
    )
    value = float(losses.mean().item())
    if not math.isfinite(value):
        raise FloatingPointError("KNN-SAA validation loss is nonfinite")
    return value, losses.detach().cpu().numpy().copy()


def _validate_problem_data(
    problem: DifferentiableStochasticProblem,
    data: FiniteCatalogData,
) -> None:
    if data.context_dim != problem.context_dim:
        raise ValueError("data and problem context dimensions differ")
    if data.scenario_dim != problem.scenario_dim:
        raise ValueError("data and problem scenario dimensions differ")


def _immutable(values: Array) -> Array:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).copy())
    array.setflags(write=False)
    return array


def _immutable_int(values: Array) -> Array:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64).copy())
    array.setflags(write=False)
    return array
