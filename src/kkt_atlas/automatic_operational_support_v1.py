"""Training-only support selection by initialized operational loss.

The selector conditions on a fixed chart partition and scenario basis.  For
each assessment fold and candidate support size, it clusters only the fitting
scenario coordinates, fits only a lightweight affine-softmax classifier, and
evaluates the resulting initialized stochastic program on the assessment
observations.  It never invokes Adam.

The candidate path is ``1,...,min(q + 1, N_distinct)``, where
``N_distinct`` is the minimum number of distinct projected fitting
observations across the internal folds.  The ``q + 1`` term is the frozen
protocol cap on the path, not the selected support size or a universal
support theorem.  All folds are formed within the training observations;
this calculation is internal resampling, not independent cross-validation
or cross-fitting.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kkt_atlas.automatic_support import fixed_cross_fitting_folds
from kkt_atlas.initialization import cluster_projected_scenarios
from kkt_atlas.probability_initialization import (
    ProbabilityHeadInitialization,
    fit_cluster_probability_head,
)
from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
    realized_loss_from_scenarios_torch,
)
from kkt_atlas.response_basis import ResponseDecomposition


@dataclass(frozen=True)
class OperationalSupportFoldDiagnostic:
    """Operational assessment score for one fold and support size."""

    fold: int
    support_size: int
    fitting_sample_size: int
    assessment_sample_size: int
    assessment_operational_loss: float
    assessment_operational_loss_sum: float
    kmeans_seed: int
    head_converged: bool
    head_iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "support_size": self.support_size,
            "fitting_sample_size": self.fitting_sample_size,
            "assessment_sample_size": self.assessment_sample_size,
            "assessment_operational_loss": self.assessment_operational_loss,
            "assessment_operational_loss_sum": (
                self.assessment_operational_loss_sum
            ),
            "kmeans_seed": self.kmeans_seed,
            "head_converged": self.head_converged,
            "head_iterations": self.head_iterations,
        }


@dataclass(frozen=True)
class OperationalSupportCandidateDiagnostic:
    """Fold-assessed operational loss for one candidate support size."""

    support_size: int
    mean_fold_assessed_operational_loss: float
    folds: tuple[OperationalSupportFoldDiagnostic, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_size": self.support_size,
            "mean_fold_assessed_operational_loss": (
                self.mean_fold_assessed_operational_loss
            ),
            "folds": [item.as_dict() for item in self.folds],
        }


@dataclass(frozen=True)
class OperationalSupportSelection:
    """Training-only operational assessment of fixed q+1 support.

    ``distinct_projected_row_cap`` is ``N_distinct``: the minimum number of
    distinct projected fitting observations across the internal folds.
    """

    selected_support_size: int
    selector_seed: int
    fold_seed: int
    assessment_folds: tuple[tuple[int, ...], ...]
    support_cap: int
    width_based_support_cap: int
    distinct_projected_row_cap: int
    width_based_cap_is_admitted: bool
    best_fold_assessed_operational_loss: float
    fold_count: int
    batched_forward_solve_count: int
    assessed_problem_count: int
    assessed_scenario_term_count: int
    scenario_kmeans_fit_count: int
    probability_head_fit_count: int
    probability_head_iteration_count: int
    slope_penalty: float
    maximum_head_iterations: int
    accept_intercept_only_fallback: bool
    distinct_projected_tolerance: float
    candidates: tuple[OperationalSupportCandidateDiagnostic, ...]
    support_cap_offset: int = 0

    @property
    def empirical_best_support_size(self) -> int:
        """Compatibility alias for the unique production support."""

        return self.selected_support_size

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rule": "fixed_q_plus_one_assessed_for_chart_selection",
            "selected_support_size": self.selected_support_size,
            "selector_seed": self.selector_seed,
            "fold_seed": self.fold_seed,
            "assessment_folds": [list(fold) for fold in self.assessment_folds],
            "support_cap": self.support_cap,
            "candidate_support_sizes": [self.support_cap],
            "protocol_support_cap": self.width_based_support_cap,
            "q_plus_one_is_protocol_search_cap": False,
            "q_plus_one_is_fixed_support_size": True,
            "width_based_support_cap": self.width_based_support_cap,
            "distinct_projected_row_cap": self.distinct_projected_row_cap,
            "N_distinct": self.distinct_projected_row_cap,
            "N_distinct_definition": (
                "minimum_distinct_projected_fitting_observations_across_"
                "training_internal_resampling_folds"
            ),
            "width_based_cap_is_admitted": self.width_based_cap_is_admitted,
            "best_fold_assessed_operational_loss": (
                self.best_fold_assessed_operational_loss
            ),
            "fold_count": self.fold_count,
            "slope_penalty": self.slope_penalty,
            "maximum_head_iterations": self.maximum_head_iterations,
            "accept_intercept_only_fallback": self.accept_intercept_only_fallback,
            "distinct_projected_tolerance": self.distinct_projected_tolerance,
            "uses_adam": False,
            "information_set": "training_observations_only",
            "resampling_role": (
                "internal_support_assessment_not_independent_cross_"
                "validation_or_cross_fitting"
            ),
            "work_counts": {
                "batched_forward_solves": self.batched_forward_solve_count,
                "assessed_problem_instances": self.assessed_problem_count,
                "assessed_scenario_terms": self.assessed_scenario_term_count,
                "scenario_kmeans_fits": self.scenario_kmeans_fit_count,
                "probability_head_fits": self.probability_head_fit_count,
                "probability_head_iterations": (
                    self.probability_head_iteration_count
                ),
            },
            "candidates": [item.as_dict() for item in self.candidates],
        }


def select_initialized_operational_support_size(
    problem: DifferentiableStochasticProblem,
    decomposition: ResponseDecomposition,
    contexts: Array,
    outcomes: Array,
    *,
    width: int,
    ridge: float,
    context_mean: Array,
    context_scale: Array,
    standardized_center: Array,
    slope_penalty: float,
    fold_count: int = 5,
    distinct_projected_tolerance: float = 1e-10,
    maximum_head_iterations: int = 500,
    accept_intercept_only_fallback: bool = False,
    seed: int = 0,
    support_cap_offset: int = 0,
    maximum_support_size: int | None = None,
) -> OperationalSupportSelection:
    """Assess fixed q+1 support for chart selection; no support-size search.

    Only the fitting observations of a fold determine its scenario locations
    and probability head.  The assessment observations enter only the
    observation-weighted realized operational loss.  The response
    decomposition and chart normalization are fixed from the full training
    chart.  The fold assessment is therefore a training-only internal-
    resampling criterion, not independent cross-validation, cross-fitting, or
    an independent risk estimate.  The quantity ``N_distinct`` used in the
    support cap is the minimum number of distinct projected observations in
    any fitting portion of these folds.
    """

    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    outcome_values = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if len(context_values) != len(outcome_values):
        raise ValueError("contexts and outcomes must contain the same observations")
    width_value = _positive_integer(width, name="width")
    if isinstance(support_cap_offset, bool) or not isinstance(support_cap_offset, int) or support_cap_offset < 0:
        raise ValueError("support_cap_offset must be a nonnegative integer")
    if maximum_support_size is not None:
        maximum_support_size = _positive_integer(
            maximum_support_size,
            name="maximum_support_size",
        )
    if width_value > decomposition.maximum_width:
        raise ValueError("width exceeds the response decomposition")
    ridge_value = _nonnegative_finite_float(ridge, name="ridge")
    penalty = _nonnegative_finite_float(slope_penalty, name="slope_penalty")
    distinct_tolerance = _nonnegative_finite_float(
        distinct_projected_tolerance,
        name="distinct_projected_tolerance",
    )
    head_iterations = _positive_integer(
        maximum_head_iterations,
        name="maximum_head_iterations",
    )
    if not isinstance(accept_intercept_only_fallback, bool):
        raise TypeError("accept_intercept_only_fallback must be a bool")
    mean = np.asarray(context_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(context_scale, dtype=np.float64).reshape(-1)
    center = np.asarray(standardized_center, dtype=np.float64).reshape(-1)
    if mean.shape != (problem.context_dim,) or scale.shape != (problem.context_dim,):
        raise ValueError("context normalization has the wrong dimension")
    if center.shape != (problem.context_dim,) or np.any(scale <= 0.0):
        raise ValueError("standardized_center and context_scale are invalid")
    if (
        np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(scale))
        or np.any(~np.isfinite(center))
    ):
        raise ValueError("context normalization must be finite")

    fold_seed = _derived_seed(seed, "operational_support_folds")
    folds = fixed_cross_fitting_folds(
        len(context_values),
        fold_count,
        seed=fold_seed,
    )
    all_indices = np.arange(len(context_values), dtype=np.int64)
    fitting_indices = tuple(
        np.setdiff1d(all_indices, assessment, assume_unique=True)
        for assessment in folds
    )
    standardized_contexts = (context_values - mean[None, :]) / scale[None, :]
    centered_contexts = standardized_contexts - center[None, :]
    projected = decomposition.coordinates(outcome_values, width_value, problem)
    distinct_gate = min(
        _count_distinct_rows(projected[indices], tolerance=distinct_tolerance)
        for indices in fitting_indices
    )
    width_based_support_cap = width_value + 1 + support_cap_offset
    support_cap = min(
        width_based_support_cap,
        distinct_gate,
        maximum_support_size if maximum_support_size is not None else width_based_support_cap,
    )
    if support_cap < 1:
        raise ValueError("no support candidate satisfies the distinct-support gate")

    if support_cap_offset != 0 or support_cap != width_value + 1:
        raise ValueError("fixed q+1 violates the sample or distinct-support gate")
    fold_loss_sums = np.empty((1, len(folds)), dtype=np.float64)
    fold_assessment_sizes = np.asarray([len(fold) for fold in folds], dtype=np.int64)
    fold_diagnostics: dict[int, list[OperationalSupportFoldDiagnostic]] = {
        support_size: [] for support_size in (support_cap,)
    }
    batched_forward_solve_count = 0
    assessed_problem_count = 0
    assessed_scenario_term_count = 0
    scenario_kmeans_fit_count = 0
    probability_head_fit_count = 0
    probability_head_iteration_count = 0

    for fold_number, (assessment, fitting) in enumerate(
        zip(folds, fitting_indices, strict=True)
    ):
        for support_size in (support_cap,):
            kmeans_seed = _derived_seed(
                seed,
                "operational_support_kmeans",
                fold_number,
                support_size,
            )
            clustering = cluster_projected_scenarios(
                projected[fitting],
                support_size,
                seed=kmeans_seed,
            )
            scenario_kmeans_fit_count += 1
            scenarios = _decode_coordinates(
                problem,
                decomposition,
                clustering.initialization.coordinates,
                width=width_value,
            )
            head = fit_cluster_probability_head(
                centered_contexts[fitting],
                clustering.assignments,
                support_size,
                slope_penalty=penalty,
                maximum_iterations=head_iterations,
            )
            probability_head_fit_count += 1
            probability_head_iteration_count += head.iterations
            if not head.converged and not accept_intercept_only_fallback:
                raise RuntimeError(
                    "cluster-logistic probability-head fitting failed closed for "
                    f"fold={fold_number}, K={support_size}: {head.fallback_reason}"
                )
            probabilities = _predict_probabilities(
                centered_contexts[assessment],
                head,
            )
            losses = _evaluate_operational_losses(
                problem,
                context_values[assessment],
                outcome_values[assessment],
                scenarios,
                probabilities,
                ridge=ridge_value,
            )
            loss = float(np.mean(losses))
            if not math.isfinite(loss):
                raise FloatingPointError("fold-assessed operational loss is nonfinite")
            loss_sum = float(np.sum(losses))
            fold_loss_sums[0, fold_number] = loss_sum
            batched_forward_solve_count += 1
            assessed_problem_count += len(assessment)
            assessed_scenario_term_count += len(assessment) * support_size
            fold_diagnostics[support_size].append(
                OperationalSupportFoldDiagnostic(
                    fold=fold_number,
                    support_size=support_size,
                    fitting_sample_size=len(fitting),
                    assessment_sample_size=len(assessment),
                    assessment_operational_loss=loss,
                    assessment_operational_loss_sum=loss_sum,
                    kmeans_seed=kmeans_seed,
                    head_converged=head.converged,
                    head_iterations=head.iterations,
                )
            )

    mean_losses, selected_index = _smallest_observation_weighted_minimizer(
        fold_loss_sums,
        fold_assessment_sizes,
    )
    candidates: list[OperationalSupportCandidateDiagnostic] = []
    for support_index in range(1):
        support_size = support_cap
        candidates.append(
            OperationalSupportCandidateDiagnostic(
                support_size=support_size,
                mean_fold_assessed_operational_loss=float(
                    mean_losses[support_index]
                ),
                folds=tuple(fold_diagnostics[support_size]),
            )
        )

    expected_batch_count = len(folds)
    expected_problem_count = len(context_values)
    expected_scenario_terms = len(context_values) * sum(
        (support_cap,)
    )
    if (
        batched_forward_solve_count != expected_batch_count
        or assessed_problem_count != expected_problem_count
        or assessed_scenario_term_count != expected_scenario_terms
    ):
        raise RuntimeError("operational support work accounting is inconsistent")
    return OperationalSupportSelection(
        selected_support_size=support_cap,
        selector_seed=int(seed),
        fold_seed=fold_seed,
        assessment_folds=tuple(
            tuple(int(index) for index in fold) for fold in folds
        ),
        support_cap=support_cap,
        width_based_support_cap=width_based_support_cap,
        distinct_projected_row_cap=distinct_gate,
        width_based_cap_is_admitted=(width_based_support_cap <= distinct_gate),
        best_fold_assessed_operational_loss=float(mean_losses[selected_index]),
        fold_count=len(folds),
        batched_forward_solve_count=batched_forward_solve_count,
        assessed_problem_count=assessed_problem_count,
        assessed_scenario_term_count=assessed_scenario_term_count,
        scenario_kmeans_fit_count=scenario_kmeans_fit_count,
        probability_head_fit_count=probability_head_fit_count,
        probability_head_iteration_count=probability_head_iteration_count,
        slope_penalty=penalty,
        maximum_head_iterations=head_iterations,
        accept_intercept_only_fallback=accept_intercept_only_fallback,
        distinct_projected_tolerance=distinct_tolerance,
        candidates=tuple(candidates),
        support_cap_offset=support_cap_offset,
    )


def _predict_probabilities(
    centered_standardized_contexts: Array,
    head: ProbabilityHeadInitialization,
) -> Array:
    contexts = np.asarray(centered_standardized_contexts, dtype=np.float64)
    if contexts.ndim != 2 or np.any(~np.isfinite(contexts)):
        raise ValueError("centered_standardized_contexts must be a finite matrix")
    if contexts.shape[1] != head.slopes.shape[1]:
        raise ValueError("context dimension does not match the fitted probability head")
    logits = head.intercepts[None, :] + contexts @ head.slopes.T
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(probabilities)):
        raise FloatingPointError("probability-head predictions are nonfinite")
    return probabilities


def _evaluate_operational_losses(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    scenarios: Array,
    probabilities: Array,
    *,
    ridge: float,
) -> Array:
    context_tensor = torch.tensor(contexts, dtype=torch.float64)
    outcome_tensor = torch.tensor(outcomes, dtype=torch.float64)
    scenario_tensor = torch.tensor(scenarios, dtype=torch.float64)
    probability_tensor = torch.tensor(probabilities, dtype=torch.float64)
    with torch.no_grad():
        losses = realized_loss_from_scenarios_torch(
            problem,
            context_tensor,
            scenario_tensor,
            probability_tensor,
            outcome_tensor,
            ridge=ridge,
        )
    loss_values = np.asarray(losses.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    if loss_values.shape != (len(contexts),):
        raise ValueError("realized operational losses must contain one value per observation")
    if np.any(~np.isfinite(loss_values)):
        raise FloatingPointError("the operational assessment returned nonfinite values")
    return loss_values


def _decode_coordinates(
    problem: DifferentiableStochasticProblem,
    decomposition: ResponseDecomposition,
    coordinates: Array,
    *,
    width: int,
) -> Array:
    values = matrix(coordinates, columns=width, name="scenario coordinates")
    if decomposition.basis_method == "identity":
        standardized = values
    else:
        standardized = values @ decomposition.basis(width).T
    transformed = (
        decomposition.transformed_mean[None, :]
        + standardized * decomposition.transformed_scale[None, :]
    )
    with torch.no_grad():
        decoded = problem.decode_torch(
            torch.tensor(transformed, dtype=torch.float64)
        )
    return matrix(
        decoded.detach().cpu().numpy(),
        columns=problem.scenario_dim,
        name="decoded scenarios",
    )


def _count_distinct_rows(values: Array, *, tolerance: float) -> int:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0 or np.any(~np.isfinite(rows)):
        raise ValueError("projected coordinates must be a nonempty finite matrix")
    order = np.lexsort(
        tuple(rows[:, column] for column in reversed(range(rows.shape[1])))
    )
    representatives: list[Array] = []
    for row in rows[order]:
        if not representatives:
            representatives.append(row)
            continue
        distances = np.max(
            np.abs(np.vstack(representatives) - row[None, :]),
            axis=1,
        )
        if np.all(distances > tolerance):
            representatives.append(row)
    return len(representatives)


def _smallest_observation_weighted_minimizer(
    fold_loss_sums: Array,
    fold_assessment_sizes: Array,
) -> tuple[Array, int]:
    """Return aggregate losses and the smallest exact minimizer."""

    sums = np.asarray(fold_loss_sums, dtype=np.float64)
    sizes = np.asarray(fold_assessment_sizes, dtype=np.int64).reshape(-1)
    if sums.ndim != 2 or sums.shape[0] < 1 or sums.shape[1] != len(sizes):
        raise ValueError("fold_loss_sums and fold_assessment_sizes are incompatible")
    if np.any(~np.isfinite(sums)) or np.any(sizes < 1):
        raise ValueError("fold loss sums must be finite and fold sizes positive")
    losses = sums.sum(axis=1) / int(sizes.sum())
    return losses, int(np.argmin(losses))


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result
