"""Training-only support-size selection from cross-fitted KKT responses.

The selector conditions on a chart partition, a local SAA decision, and a
fixed scenario basis.  Candidate-specific scenario locations and probability
heads are fitted without downstream optimization.  A paired one-standard-
error rule then returns the smallest support whose paired out-of-fold loss is
within one estimated standard error of the best candidate.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from kkt_atlas.automatic_structure import identifiable_local_parameter_count
from kkt_atlas.initialization import cluster_projected_scenarios
from kkt_atlas.probability_initialization import (
    ProbabilityHeadInitialization,
    fit_cluster_probability_head,
)
from kkt_atlas.problem import Array, DifferentiableStochasticProblem, matrix
from kkt_atlas.response_basis import ResponseDecomposition


def maximum_identifiable_support_size(
    fitting_sample_size: int,
    representation_dim: int,
    context_dim: int,
    *,
    minimum_samples_per_parameter: float,
    distinct_projected_observations: int,
    support_budget: int | None = None,
) -> int:
    """Return the largest support satisfying the frozen sample-size rule."""

    sample_size = _positive_integer(fitting_sample_size, name="fitting_sample_size")
    width = _positive_integer(representation_dim, name="representation_dim")
    dimension = _nonnegative_integer(context_dim, name="context_dim")
    distinct = _positive_integer(
        distinct_projected_observations,
        name="distinct_projected_observations",
    )
    ratio = _positive_finite_float(
        minimum_samples_per_parameter,
        name="minimum_samples_per_parameter",
    )
    closed_form_cap = math.floor((sample_size / ratio + dimension + 1) / (width + dimension + 1))
    limits = [closed_form_cap, distinct]
    if support_budget is not None:
        limits.append(_positive_integer(support_budget, name="support_budget"))
    result = min(limits)
    if result < 1:
        required = identifiable_local_parameter_count(1, width, dimension)
        raise ValueError(
            "even one scenario violates the minimum samples-per-parameter "
            f"rule: fitting_sample_size={sample_size}, required_parameters={required}"
        )
    return int(result)


def fixed_cross_fitting_folds(
    sample_size: int,
    fold_count: int,
    *,
    seed: int,
) -> tuple[Array, ...]:
    """Return deterministic, nonempty, nearly equal assessment folds."""

    size = _positive_integer(sample_size, name="sample_size")
    count = _positive_integer(fold_count, name="fold_count")
    if count < 2 or count > size:
        raise ValueError("fold_count must lie between 2 and sample_size")
    permutation = np.random.default_rng(int(seed)).permutation(size)
    folds: list[Array] = []
    for values in np.array_split(permutation, count):
        indices = np.ascontiguousarray(np.sort(values).astype(np.int64, copy=False))
        indices.setflags(write=False)
        folds.append(indices)
    return tuple(folds)


def fit_response_probability_head(
    centered_standardized_contexts: Array,
    target_responses: Array,
    atom_responses: Array,
    *,
    slope_penalty: float,
    initialization: ProbabilityHeadInitialization | None = None,
    maximum_iterations: int = 500,
) -> ProbabilityHeadInitialization:
    """Fit an affine-softmax head by squared KKT-response loss.

    The atom responses remain fixed.  Only the identifiable affine-softmax
    intercepts and slopes are optimized, so this routine never invokes a
    stochastic-program solution mapping.
    """

    contexts = np.asarray(centered_standardized_contexts, dtype=np.float64)
    targets = np.asarray(target_responses, dtype=np.float64)
    atoms = np.asarray(atom_responses, dtype=np.float64)
    if contexts.ndim != 2 or len(contexts) == 0 or np.any(~np.isfinite(contexts)):
        raise ValueError("centered_standardized_contexts must be a nonempty finite matrix")
    if targets.ndim != 2 or targets.shape[0] != len(contexts) or np.any(~np.isfinite(targets)):
        raise ValueError("target_responses must be a finite matrix aligned with contexts")
    if atoms.ndim != 2 or len(atoms) == 0 or atoms.shape[1] != targets.shape[1]:
        raise ValueError("atom_responses must be a nonempty matrix aligned with target responses")
    if np.any(~np.isfinite(atoms)):
        raise ValueError("atom_responses must be finite")
    penalty = _nonnegative_finite_float(slope_penalty, name="slope_penalty")
    iterations = _positive_integer(maximum_iterations, name="maximum_iterations")
    support_size = int(len(atoms))
    context_dim = int(contexts.shape[1])

    if initialization is None:
        intercepts = np.zeros(support_size, dtype=np.float64)
        slopes = np.zeros((support_size, context_dim), dtype=np.float64)
    else:
        intercepts = np.asarray(initialization.intercepts, dtype=np.float64).copy()
        slopes = np.asarray(initialization.slopes, dtype=np.float64).copy()
        if intercepts.shape != (support_size,) or slopes.shape != (support_size, context_dim):
            raise ValueError("initialization does not match the response-head dimensions")
    initial_parameters = np.column_stack((intercepts, slopes))
    initial_parameters -= initial_parameters.mean(axis=0, keepdims=True)
    augmented = np.column_stack((np.ones(len(contexts), dtype=np.float64), contexts))

    def objective_and_gradient(flat_parameters: Array) -> tuple[float, Array]:
        parameters = np.asarray(flat_parameters, dtype=np.float64).reshape(
            support_size,
            context_dim + 1,
        )
        centered = parameters - parameters.mean(axis=0, keepdims=True)
        logits = augmented @ centered.T
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(logits)
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        predictions = probabilities @ atoms
        residuals = predictions - targets
        value = float(np.mean(np.sum(residuals**2, axis=1)) + penalty * np.sum(centered[:, 1:] ** 2))
        probability_gradient = 2.0 * (residuals @ atoms.T) / len(contexts)
        logit_gradient = probabilities * (
            probability_gradient - np.sum(probability_gradient * probabilities, axis=1, keepdims=True)
        )
        gradient = logit_gradient.T @ augmented
        gradient[:, 1:] += 2.0 * penalty * centered[:, 1:]
        gradient -= gradient.mean(axis=0, keepdims=True)
        return value, np.asarray(gradient, dtype=np.float64).reshape(-1)

    initial_objective, _ = objective_and_gradient(initial_parameters.reshape(-1))
    if support_size == 1:
        return ProbabilityHeadInitialization(
            intercepts=np.zeros(1, dtype=np.float64),
            slopes=np.zeros((1, context_dim), dtype=np.float64),
            method="training_kkt_response_least_squares",
            converged=True,
            iterations=0,
            initial_objective=initial_objective,
            final_objective=initial_objective,
        )

    result = minimize(
        objective_and_gradient,
        initial_parameters.reshape(-1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": iterations,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    fitted = np.asarray(result.x, dtype=np.float64).reshape(
        support_size,
        context_dim + 1,
    )
    fitted -= fitted.mean(axis=0, keepdims=True)
    final_objective, _ = objective_and_gradient(fitted.reshape(-1))
    fallback_reason: str | None = None
    if (
        not bool(result.success)
        or np.any(~np.isfinite(fitted))
        or not math.isfinite(final_objective)
        or final_objective > initial_objective + 1e-10
    ):
        fallback_reason = f"L-BFGS status {int(result.status)}: {str(result.message)}"
        fitted = initial_parameters
        final_objective = initial_objective
    return ProbabilityHeadInitialization(
        intercepts=fitted[:, 0],
        slopes=fitted[:, 1:],
        method="training_kkt_response_least_squares",
        converged=fallback_reason is None,
        iterations=int(result.nit),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        fallback_reason=fallback_reason,
    )


def predict_responses(
    centered_standardized_contexts: Array,
    atom_responses: Array,
    head: ProbabilityHeadInitialization,
) -> Array:
    """Return the KKT-response predictions of a fitted probability head."""

    contexts = np.asarray(centered_standardized_contexts, dtype=np.float64)
    atoms = np.asarray(atom_responses, dtype=np.float64)
    if contexts.ndim != 2 or np.any(~np.isfinite(contexts)):
        raise ValueError("centered_standardized_contexts must be a finite matrix")
    if atoms.ndim != 2 or len(atoms) != len(head.intercepts) or np.any(~np.isfinite(atoms)):
        raise ValueError("atom_responses do not match the fitted probability head")
    if contexts.shape[1] != head.slopes.shape[1]:
        raise ValueError("context dimension does not match the fitted probability head")
    logits = head.intercepts[None, :] + contexts @ head.slopes.T
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    return probabilities @ atoms


@dataclass(frozen=True)
class SupportFoldDiagnostic:
    """One out-of-fold response score for a support candidate."""

    fold: int
    support_size: int
    fitting_sample_size: int
    assessment_sample_size: int
    response_loss: float
    head_converged: bool
    head_iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "support_size": self.support_size,
            "fitting_sample_size": self.fitting_sample_size,
            "assessment_sample_size": self.assessment_sample_size,
            "response_loss": self.response_loss,
            "head_converged": self.head_converged,
            "head_iterations": self.head_iterations,
        }


@dataclass(frozen=True)
class SupportCandidateDiagnostic:
    """Paired cross-fitting summary for one candidate support size."""

    support_size: int
    mean_response_loss: float
    paired_mean_difference: float
    paired_standard_error: float
    within_one_standard_error: bool
    folds: tuple[SupportFoldDiagnostic, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_size": self.support_size,
            "mean_response_loss": self.mean_response_loss,
            "paired_mean_difference": self.paired_mean_difference,
            "paired_standard_error": self.paired_standard_error,
            "within_one_standard_error": self.within_one_standard_error,
            "folds": [item.as_dict() for item in self.folds],
        }


@dataclass(frozen=True)
class HeadAwareSupportSelection:
    """Training-only support selected by paired cross-fitted response loss."""

    selected_support_size: int
    reference_support_size: int
    maximum_support_size: int
    structural_reference_support_size: int
    structural_reference_is_admissible: bool
    best_cross_fitted_response_loss: float
    fold_count: int
    minimum_fitting_sample_size: int
    minimum_distinct_projected_observations: int
    candidates: tuple[SupportCandidateDiagnostic, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rule": "smallest_K_within_paired_one_standard_error",
            "selected_support_size": self.selected_support_size,
            "reference_support_size": self.reference_support_size,
            "maximum_support_size": self.maximum_support_size,
            "structural_reference_support_size": (self.structural_reference_support_size),
            "structural_reference_is_admissible": self.structural_reference_is_admissible,
            "best_cross_fitted_response_loss": self.best_cross_fitted_response_loss,
            "fold_count": self.fold_count,
            "minimum_fitting_sample_size": self.minimum_fitting_sample_size,
            "minimum_distinct_projected_observations": (self.minimum_distinct_projected_observations),
            "candidates": [item.as_dict() for item in self.candidates],
        }


def select_head_aware_support_size(
    problem: DifferentiableStochasticProblem,
    decomposition: ResponseDecomposition,
    contexts: Array,
    outcomes: Array,
    *,
    width: int,
    context_mean: Array,
    context_scale: Array,
    standardized_center: Array,
    slope_penalty: float,
    minimum_samples_per_parameter: float,
    fold_count: int = 5,
    one_standard_error_multiplier: float = 1.0,
    support_budget: int | None = None,
    maximum_head_iterations: int = 500,
    seed: int = 0,
) -> HeadAwareSupportSelection:
    """Select a chart-specific support after its partition and width are fixed.

    The routine uses training observations only.  A nonconverged response-head
    fit stops selection rather than allowing a failed candidate to benefit from
    the paired comparison.
    """

    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    outcome_values = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if len(context_values) != len(outcome_values):
        raise ValueError("contexts and outcomes must contain the same observations")
    width_value = _positive_integer(width, name="width")
    if width_value > decomposition.maximum_width:
        raise ValueError("width exceeds the response decomposition")
    mean = np.asarray(context_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(context_scale, dtype=np.float64).reshape(-1)
    center = np.asarray(standardized_center, dtype=np.float64).reshape(-1)
    if mean.shape != (problem.context_dim,) or scale.shape != (problem.context_dim,):
        raise ValueError("context normalization has the wrong dimension")
    if center.shape != (problem.context_dim,) or np.any(scale <= 0.0):
        raise ValueError("standardized_center and context_scale are invalid")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(~np.isfinite(center)):
        raise ValueError("context normalization must be finite")
    multiplier = _nonnegative_finite_float(
        one_standard_error_multiplier,
        name="one_standard_error_multiplier",
    )
    folds = fixed_cross_fitting_folds(
        len(context_values),
        fold_count,
        seed=_derived_seed(seed, "support_folds"),
    )
    projected = decomposition.coordinates(outcome_values, width_value, problem)
    all_indices = np.arange(len(context_values), dtype=np.int64)
    fitting_indices = [np.setdiff1d(all_indices, fold, assume_unique=True) for fold in folds]
    minimum_fitting_size = min(len(indices) for indices in fitting_indices)
    minimum_distinct = min(len(np.unique(projected[indices], axis=0)) for indices in fitting_indices)
    support_cap = maximum_identifiable_support_size(
        minimum_fitting_size,
        width_value,
        problem.context_dim,
        minimum_samples_per_parameter=minimum_samples_per_parameter,
        distinct_projected_observations=minimum_distinct,
        support_budget=support_budget,
    )
    centered_contexts = (context_values - mean[None, :]) / scale[None, :] - center[None, :]
    target_responses = matrix(
        problem.kkt_contributions_numpy(
            decomposition.local_decision,
            decomposition.context_center,
            outcome_values,
        ),
        columns=problem.decision_dim,
        name="target KKT responses",
    )
    fold_diagnostics: dict[int, list[SupportFoldDiagnostic]] = {
        support_size: [] for support_size in range(1, support_cap + 1)
    }
    fold_losses = np.empty((support_cap, len(folds)), dtype=np.float64)

    for fold_number, (assessment, fitting) in enumerate(zip(folds, fitting_indices, strict=True)):
        for support_size in range(1, support_cap + 1):
            clustering = cluster_projected_scenarios(
                projected[fitting],
                support_size,
                seed=_derived_seed(seed, "support_kmeans", fold_number, support_size),
            )
            atom_scenarios = _decode_coordinates(
                problem,
                decomposition,
                clustering.initialization.coordinates,
                width=width_value,
            )
            atom_responses = matrix(
                problem.kkt_contributions_numpy(
                    decomposition.local_decision,
                    decomposition.context_center,
                    atom_scenarios,
                ),
                columns=problem.decision_dim,
                name="atom KKT responses",
            )
            cluster_head = fit_cluster_probability_head(
                centered_contexts[fitting],
                clustering.assignments,
                support_size,
                slope_penalty=slope_penalty,
                maximum_iterations=maximum_head_iterations,
            )
            response_head = fit_response_probability_head(
                centered_contexts[fitting],
                target_responses[fitting],
                atom_responses,
                slope_penalty=slope_penalty,
                initialization=cluster_head,
                maximum_iterations=maximum_head_iterations,
            )
            if not response_head.converged:
                raise RuntimeError(
                    "response-head fitting failed closed for "
                    f"fold={fold_number}, K={support_size}: {response_head.fallback_reason}"
                )
            predictions = predict_responses(
                centered_contexts[assessment],
                atom_responses,
                response_head,
            )
            residuals = predictions - target_responses[assessment]
            loss = float(np.mean(np.sum(residuals**2, axis=1)))
            if not math.isfinite(loss):
                raise FloatingPointError("cross-fitted KKT-response loss is nonfinite")
            fold_losses[support_size - 1, fold_number] = loss
            fold_diagnostics[support_size].append(
                SupportFoldDiagnostic(
                    fold=fold_number,
                    support_size=support_size,
                    fitting_sample_size=len(fitting),
                    assessment_sample_size=len(assessment),
                    response_loss=loss,
                    head_converged=response_head.converged,
                    head_iterations=response_head.iterations,
                )
            )

    mean_losses = fold_losses.mean(axis=1)
    reference_index = int(np.argmin(mean_losses))
    reference_support = reference_index + 1
    diagnostics: list[SupportCandidateDiagnostic] = []
    selected_support: int | None = None
    for support_index in range(support_cap):
        differences = fold_losses[support_index] - fold_losses[reference_index]
        mean_difference = float(differences.mean())
        standard_error = float(np.sqrt(np.sum((differences - mean_difference) ** 2) / (len(folds) * (len(folds) - 1))))
        eligible = mean_difference <= multiplier * standard_error + 1e-12
        support_size = support_index + 1
        if selected_support is None and eligible:
            selected_support = support_size
        diagnostics.append(
            SupportCandidateDiagnostic(
                support_size=support_size,
                mean_response_loss=float(mean_losses[support_index]),
                paired_mean_difference=mean_difference,
                paired_standard_error=standard_error,
                within_one_standard_error=eligible,
                folds=tuple(fold_diagnostics[support_size]),
            )
        )
    if selected_support is None:
        raise RuntimeError("the paired one-standard-error rule selected no support")
    return HeadAwareSupportSelection(
        selected_support_size=selected_support,
        reference_support_size=reference_support,
        maximum_support_size=support_cap,
        structural_reference_support_size=width_value + 1,
        structural_reference_is_admissible=width_value + 1 <= support_cap,
        best_cross_fitted_response_loss=float(mean_losses[reference_index]),
        fold_count=len(folds),
        minimum_fitting_sample_size=minimum_fitting_size,
        minimum_distinct_projected_observations=minimum_distinct,
        candidates=tuple(diagnostics),
    )


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
    transformed = decomposition.transformed_mean[None, :] + standardized * decomposition.transformed_scale[None, :]
    with torch.no_grad():
        decoded = problem.decode_torch(torch.tensor(transformed, dtype=torch.float64))
    return matrix(
        decoded.detach().cpu().numpy(),
        columns=problem.scenario_dim,
        name="decoded scenarios",
    )


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
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
