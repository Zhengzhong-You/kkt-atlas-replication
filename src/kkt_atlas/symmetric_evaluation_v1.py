"""Common residual-audited quality evaluation for the symmetric campaign."""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kkt_atlas.case_studies.network_reservation import NetworkReservationProblem
from kkt_atlas.confirmation_data_v1 import ConditionalTestData, FittingCatalogData
from kkt_atlas.evaluation import exact_expected_loss
from kkt_atlas.problem import Array, EqualityConstrainedReferenceProblem
from kkt_atlas.reference_evaluation import (
    ReferenceNewtonConfig,
    ReferenceSolveResult,
    solve_reference_newton,
)
from kkt_atlas.symmetric_laws_v1 import FiniteLaw

_REFERENCE_CONFIG = ReferenceNewtonConfig(
    maximum_iterations=100,
    stationarity_tolerance=1e-6,
    constraint_tolerance=1e-6,
    bound_feasibility_tolerance=1e-6,
    dual_feasibility_tolerance=1e-6,
    complementarity_tolerance=1e-6,
)
_NETWORK_TOLERANCE = 1e-6
_NETWORK_MAXIMUM_ITERATIONS = 5_000


@dataclass(frozen=True)
class SolverResidualSummary:
    maximum_stationarity: float
    maximum_equality: float
    maximum_lower_bound_violation: float
    maximum_dual_violation: float
    maximum_complementarity: float
    maximum_reservation: float
    maximum_recourse: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("solver residual summaries must be finite and nonnegative")


@dataclass(frozen=True)
class TrueReferenceEvaluation:
    eta: float
    problem_name: str
    catalog_hash: str
    evaluation_hash: str
    decisions: Array
    raw_context_risks: Array
    regularized_context_objectives: Array
    fixed_center_regularized_objectives: Array
    solver_residuals: SolverResidualSummary

    def __post_init__(self) -> None:
        decisions = np.asarray(self.decisions, dtype=np.float64)
        rows = len(decisions)
        if decisions.ndim != 2 or rows == 0 or np.any(~np.isfinite(decisions)):
            raise ValueError("reference decisions must be a nonempty finite matrix")
        for name in (
            "raw_context_risks",
            "regularized_context_objectives",
            "fixed_center_regularized_objectives",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.shape != (rows,) or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must align with reference decisions")
            object.__setattr__(self, name, _immutable(values))
        object.__setattr__(self, "decisions", _immutable(decisions))


@dataclass(frozen=True)
class FormalEvaluationResult:
    decisions: Array
    raw_context_risks: Array
    raw_operational_risk: float
    regularized_excess: float | None
    regularized_excess_numerator: float | None
    regularized_excess_denominator: float | None
    decision_rmse: float | None
    worst_feasibility: float
    solver_residuals: SolverResidualSummary

    def __post_init__(self) -> None:
        decisions = np.asarray(self.decisions, dtype=np.float64)
        risks = np.asarray(self.raw_context_risks, dtype=np.float64).reshape(-1)
        if (
            decisions.ndim != 2
            or risks.shape != (len(decisions),)
            or np.any(~np.isfinite(decisions))
            or np.any(~np.isfinite(risks))
        ):
            raise ValueError("formal decisions and risks must be finite and align")
        if not math.isfinite(self.raw_operational_risk):
            raise ValueError("raw_operational_risk must be finite")
        if not math.isfinite(self.worst_feasibility) or self.worst_feasibility < 0.0:
            raise ValueError("feasibility metric must be nonnegative")
        if self.decision_rmse is not None and (
            not math.isfinite(self.decision_rmse) or self.decision_rmse < 0.0
        ):
            raise ValueError("decision_rmse must be nonnegative or None")
        if self.regularized_excess is not None and not math.isfinite(
            self.regularized_excess
        ):
            raise ValueError("regularized_excess must be finite or None")
        for name in (
            "regularized_excess_numerator",
            "regularized_excess_denominator",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
        object.__setattr__(self, "decisions", _immutable(decisions))
        object.__setattr__(self, "raw_context_risks", _immutable(risks))


def aggregate_duplicate_law(law: FiniteLaw) -> FiniteLaw:
    """Combine exactly equal scenario rows while preserving first occurrence."""

    rows: list[np.ndarray] = []
    masses: list[float] = []
    indices: list[int] = []
    positions: dict[tuple[float, ...], int] = {}
    for row, (scenario, probability) in enumerate(
        zip(law.scenarios, law.probabilities, strict=True)
    ):
        if probability == 0.0:
            continue
        key = tuple(float(value) for value in scenario)
        position = positions.get(key)
        if position is None:
            positions[key] = len(rows)
            rows.append(scenario)
            masses.append(float(probability))
            indices.append(int(law.support_indices[row]))
        else:
            masses[position] += float(probability)
    if len(rows) == len(law.scenarios):
        return law
    return FiniteLaw(
        scenarios=np.vstack(rows),
        probabilities=np.asarray(masses),
        support_indices=np.asarray(indices),
        support_kind=law.support_kind,
    )


def compute_true_reference(
    problem: object,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
    *,
    eta: float,
    context_workers: int = 1,
) -> TrueReferenceEvaluation:
    """Solve the true finite conditional problem once per evaluation context."""

    eta_value = _nonnegative_eta(eta)
    _validate_bundles(problem, fitting, evaluation)
    support_indices = np.arange(fitting.support_size, dtype=np.int64)
    laws = tuple(
        FiniteLaw(
            fitting.catalog,
            probabilities,
            support_indices,
            "catalog",
        )
        for probabilities in evaluation.probabilities
    )
    decisions, residuals = _solve_laws_parallel(
        problem,
        evaluation.contexts,
        laws,
        eta=eta_value,
        context_workers=context_workers,
    )
    raw_risks, evaluation_recourse = _true_context_risks(
        problem,
        fitting,
        evaluation,
        decisions,
        context_workers=context_workers,
    )
    centers = _centers(problem, evaluation.contexts)
    center_risks, center_recourse = _true_context_risks(
        problem,
        fitting,
        evaluation,
        centers,
        context_workers=context_workers,
    )
    regularized = raw_risks + _regularization_penalties(
        decisions,
        centers,
        eta_value,
    )
    center_regularized = center_risks
    return TrueReferenceEvaluation(
        eta=eta_value,
        problem_name=str(problem.name),
        catalog_hash=fitting.catalog_hash,
        evaluation_hash=evaluation.data_hash,
        decisions=decisions,
        raw_context_risks=raw_risks,
        regularized_context_objectives=regularized,
        fixed_center_regularized_objectives=center_regularized,
        solver_residuals=_merge_residuals(
            residuals,
            maximum_recourse=max(evaluation_recourse, center_recourse),
        ),
    )


def evaluate_predicted_laws(
    problem: object,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
    laws: tuple[FiniteLaw, ...],
    *,
    eta: float,
    reference: TrueReferenceEvaluation | None,
    draw_indices: Array | None = None,
    context_workers: int = 1,
) -> FormalEvaluationResult:
    """Evaluate one fitted method under the true conditional distribution."""

    eta_value = _nonnegative_eta(eta)
    _validate_bundles(problem, fitting, evaluation)
    if reference is not None:
        _validate_reference(problem, fitting, evaluation, eta_value, reference)
    if len(laws) != len(evaluation.contexts):
        raise ValueError("one predicted law is required per evaluation context")
    aggregated = tuple(aggregate_duplicate_law(law) for law in laws)
    if any(law.scenarios.shape[1] != problem.scenario_dim for law in aggregated):
        raise ValueError("predicted law scenario dimensions do not match the problem")
    decisions, residuals = _solve_laws_parallel(
        problem,
        evaluation.contexts,
        aggregated,
        eta=eta_value,
        context_workers=context_workers,
    )
    if draw_indices is None:
        raw_risks, evaluation_recourse = _true_context_risks(
            problem,
            fitting,
            evaluation,
            decisions,
            context_workers=context_workers,
        )
    else:
        raw_risks, evaluation_recourse = sampled_context_risks(
            problem,
            fitting,
            evaluation,
            decisions,
            draw_indices,
            context_workers=context_workers,
        )
    regularized_excess = None
    numerator = None
    denominator = None
    decision_rmse = None
    if reference is not None:
        exact_risks, exact_recourse = _true_context_risks(
            problem,
            fitting,
            evaluation,
            decisions,
            context_workers=context_workers,
        )
        centers = _centers(problem, evaluation.contexts)
        regularized = exact_risks + _regularization_penalties(
            decisions,
            centers,
            eta_value,
        )
        numerator = math.fsum(
            float(value)
            for value in regularized - reference.regularized_context_objectives
        )
        denominator = math.fsum(
            float(value)
            for value in (
                reference.fixed_center_regularized_objectives
                - reference.regularized_context_objectives
            )
        )
        regularized_excess = None if denominator == 0.0 else numerator / denominator
        decision_rmse = float(np.sqrt(np.mean((decisions - reference.decisions) ** 2)))
        evaluation_recourse = max(evaluation_recourse, exact_recourse)
    worst_feasibility = max(
        _distance_to_feasible_set(problem, context, decision)
        for context, decision in zip(
            evaluation.contexts,
            decisions,
            strict=True,
        )
    )
    return FormalEvaluationResult(
        decisions=decisions,
        raw_context_risks=raw_risks,
        raw_operational_risk=float(np.mean(raw_risks)),
        regularized_excess=regularized_excess,
        regularized_excess_numerator=numerator,
        regularized_excess_denominator=denominator,
        decision_rmse=decision_rmse,
        worst_feasibility=float(worst_feasibility),
        solver_residuals=_merge_residuals(
            residuals,
            maximum_recourse=evaluation_recourse,
        ),
    )


def training_center_operational_losses(
    problem: object,
    fitting: FittingCatalogData,
) -> Array:
    """Return exact operational losses at the fixed center on training observations."""

    contexts = fitting.training.contexts
    outcomes = fitting.training.outcomes(fitting.catalog)
    centers = _centers(problem, contexts)
    if isinstance(problem, NetworkReservationProblem):
        if not np.all(centers == centers[0]):
            raise ValueError("NET training centers must be context independent")
        losses = problem.formal_scenario_losses_numpy(
            centers[0],
            outcomes,
            tolerance=_NETWORK_TOLERANCE,
            maximum_iterations=_NETWORK_MAXIMUM_ITERATIONS,
        )[0]
    else:
        with torch.no_grad():
            losses = (
                problem.realized_loss_torch(
                    torch.tensor(centers, dtype=torch.float64),
                    torch.tensor(outcomes, dtype=torch.float64),
                    torch.tensor(contexts, dtype=torch.float64),
                )
                .cpu()
                .numpy()
            )
    if losses.shape != (len(contexts),) or np.any(~np.isfinite(losses)):
        raise FloatingPointError("training center operational losses are invalid")
    return _immutable(losses)


def solve_predicted_laws(
    problem: object,
    contexts: Array,
    laws: Sequence[Any],
    *,
    eta: float,
    context_workers: int = 1,
) -> tuple[Array, SolverResidualSummary]:
    """Apply the common formal solver without aggregating supplied law terms."""

    values = np.ascontiguousarray(np.asarray(contexts, dtype=np.float64))
    if values.ndim != 2 or len(values) == 0 or len(values) != len(laws):
        raise ValueError("contexts and finite laws must be nonempty and align")
    for law in laws:
        scenarios = np.asarray(law.scenarios, dtype=np.float64)
        probabilities = np.asarray(law.probabilities, dtype=np.float64).reshape(-1)
        if (
            scenarios.ndim != 2
            or scenarios.shape != (len(probabilities), problem.scenario_dim)
            or np.any(~np.isfinite(scenarios))
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("formal finite law is invalid")
    return _solve_laws_parallel(
        problem,
        values,
        tuple(laws),
        eta=_nonnegative_eta(eta),
        context_workers=context_workers,
    )


def _solve_laws_parallel(
    problem: object,
    contexts: Array,
    laws: tuple[Any, ...],
    *,
    eta: float,
    context_workers: int,
) -> tuple[Array, SolverResidualSummary]:
    workers = int(context_workers)
    if workers < 1:
        raise ValueError("context_workers must be positive")
    if workers == 1 or len(contexts) == 1:
        return _solve_laws(problem, contexts, laws, eta=eta)
    with ThreadPoolExecutor(max_workers=min(workers, len(contexts))) as executor:
        results = tuple(
            executor.map(
                lambda item: _solve_laws(
                    problem,
                    np.ascontiguousarray(item[0][None, :]),
                    (item[1],),
                    eta=eta,
                ),
                zip(contexts, laws, strict=True),
            )
        )
    decisions = np.vstack([result[0][0] for result in results])
    residuals = _maximum_residuals(tuple(result[1] for result in results))
    return decisions, residuals


def _solve_laws(
    problem: object,
    contexts: Array,
    laws: tuple[FiniteLaw, ...],
    *,
    eta: float,
) -> tuple[Array, SolverResidualSummary]:
    decisions = np.empty((len(contexts), problem.decision_dim), dtype=np.float64)
    reference_results: list[ReferenceSolveResult] = []
    maximum_reservation = 0.0
    maximum_recourse = 0.0
    for index, (context, law) in enumerate(zip(contexts, laws, strict=True)):
        if isinstance(problem, NetworkReservationProblem):
            result = problem.solve_formal_numpy(
                context,
                law.scenarios,
                law.probabilities,
                ridge=eta,
                tolerance=_NETWORK_TOLERANCE,
                maximum_iterations=_NETWORK_MAXIMUM_ITERATIONS,
            )
            decisions[index] = result.decision
            maximum_reservation = max(
                maximum_reservation,
                result.reservation_residual,
            )
            maximum_recourse = max(
                maximum_recourse,
                result.maximum_recourse_residual,
            )
        elif isinstance(problem, EqualityConstrainedReferenceProblem):
            result = solve_reference_newton(
                problem,
                context,
                law.scenarios,
                law.probabilities,
                ridge=eta,
                config=_REFERENCE_CONFIG,
            )
            decisions[index] = result.decision
            reference_results.append(result)
        elif getattr(problem, "name", None) == "controlled_projected_quadratic":
            decisions[index] = problem.solve_numpy(
                context,
                law.scenarios,
                law.probabilities,
                ridge=eta,
            )
        else:
            raise TypeError("problem has no symmetric-campaign formal solver")
    return decisions, _summarize_solver_results(
        reference_results,
        maximum_reservation=maximum_reservation,
        maximum_recourse=maximum_recourse,
    )


def _true_context_risks(
    problem: object,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
    decisions: Array,
    *,
    context_workers: int = 1,
) -> tuple[Array, float]:
    if isinstance(problem, NetworkReservationProblem):
        risks = np.empty(len(decisions), dtype=np.float64)
        maximum_recourse = 0.0
        def evaluate_one(item: tuple[Array, Array]) -> tuple[float, float]:
            decision, probabilities = item
            losses, recourse = problem.formal_scenario_losses_numpy(
                decision,
                fitting.catalog,
                tolerance=_NETWORK_TOLERANCE,
                maximum_iterations=_NETWORK_MAXIMUM_ITERATIONS,
            )
            return (
                float(probabilities @ losses),
                float(np.max(recourse.residuals, initial=0.0)),
            )
        items = tuple(zip(decisions, evaluation.probabilities, strict=True))
        if context_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(
                max_workers=min(int(context_workers), len(items))
            ) as executor:
                evaluated = tuple(executor.map(evaluate_one, items))
        else:
            evaluated = tuple(evaluate_one(item) for item in items)
        risks[:] = [item[0] for item in evaluated]
        maximum_recourse = max((item[1] for item in evaluated), default=0.0)
        return risks, maximum_recourse
    evaluated = exact_expected_loss(
        problem,
        decisions,
        evaluation.contexts,
        fitting.catalog,
        evaluation.probabilities,
    )
    return np.ascontiguousarray(evaluated.context_losses), 0.0


def sampled_context_risks(
    problem: object,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
    decisions: Array,
    draw_indices: Array,
    *,
    context_workers: int = 1,
) -> tuple[Array, float]:
    """Evaluate common conditional draws without enumerating the full catalog."""

    indices = np.asarray(draw_indices, dtype=np.int64)
    if (
        indices.ndim != 2
        or indices.shape[0] != len(evaluation.contexts)
        or indices.shape[1] < 1
        or np.any(indices < 0)
        or np.any(indices >= fitting.support_size)
    ):
        raise ValueError("evaluation draw indices are invalid")
    if isinstance(problem, NetworkReservationProblem):
        def evaluate_one(item: tuple[Array, Array]) -> tuple[float, float]:
            decision, selected = item
            losses, recourse = problem.formal_scenario_losses_numpy(
                decision,
                fitting.catalog[selected],
                tolerance=_NETWORK_TOLERANCE,
                maximum_iterations=_NETWORK_MAXIMUM_ITERATIONS,
            )
            return (
                float(np.mean(losses)),
                float(np.max(recourse.residuals, initial=0.0)),
            )

        items = tuple(zip(decisions, indices, strict=True))
        if context_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(
                max_workers=min(int(context_workers), len(items))
            ) as executor:
                evaluated = tuple(executor.map(evaluate_one, items))
        else:
            evaluated = tuple(evaluate_one(item) for item in items)
        return (
            np.asarray([item[0] for item in evaluated], dtype=np.float64),
            max((item[1] for item in evaluated), default=0.0),
        )
    rows, draws = indices.shape
    repeated_decisions = np.repeat(decisions, draws, axis=0)
    repeated_contexts = np.repeat(evaluation.contexts, draws, axis=0)
    outcomes = fitting.catalog[indices.reshape(-1)]
    with torch.no_grad():
        losses = (
            problem.realized_loss_torch(
                torch.tensor(repeated_decisions, dtype=torch.float64),
                torch.tensor(outcomes, dtype=torch.float64),
                torch.tensor(repeated_contexts, dtype=torch.float64),
            )
            .cpu()
            .numpy()
            .reshape(rows, draws)
        )
    return np.ascontiguousarray(losses.mean(axis=1)), 0.0


def _summarize_solver_results(
    results: list[ReferenceSolveResult],
    *,
    maximum_reservation: float,
    maximum_recourse: float,
) -> SolverResidualSummary:
    return SolverResidualSummary(
        maximum_stationarity=max(
            (result.stationarity_residual_linf for result in results),
            default=0.0,
        ),
        maximum_equality=max(
            (result.constraint_residual_linf for result in results),
            default=0.0,
        ),
        maximum_lower_bound_violation=max(
            (result.lower_bound_violation_linf for result in results),
            default=0.0,
        ),
        maximum_dual_violation=max(
            (result.lower_bound_dual_residual_linf for result in results),
            default=0.0,
        ),
        maximum_complementarity=max(
            (result.complementarity_residual_linf for result in results),
            default=0.0,
        ),
        maximum_reservation=maximum_reservation,
        maximum_recourse=maximum_recourse,
    )


def _merge_residuals(
    residuals: SolverResidualSummary,
    *,
    maximum_recourse: float,
) -> SolverResidualSummary:
    return SolverResidualSummary(
        maximum_stationarity=residuals.maximum_stationarity,
        maximum_equality=residuals.maximum_equality,
        maximum_lower_bound_violation=(
            residuals.maximum_lower_bound_violation
        ),
        maximum_dual_violation=residuals.maximum_dual_violation,
        maximum_complementarity=residuals.maximum_complementarity,
        maximum_reservation=residuals.maximum_reservation,
        maximum_recourse=max(residuals.maximum_recourse, maximum_recourse),
    )


def _maximum_residuals(
    residuals: tuple[SolverResidualSummary, ...],
) -> SolverResidualSummary:
    if not residuals:
        raise ValueError("at least one residual summary is required")
    return SolverResidualSummary(
        **{
            name: max(getattr(item, name) for item in residuals)
            for name in SolverResidualSummary.__dataclass_fields__
        }
    )


def _validate_bundles(
    problem: object,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
) -> None:
    if fitting.catalog_hash != evaluation.catalog_hash:
        raise ValueError("fitting and evaluation catalogs differ")
    if fitting.metadata != evaluation.metadata:
        raise ValueError("fitting and evaluation metadata differ")
    if fitting.context_dim != problem.context_dim:
        raise ValueError("bundle and problem context dimensions differ")
    if fitting.scenario_dim != problem.scenario_dim:
        raise ValueError("bundle and problem scenario dimensions differ")
    if evaluation.support_size != fitting.support_size:
        raise ValueError("evaluation probabilities do not match the catalog")


def _validate_reference(
    problem: object,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
    eta: float,
    reference: TrueReferenceEvaluation,
) -> None:
    expected = (
        eta,
        str(problem.name),
        fitting.catalog_hash,
        evaluation.data_hash,
        len(evaluation.contexts),
        problem.decision_dim,
    )
    observed = (
        reference.eta,
        reference.problem_name,
        reference.catalog_hash,
        reference.evaluation_hash,
        len(reference.decisions),
        reference.decisions.shape[1],
    )
    if observed != expected:
        raise ValueError("true reference does not match this evaluation task")


def _distance_to_feasible_set(
    problem: object,
    context: Array,
    decision: Array,
) -> float:
    value = np.asarray(decision, dtype=np.float64).reshape(-1)
    if getattr(problem, "name", None) == "controlled_projected_quadratic":
        return 0.0
    if isinstance(problem, NetworkReservationProblem):
        return float(
            np.linalg.norm(
                value
                - np.clip(value, 0.0, problem.reservation_upper_bound)
            )
        )
    equality, rhs = problem.reference_equality_numpy(context)
    equality = np.asarray(equality, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lower_method = getattr(problem, "reference_lower_bounds_numpy", None)
    if callable(lower_method):
        lower = np.asarray(lower_method(context), dtype=np.float64).reshape(-1)
        if (
            equality.shape[0] == 1
            and np.all(equality[0] == 1.0)
            and np.all(lower == 0.0)
        ):
            projected = _project_simplex(value, float(rhs[0]))
            return float(np.linalg.norm(value - projected))
    residual = equality @ value - rhs
    correction = equality.T @ np.linalg.solve(
        equality @ equality.T,
        residual,
    )
    return float(np.linalg.norm(correction))


def _project_simplex(values: Array, total: float) -> Array:
    order = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    cumulative = np.cumsum(order) - total
    indices = np.arange(1, len(order) + 1)
    positive = np.flatnonzero(order - cumulative / indices > 0.0)
    if len(positive) == 0:
        raise ValueError("simplex projection has no positive coordinate")
    threshold = cumulative[int(positive[-1])] / float(positive[-1] + 1)
    return np.maximum(np.asarray(values, dtype=np.float64) - threshold, 0.0)


def _regularization_penalties(
    decisions: Array,
    centers: Array,
    eta: float,
) -> Array:
    return 0.5 * eta * np.sum((decisions - centers) ** 2, axis=1)


def _centers(problem: object, contexts: Array) -> Array:
    center_method = getattr(problem, "reference_center_numpy", None)
    if callable(center_method):
        return np.vstack([center_method(context) for context in contexts])
    if getattr(problem, "name", None) == "controlled_projected_quadratic":
        return np.zeros((len(contexts), problem.decision_dim), dtype=np.float64)
    raise TypeError("problem has no regularization center")


def _nonnegative_eta(value: float) -> float:
    eta = float(value)
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be nonnegative and finite")
    return eta


def _immutable(values: Array) -> Array:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).copy())
    array.setflags(write=False)
    return array
