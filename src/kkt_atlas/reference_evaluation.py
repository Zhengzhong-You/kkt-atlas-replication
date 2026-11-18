"""High-accuracy full-support references and dual-reference metrics.

This module is evaluation-only.  Training continues to use each problem's
differentiable solution map; the routines below provide independent,
residual-audited reference decisions for numerical comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kkt_atlas.evaluation import exact_expected_loss
from kkt_atlas.problem import (
    Array,
    EqualityConstrainedReferenceProblem,
    LowerBoundConstrainedReferenceProblem,
    matrix,
    probability_vector,
)

DUAL_REFERENCE_METRIC_SCHEMA = "kkt-atlas-dual-reference-metrics-v1"


class ReferenceSolveError(RuntimeError):
    """Raised when a reference solve cannot pass its numerical audit."""


class ReferenceMetricAuditError(RuntimeError):
    """Raised when a theoretically nonnegative metric violates tolerance."""


@dataclass(frozen=True)
class ReferenceNewtonConfig:
    """Numerical controls for equality-constrained Newton solves."""

    maximum_iterations: int = 100
    stationarity_tolerance: float = 1e-10
    constraint_tolerance: float = 1e-12
    armijo_constant: float = 1e-4
    backtracking_factor: float = 0.5
    minimum_step: float = 2.0**-40
    hessian_symmetry_tolerance: float = 1e-11
    bound_feasibility_tolerance: float = 1e-12
    dual_feasibility_tolerance: float = 1e-10
    complementarity_tolerance: float = 1e-10
    active_bound_tolerance: float = 1e-11

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        for name in (
            "stationarity_tolerance",
            "constraint_tolerance",
            "minimum_step",
            "hessian_symmetry_tolerance",
            "bound_feasibility_tolerance",
            "dual_feasibility_tolerance",
            "complementarity_tolerance",
            "active_bound_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.armijo_constant < 0.5:
            raise ValueError("armijo_constant must lie in (0,0.5)")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must lie in (0,1)")


@dataclass(frozen=True)
class ReferenceSolveResult:
    """One converged and audited reference solution."""

    decision: Array
    ridge: float
    raw_objective: float
    regularized_objective: float
    regularization_penalty: float
    multiplier: Array
    lower_bound_multipliers: Array
    lower_bounds_enforced: bool
    stationarity_residual_linf: float
    constraint_residual_linf: float
    lower_bound_violation_linf: float
    lower_bound_dual_residual_linf: float
    complementarity_residual_linf: float
    active_lower_bound_count: int
    active_set_changes: int
    iterations: int
    backtracking_steps: int
    numerical_descent_safeguard_steps: int

    def __post_init__(self) -> None:
        decision = _immutable_vector(self.decision, "decision")
        multiplier = _immutable_vector(self.multiplier, "multiplier")
        lower_bound_multipliers = _immutable_vector(
            self.lower_bound_multipliers,
            "lower_bound_multipliers",
        )
        if lower_bound_multipliers.shape != decision.shape:
            raise ValueError("lower-bound multipliers must align with the decision")
        if not isinstance(self.lower_bounds_enforced, bool):
            raise ValueError("lower_bounds_enforced must be boolean")
        scalars = (
            self.ridge,
            self.raw_objective,
            self.regularized_objective,
            self.regularization_penalty,
            self.stationarity_residual_linf,
            self.constraint_residual_linf,
            self.lower_bound_violation_linf,
            self.lower_bound_dual_residual_linf,
            self.complementarity_residual_linf,
        )
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("reference result scalars must be finite")
        if self.ridge < 0.0 or self.regularization_penalty < -1e-14:
            raise ValueError("ridge and regularization penalty must be nonnegative")
        if (
            self.stationarity_residual_linf < 0.0
            or self.constraint_residual_linf < 0.0
            or self.lower_bound_violation_linf < 0.0
            or self.lower_bound_dual_residual_linf < 0.0
            or self.complementarity_residual_linf < 0.0
        ):
            raise ValueError("residuals must be nonnegative")
        if (
            self.active_lower_bound_count < 0
            or self.active_set_changes < 0
            or self.iterations < 0
            or self.backtracking_steps < 0
            or self.numerical_descent_safeguard_steps < 0
        ):
            raise ValueError("iteration counters must be nonnegative")
        if not math.isclose(
            self.regularized_objective - self.raw_objective,
            self.regularization_penalty,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("objective values do not match the regularization penalty")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(
            self,
            "lower_bound_multipliers",
            lower_bound_multipliers,
        )

    def to_record(self, *, include_decision: bool = False) -> dict[str, object]:
        """Return a JSON-compatible audit record."""

        record: dict[str, object] = {
            "solver": (
                "lower_bound_active_set_newton" if self.lower_bounds_enforced else "equality_constrained_newton"
            ),
            "lower_bounds_enforced": self.lower_bounds_enforced,
            "converged": True,
            "ridge": self.ridge,
            "raw_objective": self.raw_objective,
            "regularized_objective": self.regularized_objective,
            "regularization_penalty": self.regularization_penalty,
            "stationarity_residual_linf": self.stationarity_residual_linf,
            "constraint_residual_linf": self.constraint_residual_linf,
            "lower_bound_violation_linf": self.lower_bound_violation_linf,
            "lower_bound_dual_residual_linf": (self.lower_bound_dual_residual_linf),
            "complementarity_residual_linf": (self.complementarity_residual_linf),
            "active_lower_bound_count": self.active_lower_bound_count,
            "active_set_changes": self.active_set_changes,
            "iterations": self.iterations,
            "backtracking_steps": self.backtracking_steps,
            "numerical_descent_safeguard_steps": (self.numerical_descent_safeguard_steps),
        }
        if include_decision:
            record["decision"] = self.decision.tolist()
            record["multiplier"] = self.multiplier.tolist()
            record["lower_bound_multipliers"] = self.lower_bound_multipliers.tolist()
        return record


@dataclass(frozen=True)
class ReferenceBatchResult:
    """Audited full-support reference results over several contexts."""

    ridge: float
    lower_bounds_enforced: bool
    decisions: Array
    raw_objectives: Array
    regularized_objectives: Array
    regularization_penalties: Array
    stationarity_residuals_linf: Array
    constraint_residuals_linf: Array
    lower_bound_violations_linf: Array
    lower_bound_dual_residuals_linf: Array
    complementarity_residuals_linf: Array
    active_lower_bound_counts: Array
    active_set_changes: Array
    iterations: Array
    backtracking_steps: Array
    numerical_descent_safeguard_steps: Array

    def __post_init__(self) -> None:
        decisions = np.asarray(self.decisions, dtype=np.float64)
        if decisions.ndim != 2 or len(decisions) == 0 or np.any(~np.isfinite(decisions)):
            raise ValueError("decisions must be a nonempty finite matrix")
        rows = len(decisions)
        immutable_decisions = np.ascontiguousarray(decisions.copy())
        immutable_decisions.setflags(write=False)
        object.__setattr__(self, "decisions", immutable_decisions)
        for field_name in (
            "raw_objectives",
            "regularized_objectives",
            "regularization_penalties",
            "stationarity_residuals_linf",
            "constraint_residuals_linf",
            "lower_bound_violations_linf",
            "lower_bound_dual_residuals_linf",
            "complementarity_residuals_linf",
            "active_lower_bound_counts",
            "active_set_changes",
            "iterations",
            "backtracking_steps",
            "numerical_descent_safeguard_steps",
        ):
            values = np.asarray(getattr(self, field_name)).reshape(-1)
            if values.shape != (rows,) or np.any(~np.isfinite(values)):
                raise ValueError(f"{field_name} must contain one finite value per context")
            immutable = np.ascontiguousarray(values.copy())
            immutable.setflags(write=False)
            object.__setattr__(self, field_name, immutable)
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        if not isinstance(self.lower_bounds_enforced, bool):
            raise ValueError("lower_bounds_enforced must be boolean")
        if np.any(self.regularization_penalties < -1e-14):
            raise ValueError("regularization penalties must be nonnegative")
        if (
            np.any(self.stationarity_residuals_linf < 0.0)
            or np.any(self.constraint_residuals_linf < 0.0)
            or np.any(self.lower_bound_violations_linf < 0.0)
            or np.any(self.lower_bound_dual_residuals_linf < 0.0)
            or np.any(self.complementarity_residuals_linf < 0.0)
        ):
            raise ValueError("reference residuals must be nonnegative")
        if (
            np.any(self.active_lower_bound_counts < 0.0)
            or np.any(self.active_set_changes < 0.0)
            or np.any(self.iterations < 0.0)
            or np.any(self.backtracking_steps < 0.0)
            or np.any(self.numerical_descent_safeguard_steps < 0.0)
        ):
            raise ValueError("reference iteration counters must be nonnegative")
        if not np.allclose(
            self.regularized_objectives - self.raw_objectives,
            self.regularization_penalties,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("batch objective values do not match their penalties")

    def to_record(self) -> dict[str, object]:
        """Summarize the batch without embedding per-context arrays."""

        return {
            "solver": (
                "lower_bound_active_set_newton" if self.lower_bounds_enforced else "equality_constrained_newton"
            ),
            "lower_bounds_enforced": self.lower_bounds_enforced,
            "converged_contexts": len(self.decisions),
            "ridge": self.ridge,
            "mean_raw_objective": float(np.mean(self.raw_objectives)),
            "mean_regularized_objective": float(np.mean(self.regularized_objectives)),
            "mean_regularization_penalty": float(np.mean(self.regularization_penalties)),
            "maximum_stationarity_residual_linf": float(np.max(self.stationarity_residuals_linf)),
            "maximum_constraint_residual_linf": float(np.max(self.constraint_residuals_linf)),
            "maximum_lower_bound_violation_linf": float(np.max(self.lower_bound_violations_linf)),
            "maximum_lower_bound_dual_residual_linf": float(np.max(self.lower_bound_dual_residuals_linf)),
            "maximum_complementarity_residual_linf": float(np.max(self.complementarity_residuals_linf)),
            "maximum_active_lower_bound_count": int(np.max(self.active_lower_bound_counts)),
            "total_active_set_changes": int(np.sum(self.active_set_changes)),
            "mean_iterations": float(np.mean(self.iterations)),
            "maximum_iterations": int(np.max(self.iterations)),
            "total_backtracking_steps": int(np.sum(self.backtracking_steps)),
            "total_numerical_descent_safeguard_steps": int(np.sum(self.numerical_descent_safeguard_steps)),
        }


@dataclass(frozen=True)
class DualReferenceResult:
    """Unregularized and same-ridge full-support references."""

    eta_zero: ReferenceBatchResult
    same_eta: ReferenceBatchResult

    def __post_init__(self) -> None:
        if self.eta_zero.ridge != 0.0:
            raise ValueError("eta_zero must use ridge=0")
        if self.eta_zero.decisions.shape != self.same_eta.decisions.shape:
            raise ValueError("dual references must align")

    def to_record(self) -> dict[str, object]:
        return {
            "eta_zero": self.eta_zero.to_record(),
            "same_eta": self.same_eta.to_record(),
        }


@dataclass(frozen=True)
class DualReferenceAggregate:
    """Aggregate metrics; L2 RMS is ``sqrt(mean_i ||e_i||_2^2)``."""

    mean_raw_operational_excess_to_eta_zero: float
    mean_same_eta_regularized_gap: float
    mean_signed_raw_difference_to_same_eta: float
    mean_reference_regularization_bias: float
    maximum_absolute_bias_identity_residual: float
    decision_l2_rms_to_eta_zero: float
    decision_l2_rms_to_same_eta: float
    portfolio_weight_l2_rms_to_eta_zero: float | None
    portfolio_weight_l2_rms_to_same_eta: float | None
    portfolio_threshold_rmse_to_eta_zero: float | None
    portfolio_threshold_rmse_to_same_eta: float | None

    def to_record(self) -> dict[str, float | None | str]:
        return {
            "mean_raw_operational_excess_to_eta_zero": (self.mean_raw_operational_excess_to_eta_zero),
            "mean_same_eta_regularized_gap": self.mean_same_eta_regularized_gap,
            "mean_signed_raw_difference_to_same_eta": (self.mean_signed_raw_difference_to_same_eta),
            "mean_reference_regularization_bias": (self.mean_reference_regularization_bias),
            "maximum_absolute_bias_identity_residual": (self.maximum_absolute_bias_identity_residual),
            "decision_l2_rms_definition": "sqrt(mean_i(||candidate_i-reference_i||_2^2))",
            "decision_l2_rms_to_eta_zero": self.decision_l2_rms_to_eta_zero,
            "decision_l2_rms_to_same_eta": self.decision_l2_rms_to_same_eta,
            "portfolio_weight_l2_rms_to_eta_zero": (self.portfolio_weight_l2_rms_to_eta_zero),
            "portfolio_weight_l2_rms_to_same_eta": (self.portfolio_weight_l2_rms_to_same_eta),
            "portfolio_threshold_rmse_to_eta_zero": (self.portfolio_threshold_rmse_to_eta_zero),
            "portfolio_threshold_rmse_to_same_eta": (self.portfolio_threshold_rmse_to_same_eta),
        }


@dataclass(frozen=True)
class DualReferenceEvaluation:
    """Per-context and aggregate comparisons against both references."""

    candidate_raw_objectives: Array
    candidate_regularized_objectives: Array
    raw_operational_excess_to_eta_zero: Array
    same_eta_regularized_gap: Array
    signed_raw_difference_to_same_eta: Array
    reference_regularization_bias: Array
    bias_identity_residual: Array
    decision_l2_to_eta_zero: Array
    decision_l2_to_same_eta: Array
    portfolio_weight_l2_to_eta_zero: Array | None
    portfolio_weight_l2_to_same_eta: Array | None
    portfolio_threshold_error_to_eta_zero: Array | None
    portfolio_threshold_error_to_same_eta: Array | None
    aggregate: DualReferenceAggregate

    def __post_init__(self) -> None:
        rows = len(np.asarray(self.candidate_raw_objectives).reshape(-1))
        if rows == 0:
            raise ValueError("dual-reference evaluation must be nonempty")
        required = (
            "candidate_raw_objectives",
            "candidate_regularized_objectives",
            "raw_operational_excess_to_eta_zero",
            "same_eta_regularized_gap",
            "signed_raw_difference_to_same_eta",
            "reference_regularization_bias",
            "bias_identity_residual",
            "decision_l2_to_eta_zero",
            "decision_l2_to_same_eta",
        )
        optional = (
            "portfolio_weight_l2_to_eta_zero",
            "portfolio_weight_l2_to_same_eta",
            "portfolio_threshold_error_to_eta_zero",
            "portfolio_threshold_error_to_same_eta",
        )
        for field_name in required:
            values = _immutable_vector(getattr(self, field_name), field_name)
            if values.shape != (rows,):
                raise ValueError(f"{field_name} must align with contexts")
            object.__setattr__(self, field_name, values)
        for field_name in optional:
            original = getattr(self, field_name)
            if original is None:
                continue
            values = _immutable_vector(original, field_name)
            if values.shape != (rows,):
                raise ValueError(f"{field_name} must align with contexts")
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True)
class _ReferenceKktAudit:
    augmented_matrix: Array
    augmented_rhs: Array
    augmented_multiplier: Array
    equality_multiplier: Array
    lower_bound_multipliers: Array
    stationarity_residual_linf: float
    equality_residual_linf: float
    lower_bound_violation_linf: float
    lower_bound_dual_residual_linf: float
    complementarity_residual_linf: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.stationarity_residual_linf,
            self.equality_residual_linf,
            self.lower_bound_violation_linf,
            self.lower_bound_dual_residual_linf,
            self.complementarity_residual_linf,
        )


def solve_reference_newton(
    problem: EqualityConstrainedReferenceProblem,
    context: Array,
    scenarios: Array,
    probabilities: Array,
    *,
    ridge: float,
    config: ReferenceNewtonConfig | None = None,
) -> ReferenceSolveResult:
    """Solve one finite-support reference problem and fail closed on KKT audit."""

    if not isinstance(problem, EqualityConstrainedReferenceProblem):
        raise TypeError("problem does not implement the reference-solver primitives")
    ridge_value = _nonnegative_finite(ridge, "ridge")
    controls = config or ReferenceNewtonConfig()
    context_value = np.asarray(context, dtype=np.float64).reshape(-1)
    if context_value.shape != (problem.context_dim,) or np.any(~np.isfinite(context_value)):
        raise ValueError("context has the wrong shape")
    scenario_values = matrix(
        scenarios,
        columns=problem.scenario_dim,
        name="scenarios",
    )
    probability_values = probability_vector(
        probabilities,
        len(scenario_values),
    )
    equality_matrix, equality_rhs = problem.reference_equality_numpy(context_value)
    equality_matrix, equality_rhs = _validate_equality(
        equality_matrix,
        equality_rhs,
        problem.decision_dim,
    )
    lower_bounds = _reference_lower_bounds(
        problem,
        context_value,
        problem.decision_dim,
    )
    finite_lower_bounds = np.isfinite(lower_bounds)
    decision = np.asarray(
        problem.reference_center_numpy(context_value),
        dtype=np.float64,
    ).reshape(-1)
    if decision.shape != (problem.decision_dim,):
        raise ValueError("reference center has the wrong shape")
    if not problem.reference_decision_is_valid_numpy(decision):
        raise ValueError("reference center lies outside the Newton domain")
    initial_constraint = equality_matrix @ decision - equality_rhs
    if np.max(np.abs(initial_constraint), initial=0.0) > controls.constraint_tolerance:
        raise ValueError("reference center does not satisfy the affine constraints")
    initial_bound_violation = np.max(
        np.maximum(lower_bounds[finite_lower_bounds] - decision[finite_lower_bounds], 0.0),
        initial=0.0,
    )
    if initial_bound_violation > controls.bound_feasibility_tolerance:
        raise ValueError("reference center violates a decision lower bound")
    regularization_center = decision.copy()
    active_bounds = finite_lower_bounds & (decision - lower_bounds <= controls.active_bound_tolerance)

    raw_value, objective, gradient, hessian = problem.reference_objective_derivatives_numpy(
        decision,
        context_value,
        scenario_values,
        probability_values,
        ridge=ridge_value,
    )
    gradient, hessian = _validate_derivatives(
        gradient,
        hessian,
        problem.decision_dim,
        controls.hessian_symmetry_tolerance,
    )
    total_backtracking = 0
    numerical_descent_safeguards = 0
    active_set_changes = 0

    for iteration in range(controls.maximum_iterations + 1):
        audit = _reference_kkt_audit(
            gradient,
            decision,
            equality_matrix,
            equality_rhs,
            lower_bounds,
            active_bounds,
        )
        converged = (
            audit.stationarity_residual_linf <= controls.stationarity_tolerance
            and audit.equality_residual_linf <= controls.constraint_tolerance
            and audit.lower_bound_violation_linf <= controls.bound_feasibility_tolerance
            and audit.lower_bound_dual_residual_linf <= controls.dual_feasibility_tolerance
            and audit.complementarity_residual_linf <= controls.complementarity_tolerance
        )
        if converged:
            displacement = decision - regularization_center
            penalty = 0.5 * ridge_value * float(displacement @ displacement)
            if not math.isclose(
                objective,
                raw_value + penalty,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ReferenceSolveError("reference objective failed the regularization identity")
            return ReferenceSolveResult(
                decision=decision,
                ridge=ridge_value,
                raw_objective=float(raw_value),
                regularized_objective=float(objective),
                regularization_penalty=penalty,
                multiplier=audit.equality_multiplier,
                lower_bound_multipliers=audit.lower_bound_multipliers,
                lower_bounds_enforced=bool(np.any(finite_lower_bounds)),
                stationarity_residual_linf=(audit.stationarity_residual_linf),
                constraint_residual_linf=audit.equality_residual_linf,
                lower_bound_violation_linf=(audit.lower_bound_violation_linf),
                lower_bound_dual_residual_linf=(audit.lower_bound_dual_residual_linf),
                complementarity_residual_linf=(audit.complementarity_residual_linf),
                active_lower_bound_count=int(np.sum(active_bounds)),
                active_set_changes=active_set_changes,
                iterations=iteration,
                backtracking_steps=total_backtracking,
                numerical_descent_safeguard_steps=(numerical_descent_safeguards),
            )
        if iteration == controls.maximum_iterations:
            break

        active_indices = np.flatnonzero(active_bounds)
        if (
            audit.stationarity_residual_linf <= controls.stationarity_tolerance
            and len(active_indices)
            and audit.lower_bound_dual_residual_linf > controls.dual_feasibility_tolerance
        ):
            release = active_indices[int(np.argmin(audit.lower_bound_multipliers[active_indices]))]
            active_bounds[release] = False
            active_set_changes += 1
            continue

        augmented_constraint = audit.augmented_matrix @ decision - audit.augmented_rhs
        kkt_matrix = np.block(
            [
                [hessian, audit.augmented_matrix.T],
                [
                    audit.augmented_matrix,
                    np.zeros(
                        (
                            len(audit.augmented_rhs),
                            len(audit.augmented_rhs),
                        ),
                        dtype=np.float64,
                    ),
                ],
            ]
        )
        right_hand_side = -np.concatenate(
            (
                gradient + audit.augmented_matrix.T @ audit.augmented_multiplier,
                augmented_constraint,
            )
        )
        try:
            newton_direction = np.linalg.solve(kkt_matrix, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise ReferenceSolveError("reference KKT system is singular") from exc
        decision_direction = newton_direction[: problem.decision_dim]

        if (
            np.max(np.abs(decision_direction), initial=0.0) <= controls.stationarity_tolerance
            and len(active_indices)
            and audit.lower_bound_dual_residual_linf > controls.dual_feasibility_tolerance
        ):
            release = active_indices[int(np.argmin(audit.lower_bound_multipliers[active_indices]))]
            active_bounds[release] = False
            active_set_changes += 1
            continue

        directional_derivative = float(gradient @ decision_direction)
        if not math.isfinite(directional_derivative):
            raise ReferenceSolveError("Newton directional derivative is nonfinite")
        if directional_derivative >= 0.0:
            predicted_curvature = float(decision_direction @ hessian @ decision_direction)
            numerical_descent_safeguards += 1
            if not math.isfinite(predicted_curvature) or predicted_curvature <= 0.0:
                raise ReferenceSolveError("Newton direction failed the positive-curvature audit")
            directional_derivative = -predicted_curvature

        maximum_feasible_step, blocking_bounds = _maximum_bound_step(
            decision,
            decision_direction,
            lower_bounds,
            active_bounds,
        )
        if maximum_feasible_step <= controls.minimum_step:
            if len(blocking_bounds) == 0:
                raise ReferenceSolveError("reference Newton step is smaller than the minimum step")
            decision[blocking_bounds] = lower_bounds[blocking_bounds]
            active_bounds[blocking_bounds] = True
            active_set_changes += len(blocking_bounds)
            continue

        step = min(1.0, maximum_feasible_step)
        accepted = False
        while step >= controls.minimum_step:
            hits_bound = maximum_feasible_step <= 1.0 and math.isclose(
                step,
                maximum_feasible_step,
                rel_tol=1e-12,
                abs_tol=controls.bound_feasibility_tolerance,
            )
            candidate_active = active_bounds.copy()
            candidate = decision + step * decision_direction
            if hits_bound:
                candidate[blocking_bounds] = lower_bounds[blocking_bounds]
                candidate_active[blocking_bounds] = True
            candidate_bound_violation = np.max(
                np.maximum(
                    lower_bounds[finite_lower_bounds] - candidate[finite_lower_bounds],
                    0.0,
                ),
                initial=0.0,
            )
            if (
                candidate_bound_violation <= controls.bound_feasibility_tolerance
                and problem.reference_decision_is_valid_numpy(candidate)
            ):
                (
                    candidate_raw,
                    candidate_objective,
                    candidate_gradient,
                    candidate_hessian,
                ) = problem.reference_objective_derivatives_numpy(
                    candidate,
                    context_value,
                    scenario_values,
                    probability_values,
                    ridge=ridge_value,
                )
                if math.isfinite(candidate_objective):
                    candidate_gradient, candidate_hessian = _validate_derivatives(
                        candidate_gradient,
                        candidate_hessian,
                        problem.decision_dim,
                        controls.hessian_symmetry_tolerance,
                    )
                    candidate_audit = _reference_kkt_audit(
                        candidate_gradient,
                        candidate,
                        equality_matrix,
                        equality_rhs,
                        lower_bounds,
                        candidate_active,
                    )
                    armijo_holds = candidate_objective <= (
                        objective + controls.armijo_constant * step * directional_derivative
                    )
                    objective_resolution = (
                        64.0
                        * np.finfo(np.float64).eps
                        * max(
                            1.0,
                            abs(objective),
                            abs(candidate_objective),
                        )
                    )
                    objective_change_is_unresolved = (
                        abs(step * directional_derivative) <= objective_resolution
                        and candidate_objective <= objective + objective_resolution
                    )
                    residual_safeguard_holds = (
                        objective_change_is_unresolved and candidate_audit.maximum_residual < audit.maximum_residual
                    )
                else:
                    armijo_holds = False
                    residual_safeguard_holds = False
                if armijo_holds or residual_safeguard_holds:
                    if residual_safeguard_holds and not armijo_holds:
                        numerical_descent_safeguards += 1
                    decision = candidate
                    active_bounds = candidate_active
                    if hits_bound:
                        active_set_changes += len(blocking_bounds)
                    raw_value = float(candidate_raw)
                    objective = float(candidate_objective)
                    gradient = candidate_gradient
                    hessian = candidate_hessian
                    accepted = True
                    break
            step *= controls.backtracking_factor
            total_backtracking += 1
        if not accepted:
            raise ReferenceSolveError("reference Newton line search failed")

    raise ReferenceSolveError("reference Newton solve did not satisfy the residual tolerances")


def solve_reference_batch(
    problem: EqualityConstrainedReferenceProblem,
    contexts: Array,
    catalog: Array,
    probabilities: Array,
    *,
    ridge: float,
    config: ReferenceNewtonConfig | None = None,
) -> ReferenceBatchResult:
    """Compute one audited high-accuracy reference per context."""

    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    catalog_values = matrix(catalog, columns=problem.scenario_dim, name="catalog")
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if probability_values.shape != (len(context_values), len(catalog_values)):
        raise ValueError("probabilities must align with contexts and catalog")
    if np.any(~np.isfinite(probability_values)) or np.any(probability_values < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(
        probability_values.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("each probability row must sum to one")

    results = [
        solve_reference_newton(
            problem,
            context,
            catalog_values,
            weights,
            ridge=ridge,
            config=config,
        )
        for context, weights in zip(
            context_values,
            probability_values,
            strict=True,
        )
    ]
    return ReferenceBatchResult(
        ridge=float(ridge),
        lower_bounds_enforced=results[0].lower_bounds_enforced,
        decisions=np.vstack([result.decision for result in results]),
        raw_objectives=np.asarray([result.raw_objective for result in results]),
        regularized_objectives=np.asarray([result.regularized_objective for result in results]),
        regularization_penalties=np.asarray([result.regularization_penalty for result in results]),
        stationarity_residuals_linf=np.asarray([result.stationarity_residual_linf for result in results]),
        constraint_residuals_linf=np.asarray([result.constraint_residual_linf for result in results]),
        lower_bound_violations_linf=np.asarray([result.lower_bound_violation_linf for result in results]),
        lower_bound_dual_residuals_linf=np.asarray([result.lower_bound_dual_residual_linf for result in results]),
        complementarity_residuals_linf=np.asarray([result.complementarity_residual_linf for result in results]),
        active_lower_bound_counts=np.asarray([result.active_lower_bound_count for result in results]),
        active_set_changes=np.asarray([result.active_set_changes for result in results]),
        iterations=np.asarray([result.iterations for result in results]),
        backtracking_steps=np.asarray([result.backtracking_steps for result in results]),
        numerical_descent_safeguard_steps=np.asarray([result.numerical_descent_safeguard_steps for result in results]),
    )


def solve_dual_references(
    problem: EqualityConstrainedReferenceProblem,
    contexts: Array,
    catalog: Array,
    probabilities: Array,
    *,
    ridge: float,
    config: ReferenceNewtonConfig | None = None,
) -> DualReferenceResult:
    """Compute the raw optimum and the same-ridge evaluation reference."""

    ridge_value = _nonnegative_finite(ridge, "ridge")
    eta_zero = solve_reference_batch(
        problem,
        contexts,
        catalog,
        probabilities,
        ridge=0.0,
        config=config,
    )
    if ridge_value == 0.0:
        same_eta = eta_zero
    else:
        same_eta = solve_reference_batch(
            problem,
            contexts,
            catalog,
            probabilities,
            ridge=ridge_value,
            config=config,
        )
    return DualReferenceResult(eta_zero=eta_zero, same_eta=same_eta)


def evaluate_dual_reference_metrics(
    problem: EqualityConstrainedReferenceProblem,
    candidate_decisions: Array,
    contexts: Array,
    catalog: Array,
    probabilities: Array,
    *,
    ridge: float,
    references: DualReferenceResult | None = None,
    newton_config: ReferenceNewtonConfig | None = None,
    nonnegative_tolerance: float = 1e-8,
    constraint_tolerance: float = 1e-7,
) -> tuple[DualReferenceResult, DualReferenceEvaluation]:
    """Evaluate candidate decisions without clipping any signed gap.

    For each context, the exact identity is

    ``raw excess to eta=0 = signed raw difference to same eta
                              + reference regularization bias``.
    """

    ridge_value = _nonnegative_finite(ridge, "ridge")
    nonnegative_tolerance = _nonnegative_finite(
        nonnegative_tolerance,
        "nonnegative_tolerance",
    )
    constraint_tolerance = _nonnegative_finite(
        constraint_tolerance,
        "constraint_tolerance",
    )
    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    catalog_values = matrix(catalog, columns=problem.scenario_dim, name="catalog")
    decision_values = matrix(
        candidate_decisions,
        columns=problem.decision_dim,
        name="candidate_decisions",
    )
    if len(decision_values) != len(context_values):
        raise ValueError("candidate decisions and contexts must align")
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if probability_values.shape != (len(context_values), len(catalog_values)):
        raise ValueError("probabilities must align with contexts and catalog")
    _audit_candidate_constraints(
        problem,
        decision_values,
        context_values,
        tolerance=constraint_tolerance,
    )

    dual = references or solve_dual_references(
        problem,
        context_values,
        catalog_values,
        probability_values,
        ridge=ridge_value,
        config=newton_config,
    )
    if dual.same_eta.ridge != ridge_value:
        raise ValueError("same-eta reference does not match the requested ridge")
    if dual.same_eta.decisions.shape != decision_values.shape:
        raise ValueError("references and candidate decisions do not align")

    candidate_raw = exact_expected_loss(
        problem,
        decision_values,
        context_values,
        catalog_values,
        probability_values,
    ).context_losses
    candidate_penalty = _regularization_penalties(
        problem,
        decision_values,
        context_values,
        ridge_value,
    )
    candidate_regularized = candidate_raw + candidate_penalty
    raw_excess = candidate_raw - dual.eta_zero.raw_objectives
    same_eta_gap = candidate_regularized - dual.same_eta.regularized_objectives
    signed_raw_difference = candidate_raw - dual.same_eta.raw_objectives
    reference_bias = dual.same_eta.raw_objectives - dual.eta_zero.raw_objectives
    identity_residual = raw_excess - (signed_raw_difference + reference_bias)

    _audit_nonnegative(
        "raw operational excess to eta=0",
        raw_excess,
        nonnegative_tolerance,
    )
    _audit_nonnegative(
        "same-eta regularized gap",
        same_eta_gap,
        nonnegative_tolerance,
    )
    _audit_nonnegative(
        "reference regularization bias",
        reference_bias,
        nonnegative_tolerance,
    )
    if np.max(np.abs(identity_residual), initial=0.0) > 1e-11:
        raise ReferenceMetricAuditError("dual-reference bias identity failed")

    eta_zero_difference = decision_values - dual.eta_zero.decisions
    same_eta_difference = decision_values - dual.same_eta.decisions
    decision_l2_eta_zero = np.linalg.norm(eta_zero_difference, axis=1)
    decision_l2_same_eta = np.linalg.norm(same_eta_difference, axis=1)

    weight_l2_eta_zero: Array | None = None
    weight_l2_same_eta: Array | None = None
    threshold_error_eta_zero: Array | None = None
    threshold_error_same_eta: Array | None = None
    if problem.name == "smoothed_cvar_portfolio":
        weight_l2_eta_zero = np.linalg.norm(
            eta_zero_difference[:, :-1],
            axis=1,
        )
        weight_l2_same_eta = np.linalg.norm(
            same_eta_difference[:, :-1],
            axis=1,
        )
        threshold_error_eta_zero = eta_zero_difference[:, -1]
        threshold_error_same_eta = same_eta_difference[:, -1]

    aggregate = DualReferenceAggregate(
        mean_raw_operational_excess_to_eta_zero=float(np.mean(raw_excess)),
        mean_same_eta_regularized_gap=float(np.mean(same_eta_gap)),
        mean_signed_raw_difference_to_same_eta=float(np.mean(signed_raw_difference)),
        mean_reference_regularization_bias=float(np.mean(reference_bias)),
        maximum_absolute_bias_identity_residual=float(np.max(np.abs(identity_residual), initial=0.0)),
        decision_l2_rms_to_eta_zero=_l2_rms(decision_l2_eta_zero),
        decision_l2_rms_to_same_eta=_l2_rms(decision_l2_same_eta),
        portfolio_weight_l2_rms_to_eta_zero=(None if weight_l2_eta_zero is None else _l2_rms(weight_l2_eta_zero)),
        portfolio_weight_l2_rms_to_same_eta=(None if weight_l2_same_eta is None else _l2_rms(weight_l2_same_eta)),
        portfolio_threshold_rmse_to_eta_zero=(
            None if threshold_error_eta_zero is None else _l2_rms(threshold_error_eta_zero)
        ),
        portfolio_threshold_rmse_to_same_eta=(
            None if threshold_error_same_eta is None else _l2_rms(threshold_error_same_eta)
        ),
    )
    return dual, DualReferenceEvaluation(
        candidate_raw_objectives=candidate_raw,
        candidate_regularized_objectives=candidate_regularized,
        raw_operational_excess_to_eta_zero=raw_excess,
        same_eta_regularized_gap=same_eta_gap,
        signed_raw_difference_to_same_eta=signed_raw_difference,
        reference_regularization_bias=reference_bias,
        bias_identity_residual=identity_residual,
        decision_l2_to_eta_zero=decision_l2_eta_zero,
        decision_l2_to_same_eta=decision_l2_same_eta,
        portfolio_weight_l2_to_eta_zero=weight_l2_eta_zero,
        portfolio_weight_l2_to_same_eta=weight_l2_same_eta,
        portfolio_threshold_error_to_eta_zero=threshold_error_eta_zero,
        portfolio_threshold_error_to_same_eta=threshold_error_same_eta,
        aggregate=aggregate,
    )


def _reference_lower_bounds(
    problem: EqualityConstrainedReferenceProblem,
    context: Array,
    decision_dim: int,
) -> Array:
    if not isinstance(problem, LowerBoundConstrainedReferenceProblem):
        return np.full(decision_dim, -np.inf, dtype=np.float64)
    bounds = np.asarray(
        problem.reference_lower_bounds_numpy(context),
        dtype=np.float64,
    ).reshape(-1)
    if bounds.shape != (decision_dim,):
        raise ValueError("reference lower bounds have the wrong shape")
    if np.any(np.isnan(bounds)) or np.any(np.isposinf(bounds)):
        raise ValueError("reference lower bounds must be finite or negative infinity")
    return np.ascontiguousarray(bounds)


def _reference_kkt_audit(
    gradient: Array,
    decision: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    lower_bounds: Array,
    active_bounds: Array,
) -> _ReferenceKktAudit:
    decision_dim = len(decision)
    active_indices = np.flatnonzero(active_bounds)
    if len(active_indices):
        active_matrix = np.eye(decision_dim, dtype=np.float64)[active_indices]
        augmented_matrix = np.vstack((equality_matrix, active_matrix))
        augmented_rhs = np.concatenate((equality_rhs, lower_bounds[active_indices]))
    else:
        augmented_matrix = equality_matrix
        augmented_rhs = equality_rhs
    if np.linalg.matrix_rank(augmented_matrix) != len(augmented_rhs):
        raise ReferenceSolveError("active lower bounds are linearly dependent with the equalities")
    try:
        augmented_multiplier = np.linalg.solve(
            augmented_matrix @ augmented_matrix.T,
            -(augmented_matrix @ gradient),
        )
    except np.linalg.LinAlgError as exc:
        raise ReferenceSolveError("reference multiplier system is singular") from exc
    equality_count = len(equality_rhs)
    equality_multiplier = augmented_multiplier[:equality_count]
    lower_bound_multipliers: Array = np.zeros(
        decision_dim,
        dtype=np.float64,
    )
    if len(active_indices):
        lower_bound_multipliers[active_indices] = -augmented_multiplier[equality_count:]
    stationarity = gradient + equality_matrix.T @ equality_multiplier - lower_bound_multipliers
    equality_residual = equality_matrix @ decision - equality_rhs
    finite = np.isfinite(lower_bounds)
    if np.any(finite):
        slack = decision[finite] - lower_bounds[finite]
        lower_bound_violation = float(np.max(np.maximum(-slack, 0.0), initial=0.0))
        lower_bound_dual_residual = float(
            np.max(
                np.maximum(-lower_bound_multipliers[finite], 0.0),
                initial=0.0,
            )
        )
        complementarity_residual = float(
            np.max(
                np.abs(lower_bound_multipliers[finite] * slack),
                initial=0.0,
            )
        )
    else:
        lower_bound_violation = 0.0
        lower_bound_dual_residual = 0.0
        complementarity_residual = 0.0
    return _ReferenceKktAudit(
        augmented_matrix=np.ascontiguousarray(augmented_matrix),
        augmented_rhs=np.ascontiguousarray(augmented_rhs),
        augmented_multiplier=np.ascontiguousarray(augmented_multiplier),
        equality_multiplier=np.ascontiguousarray(equality_multiplier),
        lower_bound_multipliers=lower_bound_multipliers,
        stationarity_residual_linf=float(np.max(np.abs(stationarity), initial=0.0)),
        equality_residual_linf=float(np.max(np.abs(equality_residual), initial=0.0)),
        lower_bound_violation_linf=lower_bound_violation,
        lower_bound_dual_residual_linf=lower_bound_dual_residual,
        complementarity_residual_linf=complementarity_residual,
    )


def _maximum_bound_step(
    decision: Array,
    direction: Array,
    lower_bounds: Array,
    active_bounds: Array,
) -> tuple[float, Array]:
    candidates = np.isfinite(lower_bounds) & ~active_bounds & (direction < 0.0)
    indices = np.flatnonzero(candidates)
    if len(indices) == 0:
        return math.inf, np.empty(0, dtype=np.int64)
    ratios = (decision[indices] - lower_bounds[indices]) / (-direction[indices])
    minimum = float(np.min(ratios))
    if minimum < -1e-12:
        raise ReferenceSolveError("current decision violates a lower bound")
    minimum = max(0.0, minimum)
    blockers = indices[
        np.isclose(
            ratios,
            minimum,
            rtol=1e-10,
            atol=1e-14,
        )
    ]
    return minimum, blockers


def _validate_equality(
    equality_matrix: Array,
    equality_rhs: Array,
    decision_dim: int,
) -> tuple[Array, Array]:
    matrix_value = np.asarray(equality_matrix, dtype=np.float64)
    rhs_value = np.asarray(equality_rhs, dtype=np.float64).reshape(-1)
    if (
        matrix_value.ndim != 2
        or matrix_value.shape[1] != decision_dim
        or matrix_value.shape[0] == 0
        or rhs_value.shape != (matrix_value.shape[0],)
        or np.any(~np.isfinite(matrix_value))
        or np.any(~np.isfinite(rhs_value))
    ):
        raise ValueError("reference equality has an invalid shape or value")
    if np.linalg.matrix_rank(matrix_value) != matrix_value.shape[0]:
        raise ValueError("reference equality matrix must have full row rank")
    return np.ascontiguousarray(matrix_value), np.ascontiguousarray(rhs_value)


def _validate_derivatives(
    gradient: Array,
    hessian: Array,
    decision_dim: int,
    symmetry_tolerance: float,
) -> tuple[Array, Array]:
    gradient_value = np.asarray(gradient, dtype=np.float64).reshape(-1)
    hessian_value = np.asarray(hessian, dtype=np.float64)
    if gradient_value.shape != (decision_dim,) or hessian_value.shape != (
        decision_dim,
        decision_dim,
    ):
        raise ReferenceSolveError("reference derivatives have the wrong shape")
    if np.any(~np.isfinite(gradient_value)) or np.any(~np.isfinite(hessian_value)):
        raise ReferenceSolveError("reference derivatives are nonfinite")
    asymmetry = np.max(np.abs(hessian_value - hessian_value.T), initial=0.0)
    scale = max(1.0, float(np.max(np.abs(hessian_value), initial=0.0)))
    if asymmetry > symmetry_tolerance * scale:
        raise ReferenceSolveError("reference Hessian failed its symmetry audit")
    return gradient_value, 0.5 * (hessian_value + hessian_value.T)


def _audit_candidate_constraints(
    problem: EqualityConstrainedReferenceProblem,
    decisions: Array,
    contexts: Array,
    *,
    tolerance: float,
) -> None:
    for decision, context in zip(decisions, contexts, strict=True):
        if not problem.reference_decision_is_valid_numpy(decision):
            raise ReferenceMetricAuditError("candidate decision is outside the model domain")
        equality_matrix, equality_rhs = problem.reference_equality_numpy(context)
        residual = np.asarray(equality_matrix) @ decision - np.asarray(equality_rhs)
        if np.max(np.abs(residual), initial=0.0) > tolerance:
            raise ReferenceMetricAuditError("candidate decision violates the affine constraint")
        lower_bounds = _reference_lower_bounds(
            problem,
            context,
            problem.decision_dim,
        )
        finite = np.isfinite(lower_bounds)
        if (
            np.max(
                np.maximum(lower_bounds[finite] - decision[finite], 0.0),
                initial=0.0,
            )
            > tolerance
        ):
            raise ReferenceMetricAuditError("candidate decision violates a lower bound")


def _regularization_penalties(
    problem: EqualityConstrainedReferenceProblem,
    decisions: Array,
    contexts: Array,
    ridge: float,
) -> Array:
    penalties: Array = np.empty(len(decisions), dtype=np.float64)
    for row, (decision, context) in enumerate(zip(decisions, contexts, strict=True)):
        center = np.asarray(
            problem.reference_center_numpy(context),
            dtype=np.float64,
        ).reshape(-1)
        displacement = decision - center
        penalties[row] = 0.5 * ridge * float(displacement @ displacement)
    return penalties


def _audit_nonnegative(
    name: str,
    values: Array,
    tolerance: float,
) -> None:
    minimum = float(np.min(values))
    if minimum < -tolerance:
        raise ReferenceMetricAuditError(f"{name} is {minimum:.6e}, below the allowed {-tolerance:.6e}")


def _l2_rms(values: Array) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(array**2)))


def _immutable_vector(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    immutable = np.ascontiguousarray(array.copy())
    immutable.setflags(write=False)
    return immutable


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result
