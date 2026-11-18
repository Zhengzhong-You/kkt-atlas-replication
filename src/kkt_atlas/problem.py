"""Problem interface used by the canonical offline KKT Atlas implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor


@runtime_checkable
class DifferentiableStochasticProblem(Protocol):
    """Operations shared by chart construction, training, and evaluation."""

    @property
    def name(self) -> str:
        """Name used in experiment records."""

    @property
    def context_dim(self) -> int:
        """Number of context coordinates."""

    @property
    def scenario_dim(self) -> int:
        """Number of coordinates in one scenario."""

    @property
    def decision_dim(self) -> int:
        """Number of decision coordinates."""

    def transform_numpy(self, outcomes: Array) -> Array:
        """Map feasible outcomes to unconstrained scenario coordinates."""

    def decode_torch(self, transformed: Tensor) -> Tensor:
        """Map transformed coordinates back to feasible outcomes."""

    def solve_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> Array:
        """Solve one finite-support stochastic program for diagnostics."""

    def solve_torch(
        self,
        contexts: Tensor,
        scenarios: Tensor,
        probabilities: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        """Solve a batch of finite-support problems with autograd support."""

    def kkt_contributions_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
    ) -> Array:
        """Return one scenario KKT contribution per outcome."""

    def realized_loss_torch(
        self,
        decisions: Tensor,
        outcomes: Tensor,
        contexts: Tensor,
    ) -> Tensor:
        """Return the unregularized realized operational loss per observation."""


@runtime_checkable
class EqualityConstrainedReferenceProblem(DifferentiableStochasticProblem, Protocol):
    """Additional primitives used only by the high-accuracy evaluator."""

    def reference_center_numpy(self, context: Array) -> Array:
        """Return the center of the quadratic decision regularizer."""

    def reference_equality_numpy(self, context: Array) -> tuple[Array, Array]:
        """Return ``(A, b)`` for the affine constraints ``A @ z = b``."""

    def reference_decision_is_valid_numpy(self, decision: Array) -> bool:
        """Return whether a decision belongs to the Newton solver's domain."""

    def reference_objective_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> tuple[float, float, Array, Array]:
        """Return raw value, regularized value, gradient, and Hessian."""


@runtime_checkable
class LowerBoundConstrainedReferenceProblem(Protocol):
    """Optional lower-bound primitives for the high-accuracy evaluator."""

    def reference_lower_bounds_numpy(self, context: Array) -> Array:
        """Return coordinatewise lower bounds for the decision."""


def realized_loss_from_scenarios_torch(
    problem: DifferentiableStochasticProblem,
    contexts: Tensor,
    scenarios: Tensor,
    probabilities: Tensor,
    outcomes: Tensor,
    *,
    ridge: float,
) -> Tensor:
    """Evaluate training losses through the most specific exact problem path."""

    scenario_fast_path = getattr(
        problem,
        "realized_loss_from_scenarios_torch",
        None,
    )
    if callable(scenario_fast_path):
        return scenario_fast_path(
            contexts,
            scenarios,
            probabilities,
            outcomes,
            ridge=ridge,
        )
    fast_path = getattr(problem, "realized_loss_from_means_torch", None)
    if callable(fast_path):
        return fast_path(
            contexts,
            probabilities @ scenarios,
            outcomes,
            ridge=ridge,
        )
    decisions = problem.solve_torch(
        contexts,
        scenarios,
        probabilities,
        ridge=ridge,
    )
    return problem.realized_loss_torch(decisions, outcomes, contexts)


def realized_loss_from_means_torch(
    problem: DifferentiableStochasticProblem,
    contexts: Tensor,
    conditional_means: Tensor,
    outcomes: Tensor,
    *,
    ridge: float,
) -> Tensor:
    """Evaluate losses for a problem whose stochastic program depends on its mean."""

    fast_path = getattr(problem, "realized_loss_from_means_torch", None)
    if callable(fast_path):
        return fast_path(
            contexts,
            conditional_means,
            outcomes,
            ridge=ridge,
        )
    solve_from_means = getattr(problem, "solve_from_means_torch", None)
    if not callable(solve_from_means):
        raise TypeError("problem does not expose an exact solve from scenario means")
    decisions = solve_from_means(
        contexts,
        conditional_means,
        ridge=ridge,
    )
    return problem.realized_loss_torch(decisions, outcomes, contexts)


def probability_vector(values: Array, expected_size: int) -> Array:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if probabilities.shape != (expected_size,):
        raise ValueError(f"probabilities must have shape ({expected_size},)")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    # Do not perturb an already unit-mass vector at every layer. Repeated
    # renormalization changed identical solver inputs in their last few bits.
    if abs(total - 1.0) <= 8 * np.finfo(np.float64).eps:
        return probabilities.copy()
    return probabilities / total


def restore_prescription(problem: object, record: dict | None) -> object:
    """Restore only explicitly supported, hash-bound solver metadata."""
    if record is None:
        return problem
    if not isinstance(record, dict) or record.get("schema") != "kkt-prescription-v1":
        raise ValueError("Invalid prescription metadata")
    base = getattr(problem, "_problem", problem)
    backend = record.get("backend")
    if backend == "scipy_slsqp":
        if set(record) != {"schema", "backend", "tolerance", "solver_ftol", "maximum_iterations"}:
            raise ValueError("Invalid SLSQP prescription fields")
        from kkt_atlas.converged_smooth_problem_v1 import ConvergedSmoothProblem
        return ConvergedSmoothProblem(base, tolerance=record["tolerance"], solver_ftol=record["solver_ftol"],
            maximum_iterations=record["maximum_iterations"])
    if backend == "clarabel_exp_cone":
        if set(record) not in ({"schema", "backend", "tolerance", "cone_tolerance"},
                {"schema", "backend", "tolerance", "cone_tolerance", "fallback_on_failure"}):
            raise ValueError("Invalid Clarabel prescription fields")
        from kkt_atlas.clarabel_smooth_problem_v1 import ClarabelSmoothProblem
        return ClarabelSmoothProblem(base, tolerance=record["tolerance"], cone_tolerance=record["cone_tolerance"],
            fallback_on_failure=record.get("fallback_on_failure",False))
    if backend == "gurobi_net":
        required = {"schema","backend","tolerance"}
        vendor = set(record) & {"joint_method","recourse_method","fallback_joint_method"}
        optional = {"last_resort_reservation_refinements","exact_duplicate_aggregation",
                    "solver_threads","context_workers","worker_solver_threads","parallelism","parallel_derivatives","unresolved_face_polish_digits"}
        if (not required <= set(record) or set(record)-required-vendor-optional
                or vendor not in (set(),{"joint_method","recourse_method"},
                    {"joint_method","recourse_method","fallback_joint_method"})):
            raise ValueError("Invalid NET prescription fields")
        if "last_resort_reservation_refinements" in record and (
                type(record["last_resort_reservation_refinements"]) is not int
                or record["last_resort_reservation_refinements"] != 50):
            raise ValueError("Unsupported NET refinement limit")
        if "exact_duplicate_aggregation" in record and record["exact_duplicate_aggregation"] is not True:
            raise ValueError("Unsupported NET duplicate aggregation")
        if "unresolved_face_polish_digits" in record and (
                type(record["unresolved_face_polish_digits"]) is not int
                or record["unresolved_face_polish_digits"] != 80):
            raise ValueError("Unsupported NET active-face precision")
        if "solver_threads" in record and (
                type(record["solver_threads"]) is not int or not 1 <= record["solver_threads"] <= 8):
            raise ValueError("Invalid NET vendor thread budget")
        controls = dict(tolerance=record["tolerance"],joint_method=record.get("joint_method",-1),
            recourse_method=record.get("recourse_method",-1),
            fallback_joint_method=record.get("fallback_joint_method"),
            solver_threads=record.get("solver_threads"))
        if "parallel_derivatives" in record and (type(record["parallel_derivatives"]) is not bool or "context_workers" not in record):
            raise ValueError("Invalid NET parallel derivative policy")
        parallel = {"context_workers","worker_solver_threads","parallelism"} & set(record)
        if parallel:
            if (parallel != {"context_workers","worker_solver_threads","parallelism"}
                    or type(record["context_workers"]) is not int or not 1 <= record["context_workers"] <= 32
                    or type(record["worker_solver_threads"]) is not int or record["worker_solver_threads"] != 1
                    or record["parallelism"] not in {"independent_context_chunks",
                        "independent_context_chunks_with_exact_probability_reuse"}):
                raise ValueError("Invalid NET context-worker prescription")
            from kkt_atlas.parallel_network_problem_v1 import ParallelNetworkProblem
            return ParallelNetworkProblem(base,workers=record["context_workers"],parallel_derivatives=record.get("parallel_derivatives",False),**controls)
        from kkt_atlas.converged_network_problem_v1 import ConvergedNetworkProblem
        return ConvergedNetworkProblem(base,**controls)
    raise ValueError("Unknown prescription backend")


def matrix(values: Array, *, columns: int, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (n,{columns})")
    if len(array) == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return np.array(array, dtype=np.float64, order="C", copy=True)
