"""Multicommodity capacity reservation with two fixed paths per commodity."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from kkt_atlas.problem import Array, Tensor, matrix, probability_vector

_PROBLEM_VERSION = "multicommodity-network-reservation-problem-v1"
_GRID_SIZE = 4
_LINK_COUNT = 48
_MASTER_COMMODITY_COUNT = 128

Node = tuple[int, int]
Link = tuple[Node, Node]
Commodity = tuple[Node, Node]


@dataclass(frozen=True)
class NetworkRecourseResult:
    routes: Array
    residuals: Array
    iterations: int

    def __post_init__(self) -> None:
        routes = np.ascontiguousarray(np.asarray(self.routes, dtype=np.float64))
        residuals = np.ascontiguousarray(
            np.asarray(self.residuals, dtype=np.float64).reshape(-1)
        )
        if routes.ndim != 2 or residuals.shape != (len(routes),):
            raise ValueError("recourse routes and residuals do not align")
        if np.any(~np.isfinite(routes)) or np.any(~np.isfinite(residuals)):
            raise ValueError("recourse result must be finite")
        if self.iterations < 0:
            raise ValueError("recourse iterations must be nonnegative")
        routes.setflags(write=False)
        residuals.setflags(write=False)
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "residuals", residuals)


@dataclass(frozen=True)
class NetworkReservationSolveResult:
    decision: Array
    raw_objective: float
    regularized_objective: float
    reservation_residual: float
    maximum_recourse_residual: float
    outer_iterations: int
    maximum_inner_iterations: int

    def __post_init__(self) -> None:
        decision = np.ascontiguousarray(
            np.asarray(self.decision, dtype=np.float64).reshape(-1)
        )
        scalars = (
            self.raw_objective,
            self.regularized_objective,
            self.reservation_residual,
            self.maximum_recourse_residual,
        )
        if np.any(~np.isfinite(decision)) or any(
            not math.isfinite(value) for value in scalars
        ):
            raise ValueError("formal NET result must be finite")
        if min(self.outer_iterations, self.maximum_inner_iterations) < 0:
            raise ValueError("formal NET iteration counts must be nonnegative")
        decision.setflags(write=False)
        object.__setattr__(self, "decision", decision)


@dataclass(frozen=True)
class NetworkReservationProblem:
    """Reserve directed-link capacity before point-to-point demand is known."""

    first_path_incidence: Array
    second_path_incidence: Array
    directed_links: tuple[Link, ...]
    commodities: tuple[Commodity, ...]
    reservation_costs: Array
    congestion_penalty: float = 10.0
    routing_penalty: float = 0.05
    reservation_upper_bound: float = 1.5
    context_dim: int = 8
    decision_iterations: int = 30
    routing_updates_per_iteration: int = 6
    realized_loss_routing_iterations: int = 80
    step_fraction: float = 0.9
    name: str = "multicommodity_network_reservation"
    _routing_difference: Array = field(init=False, repr=False, compare=False)
    _routing_lipschitz: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        first = np.asarray(self.first_path_incidence, dtype=np.float64)
        second = np.asarray(self.second_path_incidence, dtype=np.float64)
        costs = np.asarray(self.reservation_costs, dtype=np.float64).reshape(-1)
        if first.ndim != 2 or first.shape != second.shape:
            raise ValueError("path-incidence matrices must have the same shape")
        if first.shape[0] != _LINK_COUNT or first.shape[1] < 1:
            raise ValueError("path-incidence matrices have the wrong shape")
        if np.any((first != 0.0) & (first != 1.0)) or np.any(
            (second != 0.0) & (second != 1.0)
        ):
            raise ValueError("path-incidence matrices must be binary")
        if (
            costs.shape != (_LINK_COUNT,)
            or np.any(~np.isfinite(costs))
            or np.any(costs <= 0.0)
        ):
            raise ValueError("reservation costs must be positive and align with links")
        if len(self.directed_links) != _LINK_COUNT:
            raise ValueError("the directed grid must contain 48 links")
        if len(self.commodities) != first.shape[1]:
            raise ValueError("commodities must align with path columns")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.congestion_penalty,
                self.routing_penalty,
                self.reservation_upper_bound,
            )
        ):
            raise ValueError("NET penalty and bound values must be positive")
        if self.context_dim != 8:
            raise ValueError("the NET context dimension must equal eight")
        if min(
            self.decision_iterations,
            self.routing_updates_per_iteration,
            self.realized_loss_routing_iterations,
        ) < 1:
            raise ValueError("all fixed iteration counts must be positive")
        if not 0.0 < self.step_fraction <= 1.0:
            raise ValueError("step_fraction must lie in (0,1]")
        immutable = []
        for value in (first, second, costs):
            copy = np.ascontiguousarray(value.copy())
            copy.setflags(write=False)
            immutable.append(copy)
        object.__setattr__(self, "first_path_incidence", immutable[0])
        object.__setattr__(self, "second_path_incidence", immutable[1])
        object.__setattr__(self, "reservation_costs", immutable[2])
        difference = np.ascontiguousarray(immutable[0] - immutable[1])
        difference.setflags(write=False)
        object.__setattr__(self, "_routing_difference", difference)
        object.__setattr__(
            self,
            "_routing_lipschitz",
            float(
                2.0
                * self.congestion_penalty
                * np.linalg.norm(difference, ord=2) ** 2
                + 2.0 * self.routing_penalty
            ),
        )

    @property
    def scenario_dim(self) -> int:
        return int(self.first_path_incidence.shape[1])

    @property
    def decision_dim(self) -> int:
        return _LINK_COUNT

    @property
    def routing_difference(self) -> Array:
        return self._routing_difference

    @property
    def routing_lipschitz(self) -> float:
        return self._routing_lipschitz

    def reservation_lipschitz(self, ridge: float) -> float:
        """The gradient bound also holds for the unregularized business loss."""
        ridge_value = float(ridge)
        if not math.isfinite(ridge_value) or ridge_value < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        return 2.0 * self.congestion_penalty + ridge_value

    def transform_numpy(self, outcomes: Array) -> Array:
        values = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        if np.any(values <= 0.0):
            raise ValueError("NET demands must be strictly positive")
        return np.log(values)

    def decode_torch(self, transformed: Tensor) -> Tensor:
        if transformed.shape[-1] != self.scenario_dim:
            raise ValueError("transformed scenarios have the wrong dimension")
        return torch.exp(transformed)

    def reference_center_numpy(self, context: Array) -> Array:
        self._validate_context_numpy(context)
        return np.full(self.decision_dim, 0.1, dtype=np.float64)

    def reference_lower_bounds_numpy(self, context: Array) -> Array:
        self._validate_context_numpy(context)
        return np.zeros(self.decision_dim, dtype=np.float64)

    def reference_upper_bounds_numpy(self, context: Array) -> Array:
        self._validate_context_numpy(context)
        return np.full(
            self.decision_dim,
            self.reservation_upper_bound,
            dtype=np.float64,
        )

    def reference_decision_is_valid_numpy(self, decision: Array) -> bool:
        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        return bool(
            value.shape == (self.decision_dim,)
            and np.all(np.isfinite(value))
            and np.all(value >= 0.0)
            and np.all(value <= self.reservation_upper_bound)
        )

    def solve_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> Array:
        self._validate_context_numpy(context)
        demands = self._validated_demands_numpy(scenarios)
        weights = probability_vector(probabilities, len(demands))
        ridge_value = _positive_ridge(ridge)
        difference = self.routing_difference
        base_load = demands @ self.second_path_incidence.T
        routes = demands / 2.0
        decision = np.full(self.decision_dim, 0.1, dtype=np.float64)
        routing_step = self.step_fraction / self.routing_lipschitz
        reservation_step = self.step_fraction / self.reservation_lipschitz(
            ridge_value
        )
        for _ in range(self.decision_iterations):
            for _ in range(self.routing_updates_per_iteration):
                loads = base_load + routes @ difference.T
                excess = np.maximum(loads - decision[None, :], 0.0)
                route_gradient = (
                    2.0 * self.congestion_penalty * (excess @ difference)
                    + 2.0 * self.routing_penalty * (routes - demands / 2.0)
                )
                routes = np.minimum(
                    np.maximum(routes - routing_step * route_gradient, 0.0),
                    demands,
                )
            loads = base_load + routes @ difference.T
            excess = np.maximum(loads - decision[None, :], 0.0)
            gradient = self.reservation_costs.copy()
            gradient -= 2.0 * self.congestion_penalty * (weights @ excess)
            gradient += ridge_value * (decision - 0.1)
            decision = np.clip(
                decision - reservation_step * gradient,
                0.0,
                self.reservation_upper_bound,
            )
        if np.any(~np.isfinite(decision)):
            raise FloatingPointError("NET fixed decision map produced nonfinite values")
        return decision

    def solve_torch(
        self,
        contexts: Tensor,
        scenarios: Tensor,
        probabilities: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        self._validate_contexts_torch(contexts)
        if scenarios.ndim != 2 or scenarios.shape[1] != self.scenario_dim:
            raise ValueError("scenarios have the wrong shape")
        if probabilities.shape != (len(contexts), len(scenarios)):
            raise ValueError("probabilities must align with contexts and scenarios")
        if not torch.all(torch.isfinite(scenarios)) or not torch.all(scenarios > 0.0):
            raise ValueError("NET scenarios must be finite and positive")
        if not torch.all(torch.isfinite(probabilities)) or not torch.all(
            probabilities >= 0.0
        ):
            raise ValueError("probabilities must be finite and nonnegative")
        ridge_value = _positive_ridge(ridge)
        factory = {"dtype": scenarios.dtype, "device": scenarios.device}
        first = torch.tensor(self.first_path_incidence.copy(), **factory)
        second = torch.tensor(self.second_path_incidence.copy(), **factory)
        difference = first - second
        costs = torch.tensor(self.reservation_costs.copy(), **factory)
        demands = scenarios
        base_load = demands @ second.T
        routes = (demands / 2.0)[None, :, :].expand(
            len(contexts),
            -1,
            -1,
        )
        decision = torch.full((len(contexts), self.decision_dim), 0.1, **factory)
        routing_step = self.step_fraction / self.routing_lipschitz
        reservation_step = self.step_fraction / self.reservation_lipschitz(
            ridge_value
        )
        for _ in range(self.decision_iterations):
            for _ in range(self.routing_updates_per_iteration):
                loads = base_load[None, :, :] + torch.einsum(
                    "nkm,em->nke",
                    routes,
                    difference,
                )
                excess = torch.relu(loads - decision[:, None, :])
                route_gradient = (
                    2.0
                    * self.congestion_penalty
                    * torch.einsum("nke,em->nkm", excess, difference)
                    + 2.0
                    * self.routing_penalty
                    * (routes - demands[None, :, :] / 2.0)
                )
                routes = torch.minimum(
                    torch.clamp_min(routes - routing_step * route_gradient, 0.0),
                    demands[None, :, :],
                )
            loads = base_load[None, :, :] + torch.einsum(
                "nkm,em->nke",
                routes,
                difference,
            )
            excess = torch.relu(loads - decision[:, None, :])
            gradient = costs[None, :] - 2.0 * self.congestion_penalty * torch.einsum(
                "nk,nke->ne",
                probabilities,
                excess,
            )
            gradient = gradient + ridge_value * (decision - 0.1)
            decision = torch.clamp(
                decision - reservation_step * gradient,
                min=0.0,
                max=self.reservation_upper_bound,
            )
        if not torch.all(torch.isfinite(decision)):
            raise FloatingPointError("NET fixed decision map produced nonfinite values")
        return decision

    def realized_loss_torch(
        self,
        decisions: Tensor,
        outcomes: Tensor,
        contexts: Tensor,
    ) -> Tensor:
        self._validate_contexts_torch(contexts)
        if decisions.shape != (len(contexts), self.decision_dim):
            raise ValueError("decisions must align with contexts")
        if outcomes.shape != (len(contexts), self.scenario_dim):
            raise ValueError("outcomes must align with contexts")
        if not torch.all(torch.isfinite(outcomes)) or not torch.all(outcomes > 0.0):
            raise ValueError("NET outcomes must be finite and positive")
        factory = {"dtype": outcomes.dtype, "device": outcomes.device}
        second = torch.tensor(self.second_path_incidence.copy(), **factory)
        difference = torch.tensor(self.routing_difference.copy(), **factory)
        costs = torch.tensor(self.reservation_costs.copy(), **factory)
        routes = outcomes / 2.0
        base_load = outcomes @ second.T
        routing_step = self.step_fraction / self.routing_lipschitz
        for _ in range(self.realized_loss_routing_iterations):
            loads = base_load + routes @ difference.T
            excess = torch.relu(loads - decisions)
            gradient = (
                2.0 * self.congestion_penalty * (excess @ difference)
                + 2.0 * self.routing_penalty * (routes - outcomes / 2.0)
            )
            routes = torch.minimum(
                torch.clamp_min(routes - routing_step * gradient, 0.0),
                outcomes,
            )
        loads = base_load + routes @ difference.T
        excess = torch.relu(loads - decisions)
        return (
            decisions @ costs
            + self.congestion_penalty * excess.square().sum(dim=1)
            + self.routing_penalty * (routes - outcomes / 2.0).square().sum(dim=1)
        )

    def solve_recourse_numpy(
        self,
        decision: Array,
        scenarios: Array,
        *,
        tolerance: float = 1e-10,
        maximum_iterations: int = 10_000,
        initial_routes: Array | None = None,
    ) -> NetworkRecourseResult:
        value = self._validated_decision_numpy(decision)
        demands = self._validated_demands_numpy(scenarios)
        tolerance_value = _positive_finite(tolerance, "tolerance")
        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        difference = self.routing_difference
        base_load = demands @ self.second_path_incidence.T
        routes = (
            demands / 2.0
            if initial_routes is None
            else np.asarray(initial_routes, dtype=np.float64).copy()
        )
        if (
            routes.shape != demands.shape
            or np.any(~np.isfinite(routes))
            or np.any(routes < 0.0)
            or np.any(routes > demands)
        ):
            raise ValueError("initial_routes must be finite, feasible, and align with demands")
        extrapolated = routes.copy()
        step = self.step_fraction / self.routing_lipschitz
        strong_convexity = 2.0 * self.routing_penalty
        momentum = (
            1.0 - math.sqrt(strong_convexity * step)
        ) / (
            1.0 + math.sqrt(strong_convexity * step)
        )
        for iteration in range(maximum_iterations + 1):
            loads = base_load + routes @ difference.T
            excess = np.maximum(loads - value[None, :], 0.0)
            gradient = (
                2.0 * self.congestion_penalty * (excess @ difference)
                + 2.0 * self.routing_penalty * (routes - demands / 2.0)
            )
            projected = np.minimum(
                np.maximum(routes - gradient / self.routing_lipschitz, 0.0),
                demands,
            )
            residuals = self.routing_lipschitz * np.max(
                np.abs(routes - projected),
                axis=1,
            )
            if float(np.max(residuals, initial=0.0)) <= tolerance_value:
                return NetworkRecourseResult(routes, residuals, iteration)
            if iteration == maximum_iterations:
                break
            extrapolated_loads = base_load + extrapolated @ difference.T
            extrapolated_excess = np.maximum(
                extrapolated_loads - value[None, :],
                0.0,
            )
            extrapolated_gradient = (
                2.0
                * self.congestion_penalty
                * (extrapolated_excess @ difference)
                + 2.0
                * self.routing_penalty
                * (extrapolated - demands / 2.0)
            )
            following = np.minimum(
                np.maximum(
                    extrapolated - step * extrapolated_gradient,
                    0.0,
                ),
                demands,
            )
            extrapolated = following + momentum * (following - routes)
            routes = following
        raise RuntimeError("NET recourse failed its projected-gradient gate")

    def formal_scenario_losses_numpy(
        self,
        decision: Array,
        scenarios: Array,
        *,
        tolerance: float = 1e-10,
        maximum_iterations: int = 10_000,
    ) -> tuple[Array, NetworkRecourseResult]:
        value = self._validated_decision_numpy(decision)
        demands = self._validated_demands_numpy(scenarios)
        recourse = self.solve_recourse_numpy(
            value,
            demands,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        loads = (
            demands @ self.second_path_incidence.T
            + recourse.routes @ self.routing_difference.T
        )
        excess = np.maximum(loads - value[None, :], 0.0)
        losses = (
            self.reservation_costs @ value
            + self.congestion_penalty * np.sum(excess**2, axis=1)
            + self.routing_penalty
            * np.sum((recourse.routes - demands / 2.0) ** 2, axis=1)
        )
        return np.ascontiguousarray(losses), recourse

    def kkt_contributions_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
    ) -> Array:
        self._validate_context_numpy(context)
        value = self._validated_decision_numpy(decision)
        demands = self._validated_demands_numpy(outcomes)
        recourse = self.solve_recourse_numpy(value, demands)
        loads = (
            demands @ self.second_path_incidence.T
            + recourse.routes @ self.routing_difference.T
        )
        excess = np.maximum(loads - value[None, :], 0.0)
        return np.ascontiguousarray(-2.0 * self.congestion_penalty * excess)

    def sof_node_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
        *,
        ridge: float,
        active_set_tolerance: float,
    ) -> tuple[Array, Array]:
        """Return exact NET scenario gradients and its local Schur Hessian."""

        self._validate_context_numpy(context)
        value = self._validated_decision_numpy(decision)
        demands = self._validated_demands_numpy(outcomes)
        ridge_value = _positive_ridge(ridge)
        tolerance = _positive_finite(
            active_set_tolerance,
            "active_set_tolerance",
        )
        recourse = self.solve_recourse_numpy(
            value,
            demands,
            tolerance=tolerance,
        )
        difference = self.routing_difference
        loads = (
            demands @ self.second_path_incidence.T
            + recourse.routes @ difference.T
        )
        overload = loads - value[None, :]
        gradients = np.ascontiguousarray(
            -2.0 * self.congestion_penalty * np.maximum(overload, 0.0)
        )
        hessian = np.zeros((self.decision_dim, self.decision_dim))
        for row in range(len(demands)):
            active_overload = overload[row] > tolerance
            active_route = (recourse.routes[row] > tolerance) & (
                demands[row] - recourse.routes[row] > tolerance
            )
            scenario_hessian = 2.0 * self.congestion_penalty * np.diag(
                active_overload.astype(np.float64)
            )
            if np.any(active_route) and np.any(active_overload):
                free_difference = difference[:, active_route]
                weighted_difference = (
                    active_overload[:, None] * free_difference
                )
                route_system = (
                    self.congestion_penalty
                    * free_difference.T
                    @ weighted_difference
                    + self.routing_penalty * np.eye(int(np.sum(active_route)))
                )
                scenario_hessian -= (
                    2.0
                    * self.congestion_penalty**2
                    * weighted_difference
                    @ np.linalg.solve(route_system, weighted_difference.T)
                )
            hessian += 0.5 * (scenario_hessian + scenario_hessian.T)
        hessian /= len(demands)
        hessian += ridge_value * np.eye(self.decision_dim)
        if np.any(~np.isfinite(hessian)):
            raise FloatingPointError("NET SOF parent Hessian is nonfinite")
        return gradients, np.ascontiguousarray(hessian)

    def solve_formal_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
        tolerance: float = 1e-10,
        maximum_iterations: int = 10_000,
    ) -> NetworkReservationSolveResult:
        self._validate_context_numpy(context)
        demands = self._validated_demands_numpy(scenarios)
        weights = probability_vector(probabilities, len(demands))
        ridge_value = _positive_ridge(ridge)
        tolerance_value = _positive_finite(tolerance, "tolerance")
        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        lipschitz = self.reservation_lipschitz(ridge_value)
        decision, initial_routes, outer_iterations = solve_gurobi_network_qp(
            self, demands, weights, ridge=ridge_value)

        maximum_inner = 0
        for refinement in range(51):
            recourse = self.solve_recourse_numpy(
                decision,
                demands,
                tolerance=tolerance_value,
                maximum_iterations=maximum_iterations,
                initial_routes=initial_routes,
            )
            maximum_inner = max(maximum_inner, recourse.iterations)
            loads = (
                demands @ self.second_path_incidence.T
                + recourse.routes @ self.routing_difference.T
            )
            excess = np.maximum(loads - decision[None, :], 0.0)
            gradient = self.reservation_costs.copy()
            gradient -= 2.0 * self.congestion_penalty * (weights @ excess)
            gradient += ridge_value * (decision - 0.1)
            projected = np.clip(
                decision - gradient / lipschitz,
                0.0,
                self.reservation_upper_bound,
            )
            residual = lipschitz * float(np.max(np.abs(decision - projected)))
            if residual <= tolerance_value:
                break
            if refinement == 50:
                raise RuntimeError(
                    f"NET joint QP reservation residual {residual:.6e} exceeds "
                    f"{tolerance_value:.6e} after refinement"
                )
            decision = projected
            initial_routes = recourse.routes
        scenario_losses = (
            self.reservation_costs @ decision
            + self.congestion_penalty * np.sum(excess**2, axis=1)
            + self.routing_penalty
            * np.sum((recourse.routes - demands / 2.0) ** 2, axis=1)
        )
        raw = float(weights @ scenario_losses)
        regularization = 0.5 * ridge_value * float(
            (decision - 0.1) @ (decision - 0.1)
        )
        return NetworkReservationSolveResult(
            decision=decision,
            raw_objective=raw,
            regularized_objective=raw + regularization,
            reservation_residual=residual,
            maximum_recourse_residual=float(
                np.max(recourse.residuals, initial=0.0)
            ),
            outer_iterations=outer_iterations + refinement,
            maximum_inner_iterations=maximum_inner,
        )

    def _validated_demands_numpy(self, scenarios: Array) -> Array:
        values = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        if np.any(values <= 0.0):
            raise ValueError("NET demand scenarios must be strictly positive")
        return values

    def _validated_decision_numpy(self, decision: Array) -> Array:
        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        if not self.reference_decision_is_valid_numpy(value):
            raise ValueError("NET decision is outside the reservation box")
        return value

    def _validate_context_numpy(self, context: Array) -> None:
        value = np.asarray(context, dtype=np.float64).reshape(-1)
        if value.shape != (self.context_dim,) or np.any(~np.isfinite(value)):
            raise ValueError(f"context must have shape ({self.context_dim},)")

    def _validate_contexts_torch(self, contexts: Tensor) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape (n,{self.context_dim})")
        if not torch.all(torch.isfinite(contexts)):
            raise ValueError("contexts must be finite")


def solve_gurobi_network_qp(problem, demands, weights, *, ridge, fixed_decisions=None, method=-1, threads=None):
    """Shared convex QP formulation for reservation and fixed-decision routing (Gurobi primary, Clarabel fallback)."""
    try:
        import gurobipy as gp
    except ImportError:
        gp = None

    demands = problem._validated_demands_numpy(demands)
    weights = probability_vector(weights, len(demands))
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Invalid decision regularization")
    if method not in (-1,0,1,2):
        raise ValueError("Invalid continuous QP method")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("threads must be a positive integer")
    if fixed_decisions is None:
        unique, inverse = np.unique(demands, axis=0, return_inverse=True)
        if len(unique) < len(demands):
            merged = np.bincount(inverse, weights=weights, minlength=len(unique))
            decision, routes, iterations = solve_gurobi_network_qp(
                problem, unique, merged, ridge=ridge, method=method, threads=threads)
            return decision, routes[inverse], iterations
    fixed = None
    if fixed_decisions is not None:
        fixed = np.asarray(fixed_decisions, dtype=np.float64)
        if fixed.shape != (len(demands), problem.decision_dim) or np.any(~np.isfinite(fixed)):
            raise ValueError("Fixed recourse decisions must align with scenarios")
        if np.any(fixed < -1e-9) or np.any(fixed > problem.reservation_upper_bound + 1e-9):
            raise ValueError("Fixed recourse decisions are infeasible")
        # Independent routing minimizers do not depend on positive row weights.
        # Unit weights avoid degrading per-row accuracy as the batch grows.
        weights = np.ones(len(demands))
    if gp is None:
        import clarabel
        from scipy import sparse

        count, width = demands.shape
        dimension = problem.decision_dim
        route_size = count * width
        overload_size = count * dimension
        upper = np.r_[demands.ravel(), np.full(overload_size, np.inf)]
        diagonal = np.r_[
            np.repeat(weights * problem.routing_penalty, width),
            np.repeat(weights * problem.congestion_penalty, dimension),
        ]
        linear = np.r_[
            (-problem.routing_penalty * weights[:, None] * demands).ravel(),
            np.zeros(overload_size),
        ]
        blocks = [
            sparse.kron(sparse.eye(count), problem.routing_difference, format="csr"),
            -sparse.eye(overload_size, format="csr"),
        ]
        rhs = -(demands @ problem.second_path_incidence.T).ravel()
        if fixed is None:
            upper = np.r_[upper, np.full(dimension, problem.reservation_upper_bound)]
            diagonal = np.r_[diagonal, np.full(dimension, 0.5 * ridge)]
            linear = np.r_[linear, problem.reservation_costs - 0.1 * ridge]
            blocks.append(-sparse.kron(np.ones((count, 1)), sparse.eye(dimension), format="csr"))
        else:
            rhs = rhs + fixed.ravel()
        n_var = len(upper)
        p_mat = sparse.diags(2.0 * diagonal, format="csc")
        a_ineq = sparse.hstack(blocks, format="csr")
        finite_ub = np.flatnonzero(np.isfinite(upper))
        a_lb = -sparse.eye(n_var, format="csr")
        b_lb = np.zeros(n_var, dtype=np.float64)
        a_ub = sparse.csr_matrix(
            (np.ones(len(finite_ub)), (np.arange(len(finite_ub)), finite_ub)),
            shape=(len(finite_ub), n_var),
        )
        b_ub = upper[finite_ub]
        a_all = sparse.vstack([a_ineq, a_lb, a_ub], format="csc")
        b_all = np.r_[rhs, b_lb, b_ub]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.tol_gap_abs = 1e-11
        settings.tol_gap_rel = 1e-11
        settings.tol_feas = 1e-11
        solver_cb = clarabel.DefaultSolver(
            p_mat, linear, a_all, b_all, [clarabel.NonnegativeConeT(len(b_all))], settings
        )
        sol = solver_cb.solve()
        x_sol = np.asarray(sol.x, dtype=np.float64)
        decision = fixed.copy() if fixed is not None else np.clip(x_sol[-dimension:], 0.0, problem.reservation_upper_bound)
        routes = np.clip(x_sol[:route_size].reshape(demands.shape), 0.0, demands)
        return decision, routes, int(sol.iterations)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("network_reservation", env=env) as model:
            model.Params.Threads = _assigned_core_count() if threads is None else threads
            model.Params.Method = method
            model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            model.Params.BarConvTol = 1e-12
            model.Params.NumericFocus = 3
            from scipy import sparse
            count, width = demands.shape
            dimension = problem.decision_dim
            route_size = count * width
            overload_size = count * dimension
            upper = np.r_[demands.ravel(), np.full(overload_size, gp.GRB.INFINITY)]
            diagonal = np.r_[np.repeat(weights*problem.routing_penalty, width),
                             np.repeat(weights*problem.congestion_penalty, dimension)]
            linear = np.r_[(-problem.routing_penalty*weights[:, None]*demands).ravel(),
                           np.zeros(overload_size)]
            blocks = [sparse.kron(sparse.eye(count), problem.routing_difference, format="csr"),
                      -sparse.eye(overload_size, format="csr")]
            rhs = -(demands @ problem.second_path_incidence.T).ravel()
            constant = .25*problem.routing_penalty*float(weights @ np.sum(demands**2, axis=1))
            if fixed is None:
                upper = np.r_[upper, np.full(dimension, problem.reservation_upper_bound)]
                diagonal = np.r_[diagonal, np.full(dimension, .5*ridge)]
                linear = np.r_[linear, problem.reservation_costs-.1*ridge]
                blocks.append(-sparse.kron(np.ones((count, 1)), sparse.eye(dimension), format="csr"))
                constant += .005*ridge*dimension
            else:
                rhs += fixed.ravel()
            variables = model.addMVar(len(upper), lb=0., ub=upper, name="state")
            model.addMConstr(sparse.hstack(blocks, format="csr"), variables, "<", rhs)
            model.setMObjective(sparse.diags(diagonal), linear, constant,
                                variables, variables, variables, gp.GRB.MINIMIZE)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"NET Gurobi QP did not solve to optimality: {model.Status}")
            decision = fixed.copy() if fixed is not None else np.asarray(variables.X[-dimension:])
            return decision, np.clip(np.asarray(variables.X[:route_size]).reshape(demands.shape), 0., demands), int(model.BarIterCount + model.IterCount)


def make_network_reservation_problem(
    scenario_dim: int,
    *,
    seed: int = 0,
    reservation_cost: float = 1.0,
    congestion_penalty: float = 10.0,
    routing_penalty: float = 0.05,
    reservation_upper_bound: float = 1.5,
    context_dim: int = 8,
    decision_iterations: int = 30,
    routing_updates_per_iteration: int = 6,
    realized_loss_routing_iterations: int = 80,
    step_fraction: float = 0.9,
) -> NetworkReservationProblem:
    """Construct the reproducible directed-grid NET problem for one prefix."""

    if not 1 <= scenario_dim <= _MASTER_COMMODITY_COUNT:
        raise ValueError("scenario_dim must lie between one and 128")
    reservation_cost_value = _positive_finite(reservation_cost, "reservation_cost")
    links = _directed_grid_links()
    link_index = {link: index for index, link in enumerate(links)}
    eligible = tuple(
        (origin, destination)
        for origin in _grid_nodes()
        for destination in _grid_nodes()
        if origin[0] != destination[0] and origin[1] != destination[1]
    )
    if len(eligible) != 144:
        raise RuntimeError("the 4x4 grid must have 144 eligible OD pairs")
    order = _rng(seed, "commodity-permutation").permutation(len(eligible))
    commodities = tuple(eligible[int(index)] for index in order[:scenario_dim])
    first = np.zeros((_LINK_COUNT, scenario_dim), dtype=np.float64)
    second = np.zeros_like(first)
    for commodity, (origin, destination) in enumerate(commodities):
        for link in _manhattan_path(origin, destination, horizontal_first=True):
            first[link_index[link], commodity] = 1.0
        for link in _manhattan_path(origin, destination, horizontal_first=False):
            second[link_index[link], commodity] = 1.0
    if np.any(np.all(first == second, axis=0)):
        raise RuntimeError("each commodity must have two distinct paths")
    return NetworkReservationProblem(
        first_path_incidence=first,
        second_path_incidence=second,
        directed_links=links,
        commodities=commodities,
        reservation_costs=np.full(
            _LINK_COUNT,
            reservation_cost_value,
            dtype=np.float64,
        ),
        congestion_penalty=congestion_penalty,
        routing_penalty=routing_penalty,
        reservation_upper_bound=reservation_upper_bound,
        context_dim=context_dim,
        decision_iterations=decision_iterations,
        routing_updates_per_iteration=routing_updates_per_iteration,
        realized_loss_routing_iterations=realized_loss_routing_iterations,
        step_fraction=step_fraction,
    )


def _grid_nodes() -> tuple[Node, ...]:
    return tuple(
        (row, column)
        for row in range(_GRID_SIZE)
        for column in range(_GRID_SIZE)
    )


def _directed_grid_links() -> tuple[Link, ...]:
    links: list[Link] = []
    for row, column in _grid_nodes():
        if row + 1 < _GRID_SIZE:
            links.extend(
                (
                    ((row, column), (row + 1, column)),
                    ((row + 1, column), (row, column)),
                )
            )
        if column + 1 < _GRID_SIZE:
            links.extend(
                (
                    ((row, column), (row, column + 1)),
                    ((row, column + 1), (row, column)),
                )
            )
    return tuple(sorted(links))


def _manhattan_path(
    origin: Node,
    destination: Node,
    *,
    horizontal_first: bool,
) -> tuple[Link, ...]:
    current = origin
    path: list[Link] = []

    def move_horizontal() -> None:
        nonlocal current
        step = 1 if destination[1] > current[1] else -1
        while current[1] != destination[1]:
            following = (current[0], current[1] + step)
            path.append((current, following))
            current = following

    def move_vertical() -> None:
        nonlocal current
        step = 1 if destination[0] > current[0] else -1
        while current[0] != destination[0]:
            following = (current[0] + step, current[1])
            path.append((current, following))
            current = following

    if horizontal_first:
        move_horizontal()
        move_vertical()
    else:
        move_vertical()
        move_horizontal()
    return tuple(path)


def _positive_ridge(ridge: float) -> float:
    return _positive_finite(ridge, "ridge")


def _assigned_core_count() -> int:
    values = os.environ.get("KKT_ATLAS_CPU_SET_IDS", "").split(",")
    return max(1, len([value for value in values if value.strip()]))


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _rng(seed: int, token: str) -> np.random.Generator:
    payload = f"{_PROBLEM_VERSION}|{int(seed)}|{token}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)
