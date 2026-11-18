"""Smooth capacity allocation with cross-coupled demand classes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from kkt_atlas.problem import Array, Tensor, matrix, probability_vector


@dataclass(frozen=True)
class SmoothCapacityAllocationProblem:
    """Allocate a fixed resource budget under nonlinear demand penalties."""

    service_matrix: Array
    underage_costs: Array
    overage_costs: Array
    total_capacity: float
    scenario_lower_bounds: Array | None = None
    scenario_upper_bounds: Array | None = None
    smoothing: float = 0.35
    context_dim: int = 8
    newton_iterations: int = 30
    newton_damping: float = 0.8
    name: str = "smooth_capacity_allocation"

    def __post_init__(self) -> None:
        service = np.asarray(self.service_matrix, dtype=np.float64)
        underage = np.asarray(self.underage_costs, dtype=np.float64).reshape(-1)
        overage = np.asarray(self.overage_costs, dtype=np.float64).reshape(-1)
        if service.ndim != 2 or min(service.shape) < 2:
            raise ValueError("service_matrix must have shape (m,d) with m,d at least two")
        if underage.shape != (service.shape[0],) or overage.shape != underage.shape:
            raise ValueError("cost vectors must align with the demand classes")
        if np.any(~np.isfinite(service)) or np.any(service < 0.0):
            raise ValueError("service_matrix must be finite and nonnegative")
        if np.any(~np.isfinite(underage)) or np.any(underage <= 0.0):
            raise ValueError("underage costs must be finite and positive")
        if np.any(~np.isfinite(overage)) or np.any(overage <= 0.0):
            raise ValueError("overage costs must be finite and positive")
        if (self.scenario_lower_bounds is None) != (self.scenario_upper_bounds is None):
            raise ValueError("scenario bounds must be supplied together")
        lower = upper = None
        if self.scenario_lower_bounds is not None:
            lower = np.asarray(self.scenario_lower_bounds, dtype=np.float64).reshape(-1)
            upper = np.asarray(self.scenario_upper_bounds, dtype=np.float64).reshape(-1)
            if (
                lower.shape != underage.shape
                or upper.shape != underage.shape
                or np.any(~np.isfinite(lower))
                or np.any(~np.isfinite(upper))
                or np.any(lower <= 0.0)
                or np.any(lower >= upper)
            ):
                raise ValueError("scenario bounds must be finite, positive, and ordered")
        if not np.allclose(service.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("each service_matrix row must sum to one")
        if not np.isfinite(self.total_capacity) or self.total_capacity <= 0.0:
            raise ValueError("total_capacity must be positive and finite")
        if not np.isfinite(self.smoothing) or self.smoothing <= 0.0:
            raise ValueError("smoothing must be positive and finite")
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive")
        if self.newton_iterations < 1:
            raise ValueError("newton_iterations must be positive")
        if not 0.0 < self.newton_damping <= 1.0:
            raise ValueError("newton_damping must lie in (0,1]")
        immutable: list[Array] = []
        for value in (service, underage, overage):
            copy = np.ascontiguousarray(value.copy())
            copy.setflags(write=False)
            immutable.append(copy)
        object.__setattr__(self, "service_matrix", immutable[0])
        object.__setattr__(self, "underage_costs", immutable[1])
        object.__setattr__(self, "overage_costs", immutable[2])
        if lower is not None and upper is not None:
            lower = np.ascontiguousarray(lower.copy())
            upper = np.ascontiguousarray(upper.copy())
            lower.setflags(write=False)
            upper.setflags(write=False)
            object.__setattr__(self, "scenario_lower_bounds", lower)
            object.__setattr__(self, "scenario_upper_bounds", upper)

    @property
    def scenario_dim(self) -> int:
        return int(self.service_matrix.shape[0])

    @property
    def decision_dim(self) -> int:
        return int(self.service_matrix.shape[1])

    @property
    def base_capacity(self) -> float:
        return float(self.total_capacity / self.decision_dim)

    def transform_numpy(self, outcomes: Array) -> Array:
        values = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        if np.any(values <= 0.0):
            raise ValueError("demand realizations must be strictly positive")
        return np.log(values)

    def decode_torch(self, transformed: Tensor) -> Tensor:
        if transformed.shape[-1] != self.scenario_dim:
            raise ValueError("transformed scenarios have the wrong final dimension")
        decoded = torch.exp(transformed)
        if self.scenario_lower_bounds is None:
            return decoded
        lower = torch.tensor(
            self.scenario_lower_bounds.copy(),
            dtype=decoded.dtype,
            device=decoded.device,
        )
        upper = torch.tensor(
            self.scenario_upper_bounds.copy(),
            dtype=decoded.dtype,
            device=decoded.device,
        )
        return torch.minimum(torch.maximum(decoded, lower), upper)

    def reference_center_numpy(self, context: Array) -> Array:
        """Return the allocation used as the decision-ridge center."""

        self._validate_context_numpy(context)
        return np.full(
            self.decision_dim,
            self.base_capacity,
            dtype=np.float64,
        )

    def reference_equality_numpy(self, context: Array) -> tuple[Array, Array]:
        """Return the resource-budget equality used by the evaluator."""

        self._validate_context_numpy(context)
        return (
            np.ones((1, self.decision_dim), dtype=np.float64),
            np.asarray([self.total_capacity], dtype=np.float64),
        )

    def reference_lower_bounds_numpy(self, context: Array) -> Array:
        """Return the nonnegativity bounds on resource allocations."""

        self._validate_context_numpy(context)
        return np.zeros(self.decision_dim, dtype=np.float64)

    def reference_decision_is_valid_numpy(self, decision: Array) -> bool:
        """Require a finite, nonnegative resource allocation."""

        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        return bool(value.shape == (self.decision_dim,) and np.all(np.isfinite(value)) and np.all(value >= 0.0))

    def reference_objective_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> tuple[float, float, Array, Array]:
        """Evaluate the objective and exact derivatives for reference solves."""

        self._validate_context_numpy(context)
        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        if value.shape != (self.decision_dim,) or np.any(~np.isfinite(value)):
            raise ValueError("decision has the wrong shape")
        demands = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        if np.any(demands <= 0.0):
            raise ValueError("demand scenarios must be strictly positive")
        weights = probability_vector(probabilities, len(demands))
        ridge_value = _nonnegative_ridge(ridge)

        supplied = self.service_matrix @ value
        difference = supplied[None, :] - demands
        excess = self.smoothing * np.logaddexp(
            0.0,
            difference / self.smoothing,
        )
        shortage = self.smoothing * np.logaddexp(
            0.0,
            -difference / self.smoothing,
        )
        scenario_values = (self.overage_costs[None, :] * excess + self.underage_costs[None, :] * shortage).mean(axis=1)
        raw_value = float(weights @ scenario_values)
        gradient, hessian = self._expected_derivatives_numpy(
            value,
            demands,
            weights,
        )
        center: Array = np.full(
            self.decision_dim,
            self.base_capacity,
            dtype=np.float64,
        )
        displacement = value - center
        regularization = 0.5 * ridge_value * float(displacement @ displacement)
        gradient = gradient + ridge_value * displacement
        hessian = hessian + ridge_value * np.eye(self.decision_dim)
        return raw_value, raw_value + regularization, gradient, hessian

    def solve_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> Array:
        self._validate_context_numpy(context)
        demands = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        if np.any(demands <= 0.0):
            raise ValueError("demand scenarios must be strictly positive")
        weights = probability_vector(probabilities, len(demands))
        ridge_value = _nonnegative_ridge(ridge)

        decision = np.full(self.decision_dim, self.base_capacity, dtype=np.float64)
        multiplier = 0.0
        ones = np.ones(self.decision_dim, dtype=np.float64)
        for _ in range(self.newton_iterations):
            gradient, hessian = self._expected_derivatives_numpy(
                decision,
                demands,
                weights,
            )
            stationarity = gradient + ridge_value * (decision - self.base_capacity) + multiplier
            hessian = hessian + ridge_value * np.eye(self.decision_dim)
            inverse_stationarity = np.linalg.solve(hessian, stationarity)
            inverse_ones = np.linalg.solve(hessian, ones)
            constraint_residual = float(decision.sum() - self.total_capacity)
            multiplier_step = (constraint_residual - float(ones @ inverse_stationarity)) / float(ones @ inverse_ones)
            decision_step = -inverse_stationarity - inverse_ones * multiplier_step
            damping = self.newton_damping
            decreasing = decision_step < 0.0
            if np.any(decreasing):
                damping = min(
                    damping,
                    float(np.min(-0.99 * decision[decreasing] / decision_step[decreasing])),
                )
            decision += damping * decision_step
            multiplier += damping * multiplier_step
        decision = np.maximum(
            decision,
            np.finfo(np.float64).eps * self.base_capacity,
        )
        decision *= self.total_capacity / decision.sum()
        self._validate_solution_numpy(decision)
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
        if probabilities.ndim != 2 or probabilities.shape != (
            len(contexts),
            len(scenarios),
        ):
            raise ValueError("probabilities must align with contexts and scenarios")
        if not torch.all(torch.isfinite(scenarios)) or not torch.all(scenarios > 0.0):
            raise ValueError("demand scenarios must be finite and positive")
        if not torch.all(torch.isfinite(probabilities)) or not torch.all(probabilities >= 0.0):
            raise ValueError("probabilities must be finite and nonnegative")
        ridge_value = _nonnegative_ridge(ridge)

        factory = {"dtype": scenarios.dtype, "device": scenarios.device}
        service = torch.tensor(self.service_matrix.copy(), **factory)
        underage = torch.tensor(self.underage_costs.copy(), **factory)
        overage = torch.tensor(self.overage_costs.copy(), **factory)
        decision = torch.full(
            (len(contexts), self.decision_dim),
            self.base_capacity,
            **factory,
        )
        multiplier = torch.zeros((len(contexts), 1), **factory)
        ones = torch.ones((len(contexts), self.decision_dim, 1), **factory)
        identity = torch.eye(self.decision_dim, **factory)[None, :, :]
        for _ in range(self.newton_iterations):
            supplied = decision @ service.T
            difference = supplied[:, None, :] - scenarios[None, :, :]
            sigmoid = torch.sigmoid(difference / self.smoothing)
            scenario_gradients = (
                (underage + overage)[None, None, :] * sigmoid - underage[None, None, :]
            ) / self.scenario_dim
            supplied_gradient = (probabilities[:, :, None] * scenario_gradients).sum(dim=1)
            gradient = supplied_gradient @ service
            scenario_curvature = (
                (underage + overage)[None, None, :] * sigmoid * (1.0 - sigmoid) / (self.smoothing * self.scenario_dim)
            )
            supplied_curvature = (probabilities[:, :, None] * scenario_curvature).sum(dim=1)
            hessian = (
                torch.einsum(
                    "ni,ir,is->nrs",
                    supplied_curvature,
                    service,
                    service,
                )
                + ridge_value * identity
            )
            stationarity = gradient + ridge_value * (decision - self.base_capacity) + multiplier
            inverse_columns = torch.linalg.solve(
                hessian,
                torch.cat((stationarity[:, :, None], ones), dim=2),
            )
            inverse_stationarity = inverse_columns[:, :, 0]
            inverse_ones = inverse_columns[:, :, 1]
            constraint_residual = decision.sum(dim=1, keepdim=True) - self.total_capacity
            multiplier_step = (constraint_residual - inverse_stationarity.sum(dim=1, keepdim=True)) / inverse_ones.sum(
                dim=1, keepdim=True
            )
            decision_step = -inverse_stationarity - inverse_ones * multiplier_step
            decreasing = decision_step < 0.0
            denominator = torch.where(
                decreasing,
                decision_step,
                -torch.ones_like(decision_step),
            )
            feasible_damping = torch.where(
                decreasing,
                -0.99 * decision / denominator,
                torch.full_like(decision_step, torch.inf),
            ).amin(dim=1, keepdim=True)
            damping = torch.minimum(
                feasible_damping,
                torch.full_like(feasible_damping, self.newton_damping),
            )
            decision = decision + damping * decision_step
            multiplier = multiplier + damping * multiplier_step
        decision = torch.clamp_min(
            decision,
            torch.finfo(decision.dtype).eps * self.base_capacity,
        )
        decision = decision * (
            self.total_capacity / decision.sum(dim=1, keepdim=True)
        )
        if not torch.all(torch.isfinite(decision)):
            raise FloatingPointError("capacity Newton solve produced nonfinite decisions")
        if not torch.all(decision > 0.0):
            raise RuntimeError("capacity solution left the positive operating region")
        return decision

    def kkt_contributions_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
    ) -> Array:
        self._validate_context_numpy(context)
        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        if value.shape != (self.decision_dim,) or np.any(~np.isfinite(value)):
            raise ValueError("decision has the wrong shape")
        demands = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        supplied = self.service_matrix @ value
        sigmoid = _sigmoid_numpy((supplied[None, :] - demands) / self.smoothing)
        supplied_gradients = (
            (self.underage_costs + self.overage_costs)[None, :] * sigmoid - self.underage_costs[None, :]
        ) / self.scenario_dim
        return supplied_gradients @ self.service_matrix

    def sof_node_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
        *,
        ridge: float,
        active_set_tolerance: float,
    ) -> tuple[Array, Array]:
        """Return SOF observation gradients and the regularized parent Hessian."""

        tolerance = float(active_set_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("active_set_tolerance must be finite and nonnegative")
        demands = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        gradients = self.kkt_contributions_numpy(decision, context, demands)
        probabilities = np.full(len(demands), 1.0 / len(demands))
        _, _, _, hessian = self.reference_objective_derivatives_numpy(
            decision,
            context,
            demands,
            probabilities,
            ridge=ridge,
        )
        return gradients, np.ascontiguousarray(hessian)

    def realized_loss_torch(
        self,
        decisions: Tensor,
        outcomes: Tensor,
        contexts: Tensor,
    ) -> Tensor:
        self._validate_contexts_torch(contexts)
        if decisions.ndim != 2 or decisions.shape != (
            len(contexts),
            self.decision_dim,
        ):
            raise ValueError("decisions must align with contexts")
        if outcomes.ndim != 2 or outcomes.shape != (
            len(contexts),
            self.scenario_dim,
        ):
            raise ValueError("outcomes must align with contexts")
        service = torch.tensor(
            self.service_matrix.copy(),
            dtype=decisions.dtype,
            device=decisions.device,
        )
        underage = torch.tensor(
            self.underage_costs.copy(),
            dtype=decisions.dtype,
            device=decisions.device,
        )
        overage = torch.tensor(
            self.overage_costs.copy(),
            dtype=decisions.dtype,
            device=decisions.device,
        )
        difference = decisions @ service.T - outcomes
        excess = self.smoothing * functional.softplus(difference / self.smoothing)
        shortage = self.smoothing * functional.softplus(-difference / self.smoothing)
        return (overage[None, :] * excess + underage[None, :] * shortage).mean(dim=1)

    def _expected_derivatives_numpy(
        self,
        decision: Array,
        scenarios: Array,
        probabilities: Array,
    ) -> tuple[Array, Array]:
        supplied = self.service_matrix @ decision
        sigmoid = _sigmoid_numpy((supplied[None, :] - scenarios) / self.smoothing)
        gradients = (
            (self.underage_costs + self.overage_costs)[None, :] * sigmoid - self.underage_costs[None, :]
        ) / self.scenario_dim
        supplied_gradient = probabilities @ gradients
        curvature = (
            (self.underage_costs + self.overage_costs)[None, :]
            * sigmoid
            * (1.0 - sigmoid)
            / (self.smoothing * self.scenario_dim)
        )
        supplied_curvature = probabilities @ curvature
        hessian = self.service_matrix.T @ (supplied_curvature[:, None] * self.service_matrix)
        return supplied_gradient @ self.service_matrix, hessian

    def _validate_context_numpy(self, context: Array) -> None:
        value = np.asarray(context, dtype=np.float64).reshape(-1)
        if value.shape != (self.context_dim,) or np.any(~np.isfinite(value)):
            raise ValueError(f"context must have shape ({self.context_dim},)")

    def _validate_contexts_torch(self, contexts: Tensor) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape (n,{self.context_dim})")
        if not torch.all(torch.isfinite(contexts)):
            raise ValueError("contexts must be finite")

    def _validate_solution_numpy(self, decision: Array) -> None:
        if np.any(~np.isfinite(decision)):
            raise FloatingPointError("capacity Newton solve produced a nonfinite decision")
        if np.any(decision <= 0.0):
            raise RuntimeError("capacity solution left the positive operating region")


def make_smooth_capacity_allocation_problem(
    scenario_dim: int,
    *,
    resource_count: int = 8,
    base_capacity: float = 5.0,
    smoothing: float = 0.35,
    context_dim: int = 8,
    newton_iterations: int = 30,
    newton_damping: float = 0.8,
    seed: int = 0,
) -> SmoothCapacityAllocationProblem:
    """Construct one reproducible cross-coupled capacity-allocation problem."""

    if scenario_dim < 2 or resource_count < 2:
        raise ValueError("scenario_dim and resource_count must be at least two")
    if not np.isfinite(base_capacity) or base_capacity <= 0.0:
        raise ValueError("base_capacity must be positive and finite")
    generator = _rng(seed, "service-matrix")
    service = generator.gamma(
        shape=0.65,
        scale=1.0,
        size=(scenario_dim, resource_count),
    )
    service /= service.sum(axis=1, keepdims=True)
    indices = np.arange(scenario_dim, dtype=np.float64)
    underage = 3.0 + 0.4 * (1.0 + np.sin(0.31 * indices))
    overage = 0.7 + 0.15 * (1.0 + np.cos(0.23 * indices))
    return SmoothCapacityAllocationProblem(
        service_matrix=service,
        underage_costs=underage,
        overage_costs=overage,
        total_capacity=float(base_capacity) * resource_count,
        smoothing=smoothing,
        context_dim=context_dim,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
    )


def _nonnegative_ridge(ridge: float) -> float:
    value = float(ridge)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    return value


def _sigmoid_numpy(values: Array) -> Array:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponential = np.exp(array[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _rng(seed: int, token: str) -> np.random.Generator:
    payload = f"smooth-capacity-problem-v1|{int(seed)}|{token}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)
