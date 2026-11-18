"""Quadratic stochastic program driven by a low-dimensional scenario projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kkt_atlas.problem import Array, Tensor, matrix, probability_vector


@dataclass(frozen=True)
class ProjectedQuadraticProblem:
    """Track the task-relevant projection of a high-dimensional outcome."""

    task_basis: Array
    context_dim: int = 2
    name: str = "controlled_projected_quadratic"

    def __post_init__(self) -> None:
        basis = np.asarray(self.task_basis, dtype=np.float64)
        if (
            basis.ndim != 2
            or basis.shape[0] < basis.shape[1]
            or basis.shape[1] < 1
            or np.any(~np.isfinite(basis))
        ):
            raise ValueError(
                "task_basis must be a finite tall matrix with at least one column"
            )
        if not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("task_basis must have orthonormal columns")
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive")
        immutable = np.ascontiguousarray(basis.copy())
        immutable.setflags(write=False)
        object.__setattr__(self, "task_basis", immutable)

    @property
    def scenario_dim(self) -> int:
        return int(self.task_basis.shape[0])

    @property
    def decision_dim(self) -> int:
        return int(self.task_basis.shape[1])

    def transform_numpy(self, outcomes: Array) -> Array:
        """Use the ambient outcome itself as the scenario coordinate."""

        return matrix(outcomes, columns=self.scenario_dim, name="outcomes")

    def decode_torch(self, transformed: Tensor) -> Tensor:
        """Decode the unconstrained coordinates through the identity map."""

        if transformed.ndim != 2 or transformed.shape[1] != self.scenario_dim:
            raise ValueError("transformed scenarios have the wrong shape")
        if not torch.all(torch.isfinite(transformed)):
            raise ValueError("transformed scenarios must be finite")
        return transformed

    def solve_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> Array:
        self._validate_context_numpy(context)
        values = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        weights = probability_vector(probabilities, len(values))
        ridge_value = _nonnegative_ridge(ridge)
        task_mean = weights @ values @ self.task_basis
        return np.asarray(task_mean / (1.0 + ridge_value), dtype=np.float64)

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
        if not torch.all(torch.isfinite(scenarios)):
            raise ValueError("scenarios must be finite")
        if (
            not torch.all(torch.isfinite(probabilities))
            or not torch.all(probabilities >= 0.0)
            or not torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(
                    len(contexts),
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                ),
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise ValueError("each probability row must be finite and sum to one")
        ridge_value = _nonnegative_ridge(ridge)
        basis = torch.tensor(
            self.task_basis.copy(),
            dtype=scenarios.dtype,
            device=scenarios.device,
        )
        return probabilities @ scenarios @ basis / (1.0 + ridge_value)

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
        scenarios = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        return value[None, :] - scenarios @ self.task_basis

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
            raise ValueError("decisions have the wrong shape")
        if outcomes.ndim != 2 or outcomes.shape != (
            len(contexts),
            self.scenario_dim,
        ):
            raise ValueError("outcomes have the wrong shape")
        if not torch.all(torch.isfinite(decisions)) or not torch.all(
            torch.isfinite(outcomes)
        ):
            raise ValueError("decisions and outcomes must be finite")
        basis = torch.tensor(
            self.task_basis.copy(),
            dtype=outcomes.dtype,
            device=outcomes.device,
        )
        difference = decisions - outcomes @ basis
        return 0.5 * difference.square().sum(dim=1)

    def _validate_context_numpy(self, context: Array) -> None:
        value = np.asarray(context, dtype=np.float64).reshape(-1)
        if value.shape != (self.context_dim,) or np.any(~np.isfinite(value)):
            raise ValueError(f"context must have shape ({self.context_dim},)")

    def _validate_contexts_torch(self, contexts: Tensor) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(
                f"contexts must have shape (n,{self.context_dim})"
            )
        if not torch.all(torch.isfinite(contexts)):
            raise ValueError("contexts must be finite")


def _nonnegative_ridge(ridge: float) -> float:
    value = float(ridge)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    return value
