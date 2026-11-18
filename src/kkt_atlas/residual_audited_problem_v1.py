"""Validation adapter for the common residual-audited prescription."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from kkt_atlas.case_studies.network_reservation import NetworkReservationProblem
from kkt_atlas.symmetric_evaluation_v1 import solve_predicted_laws
from kkt_atlas.symmetric_laws_v1 import FiniteLaw


class ResidualAuditedProblem:
    def __init__(self, problem: object) -> None:
        self._problem = problem

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def solve_numpy(
        self,
        context: np.ndarray,
        scenarios: np.ndarray,
        probabilities: np.ndarray,
        *,
        ridge: float,
    ) -> np.ndarray:
        contexts = np.asarray(context, dtype=np.float64).reshape(1, -1)
        law = _law(scenarios, probabilities)
        decisions, _ = solve_predicted_laws(
            self._problem,
            contexts,
            (law,),
            eta=ridge,
        )
        return decisions[0]

    def solve_torch(
        self,
        contexts: torch.Tensor,
        scenarios: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        ridge: float,
    ) -> torch.Tensor:
        context_values = contexts.detach().cpu().numpy()
        scenario_values = scenarios.detach().cpu().numpy()
        probability_values = probabilities.detach().cpu().numpy()
        laws = tuple(
            _law(scenario_values, row)
            for row in probability_values
        )
        decisions, _ = solve_predicted_laws(
            self._problem,
            context_values,
            laws,
            eta=ridge,
        )
        return torch.as_tensor(
            decisions,
            dtype=contexts.dtype,
            device=contexts.device,
        )

    def realized_loss_torch(
        self,
        decisions: torch.Tensor,
        outcomes: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(self._problem, NetworkReservationProblem):
            return self._problem.realized_loss_torch(decisions, outcomes, contexts)
        decision_values = decisions.detach().cpu().numpy()
        outcome_values = outcomes.detach().cpu().numpy()
        losses = [
            self._problem.formal_scenario_losses_numpy(
                decision,
                outcome[None, :],
                tolerance=1e-6,
                maximum_iterations=5_000,
            )[0][0]
            for decision, outcome in zip(
                decision_values,
                outcome_values,
                strict=True,
            )
        ]
        return torch.as_tensor(
            losses,
            dtype=decisions.dtype,
            device=decisions.device,
        )


def _law(scenarios: np.ndarray, probabilities: np.ndarray) -> FiniteLaw:
    values = np.asarray(scenarios, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return FiniteLaw(
        values,
        weights,
        np.arange(len(weights), dtype=np.int64),
        "catalog",
    )
