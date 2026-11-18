"""Shared evaluator for every method in the canonical experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from kkt_atlas.problem import Array, DifferentiableStochasticProblem, matrix


@dataclass(frozen=True)
class DecisionEvaluation:
    """Exact finite-catalog operational risk for a collection of decisions."""

    context_losses: Array
    mean_loss: float

    def __post_init__(self) -> None:
        losses = np.asarray(self.context_losses, dtype=np.float64).reshape(-1)
        if len(losses) == 0 or np.any(~np.isfinite(losses)):
            raise ValueError("context_losses must be nonempty and finite")
        mean_loss = float(self.mean_loss)
        if not math.isfinite(mean_loss):
            raise ValueError("mean_loss must be finite")
        if not math.isclose(
            mean_loss,
            float(losses.mean()),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("mean_loss does not match context_losses")
        losses = np.ascontiguousarray(losses.copy())
        losses.setflags(write=False)
        object.__setattr__(self, "context_losses", losses)
        object.__setattr__(self, "mean_loss", mean_loss)


def full_support_decisions(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    catalog: Array,
    probabilities: Array,
    *,
    ridge: float,
    context_batch_size: int = 256,
) -> Array:
    """Solve the full finite-support problem used only as an evaluator oracle."""

    context_values, catalog_values, probability_values = _validated_inputs(
        problem,
        contexts,
        catalog,
        probabilities,
    )
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be positive")

    scenario_tensor = torch.tensor(catalog_values, dtype=torch.float64)
    decisions = np.empty(
        (len(context_values), problem.decision_dim),
        dtype=np.float64,
    )
    with torch.no_grad():
        for start in range(0, len(context_values), context_batch_size):
            stop = min(start + context_batch_size, len(context_values))
            context_tensor = torch.tensor(
                context_values[start:stop],
                dtype=torch.float64,
            )
            probability_tensor = torch.tensor(
                probability_values[start:stop],
                dtype=torch.float64,
            )
            block = problem.solve_torch(
                context_tensor,
                scenario_tensor,
                probability_tensor,
                ridge=ridge,
            )
            decisions[start:stop] = block.cpu().numpy()
    if np.any(~np.isfinite(decisions)):
        raise FloatingPointError("full-support solver returned a nonfinite decision")
    return decisions


def exact_expected_loss(
    problem: DifferentiableStochasticProblem,
    decisions: Array,
    contexts: Array,
    catalog: Array,
    probabilities: Array,
    *,
    context_batch_size: int = 64,
    scenario_batch_size: int = 256,
) -> DecisionEvaluation:
    """Evaluate decisions under the exact conditional finite-catalog law."""

    context_values, catalog_values, probability_values = _validated_inputs(
        problem,
        contexts,
        catalog,
        probabilities,
    )
    decision_values = matrix(
        decisions,
        columns=problem.decision_dim,
        name="decisions",
    )
    if len(decision_values) != len(context_values):
        raise ValueError("decisions and contexts must contain the same rows")
    if context_batch_size < 1 or scenario_batch_size < 1:
        raise ValueError("evaluation batch sizes must be positive")

    context_losses = np.zeros(len(context_values), dtype=np.float64)
    with torch.no_grad():
        for context_start in range(0, len(context_values), context_batch_size):
            context_stop = min(
                context_start + context_batch_size,
                len(context_values),
            )
            block_contexts = torch.tensor(
                context_values[context_start:context_stop],
                dtype=torch.float64,
            )
            block_decisions = torch.tensor(
                decision_values[context_start:context_stop],
                dtype=torch.float64,
            )
            block_probabilities = probability_values[context_start:context_stop]
            block_losses = torch.zeros(
                len(block_contexts),
                dtype=torch.float64,
            )

            for scenario_start in range(
                0,
                len(catalog_values),
                scenario_batch_size,
            ):
                scenario_stop = min(
                    scenario_start + scenario_batch_size,
                    len(catalog_values),
                )
                block_catalog = torch.tensor(
                    catalog_values[scenario_start:scenario_stop],
                    dtype=torch.float64,
                )
                context_count = len(block_contexts)
                scenario_count = len(block_catalog)
                repeated_decisions = (
                    block_decisions[:, None, :]
                    .expand(context_count, scenario_count, -1)
                    .reshape(-1, problem.decision_dim)
                )
                repeated_outcomes = (
                    block_catalog[None, :, :]
                    .expand(context_count, scenario_count, -1)
                    .reshape(-1, problem.scenario_dim)
                )
                repeated_contexts = (
                    block_contexts[:, None, :]
                    .expand(context_count, scenario_count, -1)
                    .reshape(-1, problem.context_dim)
                )
                losses = problem.realized_loss_torch(
                    repeated_decisions,
                    repeated_outcomes,
                    repeated_contexts,
                ).reshape(context_count, scenario_count)
                weights = torch.tensor(
                    block_probabilities[:, scenario_start:scenario_stop],
                    dtype=torch.float64,
                )
                block_losses += (weights * losses).sum(dim=1)

            context_losses[context_start:context_stop] = block_losses.cpu().numpy()

    if np.any(~np.isfinite(context_losses)):
        raise FloatingPointError("exact expected loss is nonfinite")
    return DecisionEvaluation(
        context_losses=context_losses,
        mean_loss=float(context_losses.mean()),
    )


def _validated_inputs(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    catalog: Array,
    probabilities: Array,
) -> tuple[Array, Array, Array]:
    context_values = matrix(
        contexts,
        columns=problem.context_dim,
        name="contexts",
    )
    catalog_values = matrix(
        catalog,
        columns=problem.scenario_dim,
        name="catalog",
    )
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if probability_values.shape != (len(context_values), len(catalog_values)):
        raise ValueError("probabilities must align with contexts and catalog")
    if np.any(~np.isfinite(probability_values)) or np.any(
        probability_values < 0.0
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(
        probability_values.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("each probability row must sum to one")
    return (
        context_values,
        catalog_values,
        np.ascontiguousarray(probability_values),
    )
