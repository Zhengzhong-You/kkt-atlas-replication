"""Pooled and chart-local sample-average approximation baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from kkt_atlas.charts import ContextCharts, fit_context_charts
from kkt_atlas.data import FiniteCatalogData, ObservedSplit
from kkt_atlas.problem import Array, DifferentiableStochasticProblem, matrix


@dataclass(frozen=True)
class PooledSAA:
    problem: DifferentiableStochasticProblem
    scenarios: Array
    ridge: float
    validation_loss: float

    def decisions(self, contexts: Array, *, batch_size: int = 256) -> Array:
        return _uniform_support_decisions(
            self.problem,
            contexts,
            self.scenarios,
            ridge=self.ridge,
            batch_size=batch_size,
        )


@dataclass(frozen=True)
class LocalSAA:
    problem: DifferentiableStochasticProblem
    charts: ContextCharts
    scenarios: tuple[Array, ...]
    ridge: float
    validation_loss: float

    @property
    def total_support(self) -> int:
        return sum(len(scenarios) for scenarios in self.scenarios)

    def decisions(self, contexts: Array, *, batch_size: int = 256) -> Array:
        values = matrix(
            contexts,
            columns=self.problem.context_dim,
            name="contexts",
        )
        assignments = self.charts.assign(values)
        decisions = np.empty(
            (len(values), self.problem.decision_dim),
            dtype=np.float64,
        )
        for chart, scenarios in enumerate(self.scenarios):
            indices = np.flatnonzero(assignments == chart)
            if len(indices) == 0:
                continue
            decisions[indices] = _uniform_support_decisions(
                self.problem,
                values[indices],
                scenarios,
                ridge=self.ridge,
                batch_size=batch_size,
            )
        if np.any(~np.isfinite(decisions)):
            raise FloatingPointError("local SAA returned a nonfinite decision")
        return decisions


def fit_pooled_saa(
    problem: DifferentiableStochasticProblem,
    data: FiniteCatalogData,
    *,
    ridges: tuple[float, ...],
) -> PooledSAA:
    """Select the ridge for one empirical distribution using validation loss."""

    _validate_problem_data(problem, data)
    ridge_grid = _ridge_grid(ridges)
    scenarios = data.training.outcomes(data.catalog)
    scored: list[tuple[float, float]] = []
    for ridge in ridge_grid:
        decisions = _uniform_support_decisions(
            problem,
            data.validation.contexts,
            scenarios,
            ridge=ridge,
        )
        scored.append(
            (
                _observed_mean_loss(
                    problem,
                    decisions,
                    data.validation,
                    data.catalog,
                ),
                ridge,
            )
        )
    validation_loss, ridge = min(scored, key=lambda item: (item[0], item[1]))
    return PooledSAA(
        problem=problem,
        scenarios=_immutable(scenarios),
        ridge=ridge,
        validation_loss=validation_loss,
    )


def fit_local_saa(
    problem: DifferentiableStochasticProblem,
    data: FiniteCatalogData,
    *,
    chart_counts: tuple[int, ...],
    ridges: tuple[float, ...],
    seed: int,
    minimum_chart_size: int,
) -> LocalSAA:
    """Select a fixed context partition and one ridge using validation loss."""

    _validate_problem_data(problem, data)
    ridge_grid = _ridge_grid(ridges)
    chart_grid = tuple(sorted(set(int(value) for value in chart_counts)))
    if not chart_grid or chart_grid[0] < 1:
        raise ValueError("chart_counts must contain positive integers")
    training_outcomes = data.training.outcomes(data.catalog)
    candidates: list[LocalSAA] = []
    for chart_count in chart_grid:
        try:
            charts = fit_context_charts(
                data.training.contexts,
                chart_count,
                seed=_derived_seed(seed, chart_count),
                minimum_chart_size=minimum_chart_size,
            )
        except ValueError:
            continue
        validation_assignments = charts.assign(data.validation.contexts)
        if np.any(
            np.bincount(
                validation_assignments,
                minlength=chart_count,
            )
            == 0
        ):
            continue
        local_scenarios = tuple(
            _immutable(training_outcomes[charts.indices(chart)])
            for chart in range(chart_count)
        )
        for ridge in ridge_grid:
            candidate = LocalSAA(
                problem=problem,
                charts=charts,
                scenarios=local_scenarios,
                ridge=ridge,
                validation_loss=0.0,
            )
            decisions = candidate.decisions(data.validation.contexts)
            loss = _observed_mean_loss(
                problem,
                decisions,
                data.validation,
                data.catalog,
            )
            candidates.append(
                LocalSAA(
                    problem=problem,
                    charts=charts,
                    scenarios=local_scenarios,
                    ridge=ridge,
                    validation_loss=loss,
                )
            )
    if not candidates:
        raise RuntimeError("no valid local SAA candidate")
    return min(
        candidates,
        key=lambda candidate: (
            candidate.validation_loss,
            candidate.charts.chart_count,
            candidate.ridge,
        ),
    )


def _uniform_support_decisions(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    scenarios: Array,
    *,
    ridge: float,
    batch_size: int = 256,
) -> Array:
    context_values = matrix(
        contexts,
        columns=problem.context_dim,
        name="contexts",
    )
    scenario_values = matrix(
        scenarios,
        columns=problem.scenario_dim,
        name="scenarios",
    )
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be positive and finite")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scenario_tensor = torch.tensor(scenario_values, dtype=torch.float64)
    decisions = np.empty(
        (len(context_values), problem.decision_dim),
        dtype=np.float64,
    )
    with torch.no_grad():
        for start in range(0, len(context_values), batch_size):
            stop = min(start + batch_size, len(context_values))
            context_tensor = torch.tensor(
                context_values[start:stop],
                dtype=torch.float64,
            )
            probabilities = torch.full(
                (len(context_tensor), len(scenario_tensor)),
                1.0 / len(scenario_tensor),
                dtype=torch.float64,
            )
            block = problem.solve_torch(
                context_tensor,
                scenario_tensor,
                probabilities,
                ridge=ridge,
            )
            decisions[start:stop] = block.cpu().numpy()
    return decisions


@torch.no_grad()
def _observed_mean_loss(
    problem: DifferentiableStochasticProblem,
    decisions: Array,
    split: ObservedSplit,
    catalog: Array,
) -> float:
    decision_tensor = torch.tensor(decisions, dtype=torch.float64)
    context_tensor = torch.tensor(split.contexts, dtype=torch.float64)
    outcome_tensor = torch.tensor(split.outcomes(catalog), dtype=torch.float64)
    losses = problem.realized_loss_torch(
        decision_tensor,
        outcome_tensor,
        context_tensor,
    )
    result = float(losses.mean().item())
    if not math.isfinite(result):
        raise FloatingPointError("SAA validation loss is nonfinite")
    return result


def _ridge_grid(ridges: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(sorted(set(float(value) for value in ridges)))
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("ridges must contain positive finite values")
    return values


def _immutable(values: Array) -> Array:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).copy())
    array.setflags(write=False)
    return array


def _derived_seed(seed: int, chart_count: int) -> int:
    return int((int(seed) * 1_000_003 + chart_count * 97) % (2**32))


def _validate_problem_data(
    problem: DifferentiableStochasticProblem,
    data: FiniteCatalogData,
) -> None:
    if data.context_dim != problem.context_dim:
        raise ValueError("data and problem context dimensions differ")
    if data.scenario_dim != problem.scenario_dim:
        raise ValueError("data and problem scenario dimensions differ")
