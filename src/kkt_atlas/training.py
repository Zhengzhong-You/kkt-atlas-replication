"""Deterministic Adam training for one local KKT Atlas model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
    realized_loss_from_scenarios_torch,
)
from kkt_atlas.scenario_model import LocalModelSnapshot, LocalScenarioModel


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    learning_rate: float
    save_steps: tuple[int, ...]
    batch_size: int = 0
    coordinate_penalty: float = 0.0
    slope_penalty: float = 0.0
    gradient_clip: float | None = None
    train_scenario_coordinates: bool = True
    train_probability_slopes: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if self.batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if self.coordinate_penalty < 0.0 or self.slope_penalty < 0.0:
            raise ValueError("regularization penalties must be nonnegative")
        if self.gradient_clip is not None and self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive when supplied")
        if not isinstance(self.train_scenario_coordinates, bool):
            raise TypeError("train_scenario_coordinates must be a boolean")
        if not isinstance(self.train_probability_slopes, bool):
            raise TypeError("train_probability_slopes must be a boolean")
        saves = tuple(sorted(set(int(step) for step in self.save_steps)))
        if any(step < 0 or step > self.steps for step in saves):
            raise ValueError("save_steps must lie between zero and steps")
        if 0 not in saves:
            raise ValueError("save_steps must include the initialization at step zero")
        object.__setattr__(self, "save_steps", saves)


@dataclass(frozen=True)
class TrainingResult:
    checkpoints: tuple[LocalModelSnapshot, ...]
    final_step: int

    def checkpoint(self, step: int) -> LocalModelSnapshot:
        for checkpoint in self.checkpoints:
            if checkpoint.step == step:
                return checkpoint
        raise KeyError(step)


def train_local_model(
    problem: DifferentiableStochasticProblem,
    model: LocalScenarioModel,
    contexts: Array,
    outcomes: Array,
    *,
    ridge: float,
    config: TrainingConfig,
) -> TrainingResult:
    """Train scenarios and local probabilities through realized operational loss."""

    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    outcome_values = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if len(context_values) != len(outcome_values):
        raise ValueError("contexts and outcomes must contain the same observations")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")

    device = model.scenario_coordinates.device
    dtype = model.scenario_coordinates.dtype
    context_tensor = torch.as_tensor(context_values, dtype=dtype, device=device)
    outcome_tensor = torch.as_tensor(outcome_values, dtype=dtype, device=device)
    centered_context_tensor = model._centered_standardized_contexts(context_tensor)
    model.scenario_coordinates.requires_grad_(config.train_scenario_coordinates)
    model.probability_intercepts.requires_grad_(True)
    model.probability_slopes.requires_grad_(config.train_probability_slopes)
    trained_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(
        trained_parameters,
        lr=config.learning_rate,
        foreach=False,
        fused=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    checkpoints: list[LocalModelSnapshot] = []
    save_set = set(config.save_steps)

    objective, operational = _full_objective(
        problem,
        model,
        context_tensor,
        centered_context_tensor,
        outcome_tensor,
        ridge=ridge,
        config=config,
    )
    checkpoints.append(
        model.snapshot(
            step=0,
            training_objective=objective,
            mean_operational_loss=operational,
        )
    )

    for step in range(1, config.steps + 1):
        indices = _batch_indices(
            len(context_tensor),
            config.batch_size,
            generator=generator,
            device=device,
        )
        if indices is None:
            batch_contexts = context_tensor
            batch_centered_contexts = centered_context_tensor
            batch_outcomes = outcome_tensor
        else:
            batch_contexts = context_tensor[indices]
            batch_centered_contexts = centered_context_tensor[indices]
            batch_outcomes = outcome_tensor[indices]
        optimizer.zero_grad(set_to_none=True)
        probabilities = model._probabilities_from_centered_contexts(
            batch_centered_contexts
        )
        scenarios = model.scenarios()
        losses = realized_loss_from_scenarios_torch(
            problem,
            batch_contexts,
            scenarios,
            probabilities,
            batch_outcomes,
            ridge=ridge,
        )
        if losses.shape != (len(batch_contexts),):
            raise ValueError("realized_loss_torch must return one loss per observation")
        objective_tensor = losses.mean()
        objective_tensor = objective_tensor + (
            config.coordinate_penalty
            * model.scenario_coordinates.square().sum()
            / model.support_size
        )
        objective_tensor = objective_tensor + (
            config.slope_penalty * model.probability_slopes.square().sum()
        )
        if not torch.isfinite(objective_tensor):
            raise FloatingPointError(f"nonfinite training objective at step {step}")
        objective_tensor.backward()
        if config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(trained_parameters, config.gradient_clip)
        for parameter in trained_parameters:
            if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)):
                raise FloatingPointError(f"nonfinite or missing gradient at step {step}")
        optimizer.step()
        model.project_identifiability_()

        if step in save_set:
            objective, operational = _full_objective(
                problem,
                model,
                context_tensor,
                centered_context_tensor,
                outcome_tensor,
                ridge=ridge,
                config=config,
            )
            checkpoints.append(
                model.snapshot(
                    step=step,
                    training_objective=objective,
                    mean_operational_loss=operational,
                )
            )
    return TrainingResult(checkpoints=tuple(checkpoints), final_step=config.steps)


@torch.no_grad()
def _full_objective(
    problem: DifferentiableStochasticProblem,
    model: LocalScenarioModel,
    contexts: torch.Tensor,
    centered_contexts: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    ridge: float,
    config: TrainingConfig,
) -> tuple[float, float]:
    probabilities = model._probabilities_from_centered_contexts(centered_contexts)
    scenarios = model.scenarios()
    losses = realized_loss_from_scenarios_torch(
        problem,
        contexts,
        scenarios,
        probabilities,
        outcomes,
        ridge=ridge,
    )
    operational = float(losses.mean().item())
    objective = operational
    objective += (
        config.coordinate_penalty
        * float(model.scenario_coordinates.square().sum().item())
        / model.support_size
    )
    objective += config.slope_penalty * float(model.probability_slopes.square().sum().item())
    if not math.isfinite(objective):
        raise FloatingPointError("nonfinite full training objective")
    return float(objective), operational


def _batch_indices(
    sample_size: int,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor | None:
    if batch_size == 0 or batch_size >= sample_size:
        return None
    indices = torch.randperm(sample_size, generator=generator)[:batch_size]
    return indices.to(device=device)
