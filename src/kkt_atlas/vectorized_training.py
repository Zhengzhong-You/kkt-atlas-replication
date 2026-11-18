"""Vectorized full-batch training for equal-structure local Atlas models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
    realized_loss_from_scenarios_torch,
)
from kkt_atlas.scenario_model import LocalModelSnapshot, LocalScenarioModel
from kkt_atlas.training import TrainingConfig, TrainingResult


@dataclass
class _VectorizedState:
    models: tuple[LocalScenarioModel, ...]
    configs: tuple[TrainingConfig, ...]
    contexts: torch.Tensor | None
    outcomes: torch.Tensor | None
    padded_centered_contexts: torch.Tensor
    chart_sizes: tuple[int, ...]
    chart_sizes_tensor: torch.Tensor
    transformed_means: torch.Tensor
    transformed_scales: torch.Tensor
    bases: torch.Tensor | None
    scenario_coordinates: torch.nn.Parameter
    probability_intercepts: torch.nn.Parameter
    probability_slopes: torch.nn.Parameter
    trained_parameters: tuple[torch.nn.Parameter, ...]
    checkpoints: tuple[list[LocalModelSnapshot], ...]
    grouped_loss_cache: object | None


def train_local_models_vectorized(
    problem: DifferentiableStochasticProblem,
    models: Sequence[LocalScenarioModel],
    contexts: Sequence[Array],
    outcomes: Sequence[Array],
    *,
    ridge: float,
    configs: Sequence[TrainingConfig],
) -> tuple[TrainingResult, ...]:
    """Train fixed-structure charts in one full-batch tensor program.

    This path is mathematically equivalent to independent chart training. It is
    available only when every chart has the same model shape and the problem
    exposes an exact solve from conditional scenario means.
    """

    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    solve_from_means = getattr(problem, "solve_from_means_torch", None)
    if not callable(solve_from_means):
        raise TypeError(
            "vectorized training requires solve_from_means_torch"
        )
    state = _make_vectorized_state(
        problem,
        models,
        contexts,
        outcomes,
        configs,
        ridge=ridge,
    )
    optimizer = torch.optim.Adam(
        state.trained_parameters,
        lr=state.configs[0].learning_rate,
        foreach=False,
        fused=False,
    )
    _save_checkpoints(
        problem,
        state,
        ridge=ridge,
        step=0,
    )

    save_set = set(state.configs[0].save_steps)
    for step in range(1, state.configs[0].steps + 1):
        optimizer.zero_grad(set_to_none=True)
        objectives, _ = _objectives(
            problem,
            state,
            ridge=ridge,
        )
        objectives.sum().backward()
        _clip_chart_gradients(state)
        for parameter in state.trained_parameters:
            if parameter.grad is None or not torch.all(
                torch.isfinite(parameter.grad)
            ):
                raise FloatingPointError(
                    f"nonfinite or missing gradient at step {step}"
                )
        optimizer.step()
        _project_identifiability(state)
        if step in save_set:
            _save_checkpoints(
                problem,
                state,
                ridge=ridge,
                step=step,
            )

    _copy_final_parameters_to_models(state)
    return tuple(
        TrainingResult(
            checkpoints=tuple(checkpoints),
            final_step=state.configs[0].steps,
        )
        for checkpoints in state.checkpoints
    )


def _make_vectorized_state(
    problem: DifferentiableStochasticProblem,
    models: Sequence[LocalScenarioModel],
    contexts: Sequence[Array],
    outcomes: Sequence[Array],
    configs: Sequence[TrainingConfig],
    *,
    ridge: float,
) -> _VectorizedState:
    if not models:
        raise ValueError("at least one local model is required")
    if not (len(models) == len(contexts) == len(outcomes) == len(configs)):
        raise ValueError(
            "models, contexts, outcomes, and configs must have equal lengths"
        )
    model_tuple = tuple(models)
    config_tuple = tuple(configs)
    _validate_common_structure(model_tuple, config_tuple)

    first_coordinate = model_tuple[0].scenario_coordinates
    device = first_coordinate.device
    dtype = first_coordinate.dtype
    context_arrays = tuple(
        matrix(values, columns=problem.context_dim, name="contexts")
        for values in contexts
    )
    outcome_arrays = tuple(
        matrix(values, columns=problem.scenario_dim, name="outcomes")
        for values in outcomes
    )
    for context_values, outcome_values in zip(
        context_arrays,
        outcome_arrays,
        strict=True,
    ):
        if len(context_values) != len(outcome_values):
            raise ValueError(
                "contexts and outcomes must contain the same observations"
            )

    context_tensors = tuple(
        torch.as_tensor(values, dtype=dtype, device=device)
        for values in context_arrays
    )
    outcome_tensors = tuple(
        torch.as_tensor(values, dtype=dtype, device=device)
        for values in outcome_arrays
    )
    centered_contexts = tuple(
        model._centered_standardized_contexts(values)
        for model, values in zip(model_tuple, context_tensors, strict=True)
    )
    chart_sizes = tuple(len(values) for values in context_tensors)
    maximum_size = max(chart_sizes)
    padded_centered_contexts = torch.zeros(
        (len(model_tuple), maximum_size, problem.context_dim),
        dtype=dtype,
        device=device,
    )
    padded_outcomes = torch.zeros(
        (len(model_tuple), maximum_size, problem.scenario_dim),
        dtype=dtype,
        device=device,
    )
    valid_mask = torch.zeros(
        (len(model_tuple), maximum_size),
        dtype=torch.bool,
        device=device,
    )
    for chart, (values, chart_outcomes) in enumerate(
        zip(centered_contexts, outcome_tensors, strict=True)
    ):
        chart_size = len(values)
        padded_centered_contexts[chart, :chart_size] = values
        padded_outcomes[chart, :chart_size] = chart_outcomes
        valid_mask[chart, :chart_size] = True

    coordinates = torch.nn.Parameter(
        torch.stack(
            [
                model.scenario_coordinates.detach().clone()
                for model in model_tuple
            ],
            dim=0,
        ),
        requires_grad=config_tuple[0].train_scenario_coordinates,
    )
    intercepts = torch.nn.Parameter(
        torch.stack(
            [
                model.probability_intercepts.detach().clone()
                for model in model_tuple
            ],
            dim=0,
        )
    )
    slopes = torch.nn.Parameter(
        torch.stack(
            [
                model.probability_slopes.detach().clone()
                for model in model_tuple
            ],
            dim=0,
        ),
        requires_grad=config_tuple[0].train_probability_slopes,
    )
    parameters = tuple(
        parameter
        for parameter in (coordinates, intercepts, slopes)
        if parameter.requires_grad
    )
    bases = (
        None
        if model_tuple[0]._ambient_identity
        else torch.stack([model.basis for model in model_tuple], dim=0)
    )
    grouped_loss_cache = None
    cache_builder = getattr(
        problem,
        "make_grouped_low_support_loss_cache",
        None,
    )
    if callable(cache_builder):
        grouped_loss_cache = cache_builder(
            padded_outcomes,
            valid_mask,
            chart_sizes=chart_sizes,
            support_size=model_tuple[0].support_size,
            ridge=ridge,
        )
    flat_contexts = (
        None
        if grouped_loss_cache is not None
        else torch.cat(context_tensors, dim=0)
    )
    flat_outcomes = (
        None
        if grouped_loss_cache is not None
        else torch.cat(outcome_tensors, dim=0)
    )
    return _VectorizedState(
        models=model_tuple,
        configs=config_tuple,
        contexts=flat_contexts,
        outcomes=flat_outcomes,
        padded_centered_contexts=padded_centered_contexts,
        chart_sizes=chart_sizes,
        chart_sizes_tensor=torch.as_tensor(
            chart_sizes,
            dtype=dtype,
            device=device,
        ),
        transformed_means=torch.stack(
            [model.transformed_mean for model in model_tuple],
            dim=0,
        ),
        transformed_scales=torch.stack(
            [model.transformed_scale for model in model_tuple],
            dim=0,
        ),
        bases=bases,
        scenario_coordinates=coordinates,
        probability_intercepts=intercepts,
        probability_slopes=slopes,
        trained_parameters=parameters,
        checkpoints=tuple([] for _ in model_tuple),
        grouped_loss_cache=grouped_loss_cache,
    )


def _validate_common_structure(
    models: tuple[LocalScenarioModel, ...],
    configs: tuple[TrainingConfig, ...],
) -> None:
    first_model = models[0]
    first_config = configs[0]
    model_ids = tuple(id(model) for model in models)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("vectorized chart models must be distinct")
    for model in models[1:]:
        if (
            model.width != first_model.width
            or model.support_size != first_model.support_size
            or model.context_dim != first_model.context_dim
            or model.problem.scenario_dim != first_model.problem.scenario_dim
            or model._ambient_identity != first_model._ambient_identity
            or model.scenario_coordinates.dtype
            != first_model.scenario_coordinates.dtype
            or model.scenario_coordinates.device
            != first_model.scenario_coordinates.device
        ):
            raise ValueError(
                "vectorized chart models must share one tensor structure"
            )
    shared_fields = (
        "steps",
        "learning_rate",
        "save_steps",
        "batch_size",
        "coordinate_penalty",
        "slope_penalty",
        "gradient_clip",
        "train_scenario_coordinates",
        "train_probability_slopes",
    )
    for config in configs[1:]:
        if any(
            getattr(config, field) != getattr(first_config, field)
            for field in shared_fields
        ):
            raise ValueError(
                "vectorized chart configs must share all controls except seed"
            )
    if first_config.batch_size != 0:
        raise ValueError("vectorized chart training currently requires full batch")


def _scenarios(
    problem: DifferentiableStochasticProblem,
    state: _VectorizedState,
) -> torch.Tensor:
    standardized = (
        state.scenario_coordinates
        if state.bases is None
        else torch.bmm(
            state.scenario_coordinates,
            state.bases.transpose(1, 2),
        )
    )
    transformed = (
        state.transformed_means[:, None, :]
        + standardized * state.transformed_scales[:, None, :]
    )
    return problem.decode_torch(transformed)


def _objectives(
    problem: DifferentiableStochasticProblem,
    state: _VectorizedState,
    *,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (
        state.probability_intercepts[:, None, :]
        + torch.einsum(
            "bnd,bkd->bnk",
            state.padded_centered_contexts,
            state.probability_slopes,
        )
    )
    probabilities = torch.softmax(logits, dim=2)
    scenarios = _scenarios(problem, state)
    grouped_loss = getattr(
        problem,
        "realized_loss_from_grouped_scenarios_torch",
        None,
    )
    if state.grouped_loss_cache is not None and callable(grouped_loss):
        grouped_losses = grouped_loss(
            scenarios,
            probabilities,
            state.grouped_loss_cache,
            ridge=ridge,
        )
        expected_shape = (
            len(state.models),
            state.padded_centered_contexts.shape[1],
        )
        if grouped_losses.shape != expected_shape:
            raise ValueError(
                "grouped realized losses must align with padded chart observations"
            )
        operational = (
            grouped_losses.sum(dim=1) / state.chart_sizes_tensor
        )
    else:
        if state.contexts is None or state.outcomes is None:
            raise RuntimeError(
                "flat observations are required without a grouped loss cache"
            )
        context_chunks = torch.split(state.contexts, state.chart_sizes)
        outcome_chunks = torch.split(state.outcomes, state.chart_sizes)
        losses = torch.cat(
            [
                realized_loss_from_scenarios_torch(
                    problem,
                    chart_contexts,
                    scenarios[chart],
                    probabilities[chart, :chart_size],
                    chart_outcomes,
                    ridge=ridge,
                )
                for chart, (
                    chart_size,
                    chart_contexts,
                    chart_outcomes,
                ) in enumerate(
                    zip(
                        state.chart_sizes,
                        context_chunks,
                        outcome_chunks,
                        strict=True,
                    )
                )
            ],
            dim=0,
        )
        if losses.shape != (sum(state.chart_sizes),):
            raise ValueError(
                "realized_loss_torch must return one loss per observation"
            )
        operational = torch.stack(
            [
                chart_losses.mean()
                for chart_losses in torch.split(losses, state.chart_sizes)
            ]
        )
    objectives = operational
    objectives = objectives + (
        state.configs[0].coordinate_penalty
        * state.scenario_coordinates.square().sum(dim=(1, 2))
        / state.models[0].support_size
    )
    objectives = objectives + (
        state.configs[0].slope_penalty
        * state.probability_slopes.square().sum(dim=(1, 2))
    )
    if not torch.all(torch.isfinite(objectives)):
        raise FloatingPointError("nonfinite vectorized training objective")
    return objectives, operational


def _clip_chart_gradients(state: _VectorizedState) -> None:
    threshold = state.configs[0].gradient_clip
    if threshold is None:
        return
    gradients = tuple(
        parameter.grad
        for parameter in state.trained_parameters
        if parameter.grad is not None
    )
    for chart in range(len(state.models)):
        norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(gradient[chart], ord=2)
                    for gradient in gradients
                ]
            ),
            ord=2,
        )
        coefficient = torch.clamp(
            torch.as_tensor(
                threshold,
                dtype=norm.dtype,
                device=norm.device,
            )
            / (norm + 1e-6),
            max=1.0,
        )
        for gradient in gradients:
            gradient[chart].mul_(coefficient)


@torch.no_grad()
def _project_identifiability(state: _VectorizedState) -> None:
    state.probability_intercepts.sub_(
        state.probability_intercepts.mean(dim=1, keepdim=True)
    )
    state.probability_slopes.sub_(
        state.probability_slopes.mean(dim=1, keepdim=True)
    )


@torch.no_grad()
def _copy_final_parameters_to_models(state: _VectorizedState) -> None:
    for chart, model in enumerate(state.models):
        model.scenario_coordinates.copy_(state.scenario_coordinates[chart])
        model.probability_intercepts.copy_(
            state.probability_intercepts[chart]
        )
        model.probability_slopes.copy_(state.probability_slopes[chart])
        model.project_identifiability_()


@torch.no_grad()
def _save_checkpoints(
    problem: DifferentiableStochasticProblem,
    state: _VectorizedState,
    *,
    ridge: float,
    step: int,
) -> None:
    objectives, operational = _objectives(
        problem,
        state,
        ridge=ridge,
    )
    for chart, checkpoints in enumerate(state.checkpoints):
        checkpoints.append(
            LocalModelSnapshot(
                step=step,
                scenario_coordinates=(
                    state.scenario_coordinates[chart]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                ),
                probability_intercepts=(
                    state.probability_intercepts[chart]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                ),
                probability_slopes=(
                    state.probability_slopes[chart]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                ),
                training_objective=float(objectives[chart].item()),
                mean_operational_loss=float(operational[chart].item()),
            )
        )
