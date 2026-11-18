"""Joint Adam training for independent local KKT Atlas models."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
    realized_loss_from_scenarios_torch,
)
from kkt_atlas.scenario_model import LocalModelSnapshot, LocalScenarioModel
from kkt_atlas.training import TrainingConfig, TrainingResult, _batch_indices, _full_objective


@dataclass
class _ChartState:
    model: LocalScenarioModel
    contexts: torch.Tensor
    centered_contexts: torch.Tensor
    outcomes: torch.Tensor
    config: TrainingConfig
    parameters: tuple[torch.nn.Parameter, ...]
    generator: torch.Generator
    checkpoints: list[LocalModelSnapshot]


def train_local_models_with_grouped_optimizer(
    problem: DifferentiableStochasticProblem,
    models: Sequence[LocalScenarioModel],
    contexts: Sequence[Array],
    outcomes: Sequence[Array],
    *,
    ridge: float,
    configs: Sequence[TrainingConfig],
) -> tuple[TrainingResult, ...]:
    """Train charts sequentially while sharing one Adam update per step."""

    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    if not models:
        raise ValueError("at least one local model is required")
    if not (len(models) == len(contexts) == len(outcomes) == len(configs)):
        raise ValueError(
            "models, contexts, outcomes, and configs must have equal lengths"
        )
    _validate_shared_schedule(configs)
    _validate_independent_models(models)
    states = tuple(
        _make_chart_state(
            problem,
            model,
            chart_contexts,
            chart_outcomes,
            config,
        )
        for model, chart_contexts, chart_outcomes, config in zip(
            models,
            contexts,
            outcomes,
            configs,
            strict=True,
        )
    )
    all_parameters = tuple(
        parameter for state in states for parameter in state.parameters
    )
    optimizer = torch.optim.Adam(
        all_parameters,
        lr=configs[0].learning_rate,
        foreach=False,
        fused=False,
    )
    for state in states:
        objective, operational = _full_objective(
            problem,
            state.model,
            state.contexts,
            state.centered_contexts,
            state.outcomes,
            ridge=ridge,
            config=state.config,
        )
        state.checkpoints.append(
            state.model.snapshot(
                step=0,
                training_objective=objective,
                mean_operational_loss=operational,
            )
        )

    save_set = set(configs[0].save_steps)
    for step in range(1, configs[0].steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for state in states:
            _chart_objective_tensor(
                problem,
                state,
                ridge=ridge,
                step=step,
            ).backward()
            _clip_and_validate_chart_gradients(state, step=step)
        optimizer.step()
        for state in states:
            state.model.project_identifiability_()
        if step in save_set:
            for state in states:
                objective, operational = _full_objective(
                    problem,
                    state.model,
                    state.contexts,
                    state.centered_contexts,
                    state.outcomes,
                    ridge=ridge,
                    config=state.config,
                )
                state.checkpoints.append(
                    state.model.snapshot(
                        step=step,
                        training_objective=objective,
                        mean_operational_loss=operational,
                    )
                )
    return tuple(
        TrainingResult(
            checkpoints=tuple(state.checkpoints),
            final_step=configs[0].steps,
        )
        for state in states
    )


def train_local_models_jointly(
    problem: DifferentiableStochasticProblem,
    models: Sequence[LocalScenarioModel],
    contexts: Sequence[Array],
    outcomes: Sequence[Array],
    *,
    ridge: float,
    configs: Sequence[TrainingConfig],
    checkpoint_callback: Callable[[int, tuple[LocalModelSnapshot, ...]], None] | None = None,
    step_callback: Callable[[int], None] | None = None,
    state_callback: Callable[[int, dict[str, object]], None] | None = None,
) -> tuple[TrainingResult, ...]:
    """Train independent chart models with one backward pass and one Adam optimizer.

    The chart objectives are added without sample-size weighting. Because the
    models have disjoint parameters, this sum gives every parameter the same
    gradient as training its chart separately.
    """

    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    if not models:
        raise ValueError("at least one local model is required")
    if not (len(models) == len(contexts) == len(outcomes) == len(configs)):
        raise ValueError("models, contexts, outcomes, and configs must have equal lengths")
    _validate_shared_schedule(configs)
    _validate_independent_models(models)

    states = tuple(
        _make_chart_state(
            problem,
            model,
            chart_contexts,
            chart_outcomes,
            config,
        )
        for model, chart_contexts, chart_outcomes, config in zip(
            models,
            contexts,
            outcomes,
            configs,
            strict=True,
        )
    )
    all_parameters = tuple(
        parameter
        for state in states
        for parameter in state.parameters
    )
    optimizer = torch.optim.Adam(
        all_parameters,
        lr=configs[0].learning_rate,
        foreach=False,
        fused=False,
    )

    def record_state(step: int) -> None:
        if state_callback is not None:
            state_callback(step, {"step": step,
                "models": tuple(state.model.state_dict() for state in states),
                "optimizer": optimizer.state_dict(),
                "chart_rng_states": tuple(state.generator.get_state() for state in states),
                "torch_rng_state": torch.get_rng_state()})

    record_state(0)

    for state in states:
        objective, operational = _full_objective(
            problem,
            state.model,
            state.contexts,
            state.centered_contexts,
            state.outcomes,
            ridge=ridge,
            config=state.config,
        )
        state.checkpoints.append(
            state.model.snapshot(
                step=0,
                training_objective=objective,
                mean_operational_loss=operational,
            )
        )

    if checkpoint_callback is not None:
        checkpoint_callback(0, tuple(state.checkpoints[-1] for state in states))

    steps = configs[0].steps
    save_set = set(configs[0].save_steps)
    solve_from_means = getattr(problem, "solve_from_means_torch", None)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if callable(solve_from_means):
            objective_tensor = _joint_scenario_objective_tensor(
                problem,
                states,
                ridge=ridge,
                step=step,
            )
        else:
            chart_objectives = tuple(
                _chart_objective_tensor(
                    problem,
                    state,
                    ridge=ridge,
                    step=step,
                )
                for state in states
            )
            objective_tensor = torch.stack(chart_objectives).sum()
        objective_tensor.backward()

        for state in states:
            _clip_and_validate_chart_gradients(state, step=step)

        optimizer.step()
        for state in states:
            state.model.project_identifiability_()

        if step_callback is not None:
            step_callback(step)

        if step in save_set:
            record_state(step)
            for state in states:
                objective, operational = _full_objective(
                    problem,
                    state.model,
                    state.contexts,
                    state.centered_contexts,
                    state.outcomes,
                    ridge=ridge,
                    config=state.config,
                )
                state.checkpoints.append(
                    state.model.snapshot(
                        step=step,
                        training_objective=objective,
                        mean_operational_loss=operational,
                    )
                )

            if checkpoint_callback is not None:
                checkpoint_callback(step, tuple(state.checkpoints[-1] for state in states))

    return tuple(
        TrainingResult(
            checkpoints=tuple(state.checkpoints),
            final_step=steps,
        )
        for state in states
    )


def _make_chart_state(
    problem: DifferentiableStochasticProblem,
    model: LocalScenarioModel,
    contexts: Array,
    outcomes: Array,
    config: TrainingConfig,
) -> _ChartState:
    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    outcome_values = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if len(context_values) != len(outcome_values):
        raise ValueError("contexts and outcomes must contain the same observations")

    device = model.scenario_coordinates.device
    dtype = model.scenario_coordinates.dtype
    context_tensor = torch.as_tensor(context_values, dtype=dtype, device=device)
    outcome_tensor = torch.as_tensor(outcome_values, dtype=dtype, device=device)
    centered_context_tensor = model._centered_standardized_contexts(context_tensor)
    model.scenario_coordinates.requires_grad_(config.train_scenario_coordinates)
    model.probability_intercepts.requires_grad_(True)
    model.probability_slopes.requires_grad_(config.train_probability_slopes)
    trained_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    return _ChartState(
        model=model,
        contexts=context_tensor,
        centered_contexts=centered_context_tensor,
        outcomes=outcome_tensor,
        config=config,
        parameters=trained_parameters,
        generator=generator,
        checkpoints=[],
    )


def _chart_objective_tensor(
    problem: DifferentiableStochasticProblem,
    state: _ChartState,
    *,
    ridge: float,
    step: int,
) -> torch.Tensor:
    indices = _batch_indices(
        len(state.contexts),
        state.config.batch_size,
        generator=state.generator,
        device=state.contexts.device,
    )
    if indices is None:
        batch_contexts = state.contexts
        batch_centered_contexts = state.centered_contexts
        batch_outcomes = state.outcomes
    else:
        batch_contexts = state.contexts[indices]
        batch_centered_contexts = state.centered_contexts[indices]
        batch_outcomes = state.outcomes[indices]
    probabilities = state.model._probabilities_from_centered_contexts(
        batch_centered_contexts
    )
    scenarios = state.model.scenarios()
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
    objective = losses.mean()
    objective = objective + (
        state.config.coordinate_penalty
        * state.model.scenario_coordinates.square().sum()
        / state.model.support_size
    )
    objective = objective + (
        state.config.slope_penalty
        * state.model.probability_slopes.square().sum()
    )
    if not torch.isfinite(objective):
        raise FloatingPointError(f"nonfinite training objective at step {step}")
    return objective


def _clip_and_validate_chart_gradients(
    state: _ChartState,
    *,
    step: int,
) -> None:
    if state.config.gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            state.parameters,
            state.config.gradient_clip,
        )
    for parameter in state.parameters:
        if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)):
            raise FloatingPointError(
                f"nonfinite or missing gradient at step {step}"
            )


def _joint_scenario_objective_tensor(
    problem: DifferentiableStochasticProblem,
    states: Sequence[_ChartState],
    *,
    ridge: float,
    step: int,
) -> torch.Tensor:
    """Build independent chart losses in one autograd graph."""

    chart_losses: list[torch.Tensor] = []
    for state in states:
        indices = _batch_indices(
            len(state.contexts),
            state.config.batch_size,
            generator=state.generator,
            device=state.contexts.device,
        )
        if indices is None:
            contexts = state.contexts
            centered_contexts = state.centered_contexts
            outcomes = state.outcomes
        else:
            contexts = state.contexts[indices]
            centered_contexts = state.centered_contexts[indices]
            outcomes = state.outcomes[indices]
        probabilities = state.model._probabilities_from_centered_contexts(
            centered_contexts
        )
        losses = realized_loss_from_scenarios_torch(
            problem,
            contexts,
            state.model.scenarios(),
            probabilities,
            outcomes,
            ridge=ridge,
        )
        if losses.shape != (len(contexts),):
            raise ValueError(
                "realized_loss_torch must return one loss per observation"
            )
        chart_losses.append(losses)

    chart_objectives: list[torch.Tensor] = []
    for state, losses in zip(states, chart_losses, strict=True):
        objective = losses.mean()
        objective = objective + (
            state.config.coordinate_penalty
            * state.model.scenario_coordinates.square().sum()
            / state.model.support_size
        )
        objective = objective + (
            state.config.slope_penalty
            * state.model.probability_slopes.square().sum()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"nonfinite training objective at step {step}"
            )
        chart_objectives.append(objective)
    return torch.stack(chart_objectives).sum()


def _grouped_scenario_objective_tensor(
    problem: DifferentiableStochasticProblem,
    states: Sequence[_ChartState],
    *,
    ridge: float,
    step: int,
) -> torch.Tensor:
    """Use grouped kernels without constructing a dense block support.

    Charts are grouped only when their realized mini-batches and support sizes
    have equal shapes. A problem-specific hook receives tensors with shapes
    ``(B,n,d_x)``, ``(B,K,m)``, ``(B,n,K)``, and ``(B,n,m)``. Therefore the
    kernel evaluates exactly ``B*n*K`` scenario terms. Incompatible groups and
    problems without the hook retain the established chart-by-chart path.
    """

    prepared = []
    for state in states:
        indices = _batch_indices(
            len(state.contexts),
            state.config.batch_size,
            generator=state.generator,
            device=state.contexts.device,
        )
        if indices is None:
            contexts = state.contexts
            centered_contexts = state.centered_contexts
            outcomes = state.outcomes
        else:
            contexts = state.contexts[indices]
            centered_contexts = state.centered_contexts[indices]
            outcomes = state.outcomes[indices]
        probabilities = state.model._probabilities_from_centered_contexts(
            centered_contexts
        )
        prepared.append(
            (
                state,
                contexts,
                outcomes,
                probabilities,
                state.model.scenarios(),
            )
        )

    groups: dict[tuple[int, int], list[tuple]] = {}
    for item in prepared:
        state, contexts, _, _, _ = item
        key = (len(contexts), state.model.support_size)
        groups.setdefault(key, []).append(item)

    # Keep this distinct from quadratic network flow's cache-based protocol.
    grouped_hook = getattr(problem, "realized_loss_from_chart_batches_torch", None)
    losses_by_state: dict[int, torch.Tensor] = {}
    for items in groups.values():
        if callable(grouped_hook) and len(items) > 1:
            grouped_losses = grouped_hook(
                torch.stack([item[1] for item in items]),
                torch.stack([item[4] for item in items]),
                torch.stack([item[3] for item in items]),
                torch.stack([item[2] for item in items]),
                ridge=ridge,
            )
            expected_shape = (len(items), len(items[0][1]))
            if grouped_losses.shape != expected_shape:
                raise ValueError(
                    "grouped loss must return one value per chart and observation"
                )
            for item, losses in zip(items, grouped_losses, strict=True):
                losses_by_state[id(item[0])] = losses
        else:
            for state, contexts, outcomes, probabilities, scenarios in items:
                losses = realized_loss_from_scenarios_torch(
                    problem,
                    contexts,
                    scenarios,
                    probabilities,
                    outcomes,
                    ridge=ridge,
                )
                if losses.shape != (len(contexts),):
                    raise ValueError(
                        "realized_loss_torch must return one loss per observation"
                    )
                losses_by_state[id(state)] = losses

    chart_objectives = []
    for state in states:
        losses = losses_by_state[id(state)]
        objective = losses.mean()
        objective = objective + (
            state.config.coordinate_penalty
            * state.model.scenario_coordinates.square().sum()
            / state.model.support_size
        )
        objective = objective + (
            state.config.slope_penalty
            * state.model.probability_slopes.square().sum()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"nonfinite training objective at step {step}"
            )
        chart_objectives.append(objective)
    return torch.stack(chart_objectives).sum()


def _dense_union_scenario_objective_tensor(
    problem: DifferentiableStochasticProblem,
    states: Sequence[_ChartState],
    *,
    ridge: float,
    step: int,
) -> torch.Tensor:
    """Evaluate all chart batches in one generic block-diagonal solve.

    The scenario supports are concatenated. Each observation receives zero
    probability outside its own chart support. Exact equivalence requires the
    problem mapping to be invariant to appending zero-probability scenarios.
    This policy trades additional zero-mass scenario work for one invocation
    of the problem kernel and is useful when Python/kernel-launch overhead is
    the dominant cost.
    """

    prepared = []
    support_offset = 0
    total_support = sum(state.model.support_size for state in states)
    for state in states:
        indices = _batch_indices(
            len(state.contexts),
            state.config.batch_size,
            generator=state.generator,
            device=state.contexts.device,
        )
        if indices is None:
            contexts = state.contexts
            centered_contexts = state.centered_contexts
            outcomes = state.outcomes
        else:
            contexts = state.contexts[indices]
            centered_contexts = state.centered_contexts[indices]
            outcomes = state.outcomes[indices]
        probabilities = state.model._probabilities_from_centered_contexts(
            centered_contexts
        )
        before = torch.zeros(
            (len(contexts), support_offset),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        after_size = total_support - support_offset - state.model.support_size
        after = torch.zeros(
            (len(contexts), after_size),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        prepared.append(
            (
                state,
                contexts,
                outcomes,
                torch.cat((before, probabilities, after), dim=1),
            )
        )
        support_offset += state.model.support_size

    losses = realized_loss_from_scenarios_torch(
        problem,
        torch.cat([item[1] for item in prepared], dim=0),
        torch.cat([state.model.scenarios() for state in states], dim=0),
        torch.cat([item[3] for item in prepared], dim=0),
        torch.cat([item[2] for item in prepared], dim=0),
        ridge=ridge,
    )
    expected_observations = sum(len(item[1]) for item in prepared)
    if losses.shape != (expected_observations,):
        raise ValueError(
            "realized_loss_torch must return one loss per observation"
        )

    chart_objectives = []
    observation_offset = 0
    for state, contexts, _, _ in prepared:
        chart_losses = losses[
            observation_offset : observation_offset + len(contexts)
        ]
        observation_offset += len(contexts)
        objective = chart_losses.mean()
        objective = objective + (
            state.config.coordinate_penalty
            * state.model.scenario_coordinates.square().sum()
            / state.model.support_size
        )
        objective = objective + (
            state.config.slope_penalty
            * state.model.probability_slopes.square().sum()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"nonfinite training objective at step {step}"
            )
        chart_objectives.append(objective)
    return torch.stack(chart_objectives).sum()


def _validate_shared_schedule(configs: Sequence[TrainingConfig]) -> None:
    first = configs[0]
    for config in configs[1:]:
        if config.steps != first.steps:
            raise ValueError("joint chart configs must use the same number of steps")
        if config.save_steps != first.save_steps:
            raise ValueError("joint chart configs must use the same save steps")
        if config.learning_rate != first.learning_rate:
            raise ValueError("joint chart configs must use the same learning rate")


def _validate_independent_models(models: Sequence[LocalScenarioModel]) -> None:
    model_ids = tuple(id(model) for model in models)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("joint chart models must be distinct")
    parameter_ids = tuple(
        id(parameter)
        for model in models
        for parameter in model.parameters()
    )
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("joint chart models must not share parameters")

    first = models[0].scenario_coordinates
    for model in models[1:]:
        parameter = model.scenario_coordinates
        if parameter.device != first.device or parameter.dtype != first.dtype:
            raise ValueError("joint chart models must use the same dtype and device")
