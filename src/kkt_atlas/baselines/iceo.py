"""Full-support integrated conditional estimation--optimization baseline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from kkt_atlas.data import FiniteCatalogData, ObservedSplit
from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
    realized_loss_from_scenarios_torch,
)


@dataclass(frozen=True)
class ICEOModelConfig:
    """Architecture of the context-to-full-support probability model."""

    hidden_width: int
    hidden_layers: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        if self.hidden_layers < 0:
            raise ValueError("hidden_layers must be nonnegative")


@dataclass(frozen=True)
class ICEOTrainingConfig:
    """Deterministic Adam controls for the full-support baseline."""

    steps: int
    learning_rate: float
    save_steps: tuple[int, ...]
    batch_size: int = 0
    gradient_clip: float | None = None
    cross_entropy_weight: float = 0.0
    seed: int = 0
    microbatch_size: int = 0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if self.batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if self.microbatch_size < 0:
            raise ValueError("microbatch_size must be nonnegative")
        if self.gradient_clip is not None and self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive when supplied")
        if (
            not math.isfinite(self.cross_entropy_weight)
            or self.cross_entropy_weight < 0.0
        ):
            raise ValueError("cross_entropy_weight must be finite and nonnegative")
        saves = tuple(sorted(set(int(step) for step in self.save_steps)))
        if any(step < 0 or step > self.steps for step in saves):
            raise ValueError("save_steps must lie between zero and steps")
        if 0 not in saves:
            raise ValueError("save_steps must include step zero")
        object.__setattr__(self, "save_steps", saves)


@dataclass(frozen=True)
class ICEOSnapshot:
    """Trainable probability-head parameters saved at one Adam step."""

    step: int
    parameters: tuple[tuple[str, Array], ...]
    training_loss: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be nonnegative")
        if not math.isfinite(self.training_loss):
            raise ValueError("training_loss must be finite")
        names: set[str] = set()
        immutable: list[tuple[str, Array]] = []
        for name, value in self.parameters:
            if not name or name in names:
                raise ValueError("snapshot parameter names must be unique and nonempty")
            array = np.asarray(value, dtype=np.float64)
            if np.any(~np.isfinite(array)):
                raise ValueError(f"snapshot parameter {name!r} is nonfinite")
            array = np.ascontiguousarray(array.copy())
            array.setflags(write=False)
            names.add(name)
            immutable.append((name, array))
        if not immutable:
            raise ValueError("snapshot must contain trainable parameters")
        object.__setattr__(self, "parameters", tuple(immutable))


@dataclass(frozen=True)
class ICEOTrainingResult:
    checkpoints: tuple[ICEOSnapshot, ...]
    final_step: int

    def checkpoint(self, step: int) -> ICEOSnapshot:
        for checkpoint in self.checkpoints:
            if checkpoint.step == step:
                return checkpoint
        raise KeyError(step)


@dataclass(frozen=True)
class ICEOFinalTrainingTimings:
    """Disjoint timings for one fixed final ICEO model."""

    model_setup_seconds: float
    adam_update_seconds: float
    final_scoring_seconds: float
    assembly_seconds: float
    total_construction_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.model_setup_seconds,
            self.adam_update_seconds,
            self.final_scoring_seconds,
            self.assembly_seconds,
            self.total_construction_seconds,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("ICEO timings must be finite and nonnegative")


@dataclass(frozen=True)
class ICEOFinalTrainingResult:
    """One fixed final model; no validation or checkpoint selection is run."""

    model: FullSupportICEOModel
    snapshot: ICEOSnapshot
    timings: ICEOFinalTrainingTimings
    optimizer_steps: int
    sample_presentations: int


class FullSupportICEOModel(nn.Module):
    """A neural probability head over every scenario in the reference catalog."""

    def __init__(
        self,
        problem: DifferentiableStochasticProblem,
        catalog: Array,
        training_contexts: Array,
        *,
        config: ICEOModelConfig,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        catalog_array = matrix(catalog, columns=problem.scenario_dim, name="catalog")
        context_array = matrix(
            training_contexts,
            columns=problem.context_dim,
            name="training_contexts",
        )
        mean = context_array.mean(axis=0)
        scale = context_array.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        factory = {"dtype": dtype, "device": device}
        self.problem = problem
        self.model_config = config
        self.context_dim = problem.context_dim
        self.support_size = len(catalog_array)
        self.register_buffer("catalog", torch.tensor(catalog_array, **factory))
        self.register_buffer("context_mean", torch.tensor(mean, **factory))
        self.register_buffer("context_scale", torch.tensor(scale, **factory))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.seed))
            layers: list[nn.Module] = []
            input_width = problem.context_dim
            for _ in range(config.hidden_layers):
                layers.append(nn.Linear(input_width, config.hidden_width, **factory))
                layers.append(nn.ReLU())
                input_width = config.hidden_width
            layers.append(nn.Linear(input_width, self.support_size, **factory))
            self.logit_network = nn.Sequential(*layers)
        self.project_identifiability_()

    def standardized_contexts(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(
                f"contexts must have shape (n,{self.context_dim}), "
                f"got {tuple(contexts.shape)}"
            )
        return (contexts - self.context_mean) / self.context_scale

    def _probabilities_from_standardized_contexts(
        self,
        standardized_contexts: torch.Tensor,
    ) -> torch.Tensor:
        if (
            standardized_contexts.ndim != 2
            or standardized_contexts.shape[1] != self.context_dim
        ):
            raise ValueError(
                "standardized_contexts must have shape "
                f"(n,{self.context_dim}), got {tuple(standardized_contexts.shape)}"
            )
        logits = self.logit_network(standardized_contexts)
        return torch.softmax(logits, dim=1)

    def probabilities(self, contexts: torch.Tensor) -> torch.Tensor:
        return self._probabilities_from_standardized_contexts(
            self.standardized_contexts(contexts)
        )

    def decisions(self, contexts: torch.Tensor, *, ridge: float) -> torch.Tensor:
        return self.problem.solve_torch(
            contexts,
            self.catalog,
            self.probabilities(contexts),
            ridge=ridge,
        )

    @torch.no_grad()
    def project_identifiability_(self) -> None:
        """Remove the common final-logit component without changing probabilities."""

        output = self.logit_network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("the final probability-head layer must be linear")
        output.weight.sub_(output.weight.mean(dim=0, keepdim=True))
        output.bias.sub_(output.bias.mean())

    @torch.no_grad()
    def snapshot(self, *, step: int, training_loss: float) -> ICEOSnapshot:
        parameters = tuple(
            (name, parameter.detach().cpu().numpy().copy())
            for name, parameter in self.named_parameters()
        )
        return ICEOSnapshot(
            step=int(step),
            parameters=parameters,
            training_loss=float(training_loss),
        )

    @torch.no_grad()
    def _validate_snapshot_identifiability(
        self,
        supplied: dict[str, Array],
        current: dict[str, nn.Parameter],
    ) -> None:
        output = self.logit_network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("the final probability-head layer must be linear")
        weight_name = next(
            name for name, parameter in current.items() if parameter is output.weight
        )
        bias_name = next(
            name for name, parameter in current.items() if parameter is output.bias
        )
        weight = np.asarray(supplied[weight_name])
        bias = np.asarray(supplied[bias_name])
        scale = max(
            1.0,
            float(np.max(np.abs(weight))),
            float(np.max(np.abs(bias))),
        )
        tolerance = (
            128.0
            * torch.finfo(output.weight.dtype).eps
            * max(1, self.support_size)
            * scale
        )
        residual = max(
            float(np.max(np.abs(weight.mean(axis=0)))),
            float(abs(bias.mean())),
        )
        if residual > tolerance:
            raise ValueError("snapshot violates the probability-head identifiability convention")

    @torch.no_grad()
    def load_snapshot(self, snapshot: ICEOSnapshot) -> None:
        current = dict(self.named_parameters())
        supplied = dict(snapshot.parameters)
        if current.keys() != supplied.keys():
            raise ValueError("snapshot parameters do not match the probability model")
        for name, parameter in current.items():
            value = np.asarray(supplied[name])
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(
                    f"snapshot parameter {name!r} must have a floating-point dtype"
                )
            if np.any(~np.isfinite(value)):
                raise ValueError(f"snapshot parameter {name!r} must be finite")
            if value.shape != tuple(parameter.shape):
                raise ValueError(f"snapshot parameter {name!r} has the wrong shape")
        self._validate_snapshot_identifiability(supplied, current)
        for name, parameter in current.items():
            value = supplied[name]
            parameter.copy_(
                torch.tensor(value, dtype=parameter.dtype, device=parameter.device)
            )


@dataclass(frozen=True)
class FittedICEO:
    """The validation-selected full-support model used for prediction."""

    model: FullSupportICEOModel
    training_result: ICEOTrainingResult
    selected_step: int
    validation_loss: float
    ridge: float
    validation_trace: tuple[tuple[int, float], ...] = ()

    def probabilities(self, contexts: Array) -> Array:
        values = matrix(
            contexts,
            columns=self.model.context_dim,
            name="contexts",
        )
        device = self.model.catalog.device
        tensor = torch.as_tensor(values, dtype=self.model.catalog.dtype, device=device)
        with torch.no_grad():
            probabilities = self.model.probabilities(tensor)
        return probabilities.detach().cpu().numpy().copy()

    def decisions(self, contexts: Array) -> Array:
        values = matrix(
            contexts,
            columns=self.model.context_dim,
            name="contexts",
        )
        device = self.model.catalog.device
        tensor = torch.as_tensor(values, dtype=self.model.catalog.dtype, device=device)
        with torch.no_grad():
            decisions = self.model.decisions(tensor, ridge=self.ridge)
        return decisions.detach().cpu().numpy().copy()


def train_iceo_model(
    problem: DifferentiableStochasticProblem,
    catalog: Array,
    training: ObservedSplit,
    *,
    ridge: float,
    model_config: ICEOModelConfig,
    training_config: ICEOTrainingConfig,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    callback=None,
    initial_checkpoint=None,
) -> tuple[FullSupportICEOModel, ICEOTrainingResult]:
    """Train only on observed training pairs and save the declared checkpoints."""

    catalog_values = matrix(
        catalog,
        columns=problem.scenario_dim,
        name="catalog",
    )
    _validate_problem_split(problem, catalog_values, training)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    model = FullSupportICEOModel(
        problem,
        catalog_values,
        training.contexts,
        config=model_config,
        dtype=dtype,
        device=device,
    )
    contexts = torch.tensor(
        training.contexts,
        dtype=dtype,
        device=device,
    )
    standardized_contexts = model.standardized_contexts(contexts)
    outcomes = torch.tensor(
        training.outcomes(catalog_values),
        dtype=dtype,
        device=device,
    )
    outcome_indices = (
        torch.tensor(
            training.outcome_indices,
            dtype=torch.long,
            device=device,
        )
        if training_config.cross_entropy_weight > 0.0
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        foreach=False,
        fused=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training_config.seed))
    save_set = set(training_config.save_steps)
    if initial_checkpoint is not None:
        if initial_checkpoint["snapshot"].step != 0 or initial_checkpoint["optimizer"]["state"]:
            raise ValueError("Only an untrained step-zero initialization may be reused")
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,initial_checkpoint["model"][name],rtol=0,atol=0)
        torch.testing.assert_close(generator.get_state(),initial_checkpoint["generator"],rtol=0,atol=0)
        checkpoints = [initial_checkpoint["snapshot"]]
    else:
        checkpoints = [
            model.snapshot(
                step=0,
                training_loss=_mean_training_objective(
                    problem,
                    model,
                    contexts,
                    standardized_contexts,
                    outcomes,
                    outcome_indices,
                    ridge=ridge,
                    cross_entropy_weight=training_config.cross_entropy_weight,
                    microbatch_size=training_config.microbatch_size,
                ),
            )
        ]
    def emit(step, saved):
        if callback is not None:
            state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(),
                     "generator":generator.get_state(), "snapshot":checkpoints[-1]} if saved else None
            callback(step,state)

    emit(0, True)

    for step in range(1, training_config.steps + 1):
        indices = _batch_indices(
            len(contexts),
            training_config.batch_size,
            generator=generator,
            device=model.catalog.device,
        )
        if indices is None:
            batch_contexts = contexts
            batch_standardized_contexts = standardized_contexts
            batch_outcomes = outcomes
            batch_outcome_indices = outcome_indices
        else:
            batch_contexts = contexts[indices]
            batch_standardized_contexts = standardized_contexts[indices]
            batch_outcomes = outcomes[indices]
            batch_outcome_indices = (
                None if outcome_indices is None else outcome_indices[indices]
            )
        _iceo_optimizer_step(
            problem,
            model,
            optimizer,
            batch_contexts,
            batch_standardized_contexts,
            batch_outcomes,
            batch_outcome_indices,
            ridge=ridge,
            config=training_config,
            step=step,
        )
        if step in save_set:
            checkpoints.append(
                model.snapshot(
                    step=step,
                    training_loss=_mean_training_objective(
                        problem,
                        model,
                        contexts,
                        standardized_contexts,
                        outcomes,
                        outcome_indices,
                        ridge=ridge,
                        cross_entropy_weight=training_config.cross_entropy_weight,
                        microbatch_size=training_config.microbatch_size,
                    ),
                )
            )
        emit(step, step in save_set)
    result = ICEOTrainingResult(
        checkpoints=tuple(checkpoints),
        final_step=training_config.steps,
    )
    return model, result


def train_iceo_final_model(
    problem: DifferentiableStochasticProblem,
    catalog: Array,
    training: ObservedSplit,
    *,
    ridge: float,
    model_config: ICEOModelConfig,
    training_config: ICEOTrainingConfig,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> ICEOFinalTrainingResult:
    """Fit one fixed final ICEO model with disjoint phase timers.

    The function does not read validation data, score intermediate checkpoints,
    or select a stopping step.  ``adam_update_seconds`` includes only the fixed
    number of optimizer updates.  Model construction and final full-training
    scoring are recorded separately.
    """

    started = time.perf_counter()
    setup_started = time.perf_counter()
    catalog_values = matrix(
        catalog,
        columns=problem.scenario_dim,
        name="catalog",
    )
    _validate_problem_split(problem, catalog_values, training)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    model = FullSupportICEOModel(
        problem,
        catalog_values,
        training.contexts,
        config=model_config,
        dtype=dtype,
        device=device,
    )
    contexts = torch.tensor(training.contexts, dtype=dtype, device=device)
    standardized_contexts = model.standardized_contexts(contexts)
    outcomes = torch.tensor(
        training.outcomes(catalog_values),
        dtype=dtype,
        device=device,
    )
    outcome_indices = (
        torch.tensor(
            training.outcome_indices,
            dtype=torch.long,
            device=device,
        )
        if training_config.cross_entropy_weight > 0.0
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        foreach=False,
        fused=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training_config.seed))
    model_setup_seconds = time.perf_counter() - setup_started

    adam_update_seconds = 0.0
    for step in range(1, training_config.steps + 1):
        update_started = time.perf_counter()
        indices = _batch_indices(
            len(contexts),
            training_config.batch_size,
            generator=generator,
            device=model.catalog.device,
        )
        if indices is None:
            batch_contexts = contexts
            batch_standardized_contexts = standardized_contexts
            batch_outcomes = outcomes
            batch_outcome_indices = outcome_indices
        else:
            batch_contexts = contexts[indices]
            batch_standardized_contexts = standardized_contexts[indices]
            batch_outcomes = outcomes[indices]
            batch_outcome_indices = (
                None if outcome_indices is None else outcome_indices[indices]
            )
        _iceo_optimizer_step(
            problem,
            model,
            optimizer,
            batch_contexts,
            batch_standardized_contexts,
            batch_outcomes,
            batch_outcome_indices,
            ridge=ridge,
            config=training_config,
            step=step,
        )
        adam_update_seconds += time.perf_counter() - update_started

    scoring_started = time.perf_counter()
    final_loss = _mean_training_objective(
        problem,
        model,
        contexts,
        standardized_contexts,
        outcomes,
        outcome_indices,
        ridge=ridge,
        cross_entropy_weight=training_config.cross_entropy_weight,
        microbatch_size=training_config.microbatch_size,
    )
    snapshot = model.snapshot(step=training_config.steps, training_loss=final_loss)
    final_scoring_seconds = time.perf_counter() - scoring_started
    assembly_started = time.perf_counter()
    batch_size = (
        len(contexts)
        if training_config.batch_size == 0
        or training_config.batch_size >= len(contexts)
        else training_config.batch_size
    )
    optimizer_steps = training_config.steps
    sample_presentations = optimizer_steps * batch_size
    assembly_seconds = time.perf_counter() - assembly_started
    timings = ICEOFinalTrainingTimings(
        model_setup_seconds=float(model_setup_seconds),
        adam_update_seconds=float(adam_update_seconds),
        final_scoring_seconds=float(final_scoring_seconds),
        assembly_seconds=float(assembly_seconds),
        total_construction_seconds=float(time.perf_counter() - started),
    )
    return ICEOFinalTrainingResult(
        model=model,
        snapshot=snapshot,
        timings=timings,
        optimizer_steps=optimizer_steps,
        sample_presentations=sample_presentations,
    )


def select_iceo_checkpoint(
    problem: DifferentiableStochasticProblem,
    catalog: Array,
    validation: ObservedSplit,
    model: FullSupportICEOModel,
    training_result: ICEOTrainingResult,
    *,
    ridge: float,
    microbatch_size: int = 0,
) -> FittedICEO:
    """Select one training checkpoint using only the held-out validation split."""

    if microbatch_size < 0:
        raise ValueError("microbatch_size must be nonnegative")
    catalog_values = matrix(
        catalog,
        columns=problem.scenario_dim,
        name="catalog",
    )
    _validate_problem_split(problem, catalog_values, validation)
    if len(model.catalog) != len(catalog_values):
        raise ValueError("model support does not match the input catalog")
    scored: list[tuple[float, int, ICEOSnapshot, float]] = []
    for checkpoint in training_result.checkpoints:
        model.load_snapshot(checkpoint)
        loss_sum, mean_loss = _split_loss_summary(
            problem,
            model,
            catalog_values,
            validation,
            ridge=ridge,
            microbatch_size=microbatch_size,
        )
        scored.append((loss_sum, checkpoint.step, checkpoint, mean_loss))
    _, selected_step, selected, validation_loss = min(
        scored,
        key=lambda item: (item[0], item[1]),
    )
    model.load_snapshot(selected)
    return FittedICEO(
        model=model,
        training_result=training_result,
        selected_step=selected_step,
        validation_loss=validation_loss,
        ridge=float(ridge),
        validation_trace=tuple((row[1], row[3]) for row in scored),
    )


def fit_iceo(
    problem: DifferentiableStochasticProblem,
    data: FiniteCatalogData,
    *,
    ridge: float,
    model_config: ICEOModelConfig,
    training_config: ICEOTrainingConfig,
    validation_problem: DifferentiableStochasticProblem | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    callback=None,
) -> FittedICEO:
    """Train the full-support baseline and select one saved checkpoint."""

    model, result = train_iceo_model(
        problem,
        data.catalog,
        data.training,
        ridge=ridge,
        model_config=model_config,
        training_config=training_config,
        callback=callback,
        dtype=dtype,
        device=device,
    )
    return select_iceo_checkpoint(
        problem if validation_problem is None else validation_problem,
        data.catalog,
        data.validation,
        model,
        result,
        ridge=ridge,
        microbatch_size=training_config.microbatch_size,
    )


@torch.no_grad()
def _mean_training_objective(
    problem: DifferentiableStochasticProblem,
    model: FullSupportICEOModel,
    contexts: torch.Tensor,
    standardized_contexts: torch.Tensor,
    outcomes: torch.Tensor,
    outcome_indices: torch.Tensor | None,
    *,
    ridge: float,
    cross_entropy_weight: float,
    microbatch_size: int = 0,
) -> float:
    if microbatch_size <= 0 or microbatch_size >= len(contexts):
        probabilities = model._probabilities_from_standardized_contexts(
            standardized_contexts
        )
        losses = realized_loss_from_scenarios_torch(
            problem,
            contexts,
            model.catalog,
            probabilities,
            outcomes,
            ridge=ridge,
        )
        objective = losses.mean()
        if cross_entropy_weight > 0.0:
            if outcome_indices is None:
                raise RuntimeError("outcome indices are required for cross entropy")
            objective = objective + cross_entropy_weight * _negative_log_likelihood(
                probabilities,
                outcome_indices,
            )
        result = float(objective.item())
    else:
        loss_values: list[float] = []
        log_values: list[float] = []
        for start, stop in _microbatch_ranges(len(contexts), microbatch_size):
            probabilities = model._probabilities_from_standardized_contexts(
                standardized_contexts[start:stop]
            )
            losses = realized_loss_from_scenarios_torch(
                problem,
                contexts[start:stop],
                model.catalog,
                probabilities,
                outcomes[start:stop],
                ridge=ridge,
            )
            loss_values.extend(float(value) for value in losses.cpu().tolist())
            if cross_entropy_weight > 0.0:
                if outcome_indices is None:
                    raise RuntimeError("outcome indices are required for cross entropy")
                observed = probabilities.gather(
                    1,
                    outcome_indices[start:stop, None],
                ).squeeze(1)
                log_values.extend(
                    float(value)
                    for value in (
                        -torch.log(
                            observed.clamp_min(torch.finfo(probabilities.dtype).tiny)
                        )
                    )
                    .cpu()
                    .tolist()
                )
        result = math.fsum(loss_values) / len(contexts)
        if cross_entropy_weight > 0.0:
            result += cross_entropy_weight * math.fsum(log_values) / len(contexts)
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite full-training ICEO objective")
    return result


@torch.no_grad()
def _split_loss_summary(
    problem: DifferentiableStochasticProblem,
    model: FullSupportICEOModel,
    catalog: Array,
    split: ObservedSplit,
    *,
    ridge: float,
    microbatch_size: int = 0,
) -> tuple[float, float]:
    dtype = model.catalog.dtype
    device = model.catalog.device
    contexts = torch.tensor(split.contexts, dtype=dtype, device=device)
    outcomes = torch.tensor(split.outcomes(catalog), dtype=dtype, device=device)
    values: list[float] = []
    for start, stop in _microbatch_ranges(len(contexts), microbatch_size):
        block = contexts[start:stop]
        losses = realized_loss_from_scenarios_torch(
            problem,
            block,
            model.catalog,
            model.probabilities(block),
            outcomes[start:stop],
            ridge=ridge,
        )
        values.extend(float(value) for value in losses.detach().cpu().tolist())
    loss_sum = math.fsum(float(value) for value in values)
    mean_loss = loss_sum / len(values)
    if not math.isfinite(loss_sum) or not math.isfinite(mean_loss):
        raise FloatingPointError("nonfinite ICEO validation loss")
    return loss_sum, mean_loss


def _iceo_optimizer_step(
    problem: DifferentiableStochasticProblem,
    model: FullSupportICEOModel,
    optimizer: torch.optim.Optimizer,
    contexts: torch.Tensor,
    standardized_contexts: torch.Tensor,
    outcomes: torch.Tensor,
    outcome_indices: torch.Tensor | None,
    *,
    ridge: float,
    config: ICEOTrainingConfig,
    step: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    if config.microbatch_size <= 0 or config.microbatch_size >= len(contexts):
        probabilities = model._probabilities_from_standardized_contexts(
            standardized_contexts
        )
        losses = realized_loss_from_scenarios_torch(
            problem,
            contexts,
            model.catalog,
            probabilities,
            outcomes,
            ridge=ridge,
        )
        if losses.shape != (len(contexts),):
            raise ValueError("realized_loss_torch must return one loss per observation")
        objective = losses.mean()
        if config.cross_entropy_weight > 0.0:
            if outcome_indices is None:
                raise RuntimeError("outcome indices are required for cross entropy")
            objective = objective + (
                config.cross_entropy_weight
                * _negative_log_likelihood(probabilities, outcome_indices)
            )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"nonfinite ICEO objective at step {step}")
        objective.backward()
    else:
        denominator = float(len(contexts))
        for start, stop in _microbatch_ranges(
            len(contexts),
            config.microbatch_size,
        ):
            probabilities = model._probabilities_from_standardized_contexts(
                standardized_contexts[start:stop]
            )
            losses = realized_loss_from_scenarios_torch(
                problem,
                contexts[start:stop],
                model.catalog,
                probabilities,
                outcomes[start:stop],
                ridge=ridge,
            )
            if losses.shape != (stop - start,):
                raise ValueError(
                    "realized_loss_torch must return one loss per observation"
                )
            objective = losses.sum() / denominator
            if config.cross_entropy_weight > 0.0:
                if outcome_indices is None:
                    raise RuntimeError("outcome indices are required for cross entropy")
                objective = objective + (
                    config.cross_entropy_weight
                    * _negative_log_likelihood_sum(
                        probabilities,
                        outcome_indices[start:stop],
                    )
                    / denominator
                )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"nonfinite ICEO objective at step {step}")
            objective.backward()
    if config.gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    for parameter in model.parameters():
        if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)):
            raise FloatingPointError(
                f"nonfinite or missing ICEO gradient at step {step}"
            )
    optimizer.step()
    model.project_identifiability_()


def _negative_log_likelihood(
    probabilities: torch.Tensor,
    outcome_indices: torch.Tensor,
) -> torch.Tensor:
    observed_probabilities = probabilities.gather(
        dim=1,
        index=outcome_indices[:, None],
    ).squeeze(1)
    return -torch.log(
        observed_probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    ).mean()


def _negative_log_likelihood_sum(
    probabilities: torch.Tensor,
    outcome_indices: torch.Tensor,
) -> torch.Tensor:
    observed_probabilities = probabilities.gather(
        dim=1,
        index=outcome_indices[:, None],
    ).squeeze(1)
    return -torch.log(
        observed_probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    ).sum()


def _microbatch_ranges(
    sample_size: int,
    microbatch_size: int,
) -> tuple[tuple[int, int], ...]:
    if microbatch_size <= 0 or microbatch_size >= sample_size:
        return ((0, sample_size),)
    return tuple(
        (start, min(start + microbatch_size, sample_size))
        for start in range(0, sample_size, microbatch_size)
    )


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


def _validate_problem_split(
    problem: DifferentiableStochasticProblem,
    catalog: Array,
    split: ObservedSplit,
) -> None:
    if split.contexts.shape[1] != problem.context_dim:
        raise ValueError("data and problem context dimensions differ")
    if np.asarray(catalog).shape[1] != problem.scenario_dim:
        raise ValueError("data and problem scenario dimensions differ")
    split.outcomes(catalog)
