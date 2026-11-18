"""Auditable replay of one frozen heterogeneous KKT Atlas."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from kkt_atlas.atlas import (
    CandidateLibrary,
    PartitionCandidateLibrary,
    RidgeCandidateLibrary,
    TrainedLocalCandidate,
    _derived_seed,
)
from kkt_atlas.charts import ContextCharts, fit_context_charts
from kkt_atlas.data import ObservedSplit
from kkt_atlas.initialization import cluster_projected_scenarios
from kkt_atlas.joint_training import (
    _clip_and_validate_chart_gradients,
    _dense_union_scenario_objective_tensor,
    _grouped_scenario_objective_tensor,
    _joint_scenario_objective_tensor,
    _make_chart_state,
)
from kkt_atlas.probability_initialization import fit_cluster_probability_head
from kkt_atlas.problem import Array, DifferentiableStochasticProblem
from kkt_atlas.response_basis import fit_response_decomposition
from kkt_atlas.scenario_model import LocalScenarioModel
from kkt_atlas.training import TrainingConfig, TrainingResult, _full_objective


@dataclass(frozen=True)
class FrozenChartSpecification:
    """The width, support, and stopping step frozen for one chart."""

    width: int
    support_size: int
    stop_step: int

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be positive")
        if self.support_size < 1:
            raise ValueError("support_size must be positive")
        if self.stop_step < 0:
            raise ValueError("stop_step must be nonnegative")


@dataclass(frozen=True)
class FrozenAtlasReplayConfig:
    """A fully frozen atlas structure with one specification per chart."""

    ridge: float
    charts: tuple[FrozenChartSpecification, ...]
    training: TrainingConfig
    minimum_chart_size: int = 10
    minimum_context_scale: float = 1e-10
    minimum_scenario_scale: float = 1e-8
    basis_method: str = "kkt"
    probability_head_initialization: str = "cluster_frequencies"
    execution_policy: str = "per_chart"
    seed_profile: str = "default"
    chart_batch_sizes: tuple[int, ...] = ()
    expected_partition_fingerprints: tuple[tuple[str, str], ...] = ()
    seed: int = 0
    probability_head_maximum_iterations: int = 500

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        if not self.charts:
            raise ValueError("at least one chart specification is required")
        if any(item.stop_step > self.training.steps for item in self.charts):
            raise ValueError("a chart stop_step exceeds the training budget")
        if self.probability_head_maximum_iterations < 1:
            raise ValueError("probability_head_maximum_iterations must be positive")
        if self.minimum_chart_size < 2:
            raise ValueError("minimum_chart_size must be at least two")
        if self.basis_method not in {"kkt", "pca", "random", "identity"}:
            raise ValueError("basis_method must be 'kkt', 'pca', 'random', or 'identity'")
        if self.execution_policy not in {
            "per_chart",
            "grouped",
            "dense_union",
        }:
            raise ValueError("execution_policy must be 'per_chart', 'grouped', or 'dense_union'")
        if self.probability_head_initialization not in {
            "cluster_frequencies",
            "training_cluster_multinomial_logit",
        }:
            raise ValueError(
                "probability_head_initialization must be 'cluster_frequencies' or 'training_cluster_multinomial_logit'"
            )
        if self.seed_profile not in {"default", "factorial_kkt_atlas"}:
            raise ValueError("seed_profile must be 'default' or 'factorial_kkt_atlas'")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in self.chart_batch_sizes
        ):
            raise TypeError("chart_batch_sizes must contain integers")
        batch_sizes = tuple(int(value) for value in self.chart_batch_sizes)
        if batch_sizes and len(batch_sizes) != len(self.charts):
            raise ValueError("chart_batch_sizes must contain one value per chart")
        if any(value < 1 for value in batch_sizes):
            raise ValueError("chart_batch_sizes must contain positive integers")
        object.__setattr__(self, "chart_batch_sizes", batch_sizes)
        fingerprint_names = [name for name, _ in self.expected_partition_fingerprints]
        if len(fingerprint_names) != len(set(fingerprint_names)):
            raise ValueError("partition fingerprint names must be unique")
        if any(not name or len(value) != 64 for name, value in self.expected_partition_fingerprints):
            raise ValueError("partition fingerprints must be named SHA-256 values")

    @property
    def chart_count(self) -> int:
        return len(self.charts)


@dataclass(frozen=True)
class FrozenReplayCounts:
    """Logical solve counts and actual reverse-mode calls during construction."""

    chart_partitions: int
    basis_solver_calls: int
    local_initializations: int
    update_solution_calls: int
    checkpoint_solution_calls: int
    training_solution_calls: int
    backward_calls: int
    chart_backward_participations: int
    optimizer_steps: int
    per_chart_training_solution_calls: tuple[int, ...]


@dataclass(frozen=True)
class FrozenReplayTimings:
    """Disjoint phase timers; validation is intentionally absent."""

    partition_seconds: float
    basis_seconds: float
    initialization_seconds: float
    adam_update_seconds: float
    checkpoint_scoring_seconds: float
    assembly_seconds: float
    total_construction_seconds: float


@dataclass(frozen=True)
class FrozenAtlasReplayResult:
    """The training-only candidate and its auditable construction record."""

    library: CandidateLibrary
    timings: FrozenReplayTimings
    counts: FrozenReplayCounts

    @property
    def training_seconds(self) -> float:
        """Return pure Adam-update time, excluding checkpoint scoring."""

        return self.timings.adam_update_seconds

    @property
    def construction_seconds(self) -> float:
        """Return total validation-free atlas construction time."""

        return self.timings.total_construction_seconds


@dataclass(frozen=True)
class _HeterogeneousTrainingResult:
    results: tuple[TrainingResult, ...]
    counts: FrozenReplayCounts
    adam_update_seconds: float
    checkpoint_scoring_seconds: float


def replay_frozen_heterogeneous_atlas(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: FrozenAtlasReplayConfig,
) -> FrozenAtlasReplayResult:
    """Construct exactly one frozen atlas without evaluating validation data.

    Every chart has its own representation width, support size, and stopping
    step. A chart leaves the joint computation graph after its stopping step.
    The returned timer covers chart construction, response decompositions,
    initialization, and parameter fitting; validation is a separate operation.
    """

    started = time.perf_counter()
    outcomes = training.outcomes(catalog)
    phase_started = time.perf_counter()
    charts = fit_context_charts(
        training.contexts,
        config.chart_count,
        seed=_derived_seed(config.seed, "charts", config.chart_count),
        minimum_scale=config.minimum_context_scale,
        minimum_chart_size=config.minimum_chart_size,
    )
    if config.expected_partition_fingerprints:
        expected = dict(config.expected_partition_fingerprints)
        actual = _partition_fingerprints(charts)
        if expected != actual:
            mismatches = sorted(name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name))
            raise ValueError(f"replayed context partition does not match the source partition: {mismatches}")
    partition_seconds = time.perf_counter() - phase_started
    chart_indices = tuple(charts.indices(chart) for chart in range(config.chart_count))
    chart_contexts = tuple(training.contexts[indices] for indices in chart_indices)
    chart_outcomes = tuple(outcomes[indices] for indices in chart_indices)

    decompositions = []
    initializations = []
    models = []
    training_configs = []
    basis_seconds = 0.0
    initialization_seconds = 0.0
    for chart, specification in enumerate(config.charts):
        phase_started = time.perf_counter()
        decomposition = fit_response_decomposition(
            problem,
            charts.centers[chart],
            chart_outcomes[chart],
            ridge=config.ridge,
            minimum_scenario_scale=config.minimum_scenario_scale,
            basis_method=config.basis_method,
            basis_seed=_derived_seed(
                config.seed,
                "basis",
                config.basis_method,
                config.chart_count,
                0,
                chart,
            ),
        )
        basis_seconds += time.perf_counter() - phase_started
        if specification.width > decomposition.maximum_width:
            raise ValueError(f"chart {chart} width exceeds the response-basis dimension")
        phase_started = time.perf_counter()
        projected = decomposition.coordinates(
            chart_outcomes[chart],
            specification.width,
            problem,
        )
        clustering = cluster_projected_scenarios(
            projected,
            specification.support_size,
            seed=_derived_seed(
                config.seed,
                "initialization",
                config.chart_count,
                0,
                chart,
                specification.width,
                specification.support_size,
            ),
        )
        initialization = clustering.initialization
        probability_initialization = None
        if config.probability_head_initialization == "training_cluster_multinomial_logit":
            centered_contexts = (chart_contexts[chart] - charts.mean) / charts.scale - charts.standardized_centers[
                chart
            ]
            probability_initialization = fit_cluster_probability_head(
                centered_contexts,
                clustering.assignments,
                specification.support_size,
                slope_penalty=config.training.slope_penalty,
                maximum_iterations=config.probability_head_maximum_iterations,
            )
        model = LocalScenarioModel(
            problem,
            decomposition,
            initialization,
            width=specification.width,
            context_mean=charts.mean,
            context_scale=charts.scale,
            standardized_center=charts.standardized_centers[chart],
            probability_initialization=probability_initialization,
        )
        decompositions.append(decomposition)
        initializations.append(initialization)
        models.append(model)
        training_configs.append(
            _frozen_chart_training_config(
                config,
                chart=chart,
                specification=specification,
            )
        )
        initialization_seconds += time.perf_counter() - phase_started

    trained = _train_heterogeneous_models(
        problem,
        models,
        chart_contexts,
        chart_outcomes,
        ridge=config.ridge,
        configs=training_configs,
        execution_policy=config.execution_policy,
    )
    assembly_started = time.perf_counter()
    local_candidates = []
    for chart, specification in enumerate(config.charts):
        snapshot = trained.results[chart].checkpoint(specification.stop_step)
        local_candidates.append(
            (
                TrainedLocalCandidate(
                    chart=chart,
                    ridge=config.ridge,
                    width=specification.width,
                    support_size=specification.support_size,
                    checkpoint=snapshot,
                    decomposition=decompositions[chart],
                    initialization=initializations[chart],
                ),
            )
        )
    library = CandidateLibrary(
        problem=problem,
        partitions=(
            PartitionCandidateLibrary(
                charts=charts,
                ridge_libraries=(
                    RidgeCandidateLibrary(
                        ridge=config.ridge,
                        local_candidates=tuple(local_candidates),
                    ),
                ),
            ),
        ),
        failures=(),
    )
    assembly_seconds = time.perf_counter() - assembly_started
    total_seconds = time.perf_counter() - started
    timings = FrozenReplayTimings(
        partition_seconds=float(partition_seconds),
        basis_seconds=float(basis_seconds),
        initialization_seconds=float(initialization_seconds),
        adam_update_seconds=float(trained.adam_update_seconds),
        checkpoint_scoring_seconds=float(trained.checkpoint_scoring_seconds),
        assembly_seconds=float(assembly_seconds),
        total_construction_seconds=float(total_seconds),
    )
    return FrozenAtlasReplayResult(
        library=library,
        timings=timings,
        counts=trained.counts,
    )


def _frozen_chart_training_config(
    config: FrozenAtlasReplayConfig,
    *,
    chart: int,
    specification: FrozenChartSpecification,
) -> TrainingConfig:
    template = config.training
    save_steps = (0,) if specification.stop_step == 0 else (0, specification.stop_step)
    training_seed = (
        _derived_seed(
            config.seed,
            "kkt_atlas",
            "training",
            config.chart_count,
            chart,
            specification.width,
            specification.support_size,
        )
        if config.seed_profile == "factorial_kkt_atlas"
        else _derived_seed(
            template.seed,
            "training",
            config.chart_count,
            0,
            chart,
            specification.width,
            specification.support_size,
        )
    )
    return TrainingConfig(
        steps=specification.stop_step,
        learning_rate=template.learning_rate,
        save_steps=save_steps,
        batch_size=(
            config.chart_batch_sizes[chart]
            if config.chart_batch_sizes
            else template.batch_size
        ),
        coordinate_penalty=template.coordinate_penalty,
        slope_penalty=template.slope_penalty,
        gradient_clip=template.gradient_clip,
        train_scenario_coordinates=template.train_scenario_coordinates,
        train_probability_slopes=template.train_probability_slopes,
        seed=training_seed,
    )


def _array_sha256(values: Array) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _partition_fingerprints(charts: ContextCharts) -> dict[str, str]:
    components = {
        "context_mean_sha256": _array_sha256(charts.mean),
        "context_scale_sha256": _array_sha256(charts.scale),
        "standardized_centers_sha256": _array_sha256(charts.standardized_centers),
        "centers_sha256": _array_sha256(charts.centers),
        "training_assignments_sha256": _array_sha256(charts.assignments),
    }
    components["combined_sha256"] = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return components


def _train_heterogeneous_models(
    problem: DifferentiableStochasticProblem,
    models: Sequence[LocalScenarioModel],
    contexts: Sequence[Array],
    outcomes: Sequence[Array],
    *,
    ridge: float,
    configs: Sequence[TrainingConfig],
    execution_policy: str = "per_chart",
) -> _HeterogeneousTrainingResult:
    if not (len(models) == len(contexts) == len(outcomes) == len(configs)):
        raise ValueError("models, contexts, outcomes, and configs must align")
    if not models:
        raise ValueError("at least one local model is required")
    learning_rate = configs[0].learning_rate
    if any(item.learning_rate != learning_rate for item in configs[1:]):
        raise ValueError("all charts must use the same learning rate")
    if execution_policy not in {"per_chart", "grouped", "dense_union"}:
        raise ValueError("execution_policy must be 'per_chart', 'grouped', or 'dense_union'")

    states = tuple(
        _make_chart_state(problem, model, chart_contexts, chart_outcomes, item)
        for model, chart_contexts, chart_outcomes, item in zip(
            models,
            contexts,
            outcomes,
            configs,
            strict=True,
        )
    )
    parameters = tuple(parameter for state in states for parameter in state.parameters)
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        foreach=False,
        fused=False,
    )

    checkpoint_scoring_seconds = 0.0
    for state in states:
        phase_started = time.perf_counter()
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
        checkpoint_scoring_seconds += time.perf_counter() - phase_started

    maximum_step = max(item.steps for item in configs)
    backward_calls = 0
    adam_update_seconds = 0.0
    for step in range(1, maximum_step + 1):
        active_states = tuple(state for state in states if step <= state.config.steps)
        phase_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        objective_builders = {
            "per_chart": _joint_scenario_objective_tensor,
            "grouped": _grouped_scenario_objective_tensor,
            "dense_union": _dense_union_scenario_objective_tensor,
        }
        objective_builder = objective_builders[execution_policy]
        objective = objective_builder(
            problem,
            active_states,
            ridge=ridge,
            step=step,
        )
        objective.backward()
        backward_calls += 1
        for state in active_states:
            _clip_and_validate_chart_gradients(state, step=step)
        optimizer.step()
        for state in active_states:
            state.model.project_identifiability_()
        adam_update_seconds += time.perf_counter() - phase_started

        for state in active_states:
            if step not in state.config.save_steps:
                continue
            phase_started = time.perf_counter()
            full_objective, operational = _full_objective(
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
                    training_objective=full_objective,
                    mean_operational_loss=operational,
                )
            )
            checkpoint_scoring_seconds += time.perf_counter() - phase_started

    results = tuple(
        TrainingResult(
            checkpoints=tuple(state.checkpoints),
            final_step=state.config.steps,
        )
        for state in states
    )
    stop_steps = tuple(item.steps for item in configs)
    checkpoint_calls = sum(len(item.save_steps) for item in configs)
    update_calls = sum(stop_steps)
    per_chart_calls = tuple(
        item.steps + len(item.save_steps)
        for item in configs
    )
    counts = FrozenReplayCounts(
        chart_partitions=1,
        basis_solver_calls=len(states),
        local_initializations=len(states),
        update_solution_calls=update_calls,
        checkpoint_solution_calls=checkpoint_calls,
        training_solution_calls=update_calls + checkpoint_calls,
        backward_calls=backward_calls,
        chart_backward_participations=update_calls,
        optimizer_steps=maximum_step,
        per_chart_training_solution_calls=per_chart_calls,
    )
    return _HeterogeneousTrainingResult(
        results=results,
        counts=counts,
        adam_update_seconds=float(adam_update_seconds),
        checkpoint_scoring_seconds=float(checkpoint_scoring_seconds),
    )
