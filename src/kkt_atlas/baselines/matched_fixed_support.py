"""Matched observed-support scenario reduction for KKT Atlas experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from kkt_atlas.atlas import PreparedAtlas, _derived_seed
from kkt_atlas.charts import fit_context_charts
from kkt_atlas.data import ObservedSplit
from kkt_atlas.frozen_replay import (
    FrozenAtlasReplayConfig,
    FrozenReplayCounts,
    FrozenReplayTimings,
    _frozen_chart_training_config,
    _train_heterogeneous_models,
)
from kkt_atlas.initialization import (
    ScenarioInitialization,
    cluster_projected_scenarios,
)
from kkt_atlas.probability_initialization import (
    ProbabilityHeadInitialization,
    fit_cluster_probability_head,
)
from kkt_atlas.problem import Array, DifferentiableStochasticProblem
from kkt_atlas.response_basis import (
    ResponseDecomposition,
    fit_response_decomposition,
)
from kkt_atlas.scenario_model import LocalScenarioModel
from kkt_atlas.training import TrainingConfig, TrainingResult


@dataclass(frozen=True)
class ObservedSupportInitialization:
    """Projected K-means initialization snapped to observed realizations."""

    centroid_initialization: ScenarioInitialization
    local_observation_indices: Array
    observed_scenarios: Array
    projected_observed_coordinates: Array
    cluster_assignments: Array

    def __post_init__(self) -> None:
        indices = np.asarray(self.local_observation_indices, dtype=np.int64).reshape(-1)
        scenarios = np.asarray(self.observed_scenarios, dtype=np.float64)
        coordinates = np.asarray(
            self.projected_observed_coordinates,
            dtype=np.float64,
        )
        assignments = np.asarray(
            self.cluster_assignments,
            dtype=np.int64,
        ).reshape(-1)
        support_size = self.centroid_initialization.support_size
        if indices.shape != (support_size,) or len(np.unique(indices)) != support_size:
            raise ValueError("selected observation indices must be distinct")
        if scenarios.ndim != 2 or len(scenarios) != support_size:
            raise ValueError("observed_scenarios must contain one row per atom")
        if coordinates.shape != self.centroid_initialization.coordinates.shape:
            raise ValueError("projected observed coordinates have the wrong shape")
        if np.any(~np.isfinite(scenarios)) or np.any(~np.isfinite(coordinates)):
            raise ValueError("selected scenarios and coordinates must be finite")
        if np.any(assignments < 0) or np.any(assignments >= support_size):
            raise ValueError("cluster assignments are outside the support")
        if not np.array_equal(
            np.bincount(assignments, minlength=support_size),
            self.centroid_initialization.counts,
        ):
            raise ValueError("cluster assignments do not match the centroid counts")
        immutable = []
        for values in (indices, scenarios, coordinates, assignments):
            copy = np.ascontiguousarray(values.copy())
            copy.setflags(write=False)
            immutable.append(copy)
        object.__setattr__(self, "local_observation_indices", immutable[0])
        object.__setattr__(self, "observed_scenarios", immutable[1])
        object.__setattr__(
            self,
            "projected_observed_coordinates",
            immutable[2],
        )
        object.__setattr__(self, "cluster_assignments", immutable[3])

    @property
    def support_size(self) -> int:
        return self.centroid_initialization.support_size

    def scenario_initialization(self) -> ScenarioInitialization:
        return ScenarioInitialization(
            coordinates=self.projected_observed_coordinates,
            counts=self.centroid_initialization.counts,
        )


class ObservedSupportLocalModel(LocalScenarioModel):
    """The Atlas probability head over fixed observed scenario locations."""

    def __init__(
        self,
        problem: DifferentiableStochasticProblem,
        decomposition: ResponseDecomposition,
        initialization: ObservedSupportInitialization,
        *,
        width: int,
        context_mean: Array,
        context_scale: Array,
        standardized_center: Array,
        probability_initialization: ProbabilityHeadInitialization | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            problem,
            decomposition,
            initialization.scenario_initialization(),
            width=width,
            context_mean=context_mean,
            context_scale=context_scale,
            standardized_center=standardized_center,
            probability_initialization=probability_initialization,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "fixed_observed_scenarios",
            torch.tensor(
                initialization.observed_scenarios,
                dtype=dtype,
                device=device,
            ),
        )
        self.scenario_coordinates.requires_grad_(False)

    def scenarios(self) -> torch.Tensor:
        return self.fixed_observed_scenarios


@dataclass(frozen=True)
class MatchedObservedSupportResult:
    """One fitted observed-support atlas and its construction record."""

    prepared_atlas: PreparedAtlas
    decompositions: tuple[ResponseDecomposition, ...]
    initializations: tuple[ObservedSupportInitialization, ...]
    global_observation_indices: tuple[Array, ...]
    training_results: tuple[TrainingResult, ...]
    training_configs: tuple[TrainingConfig, ...]
    timings: FrozenReplayTimings
    counts: FrozenReplayCounts

    def decisions(self, contexts: Array) -> Array:
        return self.prepared_atlas.decisions(contexts)


def snap_centroids_to_observed_scenarios(
    projected_coordinates: Array,
    observed_scenarios: Array,
    support_size: int,
    *,
    seed: int,
) -> ObservedSupportInitialization:
    """Choose the nearest observed realization within each K-means cluster."""

    coordinates = np.asarray(projected_coordinates, dtype=np.float64)
    scenarios = np.asarray(observed_scenarios, dtype=np.float64)
    if coordinates.ndim != 2 or len(coordinates) == 0:
        raise ValueError("projected_coordinates must be a nonempty matrix")
    if scenarios.ndim != 2 or len(scenarios) == 0:
        raise ValueError("observed_scenarios must be a nonempty matrix")
    if np.any(~np.isfinite(coordinates)) or np.any(~np.isfinite(scenarios)):
        raise ValueError("projected coordinates and scenarios must be finite")
    if len(coordinates) != len(scenarios):
        raise ValueError("projected coordinates and scenarios must align")
    clustering = cluster_projected_scenarios(
        coordinates,
        support_size,
        seed=seed,
    )
    centroid_initialization = clustering.initialization
    centers = centroid_initialization.coordinates
    labels = clustering.assignments
    squared_distances = ((coordinates[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

    selected = []
    for cluster in range(support_size):
        members = np.flatnonzero(labels == cluster)
        if len(members) == 0:
            raise RuntimeError("K-means produced an empty cluster")
        order = np.lexsort((members, squared_distances[members, cluster]))
        selected.append(int(members[int(order[0])]))
    indices = np.asarray(selected, dtype=np.int64)
    return ObservedSupportInitialization(
        centroid_initialization=centroid_initialization,
        local_observation_indices=indices,
        observed_scenarios=scenarios[indices],
        projected_observed_coordinates=coordinates[indices],
        cluster_assignments=labels,
    )


def fit_matched_observed_support_atlas(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: FrozenAtlasReplayConfig,
) -> MatchedObservedSupportResult:
    """Fit only local probability heads over matched observed supports."""

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
    partition_seconds = time.perf_counter() - phase_started

    decompositions = []
    initializations = []
    global_indices = []
    models = []
    training_configs = []
    chart_contexts = []
    chart_outcomes = []
    basis_seconds = 0.0
    initialization_seconds = 0.0
    for chart, specification in enumerate(config.charts):
        indices = charts.indices(chart)
        local_contexts = training.contexts[indices]
        local_outcomes = outcomes[indices]
        chart_contexts.append(local_contexts)
        chart_outcomes.append(local_outcomes)

        phase_started = time.perf_counter()
        decomposition = fit_response_decomposition(
            problem,
            charts.centers[chart],
            local_outcomes,
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
            local_outcomes,
            specification.width,
            problem,
        )
        initialization = snap_centroids_to_observed_scenarios(
            projected,
            local_outcomes,
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
        probability_initialization = None
        if config.probability_head_initialization == "training_cluster_multinomial_logit":
            centered_contexts = (local_contexts - charts.mean) / charts.scale - charts.standardized_centers[chart]
            probability_initialization = fit_cluster_probability_head(
                centered_contexts,
                initialization.cluster_assignments,
                specification.support_size,
                slope_penalty=config.training.slope_penalty,
                maximum_iterations=config.probability_head_maximum_iterations,
            )
        model = ObservedSupportLocalModel(
            problem,
            decomposition,
            initialization,
            width=specification.width,
            context_mean=charts.mean,
            context_scale=charts.scale,
            standardized_center=charts.standardized_centers[chart],
            probability_initialization=probability_initialization,
        )
        save_steps = tuple(
            sorted(
                {0, specification.stop_step}.union(
                    step
                    for step in config.training.save_steps
                    if step <= specification.stop_step
                )
            )
        )
        local_training = replace(
            _frozen_chart_training_config(
                config,
                chart=chart,
                specification=specification,
            ),
            save_steps=save_steps,
            coordinate_penalty=0.0,
            train_scenario_coordinates=False,
        )
        decompositions.append(decomposition)
        initializations.append(initialization)
        selected_global_indices = np.ascontiguousarray(
            indices[initialization.local_observation_indices],
            dtype=np.int64,
        )
        selected_global_indices.setflags(write=False)
        global_indices.append(selected_global_indices)
        models.append(model)
        training_configs.append(local_training)
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
    prepared = PreparedAtlas(
        problem=problem,
        charts=charts,
        ridge=config.ridge,
        models=tuple(models),
    )
    assembly_seconds = time.perf_counter() - assembly_started
    timings = FrozenReplayTimings(
        partition_seconds=float(partition_seconds),
        basis_seconds=float(basis_seconds),
        initialization_seconds=float(initialization_seconds),
        adam_update_seconds=float(trained.adam_update_seconds),
        checkpoint_scoring_seconds=float(trained.checkpoint_scoring_seconds),
        assembly_seconds=float(assembly_seconds),
        total_construction_seconds=float(time.perf_counter() - started),
    )
    return MatchedObservedSupportResult(
        prepared_atlas=prepared,
        decompositions=tuple(decompositions),
        initializations=tuple(initializations),
        global_observation_indices=tuple(global_indices),
        training_results=trained.results,
        training_configs=tuple(training_configs),
        timings=timings,
        counts=trained.counts,
    )
