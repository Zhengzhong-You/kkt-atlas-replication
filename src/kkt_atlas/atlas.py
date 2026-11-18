"""Training-only candidate construction and held-out selection of one atlas."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import torch

from kkt_atlas.charts import ContextCharts, fit_context_charts
from kkt_atlas.data import ObservedSplit
from kkt_atlas.initialization import (
    ScenarioClustering,
    ScenarioInitialization,
    cluster_projected_scenarios,
)
from kkt_atlas.joint_training import (
    train_local_models_jointly,
    train_local_models_with_grouped_optimizer,
)
from kkt_atlas.probability_initialization import (
    ProbabilityHeadInitialization,
    fit_cluster_probability_head,
)
from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
    realized_loss_from_scenarios_torch,
)
from kkt_atlas.response_basis import ResponseDecomposition, fit_response_decomposition
from kkt_atlas.scenario_model import LocalModelSnapshot, LocalScenarioModel
from kkt_atlas.training import TrainingConfig, TrainingResult, train_local_model
from kkt_atlas.vectorized_training import train_local_models_vectorized

if TYPE_CHECKING:
    from kkt_atlas.automatic_joint_structure_v2 import JointPartitionSelection


@dataclass(frozen=True)
class AtlasSearchConfig:
    chart_counts: tuple[int, ...]
    ridges: tuple[float, ...]
    widths: tuple[int, ...]
    support_sizes: tuple[int, ...]
    training: TrainingConfig
    minimum_chart_size: int = 10
    minimum_context_scale: float = 1e-10
    minimum_scenario_scale: float = 1e-8
    include_width_plus_one: bool = True
    basis_method: str = "kkt"
    spectral_retained_ratio: float | None = None
    probability_head_initialization: str = "cluster_frequencies"
    seed: int = 0
    probability_head_maximum_iterations: int = 500

    def __post_init__(self) -> None:
        for name in ("chart_counts", "widths", "support_sizes"):
            values = tuple(sorted(set(int(value) for value in getattr(self, name))))
            if not values or values[0] < 1:
                raise ValueError(f"{name} must contain positive integers")
            object.__setattr__(self, name, values)
        ridges = tuple(float(value) for value in self.ridges)
        if not ridges or any(not math.isfinite(value) or value < 0.0 for value in ridges):
            raise ValueError("ridges must contain nonnegative finite values")
        object.__setattr__(self, "ridges", ridges)
        if self.minimum_chart_size < 2:
            raise ValueError("minimum_chart_size must be at least two")
        if not isinstance(self.include_width_plus_one, bool):
            raise TypeError("include_width_plus_one must be a boolean")
        if self.basis_method not in {"kkt", "pca", "random", "identity"}:
            raise ValueError("basis_method must be 'kkt', 'pca', 'random', or 'identity'")
        if self.spectral_retained_ratio is not None:
            ratio = float(self.spectral_retained_ratio)
            if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
                raise ValueError("spectral_retained_ratio must lie in (0,1)")
            if self.basis_method != "kkt":
                raise ValueError("spectral_retained_ratio requires basis_method='kkt'")
            object.__setattr__(self, "spectral_retained_ratio", ratio)
        if self.probability_head_initialization not in {
            "cluster_frequencies",
            "training_cluster_multinomial_logit",
        }:
            raise ValueError(
                "probability_head_initialization must be 'cluster_frequencies' or 'training_cluster_multinomial_logit'"
            )
        if self.probability_head_maximum_iterations < 1:
            raise ValueError("probability_head_maximum_iterations must be positive")


@dataclass(frozen=True)
class CandidateFailure:
    chart_count: int
    ridge: float | None
    chart: int | None
    width: int | None
    support_size: int | None
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class TrainedLocalCandidate:
    chart: int
    ridge: float
    width: int
    support_size: int
    checkpoint: LocalModelSnapshot
    decomposition: ResponseDecomposition
    initialization: ScenarioInitialization

    def make_model(
        self,
        problem: DifferentiableStochasticProblem,
        charts: ContextCharts,
    ) -> LocalScenarioModel:
        model = LocalScenarioModel(
            problem,
            self.decomposition,
            self.initialization,
            width=self.width,
            context_mean=charts.mean,
            context_scale=charts.scale,
            standardized_center=charts.standardized_centers[self.chart],
        )
        model.load_snapshot(self.checkpoint)
        model.eval()
        return model


@dataclass(frozen=True)
class RidgeCandidateLibrary:
    ridge: float
    local_candidates: tuple[tuple[TrainedLocalCandidate, ...], ...]


@dataclass(frozen=True)
class PartitionCandidateLibrary:
    charts: ContextCharts
    ridge_libraries: tuple[RidgeCandidateLibrary, ...]


@dataclass(frozen=True)
class CandidateLibrary:
    """Frozen training-generated models; no validation outcomes were used."""

    problem: DifferentiableStochasticProblem
    partitions: tuple[PartitionCandidateLibrary, ...]
    failures: tuple[CandidateFailure, ...]


@dataclass(frozen=True)
class SelectedLocalModel:
    candidate: TrainedLocalCandidate
    validation_loss_sum: float
    validation_count: int


@dataclass(frozen=True)
class LocalCandidateValidation:
    """Validation score of one training-generated local candidate."""

    chart_count: int
    ridge: float
    chart: int
    width: int
    support_size: int
    checkpoint_step: int
    validation_loss_sum: float
    validation_count: int
    validation_indices: tuple[int, ...]
    validation_losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.chart_count < 1:
            raise ValueError("chart_count must be positive")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        if self.chart < 0 or self.chart >= self.chart_count:
            raise ValueError("chart must index the declared partition")
        if self.width < 1 or self.support_size < 1:
            raise ValueError("width and support_size must be positive")
        if self.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be nonnegative")
        if not math.isfinite(self.validation_loss_sum):
            raise ValueError("validation_loss_sum must be finite")
        if self.validation_count < 1:
            raise ValueError("validation_count must be positive")
        if len(self.validation_indices) != self.validation_count:
            raise ValueError("validation_indices must match validation_count")
        if len(set(self.validation_indices)) != self.validation_count:
            raise ValueError("validation_indices must be unique")
        if any(index < 0 for index in self.validation_indices):
            raise ValueError("validation_indices must be nonnegative")
        if len(self.validation_losses) != self.validation_count:
            raise ValueError("validation_losses must match validation_count")
        if any(not math.isfinite(loss) for loss in self.validation_losses):
            raise ValueError("validation_losses must be finite")
        if not math.isclose(
            self.validation_loss_sum,
            math.fsum(self.validation_losses),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("validation_losses must sum to validation_loss_sum")

    @property
    def validation_mean_loss(self) -> float:
        return self.validation_loss_sum / self.validation_count


@dataclass(frozen=True)
class FittedAtlas:
    problem: DifferentiableStochasticProblem
    charts: ContextCharts
    ridge: float
    local_models: tuple[SelectedLocalModel, ...]
    validation_mean_loss: float

    def __post_init__(self) -> None:
        if len(self.local_models) != self.charts.chart_count:
            raise ValueError("one selected local model is required per chart")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        if not math.isfinite(self.validation_mean_loss):
            raise ValueError("validation_mean_loss must be finite")
        for chart, selected in enumerate(self.local_models):
            if selected.candidate.chart != chart:
                raise ValueError("selected local models must follow chart order")
            if selected.candidate.ridge != self.ridge:
                raise ValueError("all selected local models must share atlas ridge")

    @property
    def chart_count(self) -> int:
        return self.charts.chart_count

    @property
    def total_support(self) -> int:
        return sum(item.candidate.support_size for item in self.local_models)

    @property
    def total_scenario_coordinates(self) -> int:
        return sum(item.candidate.support_size * item.candidate.width for item in self.local_models)

    @property
    def total_width(self) -> int:
        return sum(item.candidate.width for item in self.local_models)

    def prepare(self) -> PreparedAtlas:
        """Instantiate the selected local models once for repeated deployment."""

        return PreparedAtlas(
            problem=self.problem,
            charts=self.charts,
            ridge=self.ridge,
            models=tuple(selected.candidate.make_model(self.problem, self.charts) for selected in self.local_models),
        )

    def decisions(self, contexts: Array) -> Array:
        return self.prepare().decisions(contexts)


@dataclass(frozen=True)
class PreparedAtlas:
    """A selected atlas whose local models are loaded for repeated queries."""

    problem: DifferentiableStochasticProblem
    charts: ContextCharts
    ridge: float
    models: tuple[LocalScenarioModel, ...]

    def __post_init__(self) -> None:
        if len(self.models) != self.charts.chart_count:
            raise ValueError("one prepared model is required per chart")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")

    def decisions(self, contexts: Array) -> Array:
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
        for chart, model in enumerate(self.models):
            indices = np.flatnonzero(assignments == chart)
            if len(indices) == 0:
                continue
            with torch.no_grad():
                chart_decisions = model.decisions(
                    torch.tensor(values[indices], dtype=torch.float64),
                    ridge=self.ridge,
                )
            decisions[indices] = chart_decisions.cpu().numpy()
        if np.any(~np.isfinite(decisions)):
            raise FloatingPointError("atlas returned a nonfinite decision")
        return decisions


def build_candidate_library(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: AtlasSearchConfig,
) -> CandidateLibrary:
    """Build every candidate using training observations only."""

    outcomes = training.outcomes(catalog)
    partitions: list[PartitionCandidateLibrary] = []
    failures: list[CandidateFailure] = []
    for chart_count in config.chart_counts:
        try:
            charts = fit_context_charts(
                training.contexts,
                chart_count,
                seed=_derived_seed(config.seed, "charts", chart_count),
                minimum_scale=config.minimum_context_scale,
                minimum_chart_size=config.minimum_chart_size,
            )
        except Exception as exc:
            failures.append(
                _failure(
                    chart_count=chart_count,
                    ridge=None,
                    chart=None,
                    width=None,
                    support_size=None,
                    stage="chart_partition",
                    exc=exc,
                )
            )
            continue

        ridge_libraries: list[RidgeCandidateLibrary] = []
        for ridge_index, ridge in enumerate(config.ridges):
            local_libraries: list[tuple[TrainedLocalCandidate, ...]] = []
            ridge_valid = True
            for chart in range(chart_count):
                indices = charts.indices(chart)
                chart_outcomes = outcomes[indices]
                chart_contexts = training.contexts[indices]
                try:
                    decomposition = fit_response_decomposition(
                        problem,
                        charts.centers[chart],
                        chart_outcomes,
                        ridge=ridge,
                        minimum_scenario_scale=config.minimum_scenario_scale,
                        basis_method=config.basis_method,
                        basis_seed=_derived_seed(
                            config.seed,
                            "basis",
                            config.basis_method,
                            chart_count,
                            ridge_index,
                            chart,
                        ),
                    )
                except Exception as exc:
                    failures.append(
                        _failure(
                            chart_count=chart_count,
                            ridge=ridge,
                            chart=chart,
                            width=None,
                            support_size=None,
                            stage="response_decomposition",
                            exc=exc,
                        )
                    )
                    ridge_valid = False
                    break

                candidate_widths = config.widths
                if config.spectral_retained_ratio is not None:
                    candidate_widths = (
                        decomposition.select_width_by_spectral_ratio(
                            config.spectral_retained_ratio,
                        ).width,
                    )

                trained: list[TrainedLocalCandidate] = []
                for width in candidate_widths:
                    if width > decomposition.maximum_width:
                        continue
                    projected = decomposition.coordinates(
                        chart_outcomes,
                        width,
                        problem,
                    )
                    support_grid = set(config.support_sizes)
                    if config.include_width_plus_one:
                        support_grid.add(width + 1)
                    ordered_support_grid = sorted(support_grid)
                    for support_size in ordered_support_grid:
                        try:
                            clustering = cluster_projected_scenarios(
                                projected,
                                support_size,
                                seed=_derived_seed(
                                    config.seed,
                                    "initialization",
                                    chart_count,
                                    ridge_index,
                                    chart,
                                    width,
                                    support_size,
                                ),
                            )
                            initialization = clustering.initialization
                            probability_initialization = _fit_probability_initialization(
                                config,
                                charts,
                                chart,
                                chart_contexts,
                                clustering,
                            )
                            model = LocalScenarioModel(
                                problem,
                                decomposition,
                                initialization,
                                width=width,
                                context_mean=charts.mean,
                                context_scale=charts.scale,
                                standardized_center=charts.standardized_centers[chart],
                                probability_initialization=probability_initialization,
                            )
                            result = train_local_model(
                                problem,
                                model,
                                chart_contexts,
                                chart_outcomes,
                                ridge=ridge,
                                config=TrainingConfig(
                                    steps=config.training.steps,
                                    learning_rate=config.training.learning_rate,
                                    save_steps=config.training.save_steps,
                                    batch_size=config.training.batch_size,
                                    coordinate_penalty=config.training.coordinate_penalty,
                                    slope_penalty=config.training.slope_penalty,
                                    gradient_clip=config.training.gradient_clip,
                                    train_scenario_coordinates=(config.training.train_scenario_coordinates),
                                    train_probability_slopes=(config.training.train_probability_slopes),
                                    seed=_derived_seed(
                                        config.training.seed,
                                        "training",
                                        chart_count,
                                        ridge_index,
                                        chart,
                                        width,
                                        support_size,
                                    ),
                                ),
                            )
                            trained.extend(
                                TrainedLocalCandidate(
                                    chart=chart,
                                    ridge=ridge,
                                    width=width,
                                    support_size=support_size,
                                    checkpoint=checkpoint,
                                    decomposition=decomposition,
                                    initialization=initialization,
                                )
                                for checkpoint in result.checkpoints
                            )
                        except Exception as exc:
                            failures.append(
                                _failure(
                                    chart_count=chart_count,
                                    ridge=ridge,
                                    chart=chart,
                                    width=width,
                                    support_size=support_size,
                                    stage="local_training",
                                    exc=exc,
                                )
                            )
                if not trained:
                    ridge_valid = False
                    failures.append(
                        CandidateFailure(
                            chart_count=chart_count,
                            ridge=ridge,
                            chart=chart,
                            width=None,
                            support_size=None,
                            stage="empty_local_library",
                            error_type="NoValidCandidate",
                            message="no local model survived training-only requirements",
                        )
                    )
                    break
                local_libraries.append(tuple(trained))
            if ridge_valid:
                ridge_libraries.append(
                    RidgeCandidateLibrary(
                        ridge=ridge,
                        local_candidates=tuple(local_libraries),
                    )
                )
        if ridge_libraries:
            partitions.append(
                PartitionCandidateLibrary(
                    charts=charts,
                    ridge_libraries=tuple(ridge_libraries),
                )
            )
    if not partitions:
        raise RuntimeError("training produced no complete atlas candidate")
    return CandidateLibrary(
        problem=problem,
        partitions=tuple(partitions),
        failures=tuple(failures),
    )


def build_fixed_candidate_library_jointly(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: AtlasSearchConfig,
) -> CandidateLibrary:
    """Build one fixed atlas structure using joint independent-chart updates.

    The execution shortcut is deliberately separate from the general search
    builder. It accepts exactly one chart count, ridge, width, and effective
    support size, so one failed chart invalidates the same complete atlas that
    the sequential builder would invalidate.
    """

    return _build_fixed_candidate_library_grouped(
        problem,
        training,
        catalog,
        config,
        training_function=train_local_models_jointly,
    )


def build_fixed_candidate_library_grouped_optimizer(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: AtlasSearchConfig,
) -> CandidateLibrary:
    """Build one fixed atlas with sequential graphs and one Adam optimizer."""

    return _build_fixed_candidate_library_grouped(
        problem,
        training,
        catalog,
        config,
        training_function=train_local_models_with_grouped_optimizer,
    )


def build_fixed_candidate_library_vectorized(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: AtlasSearchConfig,
) -> CandidateLibrary:
    """Build one fixed atlas structure with a vectorized full-batch update."""

    return _build_fixed_candidate_library_grouped(
        problem,
        training,
        catalog,
        config,
        training_function=train_local_models_vectorized,
    )


def build_training_selected_candidate_library(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    structure: JointPartitionSelection,
    *,
    ridge: float,
    training_config: TrainingConfig,
    global_sample_presentations_per_step: int,
    seed: int = 0,
    probability_head_initialization: str = "cluster_frequencies",
    probability_head_maximum_iterations: int = 2_000,
    training_function: Callable[..., tuple[TrainingResult, ...]] = (
        train_local_models_jointly
    ),
) -> CandidateLibrary:
    """Train one heterogeneous structure selected from training data.

    The chart partition, response dimensions, and support sizes are taken
    directly from ``structure``. Consequently, the returned library varies
    only over the checkpoints in ``training_config``; subsequent validation
    cannot change ``B``, ``q_b``, or ``K_b``. The global sample-presentation
    budget is divided deterministically across charts, including when it is
    not divisible by the number of charts.
    """

    ridge_value = float(ridge)
    if not math.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    if not structure.eligible:
        raise ValueError("the training-selected structure must be eligible")
    if probability_head_initialization not in {
        "cluster_frequencies",
        "training_cluster_multinomial_logit",
    }:
        raise ValueError("unsupported probability-head initialization")
    if probability_head_maximum_iterations < 1:
        raise ValueError("probability_head_maximum_iterations must be positive")
    chart_count = structure.chart_count
    charts = structure.partition.context_charts
    if charts.chart_count != chart_count:
        raise ValueError("the selected partition has an inconsistent chart count")
    if len(structure.charts) != chart_count:
        raise ValueError("the selected structure must contain one record per chart")
    if len(structure.partition.decompositions) != chart_count:
        raise ValueError("the selected structure must contain one basis per chart")

    outcomes = training.outcomes(catalog)
    if len(charts.assignments) != len(training.contexts):
        raise ValueError("the selected partition does not match the training sample")
    batch_sizes = balanced_local_batch_sizes(
        global_sample_presentations_per_step,
        chart_count,
    )
    models: list[LocalScenarioModel] = []
    initializations: list[ScenarioInitialization] = []
    configurations: list[TrainingConfig] = []
    chart_contexts: list[Array] = []
    chart_outcomes: list[Array] = []

    for chart, local in enumerate(structure.charts):
        if local.chart != chart:
            raise ValueError("selected chart records must follow chart order")
        if not local.gate_eligible:
            raise ValueError("every retained chart must pass the training-only gate")
        width = int(local.representation_dim)
        support_size = int(local.selected_support_size)
        if width < 1 or support_size < 1:
            raise ValueError("selected widths and support sizes must be positive")
        if support_size > local.protocol_support_cap:
            raise ValueError("selected support exceeds the declared protocol cap")
        decomposition = structure.partition.decompositions[chart]
        if width > decomposition.maximum_width:
            raise ValueError("selected width exceeds the response decomposition")
        indices = charts.indices(chart)
        if batch_sizes[chart] > len(indices):
            raise ValueError(
                "the global sample-presentation budget exceeds a chart sample"
            )
        contexts = training.contexts[indices]
        realized = outcomes[indices]
        projected = decomposition.coordinates(realized, width, problem)
        initialization_seed, training_seed = training_selected_chart_seeds(
            seed,
            chart_count=chart_count,
            chart=chart,
            width=width,
            support_size=support_size,
        )
        clustering = cluster_projected_scenarios(
            projected,
            support_size,
            seed=initialization_seed,
        )
        initialization = clustering.initialization
        probability_initialization = None
        if probability_head_initialization == "training_cluster_multinomial_logit":
            centered_contexts = (
                (contexts - charts.mean) / charts.scale
                - charts.standardized_centers[chart]
            )
            probability_initialization = fit_cluster_probability_head(
                centered_contexts,
                clustering.assignments,
                support_size,
                slope_penalty=training_config.slope_penalty,
                maximum_iterations=probability_head_maximum_iterations,
            )
        models.append(
            LocalScenarioModel(
                problem,
                decomposition,
                initialization,
                width=width,
                context_mean=charts.mean,
                context_scale=charts.scale,
                standardized_center=charts.standardized_centers[chart],
                probability_initialization=probability_initialization,
            )
        )
        initializations.append(initialization)
        chart_contexts.append(contexts)
        chart_outcomes.append(realized)
        configurations.append(
            TrainingConfig(
                steps=training_config.steps,
                learning_rate=training_config.learning_rate,
                save_steps=training_config.save_steps,
                batch_size=batch_sizes[chart],
                coordinate_penalty=training_config.coordinate_penalty,
                slope_penalty=training_config.slope_penalty,
                gradient_clip=training_config.gradient_clip,
                train_scenario_coordinates=(
                    training_config.train_scenario_coordinates
                ),
                train_probability_slopes=training_config.train_probability_slopes,
                seed=training_seed,
            )
        )

    results = training_function(
        problem,
        models,
        tuple(chart_contexts),
        tuple(chart_outcomes),
        ridge=ridge_value,
        configs=tuple(configurations),
    )
    if len(results) != chart_count:
        raise RuntimeError("training must return one result per selected chart")
    local_candidates = tuple(
        tuple(
            TrainedLocalCandidate(
                chart=chart,
                ridge=ridge_value,
                width=int(structure.charts[chart].representation_dim),
                support_size=int(structure.charts[chart].selected_support_size),
                checkpoint=checkpoint,
                decomposition=structure.partition.decompositions[chart],
                initialization=initializations[chart],
            )
            for checkpoint in result.checkpoints
        )
        for chart, result in enumerate(results)
    )
    if any(not candidates for candidates in local_candidates):
        raise RuntimeError("training returned an empty checkpoint path")
    return CandidateLibrary(
        problem=problem,
        partitions=(
            PartitionCandidateLibrary(
                charts=charts,
                ridge_libraries=(
                    RidgeCandidateLibrary(
                        ridge=ridge_value,
                        local_candidates=local_candidates,
                    ),
                ),
            ),
        ),
        failures=(),
    )


def balanced_local_batch_sizes(
    global_sample_presentations_per_step: int,
    chart_count: int,
) -> tuple[int, ...]:
    """Divide an integer global batch budget across all charts exactly."""

    if isinstance(global_sample_presentations_per_step, bool) or not isinstance(
        global_sample_presentations_per_step,
        Integral,
    ):
        raise TypeError("global_sample_presentations_per_step must be an integer")
    total = int(global_sample_presentations_per_step)
    if total < 1:
        raise ValueError("global_sample_presentations_per_step must be positive")
    if isinstance(chart_count, bool) or not isinstance(chart_count, Integral):
        raise TypeError("chart_count must be an integer")
    count = int(chart_count)
    if count < 1:
        raise ValueError("chart_count must be positive")
    if total < count:
        raise ValueError(
            "global sample presentations must provide at least one observation "
            "per chart and step"
        )
    quotient, remainder = divmod(total, count)
    result = tuple(
        quotient + int(chart < remainder)
        for chart in range(count)
    )
    if sum(result) != total or min(result) < 1:
        raise RuntimeError("balanced batch-size accounting failed")
    return result


def training_selected_chart_seeds(
    seed: int,
    *,
    chart_count: int,
    chart: int,
    width: int,
    support_size: int,
) -> tuple[int, int]:
    """Return the initialization and Adam seeds for one frozen chart path."""

    initialization_seed = _derived_seed(
        seed,
        "initialization",
        chart_count,
        0,
        chart,
        width,
        support_size,
    )
    training_seed = _derived_seed(
        seed,
        "kkt_atlas",
        "training",
        chart_count,
        chart,
        width,
        support_size,
    )
    return initialization_seed, training_seed


def _build_fixed_candidate_library_grouped(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: AtlasSearchConfig,
    *,
    training_function: Callable[..., tuple[TrainingResult, ...]],
) -> CandidateLibrary:
    _validate_fixed_joint_structure(config)
    chart_count = config.chart_counts[0]
    failures: list[CandidateFailure] = []
    try:
        charts = fit_context_charts(
            training.contexts,
            chart_count,
            seed=_derived_seed(config.seed, "charts", chart_count),
            minimum_scale=config.minimum_context_scale,
            minimum_chart_size=config.minimum_chart_size,
        )
    except Exception as exc:
        failures.append(
            _failure(
                chart_count=chart_count,
                ridge=None,
                chart=None,
                width=None,
                support_size=None,
                stage="chart_partition",
                exc=exc,
            )
        )
        raise RuntimeError("training produced no complete atlas candidate") from exc

    ridge = config.ridges[0]
    ridge_library = _build_joint_ridge_library(
        problem=problem,
        training=training,
        outcomes=training.outcomes(catalog),
        charts=charts,
        chart_count=chart_count,
        ridge=ridge,
        ridge_index=0,
        config=config,
        failures=failures,
        training_function=training_function,
    )
    if ridge_library is None:
        raise RuntimeError("training produced no complete atlas candidate")
    return CandidateLibrary(
        problem=problem,
        partitions=(
            PartitionCandidateLibrary(
                charts=charts,
                ridge_libraries=(ridge_library,),
            ),
        ),
        failures=tuple(failures),
    )


def _validate_fixed_joint_structure(config: AtlasSearchConfig) -> None:
    if config.spectral_retained_ratio is not None:
        raise ValueError(
            "fixed joint construction requires an explicit width and does not accept spectral_retained_ratio"
        )
    if len(config.chart_counts) != 1 or len(config.ridges) != 1 or len(config.widths) != 1:
        raise ValueError("joint construction requires one chart count, ridge, and width")
    support_grid = set(config.support_sizes)
    if config.include_width_plus_one:
        support_grid.add(config.widths[0] + 1)
    if len(support_grid) != 1:
        raise ValueError("joint construction requires one effective support size")


def _build_joint_ridge_library(
    *,
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    outcomes: Array,
    charts: ContextCharts,
    chart_count: int,
    ridge: float,
    ridge_index: int,
    config: AtlasSearchConfig,
    failures: list[CandidateFailure],
    training_function: Callable[..., tuple[TrainingResult, ...]],
) -> RidgeCandidateLibrary | None:
    """Train equal-structure chart candidates with one Adam update per step."""

    chart_indices = tuple(charts.indices(chart) for chart in range(chart_count))
    chart_contexts = tuple(training.contexts[indices] for indices in chart_indices)
    chart_outcomes = tuple(outcomes[indices] for indices in chart_indices)
    decompositions: list[ResponseDecomposition] = []
    for chart in range(chart_count):
        try:
            decomposition = fit_response_decomposition(
                problem,
                charts.centers[chart],
                chart_outcomes[chart],
                ridge=ridge,
                minimum_scenario_scale=config.minimum_scenario_scale,
                basis_method=config.basis_method,
                basis_seed=_derived_seed(
                    config.seed,
                    "basis",
                    config.basis_method,
                    chart_count,
                    ridge_index,
                    chart,
                ),
            )
        except Exception as exc:
            failures.append(
                _failure(
                    chart_count=chart_count,
                    ridge=ridge,
                    chart=chart,
                    width=None,
                    support_size=None,
                    stage="response_decomposition",
                    exc=exc,
                )
            )
            return None
        decompositions.append(decomposition)

    trained_by_chart: list[list[TrainedLocalCandidate]] = [[] for _ in range(chart_count)]
    for width in config.widths:
        eligible_charts = tuple(
            chart for chart, decomposition in enumerate(decompositions) if width <= decomposition.maximum_width
        )
        if not eligible_charts:
            continue
        support_grid = set(config.support_sizes)
        if config.include_width_plus_one:
            support_grid.add(width + 1)
        for support_size in sorted(support_grid):
            models: list[LocalScenarioModel] = []
            initializations: list[ScenarioInitialization] = []
            training_configs: list[TrainingConfig] = []
            active_charts: list[int] = []
            for chart in eligible_charts:
                try:
                    projected = decompositions[chart].coordinates(
                        chart_outcomes[chart],
                        width,
                        problem,
                    )
                    clustering = cluster_projected_scenarios(
                        projected,
                        support_size,
                        seed=_derived_seed(
                            config.seed,
                            "initialization",
                            chart_count,
                            ridge_index,
                            chart,
                            width,
                            support_size,
                        ),
                    )
                    initialization = clustering.initialization
                    probability_initialization = _fit_probability_initialization(
                        config,
                        charts,
                        chart,
                        chart_contexts[chart],
                        clustering,
                    )
                    model = LocalScenarioModel(
                        problem,
                        decompositions[chart],
                        initialization,
                        width=width,
                        context_mean=charts.mean,
                        context_scale=charts.scale,
                        standardized_center=charts.standardized_centers[chart],
                        probability_initialization=probability_initialization,
                    )
                except Exception as exc:
                    failures.append(
                        _failure(
                            chart_count=chart_count,
                            ridge=ridge,
                            chart=chart,
                            width=width,
                            support_size=support_size,
                            stage="scenario_initialization",
                            exc=exc,
                        )
                    )
                    continue
                active_charts.append(chart)
                initializations.append(initialization)
                models.append(model)
                training_configs.append(
                    _chart_training_config(
                        config,
                        chart_count=chart_count,
                        ridge_index=ridge_index,
                        chart=chart,
                        width=width,
                        support_size=support_size,
                    )
                )
            if not active_charts:
                continue
            try:
                results = training_function(
                    problem,
                    models,
                    tuple(chart_contexts[chart] for chart in active_charts),
                    tuple(chart_outcomes[chart] for chart in active_charts),
                    ridge=ridge,
                    configs=training_configs,
                )
            except Exception as exc:
                failures.append(
                    _failure(
                        chart_count=chart_count,
                        ridge=ridge,
                        chart=None,
                        width=width,
                        support_size=support_size,
                        stage="joint_local_training",
                        exc=exc,
                    )
                )
                continue
            for chart, initialization, result in zip(
                active_charts,
                initializations,
                results,
                strict=True,
            ):
                trained_by_chart[chart].extend(
                    TrainedLocalCandidate(
                        chart=chart,
                        ridge=ridge,
                        width=width,
                        support_size=support_size,
                        checkpoint=checkpoint,
                        decomposition=decompositions[chart],
                        initialization=initialization,
                    )
                    for checkpoint in result.checkpoints
                )

    empty_charts = [chart for chart, local_candidates in enumerate(trained_by_chart) if not local_candidates]
    for chart in empty_charts:
        failures.append(
            CandidateFailure(
                chart_count=chart_count,
                ridge=ridge,
                chart=chart,
                width=None,
                support_size=None,
                stage="empty_local_library",
                error_type="NoValidCandidate",
                message="no local model survived training-only requirements",
            )
        )
    if empty_charts:
        return None
    return RidgeCandidateLibrary(
        ridge=ridge,
        local_candidates=tuple(tuple(local_candidates) for local_candidates in trained_by_chart),
    )


def _fit_probability_initialization(
    config: AtlasSearchConfig,
    charts: ContextCharts,
    chart: int,
    chart_contexts: Array,
    clustering: ScenarioClustering,
) -> ProbabilityHeadInitialization | None:
    if config.probability_head_initialization == "cluster_frequencies":
        return None
    centered_contexts = (
        np.asarray(chart_contexts, dtype=np.float64) - charts.mean
    ) / charts.scale - charts.standardized_centers[chart]
    return fit_cluster_probability_head(
        centered_contexts,
        clustering.assignments,
        clustering.initialization.support_size,
        slope_penalty=config.training.slope_penalty,
        maximum_iterations=config.probability_head_maximum_iterations,
    )


def _chart_training_config(
    config: AtlasSearchConfig,
    *,
    chart_count: int,
    ridge_index: int,
    chart: int,
    width: int,
    support_size: int,
) -> TrainingConfig:
    training = config.training
    return TrainingConfig(
        steps=training.steps,
        learning_rate=training.learning_rate,
        save_steps=training.save_steps,
        batch_size=training.batch_size,
        coordinate_penalty=training.coordinate_penalty,
        slope_penalty=training.slope_penalty,
        gradient_clip=training.gradient_clip,
        train_scenario_coordinates=training.train_scenario_coordinates,
        train_probability_slopes=training.train_probability_slopes,
        seed=_derived_seed(
            training.seed,
            "training",
            chart_count,
            ridge_index,
            chart,
            width,
            support_size,
        ),
    )


def select_atlas(
    library: CandidateLibrary,
    validation: ObservedSplit,
    catalog: Array,
) -> FittedAtlas:
    """Use validation outcomes once to select one complete frozen atlas."""

    fitted, _ = _select_atlas(
        library,
        validation,
        catalog,
        retain_validation_trace=False,
    )
    return fitted


def select_atlas_with_trace(
    library: CandidateLibrary,
    validation: ObservedSplit,
    catalog: Array,
) -> tuple[FittedAtlas, tuple[LocalCandidateValidation, ...]]:
    """Select one atlas and retain the validation score of every candidate."""

    return _select_atlas(
        library,
        validation,
        catalog,
        retain_validation_trace=True,
    )


def _select_atlas(
    library: CandidateLibrary,
    validation: ObservedSplit,
    catalog: Array,
    *,
    retain_validation_trace: bool,
) -> tuple[FittedAtlas, tuple[LocalCandidateValidation, ...]]:
    """Apply the common selector, optionally retaining candidate scores."""

    outcomes = validation.outcomes(catalog)
    complete_candidates: list[FittedAtlas] = []
    validation_trace: list[LocalCandidateValidation] = []
    for partition in library.partitions:
        assignments = partition.charts.assign(validation.contexts)
        validation_counts = np.bincount(
            assignments,
            minlength=partition.charts.chart_count,
        )
        if np.any(validation_counts == 0):
            continue
        for ridge_library in partition.ridge_libraries:
            selected: list[SelectedLocalModel] = []
            for chart, local_library in enumerate(ridge_library.local_candidates):
                indices = np.flatnonzero(assignments == chart)
                scored = []
                for candidate in local_library:
                    loss_sum, loss_values = _validation_score(
                        library.problem,
                        partition.charts,
                        candidate,
                        validation.contexts[indices],
                        outcomes[indices],
                    )
                    scored.append((loss_sum, candidate, loss_values))
                if retain_validation_trace:
                    validation_trace.extend(
                        LocalCandidateValidation(
                            chart_count=partition.charts.chart_count,
                            ridge=ridge_library.ridge,
                            chart=chart,
                            width=candidate.width,
                            support_size=candidate.support_size,
                            checkpoint_step=candidate.checkpoint.step,
                            validation_loss_sum=loss_sum,
                            validation_count=len(indices),
                            validation_indices=tuple(int(index) for index in indices),
                            validation_losses=loss_values,
                        )
                        for loss_sum, candidate, loss_values in scored
                    )
                loss_sum, candidate, _ = min(
                    scored,
                    key=lambda item: (
                        item[0],
                        item[1].support_size,
                        item[1].width,
                        item[1].checkpoint.step,
                    ),
                )
                selected.append(
                    SelectedLocalModel(
                        candidate=candidate,
                        validation_loss_sum=loss_sum,
                        validation_count=len(indices),
                    )
                )
            mean_loss = sum(item.validation_loss_sum for item in selected) / len(validation.contexts)
            complete_candidates.append(
                FittedAtlas(
                    problem=library.problem,
                    charts=partition.charts,
                    ridge=ridge_library.ridge,
                    local_models=tuple(selected),
                    validation_mean_loss=float(mean_loss),
                )
            )
    if not complete_candidates:
        raise RuntimeError("validation produced no complete atlas candidate with observations in every chart")
    fitted = min(
        complete_candidates,
        key=lambda atlas: (
            atlas.validation_mean_loss,
            atlas.total_support,
            atlas.total_width,
            atlas.chart_count,
            atlas.total_scenario_coordinates,
            atlas.ridge,
        ),
    )
    return fitted, tuple(validation_trace)


@torch.no_grad()
def _validation_score(
    problem: DifferentiableStochasticProblem,
    charts: ContextCharts,
    candidate: TrainedLocalCandidate,
    contexts: Array,
    outcomes: Array,
) -> tuple[float, tuple[float, ...]]:
    if len(contexts) == 0:
        return 0.0, ()
    model = candidate.make_model(problem, charts)
    context_tensor = torch.tensor(contexts, dtype=torch.float64)
    outcome_tensor = torch.tensor(outcomes, dtype=torch.float64)
    losses = realized_loss_from_scenarios_torch(
        problem,
        context_tensor,
        model.scenarios(),
        model.probabilities(context_tensor),
        outcome_tensor,
        ridge=candidate.ridge,
    )
    value = float(losses.sum().item())
    if not math.isfinite(value):
        raise FloatingPointError("validation loss is nonfinite")
    loss_values = tuple(float(loss) for loss in losses.tolist())
    if any(not math.isfinite(loss) for loss in loss_values):
        raise FloatingPointError("validation losses are nonfinite")
    return value, loss_values


def _validation_loss_sum(
    problem: DifferentiableStochasticProblem,
    charts: ContextCharts,
    candidate: TrainedLocalCandidate,
    contexts: Array,
    outcomes: Array,
) -> float:
    """Return the legacy aggregate score used by external runners."""

    value, _ = _validation_score(
        problem,
        charts,
        candidate,
        contexts,
        outcomes,
    )
    return value


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _failure(
    *,
    chart_count: int,
    ridge: float | None,
    chart: int | None,
    width: int | None,
    support_size: int | None,
    stage: str,
    exc: Exception,
) -> CandidateFailure:
    return CandidateFailure(
        chart_count=chart_count,
        ridge=ridge,
        chart=chart,
        width=width,
        support_size=support_size,
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
    )
