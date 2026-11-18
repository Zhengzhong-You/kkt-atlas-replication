"""Training-only structural choices for KKT Atlas.

This module stops before local-model fitting. It combines fixed context
charts, chart-specific spectral widths, and the structural support candidate
``K_b = q_b + 1``. A partition is eligible only when every chart supplies a
frozen minimum number of observations per identifiable local parameter.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kkt_atlas.charts import ContextCharts, fit_context_charts
from kkt_atlas.data import ObservedSplit
from kkt_atlas.problem import Array, DifferentiableStochasticProblem
from kkt_atlas.response_basis import ResponseDecomposition, fit_response_decomposition
from kkt_atlas.spectral_selection import SpectralWidthSelection


def identifiable_local_parameter_count(
    support_size: int,
    representation_dim: int,
    context_dim: int,
) -> int:
    """Return ``K*q + (K-1)*(d_x+1)`` identifiable local parameters."""

    support_size = _positive_integer(support_size, name="support_size")
    representation_dim = _positive_integer(representation_dim, name="representation_dim")
    context_dim = _nonnegative_integer(context_dim, name="context_dim")
    return support_size * representation_dim + (support_size - 1) * (context_dim + 1)


def structural_support_size(representation_dim: int) -> int:
    """Return the structural candidate ``K = q + 1``."""

    return _positive_integer(representation_dim, name="representation_dim") + 1


def canonical_dyadic_chart_counts(
    training_size: int,
    minimum_chart_size: int,
) -> tuple[int, ...]:
    """Return ``1,2,4,...`` up to the training-size feasibility ceiling."""

    training_size = _positive_integer(training_size, name="training_size")
    minimum_chart_size = _positive_integer(
        minimum_chart_size,
        name="minimum_chart_size",
    )
    maximum_chart_count = training_size // minimum_chart_size
    if maximum_chart_count < 1:
        raise ValueError("training_size must be at least minimum_chart_size")
    values: list[int] = []
    chart_count = 1
    while chart_count <= maximum_chart_count:
        values.append(chart_count)
        chart_count *= 2
    return tuple(values)


def canonical_integer_chart_counts(
    training_size: int,
    minimum_chart_size: int,
) -> tuple[int, ...]:
    """Return every integer chart count allowed by the sample-size ceiling."""

    training_size = _positive_integer(training_size, name="training_size")
    minimum_chart_size = _positive_integer(
        minimum_chart_size,
        name="minimum_chart_size",
    )
    maximum_chart_count = training_size // minimum_chart_size
    if maximum_chart_count < 1:
        raise ValueError("training_size must be at least minimum_chart_size")
    return tuple(range(1, maximum_chart_count + 1))


@dataclass(frozen=True)
class AutomaticStructureConfig:
    """Frozen controls for training-only automatic structure selection."""

    chart_counts: tuple[int, ...]
    ridge: float
    spectral_retained_ratio: float
    minimum_samples_per_parameter: float
    seed: int = 0
    minimum_chart_size: int = 2
    minimum_context_scale: float = 1e-10
    minimum_scenario_scale: float = 1e-8
    spectral_absolute_tolerance: float = 0.0
    spectral_relative_tolerance: float = 1e-12
    projected_coordinate_distinct_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        chart_counts = tuple(
            sorted({_positive_integer(value, name="chart_counts entry") for value in self.chart_counts})
        )
        if not chart_counts:
            raise ValueError("chart_counts must contain at least one positive integer")
        object.__setattr__(self, "chart_counts", chart_counts)
        object.__setattr__(self, "ridge", _nonnegative_finite_float(self.ridge, name="ridge"))
        ratio = float(self.spectral_retained_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise ValueError("spectral_retained_ratio must lie in (0,1]")
        object.__setattr__(self, "spectral_retained_ratio", ratio)
        object.__setattr__(
            self,
            "minimum_samples_per_parameter",
            _positive_finite_float(
                self.minimum_samples_per_parameter,
                name="minimum_samples_per_parameter",
            ),
        )
        object.__setattr__(
            self,
            "minimum_chart_size",
            _positive_integer(self.minimum_chart_size, name="minimum_chart_size"),
        )
        for name in ("minimum_context_scale", "minimum_scenario_scale"):
            object.__setattr__(self, name, _positive_finite_float(getattr(self, name), name=name))
        for name in (
            "spectral_absolute_tolerance",
            "spectral_relative_tolerance",
            "projected_coordinate_distinct_tolerance",
        ):
            object.__setattr__(self, name, _nonnegative_finite_float(getattr(self, name), name=name))


@dataclass(frozen=True)
class LocalStructureDiagnostic:
    """Training-only structural quantities for one fixed chart."""

    chart: int
    sample_size: int
    representation_dim: int
    support_size: int
    identifiable_parameter_count: int
    samples_per_parameter: float
    retained_spectral_ratio: float
    discarded_squared_tail: float
    effective_rank: int
    target_achieved: bool
    required_sample_size: int
    sample_size_requirement_achieved: bool
    distinct_projected_rows: int
    distinct_coordinate_tolerance: float
    distinct_support_achieved: bool
    eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart": self.chart,
            "sample_size": self.sample_size,
            "representation_dim": self.representation_dim,
            "support_size": self.support_size,
            "identifiable_parameter_count": self.identifiable_parameter_count,
            "samples_per_parameter": self.samples_per_parameter,
            "retained_spectral_ratio": self.retained_spectral_ratio,
            "discarded_squared_tail": self.discarded_squared_tail,
            "effective_rank": self.effective_rank,
            "target_achieved": self.target_achieved,
            "required_sample_size": self.required_sample_size,
            "sample_size_requirement_achieved": self.sample_size_requirement_achieved,
            "distinct_projected_rows": self.distinct_projected_rows,
            "distinct_coordinate_tolerance": self.distinct_coordinate_tolerance,
            "distinct_support_achieved": self.distinct_support_achieved,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class PartitionStructureDiagnostic:
    """Deterministic diagnostic for one candidate number of charts."""

    chart_count: int
    chart_seed: int
    distortion: float
    minimum_samples_per_parameter: float
    limiting_samples_per_parameter: float
    eligible: bool
    charts: tuple[LocalStructureDiagnostic, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_count": self.chart_count,
            "chart_seed": self.chart_seed,
            "distortion": self.distortion,
            "minimum_samples_per_parameter": self.minimum_samples_per_parameter,
            "limiting_samples_per_parameter": self.limiting_samples_per_parameter,
            "eligible": self.eligible,
            "charts": [item.as_dict() for item in self.charts],
        }


@dataclass(frozen=True)
class AutomaticPartitionCandidate:
    """One training-only partition and its chart-specific structures."""

    context_charts: ContextCharts
    decompositions: tuple[ResponseDecomposition, ...]
    spectral_selections: tuple[SpectralWidthSelection, ...]
    diagnostic: PartitionStructureDiagnostic

    def __post_init__(self) -> None:
        count = self.context_charts.chart_count
        if len(self.decompositions) != count or len(self.spectral_selections) != count:
            raise ValueError("one decomposition and spectral selection are required per chart")
        if len(self.diagnostic.charts) != count or self.diagnostic.chart_count != count:
            raise ValueError("diagnostic charts must align with the context partition")


@dataclass(frozen=True)
class PartitionConstructionFailure:
    """A deterministic record of a candidate partition that could not be built."""

    chart_count: int
    stage: str
    error_type: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_count": self.chart_count,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class AutomaticStructureSelection:
    """Largest eligible chart count under a frozen sample-complexity ratio."""

    selected: AutomaticPartitionCandidate
    candidates: tuple[AutomaticPartitionCandidate, ...]
    failures: tuple[PartitionConstructionFailure, ...]
    config: AutomaticStructureConfig

    def __post_init__(self) -> None:
        if not self.selected.diagnostic.eligible:
            raise ValueError("selected partition must be eligible")
        eligible_counts = [
            item.diagnostic.chart_count for item in self.candidates if item.diagnostic.eligible
        ]
        if not eligible_counts or self.selected.diagnostic.chart_count != max(eligible_counts):
            raise ValueError("selected partition must have the largest eligible chart count")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rule": "largest_B_with_minimum_n_over_P",
            "spectral_retained_ratio": self.config.spectral_retained_ratio,
            "minimum_samples_per_parameter": self.config.minimum_samples_per_parameter,
            "selected_chart_count": self.selected.diagnostic.chart_count,
            "candidates": [item.diagnostic.as_dict() for item in self.candidates],
            "failures": [item.as_dict() for item in self.failures],
        }


def build_automatic_partition_candidate(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    *,
    chart_count: int,
    config: AutomaticStructureConfig,
) -> AutomaticPartitionCandidate:
    """Construct one ``B`` candidate using only the supplied training split."""

    chart_count = _positive_integer(chart_count, name="chart_count")
    if chart_count not in config.chart_counts:
        raise ValueError("chart_count must belong to config.chart_counts")
    if training.contexts.shape[1] != problem.context_dim:
        raise ValueError("training context dimension does not match the problem")
    outcomes = training.outcomes(catalog)
    if outcomes.shape[1] != problem.scenario_dim:
        raise ValueError("training scenario dimension does not match the problem")

    # This seed identity must match the established atlas runners so that the
    # automatic rule and the exhaustive anchor inspect the same partition.
    chart_seed = _derived_seed(config.seed, "charts", chart_count)
    charts = fit_context_charts(
        training.contexts,
        chart_count,
        seed=chart_seed,
        minimum_scale=config.minimum_context_scale,
        minimum_chart_size=config.minimum_chart_size,
    )
    decompositions: list[ResponseDecomposition] = []
    selections: list[SpectralWidthSelection] = []
    diagnostics: list[LocalStructureDiagnostic] = []
    for chart in range(chart_count):
        indices = charts.indices(chart)
        decomposition = fit_response_decomposition(
            problem,
            charts.centers[chart],
            outcomes[indices],
            ridge=config.ridge,
            minimum_scenario_scale=config.minimum_scenario_scale,
            basis_method="kkt",
        )
        selection = decomposition.select_width_by_spectral_ratio(
            config.spectral_retained_ratio,
            absolute_tolerance=config.spectral_absolute_tolerance,
            relative_tolerance=config.spectral_relative_tolerance,
        )
        representation_dim = selection.width
        support_size = structural_support_size(representation_dim)
        parameter_count = identifiable_local_parameter_count(
            support_size,
            representation_dim,
            problem.context_dim,
        )
        sample_size = int(len(indices))
        sample_ratio = sample_size / parameter_count
        projected = decomposition.coordinates(
            outcomes[indices],
            representation_dim,
            problem,
        )
        distinct_rows = _count_distinct_rows(
            projected,
            tolerance=config.projected_coordinate_distinct_tolerance,
        )
        required_sample_size = max(support_size, representation_dim + 1)
        sample_size_achieved = sample_size >= required_sample_size
        distinct_support_achieved = distinct_rows >= support_size
        local_eligible = (
            sample_ratio >= config.minimum_samples_per_parameter
            and sample_size_achieved
            and selection.target_achieved
            and distinct_support_achieved
        )
        decompositions.append(decomposition)
        selections.append(selection)
        diagnostics.append(
            LocalStructureDiagnostic(
                chart=chart,
                sample_size=sample_size,
                representation_dim=representation_dim,
                support_size=support_size,
                identifiable_parameter_count=parameter_count,
                samples_per_parameter=sample_ratio,
                retained_spectral_ratio=selection.retained_ratio,
                discarded_squared_tail=selection.discarded_squared_tail,
                effective_rank=selection.effective_rank,
                target_achieved=selection.target_achieved,
                required_sample_size=required_sample_size,
                sample_size_requirement_achieved=sample_size_achieved,
                distinct_projected_rows=distinct_rows,
                distinct_coordinate_tolerance=(config.projected_coordinate_distinct_tolerance),
                distinct_support_achieved=distinct_support_achieved,
                eligible=local_eligible,
            )
        )

    limiting_ratio = min(item.samples_per_parameter for item in diagnostics)
    partition_diagnostic = PartitionStructureDiagnostic(
        chart_count=chart_count,
        chart_seed=chart_seed,
        distortion=charts.distortion,
        minimum_samples_per_parameter=config.minimum_samples_per_parameter,
        limiting_samples_per_parameter=limiting_ratio,
        eligible=all(item.eligible for item in diagnostics),
        charts=tuple(diagnostics),
    )
    return AutomaticPartitionCandidate(
        context_charts=charts,
        decompositions=tuple(decompositions),
        spectral_selections=tuple(selections),
        diagnostic=partition_diagnostic,
    )


def select_automatic_structure(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: AutomaticStructureConfig,
) -> AutomaticStructureSelection:
    """Build all declared ``B`` candidates and return the largest eligible one."""

    candidates: list[AutomaticPartitionCandidate] = []
    failures: list[PartitionConstructionFailure] = []
    for chart_count in config.chart_counts:
        try:
            candidate = build_automatic_partition_candidate(
                problem,
                training,
                catalog,
                chart_count=chart_count,
                config=config,
            )
        except Exception as exc:
            failures.append(
                PartitionConstructionFailure(
                    chart_count=chart_count,
                    stage="training_only_structure",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue
        candidates.append(candidate)

    eligible = [item for item in candidates if item.diagnostic.eligible]
    if not eligible:
        ratios = {
            item.diagnostic.chart_count: item.diagnostic.limiting_samples_per_parameter
            for item in candidates
        }
        raise ValueError(
            "no chart count satisfies the frozen minimum samples-per-parameter "
            f"ratio; observed limiting ratios={ratios}"
        )
    selected = max(eligible, key=lambda item: item.diagnostic.chart_count)
    return AutomaticStructureSelection(
        selected=selected,
        candidates=tuple(candidates),
        failures=tuple(failures),
        config=config,
    )


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _count_distinct_rows(values: Array, *, tolerance: float) -> int:
    """Count deterministic row representatives under an L-infinity tolerance."""

    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0 or np.any(~np.isfinite(rows)):
        raise ValueError("projected coordinates must be a nonempty finite matrix")
    tolerance = _nonnegative_finite_float(tolerance, name="tolerance")
    order = np.lexsort(tuple(rows[:, column] for column in reversed(range(rows.shape[1]))))
    representatives: list[Array] = []
    for row in rows[order]:
        if not representatives:
            representatives.append(row)
            continue
        distances = np.max(np.abs(np.vstack(representatives) - row[None, :]), axis=1)
        if np.all(distances > tolerance):
            representatives.append(row)
    return len(representatives)


def _positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result
