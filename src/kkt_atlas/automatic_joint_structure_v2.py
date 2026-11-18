"""Joint training-only selection of chart number, width, and support.

Candidate chart numbers are examined in decreasing order.  For a fixed
partition, each chart receives a spectral width and a training-only,
internal-resampling initialized-operational-loss support path.  The path is
capped by ``min(q+1+support_cap_offset,N_distinct,N_sample)``, where ``N_distinct`` is the minimum number
of distinct projected fitting observations across the internal folds.  The
``q+1`` term is a protocol choice rather than a universal support theorem.
The smallest empirical loss minimizer determines the sample-per-parameter
gate and is retained for subsequent training and deployment.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

from kkt_atlas.automatic_operational_support_v1 import (
    OperationalSupportSelection,
    select_initialized_operational_support_size,
)
from kkt_atlas.automatic_structure import (
    AutomaticPartitionCandidate,
    AutomaticStructureConfig,
    build_automatic_partition_candidate,
    identifiable_local_parameter_count,
)
from kkt_atlas.automatic_support import fixed_cross_fitting_folds
from kkt_atlas.charts import MinimumChartSizeViolation
from kkt_atlas.data import ObservedSplit
from kkt_atlas.problem import Array, DifferentiableStochasticProblem


@dataclass(frozen=True)
class JointStructureConfig:
    """Controls for selected-support-aware training-only structure selection."""

    chart_counts: tuple[int, ...]
    ridge: float
    spectral_retained_ratio: float
    minimum_samples_per_parameter: float
    seed: int = 0
    minimum_chart_size: int = 50
    fold_count: int = 5
    slope_penalty: float = 1e-5
    maximum_head_iterations: int = 2_000
    minimum_context_scale: float = 1e-10
    minimum_scenario_scale: float = 1e-8
    spectral_absolute_tolerance: float = 0.0
    spectral_relative_tolerance: float = 1e-12
    distinct_projected_tolerance: float = 1e-10
    fail_closed: bool = True
    chart_count_rule: str = "largest_eligible"
    accept_intercept_only_fallback: bool = False
    support_cap_offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.support_cap_offset, bool) or not isinstance(self.support_cap_offset, int) or self.support_cap_offset < 0:
            raise ValueError("support_cap_offset must be a nonnegative integer")
        counts = tuple(
            sorted(
                {
                    _positive_integer(value, name="chart_counts entry")
                    for value in self.chart_counts
                },
                reverse=True,
            )
        )
        if not counts:
            raise ValueError("chart_counts must contain at least one positive integer")
        object.__setattr__(self, "chart_counts", counts)
        ridge = float(self.ridge)
        if not math.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        object.__setattr__(self, "ridge", ridge)
        ratio = float(self.spectral_retained_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise ValueError("spectral_retained_ratio must lie in (0,1]")
        object.__setattr__(self, "spectral_retained_ratio", ratio)
        object.__setattr__(
            self,
            "minimum_samples_per_parameter",
            _positive_finite(
                self.minimum_samples_per_parameter,
                name="minimum_samples_per_parameter",
            ),
        )
        object.__setattr__(
            self,
            "minimum_chart_size",
            _positive_integer(self.minimum_chart_size, name="minimum_chart_size"),
        )
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        for name in ("minimum_context_scale", "minimum_scenario_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in (
            "slope_penalty",
            "spectral_absolute_tolerance",
            "spectral_relative_tolerance",
            "distinct_projected_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "maximum_head_iterations",
            _positive_integer(
                self.maximum_head_iterations,
                name="maximum_head_iterations",
            ),
        )
        if not isinstance(self.fail_closed, bool):
            raise TypeError("fail_closed must be a bool")
        if self.chart_count_rule not in {"largest_eligible", "minimum_operational_loss"}:
            raise ValueError(
                "chart_count_rule must be largest_eligible or minimum_operational_loss"
            )
        if not isinstance(self.accept_intercept_only_fallback, bool):
            raise TypeError("accept_intercept_only_fallback must be a bool")


@dataclass(frozen=True)
class JointLocalSelection:
    """Selected width and support for one chart."""

    chart: int
    sample_size: int
    minimum_fitting_fold_size: int
    representation_dim: int
    protocol_support_cap: int
    production_support_cap: int
    selected_support_size: int
    spectral_target_achieved: bool
    spectral_effective_rank: int
    spectral_rank_adequate: bool
    gate_parameter_count: int
    gate_samples_per_parameter: float
    gate_eligible: bool
    selector_seed: int
    operational: OperationalSupportSelection
    selection_seconds: float
    support_cap_offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.support_cap_offset, bool) or not isinstance(self.support_cap_offset, int) or self.support_cap_offset < 0:
            raise ValueError("support_cap_offset must be a nonnegative integer")
        if self.protocol_support_cap != self.representation_dim + 1 + self.support_cap_offset:
            raise ValueError("protocol support cap must equal q+1+support_cap_offset")
        if self.production_support_cap > self.protocol_support_cap:
            raise ValueError("production support path exceeds the protocol cap")
        if not 1 <= self.selected_support_size <= self.production_support_cap:
            raise ValueError("selected support lies outside the production path")
        if self.selected_support_size != self.operational.selected_support_size:
            raise ValueError("local and operational support selections disagree")
        if self.gate_eligible and not (
            self.spectral_target_achieved and self.spectral_rank_adequate
        ):
            raise ValueError("an eligible chart must pass the spectral gate")

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart": self.chart,
            "training_sample_size": self.sample_size,
            "minimum_fitting_fold_size": self.minimum_fitting_fold_size,
            "minimum_fitting_sample_size": self.minimum_fitting_fold_size,
            "q_b": self.representation_dim,
            "K_protocol_cap": self.protocol_support_cap,
            "support_cap_offset": self.support_cap_offset,
            "K_cap": self.production_support_cap,
            "K_best": self.selected_support_size,
            "K_small": self.selected_support_size,
            "K_score_minimizer": self.selected_support_size,
            "K_selected": self.selected_support_size,
            "spectral_target_achieved": self.spectral_target_achieved,
            "spectral_effective_rank": self.spectral_effective_rank,
            "spectral_rank_adequate": self.spectral_rank_adequate,
            "gate_identifiable_parameter_count": self.gate_parameter_count,
            "gate_samples_per_parameter": self.gate_samples_per_parameter,
            "gate_eligible": self.gate_eligible,
            "selector_seed": self.selector_seed,
            "selection_seconds": self.selection_seconds,
            "work_counts": self.operational.diagnostics()["work_counts"],
            "operational_selection": self.operational.diagnostics(),
        }


@dataclass(frozen=True)
class JointPartitionSelection:
    """One fully evaluated candidate chart partition."""

    chart_count: int
    partition: AutomaticPartitionCandidate
    charts: tuple[JointLocalSelection, ...]
    eligible: bool
    construction_seconds: float
    support_selection_seconds: float

    def __post_init__(self) -> None:
        if self.chart_count != self.partition.context_charts.chart_count:
            raise ValueError("chart count does not match the partition")
        if len(self.charts) != self.chart_count:
            raise ValueError("one local selection is required per chart")
        if self.eligible != all(item.gate_eligible for item in self.charts):
            raise ValueError("partition eligibility must equal all local gates")

    def as_dict(self) -> dict[str, Any]:
        return {
            "B": self.chart_count,
            "status": "eligible" if self.eligible else "ineligible",
            "construction_seconds": self.construction_seconds,
            "support_selection_seconds": self.support_selection_seconds,
            "assessment_operational_loss": self.assessment_operational_loss,
            "charts": [item.as_dict() for item in self.charts],
            "work_counts": _sum_work_counts(
                item.operational.diagnostics()["work_counts"] for item in self.charts
            ),
        }

    @property
    def assessment_operational_loss(self) -> float:
        total = sum(item.sample_size for item in self.charts)
        return sum(
            item.sample_size
            * float(item.operational.best_fold_assessed_operational_loss)
            / total
            for item in self.charts
        )


@dataclass(frozen=True)
class JointStructureFailure:
    """Diagnostic record for a candidate chart number that could not be built."""

    chart_count: int
    stage: str
    error_type: str
    message: str
    elapsed_seconds: float
    completed_chart_count: int = 0
    work_counts: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "B": self.chart_count,
            "status": "failed",
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "elapsed_seconds": self.elapsed_seconds,
            "completed_chart_count": self.completed_chart_count,
            "work_counts": dict(self.work_counts),
        }


@dataclass(frozen=True)
class JointPartitionRejection:
    """A chart count rejected by a prespecified structural requirement."""

    chart_count: int
    chart_seed: int
    chart_sample_sizes: tuple[int, ...]
    minimum_chart_size: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if len(self.chart_sample_sizes) != self.chart_count:
            raise ValueError("one sample size is required per rejected chart")
        if sum(self.chart_sample_sizes) < 1:
            raise ValueError("rejected partition must contain observations")
        if min(self.chart_sample_sizes) >= self.minimum_chart_size:
            raise ValueError("rejected partition satisfies the minimum chart size")

    def as_dict(self) -> dict[str, Any]:
        return {
            "B": self.chart_count,
            "status": "ineligible",
            "stage": "context_partition",
            "reason": "minimum_chart_size",
            "chart_seed": self.chart_seed,
            "chart_sample_sizes": list(self.chart_sample_sizes),
            "minimum_chart_size": self.minimum_chart_size,
            "elapsed_seconds": self.elapsed_seconds,
            "work_counts": {},
        }


class JointStructureSelectionError(RuntimeError):
    """Raised when a candidate failure stops fail-closed structure selection."""

    def __init__(self, failure: JointStructureFailure) -> None:
        self.failure = failure
        super().__init__(
            "fail-closed joint structure selection stopped at "
            f"B={failure.chart_count} during {failure.stage}: "
            f"{failure.error_type}: {failure.message}"
        )


@dataclass(frozen=True)
class JointStructureSelection:
    """Selected chart partition under the configured training-only rule."""

    selected: JointPartitionSelection
    examined_candidates: tuple[JointPartitionSelection, ...]
    failures: tuple[JointStructureFailure, ...]
    config: JointStructureConfig
    rejections: tuple[JointPartitionRejection, ...] = ()

    def __post_init__(self) -> None:
        if not self.selected.eligible:
            raise ValueError("selected partition must be eligible")
        examined_counts = [item.chart_count for item in self.examined_candidates]
        failure_counts = [item.chart_count for item in self.failures]
        rejection_counts = [item.chart_count for item in self.rejections]
        all_counts = examined_counts + rejection_counts + failure_counts
        if len(all_counts) != len(set(all_counts)):
            raise ValueError("each chart number must be examined at most once")
        eligible = [item for item in self.examined_candidates if item.eligible]
        if self.config.chart_count_rule == "largest_eligible":
            if any(count < self.selected.chart_count for count in all_counts):
                raise ValueError("selection must stop after the largest eligible chart number")
            if any(item.chart_count > self.selected.chart_count for item in eligible):
                raise ValueError("a larger eligible partition was skipped")
        else:
            expected = min(
                eligible,
                key=lambda item: (item.assessment_operational_loss, item.chart_count),
            )
            if expected.chart_count != self.selected.chart_count:
                raise ValueError("selected partition does not minimize operational loss")
        if self.config.fail_closed and self.failures:
            raise ValueError("fail-closed selection cannot return recorded failures")

    def diagnostics(self) -> dict[str, Any]:
        candidate_records = [item.as_dict() for item in self.examined_candidates]
        rejection_records = [item.as_dict() for item in self.rejections]
        failure_records = [item.as_dict() for item in self.failures]
        return {
            "rule": self.config.chart_count_rule,
            "q_plus_one_is_protocol_search_cap": False,
            "q_plus_one_is_fixed_support_size": True,
            "support_cap_offset": self.config.support_cap_offset,
            "candidate_failure_handling": (
                "fail_closed"
                if self.config.fail_closed
                else "record_and_continue_development_only"
            ),
            "selected_chart_count": self.selected.chart_count,
            "examined_chart_counts": sorted(
                [item.chart_count for item in self.examined_candidates]
                + [item.chart_count for item in self.rejections]
                + [item.chart_count for item in self.failures],
                reverse=True,
            ),
            "examined_candidates": candidate_records + rejection_records,
            "structural_rejections": rejection_records,
            "failures": failure_records,
            "work_counts": _sum_work_counts(
                record["work_counts"] for record in [
                    *candidate_records,
                    *rejection_records,
                    *failure_records,
                ]
            ),
        }


def select_joint_training_structure(
    problem: DifferentiableStochasticProblem,
    training: ObservedSplit,
    catalog: Array,
    config: JointStructureConfig,
) -> JointStructureSelection:
    """Return the first eligible candidate while examining ``B`` downward.

    A partition that violates the prespecified minimum chart size is an
    ineligible structural candidate, so the descending search continues.
    With ``config.fail_closed=True``, every other exception in partition
    construction, spectral selection, or operational support selection
    terminates the procedure. Setting it to ``False`` permits failure recording
    and continuation only for explicitly nonconfirmation diagnostic runs.
    """

    outcomes = training.outcomes(catalog)
    automatic_config = AutomaticStructureConfig(
        chart_counts=tuple(sorted(config.chart_counts)),
        ridge=config.ridge,
        spectral_retained_ratio=config.spectral_retained_ratio,
        minimum_samples_per_parameter=config.minimum_samples_per_parameter,
        seed=config.seed,
        minimum_chart_size=config.minimum_chart_size,
        minimum_context_scale=config.minimum_context_scale,
        minimum_scenario_scale=config.minimum_scenario_scale,
        spectral_absolute_tolerance=config.spectral_absolute_tolerance,
        spectral_relative_tolerance=config.spectral_relative_tolerance,
        projected_coordinate_distinct_tolerance=(
            config.distinct_projected_tolerance
        ),
    )
    candidates: list[JointPartitionSelection] = []
    rejections: list[JointPartitionRejection] = []
    failures: list[JointStructureFailure] = []

    for chart_count in config.chart_counts:
        candidate_started = time.perf_counter()
        try:
            construction_started = time.perf_counter()
            partition = build_automatic_partition_candidate(
                problem,
                training,
                catalog,
                chart_count=chart_count,
                config=automatic_config,
            )
            construction_seconds = time.perf_counter() - construction_started
        except MinimumChartSizeViolation as exc:
            rejections.append(
                JointPartitionRejection(
                    chart_count=chart_count,
                    chart_seed=_derived_seed(config.seed, "charts", chart_count),
                    chart_sample_sizes=exc.counts,
                    minimum_chart_size=exc.minimum_chart_size,
                    elapsed_seconds=time.perf_counter() - candidate_started,
                )
            )
            continue
        except Exception as exc:
            failure = JointStructureFailure(
                chart_count=chart_count,
                stage="partition_and_spectral_width",
                error_type=type(exc).__name__,
                message=str(exc),
                elapsed_seconds=time.perf_counter() - candidate_started,
            )
            if config.fail_closed:
                raise JointStructureSelectionError(failure) from exc
            failures.append(failure)
            continue

        local_records: list[JointLocalSelection] = []
        support_seconds = 0.0
        failed = False
        for chart in range(chart_count):
            indices = partition.context_charts.indices(chart)
            selector_seed = operational_selector_seed(config.seed, chart_count, chart)
            minimum_fitting = minimum_fitting_fold_size(
                len(indices),
                config.fold_count,
                selector_seed,
            )
            q_value = partition.diagnostic.charts[chart].representation_dim
            protocol_cap = q_value + 1 + config.support_cap_offset
            sample_cap = max((
                k
                for k in range(1, protocol_cap + 1)
                if minimum_fitting
                / identifiable_local_parameter_count(k, q_value, problem.context_dim)
                >= config.minimum_samples_per_parameter
            ), default=0)
            if sample_cap == 0:
                raise ValueError("no support size satisfies the sample-per-parameter gate")
            started = time.perf_counter()
            try:
                operational = select_initialized_operational_support_size(
                    problem,
                    partition.decompositions[chart],
                    training.contexts[indices],
                    outcomes[indices],
                    width=partition.diagnostic.charts[chart].representation_dim,
                    ridge=config.ridge,
                    context_mean=partition.context_charts.mean,
                    context_scale=partition.context_charts.scale,
                    standardized_center=(
                        partition.context_charts.standardized_centers[chart]
                    ),
                    slope_penalty=config.slope_penalty,
                    maximum_head_iterations=config.maximum_head_iterations,
                    accept_intercept_only_fallback=(
                        config.accept_intercept_only_fallback
                    ),
                    fold_count=config.fold_count,
                    distinct_projected_tolerance=(
                        config.distinct_projected_tolerance
                    ),
                    seed=selector_seed,
                    support_cap_offset=config.support_cap_offset,
                    maximum_support_size=sample_cap,
                )
            except Exception as exc:
                failure = JointStructureFailure(
                    chart_count=chart_count,
                    stage=f"operational_support_chart_{chart}",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    elapsed_seconds=time.perf_counter() - candidate_started,
                    completed_chart_count=len(local_records),
                    work_counts=tuple(
                        sorted(
                            _sum_work_counts(
                                item.operational.diagnostics()["work_counts"]
                                for item in local_records
                            ).items()
                        )
                    ),
                )
                if config.fail_closed:
                    raise JointStructureSelectionError(failure) from exc
                failures.append(failure)
                failed = True
                break
            elapsed = time.perf_counter() - started
            support_seconds += elapsed
            spectral = partition.spectral_selections[chart]
            spectral_rank_adequate = spectral.effective_rank >= q_value
            selected_support = operational.selected_support_size
            if operational.width_based_support_cap != protocol_cap:
                raise RuntimeError("operational path does not use the declared protocol cap")
            parameter_count = identifiable_local_parameter_count(
                selected_support,
                q_value,
                problem.context_dim,
            )
            ratio = minimum_fitting / parameter_count
            gate = (
                len(indices) >= config.minimum_chart_size
                and ratio >= config.minimum_samples_per_parameter
                and spectral.target_achieved
                and spectral_rank_adequate
            )
            local_records.append(
                JointLocalSelection(
                    chart=chart,
                    sample_size=len(indices),
                    minimum_fitting_fold_size=minimum_fitting,
                    representation_dim=q_value,
                    protocol_support_cap=protocol_cap,
                    production_support_cap=operational.support_cap,
                    selected_support_size=selected_support,
                    spectral_target_achieved=spectral.target_achieved,
                    spectral_effective_rank=spectral.effective_rank,
                    spectral_rank_adequate=spectral_rank_adequate,
                    gate_parameter_count=parameter_count,
                    gate_samples_per_parameter=ratio,
                    gate_eligible=gate,
                    selector_seed=selector_seed,
                    operational=operational,
                    selection_seconds=elapsed,
                    support_cap_offset=config.support_cap_offset,
                )
            )
        if failed:
            continue

        evaluated = JointPartitionSelection(
            chart_count=chart_count,
            partition=partition,
            charts=tuple(local_records),
            eligible=all(item.gate_eligible for item in local_records),
            construction_seconds=construction_seconds,
            support_selection_seconds=support_seconds,
        )
        candidates.append(evaluated)
        if evaluated.eligible and config.chart_count_rule == "largest_eligible":
            return JointStructureSelection(
                selected=evaluated,
                examined_candidates=tuple(candidates),
                failures=tuple(failures),
                config=config,
                rejections=tuple(rejections),
            )

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (item.assessment_operational_loss, item.chart_count),
        )
        return JointStructureSelection(
            selected=selected,
            examined_candidates=tuple(candidates),
            failures=tuple(failures),
            config=config,
            rejections=tuple(rejections),
        )

    observed = {
        item.chart_count: [chart.gate_samples_per_parameter for chart in item.charts]
        for item in candidates
    }
    raise ValueError(
        "no chart number satisfies the selected-support sample gate; "
        f"observed ratios={observed}, "
        f"rejections={[item.as_dict() for item in rejections]}, "
        f"failures={[item.as_dict() for item in failures]}"
    )


def operational_selector_seed(base_seed: int, chart_count: int, chart: int) -> int:
    """Return the fixed selector seed for one chart of one B candidate."""

    return _derived_seed(base_seed, "joint_operational_structure_v2", chart_count, chart)


def minimum_fitting_fold_size(
    sample_size: int,
    fold_count: int,
    selector_seed: int,
) -> int:
    """Return the smallest fitting-fold size used by the operational selector."""

    folds = fixed_cross_fitting_folds(
        sample_size,
        fold_count,
        seed=_derived_seed(selector_seed, "operational_support_folds"),
    )
    return min(sample_size - len(assessment) for assessment in folds)


def _sum_work_counts(records: Any) -> dict[str, int]:
    total: dict[str, int] = {}
    for record in records:
        for name, value in record.items():
            total[name] = total.get(name, 0) + int(value)
    return total


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result
