"""Validation selection of one checkpoint shared by every frozen chart."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kkt_atlas.atlas import (
    CandidateLibrary,
    FittedAtlas,
    SelectedLocalModel,
    TrainedLocalCandidate,
    _validation_score,
)
from kkt_atlas.data import ObservedSplit
from kkt_atlas.problem import Array, DifferentiableStochasticProblem


@dataclass(frozen=True)
class SharedCheckpointCandidateIdentity:
    """Unique structural identity of one chart model in a frozen library."""

    chart: int
    ridge: float
    width: int
    support_size: int
    checkpoint_step: int


@dataclass(frozen=True)
class SharedCheckpointValidationTrace:
    """Whole-validation-sample score for one atlas-wide checkpoint step."""

    chart_count: int
    ridge: float
    checkpoint_step: int
    validation_loss_sum: float
    validation_indices: tuple[int, ...]
    routed_chart_ids: tuple[int, ...]
    validation_losses: tuple[float, ...]
    candidate_identities: tuple[SharedCheckpointCandidateIdentity, ...]

    def __post_init__(self) -> None:
        count = len(self.validation_indices)
        if self.chart_count < 1:
            raise ValueError("chart_count must be positive")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")
        if self.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be nonnegative")
        if self.validation_indices != tuple(range(count)):
            raise ValueError("validation_indices must preserve the complete original order")
        if len(self.routed_chart_ids) != count or any(
            chart < 0 or chart >= self.chart_count for chart in self.routed_chart_ids
        ):
            raise ValueError("routed_chart_ids must align with the validation sample")
        if len(self.validation_losses) != count or any(
            not math.isfinite(loss) for loss in self.validation_losses
        ):
            raise ValueError("validation_losses must be finite and align with the validation sample")
        if not math.isfinite(self.validation_loss_sum):
            raise ValueError("validation_loss_sum must be finite")
        if self.validation_loss_sum != math.fsum(self.validation_losses):
            raise ValueError("validation_loss_sum must be the unrounded total in original order")
        if len(self.candidate_identities) != self.chart_count:
            raise ValueError("candidate_identities must contain one model per chart")
        for chart, identity in enumerate(self.candidate_identities):
            if identity.chart != chart:
                raise ValueError("candidate_identities must follow chart order")
            if identity.ridge != self.ridge:
                raise ValueError("candidate identities must share the trace ridge")
            if identity.checkpoint_step != self.checkpoint_step:
                raise ValueError("candidate identities must share the trace checkpoint step")

    @property
    def validation_count(self) -> int:
        return len(self.validation_indices)

    @property
    def validation_mean_loss(self) -> float:
        return self.validation_loss_sum / self.validation_count


def select_shared_checkpoint_atlas(
    library: CandidateLibrary,
    validation: ObservedSplit,
    catalog: Array,
) -> FittedAtlas:
    """Select one checkpoint step without changing the frozen atlas structure."""

    fitted, _ = select_shared_checkpoint_atlas_with_trace(
        library,
        validation,
        catalog,
    )
    return fitted


def select_shared_checkpoint_atlas_with_trace(
    library: CandidateLibrary,
    validation: ObservedSplit,
    catalog: Array,
    *,
    validation_problem: DifferentiableStochasticProblem | None = None,
) -> tuple[FittedAtlas, tuple[SharedCheckpointValidationTrace, ...]]:
    """Score common steps on the full validation sample and retain an audit trace.

    The input library must already contain exactly one partition, one ridge, and
    one fixed width and support size per chart. Validation can therefore select
    only a checkpoint step, and the selected step is shared by every chart.
    """

    scoring_problem = library.problem if validation_problem is None else validation_problem
    partition, ridge, candidates_by_chart, checkpoint_steps = _fixed_checkpoint_path(
        library
    )
    assignments = partition.charts.assign(validation.contexts)
    validation_indices = tuple(range(len(validation.contexts)))
    routed_chart_ids = tuple(int(chart) for chart in assignments)
    indices_by_chart = tuple(
        np.flatnonzero(assignments == chart)
        for chart in range(partition.charts.chart_count)
    )
    outcomes = validation.outcomes(catalog)

    scored: list[tuple[FittedAtlas, SharedCheckpointValidationTrace]] = []
    for step in checkpoint_steps:
        full_losses = np.full(len(validation.contexts), np.nan, dtype=np.float64)
        selected_models: list[SelectedLocalModel] = []
        identities: list[SharedCheckpointCandidateIdentity] = []
        for chart, candidates in enumerate(candidates_by_chart):
            candidate = candidates[step]
            indices = indices_by_chart[chart]
            _, chart_losses = _validation_score(
                scoring_problem,
                partition.charts,
                candidate,
                validation.contexts[indices],
                outcomes[indices],
            )
            full_losses[indices] = chart_losses
            chart_loss_sum = math.fsum(chart_losses)
            selected_models.append(
                SelectedLocalModel(
                    candidate=candidate,
                    validation_loss_sum=chart_loss_sum,
                    validation_count=len(indices),
                )
            )
            identities.append(_candidate_identity(candidate))
        if np.any(~np.isfinite(full_losses)):
            raise FloatingPointError("shared-checkpoint validation produced nonfinite losses")
        losses = tuple(float(loss) for loss in full_losses)
        loss_sum = math.fsum(losses)
        trace = SharedCheckpointValidationTrace(
            chart_count=partition.charts.chart_count,
            ridge=ridge,
            checkpoint_step=step,
            validation_loss_sum=loss_sum,
            validation_indices=validation_indices,
            routed_chart_ids=routed_chart_ids,
            validation_losses=losses,
            candidate_identities=tuple(identities),
        )
        scored.append(
            (
                FittedAtlas(
                    problem=library.problem,
                    charts=partition.charts,
                    ridge=ridge,
                    local_models=tuple(selected_models),
                    validation_mean_loss=loss_sum / len(validation.contexts),
                ),
                trace,
            )
        )

    fitted, _ = min(
        scored,
        key=lambda item: (
            item[1].validation_loss_sum,
            item[1].checkpoint_step,
        ),
    )
    return fitted, tuple(trace for _, trace in scored)


def _fixed_checkpoint_path(
    library: CandidateLibrary,
) -> tuple[
    object,
    float,
    tuple[dict[int, TrainedLocalCandidate], ...],
    tuple[int, ...],
]:
    if len(library.partitions) != 1:
        raise ValueError("shared-checkpoint selection requires exactly one chart partition")
    partition = library.partitions[0]
    if len(partition.ridge_libraries) != 1:
        raise ValueError("shared-checkpoint selection requires exactly one ridge")
    ridge_library = partition.ridge_libraries[0]
    if len(ridge_library.local_candidates) != partition.charts.chart_count:
        raise ValueError("the frozen library must contain one candidate path per chart")

    candidates_by_chart: list[dict[int, TrainedLocalCandidate]] = []
    expected_steps: set[int] | None = None
    for chart, local_candidates in enumerate(ridge_library.local_candidates):
        if not local_candidates:
            raise ValueError("every chart must contain a nonempty checkpoint path")
        widths = {candidate.width for candidate in local_candidates}
        support_sizes = {candidate.support_size for candidate in local_candidates}
        if len(widths) != 1 or len(support_sizes) != 1:
            raise ValueError("validation cannot change a chart width or support size")
        by_step: dict[int, TrainedLocalCandidate] = {}
        for candidate in local_candidates:
            if candidate.chart != chart or candidate.ridge != ridge_library.ridge:
                raise ValueError("candidate metadata does not match its frozen chart path")
            if candidate.checkpoint.width != candidate.width:
                raise ValueError("candidate width does not match its checkpoint")
            if candidate.checkpoint.support_size != candidate.support_size:
                raise ValueError("candidate support size does not match its checkpoint")
            step = int(candidate.checkpoint.step)
            if step in by_step:
                raise ValueError("each chart must contain exactly one candidate per checkpoint step")
            by_step[step] = candidate
        steps = set(by_step)
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError("every chart must have the identical checkpoint-step set")
        candidates_by_chart.append(by_step)
    if not expected_steps:
        raise ValueError("the frozen library contains no checkpoint steps")
    return (
        partition,
        float(ridge_library.ridge),
        tuple(candidates_by_chart),
        tuple(sorted(expected_steps)),
    )


def _candidate_identity(
    candidate: TrainedLocalCandidate,
) -> SharedCheckpointCandidateIdentity:
    return SharedCheckpointCandidateIdentity(
        chart=int(candidate.chart),
        ridge=float(candidate.ridge),
        width=int(candidate.width),
        support_size=int(candidate.support_size),
        checkpoint_step=int(candidate.checkpoint.step),
    )
