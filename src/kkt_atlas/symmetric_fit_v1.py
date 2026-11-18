"""One-method fitting entry point for the frozen symmetric campaign."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kkt_atlas.atlas import (
    AtlasSearchConfig,
    balanced_local_batch_sizes,
    build_candidate_library,
    build_training_selected_candidate_library,
)
from kkt_atlas.automatic_joint_structure_v2 import (
    JointStructureConfig,
    select_joint_training_structure,
)
from kkt_atlas.baselines.confirmation_baseline_serialization import (
    save_confirmation_baseline_bundle,
    save_matched_observed_support_bundle,
)
from kkt_atlas.baselines.contextual_empirical import fit_knn_saa
from kkt_atlas.baselines.iceo import (
    ICEOModelConfig,
    ICEOTrainingConfig,
    fit_iceo,
)
from kkt_atlas.baselines.iceo_serialization import save_fitted_iceo
from kkt_atlas.baselines.matched_fixed_support import (
    fit_matched_observed_support_atlas,
)
from kkt_atlas.baselines.sof import SOFConfig, fit_sof, save_fitted_sof
from kkt_atlas.frozen_replay import (
    FrozenAtlasReplayConfig,
    FrozenChartSpecification,
)
from kkt_atlas.residual_audited_problem_v1 import ResidualAuditedProblem
from kkt_atlas.serialization import save_fitted_atlas
from kkt_atlas.shared_checkpoint_selection import (
    select_shared_checkpoint_atlas_with_trace,
)
from kkt_atlas.symmetric_campaign_protocol_v1 import (
    ANCHOR_ABLATIONS,
    DEVELOPMENT_SEEDS,
    PRIMARY_METHODS,
    CampaignJob,
    SymmetricCampaignProtocol,
    consumer_seed32,
    derive_seed64,
)
from kkt_atlas.symmetric_campaign_runtime_v1 import CampaignRuntimeInstance
from kkt_atlas.training import TrainingConfig
from kkt_atlas.windows_job_object_v1 import query_named_job_accounting


@dataclass(frozen=True)
class FitOutcome:
    record: dict[str, Any]
    training_wall_seconds: float
    training_process_seconds: float
    training_process_tree_cpu_seconds: float | None


def fit_campaign_job(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    model_directory: Path,
    *,
    dependency_result: dict[str, Any] | None = None,
    development_horizon_multiplier: int = 1,
) -> FitOutcome:
    if development_horizon_multiplier not in {1, 2}:
        raise ValueError("development_horizon_multiplier must be one or two")
    if job.kind != "development" and development_horizon_multiplier != 1:
        raise ValueError("a formal job cannot override its frozen horizon")
    if job.kind not in {
        "primary",
        "timing_stress",
        "ablation",
        "eta_sensitivity",
        "controlled",
        "development",
    }:
        raise NotImplementedError(f"fit kind is not wired yet: {job.kind}")
    _validate_job_instance(protocol, runtime, job)
    process_tree_started = _process_tree_cpu_seconds()
    wall_started = time.perf_counter()
    process_started = time.process_time()
    if job.application == "controlled":
        record, persist = _fit_controlled(
            protocol,
            runtime,
            job,
            model_directory,
        )
    elif job.method in {"kkt_atlas", "fixed_b1"}:
        record, persist = _fit_atlas(
            protocol,
            runtime,
            job,
            model_directory,
            chart_counts=(1,) if job.method == "fixed_b1" else (1, 2, 4),
            development_horizon_multiplier=development_horizon_multiplier,
        )
    elif job.method == "iceo_a":
        record, persist = _fit_iceo(
            protocol,
            runtime,
            job,
            model_directory,
            development_horizon_multiplier=development_horizon_multiplier,
        )
    elif job.method == "knn":
        record, persist = _fit_knn(protocol, runtime, job, model_directory)
    elif job.method == "sof":
        record, persist = _fit_sof(protocol, runtime, job, model_directory)
    elif job.method == "matched_observed_support":
        if dependency_result is None:
            raise ValueError("matched observed support requires its Atlas dependency")
        record, persist = _fit_matched(
            protocol,
            runtime,
            job,
            model_directory,
            dependency_result,
            development_horizon_multiplier=development_horizon_multiplier,
        )
    else:
        raise ValueError(f"unknown campaign method: {job.method}")
    training_wall_seconds = time.perf_counter() - wall_started
    training_process_seconds = time.process_time() - process_started
    process_tree_finished = _process_tree_cpu_seconds()
    training_process_tree_cpu_seconds = (
        None
        if process_tree_started is None or process_tree_finished is None
        else process_tree_finished - process_tree_started
    )
    if (
        training_process_tree_cpu_seconds is not None
        and training_process_tree_cpu_seconds < 0.0
    ):
        raise RuntimeError("process-tree CPU accounting moved backward")
    persist()
    return FitOutcome(
        record=record,
        training_wall_seconds=training_wall_seconds,
        training_process_seconds=training_process_seconds,
        training_process_tree_cpu_seconds=training_process_tree_cpu_seconds,
    )


def training_horizon(
    protocol: SymmetricCampaignProtocol,
    training_size: int,
    *,
    multiplier: int | None = None,
) -> tuple[int, ...]:
    multiplier = protocol.formal_horizon_multiplier if multiplier is None else multiplier
    if multiplier not in {1, 2}:
        raise ValueError("formal horizon multiplier is not frozen")
    base = math.ceil(
        protocol.raw["training"]["base_passes"]
        * training_size
        / protocol.raw["training"]["global_batch_size"]
    )
    horizon = multiplier * base
    return tuple(
        sorted(
            {
                0,
                math.ceil(horizon / 8),
                math.ceil(horizon / 4),
                math.ceil(horizon / 2),
                math.ceil(3 * horizon / 4),
                horizon,
            }
        )
    )


def knn_grid(training_size: int) -> tuple[int, ...]:
    if training_size < 1:
        raise ValueError("training_size must be positive")
    powers = [1]
    while powers[-1] * 2 <= training_size:
        powers.append(powers[-1] * 2)
    return tuple(sorted({*powers, training_size}))


def _fit_atlas(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    destination: Path,
    *,
    chart_counts: tuple[int, ...],
    development_horizon_multiplier: int,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    fitting = runtime.stored.fitting
    prescription_problem = _prescription_problem(protocol, runtime, job)
    structure = select_joint_training_structure(
        prescription_problem,
        fitting.training,
        fitting.catalog,
        _structure_config(protocol, job, chart_counts),
    )
    checkpoints = _job_horizon(
        protocol,
        job,
        len(fitting.training.contexts),
        development_horizon_multiplier,
    )
    training = protocol.raw["training"]
    library = build_training_selected_candidate_library(
        runtime.problem,
        fitting.training,
        fitting.catalog,
        structure.selected,
        ridge=job.eta,
        training_config=TrainingConfig(
            steps=checkpoints[-1],
            learning_rate=training["learning_rate"],
            save_steps=checkpoints,
            batch_size=0,
            coordinate_penalty=training["coordinate_penalty"],
            slope_penalty=training["probability_slope_penalty"],
            gradient_clip=training["gradient_clip"],
            train_scenario_coordinates=True,
            train_probability_slopes=True,
            seed=_seed(job, "atlas_training"),
        ),
        global_sample_presentations_per_step=training["global_batch_size"],
        seed=job.seed,
        probability_head_initialization="training_cluster_multinomial_logit",
        probability_head_maximum_iterations=protocol.raw["structure"][
            "probability_head_maximum_iterations"
        ],
    )
    fitted, trace = select_shared_checkpoint_atlas_with_trace(
        library,
        fitting.validation,
        fitting.catalog,
        validation_problem=prescription_problem,
    )
    selected_step = fitted.local_models[0].candidate.checkpoint.step
    return {
        "method": job.method,
        "selected_checkpoint_step": selected_step,
        "checkpoint_steps": list(checkpoints),
        "selected_B": structure.selected.chart_count,
        "selected_q": [item.representation_dim for item in structure.selected.charts],
        "selected_K": [item.selected_support_size for item in structure.selected.charts],
        "assessment_operational_loss": structure.selected.assessment_operational_loss,
        "validation_loss_sum": next(
            item.validation_loss_sum for item in trace if item.checkpoint_step == selected_step
        ),
        "checkpoint_validation": [
            {
                "step": item.checkpoint_step,
                "mean_operational_loss": item.validation_mean_loss,
            }
            for item in trace
        ],
        "structure": structure.diagnostics(),
    }, lambda: save_fitted_atlas(fitted, destination)


def _fit_iceo(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    destination: Path,
    *,
    development_horizon_multiplier: int,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    fitting = runtime.stored.fitting
    prescription_problem = _prescription_problem(protocol, runtime, job)
    checkpoints = _job_horizon(
        protocol,
        job,
        len(fitting.training.contexts),
        development_horizon_multiplier,
    )
    training = protocol.raw["training"]
    iceo = protocol.raw["iceo_a"]
    fitted = fit_iceo(
        runtime.problem,
        fitting,
        ridge=job.eta,
        model_config=ICEOModelConfig(
            hidden_width=iceo["hidden_width"],
            hidden_layers=iceo["hidden_layers"],
            seed=_seed(job, "iceo_model"),
        ),
        training_config=ICEOTrainingConfig(
            steps=checkpoints[-1],
            learning_rate=training["learning_rate"],
            save_steps=checkpoints,
            batch_size=training["global_batch_size"],
            gradient_clip=training["gradient_clip"],
            cross_entropy_weight=0.0,
            seed=_seed(job, "iceo_training"),
            microbatch_size=iceo["microbatch_size"],
        ),
        validation_problem=prescription_problem,
    )
    return {
        "method": job.method,
        "selected_checkpoint_step": fitted.selected_step,
        "checkpoint_steps": list(checkpoints),
        "validation_loss": fitted.validation_loss,
        "checkpoint_validation": [
            {"step": step, "mean_operational_loss": loss}
            for step, loss in fitted.validation_trace
        ],
        "support_size": fitting.support_size,
    }, lambda: save_fitted_iceo(fitted, destination)


def _fit_knn(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    destination: Path,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    fitting = runtime.stored.fitting
    prescription_problem = _prescription_problem(protocol, runtime, job)
    candidates = knn_grid(len(fitting.training.contexts))
    fitted = fit_knn_saa(
        runtime.problem,
        fitting,
        neighbor_counts=candidates,
        ridges=(job.eta,),
        validation_problem=prescription_problem,
    )
    return {
        "method": job.method,
        "candidate_neighbor_counts": list(candidates),
        "selected_neighbor_count": fitted.neighbor_count,
        "validation_loss": fitted.validation_loss,
    }, lambda: save_confirmation_baseline_bundle(fitted, None, destination)


def _fit_sof(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    destination: Path,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    fitting = runtime.stored.fitting
    controls = protocol.raw["sof"]
    config = SOFConfig(
        tree_count=controls["tree_count"],
        max_depth=controls["maximum_depth"],
        terminal_node_size=controls["terminal_node_size"],
        minimum_child_fraction=controls["minimum_child_fraction"],
        ridge=job.eta,
        split_rule=controls["split_rule"],
        active_tolerance=controls["active_set_tolerance"],
        feasibility_tolerance=controls["feasibility_tolerance"],
        seed=_seed(job, "sof_forest"),
        worker_count=controls["worker_count"],
    )
    fitted = fit_sof(
        runtime.problem,
        fitting.training.contexts,
        fitting.training.outcomes(fitting.catalog),
        config=config,
    )
    return {
        "method": job.method,
        "tree_count": len(fitted.trees),
        "worker_count": config.worker_count,
    }, lambda: save_fitted_sof(fitted, destination)


def _fit_matched(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    destination: Path,
    dependency: dict[str, Any],
    *,
    development_horizon_multiplier: int,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    dependency_job = dependency.get("job", {})
    dependency_model = dependency.get("model", {})
    if (
        dependency_job.get("application") != job.application
        or dependency_job.get("setting") != job.setting
        or dependency_job.get("seed") != job.seed
        or dependency_job.get("method") != "kkt_atlas"
        or dependency_job.get("eta") != job.eta
    ):
        raise ValueError("matched-support dependency is not the paired primary Atlas fit")
    selected_b = int(dependency_model["selected_B"])
    selected_q = tuple(int(value) for value in dependency_model["selected_q"])
    selected_k = tuple(int(value) for value in dependency_model["selected_K"])
    if len(selected_q) != selected_b or len(selected_k) != selected_b:
        raise ValueError("primary Atlas structure is incomplete")
    candidate_records = dependency_model["structure"]["examined_candidates"]
    selected_record = next(
        record for record in candidate_records if int(record["B"]) == selected_b
    )
    if len(selected_record["charts"]) != selected_b:
        raise ValueError("primary Atlas chart records are incomplete")
    training = protocol.raw["training"]
    structure = protocol.raw["structure"]
    fitting = runtime.stored.fitting
    checkpoints = _job_horizon(
        protocol,
        job,
        len(fitting.training.contexts),
        development_horizon_multiplier,
    )
    template = TrainingConfig(
        steps=checkpoints[-1],
        learning_rate=training["learning_rate"],
        save_steps=checkpoints,
        batch_size=0,
        coordinate_penalty=0.0,
        slope_penalty=training["probability_slope_penalty"],
        gradient_clip=training["gradient_clip"],
        train_scenario_coordinates=False,
        train_probability_slopes=True,
        seed=_seed(job, "matched_training"),
    )
    matched = fit_matched_observed_support_atlas(
        runtime.problem,
        fitting.training,
        fitting.catalog,
        FrozenAtlasReplayConfig(
            ridge=job.eta,
            charts=tuple(
                FrozenChartSpecification(width=q, support_size=k, stop_step=checkpoints[-1])
                for q, k in zip(selected_q, selected_k, strict=True)
            ),
            training=template,
            minimum_chart_size=structure["minimum_chart_size"],
            minimum_context_scale=structure["minimum_context_scale"],
            minimum_scenario_scale=structure["minimum_scenario_scale"],
            probability_head_initialization="training_cluster_multinomial_logit",
            execution_policy="grouped",
            chart_batch_sizes=balanced_local_batch_sizes(
                training["global_batch_size"],
                selected_b,
            ),
            seed=job.seed,
            probability_head_maximum_iterations=structure[
                "probability_head_maximum_iterations"
            ],
        ),
    )
    selected_step, validation_loss = _select_matched_checkpoint(
        matched,
        _prescription_problem(protocol, runtime, job),
        fitting,
        checkpoints,
    )
    return {
        "method": job.method,
        "selected_checkpoint_step": selected_step,
        "checkpoint_steps": list(checkpoints),
        "selected_B": selected_b,
        "selected_q": list(selected_q),
        "selected_K": list(selected_k),
        "validation_loss": validation_loss,
        "dependency_job_key": dependency["job_key"],
    }, lambda: save_matched_observed_support_bundle(
        matched,
        destination,
        selected_checkpoint_step=selected_step,
    )


def _fit_controlled(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
    destination: Path,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    controls = protocol.raw["controlled"]
    training = protocol.raw["training"]
    basis_method = {"cross": "kkt", "pca": "pca"}.get(job.method)
    if basis_method is None:
        raise ValueError("unknown controlled basis method")
    checkpoints = (0, 8, 16, 24, 32)
    paired_job = CampaignJob("controlled", "controlled", "paired", job.seed, "paired", job.eta)
    fitting = runtime.stored.fitting
    library = build_candidate_library(
        runtime.problem,
        fitting.training,
        fitting.catalog,
        AtlasSearchConfig(
            chart_counts=(controls["chart_count"],),
            ridges=(job.eta,),
            widths=(controls["width"],),
            support_sizes=(controls["local_support_size"],),
            training=TrainingConfig(
                steps=32,
                learning_rate=training["learning_rate"],
                save_steps=checkpoints,
                batch_size=0,
                coordinate_penalty=training["coordinate_penalty"],
                slope_penalty=training["probability_slope_penalty"],
                gradient_clip=training["gradient_clip"],
                seed=_seed(paired_job, "controlled_training"),
            ),
            minimum_chart_size=100,
            minimum_context_scale=protocol.raw["structure"]["minimum_context_scale"],
            minimum_scenario_scale=protocol.raw["structure"]["minimum_scenario_scale"],
            include_width_plus_one=False,
            basis_method=basis_method,
            probability_head_initialization="training_cluster_multinomial_logit",
            seed=_seed(paired_job, "controlled_atlas"),
            probability_head_maximum_iterations=protocol.raw["structure"][
                "probability_head_maximum_iterations"
            ],
        ),
    )
    fitted, trace = select_shared_checkpoint_atlas_with_trace(
        library,
        fitting.validation,
        fitting.catalog,
    )
    selected_step = fitted.local_models[0].candidate.checkpoint.step
    task_basis = runtime.stored.extra_arrays["task_basis"]
    basis_rows = []
    chart_sizes = np.bincount(
        fitted.charts.assignments,
        minlength=fitted.chart_count,
    )
    for chart, selected in enumerate(fitted.local_models):
        decomposition = selected.candidate.decomposition
        restored = decomposition.transformed_scale[:, None] * decomposition.basis(
            selected.candidate.width
        )
        restored, _ = np.linalg.qr(restored, mode="reduced")
        correlations = np.linalg.svd(task_basis.T @ restored, compute_uv=False)
        basis_rows.append(
            {
                "chart": chart,
                "training_size": int(chart_sizes[chart]),
                "minimum_canonical_correlation": float(np.min(correlations)),
            }
        )
    weighted_correlation = math.fsum(
        row["training_size"] * row["minimum_canonical_correlation"]
        for row in basis_rows
    ) / len(fitting.training.contexts)
    return {
        "method": job.method,
        "basis_method": basis_method,
        "selected_checkpoint_step": selected_step,
        "checkpoint_steps": list(checkpoints),
        "selected_B": fitted.chart_count,
        "selected_q": [item.candidate.width for item in fitted.local_models],
        "selected_K": [item.candidate.support_size for item in fitted.local_models],
        "validation_loss_sum": next(
            item.validation_loss_sum for item in trace if item.checkpoint_step == selected_step
        ),
        "basis_diagnostics": {
            "charts": basis_rows,
            "chart_size_weighted_minimum_canonical_correlation": weighted_correlation,
            "worst_chart_minimum_canonical_correlation": min(
                row["minimum_canonical_correlation"] for row in basis_rows
            ),
        },
    }, lambda: save_fitted_atlas(fitted, destination)


def _select_matched_checkpoint(
    matched: Any,
    problem: Any,
    fitting: Any,
    checkpoints: tuple[int, ...],
) -> tuple[int, float]:
    scored = []
    for step in checkpoints:
        for model, result in zip(
            matched.prepared_atlas.models,
            matched.training_results,
            strict=True,
        ):
            model.load_snapshot(result.checkpoint(step))
        decisions = _prepared_decisions_with_problem(
            matched.prepared_atlas,
            problem,
            fitting.validation.contexts,
        )
        with torch.no_grad():
            losses = problem.realized_loss_torch(
                torch.tensor(decisions, dtype=torch.float64),
                torch.tensor(
                    fitting.validation.outcomes(fitting.catalog),
                    dtype=torch.float64,
                ),
                torch.tensor(fitting.validation.contexts, dtype=torch.float64),
            ).cpu().numpy()
        scored.append((math.fsum(float(value) for value in losses), step))
    loss_sum, selected_step = min(scored)
    for model, result in zip(
        matched.prepared_atlas.models,
        matched.training_results,
        strict=True,
    ):
        model.load_snapshot(result.checkpoint(selected_step))
    return selected_step, loss_sum / len(fitting.validation.contexts)


def _prepared_decisions_with_problem(
    prepared: Any,
    problem: Any,
    contexts: np.ndarray,
) -> np.ndarray:
    assignments = prepared.charts.assign(contexts)
    decisions = np.empty(
        (len(contexts), problem.decision_dim),
        dtype=np.float64,
    )
    for chart, model in enumerate(prepared.models):
        indices = np.flatnonzero(assignments == chart)
        if len(indices) == 0:
            continue
        values = torch.tensor(contexts[indices], dtype=torch.float64)
        with torch.no_grad():
            decisions[indices] = (
                problem.solve_torch(
                    values,
                    model.scenarios(),
                    model.probabilities(values),
                    ridge=prepared.ridge,
                )
                .cpu()
                .numpy()
            )
    return decisions


def _prescription_problem(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
) -> Any:
    del protocol, job
    return ResidualAuditedProblem(runtime.problem)


def _structure_config(
    protocol: SymmetricCampaignProtocol,
    job: CampaignJob,
    chart_counts: tuple[int, ...],
) -> JointStructureConfig:
    structure = protocol.raw["structure"]
    return JointStructureConfig(
        chart_counts=chart_counts,
        ridge=job.eta,
        spectral_retained_ratio=structure["spectral_retained_ratio"],
        minimum_samples_per_parameter=structure["minimum_samples_per_parameter"],
        seed=job.seed,
        minimum_chart_size=structure["minimum_chart_size"],
        fold_count=structure["fold_count"],
        slope_penalty=protocol.raw["training"]["probability_slope_penalty"],
        maximum_head_iterations=structure["probability_head_maximum_iterations"],
        minimum_context_scale=structure["minimum_context_scale"],
        minimum_scenario_scale=structure["minimum_scenario_scale"],
        spectral_absolute_tolerance=structure["spectral_absolute_tolerance"],
        spectral_relative_tolerance=structure["spectral_relative_tolerance"],
        distinct_projected_tolerance=structure["distinct_projected_tolerance"],
        fail_closed=True,
        chart_count_rule=structure["chart_count_rule"],
        accept_intercept_only_fallback=structure["accept_intercept_only_fallback"],
        support_cap_offset=structure.get("support_cap_offset", 0),
    )


def _validate_job_instance(
    protocol: SymmetricCampaignProtocol,
    runtime: CampaignRuntimeInstance,
    job: CampaignJob,
) -> None:
    if job.kind == "development":
        allowed_method = (
            job.method in {"cross", "pca"}
            if job.application == "controlled"
            else job.method in {*PRIMARY_METHODS, *ANCHOR_ABLATIONS}
        )
        if job.seed not in DEVELOPMENT_SEEDS or not allowed_method or job.eta not in {
            0.0,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            0.1,
            0.3,
            1.0,
        }:
            raise ValueError("development job is outside the prelaunch protocol")
    else:
        matches = [
            candidate
            for candidate in protocol.jobs()
            if (
                candidate.kind,
                candidate.application,
                candidate.setting,
                candidate.seed,
                candidate.method,
                candidate.eta,
            )
            == (
                job.kind,
                job.application,
                job.setting,
                job.seed,
                job.method,
                job.eta,
            )
        ]
        if len(matches) != 1:
            raise ValueError("job is not declared by the frozen protocol")
    spec = runtime.stored.manifest["spec"]
    if job.application == "controlled":
        if spec.get("kind") != "controlled" or spec.get("noise") != job.setting:
            raise ValueError("controlled job and standalone instance differ")
        if int(spec.get("seed", -1)) != job.seed:
            raise ValueError("controlled job and standalone seed differ")
        return
    if (
        spec.get("application") != job.application
        or spec.get("setting") != job.setting
        or int(spec.get("seed", -1)) != job.seed
    ):
        raise ValueError("job and standalone instance differ")


def _seed(job: CampaignJob, consumer: str) -> int:
    # Couple the localization ablation to its primary model; other streams stay unchanged.
    seed_method = "kkt_atlas" if job.method == "fixed_b1" else job.method
    seed_kind = "primary" if job.method == "fixed_b1" and job.kind == "ablation" else job.kind
    master = derive_seed64(
        job.seed,
        seed_kind,
        job.application,
        job.setting,
        seed_method,
        job.eta,
    )
    return consumer_seed32(master, consumer)


def _job_horizon(
    protocol: SymmetricCampaignProtocol,
    job: CampaignJob,
    training_size: int,
    development_horizon_multiplier: int,
) -> tuple[int, ...]:
    return training_horizon(
        protocol,
        training_size,
        multiplier=(
            development_horizon_multiplier
            if job.kind == "development"
            else None
        ),
    )


def _process_tree_cpu_seconds() -> float | None:
    name = os.environ.get("KKT_ATLAS_JOB_OBJECT_NAME")
    if name is None:
        return None
    return query_named_job_accounting(name).total_cpu_seconds
