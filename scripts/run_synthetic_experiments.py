#!/usr/bin/env python3
"""Reproduce all synthetic benchmark experiments (Tables 3, 4, 5, 6, 9, 10, and EC.4).

Experiments:
  - `main`: 7-method quality comparison, selected Atlas structures, construction timing, and deployment timing
            across CAP (10 seeds), PORT (10 seeds), and NET (3 seeds).
  - `basis`: Local basis ablation (Cross-covariance vs Standardized PCA vs Full space) on CAP, PORT, NET.
  - `alpha`: Spectral retention threshold sensitivity (alpha_ret in {0.9, 0.99, 0.999, 0.9999}) on CAP, PORT.
  - `sample-scaling`: Training sample size scaling (|I_tr| in {1600, 3200, 6400, 12800}) on CAP, PORT.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from grid_reserve.evaluation import fit_sof_parallel
from grid_reserve.problem import (
    compress_knn_laws_wasserstein,
    reduce_scenarios_wasserstein_kmeans,
)
from kkt_atlas.atlas import (
    FittedAtlas,
    SelectedLocalModel,
    build_training_selected_candidate_library,
)
from kkt_atlas.automatic_joint_structure_v2 import (
    JointLocalSelection,
    JointPartitionSelection,
    select_joint_training_structure,
)
from kkt_atlas.automatic_structure import AutomaticPartitionCandidate
from kkt_atlas.baselines.contextual_empirical import KNearestNeighborSAA
from kkt_atlas.baselines.iceo import (
    FittedICEO,
    FullSupportICEOModel,
    ICEOModelConfig,
    ICEOSnapshot,
    ICEOTrainingConfig,
    ICEOTrainingResult,
    _batch_indices,
    _iceo_optimizer_step,
)
from kkt_atlas.baselines.sof import SOFConfig, _parent_model
from kkt_atlas.confirmation_data_v1 import FittingCatalogData
from kkt_atlas.data import ObservedSplit
from kkt_atlas.joint_training import train_local_models_jointly
from kkt_atlas.parallel_network_problem_v1 import ParallelNetworkProblem
from kkt_atlas.parallel_smooth_problem_v1 import ParallelSmoothProblem
from kkt_atlas.response_basis import fit_response_decomposition
from kkt_atlas.runtime_helpers import (
    evaluation_worker_init,
    evaluation_worker_task,
    generate_synthetic_application_instance,
    score_validation,
    solve_and_eval_one_context,
    solver_for,
    training_config,
    write_json_atomic,
)
from kkt_atlas.shared_checkpoint_selection import _fixed_checkpoint_path
from kkt_atlas.symmetric_campaign_protocol_v1 import (
    CampaignJob,
    derive_seed64,
)
from kkt_atlas.symmetric_fit_v1 import _seed, _structure_config, knn_grid
from kkt_atlas.symmetric_laws_v1 import FiniteLaw, predict_method_laws

DEFAULT_STEPS = (0, 400, 800, 1200, 1600)
DEFAULT_SEEDS = {
    "capacity": tuple(range(260911, 260921)),
    "portfolio": tuple(range(260911, 260921)),
    "network": (260911, 260912, 260913),
}


def atlas_parameter_count(structure: JointPartitionSelection, context_dim: int) -> int:
    return int(
        sum(
            int(ch.selected_support_size) * int(ch.representation_dim)
            + (int(ch.selected_support_size) - 1) * (int(context_dim) + 1)
            for ch in structure.charts
        )
    )


def open_parallel_solver(problem: Any, app: str, *, workers: int, for_atlas: bool = False) -> Any:
    if app == "network":
        eff_w = min(8, workers) if for_atlas else workers
        ps = ParallelNetworkProblem(problem, workers=eff_w, parallel_derivatives=True, solver_threads=1)
    else:
        ps = ParallelSmoothProblem(solver_for(problem), workers=workers)
    ps.__enter__()
    return ps


def evaluate_laws_exact(
    app: str,
    problem: Any,
    laws: tuple[FiniteLaw, ...],
    catalog: np.ndarray,
    eval_contexts: np.ndarray,
    eval_probs: np.ndarray,
    eta: float,
    ref_risks: np.ndarray | None,
    *,
    workers: int,
) -> dict[str, Any]:
    tasks = [
        (app, x, law.scenarios, law.probabilities, eta, catalog, prob)
        for x, law, prob in zip(eval_contexts, laws, eval_probs, strict=True)
    ]
    if workers <= 1:
        serial = solver_for(problem)
        rows = [solve_and_eval_one_context(serial, t) for t in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=evaluation_worker_init,
            initargs=(problem,),
        ) as pool:
            rows = list(pool.map(evaluation_worker_task, tasks))
    risks = np.asarray([r[3] for r in rows], dtype=np.float64)
    max_res = float(max(r[1] for r in rows))
    supp_sizes = [int(len(lw.probabilities)) for lw in laws]
    out: dict[str, Any] = {
        "risk": float(np.mean(risks)),
        "maximum_acceptance_residual": max_res,
        "mean_query_support": float(np.mean(supp_sizes)),
        "max_query_support": int(max(supp_sizes)),
        "context_risks": risks,
    }
    if ref_risks is not None:
        ref_mean = float(np.mean(ref_risks))
        out["reference_risk"] = ref_mean
        out["relative_excess_percent"] = float(100.0 * (out["risk"] - ref_mean) / abs(ref_mean))
    return out


def compute_reference_risks(
    app: str,
    problem: Any,
    catalog: np.ndarray,
    eval_contexts: np.ndarray,
    eval_probs: np.ndarray,
    *,
    workers: int,
) -> np.ndarray:
    ref_laws = tuple(
        FiniteLaw(catalog, p, np.arange(len(catalog), dtype=np.int64), "catalog")
        for p in eval_probs
    )
    res = evaluate_laws_exact(
        app, problem, ref_laws, catalog, eval_contexts, eval_probs, 0.0, None, workers=workers
    )
    return np.asarray(res["context_risks"], dtype=np.float64)


def select_atlas_structure(
    problem: Any,
    fitting: FittingCatalogData,
    protocol: Any,
    *,
    app: str,
    seed: int,
    alpha_ret: float = 0.9999,
    chart_counts: tuple[int, ...] = (1, 2, 4),
) -> JointPartitionSelection:
    eta = protocol.eta(app)
    job = CampaignJob("replication", app, "anchor", seed, "kkt_atlas", eta)
    base_cfg = _structure_config(protocol, job, chart_counts)
    # Filter candidate B values whose smallest chart violates minimum_chart_size on small samples
    cfg = replace(base_cfg, spectral_retained_ratio=float(alpha_ret), fail_closed=False)
    res = select_joint_training_structure(solver_for(problem), fitting.training, fitting.catalog, cfg)
    return res.selected


def fit_atlas_on_structure(
    problem: Any,
    fitting: FittingCatalogData,
    protocol: Any,
    structure: JointPartitionSelection,
    par_solver: Any,
    *,
    app: str,
    seed: int,
    steps: tuple[int, ...] = DEFAULT_STEPS,
) -> dict[str, Any]:
    eta = protocol.eta(app)
    d_x = int(problem.context_dim)
    init_library = build_training_selected_candidate_library(
        problem,
        fitting.training,
        fitting.catalog,
        structure,
        ridge=eta,
        training_config=training_config(protocol, "A", (0,), seed=seed),
        global_sample_presentations_per_step=64,
        seed=seed,
        probability_head_initialization="training_cluster_multinomial_logit",
        probability_head_maximum_iterations=protocol.raw["structure"]["probability_head_maximum_iterations"],
        training_function=lambda prob, *a, **kw: train_local_models_jointly(par_solver, *a, **kw),
    )
    _, _, cached_paths, _ = _fixed_checkpoint_path(init_library)
    cached = tuple(path[0] for path in cached_paths)

    traces: list[dict[str, Any]] = []
    val_losses_by_step: dict[int, np.ndarray] = {}

    def train_arm_b(prob: Any, models: Any, contexts: Any, outcomes: Any, *, ridge: float, configs: Any) -> Any:
        for model, cand in zip(models, cached, strict=True):
            model.load_snapshot(cand.checkpoint)
        proxy = SimpleNamespace(problem=prob, charts=structure.partition.context_charts, models=models)

        def on_ckpt(step: int, snapshots: Any) -> None:
            del snapshots
            losses, _, _, residuals = score_validation(
                prob, fitting, proxy, "kkt_atlas", eta, scoring_solver=par_solver
            )
            val_losses_by_step[int(step)] = losses.copy()
            traces.append(
                {
                    "step": int(step),
                    "validation_loss": float(losses.mean()),
                    "maximum_acceptance_residual": float(residuals["maximum_acceptance_residual"]),
                }
            )

        return train_local_models_jointly(
            par_solver, models, contexts, outcomes, ridge=ridge, configs=configs, checkpoint_callback=on_ckpt
        )

    library = build_training_selected_candidate_library(
        problem,
        fitting.training,
        fitting.catalog,
        structure,
        ridge=eta,
        training_config=training_config(protocol, "B", steps, seed=seed),
        global_sample_presentations_per_step=64,
        seed=seed,
        probability_head_initialization="training_cluster_multinomial_logit",
        probability_head_maximum_iterations=protocol.raw["structure"]["probability_head_maximum_iterations"],
        training_function=train_arm_b,
    )
    partition, ridge_val, candidates, _ = _fixed_checkpoint_path(library)
    assignments = partition.charts.assign(fitting.validation.contexts)
    selected = min(traces, key=lambda r: (r["validation_loss"], r["step"]))
    sel_step = int(selected["step"])
    losses_sel = val_losses_by_step[sel_step]

    fitted = FittedAtlas(
        problem=problem,
        charts=partition.charts,
        ridge=ridge_val,
        local_models=tuple(
            SelectedLocalModel(
                candidate=replace(candidates[i][sel_step], initialization=cached[i].initialization),
                validation_loss_sum=math.fsum(losses_sel[assignments == i]),
                validation_count=int(np.sum(assignments == i)),
            )
            for i in range(len(candidates))
        ),
        validation_mean_loss=float(losses_sel.mean()),
    )
    k_list = [int(ch.selected_support_size) for ch in structure.charts]
    d_list = [int(ch.representation_dim) for ch in structure.charts]
    return {
        "fitted": fitted,
        "prepared": fitted.prepare(),
        "selected_B": int(structure.chart_count),
        "selected_q": d_list,
        "selected_K": k_list,
        "max_K": int(max(k_list)),
        "stored_K": int(sum(k_list)),
        "parameter_count": atlas_parameter_count(structure, d_x),
        "selected_step": sel_step,
        "validation_loss": float(selected["validation_loss"]),
    }


def build_basis_variant_structure(
    problem: Any,
    fitting: FittingCatalogData,
    base_structure: JointPartitionSelection,
    variant: str,
    *,
    eta: float = 0.0,
) -> JointPartitionSelection:
    """Replace only the local basis (PCA or Full space), keeping chart partition and K_b fixed."""
    if variant == "kkt":
        return base_structure
    basis_method = "pca" if variant == "pca" else "identity"
    charts = base_structure.partition.context_charts
    outcomes = fitting.training.outcomes(fitting.catalog)
    m_dim = int(problem.scenario_dim)
    slv = solver_for(problem)
    new_decomps = []
    new_locals = []
    for b_idx, ch in enumerate(base_structure.charts):
        indices = charts.indices(b_idx)
        decomp = fit_response_decomposition(
            slv,
            charts.centers[b_idx],
            outcomes[indices],
            ridge=eta,
            minimum_scenario_scale=1e-8,
            basis_method=basis_method,
        )
        new_decomps.append(decomp)
        k_b = int(ch.selected_support_size)
        d_b = int(ch.representation_dim)
        if variant == "pca":
            new_locals.append(
                replace(
                    ch,
                    representation_dim=d_b,
                    protocol_support_cap=k_b,
                    production_support_cap=max(int(ch.production_support_cap), k_b),
                    selected_support_size=k_b,
                    support_cap_offset=0,
                )
            )
        else:
            new_locals.append(
                replace(
                    ch,
                    representation_dim=m_dim,
                    protocol_support_cap=m_dim + 1,
                    production_support_cap=m_dim + 1,
                    selected_support_size=k_b,
                    support_cap_offset=0,
                )
            )
    new_part = AutomaticPartitionCandidate(
        context_charts=charts,
        decompositions=tuple(new_decomps),
        spectral_selections=base_structure.partition.spectral_selections,
        diagnostic=base_structure.partition.diagnostic,
    )
    return replace(base_structure, partition=new_part, charts=tuple(new_locals))


def train_iceo_fast(
    par_solver: Any,
    catalog: np.ndarray,
    training: ObservedSplit,
    validation: ObservedSplit,
    fitting: FittingCatalogData,
    problem: Any,
    *,
    eta: float,
    seed: int,
    steps: tuple[int, ...],
) -> tuple[FittedICEO, dict[str, Any]]:
    job_seed = derive_seed64(seed, "iceo", int(len(catalog)))
    m_cfg = ICEOModelConfig(hidden_width=32, hidden_layers=2, seed=int(job_seed & 0x7FFFFFFF))
    t_cfg = ICEOTrainingConfig(
        steps=int(steps[-1]),
        learning_rate=0.001,
        save_steps=tuple(int(s) for s in steps),
        batch_size=64,
        gradient_clip=10.0,
        seed=int((job_seed >> 17) & 0x7FFFFFFF),
        microbatch_size=32,
    )
    model = FullSupportICEOModel(
        par_solver, catalog, training.contexts, config=m_cfg, dtype=torch.float64, device="cpu"
    )
    contexts_t = torch.tensor(training.contexts, dtype=torch.float64)
    std_t = model.standardized_contexts(contexts_t)
    outcomes_t = torch.tensor(training.outcomes(catalog), dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=t_cfg.learning_rate, foreach=False, fused=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(t_cfg.seed))
    save_set = set(t_cfg.save_steps)
    snaps: list[ICEOSnapshot] = [model.snapshot(step=0, training_loss=0.0)]
    for step in range(1, t_cfg.steps + 1):
        indices = _batch_indices(len(contexts_t), t_cfg.batch_size, generator=gen, device=model.catalog.device)
        bx, bstd, bxi = (
            (contexts_t, std_t, outcomes_t)
            if indices is None
            else (contexts_t[indices], std_t[indices], outcomes_t[indices])
        )
        _iceo_optimizer_step(par_solver, model, opt, bx, bstd, bxi, None, ridge=eta, config=t_cfg, step=step)
        if step in save_set:
            snaps.append(model.snapshot(step=step, training_loss=0.0))

    best_loss = math.inf
    best_snap = snaps[0]
    for sn in snaps:
        model.load_snapshot(sn)
        probs_val = model.probabilities_numpy(validation.contexts)
        laws_val = tuple(
            FiniteLaw(catalog, p, np.arange(len(catalog), dtype=np.int64), "catalog")
            for p in probs_val
        )
        z_val, _ = par_solver.solve_laws_with_audit(validation.contexts, laws_val, ridge=eta)
        losses_val = (
            par_solver.realized_loss_torch(
                torch.tensor(z_val, dtype=torch.float64),
                torch.tensor(validation.outcomes(catalog), dtype=torch.float64),
                torch.tensor(validation.contexts, dtype=torch.float64),
            )
            .detach()
            .cpu()
            .numpy()
        )
        v_mean = float(np.mean(losses_val))
        if v_mean < best_loss:
            best_loss = v_mean
            best_snap = sn

    model.load_snapshot(best_snap)
    fitted = FittedICEO(
        model=model,
        checkpoint=best_snap,
        validation_loss=best_loss,
        ridge=eta,
        training_result=ICEOTrainingResult(checkpoints=tuple(snaps), final_step=t_cfg.steps),
    )
    return fitted, {"selected_step": int(best_snap.step), "validation_loss": best_loss}


def run_all_7_methods_on_instance(
    app: str,
    seed: int,
    *,
    training_size: int,
    steps: tuple[int, ...],
    workers: int,
    chart_counts: tuple[int, ...] = (1, 2, 4),
    eval_limit: int | None = None,
) -> dict[str, Any]:
    proto, problem, fitting, test = generate_synthetic_application_instance(
        app, seed, training_size=training_size
    )
    if eval_limit is not None:
        test = replace(
            test,
            contexts=test.contexts[:eval_limit].copy(),
            probabilities=test.probabilities[:eval_limit].copy(),
        )
    eta = proto.eta(app)
    ref_risks = compute_reference_risks(
        app, problem, fitting.catalog, test.contexts, test.probabilities, workers=workers
    )

    par_solver = open_parallel_solver(problem, app, workers=workers, for_atlas=True)
    try:
        t0 = time.perf_counter()
        structure = select_atlas_structure(
            problem, fitting, proto, app=app, seed=seed, chart_counts=chart_counts
        )
        atlas_fit = fit_atlas_from_structure_wrapper(
            problem, fitting, proto, structure, par_solver, app=app, seed=seed, steps=steps
        )
        atlas_wall = time.perf_counter() - t0
    finally:
        par_solver.__exit__(None, None, None)

    par_eval = open_parallel_solver(problem, app, workers=workers, for_atlas=False)
    try:
        methods_out: dict[str, Any] = {}
        # 1. Atlas
        atlas_laws = predict_method_laws(atlas_fit["prepared"], "kkt_atlas", test.contexts, fitting)
        m_atlas = evaluate_laws_exact(
            app, problem, atlas_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_atlas.pop("context_risks")
        m_atlas.update(
            {
                "selected_B": atlas_fit["selected_B"],
                "selected_q": atlas_fit["selected_q"],
                "selected_K": atlas_fit["selected_K"],
                "parameter_count": atlas_fit["parameter_count"],
                "construction_wall_seconds": atlas_wall,
            }
        )
        methods_out["atlas"] = m_atlas

        # 2. ICEO (full reference support)
        t0 = time.perf_counter()
        iceo_fitted, iceo_meta = train_iceo_fast(
            par_eval, fitting.catalog, fitting.training, fitting.validation, fitting, problem,
            eta=eta, seed=seed, steps=steps,
        )
        iceo_wall = time.perf_counter() - t0
        iceo_probs = iceo_fitted.model.probabilities_numpy(test.contexts)
        iceo_laws = tuple(
            FiniteLaw(fitting.catalog, p, np.arange(len(fitting.catalog), dtype=np.int64), "catalog")
            for p in iceo_probs
        )
        m_iceo = evaluate_laws_exact(
            app, problem, iceo_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_iceo.pop("context_risks")
        m_iceo.update({**iceo_meta, "construction_wall_seconds": iceo_wall})
        methods_out["iceo"] = m_iceo

        # 3. ICEO-red (2-Wasserstein scenario reduction matched to K_match = max_b K_b)
        k_match = int(atlas_fit["max_K"])
        xi_train = fitting.training.outcomes(fitting.catalog)
        t0 = time.perf_counter()
        sr_cat, _ = reduce_scenarios_wasserstein_kmeans(
            problem, xi_train, k_match, seed=int(derive_seed64(seed, app, "iceo_sr") & 0x7FFFFFFF)
        )
        # Re-index training/validation for validation loss evaluation against true observed outcomes
        iceo_sr_fitted, iceo_sr_meta = _train_iceo_reduced(
            par_eval, sr_cat, fitting,eta=eta, seed=seed + 10000, steps=steps
        )
        iceo_sr_wall = time.perf_counter() - t0
        sr_probs = iceo_sr_fitted.probabilities_numpy(test.contexts)
        sr_laws = tuple(
            FiniteLaw(sr_cat, p, np.arange(len(sr_cat), dtype=np.int64), "local")
            for p in sr_probs
        )
        m_iceo_sr = evaluate_laws_exact(
            app, problem, sr_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_iceo_sr.pop("context_risks")
        m_iceo_sr.update({**iceo_sr_meta, "construction_wall_seconds": iceo_sr_wall})
        methods_out["iceo_red"] = m_iceo_sr

        # 4. SOF
        t0 = time.perf_counter()
        sof_cfg = SOFConfig(
            ridge=eta,
            tree_count=100 if len(steps) > 2 else 10,
            bootstrap=True,
            honesty=False,
            minimum_child_fraction=0.2,
            terminal_node_size=10,
            maximum_depth=100,
            split_rule="approximate_risk",
            candidate_rule="all_midpoints",
            seed=int(derive_seed64(seed, app, "sof") & 0x7FFFFFFF),
            worker_count=min(8, workers),
        )
        sof_parent = _parent_model(solver_for(problem))
        sof_fitted = fit_sof_parallel(sof_parent, fitting.training.contexts, xi_train, sof_cfg)
        sof_wall = time.perf_counter() - t0
        sof_weights = sof_fitted.weights_matrix(test.contexts)
        sof_laws = _weights_to_laws(xi_train, sof_weights)
        m_sof = evaluate_laws_exact(
            app, problem, sof_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_sof.pop("context_risks")
        m_sof["construction_wall_seconds"] = sof_wall
        methods_out["sof"] = m_sof

        # 5. kNN & 6. kNN-red
        t0 = time.perf_counter()
        best_k = 1
        best_knn_val = math.inf
        best_knn_model = None
        for k_cand in knn_grid(len(fitting.training.contexts)):
            cand_m = KNearestNeighborSAA(
                fitting.training.contexts,
                fitting.training.outcome_indices,
                neighbors=int(k_cand),
                standardize_contexts=True,
            )
            laws_v = predict_method_laws(cand_m, "knn", fitting.validation.contexts, fitting)
            z_v, _ = par_eval.solve_laws_with_audit(fitting.validation.contexts, laws_v, ridge=eta)
            lv = (
                par_eval.realized_loss_torch(
                    torch.tensor(z_v, dtype=torch.float64),
                    torch.tensor(fitting.validation.outcomes(fitting.catalog), dtype=torch.float64),
                    torch.tensor(fitting.validation.contexts, dtype=torch.float64),
                )
                .detach()
                .cpu()
                .numpy()
            )
            mv = float(np.mean(lv))
            if mv < best_knn_val:
                best_knn_val = mv
                best_k = int(k_cand)
                best_knn_model = cand_m
        knn_wall = time.perf_counter() - t0
        assert best_knn_model is not None
        knn_laws = predict_method_laws(best_knn_model, "knn", test.contexts, fitting)
        m_knn = evaluate_laws_exact(
            app, problem, knn_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_knn.pop("context_risks")
        m_knn.update({"selected_k": best_k, "construction_wall_seconds": knn_wall})
        methods_out["knn"] = m_knn

        knn_red_laws = compress_knn_laws_wasserstein(
            problem, knn_laws, fitting.catalog, k_match,
            seed=int(derive_seed64(seed, app, "knn_sr") & 0x7FFFFFFF),
        )
        m_knn_red = evaluate_laws_exact(
            app, problem, knn_red_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_knn_red.pop("context_risks")
        m_knn_red.update({"selected_k": best_k, "construction_wall_seconds": knn_wall})
        methods_out["knn_red"] = m_knn_red

        # 7. Pooled SAA
        uniq_idx, counts = np.unique(fitting.training.outcome_indices, return_counts=True)
        saa_law = FiniteLaw(
            fitting.catalog[uniq_idx],
            counts.astype(np.float64) / float(counts.sum()),
            uniq_idx,
            "catalog",
        )
        saa_laws = tuple(saa_law for _ in range(len(test.contexts)))
        m_saa = evaluate_laws_exact(
            app, problem, saa_laws, fitting.catalog, test.contexts, test.probabilities, eta, ref_risks, workers=workers
        )
        m_saa.pop("context_risks")
        methods_out["saa"] = m_saa
    finally:
        par_eval.__exit__(None, None, None)

    return {"application": app, "seed": seed, "training_size": training_size, "methods": methods_out}


def fit_atlas_from_structure_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return fit_atlas_on_structure(*args, **kwargs)


def _train_iceo_reduced(
    par_solver: Any,
    reduced_catalog: np.ndarray,
    fitting: FittingCatalogData,
    *,
    eta: float,
    seed: int,
    steps: tuple[int, ...],
) -> tuple[FullSupportICEOModel, dict[str, Any]]:
    m_cfg = ICEOModelConfig(hidden_width=32, hidden_layers=2, seed=int(seed & 0x7FFFFFFF))
    t_cfg = ICEOTrainingConfig(
        steps=int(steps[-1]),
        learning_rate=0.001,
        save_steps=tuple(int(s) for s in steps),
        batch_size=64,
        gradient_clip=10.0,
        seed=int((seed + 7919) & 0x7FFFFFFF),
        microbatch_size=32,
    )
    model = FullSupportICEOModel(
        par_solver, reduced_catalog, fitting.training.contexts, config=m_cfg, dtype=torch.float64, device="cpu"
    )
    contexts_t = torch.tensor(fitting.training.contexts, dtype=torch.float64)
    std_t = model.standardized_contexts(contexts_t)
    outcomes_t = torch.tensor(fitting.training.outcomes(fitting.catalog), dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=t_cfg.learning_rate, foreach=False, fused=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(t_cfg.seed))
    save_set = set(t_cfg.save_steps)
    snaps: list[ICEOSnapshot] = [model.snapshot(step=0, training_loss=0.0)]
    for step in range(1, t_cfg.steps + 1):
        indices = _batch_indices(len(contexts_t), t_cfg.batch_size, generator=gen, device=model.catalog.device)
        bx, bstd, bxi = (
            (contexts_t, std_t, outcomes_t)
            if indices is None
            else (contexts_t[indices], std_t[indices], outcomes_t[indices])
        )
        _iceo_optimizer_step(par_solver, model, opt, bx, bstd, bxi, None, ridge=eta, config=t_cfg, step=step)
        if step in save_set:
            snaps.append(model.snapshot(step=step, training_loss=0.0))

    best_loss = math.inf
    best_snap = snaps[0]
    val_xi_true = fitting.validation.outcomes(fitting.catalog)
    for sn in snaps:
        model.load_snapshot(sn)
        probs_v = model.probabilities_numpy(fitting.validation.contexts)
        laws_v = tuple(
            FiniteLaw(reduced_catalog, p, np.arange(len(reduced_catalog), dtype=np.int64), "local")
            for p in probs_v
        )
        z_v, _ = par_solver.solve_laws_with_audit(fitting.validation.contexts, laws_v, ridge=eta)
        lv = (
            par_solver.realized_loss_torch(
                torch.tensor(z_v, dtype=torch.float64),
                torch.tensor(val_xi_true, dtype=torch.float64),
                torch.tensor(fitting.validation.contexts, dtype=torch.float64),
            )
            .detach()
            .cpu()
            .numpy()
        )
        mv = float(np.mean(lv))
        if mv < best_loss:
            best_loss = mv
            best_snap = sn
    model.load_snapshot(best_snap)
    return model, {"selected_step": int(best_snap.step), "validation_loss": best_loss}


def _weights_to_laws(xi_train: np.ndarray, weights_matrix: np.ndarray) -> tuple[FiniteLaw, ...]:
    laws: list[FiniteLaw] = []
    for row in weights_matrix:
        nz = np.flatnonzero(row > 0.0)
        p = row[nz].astype(np.float64)
        p /= p.sum()
        laws.append(FiniteLaw(xi_train[nz], p, nz, "local"))
    return tuple(laws)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment",
        choices=("main", "basis", "alpha", "sample-scaling"),
        default="main",
        help="Synthetic experiment suite to execute.",
    )
    parser.add_argument(
        "--applications",
        nargs="+",
        choices=("capacity", "portfolio", "network"),
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "synthetic")
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Fast end-to-end smoke check (n=400, 20 steps, 16 eval contexts, 1 seed).",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    steps = (0, 20) if args.quick_smoke else DEFAULT_STEPS
    n_train = 400 if args.quick_smoke else 12800
    chart_counts = (1, 2) if args.quick_smoke else (1, 2, 4)
    eval_limit = 16 if args.quick_smoke else None

    if args.experiment == "main":
        apps = args.applications or (("capacity",) if args.quick_smoke else ("capacity", "portfolio", "network"))
        records = []
        for app in apps:
            seeds = args.seeds or ((260911,) if args.quick_smoke else DEFAULT_SEEDS[app])
            for seed in seeds:
                rec = run_all_7_methods_on_instance(
                    app, int(seed), training_size=n_train, steps=steps,
                    workers=args.workers, chart_counts=chart_counts, eval_limit=eval_limit,
                )
                records.append(rec)
                print(
                    json.dumps(
                        {
                            "experiment": "main",
                            "app": app,
                            "seed": seed,
                            "atlas_excess_pct": round(rec["methods"]["atlas"]["relative_excess_percent"], 4),
                            "iceo_excess_pct": round(rec["methods"]["iceo"]["relative_excess_percent"], 4),
                        }
                    ),
                    flush=True,
                )
        write_json_atomic(args.output_dir / "synthetic_main_run.json", {"records": records})

    elif args.experiment == "basis":
        apps = args.applications or (("capacity",) if args.quick_smoke else ("capacity", "portfolio", "network"))
        records = []
        for app in apps:
            seeds = args.seeds or ((260911,) if args.quick_smoke else DEFAULT_SEEDS[app])
            for seed in seeds:
                proto, problem, fitting, test = generate_synthetic_application_instance(
                    app, int(seed), training_size=n_train
                )
                if eval_limit is not None:
                    test = replace(test, contexts=test.contexts[:eval_limit], probabilities=test.probabilities[:eval_limit])
                ref_risks = compute_reference_risks(
                    app, problem, fitting.catalog, test.contexts, test.probabilities, workers=args.workers
                )
                base_st = select_atlas_structure(
                    problem, fitting, proto, app=app, seed=int(seed), chart_counts=chart_counts
                )
                par_s = open_parallel_solver(problem, app, workers=args.workers, for_atlas=True)
                try:
                    var_out = {}
                    for var in ("kkt", "pca", "full"):
                        st_v = build_basis_variant_structure(
                            problem, fitting, base_st, var, eta=proto.eta(app)
                        )
                        fit_v = fit_atlas_on_structure(
                            problem, fitting, proto, st_v, par_s, app=app, seed=int(seed), steps=steps
                        )
                        laws_v = predict_method_laws(fit_v["prepared"], "kkt_atlas", test.contexts, fitting)
                        ev_v = evaluate_laws_exact(
                            app, problem, laws_v, fitting.catalog, test.contexts, test.probabilities,
                            proto.eta(app), ref_risks, workers=args.workers,
                        )
                        ev_v.pop("context_risks")
                        ev_v["parameter_count"] = fit_v["parameter_count"]
                        var_out[var] = ev_v
                    records.append({"application": app, "seed": int(seed), "variants": var_out})
                    print(json.dumps({"experiment": "basis", "app": app, "seed": seed,
                                      "kkt": round(var_out["kkt"]["relative_excess_percent"], 4),
                                      "pca": round(var_out["pca"]["relative_excess_percent"], 4)}), flush=True)
                finally:
                    par_s.__exit__(None, None, None)
        write_json_atomic(args.output_dir / "synthetic_basis_run.json", {"records": records})

    elif args.experiment == "alpha":
        apps = args.applications or (("capacity",) if args.quick_smoke else ("capacity", "portfolio"))
        alphas = (0.9, 0.9999) if args.quick_smoke else (0.9, 0.99, 0.999, 0.9999)
        records = []
        for app in apps:
            seeds = args.seeds or ((260911,) if args.quick_smoke else DEFAULT_SEEDS[app])
            for seed in seeds:
                proto, problem, fitting, test = generate_synthetic_application_instance(
                    app, int(seed), training_size=n_train
                )
                if eval_limit is not None:
                    test = replace(test, contexts=test.contexts[:eval_limit], probabilities=test.probabilities[:eval_limit])
                ref_risks = compute_reference_risks(
                    app, problem, fitting.catalog, test.contexts, test.probabilities, workers=args.workers
                )
                par_s = open_parallel_solver(problem, app, workers=args.workers, for_atlas=True)
                try:
                    a_out = {}
                    for alpha in alphas:
                        st_a = select_atlas_structure(
                            problem, fitting, proto, app=app, seed=int(seed),
                            alpha_ret=alpha, chart_counts=chart_counts,
                        )
                        fit_a = fit_atlas_on_structure(
                            problem, fitting, proto, st_a, par_s, app=app, seed=int(seed), steps=steps
                        )
                        laws_a = predict_method_laws(fit_a["prepared"], "kkt_atlas", test.contexts, fitting)
                        ev_a = evaluate_laws_exact(
                            app, problem, laws_a, fitting.catalog, test.contexts, test.probabilities,
                            proto.eta(app), ref_risks, workers=args.workers,
                        )
                        ev_a.pop("context_risks")
                        ev_a["max_K"] = fit_a["max_K"]
                        a_out[str(alpha)] = ev_a
                    records.append({"application": app, "seed": int(seed), "alphas": a_out})
                    print(json.dumps({"experiment": "alpha", "app": app, "seed": seed}), flush=True)
                finally:
                    par_s.__exit__(None, None, None)
        write_json_atomic(args.output_dir / "synthetic_alpha_run.json", {"records": records})

    elif args.experiment == "sample-scaling":
        apps = args.applications or (("capacity",) if args.quick_smoke else ("capacity", "portfolio"))
        sizes = (400, 800) if args.quick_smoke else (1600, 3200, 6400, 12800)
        seeds = args.seeds or ((260911,) if args.quick_smoke else (260911, 260912, 260913))
        records = []
        for app in apps:
            for seed in seeds:
                # Generate master 12,800 sample once and take nested training prefixes per EC.3.6
                proto, problem, fit_full, test = generate_synthetic_application_instance(
                    app, int(seed), training_size=max(sizes)
                )
                for n_sz in sizes:
                    rec = run_all_7_methods_on_instance(
                        app, int(seed), training_size=int(n_sz), steps=steps,
                        workers=args.workers, chart_counts=chart_counts, eval_limit=eval_limit,
                    )
                    records.append(rec)
                    print(json.dumps({"experiment": "sample-scaling", "app": app, "seed": seed, "n": n_sz}), flush=True)
        write_json_atomic(args.output_dir / "synthetic_sample_scaling_run.json", {"records": records})


if __name__ == "__main__":
    main()
