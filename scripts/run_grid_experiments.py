#!/usr/bin/env python3
"""Reproduce all rolling European interconnected grid experiments (Tables 7, 9, EC.2, EC.3, EC.4).

Experiments:
  - `early`: 11 early rolling quarters (2018Q1--2020Q3, 1,004 days, OPSD panel) across all 7 methods,
             plus 5-repetition deployment timing on the 2019Q1 window (Table 7 & Table EC.2).
  - `later`: 17 later rolling quarters (2020Q4--2024Q4, 1,553 days, ENTSO-E panel) across all 7 methods
             with fixed settings (Table 7 & Table EC.3).
  - `basis`: Local basis ablation (Cross-covariance vs Standardized PCA vs Full space) across all 28 quarters (Table EC.4).
  - `alpha`: Spectral retention threshold sensitivity (alpha_ret in {0.9, 0.99, 0.999, 0.9999}) across all 28 quarters (Table 9).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
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
import torch

from grid_reserve import (
    GridSmoothSolver,
    InterconnectedGridReserveProblem,
    build_early_11_quarters,
    build_later_17_quarters,
    compress_knn_laws_wasserstein,
    eval_grid_laws,
    fit_sof_parallel,
    load_grid_panel,
    newey_west_se,
    prescriptiveness_coefficient,
    reduce_scenarios_wasserstein_kmeans,
    slice_rolling_window,
    solve_grid_hindsight_optimum,
    time_grid_deployment,
)
from kkt_atlas.atlas import (
    FittedAtlas,
    SelectedLocalModel,
    build_training_selected_candidate_library,
)
from kkt_atlas.automatic_joint_structure_v2 import (
    JointPartitionSelection,
    JointStructureConfig,
    select_joint_training_structure,
)
from kkt_atlas.automatic_structure import AutomaticPartitionCandidate
from kkt_atlas.baselines.contextual_empirical import KNearestNeighborSAA
from kkt_atlas.baselines.iceo import (
    FullSupportICEOModel,
    ICEOModelConfig,
    ICEOSnapshot,
    ICEOTrainingConfig,
    _batch_indices,
    _iceo_optimizer_step,
)
from kkt_atlas.baselines.sof import SOFConfig, _parent_model
from kkt_atlas.data import ObservedSplit
from kkt_atlas.joint_training import train_local_models_jointly
from kkt_atlas.parallel_smooth_problem_v1 import ParallelSmoothProblem
from kkt_atlas.response_basis import fit_response_decomposition
from kkt_atlas.runtime_helpers import write_json_atomic
from kkt_atlas.shared_checkpoint_selection import _fixed_checkpoint_path
from kkt_atlas.symmetric_campaign_protocol_v1 import consumer_seed32, derive_seed64
from kkt_atlas.symmetric_fit_v1 import knn_grid
from kkt_atlas.symmetric_laws_v1 import FiniteLaw, predict_method_laws
from kkt_atlas.training import TrainingConfig

GRID_ETA = 1e-6
GRID_SAVE_STEPS = (0, 50, 100, 200, 400, 800, 1200, 1600)


def wseed(window_id: int, *tokens: Any) -> int:
    return consumer_seed32(derive_seed64(20261112, "grid64", int(window_id), *tokens), "grid64")


def select_grid_atlas_structure(
    problem: InterconnectedGridReserveProblem,
    x_train: np.ndarray,
    xi_train: np.ndarray,
    *,
    window_id: int,
    alpha_ret: float = 0.9999,
    chart_counts: tuple[int, ...] = (1, 2, 4),
) -> JointPartitionSelection:
    train_obs = ObservedSplit(x_train, np.arange(len(x_train), dtype=np.int64))
    cfg = JointStructureConfig(
        ridge=GRID_ETA,
        chart_counts=chart_counts,
        fold_count=5,
        minimum_chart_size=50,
        spectral_retained_ratio=float(alpha_ret),
        minimum_samples_per_parameter=1.5,
        fail_closed=False,
        support_cap_offset=0,
        accept_intercept_only_fallback=True,
        seed=wseed(window_id, "atlas_struct"),
    )
    return select_joint_training_structure(GridSmoothSolver(problem), train_obs, xi_train, cfg).selected


def build_grid_basis_variant(
    problem: InterconnectedGridReserveProblem,
    x_train: np.ndarray,
    xi_train: np.ndarray,
    base_structure: JointPartitionSelection,
    variant: str,
) -> JointPartitionSelection:
    del x_train
    if variant == "kkt":
        return base_structure
    basis_method = "pca" if variant == "pca" else "identity"
    charts = base_structure.partition.context_charts
    m_dim = int(problem.scenario_dim)
    serial = GridSmoothSolver(problem)
    new_decomps = []
    new_locals = []
    for b_idx, ch in enumerate(base_structure.charts):
        idx = charts.indices(b_idx)
        decomp = fit_response_decomposition(
            serial,
            charts.centers[b_idx],
            xi_train[idx],
            ridge=GRID_ETA,
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


def fit_grid_atlas_on_structure(
    problem: InterconnectedGridReserveProblem,
    structure: JointPartitionSelection,
    par_solver: ParallelSmoothProblem,
    x_train: np.ndarray,
    xi_train: np.ndarray,
    x_val: np.ndarray,
    xi_val: np.ndarray,
    *,
    window_id: int,
    steps: tuple[int, ...] = GRID_SAVE_STEPS,
) -> dict[str, Any]:
    train_obs = ObservedSplit(x_train, np.arange(len(x_train), dtype=np.int64))
    seed_init = wseed(window_id, "atlas_init")
    seed_fit = wseed(window_id, "atlas_fit")

    init_cfg = TrainingConfig(
        steps=0,
        learning_rate=0.025,
        save_steps=(0,),
        coordinate_penalty=1e-5,
        slope_penalty=1e-5,
        gradient_clip=10.0,
        train_scenario_coordinates=True,
        train_probability_slopes=True,
        seed=seed_init,
    )
    init_lib = build_training_selected_candidate_library(
        problem,
        train_obs,
        xi_train,
        structure,
        ridge=GRID_ETA,
        training_config=init_cfg,
        global_sample_presentations_per_step=64,
        seed=seed_init,
        probability_head_initialization="training_cluster_multinomial_logit",
        probability_head_maximum_iterations=2000,
        training_function=lambda prob, *a, **kw: train_local_models_jointly(par_solver, *a, **kw),
    )
    _, _, cached_paths, _ = _fixed_checkpoint_path(init_lib)
    cached = tuple(p[0] for p in cached_paths)

    val_losses_by_step: dict[int, np.ndarray] = {}
    traces: list[dict[str, Any]] = []

    def train_fn(prob: Any, models: Any, contexts: Any, outcomes: Any, *, ridge: float, configs: Any) -> Any:
        for model, cand in zip(models, cached, strict=True):
            model.load_snapshot(cand.checkpoint)

        def on_ckpt(step: int, snapshots: Any) -> None:
            del snapshots
            proxy = SimpleNamespace(problem=prob, charts=structure.partition.context_charts, models=models)
            laws_v = predict_method_laws(proxy, "kkt_atlas", x_val, None)
            _, l_v, max_r = eval_grid_laws(problem, par_solver, x_val, xi_val, laws_v, GRID_ETA)
            val_losses_by_step[int(step)] = l_v
            traces.append({"step": int(step), "val_loss": float(np.mean(l_v)), "max_res": max_r})

        return train_local_models_jointly(
            par_solver, models, contexts, outcomes, ridge=ridge, configs=configs, checkpoint_callback=on_ckpt
        )

    fit_cfg = TrainingConfig(
        steps=int(steps[-1]),
        learning_rate=0.025,
        save_steps=tuple(int(s) for s in steps),
        coordinate_penalty=1e-5,
        slope_penalty=1e-5,
        gradient_clip=10.0,
        train_scenario_coordinates=True,
        train_probability_slopes=True,
        seed=seed_fit,
    )
    lib = build_training_selected_candidate_library(
        problem,
        train_obs,
        xi_train,
        structure,
        ridge=GRID_ETA,
        training_config=fit_cfg,
        global_sample_presentations_per_step=64,
        seed=seed_fit,
        probability_head_initialization="training_cluster_multinomial_logit",
        probability_head_maximum_iterations=2000,
        training_function=train_fn,
    )
    partition, ridge_val, candidates, _ = _fixed_checkpoint_path(lib)
    best = min(traces, key=lambda r: (r["val_loss"], r["step"]))
    sel_step = int(best["step"])
    assign_val = partition.charts.assign(x_val)
    l_sel = val_losses_by_step[sel_step]

    fitted = FittedAtlas(
        problem=problem,
        charts=partition.charts,
        ridge=ridge_val,
        local_models=tuple(
            SelectedLocalModel(
                candidate=replace(candidates[i][sel_step], initialization=cached[i].initialization),
                validation_loss_sum=math.fsum(l_sel[assign_val == i]),
                validation_count=int(np.sum(assign_val == i)),
            )
            for i in range(len(candidates))
        ),
        validation_mean_loss=float(np.mean(l_sel)),
    )
    prepared = fitted.prepare()
    k_list = [int(ch.selected_support_size) for ch in structure.charts]
    d_list = [int(ch.representation_dim) for ch in structure.charts]
    d_x = int(problem.context_dim)
    param_cnt = sum(k * d + (k - 1) * (d_x + 1) for k, d in zip(k_list, d_list, strict=True))
    return {
        "fitted": fitted,
        "prepared": prepared,
        "B": int(structure.chart_count),
        "K_b": k_list,
        "d_b": d_list,
        "max_K": int(max(k_list)),
        "parameter_count": int(param_cnt),
        "selected_step": sel_step,
        "val_loss": float(best["val_loss"]),
    }


def predict_prepared_atlas_laws(prepared: Any, contexts: np.ndarray) -> tuple[FiniteLaw, ...]:
    return predict_method_laws(prepared, "kkt_atlas", contexts, None)


def fit_grid_iceo(
    problem: InterconnectedGridReserveProblem,
    par_solver: ParallelSmoothProblem,
    support_scenarios: np.ndarray,
    x_train: np.ndarray,
    xi_train: np.ndarray,
    x_val: np.ndarray,
    xi_val: np.ndarray,
    *,
    seed: int,
    steps: tuple[int, ...],
) -> tuple[FullSupportICEOModel, int, float]:
    m_cfg = ICEOModelConfig(hidden_width=32, hidden_layers=2, seed=int(seed & 0x7FFFFFFF))
    t_cfg = ICEOTrainingConfig(
        steps=int(steps[-1]),
        learning_rate=0.001,
        save_steps=tuple(int(s) for s in steps),
        batch_size=64,
        gradient_clip=10.0,
        seed=int((seed + 104729) & 0x7FFFFFFF),
        microbatch_size=32,
    )
    model = FullSupportICEOModel(
        par_solver, support_scenarios, x_train, config=m_cfg, dtype=torch.float64, device="cpu"
    )
    ctx_t = torch.tensor(x_train, dtype=torch.float64)
    std_t = model.standardized_contexts(ctx_t)
    out_t = torch.tensor(xi_train, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=t_cfg.learning_rate, foreach=False, fused=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(t_cfg.seed))
    save_set = set(t_cfg.save_steps)
    snaps: list[ICEOSnapshot] = [model.snapshot(step=0, training_loss=0.0)]
    for step in range(1, t_cfg.steps + 1):
        indices = _batch_indices(len(ctx_t), t_cfg.batch_size, generator=gen, device=model.catalog.device)
        bx, bstd, bxi = (ctx_t, std_t, out_t) if indices is None else (ctx_t[indices], std_t[indices], out_t[indices])
        _iceo_optimizer_step(par_solver, model, opt, bx, bstd, bxi, None, ridge=GRID_ETA, config=t_cfg, step=step)
        if step in save_set:
            snaps.append(model.snapshot(step=step, training_loss=0.0))

    best_v = math.inf
    best_sn = snaps[0]
    for sn in snaps:
        model.load_snapshot(sn)
        p_v = model.probabilities_numpy(x_val)
        laws_v = tuple(
            FiniteLaw(support_scenarios, p, np.arange(len(support_scenarios), dtype=np.int64), "local")
            for p in p_v
        )
        _, lv, _ = eval_grid_laws(problem, par_solver, x_val, xi_val, laws_v, GRID_ETA)
        mv = float(np.mean(lv))
        if mv < best_v:
            best_v = mv
            best_sn = sn
    model.load_snapshot(best_sn)
    return model, int(best_sn.step), best_v


def run_rolling_suite(
    panel_npz: Path,
    quarters: list[tuple[str, str, str, int]],
    *,
    steps: tuple[int, ...],
    workers: int,
    measure_2019q1_timing: bool = False,
) -> dict[str, Any]:
    dates, contexts, outcomes, _, problem = load_grid_panel(panel_npz)
    serial_solver = GridSmoothSolver(problem)
    method_names = ("atlas", "iceo", "iceo_red", "sof", "knn", "knn_red", "saa")
    daily_losses: dict[str, list[float]] = {m: [] for m in method_names}
    hindsight_all: list[float] = []
    window_rows: list[dict[str, Any]] = []
    deployment_timing_2019q1: dict[str, Any] | None = None

    with ParallelSmoothProblem(serial_solver, workers=workers) as par_solver:
        for q_name, start_d, end_d, wid in quarters:
            w = slice_rolling_window(dates, contexts, outcomes, start_d, end_d)
            x_tr, xi_tr = w["x_train"], w["xi_train"]
            x_va, xi_va = w["x_val"], w["xi_val"]
            x_te, xi_te = w["x_test"], w["xi_test"]
            _, pf_te, _ = solve_grid_hindsight_optimum(problem, x_te, xi_te)
            hindsight_all.extend(pf_te.tolist())

            w_res: dict[str, Any] = {"quarter": q_name, "days": len(x_te), "methods": {}}

            # 1. Atlas
            t0 = time.perf_counter()
            st = select_grid_atlas_structure(problem, x_tr, xi_tr, window_id=wid)
            afit = fit_grid_atlas_on_structure(
                problem, st, par_solver, x_tr, xi_tr, x_va, xi_va, window_id=wid, steps=steps
            )
            atlas_build_s = time.perf_counter() - t0
            atlas_laws = predict_prepared_atlas_laws(afit["prepared"], x_te)
            _, l_atlas, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, atlas_laws, GRID_ETA)
            daily_losses["atlas"].extend(l_atlas.tolist())
            w_res["methods"]["atlas"] = {
                "mean_loss": float(np.mean(l_atlas)),
                "scenarios": float(afit["max_K"]),
                "build_s": atlas_build_s,
                "B": afit["B"],
                "K_b": afit["K_b"],
            }

            # 2. ICEO (728 scenarios)
            t0 = time.perf_counter()
            iceo_m, _, _ = fit_grid_iceo(
                problem, par_solver, xi_tr, x_tr, xi_tr, x_va, xi_va, seed=wseed(wid, "iceo"), steps=steps
            )
            iceo_build_s = time.perf_counter() - t0
            iceo_laws = tuple(
                FiniteLaw(xi_tr, p, np.arange(len(xi_tr), dtype=np.int64), "catalog")
                for p in iceo_m.probabilities_numpy(x_te)
            )
            _, l_iceo, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, iceo_laws, GRID_ETA)
            daily_losses["iceo"].extend(l_iceo.tolist())
            w_res["methods"]["iceo"] = {
                "mean_loss": float(np.mean(l_iceo)),
                "scenarios": float(len(xi_tr)),
                "build_s": iceo_build_s,
            }

            # 3. ICEO-red
            k_match = int(afit["max_K"])
            t0 = time.perf_counter()
            sr_cat, _ = reduce_scenarios_wasserstein_kmeans(problem, xi_tr, k_match, seed=wseed(wid, "iceo_sr_km"))
            iceo_sr_m, _, _ = fit_grid_iceo(
                problem, par_solver, sr_cat, x_tr, xi_tr, x_va, xi_va, seed=wseed(wid, "iceo_sr"), steps=steps
            )
            iceo_sr_build_s = time.perf_counter() - t0
            iceo_sr_laws = tuple(
                FiniteLaw(sr_cat, p, np.arange(len(sr_cat), dtype=np.int64), "local")
                for p in iceo_sr_m.probabilities_numpy(x_te)
            )
            _, l_iceo_sr, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, iceo_sr_laws, GRID_ETA)
            daily_losses["iceo_red"].extend(l_iceo_sr.tolist())
            w_res["methods"]["iceo_red"] = {
                "mean_loss": float(np.mean(l_iceo_sr)),
                "scenarios": float(k_match),
                "build_s": iceo_sr_build_s,
            }

            # 4. SOF
            t0 = time.perf_counter()
            sof_cfg = SOFConfig(
                ridge=GRID_ETA,
                tree_count=100 if len(steps) > 2 else 10,
                bootstrap=True,
                honesty=False,
                minimum_child_fraction=0.2,
                terminal_node_size=10,
                maximum_depth=100,
                split_rule="approximate_risk",
                candidate_rule="all_midpoints",
                seed=wseed(wid, "sof"),
                worker_count=min(8, workers),
            )
            sof_fit = fit_sof_parallel(_parent_model(serial_solver), x_tr, xi_tr, sof_cfg)
            sof_build_s = time.perf_counter() - t0
            sof_w = sof_fit.weights_matrix(x_te)
            sof_laws = tuple(
                FiniteLaw(xi_tr[row > 0], row[row > 0] / row[row > 0].sum(), np.flatnonzero(row > 0), "local")
                for row in sof_w
            )
            _, l_sof, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, sof_laws, GRID_ETA)
            daily_losses["sof"].extend(l_sof.tolist())
            w_res["methods"]["sof"] = {
                "mean_loss": float(np.mean(l_sof)),
                "scenarios": float(np.mean([len(lw.probabilities) for lw in sof_laws])),
                "build_s": sof_build_s,
            }

            # 5. kNN & 6. kNN-red
            t0 = time.perf_counter()
            best_k = 1
            best_kv = math.inf
            best_km = None
            idx_tr = np.arange(len(x_tr), dtype=np.int64)
            for kc in knn_grid(len(x_tr)):
                cm = KNearestNeighborSAA(x_tr, idx_tr, neighbors=int(kc), standardize_contexts=True)
                d_v = cm.predict_distributions(x_va)
                laws_v = tuple(FiniteLaw(xi_tr[ii], pp, ii, "catalog") for ii, pp in d_v)
                _, lv, _ = eval_grid_laws(problem, par_solver, x_va, xi_va, laws_v, GRID_ETA)
                mv = float(np.mean(lv))
                if mv < best_kv:
                    best_kv = mv
                    best_k = int(kc)
                    best_km = cm
            knn_build_s = time.perf_counter() - t0
            assert best_km is not None
            knn_laws = tuple(
                FiniteLaw(xi_tr[ii], pp, ii, "catalog") for ii, pp in best_km.predict_distributions(x_te)
            )
            _, l_knn, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, knn_laws, GRID_ETA)
            daily_losses["knn"].extend(l_knn.tolist())
            w_res["methods"]["knn"] = {
                "mean_loss": float(np.mean(l_knn)),
                "scenarios": float(np.mean([len(lw.probabilities) for lw in knn_laws])),
                "build_s": knn_build_s,
            }

            knn_red_laws = compress_knn_laws_wasserstein(
                problem, knn_laws, xi_tr, k_match, seed=wseed(wid, "knn_sr")
            )
            _, l_knn_red, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, knn_red_laws, GRID_ETA)
            daily_losses["knn_red"].extend(l_knn_red.tolist())
            w_res["methods"]["knn_red"] = {
                "mean_loss": float(np.mean(l_knn_red)),
                "scenarios": float(np.mean([len(lw.probabilities) for lw in knn_red_laws])),
                "build_s": knn_build_s,
            }

            # 7. SAA
            saa_law = FiniteLaw(xi_tr, np.full(len(xi_tr), 1.0 / len(xi_tr)), idx_tr, "catalog")
            saa_laws = tuple(saa_law for _ in range(len(x_te)))
            _, l_saa, _ = eval_grid_laws(problem, par_solver, x_te, xi_te, saa_laws, GRID_ETA)
            daily_losses["saa"].extend(l_saa.tolist())
            w_res["methods"]["saa"] = {
                "mean_loss": float(np.mean(l_saa)),
                "scenarios": float(len(xi_tr)),
                "build_s": 0.0,
            }

            if measure_2019q1_timing and q_name == "2019Q1":
                ctx_128 = np.vstack([x_te, x_va])[:128]
                deployment_timing_2019q1 = {
                    "atlas": time_grid_deployment(
                        lambda c: predict_prepared_atlas_laws(afit["prepared"], c),
                        ctx_128, serial_solver=serial_solver, par_solver=par_solver, eta=GRID_ETA,
                    ),
                    "iceo": time_grid_deployment(
                        lambda c: tuple(
                            FiniteLaw(xi_tr, p, idx_tr, "catalog") for p in iceo_m.probabilities_numpy(c)
                        ),
                        ctx_128, serial_solver=serial_solver, par_solver=par_solver, eta=GRID_ETA,
                    ),
                }

            window_rows.append(w_res)
            print(
                json.dumps(
                    {
                        "quarter": q_name,
                        "atlas": round(w_res["methods"]["atlas"]["mean_loss"], 3),
                        "iceo": round(w_res["methods"]["iceo"]["mean_loss"], 3),
                        "saa": round(w_res["methods"]["saa"]["mean_loss"], 3),
                    }
                ),
                flush=True,
            )

    pf_arr = np.asarray(hindsight_all, dtype=np.float64)
    saa_arr = np.asarray(daily_losses["saa"], dtype=np.float64)
    atlas_arr = np.asarray(daily_losses["atlas"], dtype=np.float64)
    summary: dict[str, Any] = {}
    for m in method_names:
        arr = np.asarray(daily_losses[m], dtype=np.float64)
        q_means = [r["methods"][m]["mean_loss"] for r in window_rows]
        diff = atlas_arr - arr
        summary[m] = {
            "pooled_mean": float(np.mean(arr)),
            "newey_west_se": newey_west_se(arr, max_lag=14),
            "quarter_median": float(statistics.median(q_means)),
            "prescriptiveness_P": prescriptiveness_coefficient(arr, saa_arr, pf_arr),
            "mean_scenarios": float(np.mean([r["methods"][m]["scenarios"] for r in window_rows])),
            "mean_build_s": float(np.mean([r["methods"][m]["build_s"] for r in window_rows])),
            "atlas_minus_method_mean": float(np.mean(diff)),
            "atlas_minus_method_nw_se": newey_west_se(diff, max_lag=14),
        }
    return {
        "hindsight_pooled_mean": float(np.mean(pf_arr)),
        "summary": summary,
        "windows": window_rows,
        "deployment_timing_2019q1": deployment_timing_2019q1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=("early", "later", "basis", "alpha"), default="early")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "grid")
    parser.add_argument("--quick-smoke", action="store_true", help="Fast 1-quarter smoke check (20 steps).")
    args = parser.parse_args()

    torch.set_num_threads(1)
    opsd_npz = REPO_ROOT / "data" / "processed" / "opsd_grid64_2015_2020q3.npz"
    entsoe_npz = REPO_ROOT / "data" / "processed" / "entsoe_grid64_2017q4_2024q4.npz"
    steps = (0, 20) if args.quick_smoke else GRID_SAVE_STEPS

    if args.experiment == "early":
        qtrs = build_early_11_quarters()[:1] if args.quick_smoke else build_early_11_quarters()
        out = run_rolling_suite(
            opsd_npz, qtrs, steps=steps, workers=args.workers, measure_2019q1_timing=not args.quick_smoke
        )
        write_json_atomic(args.output_dir / "grid_early_run.json", out)

    elif args.experiment == "later":
        qtrs = build_later_17_quarters()[:1] if args.quick_smoke else build_later_17_quarters()
        out = run_rolling_suite(entsoe_npz, qtrs, steps=steps, workers=args.workers)
        write_json_atomic(args.output_dir / "grid_later_run.json", out)

    elif args.experiment in ("basis", "alpha"):
        schedules = [
            ("early", opsd_npz, build_early_11_quarters()[:1] if args.quick_smoke else build_early_11_quarters()),
            ("later", entsoe_npz, [] if args.quick_smoke else build_later_17_quarters()),
        ]
        records = []
        for period, npz_p, qtrs in schedules:
            if not qtrs:
                continue
            dates, contexts, outcomes, _, problem = load_grid_panel(npz_p)
            with ParallelSmoothProblem(GridSmoothSolver(problem), workers=args.workers) as par_s:
                for q_name, s_d, e_d, wid in qtrs:
                    w = slice_rolling_window(dates, contexts, outcomes, s_d, e_d)
                    x_tr, xi_tr, x_va, xi_va, x_te, xi_te = (
                        w["x_train"], w["xi_train"], w["x_val"], w["xi_val"], w["x_test"], w["xi_test"]
                    )
                    if args.experiment == "basis":
                        base_st = select_grid_atlas_structure(problem, x_tr, xi_tr, window_id=wid)
                        row_v = {}
                        for var in ("kkt", "pca", "full"):
                            st_v = build_grid_basis_variant(problem, x_tr, xi_tr, base_st, var)
                            fit_v = fit_grid_atlas_on_structure(
                                problem, st_v, par_s, x_tr, xi_tr, x_va, xi_va, window_id=wid, steps=steps
                            )
                            laws_v = predict_prepared_atlas_laws(fit_v["prepared"], x_te)
                            _, lv, _ = eval_grid_laws(problem, par_s, x_te, xi_te, laws_v, GRID_ETA)
                            row_v[var] = {"mean_loss": float(np.mean(lv)), "params": fit_v["parameter_count"]}
                        records.append({"period": period, "quarter": q_name, "variants": row_v})
                        print(json.dumps({"experiment": "basis", "quarter": q_name, "kkt": round(row_v["kkt"]["mean_loss"], 3)}), flush=True)
                    else:
                        alphas = (0.9, 0.9999) if args.quick_smoke else (0.9, 0.99, 0.999, 0.9999)
                        row_a = {}
                        for alpha in alphas:
                            st_a = select_grid_atlas_structure(problem, x_tr, xi_tr, window_id=wid, alpha_ret=alpha)
                            fit_a = fit_grid_atlas_on_structure(
                                problem, st_a, par_s, x_tr, xi_tr, x_va, xi_va, window_id=wid, steps=steps
                            )
                            laws_a = predict_prepared_atlas_laws(fit_a["prepared"], x_te)
                            _, la, _ = eval_grid_laws(problem, par_s, x_te, xi_te, laws_a, GRID_ETA)
                            row_a[str(alpha)] = {"mean_loss": float(np.mean(la)), "max_K": fit_a["max_K"]}
                        records.append({"period": period, "quarter": q_name, "alphas": row_a})
                        print(json.dumps({"experiment": "alpha", "quarter": q_name}), flush=True)
        write_json_atomic(args.output_dir / f"grid_{args.experiment}_run.json", {"records": records})


if __name__ == "__main__":
    main()
