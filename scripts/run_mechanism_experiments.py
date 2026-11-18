#!/usr/bin/env python3
"""Reproduce the three mechanism studies in Section 6.5 (Figures 1--2 and Table 8).

Experiments:
  - `support-size` (Figure 1): CAP fixed B=1, d=3 across local support sizes K in {1, 2, 3, 4, 5, 6}.
  - `chart-angle` (Figure 2): 4-mode quadratic problem (Eq. 21 & EC.27--EC.28) across angles
                              omega in {0, 45, 90} deg and B in {1, 2, 4, 8, 16, 32}.
  - `controlled-basis` (Table 8): 200-dim quadratic problem (Eq. 22) under decision-irrelevant
                                  variation sigma in {0.25, 4.0}, comparing Cross-covariance vs PCA.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from kkt_atlas.atlas import (
    AtlasSearchConfig,
    build_candidate_library,
    build_training_selected_candidate_library,
    select_atlas,
)
from kkt_atlas.automatic_joint_structure_v2 import JointStructureConfig, select_joint_training_structure
from kkt_atlas.case_studies.controlled_nuisance_data import (
    ControlledNuisanceDataConfig,
    generate_controlled_nuisance_pair,
)
from kkt_atlas.case_studies.projected_quadratic import ProjectedQuadraticProblem
from kkt_atlas.confirmation_data_v1 import FittingCatalogData
from kkt_atlas.data import ObservedSplit
from kkt_atlas.parallel_smooth_problem_v1 import ParallelSmoothProblem
from kkt_atlas.runtime_helpers import (
    generate_synthetic_application_instance,
    solver_for,
    write_json_atomic,
)
from kkt_atlas.symmetric_laws_v1 import predict_method_laws
from kkt_atlas.training import TrainingConfig
from run_synthetic_experiments import (
    compute_reference_risks,
    evaluate_laws_exact,
    fit_atlas_on_structure,
    select_atlas_structure,
)


def run_support_size_study(*, quick_smoke: bool, workers: int) -> dict[str, Any]:
    """Section 6.5.1 (Figure 1): CAP fixed B=1, d=3 across K in {1..6}."""
    seeds = (260911,) if quick_smoke else (260911, 260912, 260913)
    k_grid = (1, 4) if quick_smoke else (1, 2, 3, 4, 5, 6)
    steps = (0, 20) if quick_smoke else (0, 400, 800, 1200, 1600)
    n_train = 400 if quick_smoke else 12800
    eval_lim = 16 if quick_smoke else None

    records = []
    for seed in seeds:
        proto, problem, fitting, test = generate_synthetic_application_instance(
            "capacity", seed, training_size=n_train
        )
        if eval_lim is not None:
            test = replace(test, contexts=test.contexts[:eval_lim], probabilities=test.probabilities[:eval_lim])
        ref_risks = compute_reference_risks(
            "capacity", problem, fitting.catalog, test.contexts, test.probabilities, workers=workers
        )
        base_st = select_atlas_structure(
            problem, fitting, proto, app="capacity", seed=seed, chart_counts=(1,)
        )
        with ParallelSmoothProblem(solver_for(problem), workers=workers) as par_s:
            k_out = {}
            for k_val in k_grid:
                ch0 = base_st.charts[0]
                d_fix = min(3, base_st.partition.decompositions[0].left_vectors.shape[1])
                ch_k = replace(
                    ch0,
                    representation_dim=d_fix,
                    protocol_support_cap=max(int(k_val), d_fix + 1),
                    production_support_cap=max(int(ch0.production_support_cap), int(k_val)),
                    selected_support_size=int(k_val),
                    support_cap_offset=0,
                )
                st_k = replace(base_st, charts=(ch_k,))
                fit_k = fit_atlas_on_structure(
                    problem, fitting, proto, st_k, par_s, app="capacity", seed=seed, steps=steps
                )
                laws_k = predict_method_laws(fit_k["prepared"], "kkt_atlas", test.contexts, fitting)
                ev_k = evaluate_laws_exact(
                    "capacity", problem, laws_k, fitting.catalog, test.contexts, test.probabilities,
                    0.0, ref_risks, workers=workers,
                )
                k_out[str(k_val)] = float(ev_k["relative_excess_percent"])
                print(
                    json.dumps(
                        {"study": "support-size", "seed": seed, "K": k_val, "excess_pct": round(k_out[str(k_val)], 4)}
                    ),
                    flush=True,
                )
            records.append({"seed": seed, "excess_by_K": k_out})
    return {"study": "support_size_cap", "records": records}


def _generate_chart_angle_data(
    seed: int,
    angle_deg: float,
    *,
    n_train: int,
    n_val: int = 1000,
    n_eval: int = 200,
) -> tuple[ProjectedQuadraticProblem, FittingCatalogData, np.ndarray, np.ndarray, np.ndarray]:
    """Exact DGP of Section 6.5.2 & EC.3.5 (Eq. 21, EC.27--EC.28)."""
    rng = np.random.default_rng(int(seed))
    q_mat, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    v_shared = q_mat[:, :2]
    v_aux = [q_mat[:, 2 + 2 * r : 4 + 2 * r] for r in range(4)]

    omega = math.radians(float(angle_deg))
    v_modes = [math.cos(omega) * v_shared + math.sin(omega) * v_aux[r] for r in range(4)]
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
    catalog = np.vstack([(v_modes[r] @ corners.T).T for r in range(4)])

    h_pairs = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    mode_centers = np.array([2.0 * np.array([h1, h1, h1, h1, h2, h2], dtype=np.float64) for h1, h2 in h_pairs])

    def sample_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        modes = rng.integers(0, 4, size=n)
        c_e = rng.uniform(-0.25, 0.25, size=(n, 6))
        s = rng.uniform(-0.8, 0.8, size=(n, 2))
        t = np.arctanh(s)
        x = np.hstack([mode_centers[modes] + c_e, t])
        logits = t @ corners.T
        logits -= np.max(logits, axis=1, keepdims=True)
        probs_4 = np.exp(logits)
        probs_4 /= np.sum(probs_4, axis=1, keepdims=True)
        u = rng.uniform(0.0, 1.0, size=n)
        cdf = np.cumsum(probs_4, axis=1)
        corner_idx = np.clip([int(np.searchsorted(cdf[i], u[i], side="right")) for i in range(n)], 0, 3)
        cat_idx = 4 * modes + corner_idx
        probs_16 = np.zeros((n, 16), dtype=np.float64)
        for i in range(n):
            probs_16[i, 4 * modes[i] : 4 * modes[i] + 4] = probs_4[i]
        z_star = np.vstack([v_modes[modes[i]] @ s[i] for i in range(n)])
        return x, cat_idx.astype(np.int64), probs_16, z_star

    x_tr, idx_tr, _, _ = sample_split(n_train)
    x_va, idx_va, _, _ = sample_split(n_val)
    x_ev, _, probs_ev, z_star_ev = sample_split(n_eval)

    problem = ProjectedQuadraticProblem(task_basis=np.eye(10, dtype=np.float64), context_dim=8)
    fitting = FittingCatalogData(
        catalog=catalog,
        training=ObservedSplit(x_tr, idx_tr),
        validation=ObservedSplit(x_va, idx_va),
        metadata={"study": "chart_angle", "seed": int(seed), "angle_deg": float(angle_deg)},
    )
    return problem, fitting, x_ev, probs_ev, z_star_ev


def run_chart_angle_study(*, quick_smoke: bool) -> dict[str, Any]:
    """Section 6.5.2 (Figure 2): Chart count B sensitivity across mode plane separation angles."""
    seeds = (260911,) if quick_smoke else (260911, 260912, 260913)
    angles = (0.0, 90.0) if quick_smoke else (0.0, 45.0, 90.0)
    b_grid = (1, 4) if quick_smoke else (1, 2, 4, 8, 16, 32)
    n_train = 640 if quick_smoke else 12800
    steps = (0, 20) if quick_smoke else (0, 400, 800, 1200, 1600)

    records = []
    for angle in angles:
        for seed in seeds:
            problem, fitting, x_ev, probs_ev, z_star_ev = _generate_chart_angle_data(
                seed, angle, n_train=n_train, n_val=200 if quick_smoke else 1000, n_eval=64 if quick_smoke else 200
            )
            ref_losses = np.array(
                [
                    0.5 * float(probs_ev[i] @ np.sum((fitting.catalog - z_star_ev[i][None, :]) ** 2, axis=1))
                    for i in range(len(x_ev))
                ]
            )
            ref_mean = float(np.mean(ref_losses))
            b_out = {}
            for b_val in b_grid:
                cfg = JointStructureConfig(
                    ridge=0.0,
                    chart_counts=(int(b_val),),
                    fold_count=5,
                    minimum_chart_size=10 if quick_smoke else 50,
                    spectral_retained_ratio=0.9999,
                    minimum_samples_per_parameter=1.0 if quick_smoke else 3.0,
                    fail_closed=False,
                    support_cap_offset=0,
                    accept_intercept_only_fallback=True,
                    seed=int(seed + b_val),
                )
                st = select_joint_training_structure(problem, fitting.training, fitting.catalog, cfg).selected
                t_cfg = TrainingConfig(
                    steps=steps[-1],
                    learning_rate=0.001,
                    save_steps=steps,
                    coordinate_penalty=1e-5,
                    slope_penalty=1e-5,
                    gradient_clip=10.0,
                    train_scenario_coordinates=True,
                    train_probability_slopes=True,
                    seed=int(seed),
                )
                lib = build_training_selected_candidate_library(
                    problem,
                    fitting.training,
                    fitting.catalog,
                    st,
                    ridge=0.0,
                    training_config=t_cfg,
                    global_sample_presentations_per_step=64,
                    seed=int(seed),
                )
                fitted = select_atlas(lib, fitting.validation, fitting.catalog)
                decs = fitted.decisions(x_ev)
                losses = np.array(
                    [
                        0.5 * float(probs_ev[i] @ np.sum((fitting.catalog - decs[i][None, :]) ** 2, axis=1))
                        for i in range(len(x_ev))
                    ]
                )
                excess_pct = 100.0 * (float(np.mean(losses)) - ref_mean) / abs(ref_mean)
                b_out[str(b_val)] = {
                    "relative_excess_percent": excess_pct,
                    "mean_d_b": float(np.mean([c.representation_dim for c in st.charts])),
                    "mean_K_b": float(np.mean([c.selected_support_size for c in st.charts])),
                }
                print(
                    json.dumps(
                        {"study": "chart-angle", "angle": angle, "seed": seed, "B": b_val, "excess_pct": round(excess_pct, 4)}
                    ),
                    flush=True,
                )
            records.append({"angle_deg": angle, "seed": seed, "by_B": b_out})
    return {"study": "chart_angle", "records": records}


def run_controlled_basis_study(*, quick_smoke: bool) -> dict[str, Any]:
    """Section 6.5.3 (Table 8): Cross-covariance basis vs PCA under decision-irrelevant noise."""
    seeds = (241011,) if quick_smoke else (241011, 241012, 241013)
    steps = (0, 20) if quick_smoke else (0, 20, 40, 80)
    records = []
    for seed in seeds:
        cfg = ControlledNuisanceDataConfig(
            seed=int(seed),
            scenario_dim=200,
            task_rank=4,
            nuisance_rank=8,
            training_size=400 if quick_smoke else 1600,
            validation_size=100 if quick_smoke else 200,
            test_size=64 if quick_smoke else 200,
            low_noise_std=0.25,
            high_noise_std=4.0,
        )
        pair = generate_controlled_nuisance_pair(cfg)
        v_rel = pair.geometry.task_basis  # (200, 4)
        for level_name, data in (("low_0.25", pair.low_noise), ("high_4.0", pair.high_noise)):
            z_opt = data.test_probabilities @ (data.catalog @ v_rel)
            opt_signal = float(np.mean(np.sum(z_opt**2, axis=1)))
            for basis_method in ("kkt", "pca"):
                t_cfg = TrainingConfig(
                    steps=steps[-1],
                    learning_rate=0.01,
                    save_steps=steps,
                    coordinate_penalty=1e-5,
                    slope_penalty=1e-5,
                    gradient_clip=10.0,
                    seed=int(seed),
                )
                s_cfg = AtlasSearchConfig(
                    chart_counts=(4,),
                    ridges=(0.0,),
                    widths=(4,),
                    support_sizes=(5,),
                    training=t_cfg,
                    include_width_plus_one=False,
                    basis_method=basis_method,
                    seed=int(seed),
                )
                lib = build_candidate_library(pair.problem, data.training, data.catalog, s_cfg)
                fitted = select_atlas(lib, data.validation, data.catalog)
                decs = fitted.decisions(data.test_contexts)
                norm_err = float(np.mean(np.sum((decs - z_opt) ** 2, axis=1)) / max(opt_signal, 1e-12))

                assign_tr = fitted.charts.assign(data.training.contexts)
                align_num = 0.0
                for b_idx, lm in enumerate(fitted.local_models):
                    n_b = int(np.sum(assign_tr == b_idx))
                    dec = lm.candidate.decomposition
                    dv = dec.transformed_scale[:, None] * dec.left_vectors[:, :4]
                    q_orig, _ = np.linalg.qr(dv)
                    rho_b = float(np.min(np.linalg.svd(v_rel.T @ q_orig, compute_uv=False)))
                    align_num += n_b * rho_b
                alignment = align_num / float(len(data.training.contexts))

                proj_cat = data.catalog @ v_rel
                risks = [
                    0.5 * float(data.test_probabilities[i] @ np.sum((proj_cat - decs[i][None, :]) ** 2, axis=1))
                    for i in range(len(decs))
                ]
                row = {
                    "seed": int(seed),
                    "noise_level": level_name,
                    "basis": basis_method,
                    "normalized_error": norm_err,
                    "alignment": alignment,
                    "risk": float(np.mean(risks)),
                }
                records.append(row)
                print(json.dumps({"study": "controlled-basis", **row}), flush=True)
    return {"study": "controlled_basis", "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment",
        choices=("all", "support-size", "chart-angle", "controlled-basis"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mechanism")
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.experiment in ("all", "support-size"):
        write_json_atomic(
            args.output_dir / "support_size_run.json",
            run_support_size_study(quick_smoke=args.quick_smoke, workers=args.workers),
        )
    if args.experiment in ("all", "chart-angle"):
        write_json_atomic(
            args.output_dir / "chart_angle_run.json",
            run_chart_angle_study(quick_smoke=args.quick_smoke),
        )
    if args.experiment in ("all", "controlled-basis"):
        write_json_atomic(
            args.output_dir / "controlled_basis_run.json",
            run_controlled_basis_study(quick_smoke=args.quick_smoke),
        )


if __name__ == "__main__":
    main()
