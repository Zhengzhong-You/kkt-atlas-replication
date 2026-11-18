"""Automated unit and end-to-end smoke tests for the KKT Atlas replication package."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from grid_reserve import (
    GridSmoothSolver,
    build_early_11_quarters,
    build_later_17_quarters,
    load_grid_panel,
    newey_west_se,
    prescriptiveness_coefficient,
    reduce_scenarios_wasserstein_kmeans,
    slice_rolling_window,
)
from kkt_atlas import (
    JointStructureConfig,
    TrainingConfig,
    fit_response_decomposition,
    select_joint_training_structure,
)
from kkt_atlas.atlas import build_training_selected_candidate_library, select_atlas
from kkt_atlas.case_studies.controlled_nuisance_data import (
    ControlledNuisanceDataConfig,
    generate_controlled_nuisance_pair,
)
from kkt_atlas.runtime_helpers import (
    generate_synthetic_application_instance,
    solve_and_eval_one_context,
    solver_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cross_covariance_basis_beats_pca_under_irrelevant_noise() -> None:
    """Verify gradient-informed cross-covariance basis recovers V_rel under high nuisance variance."""
    cfg = ControlledNuisanceDataConfig(
        seed=241011,
        scenario_dim=60,
        task_rank=4,
        nuisance_rank=8,
        training_size=400,
        validation_size=80,
        test_size=80,
        low_noise_std=0.25,
        high_noise_std=4.0,
    )
    pair = generate_controlled_nuisance_pair(cfg)
    v_rel = pair.geometry.task_basis
    data = pair.high_noise
    xi_tr = data.training.outcomes(data.catalog)
    x_center = np.mean(data.training.contexts, axis=0)

    dec_kkt = fit_response_decomposition(pair.problem, x_center, xi_tr, ridge=0.0, basis_method="kkt")
    dec_pca = fit_response_decomposition(pair.problem, x_center, xi_tr, ridge=0.0, basis_method="pca")

    q_kkt, _ = np.linalg.qr(dec_kkt.transformed_scale[:, None] * dec_kkt.left_vectors[:, :4])
    q_pca, _ = np.linalg.qr(dec_pca.transformed_scale[:, None] * dec_pca.left_vectors[:, :4])
    align_kkt = float(np.min(np.linalg.svd(v_rel.T @ q_kkt, compute_uv=False)))
    align_pca = float(np.min(np.linalg.svd(v_rel.T @ q_pca, compute_uv=False)))

    assert align_kkt > 0.80
    assert align_pca < 0.25


def test_all_four_problem_solvers_satisfy_kkt_residual_gate() -> None:
    """Verify CAP, PORT, NET (with open-source Clarabel fallback), and GRID solve to <= 1e-6 KKT residual."""
    for app in ("capacity", "portfolio", "network"):
        proto, prob, fit, test = generate_synthetic_application_instance(app, 260911, training_size=256)
        slv = solver_for(prob)
        dec, res, _, risk = solve_and_eval_one_context(
            slv,
            (
                app,
                test.contexts[0],
                fit.catalog[:6],
                np.full(6, 1.0 / 6.0),
                proto.eta(app),
                fit.catalog[:32],
                test.probabilities[0, :32] / np.sum(test.probabilities[0, :32]),
            ),
        )
        assert dec.shape == (prob.decision_dim,)
        assert res <= 1e-6
        assert np.isfinite(risk)

    dates, ctx, out, _, gprob = load_grid_panel(REPO_ROOT / "data/processed/opsd_grid64_2015_2020q3.npz")
    w = slice_rolling_window(dates, ctx, out, "2019-01-01", "2019-03-31")
    gslv = GridSmoothSolver(gprob)
    z_g, _, aud_g, _ = gslv.solve_with_audit(w["x_test"][0], w["xi_train"][:8], np.full(8, 0.125), ridge=1e-6)
    assert z_g.shape == (8,)
    assert aud_g.maximum_residual <= 1e-6


def test_atlas_end_to_end_fit_and_predict_smoke() -> None:
    """Verify KKT Atlas structure selection, joint fitting, and prediction on a small CAP instance."""
    torch.set_num_threads(1)
    _, prob, fit, test = generate_synthetic_application_instance("capacity", 260911, training_size=200)
    slv = solver_for(prob)
    st_cfg = JointStructureConfig(
        ridge=0.0,
        chart_counts=(1, 2),
        fold_count=3,
        minimum_chart_size=25,
        spectral_retained_ratio=0.9999,
        minimum_samples_per_parameter=1.0,
        support_cap_offset=0,
        accept_intercept_only_fallback=True,
        seed=260911,
    )
    st = select_joint_training_structure(slv, fit.training, fit.catalog, st_cfg).selected
    t_cfg = TrainingConfig(
        steps=5,
        learning_rate=0.001,
        save_steps=(0, 5),
        coordinate_penalty=1e-5,
        slope_penalty=1e-5,
        gradient_clip=10.0,
        seed=260911,
    )
    lib = build_training_selected_candidate_library(
        slv,
        fit.training,
        fit.catalog,
        st,
        ridge=0.0,
        training_config=t_cfg,
        global_sample_presentations_per_step=32,
        seed=260911,
    )
    fitted = select_atlas(lib, fit.validation, fit.catalog)
    prepared = fitted.prepare()
    decs = prepared.decisions(test.contexts[:8])
    assert decs.shape == (8, prob.decision_dim)
    assert np.all(np.isfinite(decs))


def test_grid_rolling_windows_and_table_renderer() -> None:
    """Verify exact 28-quarter day counts (1,004 + 1,553 = 2,557) and table rendering script."""
    d1, c1, o1, _, p1 = load_grid_panel(REPO_ROOT / "data/processed/opsd_grid64_2015_2020q3.npz")
    early_days = sum(len(slice_rolling_window(d1, c1, o1, s, e)["x_test"]) for _, s, e, _ in build_early_11_quarters())
    assert early_days == 1004

    d2, c2, o2, _, _ = load_grid_panel(REPO_ROOT / "data/processed/entsoe_grid64_2017q4_2024q4.npz")
    later_days = sum(len(slice_rolling_window(d2, c2, o2, s, e)["x_test"]) for _, s, e, _ in build_later_17_quarters())
    assert later_days == 1553

    red_xi, red_p = reduce_scenarios_wasserstein_kmeans(p1, o1[:200], 8, seed=42)
    assert red_xi.shape == (8, 64)
    assert np.isclose(np.sum(red_p), 1.0)
    assert newey_west_se(np.linspace(0.0, 1.0, 50), max_lag=14) > 0.0
    assert np.isclose(prescriptiveness_coefficient(np.array([25.0]), np.array([28.0]), np.array([20.0])), 0.375)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "render_manuscript_tables.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Table 3:" in proc.stdout
    assert "Table EC.4:" in proc.stdout
