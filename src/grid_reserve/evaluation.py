"""Rolling-window schedule, statistical inference, and deployment timing for European grid experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import dataclasses
import math
import multiprocessing
import statistics
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from grid_reserve.problem import GridSmoothSolver, InterconnectedGridReserveProblem
from kkt_atlas.baselines.sof import FittedSOF, SOFConfig, _fit_tree_group
from kkt_atlas.parallel_smooth_problem_v1 import ParallelSmoothProblem
from kkt_atlas.symmetric_laws_v1 import FiniteLaw

GRID_TRAIN_DAYS = 728
GRID_VAL_DAYS = 364


def quarter_spec(year: int, quarter: int, window_id: int) -> tuple[str, str, str, int]:
    starts = {1: f"{year}-01-01", 2: f"{year}-04-01", 3: f"{year}-07-01", 4: f"{year}-10-01"}
    ends = {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}
    return f"{year}Q{quarter}", starts[quarter], ends[quarter], int(window_id)


def build_early_11_quarters() -> list[tuple[str, str, str, int]]:
    """11 quarterly windows 2018Q1--2020Q3 (1,004 days, OPSD compilation)."""
    out: list[tuple[str, str, str, int]] = []
    wid = 1
    for yr in (2018, 2019, 2020):
        for qq in (1, 2, 3, 4):
            if yr == 2020 and qq == 4:
                continue
            out.append(quarter_spec(yr, qq, wid))
            wid += 1
    return out


def build_later_17_quarters() -> list[tuple[str, str, str, int]]:
    """17 quarterly windows 2020Q4--2024Q4 (1,553 days, ENTSO-E compilation)."""
    out: list[tuple[str, str, str, int]] = [quarter_spec(2020, 4, 13)]
    wid = 14
    for yr in range(2021, 2025):
        for qq in (1, 2, 3, 4):
            out.append(quarter_spec(yr, qq, wid))
            wid += 1
    return out


def slice_rolling_window(
    dates: list[str],
    contexts: np.ndarray,
    outcomes: np.ndarray,
    start_date: str,
    end_date: str,
    *,
    train_days: int = GRID_TRAIN_DAYS,
    val_days: int = GRID_VAL_DAYS,
) -> dict[str, Any]:
    """Slice 728-day training, 364-day validation, and quarterly out-of-sample evaluation splits."""
    test_indices = [i for i, d in enumerate(dates) if start_date <= d <= end_date]
    if not test_indices:
        raise ValueError(f"No panel dates found in [{start_date}, {end_date}]")
    i_test_start = test_indices[0]
    i_test_end = test_indices[-1] + 1
    if i_test_end - i_test_start != len(test_indices):
        raise ValueError("Non-contiguous quarterly test slice")
    i_val_end = i_test_start
    i_val_start = i_val_end - val_days
    i_tr_end = i_val_start
    i_tr_start = i_tr_end - train_days
    if i_tr_start < 0:
        raise ValueError(f"Insufficient history ({i_test_start} days) prior to {start_date}")
    return {
        "x_train": contexts[i_tr_start:i_tr_end].copy(),
        "xi_train": outcomes[i_tr_start:i_tr_end].copy(),
        "dates_train": dates[i_tr_start:i_tr_end],
        "x_val": contexts[i_val_start:i_val_end].copy(),
        "xi_val": outcomes[i_val_start:i_val_end].copy(),
        "dates_val": dates[i_val_start:i_val_end],
        "x_test": contexts[i_test_start:i_test_end].copy(),
        "xi_test": outcomes[i_test_start:i_test_end].copy(),
        "dates_test": dates[i_test_start:i_test_end],
    }


def eval_grid_laws(
    problem: InterconnectedGridReserveProblem,
    solver: ParallelSmoothProblem | GridSmoothSolver,
    contexts: np.ndarray,
    outcomes: np.ndarray,
    laws: tuple[FiniteLaw, ...],
    eta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve downstream problem for each context law and evaluate unregularized realized operational loss."""
    if hasattr(solver, "solve_laws_with_audit"):
        decisions, residuals = solver.solve_laws_with_audit(contexts, laws, ridge=eta)
        max_res = float(np.max(residuals))
    else:
        decs = []
        res_list = []
        for x, law in zip(contexts, laws, strict=True):
            z, _, aud, rep = solver.solve_with_audit(x, law.scenarios, law.probabilities, ridge=eta)
            decs.append(z)
            res_list.append(float(rep.get("maximum_acceptance_residual", aud.maximum_residual)))
        decisions = np.vstack(decs)
        max_res = float(max(res_list))
    if not math.isfinite(max_res) or max_res > 1e-6:
        raise RuntimeError(f"Grid KKT residual exceeded 1e-6: {max_res}")
    with torch.no_grad():
        losses = (
            problem.realized_loss_torch(
                torch.as_tensor(decisions, dtype=torch.float64),
                torch.as_tensor(outcomes, dtype=torch.float64),
                torch.as_tensor(contexts, dtype=torch.float64),
            )
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
    return decisions, losses, max_res


def solve_grid_hindsight_optimum(
    problem: InterconnectedGridReserveProblem,
    contexts: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute daily hindsight-optimal commitment (perfect foresight, unregularized) for prescriptiveness P."""
    serial = GridSmoothSolver(problem)
    w1 = np.array([1.0], dtype=np.float64)
    decs = []
    max_res = 0.0
    for x, xi in zip(contexts, outcomes, strict=True):
        z, _, aud, rep = serial.solve_with_audit(x, xi[None, :], w1, ridge=0.0)
        decs.append(z)
        max_res = max(max_res, float(rep.get("maximum_acceptance_residual", aud.maximum_residual)))
    z_arr = np.vstack(decs)
    with torch.no_grad():
        pf_losses = (
            problem.realized_loss_torch(
                torch.as_tensor(z_arr, dtype=torch.float64),
                torch.as_tensor(outcomes, dtype=torch.float64),
                torch.as_tensor(contexts, dtype=torch.float64),
            )
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
    return z_arr, pf_losses, max_res


def newey_west_se(series: np.ndarray, max_lag: int = 14) -> float:
    """Newey-West (1987) heteroskedasticity- and autocorrelation-consistent standard error (Eq. EC.26)."""
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    n = len(arr)
    if n <= 1:
        return 0.0
    centered = arr - float(np.mean(arr))
    gamma0 = float(np.dot(centered, centered)) / float(n)
    l_eff = min(int(max_lag), n - 1)
    var_nw = gamma0
    for lag in range(1, l_eff + 1):
        weight = 1.0 - float(lag) / float(l_eff + 1)
        gamma_l = float(np.dot(centered[lag:], centered[:-lag])) / float(n)
        var_nw += 2.0 * weight * gamma_l
    return math.sqrt(max(0.0, var_nw) / float(n))


def prescriptiveness_coefficient(
    method_losses: np.ndarray,
    saa_losses: np.ndarray,
    hindsight_losses: np.ndarray,
) -> float:
    """Bertsimas & Kallus (2020) coefficient of prescriptiveness P = (L_SAA - L_m) / (L_SAA - L_PF)."""
    l_m = float(np.mean(method_losses))
    l_saa = float(np.mean(saa_losses))
    l_pf = float(np.mean(hindsight_losses))
    denom = l_saa - l_pf
    if denom <= 1e-12:
        return 0.0
    return (l_saa - l_m) / denom


def _sof_worker_init() -> None:
    torch.set_num_threads(1)


def fit_sof_parallel(
    serial_solver: Any,
    x_train: np.ndarray,
    xi_train: np.ndarray,
    sof_cfg: SOFConfig,
) -> FittedSOF:
    """Cross-platform parallel Stochastic Optimization Forest fitting."""
    training_contexts = np.asarray(x_train, dtype=np.float64).copy()
    training_outcomes = np.asarray(xi_train, dtype=np.float64).copy()
    n_workers = max(1, min(int(sof_cfg.worker_count), int(sof_cfg.tree_count)))
    groups = tuple(
        tuple(int(v) for v in grp.tolist())
        for grp in np.array_split(np.arange(sof_cfg.tree_count), n_workers)
    )
    serial_cfg = dataclasses.replace(sof_cfg, worker_count=1)
    if n_workers <= 1:
        with threadpool_limits(limits=1):
            fitted_groups = (
                _fit_tree_group(serial_solver, training_contexts, training_outcomes, serial_cfg, groups[0]),
            )
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_sof_worker_init,
        ) as pool:
            futs = [
                pool.submit(_fit_tree_group, serial_solver, training_contexts, training_outcomes, serial_cfg, g)
                for g in groups
            ]
            fitted_groups = tuple(f.result() for f in futs)
    trees = tuple(sorted((t for grp in fitted_groups for t in grp), key=lambda t: t.index))
    return FittedSOF(
        training_contexts=training_contexts,
        training_outcomes=training_outcomes,
        config=sof_cfg,
        trees=trees,
    )


def time_grid_deployment(
    predict_fn: Any,
    contexts_128: np.ndarray,
    *,
    serial_solver: GridSmoothSolver,
    par_solver: ParallelSmoothProblem,
    eta: float,
    warmups: int = 2,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Measure single-context (1 core), 128-context (1 core), and 128-context (parallel) deployment medians."""
    ctx_1 = np.ascontiguousarray(contexts_128[:1])
    ctx_128 = np.ascontiguousarray(contexts_128[:128])

    def run_1_1() -> None:
        laws = predict_fn(ctx_1)
        _, _, aud, rep = serial_solver.solve_with_audit(
            ctx_1[0], laws[0].scenarios, laws[0].probabilities, ridge=eta
        )
        assert float(rep.get("maximum_acceptance_residual", aud.maximum_residual)) <= 1e-6

    def run_128_1() -> None:
        laws = predict_fn(ctx_128)
        for idx in range(len(ctx_128)):
            _, _, aud, rep = serial_solver.solve_with_audit(
                ctx_128[idx], laws[idx].scenarios, laws[idx].probabilities, ridge=eta
            )
            assert float(rep.get("maximum_acceptance_residual", aud.maximum_residual)) <= 1e-6

    def run_128_par() -> None:
        laws = predict_fn(ctx_128)
        _, res = par_solver.solve_laws_with_audit(ctx_128, laws, ridge=eta)
        assert float(np.max(res)) <= 1e-6

    out: dict[str, Any] = {}
    for key, fn, mult in (
        ("ctx1_core1_ms", run_1_1, 1000.0),
        ("ctx128_core1_s", run_128_1, 1.0),
        ("ctx128_parallel_ms", run_128_par, 1000.0),
    ):
        for _ in range(warmups):
            fn()
        samples = []
        for _ in range(repetitions):
            t0 = time.perf_counter_ns()
            fn()
            samples.append(((time.perf_counter_ns() - t0) / 1e9) * mult)
        out[key] = {"median": float(statistics.median(samples)), "samples": samples}
    return out
