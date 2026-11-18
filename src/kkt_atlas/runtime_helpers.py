"""Clean cross-platform solver, data-generation, and exact-evaluation helpers for KKT Atlas."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from kkt_atlas.case_studies.capacity_continuous_data_v2 import (
    ContinuousCapacityDataConfig,
    generate_continuous_capacity_fitting_data_v2,
    generate_continuous_capacity_test_data_v2,
)
from kkt_atlas.case_studies.capacity_regime_data_v1 import regime_data
from kkt_atlas.case_studies.network_reservation import (
    NetworkReservationProblem,
    make_network_reservation_problem,
)
from kkt_atlas.case_studies.network_reservation_data_v1 import (
    NetworkReservationDataConfig,
    generate_network_reservation_fitting_data_v1,
    generate_network_reservation_test_data_v1,
)
from kkt_atlas.case_studies.portfolio_continuous_data_v2 import (
    ContinuousPortfolioDataConfig,
    generate_continuous_portfolio_fitting_data_v2,
    generate_continuous_portfolio_test_data_v2,
)
from kkt_atlas.case_studies.smooth_capacity_allocation import (
    SmoothCapacityAllocationProblem,
    make_smooth_capacity_allocation_problem,
)
from kkt_atlas.case_studies.smoothed_cvar_portfolio import SmoothedCVaRPortfolioProblem
from kkt_atlas.clarabel_smooth_problem_v1 import ClarabelSmoothProblem
from kkt_atlas.confirmation_data_v1 import ConditionalTestData, FittingCatalogData
from kkt_atlas.converged_network_problem_v1 import ConvergedNetworkProblem
from kkt_atlas.converged_smooth_problem_v1 import ConvergedSmoothProblem
from kkt_atlas.evaluation import exact_expected_loss
from kkt_atlas.symmetric_campaign_protocol_v1 import (
    SymmetricCampaignProtocol,
    derive_seed64,
    load_symmetric_campaign_protocol,
)
from kkt_atlas.symmetric_laws_v1 import predict_method_laws
from kkt_atlas.training import TrainingConfig
from kkt_atlas.windows_affinity_v1 import apply_affinity_from_environment

ARMS: dict[str, tuple[float, bool]] = {
    "A": (0.01, True),
    "B": (0.001, True),
    "C": (0.001, False),
}
CAP_FIXED_TECH_SEED = 240991


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def solver_for(problem: Any) -> Any:
    base = getattr(problem, "_problem", problem)
    if isinstance(base, NetworkReservationProblem):
        return ConvergedNetworkProblem(base)
    if isinstance(base, SmoothCapacityAllocationProblem):
        return ConvergedSmoothProblem(base)
    return ClarabelSmoothProblem(base, fallback_on_failure=True)


def training_config(
    protocol: SymmetricCampaignProtocol,
    arm: str,
    steps: tuple[int, ...],
    *,
    seed: int,
) -> TrainingConfig:
    settings = protocol.raw["training"]
    rate, move_coords = ARMS[arm]
    return TrainingConfig(
        steps=int(steps[-1]),
        learning_rate=float(rate),
        save_steps=tuple(int(s) for s in steps),
        coordinate_penalty=float(settings["coordinate_penalty"]),
        slope_penalty=float(settings["probability_slope_penalty"]),
        gradient_clip=float(settings["gradient_clip"]),
        train_scenario_coordinates=bool(move_coords),
        train_probability_slopes=True,
        seed=int(seed),
    )


def score_validation(
    problem: Any,
    fitting: FittingCatalogData,
    model: Any,
    method: str,
    eta: float,
    *,
    scoring_solver: Any = None,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, float]]:
    laws = predict_method_laws(model, method, fitting.validation.contexts, fitting)
    solver = solver_for(problem) if scoring_solver is None else scoring_solver
    if hasattr(solver, "solve_laws_with_audit"):
        z, residuals_arr = solver.solve_laws_with_audit(fitting.validation.contexts, laws, ridge=eta)
        max_res = float(np.max(residuals_arr))
        max_orig = max_res
    else:
        decisions = []
        residuals = []
        orig_kkt = []
        for x, law in zip(fitting.validation.contexts, laws, strict=True):
            if isinstance(solver, ConvergedNetworkProblem):
                res, _ = solver.solve_with_routes(x, law.scenarios, law.probabilities, ridge=eta)
                decisions.append(res.decision)
                r_val = max(res.reservation_residual, res.maximum_recourse_residual)
                residuals.append(r_val)
                orig_kkt.append(r_val)
            else:
                d, _, audit, report = solver.solve_with_audit(
                    x, law.scenarios, law.probabilities, ridge=eta
                )
                decisions.append(d)
                residuals.append(float(report.get("maximum_acceptance_residual", audit.maximum_residual)))
                orig_kkt.append(float(audit.maximum_residual))
        z = np.stack(decisions)
        max_res = float(max(residuals))
        max_orig = float(max(orig_kkt))
    if max_res > 1e-6:
        raise RuntimeError(f"Validation KKT residual {max_res:.3e} exceeded 1e-6")
    losses = (
        solver.realized_loss_torch(
            torch.tensor(z, dtype=torch.float64),
            torch.tensor(fitting.validation.outcomes(fitting.catalog), dtype=torch.float64),
            torch.tensor(fitting.validation.contexts, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    return losses, z, laws, {
        "maximum_acceptance_residual": max_res,
        "maximum_original_kkt_residual": max_orig,
    }


def generate_synthetic_application_instance(
    application: str,
    seed: int,
    *,
    protocol: SymmetricCampaignProtocol | None = None,
    training_size: int = 12800,
    support_size: int = 500,
) -> tuple[SymmetricCampaignProtocol, Any, FittingCatalogData, ConditionalTestData]:
    """Generate exact CAP, PORT, or NET synthetic problem + (train, val, eval) splits from seed."""
    proto = protocol or load_symmetric_campaign_protocol()
    app_spec = proto.application(application)
    val_size = int(app_spec["validation_size"])
    eval_size = int(app_spec["evaluation_size"])
    ctx_dim = int(app_spec["context_dim"])
    m_dim = int(app_spec["anchor"]["m"])

    if application == "capacity":
        # Fixed physical service technology across replications, per EC.2.1 / EC.3.1
        prob_kwargs = {
            k: v for k, v in app_spec["problem"].items() if k != "resource_count_rule"
        }
        problem = make_smooth_capacity_allocation_problem(
            m_dim,
            resource_count=min(8, m_dim),
            context_dim=ctx_dim,
            seed=CAP_FIXED_TECH_SEED,
            **prob_kwargs,
        )
        cfg = ContinuousCapacityDataConfig(
            support_size=int(support_size),
            scenario_dim=m_dim,
            training_size=int(training_size),
            validation_size=val_size,
            test_size=eval_size,
            context_dim=ctx_dim,
            latent_rank=int(app_spec["latent_rank"]),
            seed=int(seed),
            master_support_size=int(app_spec["master_support_size"]),
            master_scenario_dim=int(app_spec["master_scenario_dim"]),
            **app_spec["dgp"],
        )
        fit, test, _ = regime_data(
            generate_continuous_capacity_fitting_data_v2(cfg),
            generate_continuous_capacity_test_data_v2(cfg),
            problem.service_matrix,
            problem.total_capacity,
        )
        return proto, problem, fit, test

    if application == "portfolio":
        p_spec = app_spec["problem"]
        problem = SmoothedCVaRPortfolioProblem(
            asset_count=m_dim,
            context_dim=ctx_dim,
            return_bound=float(p_spec["return_bound"]),
            tail_probability=float(p_spec["tail_probability"]),
            smoothing=float(p_spec["smoothing"]),
            expected_return_weight=float(p_spec["expected_return_weight"]),
            leverage_penalty=float(p_spec["leverage_penalty"]),
            gradient_iterations=int(p_spec["gradient_iterations"]),
            step_fraction=float(p_spec["step_fraction"]),
            training_solver=str(p_spec["training_solver"]),
            newton_damping=float(p_spec["newton_damping"]),
        )
        cfg_p = ContinuousPortfolioDataConfig(
            support_size=int(support_size),
            asset_count=m_dim,
            training_size=int(training_size),
            validation_size=val_size,
            test_size=eval_size,
            context_dim=ctx_dim,
            factor_count=int(app_spec["latent_rank"]),
            seed=int(seed),
            master_support_size=int(app_spec["master_support_size"]),
            master_asset_count=int(app_spec["master_scenario_dim"]),
            **app_spec["dgp"],
        )
        return (
            proto,
            problem,
            generate_continuous_portfolio_fitting_data_v2(cfg_p),
            generate_continuous_portfolio_test_data_v2(cfg_p),
        )

    if application == "network":
        n_prob = app_spec["problem"]
        problem = make_network_reservation_problem(
            m_dim,
            seed=int(seed),
            context_dim=ctx_dim,
            reservation_cost=float(n_prob["reservation_cost"]),
            congestion_penalty=float(n_prob["congestion_penalty"]),
            routing_penalty=float(n_prob["routing_regularization"]),
            reservation_upper_bound=float(n_prob["reservation_upper_bound"]),
            decision_iterations=int(n_prob["decision_iterations"]),
            routing_updates_per_iteration=int(n_prob["routing_updates_per_iteration"]),
            realized_loss_routing_iterations=int(n_prob["realized_loss_routing_iterations"]),
            step_fraction=float(n_prob["step_fraction"]),
        )
        cfg_n = NetworkReservationDataConfig(
            support_size=int(support_size),
            scenario_dim=m_dim,
            training_size=int(training_size),
            validation_size=val_size,
            test_size=eval_size,
            context_dim=ctx_dim,
            latent_rank=int(app_spec["latent_rank"]),
            seed=int(seed),
            master_support_size=int(app_spec["master_support_size"]),
            master_scenario_dim=int(app_spec["master_scenario_dim"]),
            **app_spec["dgp"],
        )
        return (
            proto,
            problem,
            generate_network_reservation_fitting_data_v1(cfg_n),
            generate_network_reservation_test_data_v1(cfg_n),
        )

    raise ValueError(f"Unknown synthetic application: {application}")


_EVAL_SOLVER: Any = None
_EVAL_THREADS: Any = None


def evaluation_worker_init(problem: Any) -> None:
    global _EVAL_SOLVER, _EVAL_THREADS
    apply_affinity_from_environment()
    _EVAL_THREADS = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    _EVAL_SOLVER = solver_for(problem)
    if isinstance(_EVAL_SOLVER, ConvergedNetworkProblem):
        _EVAL_SOLVER.solver_threads = 1


def solve_and_eval_one_context(solver: Any, task: tuple[Any, ...]) -> tuple[np.ndarray, float, bool, float]:
    application, context, scenarios, probabilities, ridge, catalog, true_probabilities = task
    if application == "network":
        res, _ = solver.solve_with_routes(context, scenarios, probabilities, ridge=ridge)
        decision = res.decision
        residual = float(max(res.reservation_residual, res.maximum_recourse_residual))
        fallback = False
    else:
        decision, _, audit, report = solver.solve_with_audit(
            context, scenarios, probabilities, ridge=ridge
        )
        residual = float(report.get("maximum_acceptance_residual", audit.maximum_residual))
        fallback = bool(report.get("fallback_used", False))
    if residual > 1e-6:
        raise RuntimeError(f"Evaluation KKT residual {residual:.3e} exceeded 1e-6")
    risk = float(
        exact_expected_loss(
            solver,
            decision[None],
            context[None],
            catalog,
            true_probabilities[None],
            context_batch_size=1,
            scenario_batch_size=64,
        ).mean_loss
    )
    return decision, residual, fallback, risk


def evaluation_worker_task(task: tuple[Any, ...]) -> tuple[np.ndarray, float, bool, float]:
    return solve_and_eval_one_context(_EVAL_SOLVER, task)
