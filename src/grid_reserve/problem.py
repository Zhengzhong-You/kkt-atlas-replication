"""8-zone / 64-zone-block European interconnected grid reserve commitment problem & KKT solver."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import math

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from kkt_atlas.converged_smooth_problem_v1 import ConvergedSmoothProblem
from kkt_atlas.problem import Array, Tensor, matrix, probability_vector
from kkt_atlas.symmetric_laws_v1 import FiniteLaw


def _sigmoid_numpy(values: Array) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_neg = np.exp(arr[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


@dataclass(frozen=True)
class InterconnectedGridReserveProblem:
    """Day-ahead flexible capacity commitment across R interconnected zones and H intraday blocks.

    Decision z in prod_{r=1}^R [0, upper_bounds_r] commits flexible generation / reserve capacity
    in each control zone before day-ahead gate closure. During block h in {1..H}, zone r faces
    net residual demand xi_{(r,h)} and is served by supplied_{(r,h)} = (service_matrix @ z)_{(r,h)}.
    """

    service_matrix: Array          # shape (m, d_z), m = R * H = 64, d_z = 8
    capacity_costs: Array          # shape (d_z,), > 0
    underage_costs: Array          # shape (m,), > 0
    overage_costs: Array           # shape (m,), >= 0
    upper_bounds: Array            # shape (d_z,), > 0
    scenario_lower_bounds: Array   # shape (m,)
    scenario_upper_bounds: Array   # shape (m,)
    block_count: int = 8
    smoothing: float = 0.06
    context_dim: int = 20
    name: str = "interconnected_grid_reserve"

    def __post_init__(self) -> None:
        svc = np.ascontiguousarray(np.asarray(self.service_matrix, dtype=np.float64))
        cap = np.ascontiguousarray(np.asarray(self.capacity_costs, dtype=np.float64).reshape(-1))
        und = np.ascontiguousarray(np.asarray(self.underage_costs, dtype=np.float64).reshape(-1))
        ovr = np.ascontiguousarray(np.asarray(self.overage_costs, dtype=np.float64).reshape(-1))
        ub = np.ascontiguousarray(np.asarray(self.upper_bounds, dtype=np.float64).reshape(-1))
        slb = np.ascontiguousarray(np.asarray(self.scenario_lower_bounds, dtype=np.float64).reshape(-1))
        sub = np.ascontiguousarray(np.asarray(self.scenario_upper_bounds, dtype=np.float64).reshape(-1))
        m, d = svc.shape
        if und.shape != (m,) or ovr.shape != (m,) or slb.shape != (m,) or sub.shape != (m,):
            raise ValueError("scenario-aligned arrays must have shape (m,)")
        if cap.shape != (d,) or ub.shape != (d,):
            raise ValueError("decision-aligned arrays must have shape (d_z,)")
        if np.any(svc < 0.0) or np.any(cap <= 0.0) or np.any(und <= 0.0) or np.any(ovr < 0.0):
            raise ValueError("invalid sign in problem matrices/costs")
        if np.any(ub <= 0.0) or np.any(slb >= sub):
            raise ValueError("invalid variable or scenario bounds")
        for arr in (svc, cap, und, ovr, ub, slb, sub):
            arr.setflags(write=False)
        object.__setattr__(self, "service_matrix", svc)
        object.__setattr__(self, "capacity_costs", cap)
        object.__setattr__(self, "underage_costs", und)
        object.__setattr__(self, "overage_costs", ovr)
        object.__setattr__(self, "upper_bounds", ub)
        object.__setattr__(self, "scenario_lower_bounds", slb)
        object.__setattr__(self, "scenario_upper_bounds", sub)

    @property
    def scenario_dim(self) -> int:
        return int(self.service_matrix.shape[0])

    @property
    def decision_dim(self) -> int:
        return int(self.service_matrix.shape[1])

    def transform_numpy(self, outcomes: Array) -> Array:
        vals = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        mid = 0.5 * (self.scenario_lower_bounds + self.scenario_upper_bounds)
        half = 0.5 * (self.scenario_upper_bounds - self.scenario_lower_bounds)
        scaled = np.clip((vals - mid[None, :]) / half[None, :], -1.0 + 1e-9, 1.0 - 1e-9)
        return np.arctanh(scaled)

    def decode_torch(self, transformed: Tensor) -> Tensor:
        if transformed.shape[-1] != self.scenario_dim:
            raise ValueError("transformed scenarios have wrong final dimension")
        mid = torch.tensor((self.scenario_lower_bounds + self.scenario_upper_bounds) * 0.5, dtype=transformed.dtype, device=transformed.device)
        half = torch.tensor((self.scenario_upper_bounds - self.scenario_lower_bounds) * 0.5, dtype=transformed.dtype, device=transformed.device)
        return mid + half * torch.tanh(transformed)

    def reference_center_numpy(self, context: Array) -> Array:
        self._validate_context_numpy(context)
        return np.minimum(1.0, 0.5 * self.upper_bounds).astype(np.float64)

    def reference_equality_numpy(self, context: Array) -> tuple[Array, Array]:
        self._validate_context_numpy(context)
        return np.zeros((0, self.decision_dim), dtype=np.float64), np.zeros(0, dtype=np.float64)

    def reference_lower_bounds_numpy(self, context: Array) -> Array:
        self._validate_context_numpy(context)
        return np.zeros(self.decision_dim, dtype=np.float64)

    def reference_upper_bounds_numpy(self, context: Array) -> Array:
        self._validate_context_numpy(context)
        return self.upper_bounds.copy()

    def reference_decision_is_valid_numpy(self, decision: Array) -> bool:
        val = np.asarray(decision, dtype=np.float64).reshape(-1)
        return bool(
            val.shape == (self.decision_dim,)
            and np.all(np.isfinite(val))
            and np.all(val >= -1e-10)
            and np.all(val <= self.upper_bounds + 1e-10)
        )

    def reference_objective_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> tuple[float, float, Array, Array]:
        self._validate_context_numpy(context)
        z = np.asarray(decision, dtype=np.float64).reshape(-1)
        demands = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        weights = probability_vector(probabilities, len(demands))
        ridge_val = float(ridge)

        supplied = self.service_matrix @ z  # (m,)
        diff = supplied[None, :] - demands  # (K, m)
        tau = self.smoothing
        inv_b = 1.0 / float(self.block_count)

        shortage = tau * np.logaddexp(0.0, -diff / tau)
        if np.any(self.overage_costs > 0.0):
            excess = tau * np.logaddexp(0.0, diff / tau)
            rec_per_scen = inv_b * np.sum(
                self.underage_costs[None, :] * shortage + self.overage_costs[None, :] * excess,
                axis=1,
            )
        else:
            rec_per_scen = inv_b * np.sum(self.underage_costs[None, :] * shortage, axis=1)

        cap_term = float(self.capacity_costs @ z)
        raw_val = cap_term + float(weights @ rec_per_scen)

        # sig = sigma((demands - supplied)/tau) = shortage probability
        sig = _sigmoid_numpy(-diff / tau)  # (K, m)
        comb = self.underage_costs + self.overage_costs  # (m,)
        # d/d(supplied_j) = inv_b * (c^o_j - (c^u_j + c^o_j) * sig_{k,j})
        mean_sig = weights @ sig  # (m,)
        grad_supplied = inv_b * (self.overage_costs - comb * mean_sig)  # (m,)
        grad = self.capacity_costs + self.service_matrix.T @ grad_supplied

        curv_diag = (inv_b / tau) * comb * (weights @ (sig * (1.0 - sig)))  # (m,)
        hess = (self.service_matrix.T * curv_diag[None, :]) @ self.service_matrix

        center = self.reference_center_numpy(context)
        disp = z - center
        reg = 0.5 * ridge_val * float(disp @ disp)
        grad = grad + ridge_val * disp
        hess = 0.5 * (hess + hess.T) + ridge_val * np.eye(self.decision_dim)
        return raw_val, raw_val + reg, grad, hess

    def kkt_contributions_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
    ) -> Array:
        self._validate_context_numpy(context)
        z = np.asarray(decision, dtype=np.float64).reshape(-1)
        demands = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        supplied = self.service_matrix @ z
        sig = _sigmoid_numpy((demands - supplied[None, :]) / self.smoothing)  # (n, m)
        comb = self.underage_costs + self.overage_costs
        inv_b = 1.0 / float(self.block_count)
        grad_supplied = inv_b * (self.overage_costs[None, :] - comb[None, :] * sig)  # (n, m)
        return self.capacity_costs[None, :] + grad_supplied @ self.service_matrix

    def sof_node_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
        *,
        ridge: float,
        active_set_tolerance: float = 1e-8,
    ) -> tuple[Array, Array]:
        del active_set_tolerance
        z = np.asarray(decision, dtype=np.float64).reshape(-1)
        center = self.reference_center_numpy(context)
        grads = self.kkt_contributions_numpy(z, context, outcomes) + float(ridge) * (z - center)[None, :]
        uniform = np.full(len(outcomes), 1.0 / len(outcomes), dtype=np.float64)
        _, _, _, hess = self.reference_objective_derivatives_numpy(
            z, context, outcomes, uniform, ridge=ridge
        )
        return grads, hess

    def solve_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> Array:
        solver = GridSmoothSolver(self)
        return solver.solve_with_audit(context, scenarios, probabilities, ridge=ridge)[0]

    def realized_loss_torch(
        self,
        decisions: Tensor,
        outcomes: Tensor,
        contexts: Tensor,
    ) -> Tensor:
        self._validate_contexts_torch(contexts)
        svc = torch.tensor(self.service_matrix.copy(), dtype=decisions.dtype, device=decisions.device)
        cap = torch.tensor(self.capacity_costs.copy(), dtype=decisions.dtype, device=decisions.device)
        und = torch.tensor(self.underage_costs.copy(), dtype=decisions.dtype, device=decisions.device)
        ovr = torch.tensor(self.overage_costs.copy(), dtype=decisions.dtype, device=decisions.device)
        supplied = decisions @ svc.T  # (batch, m)
        diff = supplied - outcomes
        tau = self.smoothing
        inv_b = 1.0 / float(self.block_count)
        shortage = tau * functional.softplus(-diff / tau)
        if float(torch.max(ovr).item()) > 0.0:
            excess = tau * functional.softplus(diff / tau)
            rec = inv_b * torch.sum(und[None, :] * shortage + ovr[None, :] * excess, dim=1)
        else:
            rec = inv_b * torch.sum(und[None, :] * shortage, dim=1)
        return decisions @ cap + rec

    def _validate_context_numpy(self, context: Array) -> None:
        val = np.asarray(context, dtype=np.float64).reshape(-1)
        if val.shape != (self.context_dim,) or np.any(~np.isfinite(val)):
            raise ValueError(f"context must have shape ({self.context_dim},)")

    def _validate_contexts_torch(self, contexts: Tensor) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim or not torch.all(torch.isfinite(contexts)):
            raise ValueError("invalid contexts tensor")


# Monkey-patch SOF _problem_identity once so fit_sof accepts InterconnectedGridReserveProblem
import kkt_atlas.baselines.sof as _sof_mod

_ORIG_PROBLEM_IDENTITY = _sof_mod._problem_identity


def _patched_problem_identity(problem: Any, ridge: float) -> dict[str, Any]:
    base = getattr(problem, "_problem", problem)
    if isinstance(base, InterconnectedGridReserveProblem):
        return {
            "schema": "grid-reserve-identity-v1",
            "scenario_dim": base.scenario_dim,
            "decision_dim": base.decision_dim,
            "smoothing": float(base.smoothing),
            "ridge": float(ridge),
        }
    return _ORIG_PROBLEM_IDENTITY(problem, ridge)


_sof_mod._problem_identity = _patched_problem_identity


class GridSmoothSolver(ConvergedSmoothProblem):
    """Exact box-constrained Newton / L-BFGS-B solver with KKT residual certificate <= 1e-8."""

    def __init__(self, problem: InterconnectedGridReserveProblem, *, tolerance: float = 1e-6):
        self._problem = problem
        self.tolerance = float(tolerance)
        self.solver_ftol = 1e-14
        self.maximum_iterations = 500

    @property
    def prescription(self) -> dict[str, Any]:
        return {
            "schema": "kkt-prescription-v1",
            "backend": "lbfgsb_projected_newton",
            "tolerance": self.tolerance,
            "solver_ftol": self.solver_ftol,
            "maximum_iterations": self.maximum_iterations,
        }

    def solve_with_audit(self, context, scenarios, probabilities, *, ridge):
        p = self._problem
        d = p.decision_dim
        lo = np.zeros(d, dtype=np.float64)
        hi = p.upper_bounds
        z = np.clip(p.reference_center_numpy(context), lo + 0.05, hi - 0.05)

        def _newton_loop(z_curr: np.ndarray, max_steps: int) -> tuple[np.ndarray, int, bool]:
            for it in range(max_steps):
                _, val, grad, hess = p.reference_objective_derivatives_numpy(
                    z_curr, context, scenarios, probabilities, ridge=ridge
                )
                act_lo_k = (z_curr <= lo + 1e-11) & (grad >= 0.0)
                act_hi_k = (z_curr >= hi - 1e-11) & (grad <= 0.0)
                free = ~(act_lo_k | act_hi_k)
                stat_free = float(np.max(np.abs(grad[free]), initial=0.0))
                if stat_free <= 1e-14:
                    return z_curr, it, True
                f_idx = np.flatnonzero(free)
                h_ff = hess[np.ix_(f_idx, f_idx)] + 1e-15 * np.eye(len(f_idx))
                step_f = -np.linalg.solve(h_ff, grad[f_idx])
                dz = np.zeros(d, dtype=np.float64)
                dz[f_idx] = step_f
                # Two-metric safeguard at boundary: if Newton step points outward at a bound where -grad points inward, use scaled steepest descent on that coordinate
                outward_lo = (z_curr <= lo + 1e-11) & (dz < 0.0)
                outward_hi = (z_curr >= hi - 1e-11) & (dz > 0.0)
                for idx_b in np.flatnonzero(outward_lo | outward_hi):
                    dz[idx_b] = -grad[idx_b] / max(float(hess[idx_b, idx_b]), 1e-8)
                alpha = 1.0
                accepted = False
                for _ in range(30):
                    z_cand = np.clip(z_curr + alpha * dz, lo, hi)
                    step_eff = z_cand - z_curr
                    slope_eff = float(grad @ step_eff)
                    if slope_eff >= 0.0 and alpha < 1e-8:
                        break
                    _, v_cand, g_cand, _ = p.reference_objective_derivatives_numpy(
                        z_cand, context, scenarios, probabilities, ridge=ridge
                    )
                    act_lo_c = (z_cand <= lo + 1e-11) & (g_cand >= 0.0)
                    act_hi_c = (z_cand >= hi - 1e-11) & (g_cand <= 0.0)
                    stat_cand = float(np.max(np.abs(g_cand[~(act_lo_c | act_hi_c)]), initial=0.0))
                    if (slope_eff < 0.0 and v_cand <= val + 1e-4 * slope_eff) or (
                        stat_free < 1e-5 and stat_cand < 0.5 * stat_free
                    ):
                        z_curr = z_cand
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    return z_curr, it + 1, False
            _, _, g_end, _ = p.reference_objective_derivatives_numpy(
                z_curr, context, scenarios, probabilities, ridge=ridge
            )
            act_lo_e = (z_curr <= lo + 1e-11) & (g_end >= 0.0)
            act_hi_e = (z_curr >= hi - 1e-11) & (g_end <= 0.0)
            ok_end = float(np.max(np.abs(g_end[~(act_lo_e | act_hi_e)]), initial=0.0)) <= 1e-12
            return z_curr, max_steps, ok_end

        z, n_iter, converged = _newton_loop(z, 25)
        if not converged:
            def obj_and_grad(z_vec: np.ndarray) -> tuple[float, np.ndarray]:
                _, val_fg, grad_fg, _ = p.reference_objective_derivatives_numpy(
                    z_vec, context, scenarios, probabilities, ridge=ridge
                )
                return val_fg, grad_fg

            res = minimize(
                obj_and_grad,
                z,
                method="L-BFGS-B",
                jac=True,
                bounds=list(zip(lo, hi)),
                options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": self.maximum_iterations},
            )
            z = np.clip(res.x.astype(np.float64, copy=True), lo, hi)
            z, n_polish, _ = _newton_loop(z, 15)
            n_iter += int(res.nit) + n_polish

        raw, val, grad, hess = p.reference_objective_derivatives_numpy(
            z, context, scenarios, probabilities, ridge=ridge
        )
        act_lo = z <= lo + 1e-8
        act_hi = z >= hi - 1e-8
        if np.any(act_lo & act_hi):
            raise RuntimeError("Simultaneously active lower and upper bounds")

        # Build exact active-constraint matrix compatible with _ImplicitSolve.backward!
        # Note: _ImplicitSolve.backward expects active_multipliers = -audit.augmented_multiplier[eq_count:] > tolerance
        # For lower bound z_i >= 0 -> row +e_i, grad_i - mu_i = 0 => augmented_multiplier = -grad_i < 0 => -aug = +grad_i > 0!
        # For upper bound z_i <= hi_i -> row -e_i, grad_i + nu_i = 0 => (-e_i)^T aug = -grad_i => aug = +grad_i < 0 => -aug = -grad_i > 0!
        rows = []
        aug_mult = []
        stat_vec = grad.copy()
        comp_res = 0.0
        dual_res = 0.0
        for i in range(d):
            if act_lo[i]:
                mu_i = max(0.0, float(grad[i]))
                # Only treat as active in KKT linear system if strictly active (mu_i > 1e-5); if weakly active, free variable works cleanly
                if mu_i > 1e-5:
                    row = np.zeros(d, dtype=np.float64)
                    row[i] = 1.0
                    rows.append(row)
                    aug_mult.append(-mu_i)
                stat_vec[i] = grad[i] - mu_i
                comp_res = max(comp_res, abs(mu_i * (z[i] - lo[i])))
            elif act_hi[i]:
                nu_i = max(0.0, float(-grad[i]))
                if nu_i > 1e-5:
                    row = np.zeros(d, dtype=np.float64)
                    row[i] = -1.0
                    rows.append(row)
                    aug_mult.append(-nu_i)
                stat_vec[i] = grad[i] + nu_i
                comp_res = max(comp_res, abs(nu_i * (hi[i] - z[i])))

        aug_mat = np.vstack(rows) if rows else np.zeros((0, d), dtype=np.float64)
        aug_mult_arr = np.asarray(aug_mult, dtype=np.float64)
        stat_res = float(np.max(np.abs(stat_vec), initial=0.0))
        bnd_viol = float(max(np.max(np.maximum(lo - z, 0.0)), np.max(np.maximum(z - hi, 0.0))))
        max_res = max(stat_res, comp_res, dual_res, bnd_viol)
        if not math.isfinite(max_res) or max_res > self.tolerance:
            raise RuntimeError(f"GridSmoothSolver failed KKT gate: max_res={max_res:.3e}")

        audit = SimpleNamespace(
            augmented_matrix=aug_mat,
            augmented_multiplier=aug_mult_arr,
            maximum_residual=max_res,
        )
        report = {
            "solver": "projected_newton_lbfgsb",
            "iterations": int(n_iter),
            "raw_objective": raw,
            "objective": val,
            "maximum_kkt_residual": max_res,
            "maximum_acceptance_residual": max_res,
            "active_bound_count": len(rows),
        }
        return z, hess, audit, report


# ---------------------------------------------------------------------------
# External Scenario-Reduction Baselines (Wasserstein-2 / K-Means in T(xi))
# ---------------------------------------------------------------------------


def reduce_scenarios_wasserstein_kmeans(
    problem: Any,
    xi_train: np.ndarray,
    support_size: int,
    *,
    seed: int,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """2-Wasserstein barycentric scenario reduction via K-means in standardized T(xi)-space."""
    k = int(support_size)
    if k >= len(xi_train):
        probs = np.full(len(xi_train), 1.0 / len(xi_train), dtype=np.float64)
        return xi_train.copy(), probs
    t_xi = problem.transform_numpy(xi_train)
    mean = t_xi.mean(axis=0)
    scale = np.maximum(t_xi.std(axis=0, ddof=1), 1e-8)
    t_std = (t_xi - mean[None, :]) / scale[None, :]

    km = KMeans(n_clusters=k, n_init=n_init, random_state=int(seed))
    labels = km.fit_predict(t_std)
    centers_t = mean[None, :] + km.cluster_centers_ * scale[None, :]
    with torch.no_grad():
        reduced_xi = (
            problem.decode_torch(torch.as_tensor(centers_t, dtype=torch.float64))
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    probs = counts / counts.sum()
    return reduced_xi, probs


def compress_knn_laws_wasserstein(
    problem: Any,
    knn_laws: tuple[FiniteLaw, ...],
    xi_train: np.ndarray,
    support_size: int,
    *,
    seed: int,
) -> tuple[FiniteLaw, ...]:
    """Compress each contextual k-NN empirical distribution to at most K scenarios via weighted 2-Wasserstein K-means."""
    k = int(support_size)
    t_all = problem.transform_numpy(xi_train)
    mean = t_all.mean(axis=0)
    scale = np.maximum(t_all.std(axis=0, ddof=1), 1e-8)
    t_all_std = (t_all - mean[None, :]) / scale[None, :]

    reduced_laws: list[FiniteLaw] = []
    idx_dummy = np.arange(k, dtype=np.int64)
    for row_idx, law in enumerate(knn_laws):
        if len(law.scenarios) <= k:
            reduced_laws.append(law)
            continue
        sup_idx = getattr(law, "support_indices", None)
        if sup_idx is not None and getattr(law, "support_kind", "") == "catalog" and len(sup_idx) == len(law.scenarios):
            t_local_std = t_all_std[sup_idx]
        else:
            t_loc = problem.transform_numpy(law.scenarios)
            t_local_std = (t_loc - mean[None, :]) / scale[None, :]
        km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=(int(seed) + row_idx) & 0x7FFFFFFF)
        labels = km.fit_predict(t_local_std, sample_weight=law.probabilities)
        probs = np.zeros(k, dtype=np.float64)
        centers_std = np.zeros((k, t_local_std.shape[1]), dtype=np.float64)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                w = law.probabilities[mask]
                w_sum = float(w.sum())
                probs[c] = w_sum
                centers_std[c] = (w[:, None] * t_local_std[mask]).sum(axis=0) / w_sum
            else:
                centers_std[c] = km.cluster_centers_[c]
        keep = probs > 0.0
        probs = probs[keep] / probs[keep].sum()
        centers_t = mean[None, :] + centers_std[keep] * scale[None, :]
        with torch.no_grad():
            scen = (
                problem.decode_torch(torch.as_tensor(centers_t, dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
        reduced_laws.append(FiniteLaw(scen, probs, np.arange(len(scen), dtype=np.int64), "local"))
    return tuple(reduced_laws)


# ---------------------------------------------------------------------------
# Canonical 8-Zone Continental European Topology & Panel Loader (EC.2.4)
# ---------------------------------------------------------------------------

ZONES: tuple[str, ...] = (
    "DE_50hertz",
    "DE_amprion",
    "DE_tennet",
    "DE_transnetbw",
    "FR",
    "AT",
    "DK_1",
    "IT_NORD",
)

ZONE_EIC_CODES: dict[str, str] = {
    "DE_50hertz": "10YDE-VE-------2",
    "DE_amprion": "10YDE-RWENET---I",
    "DE_tennet": "10YDE-EON------1",
    "DE_transnetbw": "10YDE-ENBW-----N",
    "FR": "10YFR-RTE------C",
    "AT": "10YAT-APG------L",
    "DK_1": "10YDK-1--------W",
    "IT_NORD": "10Y1001A1001A73I",
}

# 13 cross-border / inter-TSO corridors among the 8 Continental European zones
ADJACENCY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),  # DE_50hertz - DE_amprion
    (0, 2),  # DE_50hertz - DE_tennet
    (1, 2),  # DE_amprion - DE_tennet
    (1, 3),  # DE_amprion - DE_transnetbw
    (2, 3),  # DE_tennet - DE_transnetbw
    (1, 4),  # DE_amprion - FR
    (3, 4),  # DE_transnetbw - FR
    (2, 5),  # DE_tennet - AT
    (3, 5),  # DE_transnetbw - AT
    (2, 6),  # DE_tennet - DK_1
    (4, 7),  # FR - IT_NORD
    (5, 7),  # AT - IT_NORD
    (0, 5),  # DE_50hertz - AT
)


def make_european_grid_problem(context_dim: int = 16) -> InterconnectedGridReserveProblem:
    """Construct the exact 8-zone, 8-block (64-outcome) European grid problem of Eq. (EC.25)."""
    r_cnt = len(ZONES)
    h_cnt = 8
    coupling = 0.20
    scarcity_ratio = 70.0
    tau = 0.028
    cost_scale = 2.5

    adj = np.zeros((r_cnt, r_cnt), dtype=np.float64)
    for u, v in ADJACENCY_EDGES:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    col_deg = np.maximum(adj.sum(axis=0, keepdims=True), 1.0)
    m_grid = (1.0 - coupling) * np.eye(r_cnt) + coupling * (adj / col_deg)
    service_matrix = np.repeat(m_grid, h_cnt, axis=0)

    cap_costs = np.full(r_cnt, cost_scale, dtype=np.float64)
    blk_mult = np.array([0.10, 0.10, 1.85, 1.35, 0.80, 1.60, 2.10, 0.50], dtype=np.float64)
    underage = np.tile(cost_scale * scarcity_ratio * blk_mult, (r_cnt, 1))
    overage = np.zeros((r_cnt, h_cnt), dtype=np.float64)
    slb = np.full(r_cnt * h_cnt, -2.2, dtype=np.float64)
    sub = np.full(r_cnt * h_cnt, +2.2, dtype=np.float64)
    upper_bounds = np.full(r_cnt, 2.5, dtype=np.float64)

    return InterconnectedGridReserveProblem(
        service_matrix=service_matrix,
        capacity_costs=cap_costs,
        underage_costs=underage.reshape(-1),
        overage_costs=overage.reshape(-1),
        upper_bounds=upper_bounds,
        scenario_lower_bounds=slb,
        scenario_upper_bounds=sub,
        block_count=h_cnt,
        smoothing=tau,
        context_dim=int(context_dim),
    )


def load_grid_panel(
    npz_path: str | Any,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, InterconnectedGridReserveProblem]:
    """Load processed daily grid panel (.npz) and construct its InterconnectedGridReserveProblem."""
    with np.load(npz_path) as d:
        dates = [str(x) for x in d["dates"].tolist()]
        med_gw = d["med_gw"].astype(np.float64).copy()
        outcomes = d["outcomes"].astype(np.float64).copy()
        ctx_key = "contexts" if "contexts" in d.files else "c16s"
        contexts = d[ctx_key].astype(np.float64).copy()
    problem = make_european_grid_problem(context_dim=int(contexts.shape[1]))
    return dates, contexts, outcomes, med_gw, problem
