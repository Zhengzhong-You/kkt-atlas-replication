"""Converged CAP/PORT solves with implicit KKT derivatives, not solver unrolling.

SLSQP success alone is insufficient: independently audit the original KKT
conditions. The iteration limit is a failure limit, never a prescription.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from kkt_atlas.case_studies.smooth_capacity_allocation import SmoothCapacityAllocationProblem
from kkt_atlas.case_studies.smoothed_cvar_portfolio import SmoothedCVaRPortfolioProblem
from kkt_atlas.reference_evaluation import _reference_kkt_audit, _reference_lower_bounds


class ConvergedSmoothProblem:
    def __init__(self, problem, *, tolerance=1e-6, solver_ftol=1e-13, maximum_iterations=1000):
        if not isinstance(problem, (SmoothCapacityAllocationProblem, SmoothedCVaRPortfolioProblem)):
            raise TypeError("This adapter supports only the smooth CAP and PORT objectives")
        if not 0 < tolerance <= 1e-6 or not 0 < solver_ftol <= tolerance or maximum_iterations < 1:
            raise ValueError("Invalid convergence controls")
        self._problem = problem
        self.tolerance = tolerance
        self.solver_ftol = solver_ftol
        self.maximum_iterations = maximum_iterations

    def __getattr__(self, name):
        if name == "realized_loss_from_chart_batches_torch":
            raise AttributeError("Use the converged per-chart prescription")
        return getattr(object.__getattribute__(self, "_problem"), name)

    @property
    def prescription(self):
        if type(self) is not ConvergedSmoothProblem:
            raise ValueError("This backend must declare its own prescription metadata")
        return {"schema":"kkt-prescription-v1","backend":"scipy_slsqp",
            "tolerance":self.tolerance,"solver_ftol":self.solver_ftol,"maximum_iterations":self.maximum_iterations}

    def solve_with_audit(self, context, scenarios, probabilities, *, ridge):
        problem = self._problem
        center = problem.reference_center_numpy(context)
        equality, rhs = problem.reference_equality_numpy(context)
        lower = _reference_lower_bounds(problem, context, problem.decision_dim)

        def objective(z):
            _, value, gradient, _ = problem.reference_objective_derivatives_numpy(
                z, context, scenarios, probabilities, ridge=ridge)
            return value, gradient

        result = minimize(objective, center, method="SLSQP", jac=True,
            bounds=list(zip(lower, np.full(len(center), np.inf))),
            constraints={"type": "eq", "fun": lambda z: equality @ z - rhs,
                         "jac": lambda z: equality},
            options={"ftol": self.solver_ftol, "maxiter": self.maximum_iterations})
        raw, value, gradient, hessian = problem.reference_objective_derivatives_numpy(
            result.x, context, scenarios, probabilities, ridge=ridge)
        active = np.isfinite(lower) & (result.x - lower <= 1e-8)
        audit = _reference_kkt_audit(gradient, result.x, equality, rhs, lower, active)
        if not result.success or not np.isfinite(audit.maximum_residual) or audit.maximum_residual > self.tolerance:
            raise RuntimeError(f"SLSQP failed convergence gate: {result.message}; KKT={audit.maximum_residual:.6g}")
        report = {"solver": "scipy_slsqp", "iterations": int(result.nit),
            "raw_objective": raw, "objective": value, "maximum_kkt_residual": audit.maximum_residual,
            "active_bound_count": int(active.sum())}
        return result.x, hessian, audit, report

    def solve_numpy(self, context, scenarios, probabilities, *, ridge):
        return self.solve_with_audit(context, scenarios, probabilities, ridge=ridge)[0]

    def solve_torch(self, contexts, scenarios, probabilities, *, ridge):
        if (contexts.ndim != 2 or scenarios.ndim != 2 or probabilities.shape != (len(contexts), len(scenarios))
                or len(contexts) == 0):
            raise ValueError("Misaligned contexts, scenarios, or probabilities")
        if any(t.dtype != torch.float64 or t.device.type != "cpu" for t in (contexts, scenarios, probabilities)):
            raise ValueError("The converged development adapter requires CPU float64")
        return _ImplicitSolve.apply(contexts, scenarios, probabilities, self, ridge)

    def realized_loss_torch(self, decisions, outcomes, contexts):
        return self._problem.realized_loss_torch(decisions, outcomes, contexts)


class _ImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contexts, scenarios, probabilities, adapter, ridge):
        results = [adapter.solve_with_audit(x, scenarios.detach().numpy(), p, ridge=ridge)
                   for x, p in zip(contexts.detach().numpy(), probabilities.detach().numpy(), strict=True)]
        decisions = torch.tensor(np.stack([r[0] for r in results]), dtype=torch.float64)
        ctx.save_for_backward(contexts, scenarios, probabilities, decisions)
        ctx.adapter, ctx.ridge, ctx.results = adapter, ridge, results
        return decisions

    @staticmethod
    def backward(ctx, decision_gradient):
        contexts, scenarios, probabilities, decisions = ctx.saved_tensors
        scenario_gradient = torch.zeros_like(scenarios)
        probability_gradient = torch.zeros_like(probabilities)
        for i, (_, hessian, audit, _) in enumerate(ctx.results):
            equality_count = len(ctx.adapter.reference_equality_numpy(contexts[i].numpy())[1])
            active_multipliers = -audit.augmented_multiplier[equality_count:]
            if np.any(active_multipliers <= ctx.adapter.tolerance):
                raise RuntimeError("Weakly active bound: implicit derivative not certified")
            constraints = audit.augmented_matrix
            n, m = len(hessian), len(constraints)
            kkt = np.block([[hessian, constraints.T], [constraints, np.zeros((m, m))]])
            adjoint = np.linalg.solve(kkt, np.r_[decision_gradient[i].detach().numpy(), np.zeros(m)])[:n]
            if np.any(~np.isfinite(adjoint)):
                raise RuntimeError("Nonfinite implicit derivative")
            with torch.enable_grad():
                s = scenarios.detach().requires_grad_(True)
                p = probabilities[i].detach().requires_grad_(True)
                z = decisions[i].detach().requires_grad_(True)
                losses = ctx.adapter._problem.realized_loss_torch(
                    z[None].expand(len(s), -1), s, contexts[i:i+1].expand(len(s), -1))
                center = torch.tensor(ctx.adapter.reference_center_numpy(contexts[i].numpy()))
                objective = (p / p.sum() * losses).sum() + 0.5 * ctx.ridge * (z - center).square().sum()
                gradient, = torch.autograd.grad(objective, z, create_graph=True)
                gs, gp = torch.autograd.grad(gradient @ torch.tensor(adjoint), (s, p))
            scenario_gradient -= gs
            probability_gradient[i] = -gp
        # Context enters only through the caller's probability/scenario head in these two cases.
        return None, scenario_gradient, probability_gradient, None, None
