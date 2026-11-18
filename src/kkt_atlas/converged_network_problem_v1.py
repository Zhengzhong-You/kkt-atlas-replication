"""Converged NET QP with active-face implicit derivatives and recourse envelope.

Routing fractions keep the bounds independent of the learned demand scenarios.
No differentiable fixed-step routing or reservation trajectory is used.
"""
from __future__ import annotations

import numpy as np
import torch
import hashlib
import json

from kkt_atlas.case_studies.network_reservation import NetworkReservationProblem, NetworkReservationSolveResult, NetworkRecourseResult, solve_gurobi_network_qp
from kkt_atlas.problem import probability_vector


class _ActiveFaceError(RuntimeError):
    pass


class ConvergedNetworkProblem:
    def __init__(self, problem, *, tolerance=1e-6, joint_method=2, recourse_method=0, fallback_joint_method=0, solver_threads=None):
        if not isinstance(problem, NetworkReservationProblem):
            raise TypeError("Expected the NET problem")
        if not 0 < tolerance <= 1e-6:
            raise ValueError("Invalid convergence tolerance")
        self._problem, self.tolerance = problem, tolerance
        if joint_method not in (-1,0,1,2) or recourse_method not in (-1,0,1,2):
            raise ValueError("Invalid Gurobi QP method")
        self.joint_method, self.recourse_method = joint_method, recourse_method
        if fallback_joint_method not in (None,-1,0,1,2):
            raise ValueError("Invalid Gurobi fallback method")
        self.fallback_joint_method = fallback_joint_method
        if solver_threads is not None and (type(solver_threads) is not int or solver_threads < 1):
            raise ValueError("solver_threads must be a positive integer")
        self.solver_threads = solver_threads

    def __getattr__(self, name):
        if name in {"solve_formal_numpy","solve_recourse_numpy","formal_scenario_losses_numpy","sof_node_derivatives_numpy"}:
            raise AttributeError(f"Legacy optimizer path {name} is not exposed by this prescription")
        return getattr(object.__getattribute__(self, "_problem"), name)

    @property
    def prescription(self):
        return {"schema":"kkt-prescription-v1","backend":"gurobi_net","tolerance":self.tolerance,
            "joint_method":self.joint_method,"recourse_method":self.recourse_method,"last_resort_reservation_refinements":50,"exact_duplicate_aggregation":True,
            "fallback_joint_method":self.fallback_joint_method,"unresolved_face_polish_digits":80,
            **({"solver_threads":self.solver_threads} if self.solver_threads is not None else {})}

    def solve_with_routes(self, context, scenarios, probabilities, *, ridge):
        methods = [self.joint_method]
        if self.fallback_joint_method is not None and self.fallback_joint_method not in methods:
            methods.append(self.fallback_joint_method)
        reports = []
        for method in methods:
            result,routes = self._solve_with_method(context,scenarios,probabilities,ridge=ridge,method=method)
            reports.append({"method":method,"reservation":result.reservation_residual,"recourse":result.maximum_recourse_residual})
            if max(result.reservation_residual,result.maximum_recourse_residual)<=self.tolerance:
                return result,routes
        result,routes = self._refine_reservation(scenarios,probabilities,result,routes,ridge)
        reports.append({"method":"projected_refinement","reservation":result.reservation_residual,"recourse":result.maximum_recourse_residual})
        if max(result.reservation_residual,result.maximum_recourse_residual)<=self.tolerance:
            return result,routes
        error = RuntimeError(f"NET convergence gate failed after vendor methods: {reports}")
        error.solver_inputs = {"context":np.asarray(context),"scenarios":np.asarray(scenarios),
            "probabilities":np.asarray(probabilities),"decision":result.decision,"routes":routes}
        raise error


    def _refine_reservation(self, scenarios, probabilities, result, routes, ridge):
        """Native projected refinement, used only after both vendor solves fail."""
        p = self._problem
        scenarios = p._validated_demands_numpy(scenarios)
        weights = probability_vector(probabilities, len(scenarios))
        z = result.decision.copy()
        initial_z, initial_residual = z.copy(), result.reservation_residual
        inner_iterations = result.maximum_inner_iterations
        lipschitz = p.reservation_lipschitz(ridge)
        for refinement in range(51):
            excess = np.maximum(scenarios @ p.second_path_incidence.T + routes @ p.routing_difference.T - z, 0.)
            gradient = p.reservation_costs - 2*p.congestion_penalty*(weights @ excess) + ridge*(z-.1)
            projected = np.clip(z-gradient/lipschitz, 0., p.reservation_upper_bound)
            residual = lipschitz*float(np.max(np.abs(z-projected)))
            recourse_residual = _recourse_residual(p,z,scenarios,routes)
            if max(residual,recourse_residual) <= self.tolerance or refinement == 50:
                break
            z = projected
            routes, count = self._fixed_routes(np.tile(z,(len(scenarios),1)),scenarios)
            inner_iterations += count
        raw = float(p.reservation_costs@z + weights@(p.congestion_penalty*np.sum(excess**2,axis=1)
            + p.routing_penalty*np.sum((routes-scenarios/2)**2,axis=1)))
        refined = NetworkReservationSolveResult(z,raw,raw+.5*ridge*float((z-.1)@(z-.1)),
            residual,recourse_residual,result.outer_iterations+refinement,inner_iterations)
        print(json.dumps(dict(event="NET_RESERVATION_REFINEMENT",steps=refinement,
            initial_residual=initial_residual,final_residual=residual,
            maximum_decision_change=float(np.max(np.abs(z-initial_z))))),flush=True)
        return refined,routes

    def _solve_with_method(self, context, scenarios, probabilities, *, ridge, method):
        p = self._problem
        p._validate_context_numpy(context)
        scenarios = p._validated_demands_numpy(scenarios)
        weights = probability_vector(probabilities, len(scenarios))
        z, routes, iterations = solve_gurobi_network_qp(p, scenarios, weights, ridge=ridge,method=method,threads=self.solver_threads)
        feasible_z=np.clip(z,0.,p.reservation_upper_bound)
        bound_roundoff=float(np.max(np.abs(z-feasible_z)))
        if bound_roundoff>self.tolerance:
            error=RuntimeError(f"Gurobi reservation exceeds declared bound tolerance: {bound_roundoff:.9g}")
            error.solver_inputs={"context":np.asarray(context),"scenarios":scenarios,"probabilities":weights,"decision":z,"routes":routes}
            raise error
        # Canonicalize only solver-scale bound roundoff, then re-audit the original
        # KKT conditions below. Downstream recourse must receive a feasible decision.
        z=feasible_z
        recourse_residual = _recourse_residual(p,z,scenarios,routes)
        inner_iterations = 0
        if recourse_residual > self.tolerance:
            # Tiny scenario weights can leave accurate joint solutions with inaccurate
            # individual routes. Resolve the fixed-z QPs with unit row weights.
            routes, inner_iterations = self._fixed_routes(np.tile(z,(len(scenarios),1)),scenarios)
            recourse_residual = _recourse_residual(p,z,scenarios,routes)
        excess = np.maximum(scenarios @ p.second_path_incidence.T + routes @ p.routing_difference.T - z, 0.)
        gradient = p.reservation_costs - 2*p.congestion_penalty*(weights @ excess) + ridge*(z-.1)
        lipschitz = p.reservation_lipschitz(ridge)
        residual = lipschitz * float(np.max(np.abs(z-np.clip(z-gradient/lipschitz,0.,p.reservation_upper_bound))))
        raw = float(p.reservation_costs @ z + weights @ (p.congestion_penalty*(excess**2).sum(axis=1)
            + p.routing_penalty*((routes-scenarios/2)**2).sum(axis=1)))
        result = NetworkReservationSolveResult(z,raw,raw+.5*ridge*float((z-.1)@(z-.1)),
            residual,recourse_residual,iterations,inner_iterations)
        return result, routes

    def _fixed_routes(self, decisions, scenarios):
        # With reservation fixed, row QPs are independent. This bounds model
        # assembly, not the scenario law; all rows and residual checks remain.
        joined = np.column_stack((decisions, scenarios))
        unique, inverse = np.unique(joined, axis=0, return_inverse=True)
        if len(unique) < len(joined):
            decisions, scenarios = unique[:,:self.decision_dim], unique[:,self.decision_dim:]
        else:
            inverse = None
        blocks, iterations = [], 0
        for start in range(0,len(scenarios),64):
            values=scenarios[start:start+64]
            _,routes,count=solve_gurobi_network_qp(self._problem,values,np.ones(len(values))/len(values),
                ridge=0.,fixed_decisions=decisions[start:start+64],method=self.recourse_method,threads=self.solver_threads)
            blocks.append(routes)
            iterations+=count
        routes = np.concatenate(blocks)
        return (routes if inverse is None else routes[inverse]),iterations

    def kkt_contributions_numpy(self, decision, context, outcomes):
        p=self._problem
        p._validate_context_numpy(context)
        z=p._validated_decision_numpy(decision); y=p._validated_demands_numpy(outcomes)
        routes,_=self._fixed_routes(np.tile(z,(len(y),1)),y)
        if _recourse_residual(p,z,y,routes)>self.tolerance:
            error=RuntimeError("NET Gurobi basis-contribution recourse failed its residual gate")
            error.solver_inputs={"context":np.asarray(context),"decision":z,"scenarios":y,"routes":routes}
            raise error
        loads=y@p.second_path_incidence.T+routes@p.routing_difference.T
        return np.ascontiguousarray(-2*p.congestion_penalty*np.maximum(loads-z,0.))

    def solve_recourse_numpy(self, decision, scenarios, *, tolerance=1e-6):
        """Solve fixed-reservation recourse for the SOF derivative interface."""
        p = self._problem
        z = p._validated_decision_numpy(decision)
        demands = p._validated_demands_numpy(scenarios)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Invalid recourse tolerance")
        unique, inverse = np.unique(demands, axis=0, return_inverse=True)
        routes, iterations = self._fixed_routes(np.tile(z, (len(unique), 1)), unique)
        excess = np.maximum(unique @ p.second_path_incidence.T
                            + routes @ p.routing_difference.T - z, 0.)
        gradient = 2*p.congestion_penalty*(excess @ p.routing_difference)
        gradient += 2*p.routing_penalty*(routes-unique/2)
        projected = np.clip(routes-gradient/p.routing_lipschitz, 0., unique)
        residuals = p.routing_lipschitz*np.max(np.abs(routes-projected), axis=1)
        if np.max(residuals) > tolerance:
            raise RuntimeError("NET SOF recourse failed its declared residual tolerance")
        return NetworkRecourseResult(routes[inverse], residuals[inverse], iterations)

    def sof_node_derivatives_numpy(self, decision, context, outcomes, *,
                                   ridge, active_set_tolerance):
        return NetworkReservationProblem.sof_node_derivatives_numpy(
            self, decision, context, outcomes, ridge=ridge,
            active_set_tolerance=active_set_tolerance)

    def solve_numpy(self, context, scenarios, probabilities, *, ridge):
        return self.solve_with_routes(context, scenarios, probabilities, ridge=ridge)[0].decision

    def solve_torch(self, contexts, scenarios, probabilities, *, ridge):
        if (contexts.ndim != 2 or scenarios.ndim != 2 or probabilities.shape != (len(contexts), len(scenarios))
                or len(contexts) == 0):
            raise ValueError("Misaligned contexts, scenarios, or probabilities")
        if any(t.dtype != torch.float64 or t.device.type != "cpu" for t in (contexts, scenarios, probabilities)):
            raise ValueError("The converged development adapter requires CPU float64")
        if not torch.is_grad_enabled() or not any(t.requires_grad for t in (contexts,scenarios,probabilities)):
            return torch.tensor(np.stack([self.solve_numpy(x,scenarios.detach().numpy(),p,ridge=ridge)
                for x,p in zip(contexts.detach().numpy(),probabilities.detach().numpy(),strict=True)]))
        return _ImplicitNetworkSolve.apply(contexts, scenarios, probabilities, self, ridge)

    def realized_loss_torch(self, decisions, outcomes, contexts):
        if outcomes.requires_grad:
            raise ValueError("Observed outcomes must be fixed in the recourse envelope")
        # Envelope theorem: routes are converged, so differentiate the loss in z only.
        z,y = decisions.detach().numpy(),outcomes.detach().numpy()
        routes,_ = self._fixed_routes(z,y)
        if _recourse_residual(self._problem,z,y,routes)>self.tolerance:
            raise RuntimeError("NET Gurobi recourse failed its residual gate")
        routes = torch.tensor(routes, dtype=torch.float64)
        second = torch.tensor(self.second_path_incidence.copy())
        difference = torch.tensor(self.routing_difference.copy())
        loads = outcomes @ second.T + routes @ difference.T
        return (decisions @ torch.tensor(self.reservation_costs.copy())
            + self.congestion_penalty * torch.relu(loads-decisions).square().sum(dim=1)
            + self.routing_penalty * (routes-outcomes/2).square().sum(dim=1))


def _recourse_residual(problem, decisions, scenarios, routes):
    excess=np.maximum(scenarios @ problem.second_path_incidence.T + routes @ problem.routing_difference.T-decisions,0.)
    gradient=2*problem.congestion_penalty*(excess @ problem.routing_difference)+2*problem.routing_penalty*(routes-scenarios/2)
    projected=np.clip(routes-gradient/problem.routing_lipschitz,0.,scenarios)
    return problem.routing_lipschitz*float(np.max(np.abs(routes-projected)))


def _objective(problem, state, scenarios, probabilities, ridge):
    d = problem.decision_dim
    z, fractions = state[:d], state[d:].reshape(scenarios.shape)
    routes = fractions * scenarios
    load = (scenarios @ torch.tensor(problem.second_path_incidence.copy()).T
        + routes @ torch.tensor(problem.routing_difference.copy()).T)
    losses = (problem.congestion_penalty * torch.relu(load-z).square().sum(dim=1)
        + problem.routing_penalty * (routes-scenarios/2).square().sum(dim=1))
    return (z @ torch.tensor(problem.reservation_costs.copy())
        + (probabilities/probabilities.sum() * losses).sum()
        + 0.5 * ridge * (z-0.1).square().sum())


def _hessian(problem, state, scenarios, probabilities, ridge):
    d, m = problem.decision_dim, problem.scenario_dim
    z, fractions = state[:d], state[d:].reshape(scenarios.shape)
    loads = scenarios @ problem.second_path_incidence.T + (fractions*scenarios) @ problem.routing_difference.T
    active = (loads-z) > 0
    weights = probabilities/probabilities.sum()
    matrix = np.zeros((len(state), len(state)))
    matrix[:d,:d] = np.diag(2*problem.congestion_penalty*(weights @ active)+ridge)
    for k, (s,p,mask) in enumerate(zip(scenarios,weights,active,strict=True)):
        indices = slice(d+k*m,d+(k+1)*m)
        design = problem.routing_difference * s[None,:]
        cross = -2*p*problem.congestion_penalty*(mask[:,None]*design)
        matrix[:d,indices] = cross
        matrix[indices,:d] = cross.T
        matrix[indices,indices] = 2*p*(problem.congestion_penalty*(design.T @ (mask[:,None]*design))
            + problem.routing_penalty*np.diag(s*s))
    return matrix




def _block_adjoint(problem, state, scenarios, probabilities, ridge, free, decision_gradient):
    """Eliminate independent free route blocks from the NET adjoint system."""
    d, m = problem.decision_dim, problem.scenario_dim
    result = np.zeros_like(state)
    reservation = np.flatnonzero(free[:d])
    if not len(reservation):
        return result
    routes = state[d:].reshape(scenarios.shape) * scenarios
    loads = scenarios @ problem.second_path_incidence.T + routes @ problem.routing_difference.T
    active = loads - state[:d] > 0
    weights = probabilities / probabilities.sum()
    penalty = problem.congestion_penalty
    reduced = np.diag((2 * penalty * (weights @ active) + ridge)[reservation])
    blocks = []
    for k, (scenario, weight, mask) in enumerate(zip(scenarios, weights, active, strict=True)):
        route = np.flatnonzero(free[d + k*m:d + (k+1)*m])
        if not len(route):
            continue
        design = problem.routing_difference[:, route]
        route_system = penalty * (design.T @ (mask[:, None] * design))
        route_system += problem.routing_penalty * np.eye(len(route))
        coupling = penalty * mask[reservation, None] * design[reservation]
        solved = np.linalg.solve(route_system, coupling.T)
        reduced -= 2 * weight * (coupling @ solved)
        blocks.append((k, route, solved, scenario[route]))
    reduced = .5 * (reduced + reduced.T)
    reservation_adjoint = np.linalg.solve(reduced, np.asarray(decision_gradient)[reservation])
    result[reservation] = reservation_adjoint
    for k, route, solved, scale in blocks:
        result[d + k*m + route] = (solved @ reservation_adjoint) / scale
    if np.any(~np.isfinite(result)):
        raise RuntimeError("Nonfinite NET block adjoint")
    return result


def _high_precision_state(adapter,candidate,routes,s,p,ridge,reference):
    from kkt_atlas.network_face_precision_v1 import polish_network_face
    initial=np.r_[candidate.decision,(routes/s).ravel()]
    try:
        state,free,precision=polish_network_face(adapter._problem,initial,s,p,ridge)
    except RuntimeError as first_error:
        # A vendor solution can round an actually interior variable onto a bound.
        # Retry only when the original gradient points strictly into the box;
        # the high-precision routine must still certify the corrected face.
        with torch.enable_grad():
            value=torch.tensor(initial,requires_grad=True)
            gradient,=torch.autograd.grad(
                _objective(adapter._problem,value,torch.tensor(s),torch.tensor(p),ridge),value)
        d=adapter.decision_dim
        upper=np.r_[np.full(d,adapter._problem.reservation_upper_bound),np.ones(len(initial)-d)]
        weights=p/p.sum()
        scale=np.r_[np.ones(d),(weights[:,None]*s).ravel()]
        physical_gradient=gradient.detach().numpy()/scale
        low,high=initial<=1e-8,upper-initial<=1e-8
        release=(low & (physical_gradient<0)) | (high & (physical_gradient>0))
        if not np.any(release):
            raise
        retry=initial.copy()
        retry[release & low]=2e-8
        retry[release & high]=upper[release & high]-2e-8
        try:
            state,free,precision=polish_network_face(adapter._problem,retry,s,p,ridge)
        except RuntimeError as second_error:
            raise second_error from first_error
        precision['reclassified_wrong_sign_bounds']=int(release.sum())
    with torch.enable_grad():
        u=torch.tensor(state,requires_grad=True)
        objective=_objective(adapter._problem,u,torch.tensor(s),torch.tensor(p),ridge)
        gradient,=torch.autograd.grad(objective,u)
    d=adapter.decision_dim
    lipschitz=adapter._problem.reservation_lipschitz(ridge)
    projected=np.clip(state[:d]-gradient.detach().numpy()[:d]/lipschitz,0.,adapter._problem.reservation_upper_bound)
    reservation_residual=lipschitz*float(np.max(abs(state[:d]-projected)))
    recourse_residual=_recourse_residual(adapter._problem,state[:d],s,state[d:].reshape(s.shape)*s)
    decision_change=float(np.max(abs(state[:d]-reference.decision)))
    objective_change=abs(float(objective.detach())-reference.regularized_objective)/max(1.,abs(reference.regularized_objective))
    if max(reservation_residual,recourse_residual,decision_change,objective_change)>adapter.tolerance:
        raise RuntimeError("NET high-precision face failed original residual/decision/objective checks")
    print(json.dumps(dict(event="NET_HIGH_PRECISION_FACE_CHECK",reservation_residual=reservation_residual,
        recourse_residual=recourse_residual,decision_change=decision_change,**precision)),flush=True)
    return state,free


class _ImplicitNetworkSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contexts, scenarios, probabilities, adapter, ridge):
        s = scenarios.detach().numpy()
        values,faces=[],[]
        for x,p in zip(contexts.detach().numpy(),probabilities.detach().numpy(),strict=True):
            try:
                result,routes=adapter.solve_with_routes(x,s,p,ridge=ridge)
                try:
                    state,free=_certified_state(adapter._problem,result,routes,s,p,ridge)
                except _ActiveFaceError:
                    try:
                        state,free=_high_precision_state(adapter,result,routes,s,p,ridge,result)
                    except RuntimeError:
                        pass
                    else:
                        values.append(state);faces.append(free)
                        continue
                    if adapter.fallback_joint_method is None:
                        raise
                    refined,refined_routes=adapter._solve_with_method(x,s,p,ridge=ridge,method=adapter.fallback_joint_method)
                    change=float(np.max(np.abs(refined.decision-result.decision)))
                    if max(refined.reservation_residual,refined.maximum_recourse_residual,change)>adapter.tolerance:
                        raise RuntimeError("NET derivative refinement failed residual/decision-agreement gate")
                    try:
                        state,free=_certified_state(adapter._problem,refined,refined_routes,s,p,ridge)
                    except _ActiveFaceError:
                        state,free=_high_precision_state(adapter,refined,refined_routes,s,p,ridge,result)
                    print(json.dumps({"event":"NET_ACTIVE_FACE_REFINEMENT","method":adapter.fallback_joint_method,
                        "capacity_change":change,"objective_change":refined.regularized_objective-result.regularized_objective,
                        "residual":max(refined.reservation_residual,refined.maximum_recourse_residual),
                        "input_sha256":hashlib.sha256(x.tobytes()+s.tobytes()+p.tobytes()).hexdigest()}),flush=True)
                values.append(state);faces.append(free)
            except RuntimeError as error:
                error.solver_inputs={**getattr(error,"solver_inputs",{}),"context":x,"scenarios":s,"probabilities":p,"ridge":np.asarray(ridge)}
                raise
        states=torch.tensor(np.stack(values));free_masks=torch.tensor(np.stack(faces))
        ctx.save_for_backward(scenarios,probabilities,states,free_masks)
        ctx.adapter, ctx.ridge = adapter,ridge
        return states[:,:adapter.decision_dim].clone()

    @staticmethod
    def backward(ctx, decision_gradient):
        scenarios, probabilities, states, free_masks = ctx.saved_tensors
        problem = ctx.adapter._problem
        d = problem.decision_dim
        sg, pg = torch.zeros_like(scenarios), torch.zeros_like(probabilities)
        for i, state in enumerate(states):
            if torch.any(probabilities[i] <= 0):
                raise RuntimeError("NET implicit derivative requires positive scenario weights")
            with torch.enable_grad():
                s = scenarios.detach().requires_grad_(True)
                p = probabilities[i].detach().requires_grad_(True)
                u = state.detach().requires_grad_(True)
                gradient, = torch.autograd.grad(_objective(problem,u,s,p,ctx.ridge),u,create_graph=True)
                values,free=state.numpy(),free_masks[i].numpy()
                adjoint = _block_adjoint(problem, values, s.detach().numpy(),
                    p.detach().numpy(), ctx.ridge, free, decision_gradient[i].detach().numpy())
                if np.any(~np.isfinite(adjoint)):
                    raise RuntimeError("Nonfinite NET implicit derivative")
                a,b = torch.autograd.grad(gradient @ torch.tensor(adjoint), (s,p))
                sg -= a
                pg[i] = -b
        return None,sg,pg,None,None


def _certified_state(problem,result,routes,scenarios,probabilities,ridge):
    values=np.r_[result.decision,(routes/scenarios).ravel()]
    with torch.enable_grad():
        state=torch.tensor(values,requires_grad=True)
        gradient,=torch.autograd.grad(_objective(problem,state,torch.tensor(scenarios),torch.tensor(probabilities),ridge),state)
    g=gradient.detach().numpy()
    try:
        return values,_free_variables(problem,values,g,scenarios,probabilities)
    except _ActiveFaceError as error:
        error.solver_inputs={"values":values,"weighted_gradient":g}
        raise


def _free_variables(problem, values, weighted_gradient, scenarios, probabilities):
    d = problem.decision_dim
    upper = np.r_[np.full(d,problem.reservation_upper_bound),np.ones(len(values)-d)]
    low,high = values<=1e-8,upper-values<=1e-8
    weights = probabilities/probabilities.sum()
    scale = np.r_[np.ones(d),(weights[:,None]*scenarios).ravel()]
    if np.any(scale<=0) or np.any(~np.isfinite(scale)):
        raise RuntimeError("Invalid NET probability-demand derivative scale")
    # Strict-complementarity signs are invariant to positive row scaling.
    # Apply the numerical margin in physical route units, before p*xi scaling.
    physical_gradient = weighted_gradient/scale
    ambiguous = (low & (physical_gradient<=1e-8)) | (high & (physical_gradient>=-1e-8))
    # A strictly interior stationary variable can be closer than 1e-8 to a
    # bound. Proximity alone must not fix it in the sensitivity system.
    # Require much tighter stationarity here; an exactly weak bound still fails.
    stationary_interior = (values>0) & (values<upper) & (np.abs(physical_gradient)<=1e-12)
    if np.any(ambiguous & ~stationary_interior):
        raise _ActiveFaceError("Weak or inconsistent NET active bound; derivative not certified")
    return ~(low|high) | stationary_interior
