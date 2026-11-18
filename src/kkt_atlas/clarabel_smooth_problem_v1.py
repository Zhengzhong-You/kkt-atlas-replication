"""Exact exponential-cone reformulation of the original smooth CAP/PORT loss.

This module builds solver input; it does not implement an optimization algorithm.
"""
import clarabel
import hashlib
import json
import numpy as np
from scipy import sparse

from kkt_atlas.case_studies.smooth_capacity_allocation import SmoothCapacityAllocationProblem
from kkt_atlas.converged_smooth_problem_v1 import ConvergedSmoothProblem
from kkt_atlas.problem import probability_vector
from kkt_atlas.reference_evaluation import _reference_kkt_audit, _reference_lower_bounds


class _ClarabelFailure(RuntimeError):
    """A vendor termination or failed numerical certificate, not an input error."""


class ClarabelSmoothProblem(ConvergedSmoothProblem):
    def __init__(self,problem,*,tolerance=1e-6,cone_tolerance=1e-9,fallback_on_failure=False):
        super().__init__(problem,tolerance=tolerance)
        if not 0 < cone_tolerance <= 1e-6:
            raise ValueError("Invalid cone tolerance")
        self.cone_tolerance=cone_tolerance
        if type(fallback_on_failure) is not bool:
            raise ValueError("fallback_on_failure must be boolean")
        self.fallback_on_failure=fallback_on_failure

    @property
    def prescription(self):
        record={"schema":"kkt-prescription-v1","backend":"clarabel_exp_cone",
            "tolerance":self.tolerance,"cone_tolerance":self.cone_tolerance}
        if self.__dict__.get('fallback_on_failure',False):
            record['fallback_on_failure']=True
        return record

    def solve_with_audit(self,context,scenarios,probabilities,*,ridge):
        try:
            return self._solve_clarabel(context,scenarios,probabilities,ridge=ridge)
        except _ClarabelFailure as primary:
            inputs={'context':np.asarray(context),'scenarios':np.asarray(scenarios),
                'probabilities':np.asarray(probabilities),'ridge':np.asarray(ridge)}
            primary.solver_inputs=inputs
            if not self.__dict__.get('fallback_on_failure',False):
                raise
            try:
                decision,hessian,audit,report=super().solve_with_audit(context,scenarios,probabilities,ridge=ridge)
                report.update(primary_failure=str(primary),fallback_used=True,acceptance_kind='original_kkt')
                measures=[audit.maximum_residual]
                if not isinstance(self._problem,SmoothCapacityAllocationProblem):
                    if ridge<=0:
                        raise RuntimeError('PORT fallback gap certificate requires positive ridge')
                    p=self._problem; equality,rhs=p.reference_equality_numpy(context)
                    # Canonicalize equality roundoff; this is not an optimization iteration.
                    decision=decision+equality.T@np.linalg.solve(equality@equality.T,rhs-equality@decision)
                    raw,value,gradient,hessian=p.reference_objective_derivatives_numpy(decision,context,scenarios,probabilities,ridge=ridge)
                    lower=_reference_lower_bounds(p,context,p.decision_dim)
                    assert not np.any(np.isfinite(lower))
                    audit=_reference_kkt_audit(gradient,decision,equality,rhs,lower,np.zeros(p.decision_dim,dtype=bool))
                    projected=gradient+equality.T@np.linalg.lstsq(equality.T,-gradient,rcond=None)[0]
                    # The original PORT Hessian is >= ridge*I. On its affine
                    # feasible set, ||projected gradient||^2/(2*ridge) bounds suboptimality.
                    gap_bound=float(projected@projected)/(2*ridge)
                    relative_gap_bound=gap_bound/max(1.,abs(value))
                    measures=[audit.maximum_residual,relative_gap_bound]
                    report.update(raw_objective=raw,objective=value,maximum_kkt_residual=audit.maximum_residual,
                        objective_gap_upper_bound=gap_bound,relative_objective_gap_upper_bound=relative_gap_bound,
                        acceptance_kind='original_kkt_and_strong_convexity_gap')
                if not all(np.isfinite(v) for v in measures) or max(measures)>self.tolerance:
                    raise RuntimeError(f'SLSQP fallback certificate failed: {report}')
                report['maximum_acceptance_residual']=max(measures)
            except Exception as secondary:
                error=RuntimeError(f'Clarabel and SLSQP fallback failed: {primary}; {secondary}')
                error.solver_inputs=inputs
                raise error from secondary
            fingerprint=hashlib.sha256(b''.join(np.asarray(inputs[k],dtype=np.float64).tobytes() for k in sorted(inputs))).hexdigest()
            print(json.dumps({'event':'numerical_solver_fallback','input_sha256':fingerprint,'report':report}),flush=True)
            return decision,hessian,audit,report

    def _solve_clarabel(self,context,scenarios,probabilities,*,ridge):
        p=self._problem; center=p.reference_center_numpy(context)
        p.reference_objective_derivatives_numpy(center,context,scenarios,probabilities,ridge=ridge)
        scenarios=np.asarray(scenarios,dtype=np.float64)
        weights=probability_vector(probabilities,len(scenarios))
        equality,rhs=p.reference_equality_numpy(context)
        lower=_reference_lower_bounds(p,context,p.decision_dim)
        d=p.decision_dim; tau=p.smoothing
        quadratic=np.full(d,ridge); linear=-ridge*center
        constant=.5*ridge*float(center@center)
        if isinstance(p,SmoothCapacityAllocationProblem):
            # Keep the original two positive costs: eliminating one softplus
            # introduced large affine cancellation in the conic objective.
            service=np.tile(p.service_matrix,(len(scenarios),1))
            design=np.vstack((service,-service)); offset=np.r_[-scenarios.ravel(),scenarios.ravel()]
            coefficient=np.r_[(weights[:,None]*p.overage_costs[None,:]/p.scenario_dim).ravel(),
                (weights[:,None]*p.underage_costs[None,:]/p.scenario_dim).ravel()]
        else:
            design=np.column_stack((-scenarios,-np.ones(len(scenarios)))); offset=np.zeros(len(scenarios))
            coefficient=weights/p.tail_probability
            linear[:-1]-=p.expected_return_weight*(weights@scenarios); linear[-1]+=1.
            quadratic[:-1]+=p.leverage_penalty
        n=len(offset); size=d+3*n
        # Variable order: decision, softplus epigraph, first/second exponential slack.
        identity=sparse.eye(n,format="csc"); zero=sparse.csc_matrix((n,n)); zblock=sparse.csc_matrix((n,d))
        first=sparse.vstack((sparse.hstack((zblock,identity/tau,zero,zero)),
            sparse.csc_matrix((n,size)),sparse.hstack((zblock,zero,-identity,zero))),format="csc")
        second=sparse.vstack((sparse.hstack((-sparse.csc_matrix(design)/tau,identity/tau,zero,zero)),
            sparse.csc_matrix((n,size)),sparse.hstack((zblock,zero,zero,-identity))),format="csc")
        order=np.arange(3*n).reshape(3,n).T.ravel()
        first,second=first[order],second[order]
        finite=np.flatnonzero(np.isfinite(lower))
        bounds=sparse.hstack((-sparse.eye(d,format="csc")[finite],sparse.csc_matrix((len(finite),3*n))))
        sums=sparse.hstack((zblock,zero,identity,identity))
        eq=sparse.hstack((sparse.csc_matrix(equality),sparse.csc_matrix((len(rhs),3*n))))
        a=sparse.vstack((eq,bounds,sums,first,second),format="csc")
        b=np.r_[rhs,-lower[finite],np.ones(n),np.tile([0.,1.,0.],n),np.column_stack((offset/tau,np.ones(n),np.zeros(n))).ravel()]
        cones=[clarabel.ZeroConeT(len(rhs)),clarabel.NonnegativeConeT(len(finite)+n)]
        cones += [clarabel.ExponentialConeT() for _ in range(2*n)]
        settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_threads=1
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=self.cone_tolerance
        settings.time_limit=30.
        result=clarabel.DefaultSolver(sparse.diags(np.r_[quadratic,np.zeros(3*n)],format="csc"),
            np.r_[linear,coefficient,np.zeros(2*n)],a,b,cones,settings).solve()
        if result.status not in (clarabel.SolverStatus.Solved,clarabel.SolverStatus.AlmostSolved):
            raise _ClarabelFailure(f"Clarabel did not converge: {result.status}")
        decision=np.asarray(result.x[:d])
        raw,value,gradient,hessian=p.reference_objective_derivatives_numpy(decision,context,scenarios,weights,ridge=ridge)
        active=np.zeros(d,dtype=bool)
        bound_dual=np.asarray(result.z[len(rhs):len(rhs)+len(finite)])
        active[finite]=(decision[finite]-lower[finite]<=self.tolerance)&(bound_dual>self.tolerance)
        audit=_reference_kkt_audit(gradient,decision,equality,rhs,lower,active)
        report={"solver":"clarabel_exponential_cone","solver_status":str(result.status),"iterations":result.iterations,"runtime_s":result.solve_time,
            "raw_objective":raw,"objective":value,"conic_objective":result.obj_val+constant,
            "duality_gap":abs(result.obj_val-result.obj_val_dual),"primal_residual":result.r_prim,"dual_residual":result.r_dual,
            "maximum_kkt_residual":audit.maximum_residual,"cone_tolerance":self.cone_tolerance}
        # The conic solver certifies feasibility and objective gap. The original
        # smooth-gradient residual has different scaling and is reported separately.
        scale=max(1.,abs(value),abs(result.obj_val_dual+constant))
        feasibility=max(audit.equality_residual_linf,audit.lower_bound_violation_linf,result.r_prim,result.r_dual)
        report.update(feasibility_residual=feasibility,relative_objective_gap=report["duality_gap"]/scale,
            relative_objective_reconstruction_error=abs(value-report["conic_objective"])/scale)
        certificates=(feasibility,report["relative_objective_gap"],report["relative_objective_reconstruction_error"])
        if not all(np.isfinite(v) for v in (*certificates,audit.maximum_residual)) or max(certificates)>self.tolerance:
            raise _ClarabelFailure(f"Clarabel original-objective audit failed: {report}")
        report['maximum_acceptance_residual']=max(certificates)
        return decision,hessian,audit,report
