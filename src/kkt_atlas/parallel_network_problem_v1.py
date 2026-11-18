"""Converged NET batch solves within one existing CPU slot."""
import multiprocessing
import os
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from kkt_atlas.converged_network_problem_v1 import ConvergedNetworkProblem, _ImplicitNetworkSolve

_WORKER_SOLVER = None
_THREAD_LIMIT = None


def _initialize(problem, controls, barrier):
    global _WORKER_SOLVER, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    if os.environ.get("KKT_ATLAS_CPU_SET_IDS"):
        from kkt_atlas.windows_affinity_v1 import apply_affinity_from_environment
        apply_affinity_from_environment()
    _WORKER_SOLVER = ConvergedNetworkProblem(problem, solver_threads=1, **controls)
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", 0)
            environment.start()
    except Exception:
        pass
    barrier.wait(timeout=max(60,5*barrier.parties))


def _ready(unused):
    return os.getpid()


def _solve_chunk(task):
    contexts, scenarios, probabilities, ridge, derivatives = task
    with torch.set_grad_enabled(derivatives):
        x = torch.tensor(contexts)
        s = torch.tensor(scenarios, requires_grad=derivatives)
        p = torch.tensor(probabilities, requires_grad=derivatives)
        z = _WORKER_SOLVER.solve_torch(x,s,p,ridge=ridge)
        if derivatives:
            _, _, states, faces = z.grad_fn.saved_tensors
            return states.numpy(), faces.numpy()
        return z.numpy(), None



def _audit_law(task):
    context,scenarios,probabilities,ridge = task
    result,_ = _WORKER_SOLVER.solve_with_routes(context,scenarios,probabilities,ridge=ridge)
    return result.decision,max(result.reservation_residual,result.maximum_recourse_residual)


def _backward_chunk(task):
    scenarios, probabilities, states, faces, gradients, ridge = task
    sg, pg = [], []
    for i in range(len(states)):
        ctx = SimpleNamespace(adapter=_WORKER_SOLVER, ridge=ridge,
            saved_tensors=tuple(torch.tensor(a) for a in
                (scenarios, probabilities[i:i+1], states[i:i+1], faces[i:i+1])))
        result = _ImplicitNetworkSolve.backward(ctx, torch.tensor(gradients[i:i+1]))
        sg.append(result[1].numpy()); pg.append(result[2].numpy()[0])
    return np.stack(sg), np.stack(pg)


class ParallelNetworkProblem(ConvergedNetworkProblem):
    """Use in a context manager; every child inherits the parent's CPU slot."""
    def __init__(self, problem, *, workers=8, parallel_derivatives=False, **controls):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        assigned = [x for x in os.environ.get("KKT_ATLAS_CPU_SET_IDS", "").split(",") if x.strip()]
        if assigned and workers > len(assigned):
            raise ValueError("Context workers exceed the assigned CPU slot")
        super().__init__(problem, **controls)
        if type(parallel_derivatives) is not bool:
            raise ValueError("parallel_derivatives must be boolean")
        self.parallel_derivatives = parallel_derivatives
        self.workers = workers
        self._pool = None
        self._controls = dict(tolerance=self.tolerance, joint_method=self.joint_method,
            recourse_method=self.recourse_method, fallback_joint_method=self.fallback_joint_method)

    @property
    def prescription(self):
        return {**super().prescription, "context_workers":self.workers,
                **({"parallel_derivatives":True} if self.parallel_derivatives else {}),
                "worker_solver_threads":1, "parallelism":"independent_context_chunks_with_exact_probability_reuse"}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _ensure_pool(self):
        if self._pool is None:
            context = multiprocessing.get_context("spawn")
            barrier = context.Barrier(self.workers)
            self._pool = ProcessPoolExecutor(max_workers=self.workers,mp_context=context,
                initializer=_initialize, initargs=(self._problem,self._controls,barrier))
            # Start every allocated worker before measuring or dispatching a batch.
            list(self._pool.map(_ready,range(self.workers)))
    def solve_laws_with_audit(self, contexts, laws, *, ridge):
        self._problem._validate_contexts_torch(torch.tensor(contexts,dtype=torch.float64))
        if len(contexts)!=len(laws) or not len(laws):
            raise ValueError("Misaligned contexts and laws")
        self._ensure_pool()
        tasks=[(x,law.scenarios,law.probabilities,ridge) for x,law in zip(contexts,laws,strict=True)]
        results=list(self._pool.map(_audit_law,tasks))
        return np.stack([r[0] for r in results]),np.asarray([r[1] for r in results])

    def _solve_batch(self, contexts, scenarios, probabilities, ridge, derivatives):
        self._ensure_pool()
        x, s, p = (a.detach().numpy() for a in (contexts,scenarios,probabilities))
        unique, first, inverse = np.unique(p,axis=0,return_index=True,return_inverse=True)
        if len(unique) < len(p):
            x, p = x[first], unique
        else:
            inverse = None
        indices = np.array_split(np.arange(len(x)), min(self.workers,len(x)))
        tasks = [(x[index],s,p[index],ridge,derivatives) for index in indices]
        result = list(self._pool.map(_solve_chunk,tasks))
        values = torch.tensor(np.concatenate([item[0] for item in result]))
        faces = torch.tensor(np.concatenate([item[1] for item in result])) if derivatives else None
        if inverse is not None:
            values = values[inverse]
            if faces is not None: faces = faces[inverse]
        return values, faces

    def solve_torch(self, contexts, scenarios, probabilities, *, ridge):
        if (contexts.ndim!=2 or scenarios.ndim!=2 or probabilities.shape!=(len(contexts),len(scenarios))
                or not len(contexts)):
            raise ValueError("Misaligned contexts, scenarios, or probabilities")
        if any(t.dtype!=torch.float64 or t.device.type!="cpu" for t in (contexts,scenarios,probabilities)):
            raise ValueError("Parallel NET requires CPU float64")
        self._problem._validate_contexts_torch(contexts)
        derivatives = torch.is_grad_enabled() and any(t.requires_grad for t in (contexts,scenarios,probabilities))
        if derivatives:
            return _ParallelImplicitSolve.apply(contexts,scenarios,probabilities,self,ridge)
        return self._solve_batch(contexts,scenarios,probabilities,ridge,False)[0]


class _ParallelImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contexts, scenarios, probabilities, adapter, ridge):
        states, faces = adapter._solve_batch(contexts,scenarios,probabilities,ridge,True)
        ctx.save_for_backward(scenarios,probabilities,states,faces)
        ctx.adapter, ctx.ridge = adapter,ridge
        return states[:,:adapter.decision_dim].clone()

    @staticmethod
    def backward(ctx, decision_gradient):
        if not ctx.adapter.parallel_derivatives:
            return _ImplicitNetworkSolve.backward(ctx, decision_gradient)
        scenarios, probabilities, states, faces = ctx.saved_tensors
        indices = np.array_split(np.arange(len(states)), min(ctx.adapter.workers,len(states)))
        tasks = [(scenarios.detach().numpy(), probabilities[index].detach().numpy(), states[index].detach().numpy(),
                  faces[index].numpy(), decision_gradient[index].detach().numpy(), ctx.ridge)
                 for index in indices]
        results = list(ctx.adapter._pool.map(_backward_chunk,tasks))
        sg = torch.zeros_like(scenarios)
        # Preserve the original per-context scenario-gradient accumulation order.
        for values, unused in results:
            for value in values:
                sg += torch.tensor(value)
        pg = torch.tensor(np.concatenate([value[1] for value in results]))
        return None,sg,pg,None,None
