"""Run the existing CAP/PORT solvers and implicit derivatives by context."""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from kkt_atlas.converged_smooth_problem_v1 import ConvergedSmoothProblem, _ImplicitSolve

_SOLVER = None
_LIMIT = None


def _initialize(solver, barrier):
    global _SOLVER, _LIMIT
    _LIMIT = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    from kkt_atlas.windows_affinity_v1 import apply_affinity_from_environment
    apply_affinity_from_environment()
    _SOLVER = solver
    barrier.wait(timeout=600)


def _ready(_):
    return os.getpid()


def _forward_chunk(task):
    contexts, scenarios, probabilities, ridge = task
    return [_SOLVER.solve_with_audit(x, scenarios, p, ridge=ridge)
            for x, p in zip(contexts, probabilities, strict=True)]


def _law(task):
    x, s, p, ridge = task
    z, _, audit, report = _SOLVER.solve_with_audit(x, s, p, ridge=ridge)
    return z, report.get('maximum_acceptance_residual', audit.maximum_residual)


def _backward_chunk(task):
    contexts, scenarios, probabilities, results, gradients, ridge = task
    scenario_gradients, probability_gradients = [], []
    for x, p, result, gradient in zip(contexts, probabilities, results, gradients, strict=True):
        ctx = SimpleNamespace(adapter=_SOLVER, ridge=ridge, results=[result],
            saved_tensors=tuple(torch.tensor(a) for a in
                (x[None], scenarios, p[None], result[0][None])))
        _, gs, gp, _, _ = _ImplicitSolve.backward(ctx, torch.tensor(gradient[None]))
        scenario_gradients.append(gs.numpy())
        probability_gradients.append(gp.numpy()[0])
    return scenario_gradients, probability_gradients


class ParallelSmoothProblem(ConvergedSmoothProblem):
    def __init__(self, serial, workers=32):
        if type(workers) is not int or workers < 1:
            raise ValueError('workers must be a positive integer')
        assigned = os.environ.get('KKT_ATLAS_CPU_SET_IDS', '')
        if assigned and workers > len(assigned.split(',')):
            raise ValueError('workers exceed the assigned physical cores')
        self._serial, self._problem = serial, serial._problem
        self.tolerance, self.workers = serial.tolerance, workers
        self._pool = None

    @property
    def prescription(self):
        # Execution parallelism is recorded in the timing recipe, not the saved solver.
        return self._serial.prescription

    def solve_with_audit(self, *args, **kwargs):
        return self._serial.solve_with_audit(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _ensure_pool(self):
        if self._pool is None:
            context = multiprocessing.get_context('spawn')
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                initializer=_initialize,
                initargs=(self._serial, context.Barrier(self.workers)))
            list(self._pool.map(_ready, range(self.workers)))

    def solve_laws_with_audit(self, contexts, laws, *, ridge):
        if len(contexts) != len(laws) or not len(laws):
            raise ValueError('Misaligned contexts and laws')
        self._ensure_pool()
        results = list(self._pool.map(_law,
            [(x, law.scenarios, law.probabilities, ridge)
             for x, law in zip(contexts, laws, strict=True)]))
        return np.stack([r[0] for r in results]), np.array([r[1] for r in results])

    def solve_torch(self, contexts, scenarios, probabilities, *, ridge):
        if (contexts.ndim != 2 or scenarios.ndim != 2 or
                probabilities.shape != (len(contexts), len(scenarios)) or not len(contexts)):
            raise ValueError('Misaligned contexts, scenarios, or probabilities')
        if any(t.dtype != torch.float64 or t.device.type != 'cpu'
               for t in (contexts, scenarios, probabilities)):
            raise ValueError('Parallel CAP/PORT requires CPU float64')
        self._ensure_pool()
        return _ParallelImplicitSolve.apply(contexts, scenarios, probabilities, self, ridge)


class _ParallelImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contexts, scenarios, probabilities, adapter, ridge):
        x, s, p = (t.detach().numpy() for t in (contexts, scenarios, probabilities))
        chunks = np.array_split(np.arange(len(x)), min(adapter.workers, len(x)))
        result_chunks = list(adapter._pool.map(_forward_chunk,
            [(x[ix], s, p[ix], ridge) for ix in chunks]))
        ctx.results = [r for chunk in result_chunks for r in chunk]
        ctx.save_for_backward(contexts, scenarios, probabilities)
        ctx.adapter, ctx.ridge, ctx.chunks = adapter, ridge, chunks
        return torch.tensor(np.stack([r[0] for r in ctx.results]))

    @staticmethod
    def backward(ctx, decision_gradient):
        x, s, p = (t.detach().numpy() for t in ctx.saved_tensors)
        gradient = decision_gradient.detach().numpy()
        tasks = [(x[ix], s, p[ix], [ctx.results[i] for i in ix], gradient[ix], ctx.ridge)
                 for ix in ctx.chunks]
        results = list(ctx.adapter._pool.map(_backward_chunk, tasks))
        gs = torch.zeros_like(ctx.saved_tensors[1])
        # Keep the serial per-context accumulation order.
        for chunk, _ in results:
            for value in chunk:
                gs += torch.tensor(value)
        gp = torch.tensor(np.concatenate([r[1] for r in results]))
        return None, gs, gp, None, None
