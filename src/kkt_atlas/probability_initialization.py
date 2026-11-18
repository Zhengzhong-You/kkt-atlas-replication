"""Training-only initialization of a local affine-softmax probability model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from kkt_atlas.problem import Array


@dataclass(frozen=True)
class ProbabilityHeadInitialization:
    """A fitted affine-softmax initialization and its optimization record."""

    intercepts: Array
    slopes: Array
    method: str
    converged: bool
    iterations: int
    initial_objective: float
    final_objective: float
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        intercepts = np.asarray(self.intercepts, dtype=np.float64).reshape(-1)
        slopes = np.asarray(self.slopes, dtype=np.float64)
        if len(intercepts) == 0:
            raise ValueError("probability intercepts must be nonempty")
        if slopes.ndim != 2 or slopes.shape[0] != len(intercepts):
            raise ValueError("probability slopes do not match the intercepts")
        if np.any(~np.isfinite(intercepts)) or np.any(~np.isfinite(slopes)):
            raise ValueError("probability initialization must be finite")
        if not self.method:
            raise ValueError("probability initialization method must be nonempty")
        if self.iterations < 0:
            raise ValueError("probability initialization iterations must be nonnegative")
        if not math.isfinite(self.initial_objective) or not math.isfinite(self.final_objective):
            raise ValueError("probability initialization objectives must be finite")
        if abs(float(intercepts.sum())) > 1e-10:
            raise ValueError("probability intercepts must satisfy the sum-to-zero convention")
        if np.max(np.abs(slopes.sum(axis=0)), initial=0.0) > 1e-10:
            raise ValueError("probability slopes must satisfy the sum-to-zero convention")
        immutable: list[Array] = []
        for values in (intercepts, slopes):
            copy = np.ascontiguousarray(values.copy())
            copy.setflags(write=False)
            immutable.append(copy)
        object.__setattr__(self, "intercepts", immutable[0])
        object.__setattr__(self, "slopes", immutable[1])


def fit_cluster_probability_head(
    centered_standardized_contexts: Array,
    cluster_assignments: Array,
    support_size: int,
    *,
    slope_penalty: float,
    maximum_iterations: int = 500,
) -> ProbabilityHeadInitialization:
    """Fit an affine-softmax model to training-only scenario-cluster labels.

    The objective is the mean multinomial negative log likelihood plus the
    same squared-slope penalty used by the subsequent operational training.
    A deterministic L-BFGS solve starts from the cluster-frequency model.
    """

    contexts = np.asarray(centered_standardized_contexts, dtype=np.float64)
    assignments = np.asarray(cluster_assignments, dtype=np.int64).reshape(-1)
    if contexts.ndim != 2 or len(contexts) == 0:
        raise ValueError("centered_standardized_contexts must be a nonempty matrix")
    if np.any(~np.isfinite(contexts)):
        raise ValueError("centered standardized contexts must be finite")
    if assignments.shape != (len(contexts),):
        raise ValueError("cluster assignments must align with the contexts")
    if not isinstance(support_size, int) or isinstance(support_size, bool):
        raise TypeError("support_size must be an integer")
    if support_size < 1:
        raise ValueError("support_size must be positive")
    if np.any(assignments < 0) or np.any(assignments >= support_size):
        raise ValueError("cluster assignments are outside the support")
    if not math.isfinite(slope_penalty) or slope_penalty < 0.0:
        raise ValueError("slope_penalty must be finite and nonnegative")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    counts = np.bincount(assignments, minlength=support_size).astype(np.float64)
    if np.any(counts <= 0.0):
        raise ValueError("every scenario cluster must contain an observation")
    intercepts = np.log(counts / counts.sum())
    intercepts -= intercepts.mean()
    slopes = np.zeros((support_size, contexts.shape[1]), dtype=np.float64)
    if support_size == 1:
        return ProbabilityHeadInitialization(
            intercepts=np.zeros(1, dtype=np.float64),
            slopes=slopes,
            method="training_cluster_multinomial_logit",
            converged=True,
            iterations=0,
            initial_objective=0.0,
            final_objective=0.0,
        )

    augmented = np.column_stack((np.ones(len(contexts), dtype=np.float64), contexts))
    initial_parameters = np.column_stack((intercepts, slopes))

    def objective_and_gradient(flat_parameters: Array) -> tuple[float, Array]:
        parameters = np.asarray(flat_parameters, dtype=np.float64).reshape(
            support_size,
            augmented.shape[1],
        )
        centered = parameters - parameters.mean(axis=0, keepdims=True)
        logits = augmented @ centered.T
        log_normalizers = logsumexp(logits, axis=1)
        value = float(
            np.mean(log_normalizers - logits[np.arange(len(assignments)), assignments])
            + slope_penalty * np.sum(centered[:, 1:] ** 2)
        )
        probabilities = np.exp(logits - log_normalizers[:, None])
        probabilities[np.arange(len(assignments)), assignments] -= 1.0
        gradient = probabilities.T @ augmented / len(assignments)
        gradient[:, 1:] += 2.0 * slope_penalty * centered[:, 1:]
        gradient -= gradient.mean(axis=0, keepdims=True)
        return value, np.asarray(gradient, dtype=np.float64).reshape(-1)

    initial_objective, _ = objective_and_gradient(initial_parameters.reshape(-1))
    result = minimize(
        objective_and_gradient,
        initial_parameters.reshape(-1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maximum_iterations),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    fitted = np.asarray(result.x, dtype=np.float64).reshape(
        support_size,
        augmented.shape[1],
    )
    fitted -= fitted.mean(axis=0, keepdims=True)
    final_objective, _ = objective_and_gradient(fitted.reshape(-1))
    fallback_reason: str | None = None
    if (
        not bool(result.success)
        or np.any(~np.isfinite(fitted))
        or not math.isfinite(final_objective)
        or final_objective > initial_objective + 1e-10
    ):
        fallback_reason = f"L-BFGS status {int(result.status)}: {str(result.message)}"
        fitted = initial_parameters
        final_objective = initial_objective

    return ProbabilityHeadInitialization(
        intercepts=fitted[:, 0],
        slopes=fitted[:, 1:],
        method="training_cluster_multinomial_logit",
        converged=fallback_reason is None,
        iterations=int(result.nit),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        fallback_reason=fallback_reason,
    )
