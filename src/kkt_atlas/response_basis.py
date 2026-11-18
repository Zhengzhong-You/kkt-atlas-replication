"""KKT-informed scenario coordinates for one fixed context chart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from kkt_atlas.problem import (
    Array,
    DifferentiableStochasticProblem,
    matrix,
)
from kkt_atlas.spectral_selection import (
    SpectralWidthSelection,
    select_spectral_width,
)


@dataclass(frozen=True)
class ResponseDecomposition:
    """Local SAA decision and standardized scenario-coordinate basis.

    For ``basis_method="identity"``, ``singular_values`` and
    ``cross_covariance`` are zero arrays retained only for a stable
    serialization schema; they do not represent a KKT spectrum.
    """

    context_center: Array
    local_decision: Array
    transformed_mean: Array
    transformed_scale: Array
    left_vectors: Array
    singular_values: Array
    cross_covariance: Array
    basis_method: str = "kkt"

    @property
    def maximum_width(self) -> int:
        return int(self.left_vectors.shape[1])

    def coordinates(self, outcomes: Array, width: int, problem: DifferentiableStochasticProblem) -> Array:
        transformed = problem.transform_numpy(outcomes)
        standardized = (
            transformed - self.transformed_mean[None, :]
        ) / self.transformed_scale[None, :]
        if self.basis_method == "identity":
            if width != self.maximum_width:
                raise ValueError(
                    "identity basis requires the full ambient scenario width"
                )
            return standardized
        basis = self.basis(width)
        return standardized @ basis

    def basis(self, width: int) -> Array:
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError("width must be an integer")
        if width < 1 or width > self.maximum_width:
            raise ValueError(f"width must lie in [1,{self.maximum_width}]")
        return self.left_vectors[:, :width]

    def select_width_by_spectral_ratio(
        self,
        target_ratio: float,
        *,
        minimum_width: int = 1,
        maximum_width: int | None = None,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 1e-12,
    ) -> SpectralWidthSelection:
        """Select a chart-specific width without using validation outcomes."""

        if self.basis_method == "identity":
            raise ValueError("the identity basis does not have a KKT spectrum")
        available_width = min(self.maximum_width, len(self.singular_values))
        upper_width = available_width if maximum_width is None else maximum_width
        return select_spectral_width(
            self.singular_values,
            target_ratio,
            minimum_width=minimum_width,
            maximum_width=upper_width,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )


def fit_response_decomposition(
    problem: DifferentiableStochasticProblem,
    context_center: Array,
    outcomes: Array,
    *,
    ridge: float,
    minimum_scenario_scale: float = 1e-8,
    basis_method: Literal["kkt", "pca", "random", "identity"] = "kkt",
    basis_seed: int = 0,
) -> ResponseDecomposition:
    """Solve the local SAA problem and construct the declared scenario basis."""

    context = np.asarray(context_center, dtype=np.float64).reshape(-1)
    if context.shape != (problem.context_dim,) or np.any(~np.isfinite(context)):
        raise ValueError(f"context_center must have shape ({problem.context_dim},)")
    scenarios = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    probabilities = np.full(len(scenarios), 1.0 / len(scenarios), dtype=np.float64)
    local_decision = np.asarray(
        problem.solve_numpy(context, scenarios, probabilities, ridge=ridge),
        dtype=np.float64,
    ).reshape(-1)
    if local_decision.shape != (problem.decision_dim,) or np.any(~np.isfinite(local_decision)):
        raise ValueError("local SAA solve returned an invalid decision")

    transformed = matrix(
        problem.transform_numpy(scenarios),
        columns=problem.scenario_dim,
        name="transformed outcomes",
    )
    transformed_mean = transformed.mean(axis=0)
    transformed_scale = (
        transformed.std(axis=0, ddof=1)
        if len(transformed) > 1
        else np.zeros(problem.scenario_dim)
    )
    transformed_scale = np.maximum(transformed_scale, float(minimum_scenario_scale))
    standardized = (transformed - transformed_mean[None, :]) / transformed_scale[None, :]

    if basis_method == "identity":
        left = np.eye(problem.scenario_dim, dtype=np.float64)
        singular_values = np.zeros(
            min(problem.scenario_dim, problem.decision_dim),
            dtype=np.float64,
        )
        cross_covariance = np.zeros(
            (problem.scenario_dim, problem.decision_dim),
            dtype=np.float64,
        )
    else:
        contributions = matrix(
            problem.kkt_contributions_numpy(local_decision, context, scenarios),
            columns=problem.decision_dim,
            name="KKT contributions",
        )
        if len(contributions) != len(scenarios):
            raise ValueError("KKT contributions must align with outcomes")
        centered_contributions = contributions - contributions.mean(
            axis=0,
            keepdims=True,
        )
        cross_covariance = (
            standardized.T @ centered_contributions / len(scenarios)
        )
        kkt_left, singular_values, _ = np.linalg.svd(
            cross_covariance,
            full_matrices=False,
        )
        if basis_method == "kkt":
            left = kkt_left
        elif basis_method == "pca":
            _, _, right_transpose = np.linalg.svd(
                standardized,
                full_matrices=True,
            )
            left = right_transpose.T
        elif basis_method == "random":
            generator = np.random.default_rng(int(basis_seed))
            gaussian = generator.normal(
                size=(problem.scenario_dim, problem.scenario_dim)
            )
            left, _ = np.linalg.qr(gaussian)
        else:
            raise ValueError(f"unsupported basis_method: {basis_method!r}")
    left = _canonicalize_column_signs(left)

    arrays = (
        context,
        local_decision,
        transformed_mean,
        transformed_scale,
        left,
        singular_values,
        cross_covariance,
    )
    immutable: list[Array] = []
    for array in arrays:
        copy = np.ascontiguousarray(array.copy())
        copy.setflags(write=False)
        immutable.append(copy)
    return ResponseDecomposition(
        context_center=immutable[0],
        local_decision=immutable[1],
        transformed_mean=immutable[2],
        transformed_scale=immutable[3],
        left_vectors=immutable[4],
        singular_values=immutable[5],
        cross_covariance=immutable[6],
        basis_method=basis_method,
    )


def _canonicalize_column_signs(vectors: Array) -> Array:
    result = np.asarray(vectors, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result
