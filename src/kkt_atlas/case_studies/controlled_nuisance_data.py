"""Paired contextual data with high-variance, decision-invariant nuisance factors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from kkt_atlas.case_studies.projected_quadratic import (
    ProjectedQuadraticProblem,
)
from kkt_atlas.data import FiniteCatalogData, ObservedSplit
from kkt_atlas.problem import Array, matrix

_DGP_VERSION = "controlled-contextual-nuisance-dgp-v1"


@dataclass(frozen=True)
class ControlledNuisanceDataConfig:
    """Frozen controls for the contextual q-attribution experiment."""

    seed: int
    scenario_dim: int = 200
    task_rank: int = 4
    task_microstates_per_group: int = 10
    nuisance_rank: int = 8
    nuisance_state_count: int = 10
    training_size: int = 4000
    validation_size: int = 1000
    test_size: int = 1000
    context_dim: int = 2
    context_scale: float = 1.0
    context_center_separation: float = 5.0
    task_vertex_scale: float = 1.5
    task_microstate_scale: float = 0.1
    probability_slope_scale: float = 1.4
    low_noise_std: float = 0.05
    high_noise_std: float = 6.0

    def __post_init__(self) -> None:
        if self.scenario_dim < self.task_rank + self.nuisance_rank:
            raise ValueError(
                "scenario_dim must contain the task and nuisance subspaces"
            )
        if self.task_rank < 1 or self.nuisance_rank < 1:
            raise ValueError("task_rank and nuisance_rank must be positive")
        if self.task_microstates_per_group < 1:
            raise ValueError("task_microstates_per_group must be positive")
        if self.nuisance_state_count <= self.nuisance_rank:
            raise ValueError(
                "nuisance_state_count must exceed nuisance_rank after centering"
            )
        if min(self.training_size, self.validation_size, self.test_size) < 1:
            raise ValueError("all sample sizes must be positive")
        if self.context_dim < 2:
            raise ValueError("context_dim must be at least two for four regions")
        positive_values = (
            self.context_scale,
            self.context_center_separation,
            self.task_vertex_scale,
            self.probability_slope_scale,
            self.high_noise_std,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError("positive DGP scales must be positive and finite")
        if (
            not np.isfinite(self.task_microstate_scale)
            or self.task_microstate_scale < 0.0
        ):
            raise ValueError("task_microstate_scale must be finite and nonnegative")
        if (
            not np.isfinite(self.low_noise_std)
            or self.low_noise_std < 0.0
            or self.low_noise_std >= self.high_noise_std
        ):
            raise ValueError(
                "noise levels must satisfy 0 <= low_noise_std < high_noise_std"
            )

    @property
    def task_group_count(self) -> int:
        return self.task_rank + 1

    @property
    def task_state_count(self) -> int:
        return self.task_group_count * self.task_microstates_per_group

    @property
    def support_size(self) -> int:
        return self.task_state_count * self.nuisance_state_count

    @property
    def context_region_count(self) -> int:
        return 4


@dataclass(frozen=True)
class ControlledNuisanceGeometry:
    """Shared geometry used by both nuisance levels."""

    task_basis: Array
    nuisance_basis: Array
    task_vertices: Array
    task_states: Array
    nuisance_states: Array
    context_centers: Array
    probability_slopes: Array

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (
                self.task_basis,
                self.nuisance_basis,
                self.task_vertices,
                self.task_states,
                self.nuisance_states,
                self.context_centers,
                self.probability_slopes,
            )
        ]
        immutable: list[Array] = []
        for array in arrays:
            if array.ndim != 2 or len(array) == 0 or np.any(~np.isfinite(array)):
                raise ValueError("all geometry components must be finite matrices")
            copy = np.ascontiguousarray(array.copy())
            copy.setflags(write=False)
            immutable.append(copy)
        (
            task_basis,
            nuisance_basis,
            task_vertices,
            task_states,
            nuisance_states,
            context_centers,
            probability_slopes,
        ) = immutable
        if task_basis.shape[0] != nuisance_basis.shape[0]:
            raise ValueError("task and nuisance bases must share the ambient dimension")
        if task_vertices.shape[1] != task_basis.shape[1]:
            raise ValueError("task vertices have the wrong dimension")
        if task_states.shape[1] != task_basis.shape[1]:
            raise ValueError("task states have the wrong dimension")
        if nuisance_states.shape[1] != nuisance_basis.shape[1]:
            raise ValueError("nuisance states have the wrong dimension")
        if probability_slopes.shape != (
            len(task_vertices),
            context_centers.shape[1],
        ):
            raise ValueError("probability slopes have the wrong shape")
        object.__setattr__(self, "task_basis", task_basis)
        object.__setattr__(self, "nuisance_basis", nuisance_basis)
        object.__setattr__(self, "task_vertices", task_vertices)
        object.__setattr__(self, "task_states", task_states)
        object.__setattr__(self, "nuisance_states", nuisance_states)
        object.__setattr__(self, "context_centers", context_centers)
        object.__setattr__(self, "probability_slopes", probability_slopes)


@dataclass(frozen=True)
class ControlledNuisancePair:
    """Low- and high-nuisance data with identical shared random quantities."""

    problem: ProjectedQuadraticProblem
    low_noise: FiniteCatalogData
    high_noise: FiniteCatalogData
    geometry: ControlledNuisanceGeometry
    diagnostics: dict[str, Any]


def generate_controlled_nuisance_pair(
    config: ControlledNuisanceDataConfig,
) -> ControlledNuisancePair:
    """Generate the paired finite-catalog inputs for one seed."""

    geometry = _make_geometry(config)
    problem = ProjectedQuadraticProblem(
        task_basis=geometry.task_basis,
        context_dim=config.context_dim,
    )
    training_contexts = _sample_contexts(
        config,
        geometry.context_centers,
        "training",
        config.training_size,
    )
    validation_contexts = _sample_contexts(
        config,
        geometry.context_centers,
        "validation",
        config.validation_size,
    )
    test_contexts = _sample_contexts(
        config,
        geometry.context_centers,
        "test",
        config.test_size,
    )
    training_group_probabilities = conditional_group_probabilities(
        training_contexts,
        geometry,
    )
    validation_group_probabilities = conditional_group_probabilities(
        validation_contexts,
        geometry,
    )
    test_group_probabilities = conditional_group_probabilities(
        test_contexts,
        geometry,
    )
    training_indices = _sample_catalog_indices(
        training_group_probabilities,
        config,
        _rng(config.seed, "training-outcomes"),
    )
    validation_indices = _sample_catalog_indices(
        validation_group_probabilities,
        config,
        _rng(config.seed, "validation-outcomes"),
    )
    training = ObservedSplit(training_contexts, training_indices)
    validation = ObservedSplit(validation_contexts, validation_indices)
    test_probabilities = _catalog_probabilities(
        test_group_probabilities,
        config,
    )
    paired_design_hash = _shared_design_hash(
        config,
        geometry,
        training,
        validation,
        test_contexts,
        test_probabilities,
    )
    low_noise = _make_data(
        config,
        geometry,
        training,
        validation,
        test_contexts,
        test_probabilities,
        noise_std=config.low_noise_std,
        paired_design_hash=paired_design_hash,
    )
    high_noise = _make_data(
        config,
        geometry,
        training,
        validation,
        test_contexts,
        test_probabilities,
        noise_std=config.high_noise_std,
        paired_design_hash=paired_design_hash,
    )
    diagnostics = paired_nuisance_diagnostics(
        low_noise,
        high_noise,
        geometry,
    )
    if not diagnostics["exact_shared_random_quantities"]:
        raise AssertionError("paired nuisance construction lost exact pairing")
    return ControlledNuisancePair(
        problem=problem,
        low_noise=low_noise,
        high_noise=high_noise,
        geometry=geometry,
        diagnostics=diagnostics,
    )


def make_controlled_nuisance_problem(
    config: ControlledNuisanceDataConfig,
) -> ProjectedQuadraticProblem:
    """Reconstruct the stochastic problem deterministically from the DGP seed."""

    geometry = _make_geometry(config)
    return ProjectedQuadraticProblem(
        task_basis=geometry.task_basis,
        context_dim=config.context_dim,
    )


def conditional_group_probabilities(
    contexts: Array,
    geometry: ControlledNuisanceGeometry,
) -> Array:
    """Return the chart-local affine-softmax task-group probabilities."""

    values = matrix(
        contexts,
        columns=geometry.context_centers.shape[1],
        name="contexts",
    )
    squared_distances = np.sum(
        (
            values[:, None, :]
            - geometry.context_centers[None, :, :]
        )
        ** 2,
        axis=2,
    )
    regions = np.argmin(squared_distances, axis=1)
    local_contexts = values - geometry.context_centers[regions]
    logits = local_contexts @ geometry.probability_slopes.T
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(probabilities)


def paired_nuisance_diagnostics(
    low_noise: FiniteCatalogData,
    high_noise: FiniteCatalogData,
    geometry: ControlledNuisanceGeometry,
) -> dict[str, Any]:
    """Check exact pairing and decision invariance across nuisance levels."""

    exact_contexts = (
        np.array_equal(
            low_noise.training.contexts,
            high_noise.training.contexts,
        )
        and np.array_equal(
            low_noise.validation.contexts,
            high_noise.validation.contexts,
        )
        and np.array_equal(low_noise.test_contexts, high_noise.test_contexts)
    )
    exact_indices = np.array_equal(
        low_noise.training.outcome_indices,
        high_noise.training.outcome_indices,
    ) and np.array_equal(
        low_noise.validation.outcome_indices,
        high_noise.validation.outcome_indices,
    )
    exact_probabilities = np.array_equal(
        low_noise.test_probabilities,
        high_noise.test_probabilities,
    )
    task_projection_difference = (
        low_noise.catalog @ geometry.task_basis
        - high_noise.catalog @ geometry.task_basis
    )
    nuisance_task_inner_product = (
        geometry.nuisance_basis.T @ geometry.task_basis
    )
    return {
        "exact_contexts": bool(exact_contexts),
        "exact_outcome_indices": bool(exact_indices),
        "exact_test_probabilities": bool(exact_probabilities),
        "exact_shared_random_quantities": bool(
            exact_contexts and exact_indices and exact_probabilities
        ),
        "max_abs_task_projection_difference": float(
            np.max(np.abs(task_projection_difference))
        ),
        "max_abs_task_nuisance_inner_product": float(
            np.max(np.abs(nuisance_task_inner_product))
        ),
        "support_size": int(low_noise.support_size),
        "scenario_dim": int(low_noise.scenario_dim),
    }


def _make_geometry(
    config: ControlledNuisanceDataConfig,
) -> ControlledNuisanceGeometry:
    column_count = config.task_rank + config.nuisance_rank
    raw_geometry = _rng(config.seed, "ambient-geometry").normal(
        size=(config.scenario_dim, column_count)
    )
    orthogonal, _ = np.linalg.qr(raw_geometry, mode="reduced")
    orthogonal = _canonicalize_columns(orthogonal)
    task_basis = orthogonal[:, : config.task_rank]
    nuisance_basis = orthogonal[:, config.task_rank :]

    task_vertices = (
        config.task_vertex_scale * _regular_simplex(config.task_rank)
    )
    task_states: list[Array] = []
    for group, vertex in enumerate(task_vertices):
        perturbations = _rng(
            config.seed,
            f"task-microstates-{group}",
        ).normal(
            size=(
                config.task_microstates_per_group,
                config.task_rank,
            )
        )
        perturbations -= perturbations.mean(axis=0, keepdims=True)
        root_mean_square = float(np.sqrt(np.mean(perturbations**2)))
        if (
            config.task_microstate_scale > 0.0
            and root_mean_square > 1e-14
        ):
            perturbations *= (
                config.task_microstate_scale / root_mean_square
            )
        else:
            perturbations.fill(0.0)
        task_states.append(vertex[None, :] + perturbations)

    raw_nuisance = _rng(config.seed, "nuisance-states").normal(
        size=(config.nuisance_state_count, config.nuisance_rank)
    )
    raw_nuisance -= raw_nuisance.mean(axis=0, keepdims=True)
    covariance = raw_nuisance.T @ raw_nuisance / config.nuisance_state_count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(np.min(eigenvalues)) <= 1e-12:
        raise RuntimeError("nuisance-state construction lost rank")
    inverse_root = (
        eigenvectors
        @ np.diag(eigenvalues**-0.5)
        @ eigenvectors.T
    )
    nuisance_states = raw_nuisance @ inverse_root

    context_centers = np.zeros(
        (config.context_region_count, config.context_dim),
        dtype=np.float64,
    )
    context_centers[:, :2] = config.context_center_separation * np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    angles = (
        2.0
        * np.pi
        * np.arange(config.task_group_count)
        / config.task_group_count
    )
    probability_slopes = np.zeros(
        (config.task_group_count, config.context_dim),
        dtype=np.float64,
    )
    probability_slopes[:, 0] = (
        config.probability_slope_scale * np.cos(angles)
    )
    probability_slopes[:, 1] = (
        config.probability_slope_scale * np.sin(angles)
    )
    return ControlledNuisanceGeometry(
        task_basis=task_basis,
        nuisance_basis=nuisance_basis,
        task_vertices=task_vertices,
        task_states=np.vstack(task_states),
        nuisance_states=nuisance_states,
        context_centers=context_centers,
        probability_slopes=probability_slopes,
    )


def _sample_contexts(
    config: ControlledNuisanceDataConfig,
    context_centers: Array,
    split: str,
    sample_size: int,
) -> Array:
    rng = _rng(config.seed, f"{split}-contexts")
    regions = rng.integers(
        0,
        config.context_region_count,
        size=sample_size,
    )
    local_contexts = rng.normal(
        scale=config.context_scale,
        size=(sample_size, config.context_dim),
    )
    return np.asarray(
        context_centers[regions] + local_contexts,
        dtype=np.float64,
    )


def _sample_catalog_indices(
    group_probabilities: Array,
    config: ControlledNuisanceDataConfig,
    rng: np.random.Generator,
) -> Array:
    uniforms = rng.random(len(group_probabilities))
    cumulative = np.cumsum(group_probabilities, axis=1)
    groups = np.sum(
        uniforms[:, None] > cumulative,
        axis=1,
    ).astype(np.int64)
    microstates = rng.integers(
        config.task_microstates_per_group,
        size=len(groups),
    )
    nuisance_states = rng.integers(
        config.nuisance_state_count,
        size=len(groups),
    )
    task_states = (
        groups * config.task_microstates_per_group + microstates
    )
    return (
        task_states * config.nuisance_state_count + nuisance_states
    ).astype(np.int64)


def _catalog_probabilities(
    group_probabilities: Array,
    config: ControlledNuisanceDataConfig,
) -> Array:
    per_scenario = (
        group_probabilities[:, :, None, None]
        / (
            config.task_microstates_per_group
            * config.nuisance_state_count
        )
    )
    expanded = np.broadcast_to(
        per_scenario,
        (
            len(group_probabilities),
            config.task_group_count,
            config.task_microstates_per_group,
            config.nuisance_state_count,
        ),
    )
    return np.ascontiguousarray(
        expanded.reshape(len(group_probabilities), config.support_size)
    )


def _make_data(
    config: ControlledNuisanceDataConfig,
    geometry: ControlledNuisanceGeometry,
    training: ObservedSplit,
    validation: ObservedSplit,
    test_contexts: Array,
    test_probabilities: Array,
    *,
    noise_std: float,
    paired_design_hash: str,
) -> FiniteCatalogData:
    task_component = geometry.task_states @ geometry.task_basis.T
    nuisance_component = (
        geometry.nuisance_states @ geometry.nuisance_basis.T
    )
    catalog = (
        task_component[:, None, :]
        + float(noise_std) * nuisance_component[None, :, :]
    ).reshape(config.support_size, config.scenario_dim)
    return FiniteCatalogData(
        catalog=catalog,
        training=training,
        validation=validation,
        test_contexts=test_contexts,
        test_probabilities=test_probabilities,
        metadata={
            "schema": _DGP_VERSION,
            "seed": int(config.seed),
            "paired_design_hash": paired_design_hash,
            "support_size": config.support_size,
            "scenario_dim": config.scenario_dim,
            "task_rank": config.task_rank,
            "task_group_count": config.task_group_count,
            "task_microstates_per_group": (
                config.task_microstates_per_group
            ),
            "nuisance_rank": config.nuisance_rank,
            "nuisance_state_count": config.nuisance_state_count,
            "noise_std": float(noise_std),
            "context_dim": config.context_dim,
            "context_region_count": config.context_region_count,
            "training_size": config.training_size,
            "validation_size": config.validation_size,
            "test_size": config.test_size,
            "support_order": "task-group--microstate--nuisance",
        },
    )


def _shared_design_hash(
    config: ControlledNuisanceDataConfig,
    geometry: ControlledNuisanceGeometry,
    training: ObservedSplit,
    validation: ObservedSplit,
    test_contexts: Array,
    test_probabilities: Array,
) -> str:
    digest = hashlib.sha256()
    digest.update(_DGP_VERSION.encode("ascii") + b"\0")
    digest.update(str(int(config.seed)).encode("ascii") + b"\0")
    for name, array in (
        ("task_basis", geometry.task_basis),
        ("nuisance_basis", geometry.nuisance_basis),
        ("task_states", geometry.task_states),
        ("nuisance_states", geometry.nuisance_states),
        ("context_centers", geometry.context_centers),
        ("probability_slopes", geometry.probability_slopes),
        ("training_contexts", training.contexts),
        ("training_indices", training.outcome_indices),
        ("validation_contexts", validation.contexts),
        ("validation_indices", validation.outcome_indices),
        ("test_contexts", test_contexts),
        ("test_probabilities", test_probabilities),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(str(contiguous.shape).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _regular_simplex(dimension: int) -> Array:
    count = dimension + 1
    helmert = np.zeros((dimension, count), dtype=np.float64)
    for row in range(1, count):
        scale = np.sqrt(row * (row + 1.0))
        helmert[row - 1, :row] = 1.0 / scale
        helmert[row - 1, row] = -row / scale
    return np.sqrt(count) * helmert.T


def _canonicalize_columns(values: Array) -> Array:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _rng(seed: int, token: str) -> np.random.Generator:
    payload = f"{_DGP_VERSION}|{int(seed)}|{token}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)
