"""Deterministic initialization of a local artificial support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from kkt_atlas.problem import Array


@dataclass(frozen=True)
class ScenarioInitialization:
    coordinates: Array
    counts: Array

    @property
    def support_size(self) -> int:
        return int(len(self.coordinates))


@dataclass(frozen=True)
class ScenarioClustering:
    """A canonical projected-scenario clustering and its assignments."""

    initialization: ScenarioInitialization
    assignments: Array

    def __post_init__(self) -> None:
        labels = np.asarray(self.assignments, dtype=np.int64).reshape(-1)
        if len(labels) == 0:
            raise ValueError("scenario-cluster assignments must be nonempty")
        if np.any(labels < 0) or np.any(labels >= self.initialization.support_size):
            raise ValueError("scenario-cluster assignments are outside the support")
        counts = np.bincount(
            labels,
            minlength=self.initialization.support_size,
        ).astype(np.int64)
        if not np.array_equal(counts, self.initialization.counts):
            raise ValueError("scenario-cluster assignments do not match the counts")
        immutable = np.ascontiguousarray(labels.copy())
        immutable.setflags(write=False)
        object.__setattr__(self, "assignments", immutable)


def cluster_projected_scenarios(
    projected_coordinates: Array,
    support_size: int,
    *,
    seed: int,
) -> ScenarioClustering:
    """Cluster projected realizations and retain the canonical assignments."""

    values = np.asarray(projected_coordinates, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("projected_coordinates must be a nonempty finite matrix")
    if not isinstance(support_size, int) or isinstance(support_size, bool):
        raise TypeError("support_size must be an integer")
    distinct = np.unique(values, axis=0)
    if support_size < 1 or support_size > len(distinct):
        raise ValueError(f"support_size must lie in [1,{len(distinct)}] for the projected observations")
    fitted = KMeans(
        n_clusters=support_size,
        init="k-means++",
        n_init=20,
        max_iter=500,
        tol=1e-8,
        random_state=int(seed),
        algorithm="lloyd",
    ).fit(values)
    centers = np.asarray(fitted.cluster_centers_, dtype=np.float64)
    order = np.lexsort(tuple(centers[:, j] for j in reversed(range(centers.shape[1]))))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(support_size)
    labels = inverse[np.asarray(fitted.labels_, dtype=np.int64)]
    centers = centers[order]
    counts = np.bincount(labels, minlength=support_size).astype(np.int64)
    if np.any(counts <= 0):
        raise RuntimeError("K-means returned an empty scenario cluster")
    centers.setflags(write=False)
    counts.setflags(write=False)
    return ScenarioClustering(
        initialization=ScenarioInitialization(
            coordinates=centers,
            counts=counts,
        ),
        assignments=labels,
    )


def initialize_scenarios(
    projected_coordinates: Array,
    support_size: int,
    *,
    seed: int,
) -> ScenarioInitialization:
    """Return the canonical scenario initialization for backward compatibility."""

    return cluster_projected_scenarios(
        projected_coordinates,
        support_size,
        seed=seed,
    ).initialization
