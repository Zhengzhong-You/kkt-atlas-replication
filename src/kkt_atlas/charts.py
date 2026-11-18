"""Deterministic K-means partition of standardized training contexts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from kkt_atlas.problem import Array, matrix


class MinimumChartSizeViolation(ValueError):
    """Raised when a fitted partition contains an undersized chart."""

    def __init__(self, counts: Array, minimum_chart_size: int) -> None:
        self.counts = tuple(int(value) for value in np.asarray(counts).reshape(-1))
        self.minimum_chart_size = int(minimum_chart_size)
        super().__init__(
            "context partition violates minimum_chart_size: "
            f"counts={list(self.counts)}, minimum={self.minimum_chart_size}"
        )


@dataclass(frozen=True)
class ContextCharts:
    """Frozen context standardization, centers, and training assignments."""

    mean: Array
    scale: Array
    standardized_centers: Array
    centers: Array
    assignments: Array
    distortion: float

    @property
    def chart_count(self) -> int:
        return int(len(self.centers))

    @property
    def context_dim(self) -> int:
        return int(self.centers.shape[1])

    def standardize(self, contexts: Array) -> Array:
        values = matrix(contexts, columns=self.context_dim, name="contexts")
        return (values - self.mean[None, :]) / self.scale[None, :]

    def assign(self, contexts: Array) -> Array:
        standardized = self.standardize(contexts)
        squared = np.sum(
            (standardized[:, None, :] - self.standardized_centers[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(squared, axis=1).astype(np.int64)

    def indices(self, chart: int) -> Array:
        if chart < 0 or chart >= self.chart_count:
            raise IndexError(chart)
        return np.flatnonzero(self.assignments == chart)


def fit_context_charts(
    contexts: Array,
    chart_count: int,
    *,
    seed: int,
    minimum_scale: float = 1e-10,
    minimum_chart_size: int = 2,
) -> ContextCharts:
    """Fit and canonically label a fixed context partition."""

    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("contexts must be a nonempty finite matrix")
    if not isinstance(chart_count, int) or isinstance(chart_count, bool):
        raise TypeError("chart_count must be an integer")
    if chart_count < 1 or chart_count > len(values):
        raise ValueError("chart_count must lie between one and the sample size")
    if minimum_chart_size < 1:
        raise ValueError("minimum_chart_size must be positive")

    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
    if np.any(scale <= minimum_scale):
        bad = np.flatnonzero(scale <= minimum_scale).tolist()
        raise ValueError(f"context features have degenerate training scale: {bad}")
    standardized = (values - mean[None, :]) / scale[None, :]

    fitted = KMeans(
        n_clusters=chart_count,
        init="k-means++",
        n_init=20,
        max_iter=500,
        tol=1e-8,
        random_state=int(seed),
        algorithm="lloyd",
    ).fit(standardized)
    centers = np.asarray(fitted.cluster_centers_, dtype=np.float64)
    order = np.lexsort(tuple(centers[:, j] for j in reversed(range(centers.shape[1]))))
    centers = centers[order]
    squared = np.sum(
        (standardized[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    assignments = np.argmin(squared, axis=1).astype(np.int64)
    counts = np.bincount(assignments, minlength=chart_count)
    if np.any(counts < minimum_chart_size):
        raise MinimumChartSizeViolation(counts, minimum_chart_size)
    distortion = float(np.mean(squared[np.arange(len(values)), assignments]))
    original_centers = mean[None, :] + scale[None, :] * centers

    arrays = (mean, scale, centers, original_centers, assignments)
    immutable: list[Array] = []
    for array in arrays:
        copy = np.ascontiguousarray(array.copy())
        copy.setflags(write=False)
        immutable.append(copy)
    return ContextCharts(
        mean=immutable[0],
        scale=immutable[1],
        standardized_centers=immutable[2],
        centers=immutable[3],
        assignments=immutable[4],
        distortion=distortion,
    )


def context_partition_sample_sizes(
    contexts: Array,
    chart_count: int,
    *,
    seed: int,
    minimum_scale: float = 1e-10,
) -> tuple[int, ...]:
    """Reconstruct the sample sizes of a deterministic chart partition."""

    try:
        charts = fit_context_charts(
            contexts,
            chart_count,
            seed=seed,
            minimum_scale=minimum_scale,
            minimum_chart_size=1,
        )
    except MinimumChartSizeViolation as error:
        if error.minimum_chart_size != 1:
            raise
        return error.counts
    return tuple(
        int(value)
        for value in np.bincount(
            charts.assignments,
            minlength=chart_count,
        )
    )
