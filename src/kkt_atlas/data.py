"""Immutable finite-catalog data shared by KKT Atlas and ICEO baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from kkt_atlas.problem import Array, matrix


@dataclass(frozen=True)
class ObservedSplit:
    contexts: Array
    outcome_indices: Array

    def __post_init__(self) -> None:
        contexts = np.asarray(self.contexts, dtype=np.float64)
        indices = np.asarray(self.outcome_indices, dtype=np.int64).reshape(-1)
        if contexts.ndim != 2 or len(contexts) == 0 or np.any(~np.isfinite(contexts)):
            raise ValueError("contexts must be a nonempty finite matrix")
        if indices.shape != (len(contexts),) or np.any(indices < 0):
            raise ValueError("outcome_indices must be nonnegative and align with contexts")
        contexts = np.ascontiguousarray(contexts.copy())
        indices = np.ascontiguousarray(indices.copy())
        contexts.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "outcome_indices", indices)

    def outcomes(self, catalog: Array) -> Array:
        values = np.asarray(catalog, dtype=np.float64)
        if np.any(self.outcome_indices >= len(values)):
            raise ValueError("outcome index exceeds the catalog")
        return np.ascontiguousarray(values[self.outcome_indices])


@dataclass(frozen=True)
class FiniteCatalogData:
    """One immutable input bundle; true test probabilities are evaluator-only."""

    catalog: Array
    training: ObservedSplit
    validation: ObservedSplit
    test_contexts: Array
    test_probabilities: Array
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        catalog = np.asarray(self.catalog, dtype=np.float64)
        if catalog.ndim != 2 or len(catalog) == 0 or np.any(~np.isfinite(catalog)):
            raise ValueError("catalog must be a nonempty finite matrix")
        context_dim = self.training.contexts.shape[1]
        if self.validation.contexts.shape[1] != context_dim:
            raise ValueError("training and validation context dimensions differ")
        test_contexts = matrix(
            self.test_contexts,
            columns=context_dim,
            name="test_contexts",
        )
        probabilities = np.asarray(self.test_probabilities, dtype=np.float64)
        if probabilities.shape != (len(test_contexts), len(catalog)):
            raise ValueError("test_probabilities have the wrong shape")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("test_probabilities must be finite and nonnegative")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("each test probability row must sum to one")
        if np.any(self.training.outcome_indices >= len(catalog)):
            raise ValueError("training outcome index exceeds the catalog")
        if np.any(self.validation.outcome_indices >= len(catalog)):
            raise ValueError("validation outcome index exceeds the catalog")
        catalog = np.ascontiguousarray(catalog.copy())
        test_contexts = np.ascontiguousarray(test_contexts.copy())
        probabilities = np.ascontiguousarray(probabilities.copy())
        for array in (catalog, test_contexts, probabilities):
            array.setflags(write=False)
        object.__setattr__(self, "catalog", catalog)
        object.__setattr__(self, "test_contexts", test_contexts)
        object.__setattr__(self, "test_probabilities", probabilities)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def support_size(self) -> int:
        return int(len(self.catalog))

    @property
    def scenario_dim(self) -> int:
        return int(self.catalog.shape[1])

    @property
    def context_dim(self) -> int:
        return int(self.training.contexts.shape[1])

    @property
    def data_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"kkt-atlas-finite-catalog-data-v1\0")
        for name, array in (
            ("catalog", self.catalog),
            ("training_contexts", self.training.contexts),
            ("training_indices", self.training.outcome_indices),
            ("validation_contexts", self.validation.contexts),
            ("validation_indices", self.validation.outcome_indices),
            ("test_contexts", self.test_contexts),
            ("test_probabilities", self.test_probabilities),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(contiguous.shape).encode("ascii") + b"\0")
            digest.update(contiguous.tobytes(order="C"))
        metadata_bytes = json.dumps(
            self.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(b"metadata\0" + metadata_bytes)
        return digest.hexdigest()
