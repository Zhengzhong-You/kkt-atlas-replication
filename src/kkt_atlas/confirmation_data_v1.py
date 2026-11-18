"""Disjoint fitting and evaluation data objects for confirmation studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from kkt_atlas.data import FiniteCatalogData, ObservedSplit
from kkt_atlas.problem import Array, matrix


@dataclass(frozen=True)
class FittingCatalogData:
    """Finite catalog with training and validation observations but no test data."""

    catalog: Array
    training: ObservedSplit
    validation: ObservedSplit
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        catalog = np.asarray(self.catalog, dtype=np.float64)
        if catalog.ndim != 2 or len(catalog) == 0 or np.any(~np.isfinite(catalog)):
            raise ValueError("catalog must be a nonempty finite matrix")
        context_dim = self.training.contexts.shape[1]
        if self.validation.contexts.shape[1] != context_dim:
            raise ValueError("training and validation context dimensions differ")
        if np.any(self.training.outcome_indices >= len(catalog)):
            raise ValueError("training outcome index exceeds the catalog")
        if np.any(self.validation.outcome_indices >= len(catalog)):
            raise ValueError("validation outcome index exceeds the catalog")
        catalog = _immutable_float_array(catalog)
        object.__setattr__(self, "catalog", catalog)
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
    def catalog_hash(self) -> str:
        return array_sha256(self.catalog)

    @property
    def data_hash(self) -> str:
        return _bundle_hash(
            b"kkt-atlas-fitting-catalog-data-v1\0",
            (
                ("catalog", self.catalog),
                ("training_contexts", self.training.contexts),
                ("training_indices", self.training.outcome_indices),
                ("validation_contexts", self.validation.contexts),
                ("validation_indices", self.validation.outcome_indices),
            ),
            self.metadata,
        )


@dataclass(frozen=True)
class ConditionalTestData:
    """Evaluator-only contexts and exact conditional probabilities."""

    contexts: Array
    probabilities: Array
    catalog_hash: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        contexts = np.asarray(self.contexts, dtype=np.float64)
        if contexts.ndim != 2 or len(contexts) == 0 or np.any(~np.isfinite(contexts)):
            raise ValueError("test contexts must be a nonempty finite matrix")
        raw_probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if raw_probabilities.ndim != 2 or raw_probabilities.shape[1] == 0:
            raise ValueError("test probabilities must be a nonempty matrix")
        probabilities = matrix(
            raw_probabilities,
            columns=raw_probabilities.shape[1],
            name="test_probabilities",
        )
        if len(probabilities) != len(contexts):
            raise ValueError("test probabilities must align with test contexts")
        if np.any(probabilities < 0.0):
            raise ValueError("test probabilities must be nonnegative")
        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("each test probability row must sum to one")
        digest = str(self.catalog_hash)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("catalog_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "contexts", _immutable_float_array(contexts))
        object.__setattr__(self, "probabilities", _immutable_float_array(probabilities))
        object.__setattr__(self, "catalog_hash", digest)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def support_size(self) -> int:
        return int(self.probabilities.shape[1])

    @property
    def context_dim(self) -> int:
        return int(self.contexts.shape[1])

    @property
    def data_hash(self) -> str:
        metadata = dict(self.metadata)
        metadata["catalog_hash"] = self.catalog_hash
        return _bundle_hash(
            b"kkt-atlas-conditional-test-data-v1\0",
            (
                ("test_contexts", self.contexts),
                ("test_probabilities", self.probabilities),
            ),
            metadata,
        )


def combine_fitting_and_test_data(
    fitting: FittingCatalogData,
    test: ConditionalTestData,
) -> FiniteCatalogData:
    """Build the legacy all-split object after verifying catalog identity."""

    if fitting.catalog_hash != test.catalog_hash:
        raise ValueError("fitting and test data refer to different catalogs")
    if fitting.context_dim != test.context_dim:
        raise ValueError("fitting and test context dimensions differ")
    if fitting.support_size != test.support_size:
        raise ValueError("fitting and test support sizes differ")
    if fitting.metadata != test.metadata:
        raise ValueError("fitting and test metadata differ")
    return FiniteCatalogData(
        catalog=fitting.catalog,
        training=fitting.training,
        validation=fitting.validation,
        test_contexts=test.contexts,
        test_probabilities=test.probabilities,
        metadata=fitting.metadata,
    )


def array_sha256(values: Array) -> str:
    """Return a shape- and dtype-aware SHA-256 digest for one array."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_float_array(values: Array) -> Array:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).copy())
    array.setflags(write=False)
    return array


def _bundle_hash(
    prefix: bytes,
    arrays: tuple[tuple[str, Array], ...],
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(prefix)
    for name, values in arrays:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array_sha256(values).encode("ascii") + b"\0")
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest.update(b"metadata\0" + metadata_bytes)
    return digest.hexdigest()
