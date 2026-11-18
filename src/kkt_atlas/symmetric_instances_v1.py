"""Standalone, checksummed input bundles for the symmetric campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kkt_atlas.confirmation_data_v1 import ConditionalTestData, FittingCatalogData
from kkt_atlas.data import ObservedSplit

SCHEMA = "kkt-atlas-standalone-instance-v1"
_DATA_FILES = {
    "catalog": "catalog.npy",
    "training_contexts": "training_contexts.npy",
    "training_indices": "training_indices.npy",
    "validation_contexts": "validation_contexts.npy",
    "validation_indices": "validation_indices.npy",
    "evaluation_contexts": "evaluation_contexts.npy",
    "evaluation_probabilities": "evaluation_probabilities.npy",
}


@dataclass(frozen=True)
class StandaloneInstance:
    fitting: FittingCatalogData
    evaluation: ConditionalTestData
    extra_arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def write_standalone_instance(
    path: Path,
    fitting: FittingCatalogData,
    evaluation: ConditionalTestData,
    *,
    protocol_sha256: str,
    spec: Mapping[str, Any],
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write one immutable directory; an existing target is never overwritten."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    if fitting.catalog_hash != evaluation.catalog_hash or fitting.metadata != evaluation.metadata:
        raise ValueError("fitting and evaluation data do not describe the same instance")
    if len(protocol_sha256) != 64:
        raise ValueError("protocol_sha256 must be a full SHA-256 digest")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        arrays = {
            "catalog": fitting.catalog,
            "training_contexts": fitting.training.contexts,
            "training_indices": fitting.training.outcome_indices,
            "validation_contexts": fitting.validation.contexts,
            "validation_indices": fitting.validation.outcome_indices,
            "evaluation_contexts": evaluation.contexts,
            "evaluation_probabilities": evaluation.probabilities,
        }
        files: dict[str, dict[str, Any]] = {}
        for name, values in arrays.items():
            filename = _DATA_FILES[name]
            np.save(temporary / filename, np.asarray(values), allow_pickle=False)
            files[name] = _file_record(temporary / filename, values)

        extras: dict[str, dict[str, Any]] = {}
        for name, values in sorted((extra_arrays or {}).items()):
            if not name or not name.replace("_", "").isalnum():
                raise ValueError(f"invalid extra array name: {name!r}")
            array = np.asarray(values)
            if array.dtype.hasobject or np.any(~np.isfinite(array)):
                raise ValueError(f"extra array {name!r} must be finite and numeric")
            filename = f"extra_{name}.npy"
            np.save(temporary / filename, array, allow_pickle=False)
            extras[name] = _file_record(temporary / filename, array)

        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(_canonical_json(fitting.metadata) + "\n", encoding="utf-8")
        manifest = {
            "schema": SCHEMA,
            "protocol_sha256": protocol_sha256,
            "spec": dict(spec),
            "catalog_sha256": fitting.catalog_hash,
            "fitting_data_sha256": fitting.data_hash,
            "evaluation_data_sha256": evaluation.data_hash,
            "files": files,
            "extra_arrays": extras,
            "metadata_file": {
                "filename": metadata_path.name,
                "sha256": _file_sha256(metadata_path),
                "bytes": metadata_path.stat().st_size,
            },
        }
        (temporary / "MANIFEST.json").write_text(
            _canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_standalone_instance(path: Path) -> StandaloneInstance:
    """Load one bundle only after verifying every declared file hash."""

    root = Path(path)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("wrong standalone instance schema")
    metadata_record = manifest["metadata_file"]
    metadata_path = root / metadata_record["filename"]
    _verify_file(metadata_path, metadata_record)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    arrays: dict[str, np.ndarray] = {}
    for name, record in manifest["files"].items():
        path_value = root / record["filename"]
        _verify_file(path_value, record)
        arrays[name] = np.load(path_value, allow_pickle=False)
        _verify_array(arrays[name], record)

    extras: dict[str, np.ndarray] = {}
    for name, record in manifest.get("extra_arrays", {}).items():
        path_value = root / record["filename"]
        _verify_file(path_value, record)
        extras[name] = np.load(path_value, allow_pickle=False)
        _verify_array(extras[name], record)

    fitting = FittingCatalogData(
        catalog=arrays["catalog"],
        training=ObservedSplit(arrays["training_contexts"], arrays["training_indices"]),
        validation=ObservedSplit(arrays["validation_contexts"], arrays["validation_indices"]),
        metadata=metadata,
    )
    evaluation = ConditionalTestData(
        contexts=arrays["evaluation_contexts"],
        probabilities=arrays["evaluation_probabilities"],
        catalog_hash=fitting.catalog_hash,
        metadata=metadata,
    )
    if fitting.data_hash != manifest["fitting_data_sha256"]:
        raise ValueError("fitting-data hash mismatch")
    if evaluation.data_hash != manifest["evaluation_data_sha256"]:
        raise ValueError("evaluation-data hash mismatch")
    return StandaloneInstance(fitting, evaluation, extras, manifest)


def _file_record(path: Path, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "filename": path.name,
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _verify_file(path: Path, record: Mapping[str, Any]) -> None:
    if path.stat().st_size != record["bytes"] or _file_sha256(path) != record["sha256"]:
        raise ValueError(f"file hash mismatch: {path.name}")


def _verify_array(values: np.ndarray, record: Mapping[str, Any]) -> None:
    if values.dtype.str != record["dtype"] or list(values.shape) != record["shape"]:
        raise ValueError(f"array metadata mismatch: {record['filename']}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
