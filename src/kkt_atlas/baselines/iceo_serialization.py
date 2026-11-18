"""Round-trip serialization for validation-selected full-support ICEO models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kkt_atlas.baselines.iceo import (
    FittedICEO,
    FullSupportICEOModel,
    ICEOModelConfig,
    ICEOSnapshot,
    ICEOTrainingResult,
)
from kkt_atlas.problem import DifferentiableStochasticProblem

_SCHEMA = "kkt-atlas-fitted-full-support-iceo-v1"


def save_fitted_iceo(fitted: FittedICEO, path: str | Path) -> Path:
    """Write the selected network and every saved training checkpoint."""

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    model = fitted.model
    arrays: dict[str, np.ndarray] = {
        "catalog": model.catalog.detach().cpu().numpy().copy(),
        "context_mean": model.context_mean.detach().cpu().numpy().copy(),
        "context_scale": model.context_scale.detach().cpu().numpy().copy(),
    }
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(
        fitted.training_result.checkpoints
    ):
        parameters: list[dict[str, Any]] = []
        for parameter_index, (name, value) in enumerate(checkpoint.parameters):
            key = f"checkpoint_{checkpoint_index}_parameter_{parameter_index}"
            arrays[key] = np.asarray(value, dtype=np.float64)
            parameters.append({"name": name, "array": key})
        checkpoint_rows.append(
            {
                "step": checkpoint.step,
                "training_loss": checkpoint.training_loss,
                "parameters": parameters,
            }
        )

    arrays_path = destination / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "schema": _SCHEMA,
        "problem": {
            "name": model.problem.name,
            "context_dim": model.problem.context_dim,
            "scenario_dim": model.problem.scenario_dim,
            "decision_dim": model.problem.decision_dim,
        },
        "model_config": {
            "hidden_width": model.model_config.hidden_width,
            "hidden_layers": model.model_config.hidden_layers,
            "seed": model.model_config.seed,
        },
        "ridge": fitted.ridge,
        "selected_step": fitted.selected_step,
        "validation_loss": fitted.validation_loss,
        "final_step": fitted.training_result.final_step,
        "checkpoints": checkpoint_rows,
        "arrays_sha256": _sha256(arrays_path),
    }
    metadata_path = destination / "iceo.json"
    metadata_path.write_bytes(_canonical_json_bytes(metadata))
    completion = {
        "schema": _SCHEMA,
        "metadata_sha256": _sha256(metadata_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    (destination / "COMPLETED.json").write_bytes(
        _canonical_json_bytes(completion)
    )
    return destination


def load_fitted_iceo(
    path: str | Path,
    problem: DifferentiableStochasticProblem,
) -> FittedICEO:
    """Load a complete ICEO fit and reject any hash or dimension mismatch."""

    source = Path(path)
    metadata_path = source / "iceo.json"
    arrays_path = source / "arrays.npz"
    completion_path = source / "COMPLETED.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"incomplete or invalid ICEO serialization: {source}") from error
    if metadata.get("schema") != _SCHEMA or completion.get("schema") != _SCHEMA:
        raise ValueError("unsupported ICEO serialization schema")
    arrays_hash = _sha256(arrays_path)
    if (
        completion.get("metadata_sha256") != _sha256(metadata_path)
        or completion.get("arrays_sha256") != arrays_hash
        or metadata.get("arrays_sha256") != arrays_hash
    ):
        raise ValueError("ICEO serialization hash mismatch")
    expected_problem = {
        "name": problem.name,
        "context_dim": problem.context_dim,
        "scenario_dim": problem.scenario_dim,
        "decision_dim": problem.decision_dim,
    }
    if metadata.get("problem") != expected_problem:
        raise ValueError("serialized ICEO model does not match the supplied problem")

    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {
            name: np.ascontiguousarray(stored[name].copy())
            for name in stored.files
        }
    catalog = _take(arrays, "catalog")
    context_mean = _take(arrays, "context_mean")
    context_scale = _take(arrays, "context_scale")
    config = ICEOModelConfig(**metadata["model_config"])
    model = FullSupportICEOModel(
        problem,
        catalog,
        np.zeros((1, problem.context_dim), dtype=np.float64),
        config=config,
        dtype=torch.float64,
        device="cpu",
    )
    with torch.no_grad():
        model.context_mean.copy_(torch.as_tensor(context_mean, dtype=torch.float64))
        model.context_scale.copy_(torch.as_tensor(context_scale, dtype=torch.float64))

    checkpoints: list[ICEOSnapshot] = []
    for checkpoint_row in metadata["checkpoints"]:
        parameters: list[tuple[str, np.ndarray]] = []
        for parameter_row in checkpoint_row["parameters"]:
            parameters.append(
                (
                    str(parameter_row["name"]),
                    _take(arrays, str(parameter_row["array"])),
                )
            )
        checkpoints.append(
            ICEOSnapshot(
                step=int(checkpoint_row["step"]),
                parameters=tuple(parameters),
                training_loss=float(checkpoint_row["training_loss"]),
            )
        )
    if arrays:
        raise ValueError(f"unexpected serialized ICEO arrays: {sorted(arrays)}")
    training_result = ICEOTrainingResult(
        checkpoints=tuple(checkpoints),
        final_step=int(metadata["final_step"]),
    )
    selected_step = int(metadata["selected_step"])
    selected = training_result.checkpoint(selected_step)
    model.load_snapshot(selected)
    model.eval()
    return FittedICEO(
        model=model,
        training_result=training_result,
        selected_step=selected_step,
        validation_loss=float(metadata["validation_loss"]),
        ridge=float(metadata["ridge"]),
    )


def _take(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    try:
        value = arrays.pop(name)
    except KeyError as error:
        raise ValueError(f"serialized ICEO model is missing {name!r}") from error
    if np.any(~np.isfinite(value)):
        raise ValueError(f"serialized ICEO array {name!r} is nonfinite")
    return value


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
