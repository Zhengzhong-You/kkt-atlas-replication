"""Round-trip serialization for a validation-selected KKT Atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from kkt_atlas.atlas import (
    FittedAtlas,
    SelectedLocalModel,
    TrainedLocalCandidate,
)
from kkt_atlas.charts import ContextCharts
from kkt_atlas.initialization import ScenarioInitialization
from kkt_atlas.problem import Array, DifferentiableStochasticProblem, restore_prescription
from kkt_atlas.response_basis import ResponseDecomposition
from kkt_atlas.scenario_model import LocalModelSnapshot

_SCHEMA = "kkt-atlas-fitted-model-v1"


def save_fitted_atlas(atlas: FittedAtlas, path: str | Path, *, prescription: dict | None = None) -> Path:
    """Write a selected atlas to a new directory without overwriting."""

    restore_prescription(atlas.problem, prescription)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, Array] = {
        "charts_mean": atlas.charts.mean,
        "charts_scale": atlas.charts.scale,
        "charts_standardized_centers": atlas.charts.standardized_centers,
        "charts_centers": atlas.charts.centers,
        "charts_assignments": atlas.charts.assignments,
    }
    local_metadata: list[dict[str, Any]] = []
    for chart, selected in enumerate(atlas.local_models):
        candidate = selected.candidate
        decomposition = candidate.decomposition
        initialization = candidate.initialization
        checkpoint = candidate.checkpoint
        prefix = f"chart_{chart}"
        arrays.update(
            {
                f"{prefix}_context_center": decomposition.context_center,
                f"{prefix}_local_decision": decomposition.local_decision,
                f"{prefix}_transformed_mean": decomposition.transformed_mean,
                f"{prefix}_transformed_scale": decomposition.transformed_scale,
                f"{prefix}_left_vectors": decomposition.left_vectors,
                f"{prefix}_singular_values": decomposition.singular_values,
                f"{prefix}_cross_covariance": decomposition.cross_covariance,
                f"{prefix}_initial_coordinates": initialization.coordinates,
                f"{prefix}_initial_counts": initialization.counts,
                f"{prefix}_scenario_coordinates": checkpoint.scenario_coordinates,
                f"{prefix}_probability_intercepts": checkpoint.probability_intercepts,
                f"{prefix}_probability_slopes": checkpoint.probability_slopes,
            }
        )
        local_metadata.append(
            {
                "chart": chart,
                "basis_method": decomposition.basis_method,
                "width": candidate.width,
                "support_size": candidate.support_size,
                "checkpoint_step": checkpoint.step,
                "training_objective": checkpoint.training_objective,
                "mean_operational_loss": checkpoint.mean_operational_loss,
                "validation_loss_sum": selected.validation_loss_sum,
                "validation_count": selected.validation_count,
            }
        )

    arrays_path = destination / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "schema": _SCHEMA,
        "problem_name": atlas.problem.name,
        "context_dim": atlas.problem.context_dim,
        "scenario_dim": atlas.problem.scenario_dim,
        "decision_dim": atlas.problem.decision_dim,
        "ridge": atlas.ridge,
        "validation_mean_loss": atlas.validation_mean_loss,
        "chart_distortion": atlas.charts.distortion,
        "local_models": local_metadata,
        "arrays_sha256": _sha256(arrays_path),
    }
    if prescription is not None:
        metadata["prescription"] = prescription
    metadata_path = destination / "atlas.json"
    metadata_path.write_bytes(_canonical_json_bytes(metadata))
    completion = {
        "schema": _SCHEMA,
        "atlas_sha256": _sha256(metadata_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    (destination / "COMPLETED.json").write_bytes(_canonical_json_bytes(completion))
    return destination


def load_fitted_atlas(
    path: str | Path,
    problem: DifferentiableStochasticProblem,
) -> FittedAtlas:
    """Load a complete atlas and verify its stored hashes and problem dimensions."""

    source = Path(path)
    metadata_path = source / "atlas.json"
    arrays_path = source / "arrays.npz"
    completion_path = source / "COMPLETED.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"incomplete or invalid atlas serialization: {source}") from exc
    if completion.get("schema") != _SCHEMA or metadata.get("schema") != _SCHEMA:
        raise ValueError("unsupported atlas serialization schema")
    if completion.get("atlas_sha256") != _sha256(metadata_path):
        raise ValueError("atlas metadata hash does not match")
    arrays_hash = _sha256(arrays_path)
    if (
        completion.get("arrays_sha256") != arrays_hash
        or metadata.get("arrays_sha256") != arrays_hash
    ):
        raise ValueError("atlas array hash does not match")
    expected_problem = (
        problem.name,
        problem.context_dim,
        problem.scenario_dim,
        problem.decision_dim,
    )
    stored_problem = (
        metadata.get("problem_name"),
        metadata.get("context_dim"),
        metadata.get("scenario_dim"),
        metadata.get("decision_dim"),
    )
    if stored_problem != expected_problem:
        raise ValueError("serialized atlas does not match the supplied problem")
    problem = restore_prescription(problem, metadata.get("prescription"))

    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: _immutable(stored[name]) for name in stored.files}
    charts = ContextCharts(
        mean=_take(arrays, "charts_mean"),
        scale=_take(arrays, "charts_scale"),
        standardized_centers=_take(arrays, "charts_standardized_centers"),
        centers=_take(arrays, "charts_centers"),
        assignments=_take(arrays, "charts_assignments"),
        distortion=float(metadata["chart_distortion"]),
    )
    local_models: list[SelectedLocalModel] = []
    for item in metadata["local_models"]:
        chart = int(item["chart"])
        prefix = f"chart_{chart}"
        decomposition = ResponseDecomposition(
            context_center=_take(arrays, f"{prefix}_context_center"),
            local_decision=_take(arrays, f"{prefix}_local_decision"),
            transformed_mean=_take(arrays, f"{prefix}_transformed_mean"),
            transformed_scale=_take(arrays, f"{prefix}_transformed_scale"),
            left_vectors=_take(arrays, f"{prefix}_left_vectors"),
            singular_values=_take(arrays, f"{prefix}_singular_values"),
            cross_covariance=_take(arrays, f"{prefix}_cross_covariance"),
            basis_method=str(item["basis_method"]),
        )
        initialization = ScenarioInitialization(
            coordinates=_take(arrays, f"{prefix}_initial_coordinates"),
            counts=_take(arrays, f"{prefix}_initial_counts"),
        )
        checkpoint = LocalModelSnapshot(
            step=int(item["checkpoint_step"]),
            scenario_coordinates=_take(arrays, f"{prefix}_scenario_coordinates"),
            probability_intercepts=_take(arrays, f"{prefix}_probability_intercepts"),
            probability_slopes=_take(arrays, f"{prefix}_probability_slopes"),
            training_objective=float(item["training_objective"]),
            mean_operational_loss=float(item["mean_operational_loss"]),
        )
        candidate = TrainedLocalCandidate(
            chart=chart,
            ridge=float(metadata["ridge"]),
            width=int(item["width"]),
            support_size=int(item["support_size"]),
            checkpoint=checkpoint,
            decomposition=decomposition,
            initialization=initialization,
        )
        local_models.append(
            SelectedLocalModel(
                candidate=candidate,
                validation_loss_sum=float(item["validation_loss_sum"]),
                validation_count=int(item["validation_count"]),
            )
        )
    local_models.sort(key=lambda selected: selected.candidate.chart)
    if arrays:
        raise ValueError(f"unexpected serialized arrays: {sorted(arrays)}")
    return FittedAtlas(
        problem=problem,
        charts=charts,
        ridge=float(metadata["ridge"]),
        local_models=tuple(local_models),
        validation_mean_loss=float(metadata["validation_mean_loss"]),
    )


def _take(arrays: dict[str, Array], name: str) -> Array:
    try:
        return arrays.pop(name)
    except KeyError as exc:
        raise ValueError(f"serialized atlas is missing array {name!r}") from exc


def _immutable(values: Array) -> Array:
    array = np.ascontiguousarray(np.asarray(values).copy())
    array.setflags(write=False)
    return array


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
