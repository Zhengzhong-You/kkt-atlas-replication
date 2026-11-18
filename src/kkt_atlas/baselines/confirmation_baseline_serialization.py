"""Round-trip serialization for frozen non-neural confirmation baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from kkt_atlas.atlas import PreparedAtlas
from kkt_atlas.baselines.contextual_empirical import KNearestNeighborSAA
from kkt_atlas.baselines.matched_fixed_support import (
    MatchedObservedSupportResult,
    ObservedSupportInitialization,
    ObservedSupportLocalModel,
)
from kkt_atlas.charts import ContextCharts
from kkt_atlas.initialization import ScenarioInitialization
from kkt_atlas.problem import DifferentiableStochasticProblem, restore_prescription
from kkt_atlas.response_basis import ResponseDecomposition
from kkt_atlas.scenario_model import LocalModelSnapshot
from kkt_atlas.scientific_identity import (
    capacity_problem_identity,
    network_reservation_problem_identity,
    portfolio_problem_identity,
)

SCHEMA = "kkt-atlas-confirmation-baseline-bundle-v1"
METADATA_FILE = "baselines.json"
ARRAYS_FILE = "arrays.npz"
COMPLETION_FILE = "COMPLETED.json"
MATCHED_ONLY_SCHEMA = "kkt-atlas-matched-observed-support-bundle-v1"


@dataclass(frozen=True)
class LoadedConfirmationBaselines:
    """Frozen non-neural baselines reconstructed without fitting data."""

    knn_saa: KNearestNeighborSAA
    matched_fixed_support: PreparedAtlas | None
    matched_selected_checkpoint_step: int | None
    identities: dict[str, str]


@dataclass(frozen=True)
class LoadedMatchedObservedSupport:
    atlas: PreparedAtlas
    selected_checkpoint_step: int
    identity_sha256: str


def save_matched_observed_support_bundle(
    fitted: MatchedObservedSupportResult,
    path: str | Path,
    *,
    selected_checkpoint_step: int,
    support_cap_offset: int = 0,
) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    problem = fitted.prepared_atlas.problem
    problem_record = _problem_record(problem, ridge=fitted.prepared_atlas.ridge)
    matched = _serialize_matched(
        fitted,
        arrays,
        problem,
        problem_record,
        selected_checkpoint_step=selected_checkpoint_step,
        support_cap_offset=support_cap_offset,
    )
    arrays_path = destination / ARRAYS_FILE
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "schema": MATCHED_ONLY_SCHEMA,
        "problem": problem_record,
        "model": matched,
        "arrays_sha256": _sha256_file(arrays_path),
    }
    metadata_path = destination / METADATA_FILE
    metadata_path.write_bytes(_canonical_json_bytes(metadata))
    completion = {
        "schema": MATCHED_ONLY_SCHEMA,
        "state": "COMPLETE",
        "metadata_sha256": _sha256_file(metadata_path),
        "arrays_sha256": _sha256_file(arrays_path),
        "identity_sha256": matched["identity_sha256"],
    }
    (destination / COMPLETION_FILE).write_bytes(_canonical_json_bytes(completion))
    return destination


def load_matched_observed_support_bundle(
    path: str | Path,
    problem: DifferentiableStochasticProblem,
) -> LoadedMatchedObservedSupport:
    source = Path(path)
    metadata_path = source / METADATA_FILE
    arrays_path = source / ARRAYS_FILE
    completion_path = source / COMPLETION_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != MATCHED_ONLY_SCHEMA or completion.get("schema") != MATCHED_ONLY_SCHEMA:
        raise ValueError("unsupported matched observed-support bundle schema")
    observed_arrays_hash = _sha256_file(arrays_path)
    if (
        completion.get("state") != "COMPLETE"
        or completion.get("metadata_sha256") != _sha256_file(metadata_path)
        or completion.get("arrays_sha256") != observed_arrays_hash
        or metadata.get("arrays_sha256") != observed_arrays_hash
    ):
        raise ValueError("matched observed-support bundle hash mismatch")
    record = metadata.get("model")
    if not isinstance(record, dict):
        raise ValueError("matched observed-support metadata omit the model")
    ridge = _nonnegative_float(record.get("ridge"), "ridge")
    if metadata.get("problem") != _problem_record(problem, ridge=ridge):
        raise ValueError("matched observed-support bundle does not match the problem")
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: np.ascontiguousarray(stored[name].copy()) for name in stored.files}
    atlas = _deserialize_matched(record, arrays, problem)
    if arrays:
        raise ValueError(f"unexpected matched-support arrays: {sorted(arrays)}")
    identity = _identity_from_record(record)
    if completion.get("identity_sha256") != identity:
        raise ValueError("matched observed-support identity hash mismatch")
    return LoadedMatchedObservedSupport(
        atlas=atlas,
        selected_checkpoint_step=_nonnegative_integer(
            record.get("selected_checkpoint_step"),
            "matched selected checkpoint step",
        ),
        identity_sha256=identity,
    )


def save_confirmation_baseline_bundle(
    knn_saa: KNearestNeighborSAA,
    matched_fixed_support: MatchedObservedSupportResult | None,
    path: str | Path,
    *,
    matched_selected_checkpoint_step: int | None = None,
    prescription: dict | None = None,
) -> Path:
    """Save KNN-SAA and an optional matched fixed-support atlas."""

    if prescription is not None and matched_fixed_support is not None:
        raise ValueError("Explicit prescription currently supports the standalone kNN bundle only")
    restore_prescription(knn_saa.problem, prescription)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    problem_record = _problem_record(knn_saa.problem, ridge=knn_saa.ridge)
    knn_record = _serialize_knn(knn_saa, arrays, problem_record)
    if matched_fixed_support is None:
        if matched_selected_checkpoint_step is not None:
            raise ValueError(
                "matched_selected_checkpoint_step requires a matched model"
            )
        matched_record = None
    else:
        if matched_selected_checkpoint_step is None:
            raise ValueError(
                "matched_selected_checkpoint_step is required for a matched model"
            )
        matched_record = _serialize_matched(
            matched_fixed_support,
            arrays,
            knn_saa.problem,
            problem_record,
            selected_checkpoint_step=matched_selected_checkpoint_step,
        )
    arrays_path = destination / ARRAYS_FILE
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "schema": SCHEMA,
        "problem": problem_record,
        "models": {
            "knn_saa": knn_record,
            "matched_fixed_support": matched_record,
        },
        "arrays_sha256": _sha256_file(arrays_path),
    }
    if prescription is not None:
        metadata["prescription"] = prescription
    metadata_path = destination / METADATA_FILE
    metadata_path.write_bytes(_canonical_json_bytes(metadata))
    completion = {
        "schema": SCHEMA,
        "state": "COMPLETE",
        "metadata_sha256": _sha256_file(metadata_path),
        "arrays_sha256": _sha256_file(arrays_path),
        "model_identity_sha256": {
            "knn_saa": knn_record["identity_sha256"],
            **(
                {}
                if matched_record is None
                else {
                    "matched_fixed_support": matched_record["identity_sha256"]
                }
            ),
        },
    }
    (destination / COMPLETION_FILE).write_bytes(
        _canonical_json_bytes(completion)
    )
    return destination


def load_confirmation_baseline_bundle(
    path: str | Path,
    problem: DifferentiableStochasticProblem,
) -> LoadedConfirmationBaselines:
    """Load the baseline bundle and reject any mutation or mismatch."""

    source = Path(path)
    metadata_path = source / METADATA_FILE
    arrays_path = source / ARRAYS_FILE
    completion_path = source / COMPLETION_FILE
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"incomplete or invalid baseline bundle: {source}") from error
    if metadata.get("schema") != SCHEMA or completion.get("schema") != SCHEMA:
        raise ValueError("unsupported confirmation baseline schema")
    if completion.get("state") != "COMPLETE":
        raise ValueError("baseline completion state is not COMPLETE")
    observed_arrays_hash = _sha256_file(arrays_path)
    if (
        completion.get("metadata_sha256") != _sha256_file(metadata_path)
        or completion.get("arrays_sha256") != observed_arrays_hash
        or metadata.get("arrays_sha256") != observed_arrays_hash
    ):
        raise ValueError("confirmation baseline bundle hash mismatch")
    knn_record = metadata.get("models", {}).get("knn_saa")
    if not isinstance(knn_record, dict):
        raise ValueError("baseline metadata omit KNN-SAA")
    ridge = _nonnegative_float(knn_record.get("ridge"), "ridge")
    if metadata.get("problem") != _problem_record(problem, ridge=ridge):
        raise ValueError("serialized baselines do not match the supplied problem")

    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {
            name: np.ascontiguousarray(stored[name].copy())
            for name in stored.files
        }
    models = metadata.get("models")
    if not isinstance(models, dict):
        raise ValueError("baseline metadata omit the model records")
    knn_record = models.get("knn_saa")
    if not isinstance(knn_record, dict):
        raise ValueError("baseline metadata omit KNN-SAA")
    knn = _deserialize_knn(knn_record, arrays, problem)
    matched_record = models.get("matched_fixed_support")
    matched = (
        None
        if matched_record is None
        else _deserialize_matched(matched_record, arrays, problem)
    )
    matched_selected_step = (
        None
        if matched_record is None
        else _nonnegative_integer(
            matched_record.get("selected_checkpoint_step"),
            "matched selected checkpoint step",
        )
    )
    if arrays:
        raise ValueError(f"unexpected serialized baseline arrays: {sorted(arrays)}")
    identities = {
        "knn_saa": _identity_from_record(knn_record),
        **(
            {}
            if matched_record is None
            else {
                "matched_fixed_support": _identity_from_record(matched_record)
            }
        ),
    }
    if completion.get("model_identity_sha256") != identities:
        raise ValueError("baseline completion identities do not match metadata")
    if metadata.get("prescription") is not None:
        if matched is not None:
            raise ValueError("Explicit prescription does not support a combined matched bundle")
        knn = replace(knn, problem=restore_prescription(problem, metadata["prescription"]))
    return LoadedConfirmationBaselines(
        knn_saa=knn,
        matched_fixed_support=matched,
        matched_selected_checkpoint_step=matched_selected_step,
        identities=identities,
    )


def baseline_bundle_artifacts(path: str | Path) -> dict[str, str]:
    """Return canonical relative paths and hashes for a completed bundle."""

    source = Path(path)
    return {
        name: _sha256_file(source / name)
        for name in (METADATA_FILE, ARRAYS_FILE, COMPLETION_FILE)
    }


def _serialize_knn(
    fitted: KNearestNeighborSAA,
    arrays: dict[str, np.ndarray],
    problem_record: dict[str, Any],
) -> dict[str, Any]:
    keys = {
        "catalog": _put(arrays, "knn__catalog", fitted.catalog),
        "training_contexts": _put(
            arrays,
            "knn__training_contexts",
            fitted.training_contexts,
        ),
        "training_outcome_indices": _put(
            arrays,
            "knn__training_outcome_indices",
            fitted.training_outcome_indices,
        ),
        "context_mean": _put(arrays, "knn__context_mean", fitted.context_mean),
        "context_scale": _put(arrays, "knn__context_scale", fitted.context_scale),
    }
    identity = {
        "model": "knn_saa",
        "problem_sha256": problem_record["scientific_identity"]["sha256"],
        "neighbor_count": int(fitted.neighbor_count),
        "ridge": float(fitted.ridge),
        "validation_loss": float(fitted.validation_loss),
        "array_sha256": {
            name: _array_sha256(arrays[key]) for name, key in keys.items()
        },
    }
    return {
        "neighbor_count": int(fitted.neighbor_count),
        "ridge": float(fitted.ridge),
        "validation_loss": float(fitted.validation_loss),
        "arrays": keys,
        "identity": identity,
        "identity_sha256": _canonical_digest(identity),
    }


def _deserialize_knn(
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    problem: DifferentiableStochasticProblem,
) -> KNearestNeighborSAA:
    identity = _validated_identity(record, "knn_saa")
    scalar_values = {
        "neighbor_count": _positive_integer(
            record.get("neighbor_count"),
            "neighbor_count",
        ),
        "ridge": _nonnegative_float(record.get("ridge"), "ridge"),
        "validation_loss": _finite_float(
            record.get("validation_loss"),
            "validation_loss",
        ),
    }
    expected_problem = _problem_record(problem, ridge=scalar_values["ridge"])
    if identity.get("problem_sha256") != expected_problem["scientific_identity"]["sha256"]:
        raise ValueError("KNN-SAA identity does not match the supplied problem")
    if any(identity.get(name) != value for name, value in scalar_values.items()):
        raise ValueError("KNN-SAA scalar identity does not match metadata")
    keys = _array_key_map(record, required={
        "catalog",
        "training_contexts",
        "training_outcome_indices",
        "context_mean",
        "context_scale",
    })
    values = {name: _take(arrays, key) for name, key in keys.items()}
    _validate_array_hashes(identity, values)
    return KNearestNeighborSAA(
        problem=problem,
        catalog=values["catalog"],
        training_contexts=values["training_contexts"],
        training_outcome_indices=values["training_outcome_indices"],
        context_mean=values["context_mean"],
        context_scale=values["context_scale"],
        neighbor_count=int(scalar_values["neighbor_count"]),
        ridge=float(scalar_values["ridge"]),
        validation_loss=float(scalar_values["validation_loss"]),
    )


def _serialize_matched(
    fitted: MatchedObservedSupportResult,
    arrays: dict[str, np.ndarray],
    problem: DifferentiableStochasticProblem,
    problem_record: dict[str, Any],
    *,
    selected_checkpoint_step: int,
    support_cap_offset: int = 0,
) -> dict[str, Any]:
    if type(support_cap_offset) is not int or support_cap_offset not in (0,1):
        raise ValueError('support_cap_offset must be zero or one')
    prepared = fitted.prepared_atlas
    if _problem_record(prepared.problem, ridge=prepared.ridge) != problem_record:
        raise ValueError("KNN and matched baselines must use the same stochastic program")
    if prepared.ridge != float(problem_record["scientific_identity"]["scalars"]["decision_regularization"]):
        raise ValueError("KNN and matched baselines must use the same regularization parameter")
    chart_count = prepared.charts.chart_count
    selected_step = _nonnegative_integer(
        selected_checkpoint_step,
        "matched selected checkpoint step",
    )
    aligned = (
        len(prepared.models)
        == len(fitted.decompositions)
        == len(fitted.initializations)
        == chart_count
    )
    if not aligned:
        raise ValueError("matched fixed-support components do not align by chart")
    if len(fitted.training_results) != chart_count:
        raise ValueError("matched training results do not align by chart")
    _validate_chart_router(prepared.charts, problem)
    chart_arrays = {
        "mean": _put(arrays, "matched__charts__mean", prepared.charts.mean),
        "scale": _put(arrays, "matched__charts__scale", prepared.charts.scale),
        "standardized_centers": _put(
            arrays,
            "matched__charts__standardized_centers",
            prepared.charts.standardized_centers,
        ),
        "centers": _put(arrays, "matched__charts__centers", prepared.charts.centers),
        "assignments": _put(
            arrays,
            "matched__charts__assignments",
            prepared.charts.assignments,
        ),
    }
    chart_records = []
    identity_arrays = {
        f"charts.{name}": _array_sha256(arrays[key])
        for name, key in chart_arrays.items()
    }
    for chart, (model, decomposition, initialization) in enumerate(
        zip(
            prepared.models,
            fitted.decompositions,
            fitted.initializations,
            strict=True,
        )
    ):
        if not isinstance(model, ObservedSupportLocalModel):
            raise TypeError("matched baseline must contain observed-support local models")
        if model.problem is not prepared.problem:
            raise ValueError("matched local models must use the prepared-atlas problem")
        width = int(model.width)
        support_size = int(model.support_size)
        if support_size > width + 1 + support_cap_offset:
            raise ValueError(f"matched fixed support exceeds the declared q+{1+support_cap_offset} cap")
        if support_size != initialization.support_size:
            raise ValueError("matched model and initialization support sizes differ")
        if width > decomposition.maximum_width:
            raise ValueError("matched width exceeds its response decomposition")
        if decomposition.basis_method != "kkt":
            raise ValueError("confirmation matched support requires a KKT response basis")
        _validate_decomposition(decomposition, problem, width=width)
        _validate_initialization(
            initialization,
            problem,
            width=width,
            support_size=support_size,
        )
        fixed_scenarios = model.fixed_observed_scenarios.detach().cpu().numpy()
        if not np.array_equal(fixed_scenarios, initialization.observed_scenarios):
            raise ValueError("matched local model does not contain the declared fixed support")
        selected_snapshot = fitted.training_results[chart].checkpoint(
            selected_step
        )
        _validate_live_matched_snapshot(model, selected_snapshot)
        prefix = f"matched__chart_{chart:04d}"
        local_arrays = {
            "decomposition_context_center": _put(
                arrays,
                f"{prefix}__decomposition_context_center",
                decomposition.context_center,
            ),
            "decomposition_local_decision": _put(
                arrays,
                f"{prefix}__decomposition_local_decision",
                decomposition.local_decision,
            ),
            "decomposition_transformed_mean": _put(
                arrays,
                f"{prefix}__decomposition_transformed_mean",
                decomposition.transformed_mean,
            ),
            "decomposition_transformed_scale": _put(
                arrays,
                f"{prefix}__decomposition_transformed_scale",
                decomposition.transformed_scale,
            ),
            "decomposition_left_vectors": _put(
                arrays,
                f"{prefix}__decomposition_left_vectors",
                decomposition.left_vectors,
            ),
            "decomposition_singular_values": _put(
                arrays,
                f"{prefix}__decomposition_singular_values",
                decomposition.singular_values,
            ),
            "decomposition_cross_covariance": _put(
                arrays,
                f"{prefix}__decomposition_cross_covariance",
                decomposition.cross_covariance,
            ),
            "centroid_coordinates": _put(
                arrays,
                f"{prefix}__centroid_coordinates",
                initialization.centroid_initialization.coordinates,
            ),
            "centroid_counts": _put(
                arrays,
                f"{prefix}__centroid_counts",
                initialization.centroid_initialization.counts,
            ),
            "local_observation_indices": _put(
                arrays,
                f"{prefix}__local_observation_indices",
                initialization.local_observation_indices,
            ),
            "observed_scenarios": _put(
                arrays,
                f"{prefix}__observed_scenarios",
                initialization.observed_scenarios,
            ),
            "projected_observed_coordinates": _put(
                arrays,
                f"{prefix}__projected_observed_coordinates",
                initialization.projected_observed_coordinates,
            ),
            "cluster_assignments": _put(
                arrays,
                f"{prefix}__cluster_assignments",
                initialization.cluster_assignments,
            ),
            "scenario_coordinates": _put(
                arrays,
                f"{prefix}__scenario_coordinates",
                model.scenario_coordinates.detach().cpu().numpy(),
            ),
            "probability_intercepts": _put(
                arrays,
                f"{prefix}__probability_intercepts",
                model.probability_intercepts.detach().cpu().numpy(),
            ),
            "probability_slopes": _put(
                arrays,
                f"{prefix}__probability_slopes",
                model.probability_slopes.detach().cpu().numpy(),
            ),
        }
        for name, key in local_arrays.items():
            identity_arrays[f"chart.{chart}.{name}"] = _array_sha256(arrays[key])
        chart_records.append(
            {
                "chart": chart,
                "q": width,
                "K": support_size,
                "basis_method": decomposition.basis_method,
                "arrays": local_arrays,
            }
        )
    identity = {
        "model": "matched_fixed_support",
        "problem_sha256": problem_record["scientific_identity"]["sha256"],
        "ridge": float(prepared.ridge),
        "chart_count": chart_count,
        "distortion": float(prepared.charts.distortion),
        "selected_checkpoint_step": selected_step,
        "q": [row["q"] for row in chart_records],
        "K": [row["K"] for row in chart_records],
        "array_sha256": identity_arrays,
    }
    if support_cap_offset:
        identity["support_cap_offset"] = support_cap_offset
    return {
        "ridge": float(prepared.ridge),
        "chart_count": chart_count,
        "distortion": float(prepared.charts.distortion),
        "selected_checkpoint_step": selected_step,
        "chart_arrays": chart_arrays,
        "charts": chart_records,
        "identity": identity,
        "identity_sha256": _canonical_digest(identity),
    }


def _deserialize_matched(
    record: Any,
    arrays: dict[str, np.ndarray],
    problem: DifferentiableStochasticProblem,
) -> PreparedAtlas:
    if not isinstance(record, dict):
        raise ValueError("matched fixed-support metadata must be an object or null")
    identity = _validated_identity(record, "matched_fixed_support")
    support_cap_offset = identity.get("support_cap_offset",0)
    if type(support_cap_offset) is not int or support_cap_offset not in (0,1):
        raise ValueError("support_cap_offset must be zero or one")
    chart_count = _positive_integer(record.get("chart_count"), "chart_count")
    ridge = _nonnegative_float(record.get("ridge"), "ridge")
    distortion = _nonnegative_float(record.get("distortion"), "distortion")
    selected_step = _nonnegative_integer(
        record.get("selected_checkpoint_step"),
        "matched selected checkpoint step",
    )
    expected_problem = _problem_record(problem, ridge=ridge)
    if identity.get("problem_sha256") != expected_problem["scientific_identity"]["sha256"]:
        raise ValueError("matched fixed-support identity does not match the problem")
    if (
        identity.get("ridge") != ridge
        or identity.get("chart_count") != chart_count
        or identity.get("distortion") != distortion
        or identity.get("selected_checkpoint_step") != selected_step
    ):
        raise ValueError("matched fixed-support scalar identity does not match metadata")
    chart_keys = _array_key_map(
        {"arrays": record.get("chart_arrays")},
        required={"mean", "scale", "standardized_centers", "centers", "assignments"},
    )
    chart_values = {name: _take(arrays, key) for name, key in chart_keys.items()}
    charts = ContextCharts(
        mean=_finite_array(chart_values["mean"], "chart mean"),
        scale=_positive_array(chart_values["scale"], "chart scale"),
        standardized_centers=_finite_array(
            chart_values["standardized_centers"],
            "standardized chart centers",
        ),
        centers=_finite_array(chart_values["centers"], "chart centers"),
        assignments=_integer_array(chart_values["assignments"], "chart assignments"),
        distortion=distortion,
    )
    if charts.chart_count != chart_count or charts.context_dim != problem.context_dim:
        raise ValueError("serialized chart router has incompatible dimensions")
    _validate_chart_router(charts, problem)
    rows = record.get("charts")
    if not isinstance(rows, list) or len(rows) != chart_count:
        raise ValueError("matched baseline must have one record per chart")
    if identity.get("q") != [row.get("q") for row in rows] or identity.get("K") != [
        row.get("K") for row in rows
    ]:
        raise ValueError("matched q/K identity does not match metadata")
    models = []
    observed_hashes = {
        f"charts.{name}": _array_sha256(values)
        for name, values in chart_values.items()
    }
    required_local = {
        "decomposition_context_center",
        "decomposition_local_decision",
        "decomposition_transformed_mean",
        "decomposition_transformed_scale",
        "decomposition_left_vectors",
        "decomposition_singular_values",
        "decomposition_cross_covariance",
        "centroid_coordinates",
        "centroid_counts",
        "local_observation_indices",
        "observed_scenarios",
        "projected_observed_coordinates",
        "cluster_assignments",
        "scenario_coordinates",
        "probability_intercepts",
        "probability_slopes",
    }
    for chart, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("chart") != chart:
            raise ValueError("matched chart records are not in canonical order")
        width = _positive_integer(row.get("q"), "q")
        support_size = _positive_integer(row.get("K"), "K")
        if support_size > width + 1 + support_cap_offset:
            raise ValueError(f"matched fixed support exceeds the declared q+{1+support_cap_offset} cap")
        local_keys = _array_key_map(row, required=required_local)
        local = {name: _take(arrays, key) for name, key in local_keys.items()}
        for name, values in local.items():
            observed_hashes[f"chart.{chart}.{name}"] = _array_sha256(values)
        decomposition = ResponseDecomposition(
            context_center=_finite_array(
                local["decomposition_context_center"],
                "decomposition context center",
            ),
            local_decision=_finite_array(
                local["decomposition_local_decision"],
                "decomposition local decision",
            ),
            transformed_mean=_finite_array(
                local["decomposition_transformed_mean"],
                "transformed mean",
            ),
            transformed_scale=_positive_array(
                local["decomposition_transformed_scale"],
                "transformed scale",
            ),
            left_vectors=_finite_array(
                local["decomposition_left_vectors"],
                "left vectors",
            ),
            singular_values=_nonnegative_array(
                local["decomposition_singular_values"],
                "singular values",
            ),
            cross_covariance=_finite_array(
                local["decomposition_cross_covariance"],
                "cross covariance",
            ),
            basis_method=str(row.get("basis_method")),
        )
        if decomposition.basis_method != "kkt":
            raise ValueError("confirmation matched support requires a KKT response basis")
        if width > decomposition.maximum_width:
            raise ValueError("matched width exceeds its response decomposition")
        _validate_decomposition(decomposition, problem, width=width)
        initialization = ObservedSupportInitialization(
            centroid_initialization=ScenarioInitialization(
                coordinates=_finite_array(
                    local["centroid_coordinates"],
                    "centroid coordinates",
                ),
                counts=_integer_array(local["centroid_counts"], "centroid counts"),
            ),
            local_observation_indices=_integer_array(
                local["local_observation_indices"],
                "local observation indices",
            ),
            observed_scenarios=_finite_array(
                local["observed_scenarios"],
                "observed scenarios",
            ),
            projected_observed_coordinates=_finite_array(
                local["projected_observed_coordinates"],
                "projected observed coordinates",
            ),
            cluster_assignments=_integer_array(
                local["cluster_assignments"],
                "cluster assignments",
            ),
        )
        if initialization.support_size != support_size:
            raise ValueError("serialized matched support size is inconsistent")
        _validate_initialization(
            initialization,
            problem,
            width=width,
            support_size=support_size,
        )
        model = ObservedSupportLocalModel(
            problem,
            decomposition,
            initialization,
            width=width,
            context_mean=charts.mean,
            context_scale=charts.scale,
            standardized_center=charts.standardized_centers[chart],
        )
        snapshot = LocalModelSnapshot(
            step=selected_step,
            scenario_coordinates=_finite_array(
                local["scenario_coordinates"],
                "scenario coordinates",
            ),
            probability_intercepts=_finite_array(
                local["probability_intercepts"],
                "probability intercepts",
            ),
            probability_slopes=_finite_array(
                local["probability_slopes"],
                "probability slopes",
            ),
            training_objective=0.0,
            mean_operational_loss=0.0,
        )
        model.load_snapshot(snapshot)
        model.eval()
        models.append(model)
    expected_hashes = identity.get("array_sha256")
    if expected_hashes != observed_hashes:
        raise ValueError("matched fixed-support array identities do not match")
    return PreparedAtlas(
        problem=problem,
        charts=charts,
        ridge=ridge,
        models=tuple(models),
    )


def _validate_live_matched_snapshot(
    model: ObservedSupportLocalModel,
    snapshot: LocalModelSnapshot,
) -> None:
    """Bind the serialized live model to its selected training checkpoint."""

    expected = {
        "scenario coordinates": snapshot.scenario_coordinates,
        "probability intercepts": snapshot.probability_intercepts,
        "probability slopes": snapshot.probability_slopes,
    }
    observed = {
        "scenario coordinates": model.scenario_coordinates.detach().cpu().numpy(),
        "probability intercepts": model.probability_intercepts.detach().cpu().numpy(),
        "probability slopes": model.probability_slopes.detach().cpu().numpy(),
    }
    for name, values in expected.items():
        if not np.array_equal(observed[name], np.asarray(values)):
            raise ValueError(
                "matched live model does not equal the declared selected "
                f"checkpoint: {name}"
            )


def _validated_identity(record: dict[str, Any], model: str) -> dict[str, Any]:
    identity = record.get("identity")
    if not isinstance(identity, dict) or identity.get("model") != model:
        raise ValueError(f"{model} identity is missing or incompatible")
    if record.get("identity_sha256") != _canonical_digest(identity):
        raise ValueError(f"{model} identity digest mismatch")
    return identity


def _identity_from_record(record: Any) -> str:
    if not isinstance(record, dict):
        raise ValueError("model record must be an object")
    identity = record.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("model identity must be an object")
    digest = _canonical_digest(identity)
    if record.get("identity_sha256") != digest:
        raise ValueError("model identity digest mismatch")
    return digest


def _array_key_map(record: dict[str, Any], *, required: set[str]) -> dict[str, str]:
    keys = record.get("arrays")
    if not isinstance(keys, dict) or set(keys) != required:
        raise ValueError("serialized baseline array-key contract is incompatible")
    if any(not isinstance(value, str) or not value for value in keys.values()):
        raise ValueError("serialized baseline array keys must be nonempty strings")
    if len(set(keys.values())) != len(keys):
        raise ValueError("serialized baseline array keys must be unique")
    return dict(keys)


def _validate_array_hashes(
    identity: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    observed = {name: _array_sha256(values) for name, values in arrays.items()}
    if identity.get("array_sha256") != observed:
        raise ValueError("serialized baseline array identities do not match")


def _problem_record(
    problem: DifferentiableStochasticProblem,
    *,
    ridge: float,
) -> dict[str, Any]:
    if hasattr(problem, "service_matrix"):
        identity = capacity_problem_identity(problem, ridge=ridge)
    elif hasattr(problem, "asset_count") and hasattr(problem, "tail_probability"):
        identity = portfolio_problem_identity(problem, ridge=ridge)
    elif hasattr(problem, "first_path_incidence") and hasattr(problem, "routing_penalty"):
        identity = network_reservation_problem_identity(problem, ridge=ridge)
    else:
        raise TypeError(
            "confirmation baseline serialization supports Capacity, Portfolio, and NET problems"
        )
    return {
        "name": problem.name,
        "class": f"{type(problem).__module__}.{type(problem).__qualname__}",
        "context_dim": int(problem.context_dim),
        "scenario_dim": int(problem.scenario_dim),
        "decision_dim": int(problem.decision_dim),
        "scientific_identity": identity,
    }


def _validate_chart_router(
    charts: ContextCharts,
    problem: DifferentiableStochasticProblem,
) -> None:
    chart_count = charts.chart_count
    context_dim = problem.context_dim
    if np.asarray(charts.mean).shape != (context_dim,):
        raise ValueError("chart mean has incompatible dimensions")
    if np.asarray(charts.scale).shape != (context_dim,) or np.any(charts.scale <= 0.0):
        raise ValueError("chart scale has incompatible dimensions")
    if np.asarray(charts.standardized_centers).shape != (chart_count, context_dim):
        raise ValueError("standardized chart centers have incompatible dimensions")
    if np.asarray(charts.centers).shape != (chart_count, context_dim):
        raise ValueError("chart centers have incompatible dimensions")
    assignments = np.asarray(charts.assignments)
    if assignments.ndim != 1 or len(assignments) == 0:
        raise ValueError("chart assignments must be a nonempty vector")
    if np.any(assignments < 0) or np.any(assignments >= chart_count):
        raise ValueError("chart assignments lie outside the router")
    reconstructed = (
        charts.mean[None, :]
        + charts.scale[None, :] * charts.standardized_centers
    )
    if not np.allclose(
        reconstructed,
        np.asarray(charts.centers),
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("chart centers do not match the saved standardization")


def _validate_decomposition(
    decomposition: ResponseDecomposition,
    problem: DifferentiableStochasticProblem,
    *,
    width: int,
) -> None:
    scenario_dim = problem.scenario_dim
    decision_dim = problem.decision_dim
    context_dim = problem.context_dim
    left = np.asarray(decomposition.left_vectors)
    if left.ndim != 2 or left.shape[0] != scenario_dim:
        raise ValueError("decomposition left vectors have incompatible dimensions")
    expected = {
        "context_center": (context_dim,),
        "local_decision": (decision_dim,),
        "transformed_mean": (scenario_dim,),
        "transformed_scale": (scenario_dim,),
        "left_vectors": (scenario_dim, left.shape[1]),
        "cross_covariance": (scenario_dim, decision_dim),
    }
    for name, shape in expected.items():
        if np.asarray(getattr(decomposition, name)).shape != shape:
            raise ValueError(f"decomposition {name} has incompatible dimensions")
    if len(decomposition.singular_values) != min(scenario_dim, decision_dim):
        raise ValueError("decomposition singular values have incompatible dimensions")
    if width > decomposition.maximum_width:
        raise ValueError("matched width exceeds its response decomposition")


def _validate_initialization(
    initialization: ObservedSupportInitialization,
    problem: DifferentiableStochasticProblem,
    *,
    width: int,
    support_size: int,
) -> None:
    centroid = initialization.centroid_initialization
    if np.asarray(centroid.coordinates).shape != (support_size, width):
        raise ValueError("matched centroid coordinates have incompatible dimensions")
    if np.asarray(centroid.counts).shape != (support_size,) or np.any(centroid.counts <= 0):
        raise ValueError("matched centroid counts have incompatible dimensions")
    if np.asarray(initialization.observed_scenarios).shape != (
        support_size,
        problem.scenario_dim,
    ):
        raise ValueError("matched observed scenarios have incompatible dimensions")
    if np.asarray(initialization.projected_observed_coordinates).shape != (
        support_size,
        width,
    ):
        raise ValueError("matched projected coordinates have incompatible dimensions")


def _put(arrays: dict[str, np.ndarray], key: str, values: Any) -> str:
    if key in arrays:
        raise ValueError(f"duplicate baseline array key: {key}")
    array = np.ascontiguousarray(np.asarray(values).copy())
    if array.dtype.kind not in "fiu" or np.any(~np.isfinite(array)):
        raise ValueError(f"baseline array {key} must be finite and numeric")
    arrays[key] = array
    return key


def _take(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return arrays.pop(key)
    except KeyError as error:
        raise ValueError(f"serialized baseline bundle is missing {key!r}") from error


def _finite_array(values: Any, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).copy())
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _positive_array(values: Any, name: str) -> np.ndarray:
    array = _finite_array(values, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def _nonnegative_array(values: Any, name: str) -> np.ndarray:
    array = _finite_array(values, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _integer_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    result = np.ascontiguousarray(array.astype(np.int64, copy=True))
    result.setflags(write=False)
    return result


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
