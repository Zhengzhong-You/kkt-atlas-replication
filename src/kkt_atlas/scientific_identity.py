"""Canonical scientific identities for problems and fitted models.

The records in this module bind a development-selected model to the numerical
problem primitives and fitted parameters that are replayed by fixed-final
timing studies.  They intentionally exclude implementation-class names and
wall-clock measurements.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from kkt_atlas.problem import Array, DifferentiableStochasticProblem

_CAPACITY_FAMILY = "smooth_capacity_allocation"
_PORTFOLIO_FAMILY = "smoothed_cvar_portfolio"
_NETWORK_RESERVATION_FAMILY = "multicommodity_network_reservation"
TRAINING_INPUT_HASH_SCHEMA = "kkt-atlas-finite-catalog-training-input-v1"
ICEO_MODEL_IDENTITY_SCHEMA = "kkt-atlas-full-support-iceo-model-identity-v1"


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_identity(values: Array) -> dict[str, Any]:
    """Return a dtype- and shape-aware SHA-256 identity for one array."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def finite_catalog_training_input_hash(data: Any) -> str:
    """Return the versioned identity of the finite-catalog training input."""

    digest = hashlib.sha256((TRAINING_INPUT_HASH_SCHEMA + "\0").encode("utf-8"))
    for name, values in (
        ("catalog", data.catalog),
        ("training_contexts", data.training.contexts),
        ("training_outcome_indices", data.training.outcome_indices),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def capacity_problem_identity(
    problem: DifferentiableStochasticProblem,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fingerprint the numerical primitives of a smooth-capacity problem.

    A grouped timing kernel and the ordinary problem receive the same identity
    whenever their numerical stochastic programs are identical.
    """

    specific: Any = problem
    required = (
        "service_matrix",
        "underage_costs",
        "overage_costs",
        "total_capacity",
        "smoothing",
        "context_dim",
        "newton_iterations",
        "newton_damping",
    )
    missing = [name for name in required if not hasattr(specific, name)]
    if missing:
        raise TypeError("capacity problem is missing numerical primitives: " + ", ".join(missing))
    primitives = {
        "family": _CAPACITY_FAMILY,
        "scalars": {
            "total_capacity": float(specific.total_capacity),
            "smoothing": float(specific.smoothing),
            "context_dim": int(specific.context_dim),
            "newton_iterations": int(specific.newton_iterations),
            "newton_damping": float(specific.newton_damping),
            "decision_regularization": float(ridge),
        },
        "arrays": {
            name: array_identity(getattr(specific, name))
            for name in (
                "service_matrix",
                "underage_costs",
                "overage_costs",
            )
        },
    }
    if specific.scenario_lower_bounds is not None:
        primitives["arrays"]["scenario_lower_bounds"] = array_identity(
            specific.scenario_lower_bounds
        )
        primitives["arrays"]["scenario_upper_bounds"] = array_identity(
            specific.scenario_upper_bounds
        )
    return {**primitives, "sha256": _canonical_json_sha256(primitives)}


def portfolio_problem_identity(
    problem: DifferentiableStochasticProblem,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fingerprint the numerical primitives of a smoothed-CVaR problem."""

    specific: Any = problem
    names = (
        "asset_count",
        "return_bound",
        "tail_probability",
        "smoothing",
        "expected_return_weight",
        "leverage_penalty",
        "context_dim",
        "gradient_iterations",
        "step_fraction",
        "training_solver",
        "newton_damping",
    )
    missing = [name for name in names if not hasattr(specific, name)]
    if missing:
        raise TypeError("portfolio problem is missing numerical primitives: " + ", ".join(missing))
    primitives = {
        "family": _PORTFOLIO_FAMILY,
        "scalars": {
            "asset_count": int(specific.asset_count),
            "return_bound": float(specific.return_bound),
            "tail_probability": float(specific.tail_probability),
            "smoothing": float(specific.smoothing),
            "expected_return_weight": float(specific.expected_return_weight),
            "leverage_penalty": float(specific.leverage_penalty),
            "context_dim": int(specific.context_dim),
            "gradient_iterations": int(specific.gradient_iterations),
            "step_fraction": float(specific.step_fraction),
            "training_solver": str(specific.training_solver),
            "newton_damping": float(specific.newton_damping),
            "decision_regularization": float(ridge),
        },
    }
    return {**primitives, "sha256": _canonical_json_sha256(primitives)}


def network_reservation_problem_identity(
    problem: DifferentiableStochasticProblem,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fingerprint the numerical primitives of a NET reservation problem."""

    specific: Any = problem
    names = (
        "first_path_incidence",
        "second_path_incidence",
        "reservation_costs",
        "directed_links",
        "commodities",
        "congestion_penalty",
        "routing_penalty",
        "reservation_upper_bound",
        "context_dim",
        "decision_iterations",
        "routing_updates_per_iteration",
        "realized_loss_routing_iterations",
        "step_fraction",
    )
    missing = [name for name in names if not hasattr(specific, name)]
    if missing:
        raise TypeError("network problem is missing numerical primitives: " + ", ".join(missing))
    def node_pairs(pairs: Any) -> Array:
        return np.asarray(
            [(*origin, *destination) for origin, destination in pairs],
            dtype=np.int64,
        )
    primitives = {
        "family": _NETWORK_RESERVATION_FAMILY,
        "scalars": {
            "congestion_penalty": float(specific.congestion_penalty),
            "routing_penalty": float(specific.routing_penalty),
            "reservation_upper_bound": float(specific.reservation_upper_bound),
            "context_dim": int(specific.context_dim),
            "decision_iterations": int(specific.decision_iterations),
            "routing_updates_per_iteration": int(specific.routing_updates_per_iteration),
            "realized_loss_routing_iterations": int(
                specific.realized_loss_routing_iterations
            ),
            "step_fraction": float(specific.step_fraction),
            "decision_regularization": float(ridge),
        },
        "arrays": {
            "first_path_incidence": array_identity(specific.first_path_incidence),
            "second_path_incidence": array_identity(specific.second_path_incidence),
            "reservation_costs": array_identity(specific.reservation_costs),
            "directed_links": array_identity(node_pairs(specific.directed_links)),
            "commodities": array_identity(node_pairs(specific.commodities)),
        },
    }
    return {**primitives, "sha256": _canonical_json_sha256(primitives)}


def _hash_named_arrays(
    *,
    scalars: Mapping[str, Any],
    arrays: Iterable[tuple[str, Array]],
) -> dict[str, Any]:
    array_records: dict[str, dict[str, Any]] = {}
    for name, values in arrays:
        if not name or name in array_records:
            raise ValueError("fingerprinted array names must be unique and nonempty")
        array_records[name] = array_identity(values)
    payload = {
        "scalars": dict(scalars),
        "arrays": array_records,
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def atlas_model_identity(
    problem: DifferentiableStochasticProblem,
    charts: Any,
    candidates: Sequence[Any],
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fingerprint all parameters defining one deployed KKT Atlas."""

    arrays: list[tuple[str, Array]] = [
        ("charts.mean", charts.mean),
        ("charts.scale", charts.scale),
        ("charts.standardized_centers", charts.standardized_centers),
        ("charts.centers", charts.centers),
        ("charts.assignments", charts.assignments),
    ]
    chart_records: list[dict[str, int]] = []
    for position, candidate in enumerate(candidates):
        chart = int(candidate.chart)
        if chart != position:
            raise ValueError("Atlas candidates must follow chart order")
        prefix = f"chart_{chart}"
        decomposition = candidate.decomposition
        checkpoint = candidate.checkpoint
        arrays.extend(
            (
                (f"{prefix}.context_center", decomposition.context_center),
                (f"{prefix}.local_decision", decomposition.local_decision),
                (f"{prefix}.transformed_mean", decomposition.transformed_mean),
                (f"{prefix}.transformed_scale", decomposition.transformed_scale),
                (f"{prefix}.left_vectors", decomposition.left_vectors),
                (f"{prefix}.singular_values", decomposition.singular_values),
                (f"{prefix}.cross_covariance", decomposition.cross_covariance),
                (
                    f"{prefix}.scenario_coordinates",
                    checkpoint.scenario_coordinates,
                ),
                (
                    f"{prefix}.probability_intercepts",
                    checkpoint.probability_intercepts,
                ),
                (
                    f"{prefix}.probability_slopes",
                    checkpoint.probability_slopes,
                ),
                (
                    f"{prefix}.decoded_scenarios",
                    candidate.make_model(problem, charts).scenarios().detach().cpu().numpy(),
                ),
            )
        )
        chart_records.append(
            {
                "chart": chart,
                "q_b": int(candidate.width),
                "K_b": int(candidate.support_size),
                "checkpoint_step": int(checkpoint.step),
            }
        )
    return _hash_named_arrays(
        scalars={
            "ridge": float(ridge),
            "chart_count": len(chart_records),
            "charts": chart_records,
        },
        arrays=arrays,
    )


def fitted_atlas_identity(atlas: Any) -> dict[str, Any]:
    """Fingerprint a validation-selected ``FittedAtlas``."""

    return atlas_model_identity(
        atlas.problem,
        atlas.charts,
        tuple(selected.candidate for selected in atlas.local_models),
        ridge=atlas.ridge,
    )


def replayed_atlas_identity(
    problem: DifferentiableStochasticProblem,
    replay: Any,
) -> dict[str, Any]:
    """Fingerprint the unique model returned by a frozen replay."""

    partition = replay.library.partitions[0]
    ridge_library = partition.ridge_libraries[0]
    candidates = tuple(items[0] for items in ridge_library.local_candidates)
    if any(len(items) != 1 for items in ridge_library.local_candidates):
        raise ValueError("frozen replay must contain exactly one model per chart")
    return atlas_model_identity(
        problem,
        partition.charts,
        candidates,
        ridge=ridge_library.ridge,
    )


def iceo_snapshot_identity(snapshot: Any) -> dict[str, Any]:
    """Fingerprint one selected full-support ICEO parameter snapshot."""

    parameters = tuple(snapshot.parameters)
    return _hash_named_arrays(
        scalars={"checkpoint_step": int(snapshot.step)},
        arrays=((f"parameter.{name}", values) for name, values in parameters),
    )


def iceo_model_identity(model: Any, snapshot: Any, *, ridge: float) -> dict[str, Any]:
    """Fingerprint every inference-relevant part of one selected ICEO model.

    The selected checkpoint alone is insufficient: inference also depends on
    the reference catalog, context normalization, architecture, decision
    regularization, and the live parameter/buffer state.
    """

    arrays: list[tuple[str, Array]] = [
        (f"selected_checkpoint.parameter.{name}", values)
        for name, values in snapshot.parameters
    ]
    arrays.extend(
        (
            f"inference_state.{name}",
            value.detach().cpu().numpy().copy(),
        )
        for name, value in model.state_dict().items()
    )
    return _hash_named_arrays(
        scalars={
            "identity_schema": ICEO_MODEL_IDENTITY_SCHEMA,
            "method": "full_support_iceo",
            "problem": {
                "name": str(model.problem.name),
                "context_dim": int(model.problem.context_dim),
                "scenario_dim": int(model.problem.scenario_dim),
                "decision_dim": int(model.problem.decision_dim),
            },
            "ridge": float(ridge),
            "selected_checkpoint_step": int(snapshot.step),
            "architecture": {
                "hidden_width": int(model.model_config.hidden_width),
                "hidden_layers": int(model.model_config.hidden_layers),
                "initialization_seed": int(model.model_config.seed),
                "activation": "ReLU",
                "probability_link": "softmax",
                "context_dim": int(model.context_dim),
                "support_size": int(model.support_size),
            },
        },
        arrays=arrays,
    )


def fitted_iceo_identity(fitted: Any) -> dict[str, Any]:
    """Fingerprint a validation-selected full-support ICEO model."""

    snapshot = fitted.training_result.checkpoint(fitted.selected_step)
    return iceo_model_identity(
        fitted.model,
        snapshot,
        ridge=fitted.ridge,
    )
