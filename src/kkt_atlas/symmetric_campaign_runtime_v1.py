"""Load one standalone campaign instance and its exact downstream problem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kkt_atlas.case_studies.network_reservation import NetworkReservationProblem
from kkt_atlas.case_studies.projected_quadratic import ProjectedQuadraticProblem
from kkt_atlas.case_studies.smooth_capacity_allocation import (
    SmoothCapacityAllocationProblem,
)
from kkt_atlas.case_studies.smoothed_cvar_portfolio import (
    SmoothedCVaRPortfolioProblem,
)
from kkt_atlas.symmetric_campaign_protocol_v1 import SymmetricCampaignProtocol
from kkt_atlas.symmetric_instances_v1 import StandaloneInstance, read_standalone_instance


@dataclass(frozen=True)
class CampaignRuntimeInstance:
    stored: StandaloneInstance
    problem: Any


def load_campaign_runtime_instance(
    path: Path,
    protocol: SymmetricCampaignProtocol,
) -> CampaignRuntimeInstance:
    stored = read_standalone_instance(path)
    if stored.manifest["protocol_sha256"] != protocol.sha256:
        raise ValueError("standalone instance belongs to a different protocol")
    return CampaignRuntimeInstance(stored, campaign_problem(protocol, stored))


def campaign_problem(
    protocol: SymmetricCampaignProtocol,
    stored: StandaloneInstance,
) -> Any:
    spec = stored.manifest["spec"]
    application = spec.get("application")
    if spec.get("kind") == "controlled":
        return ProjectedQuadraticProblem(
            task_basis=stored.extra_arrays["task_basis"],
            context_dim=stored.fitting.context_dim,
        )
    if spec.get("kind") != "application" or application not in {
        "capacity",
        "portfolio",
        "network",
    }:
        raise ValueError("unsupported campaign instance spec")
    cell = protocol.cell(application, spec["setting"])
    if (
        int(spec["seed"]) != int(stored.fitting.metadata["seed"])
        or cell.support_size != stored.fitting.support_size
        or cell.scenario_dim != stored.fitting.scenario_dim
        or cell.training_size != len(stored.fitting.training.contexts)
    ):
        raise ValueError("standalone instance dimensions differ from the protocol cell")
    app = protocol.application(application)
    if application == "capacity":
        problem = app["problem"]
        return SmoothCapacityAllocationProblem(
            service_matrix=stored.extra_arrays["service_matrix"],
            underage_costs=stored.extra_arrays["underage_costs"],
            overage_costs=stored.extra_arrays["overage_costs"],
            total_capacity=problem["base_capacity"] * min(8, cell.scenario_dim),
            scenario_lower_bounds=stored.fitting.catalog.min(axis=0),
            scenario_upper_bounds=stored.fitting.catalog.max(axis=0),
            smoothing=problem["smoothing"],
            context_dim=app["context_dim"],
            newton_iterations=problem["newton_iterations"],
            newton_damping=problem["newton_damping"],
        )
    if application == "portfolio":
        problem = app["problem"]
        return SmoothedCVaRPortfolioProblem(
            asset_count=cell.scenario_dim,
            context_dim=app["context_dim"],
            return_bound=problem["return_bound"],
            tail_probability=problem["tail_probability"],
            smoothing=problem["smoothing"],
            expected_return_weight=problem["expected_return_weight"],
            leverage_penalty=problem["leverage_penalty"],
            gradient_iterations=problem["gradient_iterations"],
            step_fraction=problem["step_fraction"],
            training_solver=problem.get("training_solver", "accelerated_gradient"),
            newton_damping=problem.get("newton_damping", 0.5),
        )
    problem = app["problem"]
    return NetworkReservationProblem(
        first_path_incidence=stored.extra_arrays["first_path_incidence"],
        second_path_incidence=stored.extra_arrays["second_path_incidence"],
        directed_links=_links(stored.extra_arrays["directed_links"]),
        commodities=_links(stored.extra_arrays["commodities"]),
        reservation_costs=stored.extra_arrays["reservation_costs"],
        congestion_penalty=problem["congestion_penalty"],
        routing_penalty=problem["routing_regularization"],
        reservation_upper_bound=problem["reservation_upper_bound"],
        context_dim=app["context_dim"],
        decision_iterations=problem["decision_iterations"],
        routing_updates_per_iteration=problem["routing_updates_per_iteration"],
        realized_loss_routing_iterations=problem["realized_loss_routing_iterations"],
        step_fraction=problem["step_fraction"],
    )


def _links(values: np.ndarray) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("stored node-pair array must have four columns")
    return tuple(
        ((int(row[0]), int(row[1])), (int(row[2]), int(row[3])))
        for row in array
    )
