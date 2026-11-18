"""Versioned protocol and job identities for the symmetric KKT Atlas campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_V3 = "kkt-atlas-symmetric-campaign-protocol-v3"
SCHEMA_V4 = "kkt-atlas-symmetric-campaign-protocol-v4"
SCHEMA_V5 = "kkt-atlas-symmetric-campaign-protocol-v5"
APPLICATIONS = ("capacity", "portfolio", "network")
PRIMARY_METHODS = ("kkt_atlas", "iceo_a", "sof", "knn")
ANCHOR_ABLATIONS = ("matched_observed_support", "fixed_b1")
FORMAL_SEEDS = (241001, 241002, 241003)
DEVELOPMENT_SEEDS = (240991,)
CONTROLLED_SEEDS = (241011, 241012, 241013)


@dataclass(frozen=True)
class CampaignCell:
    application: str
    setting: str
    axis: str
    support_size: int
    scenario_dim: int
    training_size: int
    validation_size: int
    evaluation_size: int

    @property
    def key(self) -> str:
        return f"{self.application}__{self.setting}"


@dataclass(frozen=True)
class CampaignJob:
    kind: str
    application: str
    setting: str
    seed: int
    method: str
    eta: float
    dependencies: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        eta = str(self.eta).replace(".", "p")
        return (
            f"{self.kind}__{self.application}__{self.setting}__"
            f"seed_{self.seed}__{self.method}__eta_{eta}"
        )


@dataclass(frozen=True)
class SymmetricCampaignProtocol:
    path: Path
    raw: dict[str, Any]
    sha256: str
    formal_horizon_multiplier: int | None

    @property
    def is_frozen(self) -> bool:
        return self.raw["phase"] == "formal" and self.formal_horizon_multiplier in {1, 2}

    def application(self, name: str) -> dict[str, Any]:
        if name not in APPLICATIONS:
            raise KeyError(name)
        return self.raw["applications"][name]

    def eta(self, application: str) -> float:
        if application == "controlled":
            return float(
                self.raw.get("regularization", {})
                .get("controlled", {})
                .get("raw_eta", self.raw["training"].get("eta", 0.1))
            )
        self.application(application)
        return float(
            self.raw.get("regularization", {})
            .get(application, {})
            .get("raw_eta", self.raw["training"].get("eta", 0.1))
        )

    def cells(self) -> tuple[CampaignCell, ...]:
        result: list[CampaignCell] = []
        for application in APPLICATIONS:
            spec = self.application(application)
            anchor = spec["anchor"]
            values = [
                ("anchor", "anchor", anchor),
                (
                    "m_high",
                    "m",
                    {**anchor, "m": spec["quality_stresses"]["m_high"]},
                ),
                (
                    "n_low",
                    "n",
                    {**anchor, "n": spec["quality_stresses"]["n_low"]},
                ),
                (
                    "k_timing",
                    "K",
                    {**anchor, "K": spec["timing_support_size"]},
                ),
            ]
            for setting, axis, value in values:
                result.append(
                    CampaignCell(
                        application=application,
                        setting=setting,
                        axis=axis,
                        support_size=int(value["K"]),
                        scenario_dim=int(value["m"]),
                        training_size=int(value["n"]),
                        validation_size=int(spec["validation_size"]),
                        evaluation_size=int(spec["evaluation_size"]),
                    )
                )
        return tuple(result)

    def cell(self, application: str, setting: str) -> CampaignCell:
        matches = [cell for cell in self.cells() if cell.application == application and cell.setting == setting]
        if len(matches) != 1:
            raise KeyError((application, setting))
        return matches[0]

    def jobs(self) -> tuple[CampaignJob, ...]:
        jobs: list[CampaignJob] = []
        cells = self.cells()
        setting_indices = {
            (application, cell.setting): position
            for application in APPLICATIONS
            for position, cell in enumerate(
                candidate for candidate in cells if candidate.application == application
            )
        }
        for cell in cells:
            eta = self.eta(cell.application)
            application_index = APPLICATIONS.index(cell.application)
            setting_index = setting_indices[(cell.application, cell.setting)]
            seeds = FORMAL_SEEDS if cell.setting == "anchor" else FORMAL_SEEDS[:1]
            methods = (
                ("kkt_atlas", "iceo_a")
                if cell.setting == "k_timing"
                else PRIMARY_METHODS
            )
            kind = "timing_stress" if cell.setting == "k_timing" else "primary"
            for seed_index, seed in enumerate(seeds):
                ordered = cyclic_primary_methods(
                    application_index + setting_index + seed_index
                )
                for method in (item for item in ordered if item in methods):
                    jobs.append(
                        CampaignJob(
                            kind,
                            cell.application,
                            cell.setting,
                            seed,
                            method,
                            eta,
                        )
                    )
        for application in APPLICATIONS:
            eta = self.eta(application)
            for seed in FORMAL_SEEDS:
                atlas = CampaignJob("primary", application, "anchor", seed, "kkt_atlas", eta).key
                for method in ANCHOR_ABLATIONS:
                    dependencies = (atlas,) if method == "matched_observed_support" else ()
                    jobs.append(
                        CampaignJob("ablation", application, "anchor", seed, method, eta, dependencies)
                    )
        for seed in CONTROLLED_SEEDS:
            for noise in ("low", "high"):
                for basis in ("cross", "pca"):
                    jobs.append(
                        CampaignJob("controlled", "controlled", noise, seed, basis, self.eta("controlled"))
                    )
        keys = [job.key for job in jobs]
        if len(jobs) != 96 or len(set(keys)) != len(keys):
            raise RuntimeError("symmetric campaign job enumeration is invalid")
        declared = set(keys)
        for job in jobs:
            if not set(job.dependencies).issubset(declared):
                raise RuntimeError(f"unknown dependency for {job.key}")
        return tuple(jobs)

    def require_frozen(self) -> None:
        if not self.is_frozen:
            raise RuntimeError("formal execution requires a frozen horizon multiplier and formal phase")


def derive_seed64(base_seed: int, *tokens: str | int) -> int:
    encoded = [str(base_seed), *(str(token) for token in tokens)]
    if base_seed < 0 or any("|" in token for token in encoded):
        raise ValueError("seed inputs must be nonnegative and cannot contain '|'")
    return int.from_bytes(hashlib.sha256("|".join(encoded).encode()).digest()[:8], "little")


def cyclic_primary_methods(offset: int) -> tuple[str, ...]:
    shift = offset % len(PRIMARY_METHODS)
    return PRIMARY_METHODS[shift:] + PRIMARY_METHODS[:shift]


def consumer_seed32(seed64: int, consumer: str, counter: int = 0) -> int:
    if seed64 < 0 or counter < 0 or "|" in consumer:
        raise ValueError("invalid consumer seed input")
    payload = f"{seed64}|{consumer}|{counter}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def collision_free_seed_manifest(
    namespaces: tuple[tuple[int, tuple[str | int, ...], str], ...],
) -> tuple[dict[str, Any], ...]:
    used: set[int] = set()
    records: list[dict[str, Any]] = []
    for base_seed, tokens, consumer in namespaces:
        master = derive_seed64(base_seed, *tokens)
        counter = 0
        value = consumer_seed32(master, consumer, counter)
        while value in used:
            counter += 1
            value = consumer_seed32(master, consumer, counter)
        used.add(value)
        records.append(
            {
                "base_seed": base_seed,
                "tokens": [str(token) for token in tokens],
                "consumer": consumer,
                "seed64": master,
                "seed32": value,
                "collision_counter": counter,
            }
        )
    return tuple(records)


def load_symmetric_campaign_protocol(path: Path | None = None) -> SymmetricCampaignProtocol:
    source = path or (
        Path(__file__).resolve().parents[2] / "data" / "configs" / "synthetic_protocol.json"
    )
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    _validate(payload)
    raw = Path(source).read_bytes()
    protocol = SymmetricCampaignProtocol(
        path=Path(source).resolve(),
        raw=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
        formal_horizon_multiplier=payload["training"]["formal_horizon_multiplier"],
    )
    if len(protocol.cells()) != 12:
        raise ValueError("protocol must define 12 application settings")
    _ = protocol.jobs()
    return protocol


def _validate(payload: dict[str, Any]) -> None:
    schema = payload.get("schema")
    if schema not in {SCHEMA_V3, SCHEMA_V4, SCHEMA_V5}:
        raise ValueError("wrong symmetric campaign schema")
    if payload.get("phase") not in {"development", "formal"}:
        raise ValueError("phase must be development or formal")
    if tuple(payload.get("applications", {})) != APPLICATIONS:
        raise ValueError("applications must be capacity, portfolio, network in order")
    if tuple(payload["seeds"]["formal"]) != FORMAL_SEEDS:
        raise ValueError("formal seed block differs")
    if tuple(payload["seeds"]["development"]) != DEVELOPMENT_SEEDS:
        raise ValueError("development seed block differs")
    if tuple(payload["controlled"]["seeds"]) != CONTROLLED_SEEDS:
        raise ValueError("controlled seed block differs")
    if tuple(payload["methods"]) != PRIMARY_METHODS:
        raise ValueError("primary method order differs")
    if tuple(payload["anchor_ablations"]) != ANCHOR_ABLATIONS:
        raise ValueError("ablation order differs")
    training = payload["training"]
    if training["global_batch_size"] != 64 or training["base_passes"] != 32:
        raise ValueError("training constants differ")
    if schema == SCHEMA_V3:
        if training.get("eta") != 0.1:
            raise ValueError("legacy training eta differs")
    else:
        if "eta" in training:
            raise ValueError("v4 uses application-specific regularization")
        regularization = payload.get("regularization", {})
        if (
            regularization.get("capacity")
            != {
                "raw_eta": 0.0,
                "natural_curvature": True,
                "scenario_domain": "catalog_coordinate_envelope",
            }
            or regularization.get("portfolio") != {"raw_eta": 1e-6}
            or regularization.get("network")
            != {
                "raw_eta": 1e-6,
                "normalized_eta": 1.08e-4,
                "decision_dimension": 48,
                "coordinate_scale": 1.5,
                "normalized_penalty": "eta_bar/(2*d)*||(z-z0)/1.5||_2^2",
            }
            or regularization.get("controlled") != {"raw_eta": 0.0}
        ):
            raise ValueError("application-specific regularization differs")
    if training["formal_horizon_multiplier"] not in {None, 1, 2}:
        raise ValueError("formal horizon multiplier must be null, one, or two")
    expected_checkpoints = (
        [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
        if schema == SCHEMA_V5
        else [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    if training["checkpoints"] != expected_checkpoints:
        raise ValueError("training checkpoint path differs")
    structure = payload["structure"]
    if (
        tuple(structure["chart_counts"]) != (1, 2, 4)
        or structure["chart_count_rule"] != "minimum_operational_loss"
        or structure["fold_count"] != 5
        or structure["minimum_chart_size"] != 50
        or structure["spectral_retained_ratio"] != 0.9999
        or structure["minimum_samples_per_parameter"] != 3.0
        or not structure["accept_intercept_only_fallback"]
    ):
        raise ValueError("automatic-structure protocol differs")
    if payload["iceo_a"]["microbatch_size"] != 32:
        raise ValueError("ICEO-A microbatch protocol differs")
    sof = payload["sof"]
    if (
        sof["tree_count"] != 100
        or sof["honesty"]
        or sof["minimum_child_fraction"] != 0.2
        or sof["terminal_node_size"] != 10
        or sof["maximum_depth"] != 100
        or sof["split_rule"] != "approximate_risk"
    ):
        raise ValueError("SOF protocol differs")
    if payload["knn"]["candidate_rule"] != "powers_of_two_plus_n":
        raise ValueError("kNN protocol differs")
    solver = payload["formal_solver"]
    common_solver_ok = (
        solver["stationarity_tolerance"] != 1e-6
        or solver["equality_tolerance"] != 1e-6
        or solver["bound_tolerance"] != 1e-6
        or solver["dual_tolerance"] != 1e-6
        or solver["complementarity_tolerance"] != 1e-6
        or solver["network_residual_tolerance"] != 1e-6
        or solver["network_inner_maximum_iterations"] != 5_000
        or solver["network_outer_maximum_iterations"] != 5_000
    )
    algorithms_ok = (
        solver["network_recourse_algorithm"] == "accelerated_projected_gradient"
        and solver["network_outer_algorithm"] == "accelerated_projected_gradient"
        and solver["network_recourse_warm_start_across_outer_iterations"]
        if schema == SCHEMA_V3
        else solver["network_recourse_algorithm"] == "exact_joint_convex_qp"
        and solver["network_outer_algorithm"] == "projected_residual_refinement"
        and not solver["network_recourse_warm_start_across_outer_iterations"]
    )
    if common_solver_ok or not algorithms_ok:
        raise ValueError("formal-solver tolerances differ")
    validation_size = payload.get("development_validation_size", 200)
    if "development_validation_size" in payload and (
        payload["phase"] != "development" or type(validation_size) is not int or validation_size != 1000
    ):
        raise ValueError("development validation override must be 1000 and development-only")
    training_size = payload.get("development_training_size", 1600)
    if "development_training_size" in payload and (
        payload["phase"] != "development" or type(training_size) is not int or training_size not in (3200, 6400, 12800)
    ):
        raise ValueError("development training override must be 3200, 6400 or 12800 and development-only")
    for name in APPLICATIONS:
        spec = payload["applications"][name]
        if spec["master_support_size"] != 1_000 or spec["master_scenario_dim"] != 128:
            raise ValueError(f"{name} master dimensions differ")
        if spec["validation_size"] != validation_size or spec["evaluation_size"] != 200:
            raise ValueError(f"{name} evaluation split differs")
        anchor = spec["anchor"]
        if (
            anchor["K"] != 500
            or anchor["n"] != training_size
            or spec["quality_stresses"]
            != {"m_high": anchor["m"] * 2, "n_low": 800}
            or spec["timing_support_size"] != 1_000
            or spec["evaluation_draws_per_context"] != 50
        ):
            raise ValueError(f"{name} lean design differs")
    portfolio_problem = payload["applications"]["portfolio"]["problem"]
    if schema == SCHEMA_V5:
        if (
            portfolio_problem.get("gradient_iterations") != 24
            or portfolio_problem.get("training_solver") != "damped_newton"
            or portfolio_problem.get("newton_damping") != 0.5
        ):
            raise ValueError("PORT differentiable Newton protocol differs")
    elif portfolio_problem.get("gradient_iterations") != 80:
        raise ValueError("legacy PORT training map differs")
    network_dgp = payload["applications"]["network"]["dgp"]
    if (
        network_dgp["baseline_demand_minimum"] != 0.04
        or network_dgp["baseline_demand_maximum"] != 0.06
    ):
        raise ValueError("NET demand scale differs")
    execution = payload["execution"]
    wall_hours = payload.get("development_job_wall_hours", 12)
    if "development_job_wall_hours" in payload and (
        payload["phase"] != "development" or payload.get("paper_evidence") is not False
        or type(wall_hours) is not int or wall_hours != 36
    ):
        raise ValueError("development job-wall override must be36 hours and development-only")
    if execution["cores_per_fit"] != 8 or execution["context_workers"] != 1:
        raise ValueError("common execution profile differs")
    if (
        execution["max_parallel_fits_windows"] != 24
        or execution["anchor_timing_workers_windows"] != 24
        or execution["training_time_primary"] != "process_tree_cpu_seconds"
        or execution["wall_time_role"] != "secondary_descriptive"
        or execution["formal_makespan_limit_hours"] != 48
        or execution["maximum_job_wall_hours"] != wall_hours
        or execution["evaluation_block_size"] != 10
        or execution["deployment_repetitions"] != 5
    ):
        raise ValueError("revised Windows execution protocol differs")
