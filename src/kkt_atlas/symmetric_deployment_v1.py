"""Deployment timing and matched-number-of-terms controls."""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kkt_atlas.confirmation_data_v1 import FittingCatalogData
from kkt_atlas.symmetric_evaluation_v1 import (
    SolverResidualSummary,
    solve_predicted_laws,
)
from kkt_atlas.symmetric_laws_v1 import FiniteLaw, predict_method_laws

_COMPARATORS = ("iceo_a", "sof", "knn")


@dataclass(frozen=True)
class _PaddedLaw:
    scenarios: np.ndarray
    probabilities: np.ndarray


def run_deployment_timing(
    models: Mapping[str, Any],
    problem: object,
    fitting: FittingCatalogData,
    contexts: np.ndarray,
    *,
    eta: float,
    batch_sizes: Sequence[int] = (1, 128),
    warmups: int = 2,
    repetitions: int = 10,
    context_workers: int = 8,
    include_matched_controls: bool = True,
) -> dict[str, Any]:
    """Time prediction plus the common formal solver for preloaded models."""

    raw_contexts = np.ascontiguousarray(np.asarray(contexts, dtype=np.float64))
    if raw_contexts.ndim != 2 or raw_contexts.shape[1] != problem.context_dim:
        raise ValueError("deployment contexts have the wrong shape")
    sizes = tuple(int(value) for value in batch_sizes)
    if not sizes or any(value < 1 or value > len(raw_contexts) for value in sizes):
        raise ValueError("batch sizes must lie within the supplied contexts")
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    if context_workers < 1:
        raise ValueError("context_workers must be positive")
    eta_value = float(eta)
    if not math.isfinite(eta_value) or eta_value < 0.0:
        raise ValueError("eta must be finite and nonnegative")

    method_records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for batch_size in sizes:
        raw_batch = np.array(raw_contexts[:batch_size], copy=True, order="C")
        for method, model in models.items():
            method_records.append(
                {
                    "method": method,
                    "batch_size": batch_size,
                    **_measure(
                        lambda model=model, method=method, raw_batch=raw_batch: _predict_and_solve(
                            model,
                            method,
                            problem,
                            fitting,
                            raw_batch,
                            eta_value,
                            context_workers,
                        ),
                        warmups,
                        repetitions,
                    ),
                }
            )
        atlas = models.get("kkt_atlas")
        if include_matched_controls and atlas is not None:
            for comparator in _COMPARATORS:
                if comparator in models:
                    controls.append(
                        _matched_control(
                            atlas,
                            models[comparator],
                            comparator,
                            problem,
                            fitting,
                            raw_batch,
                            eta_value,
                            warmups,
                            repetitions,
                            context_workers,
                        )
                    )
    return {
        "schema": "kkt-atlas-symmetric-deployment-timing-v1",
        "eta": eta_value,
        "batch_sizes": list(sizes),
        "warmups": warmups,
        "repetitions": repetitions,
        "context_workers": context_workers,
        "methods": method_records,
        "matched_number_of_terms": controls,
    }


def pad_atlas_law(law: FiniteLaw, target_terms: int) -> _PaddedLaw:
    """Split each atom by quotient/remainder without deduplication."""

    support_size = len(law.probabilities)
    if not isinstance(target_terms, int) or isinstance(target_terms, bool):
        raise TypeError("target_terms must be an integer")
    if target_terms < support_size:
        raise ValueError("target_terms cannot be smaller than the Atlas support")
    quotient, remainder = divmod(target_terms, support_size)
    copies = np.full(support_size, quotient, dtype=np.int64)
    copies[:remainder] += 1
    scenarios = np.repeat(law.scenarios, copies, axis=0)
    probabilities = np.concatenate(
        [
            np.full(int(count), law.probabilities[index] / count)
            for index, count in enumerate(copies)
        ]
    )
    if len(scenarios) != target_terms or not np.isclose(
        probabilities.sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("padded Atlas law is invalid")
    return _PaddedLaw(
        np.ascontiguousarray(scenarios),
        np.ascontiguousarray(probabilities),
    )


def _predict_and_solve(
    model: Any,
    method: str,
    problem: object,
    fitting: FittingCatalogData,
    raw_contexts: np.ndarray,
    eta: float,
    context_workers: int,
) -> None:
    laws = predict_method_laws(model, method, raw_contexts, fitting)
    _, residuals = solve_predicted_laws(
        problem,
        raw_contexts,
        laws,
        eta=eta,
        context_workers=min(context_workers, len(raw_contexts)),
    )
    _require_residual_gate(residuals)


def _matched_control(
    atlas_model: Any,
    comparator_model: Any,
    comparator: str,
    problem: object,
    fitting: FittingCatalogData,
    contexts: np.ndarray,
    eta: float,
    warmups: int,
    repetitions: int,
    context_workers: int,
) -> dict[str, Any]:
    atlas_laws = predict_method_laws(atlas_model, "kkt_atlas", contexts, fitting)
    comparator_laws = predict_method_laws(
        comparator_model,
        comparator,
        contexts,
        fitting,
    )
    available_contexts: list[np.ndarray] = []
    original_laws: list[FiniteLaw] = []
    padded_laws: list[_PaddedLaw] = []
    unavailable = Counter()
    maximum_objective_difference = 0.0
    maximum_objective_tolerance_ratio = 0.0
    maximum_decision_difference = 0.0
    for context, atlas_law, comparator_law in zip(
        contexts,
        atlas_laws,
        comparator_laws,
        strict=True,
    ):
        if len(comparator_law.probabilities) < len(atlas_law.probabilities):
            unavailable["comparator_support_smaller"] += 1
            continue
        padded = pad_atlas_law(atlas_law, len(comparator_law.probabilities))
        gate = _equivalence_gate(problem, context, atlas_law, padded, eta)
        if not gate["passed"]:
            unavailable["equivalence_gate_failed"] += 1
            continue
        available_contexts.append(context)
        original_laws.append(atlas_law)
        padded_laws.append(padded)
        maximum_objective_difference = max(
            maximum_objective_difference,
            gate["objective_difference"],
        )
        maximum_objective_tolerance_ratio = max(
            maximum_objective_tolerance_ratio,
            gate["objective_tolerance_ratio"],
        )
        maximum_decision_difference = max(
            maximum_decision_difference,
            gate["decision_difference"],
        )
    requested = len(contexts)
    available = len(available_contexts)
    base = {
        "comparator": comparator,
        "batch_size": requested,
        "available_contexts": available,
        "coverage": available / requested,
        "unavailable_reasons": dict(sorted(unavailable.items())),
        "maximum_objective_difference": maximum_objective_difference,
        "maximum_objective_tolerance_ratio": (
            maximum_objective_tolerance_ratio
        ),
        "maximum_decision_difference": maximum_decision_difference,
    }
    if available == 0:
        return {**base, "original": None, "padded": None}
    selected_contexts = np.ascontiguousarray(np.vstack(available_contexts))

    def solve(laws: Sequence[Any]) -> None:
        _, residuals = solve_predicted_laws(
            problem,
            selected_contexts,
            laws,
            eta=eta,
            context_workers=min(context_workers, len(selected_contexts)),
        )
        _require_residual_gate(residuals)

    return {
        **base,
        "original": _measure(
            lambda: solve(original_laws),
            warmups,
            repetitions,
        ),
        "padded": _measure(
            lambda: solve(padded_laws),
            warmups,
            repetitions,
        ),
    }


def _equivalence_gate(
    problem: object,
    context: np.ndarray,
    original: FiniteLaw,
    padded: _PaddedLaw,
    eta: float,
) -> dict[str, float | bool]:
    one_context = np.ascontiguousarray(context[None, :])
    original_decision, original_residuals = solve_predicted_laws(
        problem,
        one_context,
        (original,),
        eta=eta,
    )
    padded_decision, padded_residuals = solve_predicted_laws(
        problem,
        one_context,
        (padded,),
        eta=eta,
    )
    original_objective = _regularized_objective(
        problem,
        context,
        original,
        original_decision[0],
        eta,
    )
    padded_objective = _regularized_objective(
        problem,
        context,
        padded,
        padded_decision[0],
        eta,
    )
    objective_difference = abs(original_objective - padded_objective)
    objective_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(original_objective),
        abs(padded_objective),
    )
    decision_difference = float(
        np.max(np.abs(original_decision[0] - padded_decision[0]), initial=0.0)
    )
    passed = (
        objective_difference <= objective_tolerance
        and decision_difference <= 1e-10
        and _residuals_pass(original_residuals)
        and _residuals_pass(padded_residuals)
    )
    return {
        "passed": passed,
        "objective_difference": objective_difference,
        "objective_tolerance": objective_tolerance,
        "objective_tolerance_ratio": (
            objective_difference / objective_tolerance
        ),
        "decision_difference": decision_difference,
    }


def _regularized_objective(
    problem: object,
    context: np.ndarray,
    law: Any,
    decision: np.ndarray,
    eta: float,
) -> float:
    derivatives = getattr(problem, "reference_objective_derivatives_numpy", None)
    if callable(derivatives):
        return float(
            derivatives(
                decision,
                context,
                law.scenarios,
                law.probabilities,
                ridge=eta,
            )[1]
        )
    scenario_losses = getattr(problem, "formal_scenario_losses_numpy", None)
    center_method = getattr(problem, "reference_center_numpy", None)
    if not callable(scenario_losses) or not callable(center_method):
        raise TypeError("problem cannot evaluate a formal regularized objective")
    losses, _ = scenario_losses(
        decision,
        law.scenarios,
        tolerance=1e-10,
        maximum_iterations=10_000,
    )
    center = np.asarray(center_method(context), dtype=np.float64)
    penalty = 0.5 * eta * float((decision - center) @ (decision - center))
    return float(law.probabilities @ losses) + penalty


def _measure(
    operation: Callable[[], None],
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        operation()
    records = []
    for repetition in range(repetitions):
        wall_started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        operation()
        cpu_seconds = (time.process_time_ns() - cpu_started) / 1e9
        wall_seconds = (time.perf_counter_ns() - wall_started) / 1e9
        records.append(
            {
                "repetition": repetition,
                "wall_seconds": wall_seconds,
                "self_cpu_seconds": cpu_seconds,
            }
        )
    return {
        "median_wall_seconds": statistics.median(
            record["wall_seconds"] for record in records
        ),
        "median_self_cpu_seconds": statistics.median(
            record["self_cpu_seconds"] for record in records
        ),
        "repetitions": records,
    }


def _require_residual_gate(residuals: SolverResidualSummary) -> None:
    if not _residuals_pass(residuals):
        raise RuntimeError("formal deployment solve failed its residual gate")


def _residuals_pass(residuals: SolverResidualSummary) -> bool:
    return bool(
        residuals.maximum_stationarity <= 1e-6
        and residuals.maximum_equality <= 1e-6
        and residuals.maximum_lower_bound_violation <= 1e-6
        and residuals.maximum_dual_violation <= 1e-6
        and residuals.maximum_complementarity <= 1e-6
        and residuals.maximum_reservation <= 1e-6
        and residuals.maximum_recourse <= 1e-6
    )
