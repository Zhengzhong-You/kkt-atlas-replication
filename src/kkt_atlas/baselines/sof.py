"""Serial stochastic optimization forests with approximate-solution splits."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

from kkt_atlas.problem import Array, DifferentiableStochasticProblem, matrix
from kkt_atlas.scientific_identity import (
    array_identity,
    capacity_problem_identity,
    network_reservation_problem_identity,
    portfolio_problem_identity,
)

_SERIALIZATION_SCHEMA = "kkt-atlas-sof-model-v1"

@dataclass(frozen=True)
class SOFConfig:
    tree_count: int = 500
    max_depth: int = 100
    terminal_node_size: int = 10
    minimum_child_fraction: float = 0.2
    ridge: float = 0.1
    split_rule: str = "approximate_solution"
    active_tolerance: float = 1e-10
    feasibility_tolerance: float = 1e-10
    seed: int = 0
    worker_count: int = 1

    def __post_init__(self) -> None:
        if self.tree_count <= 0:
            raise ValueError("tree_count must be positive")
        if not 0 <= self.max_depth <= 100:
            raise ValueError("max_depth must lie between zero and 100")
        if self.terminal_node_size < 1:
            raise ValueError("terminal_node_size must be positive")
        if not 0.0 < self.minimum_child_fraction < 0.5:
            raise ValueError("minimum_child_fraction must lie in (0,0.5)")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be finite and nonnegative")
        if not math.isfinite(self.active_tolerance) or self.active_tolerance <= 0.0:
            raise ValueError("active_tolerance must be finite and positive")
        if not math.isfinite(self.feasibility_tolerance) or self.feasibility_tolerance <= 0.0:
            raise ValueError("feasibility_tolerance must be finite and positive")
        if self.split_rule not in {
            "approximate_solution",
            "approximate_risk",
        }:
            raise ValueError("split_rule is unsupported")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")


@dataclass(frozen=True)
class SOFNode:
    feature: int | None
    threshold: float | None
    bootstrap_indices: Array | None
    left: SOFNode | None = None
    right: SOFNode | None = None

    @classmethod
    def leaf(cls, indices: Array) -> SOFNode:
        values = np.asarray(indices, dtype=np.int64).reshape(-1).copy()
        if len(values) == 0:
            raise ValueError("a SOF leaf cannot be empty")
        values.setflags(write=False)
        return cls(None, None, values)

    @classmethod
    def branch(
        cls,
        feature: int,
        threshold: float,
        left: SOFNode,
        right: SOFNode,
    ) -> SOFNode:
        return cls(int(feature), float(threshold), None, left, right)

    @property
    def is_leaf(self) -> bool:
        return self.bootstrap_indices is not None

    def leaf_for(self, context: Array) -> Array:
        node = self
        while not node.is_leaf:
            if node.feature is None or node.threshold is None:
                raise RuntimeError("invalid SOF branch")
            child = node.left if context[node.feature] <= node.threshold else node.right
            if child is None:
                raise RuntimeError("SOF branch is missing a child")
            node = child
        assert node.bootstrap_indices is not None
        return node.bootstrap_indices


@dataclass(frozen=True)
class SOFTree:
    index: int
    seed: int
    root: SOFNode


@dataclass(frozen=True)
class FittedSOF:
    problem: DifferentiableStochasticProblem
    contexts: Array
    outcomes: Array
    trees: tuple[SOFTree, ...]
    config: SOFConfig

    def __post_init__(self) -> None:
        contexts = matrix(self.contexts, columns=self.problem.context_dim, name="contexts")
        outcomes = matrix(self.outcomes, columns=self.problem.scenario_dim, name="outcomes")
        if len(contexts) != len(outcomes):
            raise ValueError("SOF contexts and outcomes must align")
        if tuple(tree.index for tree in self.trees) != tuple(range(self.config.tree_count)):
            raise ValueError("SOF trees must be complete and ordered")
        contexts = np.ascontiguousarray(contexts.copy())
        outcomes = np.ascontiguousarray(outcomes.copy())
        contexts.setflags(write=False)
        outcomes.setflags(write=False)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "outcomes", outcomes)

    def weights(self, contexts: Array) -> Array:
        queries = matrix(contexts, columns=self.problem.context_dim, name="contexts")
        result = np.zeros((len(queries), len(self.contexts)), dtype=np.float64)
        for row, context in enumerate(queries):
            for tree in self.trees:
                indices = tree.root.leaf_for(context)
                result[row] += np.bincount(indices, minlength=len(self.contexts)) / len(indices)
        result /= len(self.trees)
        if np.any(~np.isfinite(result)) or not np.allclose(result.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError("SOF prediction weights are not normalized")
        return result

    def decisions(self, contexts: Array) -> Array:
        queries = matrix(contexts, columns=self.problem.context_dim, name="contexts")
        weights = self.weights(queries)
        decisions = np.empty((len(queries), self.problem.decision_dim), dtype=np.float64)
        for row, context in enumerate(queries):
            decisions[row] = self.problem.solve_numpy(
                context,
                self.outcomes,
                weights[row],
                ridge=self.config.ridge,
            )
        return decisions


@dataclass(frozen=True)
class _Candidate:
    feature: int
    threshold: float
    left_positions: Array
    right_positions: Array


@dataclass(frozen=True)
class _ParentModel:
    decision: Array
    gradients: Array
    hessian: Array
    lu: tuple[Array, Array]
    constraint_count: int
    equality: tuple[Array, Array] | None
    lower: Array | None
    upper: Array | None


def fit_sof(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    *,
    config: SOFConfig | None = None,
) -> FittedSOF:
    """Fit trees and restore deterministic tree-index order."""

    config = SOFConfig() if config is None else config
    training_contexts = matrix(contexts, columns=problem.context_dim, name="contexts")
    training_outcomes = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if len(training_contexts) != len(training_outcomes):
        raise ValueError("contexts and outcomes must contain the same observations")
    groups = tuple(
        tuple(group.tolist())
        for group in np.array_split(
            np.arange(config.tree_count), min(config.worker_count, config.tree_count)
        )
    )
    serial_config = replace(config, worker_count=1)
    if config.worker_count == 1:
        fitted_groups = (
            _fit_tree_group(
                problem,
                training_contexts,
                training_outcomes,
                serial_config,
                groups[0],
            ),
        )
    else:
        with ProcessPoolExecutor(
            max_workers=config.worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_tree_worker,
        ) as executor:
            futures = [
                executor.submit(
                    _fit_tree_group,
                    problem,
                    training_contexts,
                    training_outcomes,
                    serial_config,
                    group,
                )
                for group in groups
            ]
            fitted_groups = tuple(future.result() for future in futures)
    trees = tuple(
        sorted(
            (tree for group in fitted_groups for tree in group),
            key=lambda tree: tree.index,
        )
    )
    if tuple(tree.index for tree in trees) != tuple(range(config.tree_count)):
        raise RuntimeError("SOF parallel tree order is incomplete")
    training_contexts.setflags(write=False)
    training_outcomes.setflags(write=False)
    return FittedSOF(
        problem=problem,
        contexts=training_contexts,
        outcomes=training_outcomes,
        trees=trees,
        config=config,
    )


def fit_sof_tree(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    *,
    config: SOFConfig,
    tree_index: int = 0,
    tree_seed: int = 0,
    bootstrap_indices: Array | None = None,
) -> SOFTree:
    """Fit one tree; explicit bootstrap indices support deterministic audits."""

    if config.worker_count != 1:
        raise NotImplementedError("SOF multiprocessing is not implemented; use worker_count=1")
    training_contexts = matrix(contexts, columns=problem.context_dim, name="contexts")
    training_outcomes = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if len(training_contexts) != len(training_outcomes):
        raise ValueError("contexts and outcomes must align")
    if bootstrap_indices is None:
        indices = np.random.default_rng(tree_seed).integers(
            0,
            len(training_contexts),
            size=len(training_contexts),
            dtype=np.int64,
        )
    else:
        indices = np.asarray(bootstrap_indices, dtype=np.int64).reshape(-1).copy()
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(training_contexts)):
        raise ValueError("bootstrap_indices are empty or outside the training sample")
    root = _grow_node(
        problem,
        training_contexts,
        training_outcomes,
        indices,
        depth=0,
        config=config,
    )
    return SOFTree(index=int(tree_index), seed=int(tree_seed), root=root)


_SOF_THREAD_LIMIT = None

def _initialize_tree_worker():
    global _SOF_THREAD_LIMIT
    from threadpoolctl import threadpool_limits
    _SOF_THREAD_LIMIT = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    if os.environ.get("KKT_ATLAS_CPU_SET_IDS"):
        from kkt_atlas.windows_affinity_v1 import apply_affinity_from_environment
        apply_affinity_from_environment()


def _fit_tree_group(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    config: SOFConfig,
    tree_indices: tuple[int, ...],
) -> tuple[SOFTree, ...]:
    trees = []
    for tree_index in tree_indices:
        seed = _tree_seed(config.seed, tree_index)
        bootstrap = np.random.default_rng(seed).integers(
            0,
            len(contexts),
            size=len(contexts),
            dtype=np.int64,
        )
        trees.append(
            fit_sof_tree(
                problem,
                contexts,
                outcomes,
                config=config,
                tree_index=tree_index,
                tree_seed=seed,
                bootstrap_indices=bootstrap,
            )
        )
    return tuple(trees)


def _grow_node(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    indices: Array,
    *,
    depth: int,
    config: SOFConfig,
) -> SOFNode:
    if len(indices) <= config.terminal_node_size or depth >= config.max_depth:
        return SOFNode.leaf(indices)
    local_contexts = contexts[indices]
    if config.split_rule == "approximate_risk":
        partitions = _risk_partitions(local_contexts,config.minimum_child_fraction)
        if not partitions:
            return SOFNode.leaf(indices)
    else:
        candidates = _candidate_splits(local_contexts,minimum_child_fraction=config.minimum_child_fraction)
        if not candidates:
            return SOFNode.leaf(indices)
    parent = _parent_model(problem,local_contexts,outcomes[indices],config)
    if parent is None:
        return SOFNode.leaf(indices)
    if config.split_rule == "approximate_risk":
        best = _best_risk_split(local_contexts,parent,partitions)
    else:
        best = None
        for candidate in candidates:
            score = _candidate_score(problem,local_contexts,outcomes[indices],candidate,parent,config)
            if score is not None and (best is None or score < best[0]):
                best = (score,candidate)
    if best is None:
        return SOFNode.leaf(indices)
    selected = best[1]
    left = _grow_node(
        problem,
        contexts,
        outcomes,
        indices[selected.left_positions],
        depth=depth + 1,
        config=config,
    )
    right = _grow_node(
        problem,
        contexts,
        outcomes,
        indices[selected.right_positions],
        depth=depth + 1,
        config=config,
    )
    return SOFNode.branch(selected.feature, selected.threshold, left, right)


def _risk_partitions(contexts, minimum_child_fraction):
    minimum = max(1, math.ceil(len(contexts)*minimum_child_fraction))
    partitions = []
    for feature in range(contexts.shape[1]):
        order = np.argsort(contexts[:,feature], kind="stable")
        values = contexts[order,feature]
        distinct = np.flatnonzero(values[:-1] != values[1:])
        thresholds = values[distinct] + (values[distinct+1]-values[distinct])/2
        counts = np.searchsorted(values, thresholds, side="right")
        eligible = (counts >= minimum) & (len(contexts)-counts >= minimum)
        if np.any(eligible):
            partitions.append((feature,order,thresholds[eligible],counts[eligible]))
    return partitions


def _best_risk_split(contexts, parent, partitions):
    # The quadratic score needs child counts and gradient sums, not every
    # candidate's materialized membership arrays. Centering removes constants.
    centered = parent.gradients-parent.gradients[0]
    total = centered.sum(axis=0)
    mean = total/len(contexts)
    best = None
    for feature,order,thresholds,left_counts in partitions:
        prefix = np.cumsum(centered[order],axis=0)[left_counts-1]
        right_counts = len(contexts)-left_counts
        differences = np.concatenate((prefix/left_counts[:,None]-mean,
            (total-prefix)/right_counts[:,None]-mean))
        rhs = np.column_stack((-differences,
            np.zeros((len(differences),parent.constraint_count))))
        try:
            directions = lu_solve(parent.lu,rhs.T,check_finite=False).T[:,:parent.gradients.shape[1]]
        except (ValueError,np.linalg.LinAlgError):
            continue
        terms = np.sum((directions@parent.hessian)*directions,axis=1)
        terms += 2*np.sum(directions*differences,axis=1)
        size = len(left_counts)
        scores = (left_counts*terms[:size]+right_counts*terms[size:])/len(contexts)
        finite = np.all(np.isfinite(directions),axis=1)
        valid = finite[:size] & finite[size:] & np.isfinite(scores)
        scores = np.where(valid,scores,np.inf)
        index = int(np.argmin(scores))
        value = float(scores[index])
        if math.isfinite(value) and (best is None or value < best[0]):
            threshold = float(thresholds[index])
            best = (value,_Candidate(feature,threshold,
                np.flatnonzero(contexts[:,feature] <= threshold),
                np.flatnonzero(contexts[:,feature] > threshold)))
    return best


def _candidate_splits(
    node_contexts: Array,
    *,
    minimum_child_fraction: float,
) -> tuple[_Candidate, ...]:
    minimum_child = math.ceil(minimum_child_fraction * len(node_contexts))
    result = []
    for feature in range(node_contexts.shape[1]):
        values = node_contexts[:, feature]
        distinct = np.unique(values)
        for lower, upper in zip(distinct[:-1], distinct[1:], strict=True):
            threshold = float(lower + (upper - lower) / 2.0)
            left = np.flatnonzero(values <= threshold)
            right = np.flatnonzero(values > threshold)
            if len(left) >= minimum_child and len(right) >= minimum_child:
                result.append(_Candidate(feature, threshold, left, right))
    return tuple(result)


def _parent_model(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    config: SOFConfig,
) -> _ParentModel | None:
    context = contexts.mean(axis=0)
    unique, counts = np.unique(outcomes, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / len(outcomes)
    derivatives = getattr(problem, "sof_node_derivatives_numpy", None)
    if not callable(derivatives):
        raise TypeError("problem must implement sof_node_derivatives_numpy")
    try:
        decision = np.asarray(
            problem.solve_numpy(context, unique, probabilities, ridge=config.ridge),
            dtype=np.float64,
        ).reshape(-1)
        gradients, hessian = derivatives(
            decision,
            context,
            outcomes,
            ridge=config.ridge,
            active_set_tolerance=config.active_tolerance,
        )
        gradients = np.asarray(gradients, dtype=np.float64)
        hessian = np.asarray(hessian, dtype=np.float64)
        if (
            decision.shape != (problem.decision_dim,)
            or gradients.shape != (len(outcomes), problem.decision_dim)
            or hessian.shape != (problem.decision_dim, problem.decision_dim)
            or np.any(~np.isfinite(decision))
            or np.any(~np.isfinite(gradients))
            or np.any(~np.isfinite(hessian))
        ):
            return None
        equality, lower, upper = _constraints(problem, context, decision, config.active_tolerance)
        rows = [] if equality is None else [equality[0]]
        if lower is not None:
            active = np.flatnonzero(decision <= lower + config.active_tolerance)
            rows.extend(-np.eye(problem.decision_dim)[active])
        if upper is not None:
            active = np.flatnonzero(decision >= upper - config.active_tolerance)
            rows.extend(np.eye(problem.decision_dim)[active])
        constraints = (
            np.vstack(rows)
            if rows
            else np.empty((0, problem.decision_dim), dtype=np.float64)
        )
        kkt = np.block(
            [
                [hessian, constraints.T],
                [constraints, np.zeros((len(constraints), len(constraints)))],
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            factor = lu_factor(kkt, check_finite=False)
        return _ParentModel(
            decision=decision,
            gradients=gradients,
            hessian=hessian,
            lu=factor,
            constraint_count=len(constraints),
            equality=equality,
            lower=lower,
            upper=upper,
        )
    except (
        FloatingPointError,
        LinAlgWarning,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None


def _constraints(
    problem: DifferentiableStochasticProblem,
    context: Array,
    decision: Array,
    tolerance: float,
) -> tuple[tuple[Array, Array] | None, Array | None, Array | None]:
    equality_method = getattr(problem, "reference_equality_numpy", None)
    equality = None
    if callable(equality_method):
        matrix_value, rhs = equality_method(context)
        equality = (
            np.asarray(matrix_value, dtype=np.float64),
            np.asarray(rhs, dtype=np.float64).reshape(-1),
        )
        if equality[0].shape != (len(equality[1]), len(decision)):
            raise ValueError("reference equality has the wrong shape")
    lower = _bounds(problem, "reference_lower_bounds_numpy", context, len(decision))
    upper = _bounds(problem, "reference_upper_bounds_numpy", context, len(decision))
    if lower is not None and upper is not None and np.any(lower > upper + tolerance):
        raise ValueError("reference bounds are inconsistent")
    return equality, lower, upper


def _bounds(problem: Any, name: str, context: Array, size: int) -> Array | None:
    method = getattr(problem, name, None)
    if not callable(method):
        return None
    values = np.asarray(method(context), dtype=np.float64).reshape(-1)
    if values.shape != (size,) or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} returned invalid bounds")
    return values


def _candidate_score(
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
    candidate: _Candidate,
    parent: _ParentModel,
    config: SOFConfig,
) -> float | None:
    parent_gradient = parent.gradients.mean(axis=0)
    scores = []
    for positions in (candidate.left_positions, candidate.right_positions):
        difference = parent.gradients[positions].mean(axis=0) - parent_gradient
        rhs = np.concatenate((-difference, np.zeros(parent.constraint_count)))
        try:
            solution = lu_solve(parent.lu, rhs, check_finite=False)
        except (ValueError, np.linalg.LinAlgError):
            return None
        direction = solution[: problem.decision_dim]
        if np.any(~np.isfinite(direction)):
            return None
        if config.split_rule == "approximate_risk":
            score = float(
                direction @ parent.hessian @ direction
                + 2.0 * direction @ difference
            )
        else:
            decision = parent.decision + direction
            if np.any(~np.isfinite(decision)):
                return None
            try:
                score = _child_score(
                    problem,
                    decision,
                    contexts[positions],
                    outcomes[positions],
                    config.ridge,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                return None
        if not math.isfinite(score):
            return None
        scores.append(score)
    left_fraction = len(candidate.left_positions) / len(contexts)
    return float(left_fraction * scores[0] + (1.0 - left_fraction) * scores[1])


def _child_score(
    problem: DifferentiableStochasticProblem,
    decision: Array,
    contexts: Array,
    outcomes: Array,
    ridge: float,
) -> float:
    with torch.no_grad():
        decisions = torch.as_tensor(
            np.repeat(decision[None, :], len(outcomes), axis=0),
            dtype=torch.float64,
        )
        outcomes_tensor = torch.as_tensor(outcomes, dtype=torch.float64)
        contexts_tensor = torch.as_tensor(contexts, dtype=torch.float64)
        losses = problem.realized_loss_torch(decisions, outcomes_tensor, contexts_tensor)
        raw = float(losses.mean().item())
    center_method = getattr(problem, "reference_center_numpy", None)
    if callable(center_method):
        centers = np.vstack([center_method(context) for context in contexts])
    else:
        centers = np.zeros((len(contexts), len(decision)), dtype=np.float64)
    penalty = 0.5 * ridge * float(np.mean(np.sum((decision[None, :] - centers) ** 2, axis=1)))
    return raw + penalty


def _tree_seed(seed: int, tree_index: int) -> int:
    payload = f"{seed}|sof|tree|{tree_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def save_fitted_sof(model: FittedSOF, path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    encoded = _encode_trees(model.trees)
    arrays_path = destination / "arrays.npz"
    np.savez_compressed(arrays_path, **encoded)
    metadata = {
        "schema": _SERIALIZATION_SCHEMA,
        "config": asdict(model.config),
        "problem_identity": _problem_identity(model.problem, model.config.ridge),
        "contexts": array_identity(model.contexts),
        "outcomes": array_identity(model.outcomes),
        "arrays_sha256": _file_sha256(arrays_path),
    }
    metadata_path = destination / "model.json"
    metadata_path.write_bytes(_canonical_json(metadata))
    completion = {
        "schema": _SERIALIZATION_SCHEMA,
        "metadata_sha256": _file_sha256(metadata_path),
        "arrays_sha256": _file_sha256(arrays_path),
    }
    completion_path = destination / "COMPLETED.json"
    with completion_path.open("xb") as stream:
        stream.write(_canonical_json(completion))
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def load_fitted_sof(
    path: str | Path,
    problem: DifferentiableStochasticProblem,
    contexts: Array,
    outcomes: Array,
) -> FittedSOF:
    source = Path(path)
    metadata_path = source / "model.json"
    arrays_path = source / "arrays.npz"
    completion = json.loads((source / "COMPLETED.json").read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if completion.get("schema") != _SERIALIZATION_SCHEMA or metadata.get("schema") != _SERIALIZATION_SCHEMA:
        raise ValueError("wrong SOF serialization schema")
    if (
        completion.get("metadata_sha256") != _file_sha256(metadata_path)
        or completion.get("arrays_sha256") != _file_sha256(arrays_path)
        or metadata.get("arrays_sha256") != _file_sha256(arrays_path)
    ):
        raise ValueError("SOF model hash mismatch")
    config = SOFConfig(**metadata["config"])
    context_values = matrix(contexts, columns=problem.context_dim, name="contexts")
    outcome_values = matrix(outcomes, columns=problem.scenario_dim, name="outcomes")
    if (
        metadata["problem_identity"] != _problem_identity(problem, config.ridge)
        or metadata["contexts"] != array_identity(context_values)
        or metadata["outcomes"] != array_identity(outcome_values)
    ):
        raise ValueError("SOF model does not match its problem or training data")
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    return FittedSOF(
        problem=problem,
        contexts=context_values,
        outcomes=outcome_values,
        trees=_decode_trees(arrays, config.tree_count),
        config=config,
    )


def _encode_trees(trees: tuple[SOFTree, ...]) -> dict[str, Array]:
    features: list[int] = []
    thresholds: list[float] = []
    left: list[int] = []
    right: list[int] = []
    leaf_offsets: list[int] = []
    leaf_counts: list[int] = []
    samples: list[int] = []

    def encode(node: SOFNode) -> int:
        index = len(features)
        features.append(-1 if node.feature is None else node.feature)
        thresholds.append(math.nan if node.threshold is None else node.threshold)
        left.append(-1)
        right.append(-1)
        leaf_offsets.append(-1)
        leaf_counts.append(0)
        if node.is_leaf:
            assert node.bootstrap_indices is not None
            leaf_offsets[index] = len(samples)
            leaf_counts[index] = len(node.bootstrap_indices)
            samples.extend(int(value) for value in node.bootstrap_indices)
        else:
            if node.left is None or node.right is None:
                raise ValueError("SOF branch is incomplete")
            left[index] = encode(node.left)
            right[index] = encode(node.right)
        return index

    roots = [encode(tree.root) for tree in trees]
    return {
        "tree_indices": np.asarray([tree.index for tree in trees], dtype=np.int64),
        "tree_seeds": np.asarray([tree.seed for tree in trees], dtype=np.uint64),
        "roots": np.asarray(roots, dtype=np.int64),
        "features": np.asarray(features, dtype=np.int16),
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "leaf_offsets": np.asarray(leaf_offsets, dtype=np.int64),
        "leaf_counts": np.asarray(leaf_counts, dtype=np.int64),
        "samples": np.asarray(samples, dtype=np.int64),
    }


def _decode_trees(arrays: dict[str, Array], tree_count: int) -> tuple[SOFTree, ...]:
    required = {
        "tree_indices",
        "tree_seeds",
        "roots",
        "features",
        "thresholds",
        "left",
        "right",
        "leaf_offsets",
        "leaf_counts",
        "samples",
    }
    if set(arrays) != required:
        raise ValueError("SOF arrays are incomplete")
    node_count = len(arrays["features"])
    if any(len(arrays[name]) != node_count for name in required - {"tree_indices", "tree_seeds", "roots", "samples"}):
        raise ValueError("SOF node arrays do not align")

    def decode(index: int) -> SOFNode:
        if not 0 <= index < node_count:
            raise ValueError("SOF child index is outside the node arrays")
        feature = int(arrays["features"][index])
        if feature == -1:
            start = int(arrays["leaf_offsets"][index])
            stop = start + int(arrays["leaf_counts"][index])
            if start < 0 or stop > len(arrays["samples"]):
                raise ValueError("SOF leaf slice is outside the sample array")
            return SOFNode.leaf(arrays["samples"][start:stop])
        return SOFNode.branch(
            feature,
            float(arrays["thresholds"][index]),
            decode(int(arrays["left"][index])),
            decode(int(arrays["right"][index])),
        )

    if any(len(arrays[name]) != tree_count for name in ("tree_indices", "tree_seeds", "roots")):
        raise ValueError("SOF tree arrays do not match tree_count")
    return tuple(
        SOFTree(
            index=int(arrays["tree_indices"][position]),
            seed=int(arrays["tree_seeds"][position]),
            root=decode(int(arrays["roots"][position])),
        )
        for position in range(tree_count)
    )


def _problem_identity(problem: Any, ridge: float) -> dict[str, Any]:
    if hasattr(problem, "service_matrix"):
        return capacity_problem_identity(problem, ridge=ridge)
    if hasattr(problem, "asset_count"):
        return portfolio_problem_identity(problem, ridge=ridge)
    if hasattr(problem, "first_path_incidence"):
        return network_reservation_problem_identity(problem, ridge=ridge)
    raise TypeError("unsupported SOF problem identity")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
