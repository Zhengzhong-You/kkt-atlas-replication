"""Local artificial scenarios and context-dependent probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from kkt_atlas.initialization import ScenarioInitialization
from kkt_atlas.probability_initialization import ProbabilityHeadInitialization
from kkt_atlas.problem import Array, DifferentiableStochasticProblem, Tensor
from kkt_atlas.response_basis import ResponseDecomposition


@dataclass(frozen=True)
class LocalModelSnapshot:
    step: int
    scenario_coordinates: Array
    probability_intercepts: Array
    probability_slopes: Array
    training_objective: float
    mean_operational_loss: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("snapshot step must be nonnegative")
        if not np.isfinite(self.training_objective) or not np.isfinite(self.mean_operational_loss):
            raise ValueError("snapshot losses must be finite")
        coordinates = np.asarray(
            self.scenario_coordinates,
            dtype=np.float64,
        )
        intercepts = np.asarray(
            self.probability_intercepts,
            dtype=np.float64,
        ).reshape(-1)
        slopes = np.asarray(
            self.probability_slopes,
            dtype=np.float64,
        )
        if (
            coordinates.ndim != 2
            or len(coordinates) == 0
            or coordinates.shape[1] == 0
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError("snapshot scenario coordinates must be a finite matrix")
        if intercepts.shape != (len(coordinates),) or np.any(~np.isfinite(intercepts)):
            raise ValueError("snapshot probability intercepts have the wrong shape")
        if slopes.ndim != 2 or slopes.shape[0] != len(coordinates) or np.any(~np.isfinite(slopes)):
            raise ValueError("snapshot probability slopes have the wrong shape")
        immutable: list[Array] = []
        for array in (coordinates, intercepts, slopes):
            copy = np.ascontiguousarray(array.copy())
            copy.setflags(write=False)
            immutable.append(copy)
        object.__setattr__(self, "scenario_coordinates", immutable[0])
        object.__setattr__(self, "probability_intercepts", immutable[1])
        object.__setattr__(self, "probability_slopes", immutable[2])

    @property
    def support_size(self) -> int:
        return int(len(self.scenario_coordinates))

    @property
    def width(self) -> int:
        return int(self.scenario_coordinates.shape[1])


class LocalScenarioModel(nn.Module):
    """A fixed local support with an affine-softmax probability head."""

    def __init__(
        self,
        problem: DifferentiableStochasticProblem,
        decomposition: ResponseDecomposition,
        initialization: ScenarioInitialization,
        *,
        width: int,
        context_mean: Array,
        context_scale: Array,
        standardized_center: Array,
        probability_initialization: ProbabilityHeadInitialization | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if initialization.coordinates.shape[1] != width:
            raise ValueError("initial scenario coordinates do not match width")
        ambient_identity = decomposition.basis_method == "identity"
        if ambient_identity and width != problem.scenario_dim:
            raise ValueError("identity basis requires the full ambient scenario width")
        basis = np.empty((0, 0), dtype=np.float64) if ambient_identity else decomposition.basis(width)
        context_mean_array = np.asarray(context_mean, dtype=np.float64).reshape(-1)
        context_scale_array = np.asarray(context_scale, dtype=np.float64).reshape(-1)
        center_array = np.asarray(standardized_center, dtype=np.float64).reshape(-1)
        if context_mean_array.shape != (problem.context_dim,):
            raise ValueError("context_mean has the wrong dimension")
        if context_scale_array.shape != (problem.context_dim,) or np.any(context_scale_array <= 0.0):
            raise ValueError("context_scale must be positive with the problem context dimension")
        if center_array.shape != (problem.context_dim,):
            raise ValueError("standardized_center has the wrong dimension")

        self.problem = problem
        self.width = int(width)
        self.support_size = initialization.support_size
        self.context_dim = problem.context_dim
        self._ambient_identity = ambient_identity
        factory = {"dtype": dtype, "device": device}
        self.register_buffer(
            "transformed_mean",
            torch.tensor(decomposition.transformed_mean, **factory),
        )
        self.register_buffer(
            "transformed_scale",
            torch.tensor(decomposition.transformed_scale, **factory),
        )
        self.register_buffer("basis", torch.tensor(basis, **factory))
        self.register_buffer("context_mean", torch.tensor(context_mean_array, **factory))
        self.register_buffer("context_scale", torch.tensor(context_scale_array, **factory))
        self.register_buffer("standardized_center", torch.tensor(center_array, **factory))

        if probability_initialization is None:
            frequencies = initialization.counts.astype(np.float64)
            frequencies /= frequencies.sum()
            intercepts = np.log(frequencies)
            intercepts -= intercepts.mean()
            slopes = np.zeros(
                (self.support_size, self.context_dim),
                dtype=np.float64,
            )
        else:
            intercepts = np.asarray(
                probability_initialization.intercepts,
                dtype=np.float64,
            )
            slopes = np.asarray(
                probability_initialization.slopes,
                dtype=np.float64,
            )
            if intercepts.shape != (self.support_size,):
                raise ValueError("probability initialization intercepts have the wrong shape")
            if slopes.shape != (self.support_size, self.context_dim):
                raise ValueError("probability initialization slopes have the wrong shape")
        self.scenario_coordinates = nn.Parameter(torch.tensor(initialization.coordinates, **factory))
        self.probability_intercepts = nn.Parameter(torch.tensor(intercepts, **factory))
        self.probability_slopes = nn.Parameter(torch.tensor(slopes, **factory))

    def standardized_contexts(self, contexts: Tensor) -> Tensor:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape (n,{self.context_dim}), got {tuple(contexts.shape)}")
        return (contexts - self.context_mean) / self.context_scale

    def _centered_standardized_contexts(self, contexts: Tensor) -> Tensor:
        return self.standardized_contexts(contexts) - self.standardized_center

    def _probabilities_from_centered_contexts(self, centered_contexts: Tensor) -> Tensor:
        if centered_contexts.ndim != 2 or centered_contexts.shape[1] != self.context_dim:
            raise ValueError(
                f"centered_contexts must have shape (n,{self.context_dim}), got {tuple(centered_contexts.shape)}"
            )
        logits = self.probability_intercepts[None, :] + centered_contexts @ self.probability_slopes.T
        return torch.softmax(logits, dim=1)

    def probabilities(self, contexts: Tensor) -> Tensor:
        return self._probabilities_from_centered_contexts(self._centered_standardized_contexts(contexts))

    def scenarios(self) -> Tensor:
        standardized = self.scenario_coordinates if self._ambient_identity else self.scenario_coordinates @ self.basis.T
        transformed = self.transformed_mean[None, :] + standardized * self.transformed_scale[None, :]
        return self.problem.decode_torch(transformed)

    def decisions(self, contexts: Tensor, *, ridge: float) -> Tensor:
        return self.problem.solve_torch(
            contexts,
            self.scenarios(),
            self.probabilities(contexts),
            ridge=ridge,
        )

    @torch.no_grad()
    def project_identifiability_(self) -> None:
        """Remove the common-logit directions without changing probabilities."""

        self.probability_intercepts.sub_(self.probability_intercepts.mean())
        self.probability_slopes.sub_(self.probability_slopes.mean(dim=0, keepdim=True))

    @torch.no_grad()
    def snapshot(
        self,
        *,
        step: int,
        training_objective: float,
        mean_operational_loss: float,
    ) -> LocalModelSnapshot:
        return LocalModelSnapshot(
            step=int(step),
            scenario_coordinates=self.scenario_coordinates.detach().cpu().numpy().copy(),
            probability_intercepts=self.probability_intercepts.detach().cpu().numpy().copy(),
            probability_slopes=self.probability_slopes.detach().cpu().numpy().copy(),
            training_objective=float(training_objective),
            mean_operational_loss=float(mean_operational_loss),
        )

    @torch.no_grad()
    def load_snapshot(self, snapshot: LocalModelSnapshot) -> None:
        if snapshot.scenario_coordinates.shape != tuple(self.scenario_coordinates.shape):
            raise ValueError("snapshot scenario coordinates have the wrong shape")
        if snapshot.probability_intercepts.shape != tuple(self.probability_intercepts.shape):
            raise ValueError("snapshot probability intercepts have the wrong shape")
        if snapshot.probability_slopes.shape != tuple(self.probability_slopes.shape):
            raise ValueError("snapshot probability slopes have the wrong shape")
        self.scenario_coordinates.copy_(
            torch.tensor(
                snapshot.scenario_coordinates,
                dtype=self.scenario_coordinates.dtype,
                device=self.scenario_coordinates.device,
            )
        )
        self.probability_intercepts.copy_(
            torch.tensor(
                snapshot.probability_intercepts,
                dtype=self.probability_intercepts.dtype,
                device=self.probability_intercepts.device,
            )
        )
        self.probability_slopes.copy_(
            torch.tensor(
                snapshot.probability_slopes,
                dtype=self.probability_slopes.dtype,
                device=self.probability_slopes.device,
            )
        )
