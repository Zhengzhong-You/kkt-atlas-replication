"""Continuous portfolio allocation with a smoothed CVaR objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from kkt_atlas.problem import Array, Tensor, matrix, probability_vector


@dataclass(frozen=True)
class SmoothedCVaRPortfolioProblem:
    """Choose affine-budget portfolio weights under smoothed tail risk."""

    asset_count: int
    return_bound: float = 0.25
    tail_probability: float = 0.10
    smoothing: float = 0.015
    expected_return_weight: float = 0.25
    leverage_penalty: float = 0.02
    context_dim: int = 8
    gradient_iterations: int = 80
    step_fraction: float = 0.9
    training_solver: str = "accelerated_gradient"
    newton_damping: float = 0.5
    name: str = "smoothed_cvar_portfolio"

    def __post_init__(self) -> None:
        if self.asset_count < 2:
            raise ValueError("asset_count must be at least two")
        if not np.isfinite(self.return_bound) or self.return_bound <= 0.0:
            raise ValueError("return_bound must be positive and finite")
        if not 0.0 < self.tail_probability < 1.0:
            raise ValueError("tail_probability must lie in (0,1)")
        if not np.isfinite(self.smoothing) or self.smoothing <= 0.0:
            raise ValueError("smoothing must be positive and finite")
        if self.expected_return_weight < 0.0 or not np.isfinite(self.expected_return_weight):
            raise ValueError("expected_return_weight must be finite and nonnegative")
        if self.leverage_penalty < 0.0 or not np.isfinite(self.leverage_penalty):
            raise ValueError("leverage_penalty must be finite and nonnegative")
        if self.context_dim < 1 or self.gradient_iterations < 1:
            raise ValueError("context_dim and gradient_iterations must be positive")
        if not 0.0 < self.step_fraction <= 1.0:
            raise ValueError("step_fraction must lie in (0,1]")
        if self.training_solver not in {"accelerated_gradient", "damped_newton"}:
            raise ValueError("unknown portfolio training solver")
        if not np.isfinite(self.newton_damping) or not 0.0 < self.newton_damping <= 1.0:
            raise ValueError("newton_damping must lie in (0,1]")

    @property
    def scenario_dim(self) -> int:
        return self.asset_count

    @property
    def decision_dim(self) -> int:
        return self.asset_count + 1

    def transform_numpy(self, outcomes: Array) -> Array:
        values = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        if np.any(np.abs(values) >= self.return_bound):
            raise ValueError("portfolio returns must lie strictly inside the return bound")
        return np.arctanh(values / self.return_bound)

    def decode_torch(self, transformed: Tensor) -> Tensor:
        if transformed.shape[-1] != self.scenario_dim:
            raise ValueError("transformed scenarios have the wrong final dimension")
        return self.return_bound * torch.tanh(transformed)

    def reference_center_numpy(self, context: Array) -> Array:
        """Return equal portfolio weights and a zero CVaR threshold."""

        self._validate_context_numpy(context)
        center: Array = np.empty(self.decision_dim, dtype=np.float64)
        center[:-1] = 1.0 / self.asset_count
        center[-1] = 0.0
        return center

    def reference_equality_numpy(self, context: Array) -> tuple[Array, Array]:
        """Return the affine portfolio-budget equality used by the evaluator."""

        self._validate_context_numpy(context)
        matrix_value: Array = np.zeros(
            (1, self.decision_dim),
            dtype=np.float64,
        )
        matrix_value[0, :-1] = 1.0
        return matrix_value, np.ones(1, dtype=np.float64)

    def reference_decision_is_valid_numpy(self, decision: Array) -> bool:
        """The affine-budget model permits any finite weights and threshold."""

        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        return bool(
            value.shape == (self.decision_dim,)
            and np.all(np.isfinite(value))
        )

    def reference_objective_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> tuple[float, float, Array, Array]:
        """Evaluate the smoothed-CVaR objective and exact derivatives."""

        self._validate_context_numpy(context)
        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        if value.shape != (self.decision_dim,) or np.any(~np.isfinite(value)):
            raise ValueError("decision has the wrong shape")
        returns = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        self._validate_returns_numpy(returns)
        weights = probability_vector(probabilities, len(returns))
        ridge_value = _nonnegative_ridge(ridge)

        portfolio = value[:-1]
        threshold = float(value[-1])
        scenario_returns = returns @ portfolio
        scaled_shortfall = (
            -scenario_returns - threshold
        ) / self.smoothing
        smoothed_shortfall = self.smoothing * np.logaddexp(
            0.0,
            scaled_shortfall,
        )
        scenario_values = (
            -self.expected_return_weight * scenario_returns
            + threshold
            + smoothed_shortfall / self.tail_probability
        )
        raw_value = float(weights @ scenario_values)
        raw_value += 0.5 * self.leverage_penalty * float(portfolio @ portfolio)

        sigmoid = _sigmoid_numpy(scaled_shortfall)
        tail_weights = sigmoid / self.tail_probability
        coefficients = weights * (
            self.expected_return_weight + tail_weights
        )
        gradient: Array = np.empty(self.decision_dim, dtype=np.float64)
        gradient[:-1] = -(coefficients @ returns)
        gradient[:-1] += self.leverage_penalty * portfolio
        gradient[-1] = 1.0 - float(weights @ tail_weights)

        curvature = (
            weights
            * sigmoid
            * (1.0 - sigmoid)
            / (self.smoothing * self.tail_probability)
        )
        augmented_returns = np.column_stack(
            (returns, np.ones(len(returns), dtype=np.float64))
        )
        hessian = augmented_returns.T @ (
            curvature[:, None] * augmented_returns
        )
        hessian[:-1, :-1] += self.leverage_penalty * np.eye(
            self.asset_count,
        )

        center = self.reference_center_numpy(context)
        displacement = value - center
        regularization = 0.5 * ridge_value * float(displacement @ displacement)
        gradient += ridge_value * displacement
        hessian += ridge_value * np.eye(self.decision_dim)
        return raw_value, raw_value + regularization, gradient, hessian

    def solve_numpy(
        self,
        context: Array,
        scenarios: Array,
        probabilities: Array,
        *,
        ridge: float,
    ) -> Array:
        self._validate_context_numpy(context)
        returns = matrix(scenarios, columns=self.scenario_dim, name="scenarios")
        self._validate_returns_numpy(returns)
        weights = probability_vector(probabilities, len(returns))
        ridge_value = _nonnegative_ridge(ridge)
        if self.training_solver == "damped_newton":
            return self._solve_newton_numpy(context, returns, weights, ridge_value)
        step, strong_momentum = self._step_and_momentum(ridge_value)

        portfolio = np.full(self.asset_count, 1.0 / self.asset_count)
        threshold = 0.0
        extrapolated_portfolio = portfolio.copy()
        extrapolated_threshold = threshold
        for iteration in range(self.gradient_iterations):
            momentum = _momentum(ridge_value, strong_momentum, iteration)
            gradient_portfolio, gradient_threshold = self._gradients_numpy(
                extrapolated_portfolio,
                extrapolated_threshold,
                returns,
                weights,
                ridge_value,
            )
            next_portfolio = extrapolated_portfolio - step * gradient_portfolio
            next_portfolio -= (next_portfolio.sum() - 1.0) / self.asset_count
            next_threshold = extrapolated_threshold - step * gradient_threshold
            extrapolated_portfolio = next_portfolio + momentum * (next_portfolio - portfolio)
            extrapolated_threshold = next_threshold + momentum * (next_threshold - threshold)
            portfolio = next_portfolio
            threshold = next_threshold
        decision = np.concatenate((portfolio, np.asarray([threshold])))
        if np.any(~np.isfinite(decision)):
            raise FloatingPointError("portfolio solution map produced a nonfinite decision")
        return decision

    def solve_torch(
        self,
        contexts: Tensor,
        scenarios: Tensor,
        probabilities: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        self._validate_contexts_torch(contexts)
        if scenarios.ndim != 2 or scenarios.shape[1] != self.scenario_dim:
            raise ValueError("scenarios have the wrong shape")
        if probabilities.ndim != 2 or probabilities.shape != (
            len(contexts),
            len(scenarios),
        ):
            raise ValueError("probabilities must align with contexts and scenarios")
        if not torch.all(torch.isfinite(scenarios)):
            raise ValueError("portfolio returns must be finite")
        if not torch.all(torch.abs(scenarios) < self.return_bound):
            raise ValueError("portfolio returns must lie strictly inside the return bound")
        if not torch.all(torch.isfinite(probabilities)) or not torch.all(probabilities >= 0.0):
            raise ValueError("probabilities must be finite and nonnegative")
        ridge_value = _nonnegative_ridge(ridge)
        if self.training_solver == "damped_newton":
            return self._solve_newton_torch(scenarios, probabilities, ridge_value)
        step, strong_momentum = self._step_and_momentum(ridge_value)

        factory = {"dtype": scenarios.dtype, "device": scenarios.device}
        portfolio = torch.full(
            (len(contexts), self.asset_count),
            1.0 / self.asset_count,
            **factory,
        )
        threshold = torch.zeros((len(contexts), 1), **factory)
        extrapolated_portfolio = portfolio
        extrapolated_threshold = threshold
        base = torch.full((1, self.asset_count), 1.0 / self.asset_count, **factory)
        for iteration in range(self.gradient_iterations):
            momentum = _momentum(ridge_value, strong_momentum, iteration)
            scenario_returns = extrapolated_portfolio @ scenarios.T
            tail_argument = (-scenario_returns - extrapolated_threshold) / self.smoothing
            tail_weights = torch.sigmoid(tail_argument) / self.tail_probability
            coefficients = probabilities * (self.expected_return_weight + tail_weights)
            gradient_portfolio = -(coefficients @ scenarios)
            gradient_portfolio = gradient_portfolio + (
                self.leverage_penalty * extrapolated_portfolio + ridge_value * (extrapolated_portfolio - base)
            )
            gradient_threshold = (
                1.0 - (probabilities * tail_weights).sum(dim=1, keepdim=True) + ridge_value * extrapolated_threshold
            )
            next_portfolio = extrapolated_portfolio - step * gradient_portfolio
            next_portfolio = next_portfolio - (next_portfolio.sum(dim=1, keepdim=True) - 1.0) / self.asset_count
            next_threshold = extrapolated_threshold - step * gradient_threshold
            extrapolated_portfolio = next_portfolio + momentum * (next_portfolio - portfolio)
            extrapolated_threshold = next_threshold + momentum * (next_threshold - threshold)
            portfolio = next_portfolio
            threshold = next_threshold
        decision = torch.cat((portfolio, threshold), dim=1)
        if not torch.all(torch.isfinite(decision)):
            raise FloatingPointError("portfolio solution map produced nonfinite decisions")
        return decision

    def solve_grouped_torch(
        self,
        contexts: Tensor,
        scenarios: Tensor,
        probabilities: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        """Solve equal-size chart batches without forming a block support."""

        if contexts.ndim != 3 or contexts.shape[2] != self.context_dim:
            raise ValueError(
                f"contexts must have shape (b,n,{self.context_dim})"
            )
        if scenarios.ndim != 3 or scenarios.shape[2] != self.scenario_dim:
            raise ValueError("scenarios must have shape (b,k,m)")
        chart_count, observation_count, _ = contexts.shape
        if scenarios.shape[0] != chart_count:
            raise ValueError("scenarios must align with charts")
        support_size = scenarios.shape[1]
        if probabilities.shape != (
            chart_count,
            observation_count,
            support_size,
        ):
            raise ValueError(
                "probabilities must align with charts, observations, and scenarios"
            )
        if not torch.all(torch.isfinite(contexts)):
            raise ValueError("contexts must be finite")
        if not torch.all(torch.isfinite(scenarios)):
            raise ValueError("portfolio returns must be finite")
        if not torch.all(torch.abs(scenarios) < self.return_bound):
            raise ValueError(
                "portfolio returns must lie strictly inside the return bound"
            )
        if not torch.all(torch.isfinite(probabilities)) or not torch.all(
            probabilities >= 0.0
        ):
            raise ValueError("probabilities must be finite and nonnegative")

        ridge_value = _nonnegative_ridge(ridge)
        if self.training_solver == "damped_newton":
            return self._solve_grouped_newton_torch(scenarios, probabilities, ridge_value)
        step, strong_momentum = self._step_and_momentum(ridge_value)
        factory = {"dtype": scenarios.dtype, "device": scenarios.device}
        portfolio = torch.full(
            (chart_count, observation_count, self.asset_count),
            1.0 / self.asset_count,
            **factory,
        )
        threshold = torch.zeros(
            (chart_count, observation_count, 1),
            **factory,
        )
        extrapolated_portfolio = portfolio
        extrapolated_threshold = threshold
        base = torch.full(
            (1, 1, self.asset_count),
            1.0 / self.asset_count,
            **factory,
        )
        for iteration in range(self.gradient_iterations):
            momentum = _momentum(ridge_value, strong_momentum, iteration)
            scenario_returns = torch.einsum(
                "bnm,bkm->bnk",
                extrapolated_portfolio,
                scenarios,
            )
            tail_argument = (
                -scenario_returns - extrapolated_threshold
            ) / self.smoothing
            tail_weights = torch.sigmoid(tail_argument) / self.tail_probability
            coefficients = probabilities * (
                self.expected_return_weight + tail_weights
            )
            gradient_portfolio = -torch.einsum(
                "bnk,bkm->bnm",
                coefficients,
                scenarios,
            )
            gradient_portfolio = gradient_portfolio + (
                self.leverage_penalty * extrapolated_portfolio
                + ridge_value * (extrapolated_portfolio - base)
            )
            gradient_threshold = (
                1.0
                - (probabilities * tail_weights).sum(dim=2, keepdim=True)
                + ridge_value * extrapolated_threshold
            )
            next_portfolio = extrapolated_portfolio - step * gradient_portfolio
            next_portfolio = next_portfolio - (
                next_portfolio.sum(dim=2, keepdim=True) - 1.0
            ) / self.asset_count
            next_threshold = extrapolated_threshold - step * gradient_threshold
            extrapolated_portfolio = next_portfolio + momentum * (
                next_portfolio - portfolio
            )
            extrapolated_threshold = next_threshold + momentum * (
                next_threshold - threshold
            )
            portfolio = next_portfolio
            threshold = next_threshold
        decision = torch.cat((portfolio, threshold), dim=2)
        if not torch.all(torch.isfinite(decision)):
            raise FloatingPointError(
                "portfolio solution map produced nonfinite decisions"
            )
        return decision

    def realized_loss_from_chart_batches_torch(
        self,
        contexts: Tensor,
        scenarios: Tensor,
        probabilities: Tensor,
        outcomes: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        """Evaluate independent chart batches with B*n*K scenario terms."""

        decisions = self.solve_grouped_torch(
            contexts,
            scenarios,
            probabilities,
            ridge=ridge,
        )
        if outcomes.shape != contexts.shape[:2] + (self.scenario_dim,):
            raise ValueError("outcomes must align with charts and contexts")
        portfolio = decisions[:, :, :-1]
        threshold = decisions[:, :, -1]
        portfolio_returns = (portfolio * outcomes).sum(dim=2)
        shortfall = -portfolio_returns - threshold
        tail = self.smoothing * functional.softplus(
            shortfall / self.smoothing
        )
        leverage = (
            0.5
            * self.leverage_penalty
            * portfolio.square().sum(dim=2)
        )
        return (
            -self.expected_return_weight * portfolio_returns
            + threshold
            + tail / self.tail_probability
            + leverage
        )

    def kkt_contributions_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
    ) -> Array:
        self._validate_context_numpy(context)
        value = np.asarray(decision, dtype=np.float64).reshape(-1)
        if value.shape != (self.decision_dim,) or np.any(~np.isfinite(value)):
            raise ValueError("decision has the wrong shape")
        returns = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        self._validate_returns_numpy(returns)
        portfolio = value[:-1]
        threshold = value[-1]
        scenario_returns = returns @ portfolio
        sigmoid = _sigmoid_numpy((-scenario_returns - threshold) / self.smoothing)
        multiplier = self.expected_return_weight + sigmoid / self.tail_probability
        portfolio_gradient = -multiplier[:, None] * returns
        threshold_gradient = 1.0 - sigmoid / self.tail_probability
        return np.column_stack((portfolio_gradient, threshold_gradient))

    def sof_node_derivatives_numpy(
        self,
        decision: Array,
        context: Array,
        outcomes: Array,
        *,
        ridge: float,
        active_set_tolerance: float,
    ) -> tuple[Array, Array]:
        """Return SOF observation gradients and the regularized parent Hessian."""

        tolerance = float(active_set_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("active_set_tolerance must be finite and nonnegative")
        returns = matrix(outcomes, columns=self.scenario_dim, name="outcomes")
        gradients = self.kkt_contributions_numpy(decision, context, returns)
        probabilities = np.full(len(returns), 1.0 / len(returns))
        _, _, _, hessian = self.reference_objective_derivatives_numpy(
            decision,
            context,
            returns,
            probabilities,
            ridge=ridge,
        )
        return gradients, np.ascontiguousarray(hessian)

    def realized_loss_torch(
        self,
        decisions: Tensor,
        outcomes: Tensor,
        contexts: Tensor,
    ) -> Tensor:
        self._validate_contexts_torch(contexts)
        if decisions.ndim != 2 or decisions.shape != (
            len(contexts),
            self.decision_dim,
        ):
            raise ValueError("decisions must align with contexts")
        if outcomes.ndim != 2 or outcomes.shape != (
            len(contexts),
            self.scenario_dim,
        ):
            raise ValueError("outcomes must align with contexts")
        portfolio = decisions[:, :-1]
        threshold = decisions[:, -1]
        portfolio_returns = (portfolio * outcomes).sum(dim=1)
        shortfall = -portfolio_returns - threshold
        tail = self.smoothing * functional.softplus(shortfall / self.smoothing)
        leverage = 0.5 * self.leverage_penalty * portfolio.square().sum(dim=1)
        return -self.expected_return_weight * portfolio_returns + threshold + tail / self.tail_probability + leverage

    def _solve_newton_numpy(
        self,
        context: Array,
        returns: Array,
        weights: Array,
        ridge: float,
    ) -> Array:
        decision = self.reference_center_numpy(context)
        equality = np.zeros(self.decision_dim, dtype=np.float64)
        equality[:-1] = 1.0
        for _ in range(self.gradient_iterations):
            _, _, gradient, hessian = self.reference_objective_derivatives_numpy(
                decision,
                context,
                returns,
                weights,
                ridge=ridge,
            )
            kkt = np.zeros((self.decision_dim + 1, self.decision_dim + 1), dtype=np.float64)
            kkt[:-1, :-1] = hessian
            kkt[:-1, -1] = equality
            kkt[-1, :-1] = equality
            residual = np.concatenate((gradient, [decision[:-1].sum() - 1.0]))
            decision += self.newton_damping * np.linalg.solve(kkt, -residual)[:-1]
        if np.any(~np.isfinite(decision)):
            raise FloatingPointError("portfolio Newton map produced a nonfinite decision")
        return decision

    def _solve_newton_torch(
        self,
        scenarios: Tensor,
        probabilities: Tensor,
        ridge: float,
    ) -> Tensor:
        batch = len(probabilities)
        factory = {"dtype": scenarios.dtype, "device": scenarios.device}
        portfolio = torch.full((batch, self.asset_count), 1.0 / self.asset_count, **factory)
        threshold = torch.zeros((batch, 1), **factory)
        center = portfolio.clone()
        augmented = torch.cat((scenarios, torch.ones((len(scenarios), 1), **factory)), dim=1)
        equality = torch.zeros(self.decision_dim, **factory)
        equality[:-1] = 1.0
        equality_column = equality.view(1, -1, 1).expand(batch, -1, -1)
        equality_row = equality.view(1, 1, -1).expand(batch, -1, -1)
        zero = torch.zeros((batch, 1, 1), **factory)
        diagonal = torch.full((self.decision_dim,), ridge, **factory)
        diagonal[:-1] += self.leverage_penalty
        diagonal = torch.diag(diagonal).unsqueeze(0)
        for _ in range(self.gradient_iterations):
            scenario_returns = portfolio @ scenarios.T
            sigmoid = torch.sigmoid((-scenario_returns - threshold) / self.smoothing)
            tail = sigmoid / self.tail_probability
            coefficients = probabilities * (self.expected_return_weight + tail)
            gradient_portfolio = -(coefficients @ scenarios)
            gradient_portfolio += self.leverage_penalty * portfolio + ridge * (portfolio - center)
            gradient_threshold = 1.0 - (probabilities * tail).sum(dim=1, keepdim=True) + ridge * threshold
            gradient = torch.cat((gradient_portfolio, gradient_threshold), dim=1)
            curvature = probabilities * sigmoid * (1.0 - sigmoid) / (self.smoothing * self.tail_probability)
            hessian = torch.einsum("bk,kd,ke->bde", curvature, augmented, augmented) + diagonal
            kkt = torch.cat(
                (
                    torch.cat((hessian, equality_column), dim=2),
                    torch.cat((equality_row, zero), dim=2),
                ),
                dim=1,
            )
            residual = torch.cat((gradient, portfolio.sum(dim=1, keepdim=True) - 1.0), dim=1)
            step = torch.linalg.solve(kkt, -residual.unsqueeze(2)).squeeze(2)[:, :-1]
            portfolio = portfolio + self.newton_damping * step[:, :-1]
            threshold = threshold + self.newton_damping * step[:, -1:]
        decision = torch.cat((portfolio, threshold), dim=1)
        if not torch.all(torch.isfinite(decision)):
            raise FloatingPointError("portfolio Newton map produced nonfinite decisions")
        return decision

    def _solve_grouped_newton_torch(
        self,
        scenarios: Tensor,
        probabilities: Tensor,
        ridge: float,
    ) -> Tensor:
        chart_count, observation_count, support_size = probabilities.shape
        factory = {"dtype": scenarios.dtype, "device": scenarios.device}
        portfolio = torch.full(
            (chart_count, observation_count, self.asset_count),
            1.0 / self.asset_count,
            **factory,
        )
        threshold = torch.zeros((chart_count, observation_count, 1), **factory)
        center = portfolio.clone()
        augmented = torch.cat(
            (scenarios, torch.ones((chart_count, support_size, 1), **factory)), dim=2
        )
        equality = torch.zeros(self.decision_dim, **factory)
        equality[:-1] = 1.0
        equality_column = equality.view(1, 1, -1, 1).expand(
            chart_count, observation_count, -1, -1
        )
        equality_row = equality.view(1, 1, 1, -1).expand(
            chart_count, observation_count, -1, -1
        )
        zero = torch.zeros((chart_count, observation_count, 1, 1), **factory)
        diagonal = torch.full((self.decision_dim,), ridge, **factory)
        diagonal[:-1] += self.leverage_penalty
        diagonal = torch.diag(diagonal).view(1, 1, self.decision_dim, self.decision_dim)
        for _ in range(self.gradient_iterations):
            scenario_returns = torch.einsum("bnm,bkm->bnk", portfolio, scenarios)
            sigmoid = torch.sigmoid((-scenario_returns - threshold) / self.smoothing)
            tail = sigmoid / self.tail_probability
            coefficients = probabilities * (self.expected_return_weight + tail)
            gradient_portfolio = -torch.einsum("bnk,bkm->bnm", coefficients, scenarios)
            gradient_portfolio += self.leverage_penalty * portfolio + ridge * (portfolio - center)
            gradient_threshold = 1.0 - (probabilities * tail).sum(dim=2, keepdim=True) + ridge * threshold
            gradient = torch.cat((gradient_portfolio, gradient_threshold), dim=2)
            curvature = probabilities * sigmoid * (1.0 - sigmoid) / (self.smoothing * self.tail_probability)
            hessian = torch.einsum("bnk,bkd,bke->bnde", curvature, augmented, augmented) + diagonal
            kkt = torch.cat(
                (
                    torch.cat((hessian, equality_column), dim=3),
                    torch.cat((equality_row, zero), dim=3),
                ),
                dim=2,
            )
            residual = torch.cat((gradient, portfolio.sum(dim=2, keepdim=True) - 1.0), dim=2)
            step = torch.linalg.solve(kkt, -residual.unsqueeze(3)).squeeze(3)[:, :, :-1]
            portfolio = portfolio + self.newton_damping * step[:, :, :-1]
            threshold = threshold + self.newton_damping * step[:, :, -1:]
        decision = torch.cat((portfolio, threshold), dim=2)
        if not torch.all(torch.isfinite(decision)):
            raise FloatingPointError("portfolio grouped Newton map produced nonfinite decisions")
        return decision

    def _gradients_numpy(
        self,
        portfolio: Array,
        threshold: float,
        returns: Array,
        probabilities: Array,
        ridge: float,
    ) -> tuple[Array, float]:
        scenario_returns = returns @ portfolio
        tail_weights = _sigmoid_numpy((-scenario_returns - threshold) / self.smoothing) / self.tail_probability
        coefficients = probabilities * (self.expected_return_weight + tail_weights)
        base = np.full(self.asset_count, 1.0 / self.asset_count)
        gradient_portfolio = -(coefficients @ returns)
        gradient_portfolio += self.leverage_penalty * portfolio + ridge * (portfolio - base)
        gradient_threshold = 1.0 - float(probabilities @ tail_weights) + ridge * threshold
        return gradient_portfolio, gradient_threshold

    def _step_and_momentum(self, ridge: float) -> tuple[float, float]:
        lipschitz = (
            self.leverage_penalty
            + ridge
            + (self.asset_count * self.return_bound**2 + 1.0) / (4.0 * self.smoothing * self.tail_probability)
        )
        strong_convexity = ridge
        step = self.step_fraction / lipschitz
        root_lipschitz = np.sqrt(lipschitz)
        root_strong = np.sqrt(strong_convexity)
        momentum = (root_lipschitz - root_strong) / (root_lipschitz + root_strong)
        return float(step), float(momentum)

    def _validate_returns_numpy(self, returns: Array) -> None:
        if np.any(np.abs(returns) >= self.return_bound):
            raise ValueError("portfolio returns must lie strictly inside the return bound")

    def _validate_context_numpy(self, context: Array) -> None:
        value = np.asarray(context, dtype=np.float64).reshape(-1)
        if value.shape != (self.context_dim,) or np.any(~np.isfinite(value)):
            raise ValueError(f"context must have shape ({self.context_dim},)")

    def _validate_contexts_torch(self, contexts: Tensor) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape (n,{self.context_dim})")
        if not torch.all(torch.isfinite(contexts)):
            raise ValueError("contexts must be finite")


def make_smoothed_cvar_portfolio_problem(
    asset_count: int,
    *,
    context_dim: int = 8,
    return_bound: float = 0.25,
    tail_probability: float = 0.10,
    smoothing: float = 0.015,
    expected_return_weight: float = 0.25,
    leverage_penalty: float = 0.02,
    gradient_iterations: int = 80,
    step_fraction: float = 0.9,
    training_solver: str = "accelerated_gradient",
    newton_damping: float = 0.5,
) -> SmoothedCVaRPortfolioProblem:
    return SmoothedCVaRPortfolioProblem(
        asset_count=asset_count,
        context_dim=context_dim,
        return_bound=return_bound,
        tail_probability=tail_probability,
        smoothing=smoothing,
        expected_return_weight=expected_return_weight,
        leverage_penalty=leverage_penalty,
        gradient_iterations=gradient_iterations,
        step_fraction=step_fraction,
        training_solver=training_solver,
        newton_damping=newton_damping,
    )


def _nonnegative_ridge(ridge: float) -> float:
    value = float(ridge)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("ridge must be nonnegative and finite")
    return value


def _momentum(ridge: float, strong_momentum: float, iteration: int) -> float:
    return iteration / (iteration + 3.0) if ridge == 0.0 else strong_momentum


def _sigmoid_numpy(values: Array) -> Array:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponential = np.exp(array[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result
