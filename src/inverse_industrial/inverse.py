from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .data import Observation
from .forward import ProductionModel, solve_forward


@dataclass(frozen=True)
class InverseResult:
    theta: np.ndarray
    slacks: np.ndarray
    iterations: int
    challenger_count: int
    converged: bool


def _solve_master(
    observations: list[Observation],
    challengers: list[tuple[int, np.ndarray]],
    n_products: int,
    *,
    slack_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_obs = len(observations)
    prior = np.full(n_products, 1.0 / n_products)

    def objective(z: np.ndarray) -> float:
        theta = z[:n_products]
        slacks = z[n_products:]
        return float(0.5 * np.sum((theta - prior) ** 2) + slack_penalty * np.sum(slacks**2))

    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda z: float(np.sum(z[:n_products]) - 1.0)}
    ]
    for obs_index, challenger in challengers:
        observed = observations[obs_index].decision.copy()
        rival = challenger.copy()
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda z, i=obs_index, x=observed, y=rival: float(
                    z[:n_products] @ (x - y) + z[n_products + i]
                ),
            }
        )

    start = np.concatenate([prior, np.full(n_obs, 1e-6)])
    bounds = [(0.0, 1.0)] * n_products + [(0.0, None)] * n_obs
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"inverse master failed: {result.message}")
    theta = np.asarray(result.x[:n_products], dtype=float)
    slacks = np.asarray(result.x[n_products:], dtype=float)
    return theta, slacks


def recover_objective(
    model: ProductionModel,
    observations: list[Observation],
    *,
    slack_penalty: float = 50.0,
    tolerance: float = 1e-7,
    max_iterations: int = 30,
) -> InverseResult:
    if not observations:
        raise ValueError("observations must not be empty")
    challengers: list[tuple[int, np.ndarray]] = []

    for iteration in range(1, max_iterations + 1):
        theta, slacks = _solve_master(
            observations,
            challengers,
            model.n_products,
            slack_penalty=slack_penalty,
        )
        added = 0
        for i, observation in enumerate(observations):
            rival = np.asarray(solve_forward(model, observation.capacities, theta)["x"])
            violation = float(theta @ rival - theta @ observation.decision - slacks[i])
            if violation > tolerance:
                challengers.append((i, rival))
                added += 1
        if added == 0:
            return InverseResult(theta, slacks, iteration, len(challengers), True)

    theta, slacks = _solve_master(
        observations,
        challengers,
        model.n_products,
        slack_penalty=slack_penalty,
    )
    return InverseResult(theta, slacks, max_iterations, len(challengers), False)
