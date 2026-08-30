from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

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
    """Solve the inverse master as an LP with L1 prior regularization.

    Variables are normalized objective weights ``theta``, one nonnegative slack per
    observation, and absolute-deviation variables measuring distance from the neutral
    simplex prior. This LP is numerically more reliable than an SLSQP quadratic master
    when demonstrations are feasible but intentionally suboptimal/noisy.
    """
    n_obs = len(observations)
    prior = np.full(n_products, 1.0 / n_products)
    theta_start = 0
    slack_start = n_products
    dev_start = n_products + n_obs
    n_vars = n_products + n_obs + n_products

    c = np.zeros(n_vars, dtype=float)
    c[slack_start:dev_start] = slack_penalty
    c[dev_start:] = 1.0

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # Challenger optimality: theta @ (x - y) + slack_i >= 0.
    for obs_index, challenger in challengers:
        observed = observations[obs_index].decision
        row = np.zeros(n_vars, dtype=float)
        row[theta_start:slack_start] = -(observed - challenger)
        row[slack_start + obs_index] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    # L1 distance to neutral prior: d_j >= |theta_j - prior_j|.
    for j in range(n_products):
        upper = np.zeros(n_vars, dtype=float)
        upper[j] = 1.0
        upper[dev_start + j] = -1.0
        a_ub.append(upper)
        b_ub.append(float(prior[j]))

        lower = np.zeros(n_vars, dtype=float)
        lower[j] = -1.0
        lower[dev_start + j] = -1.0
        a_ub.append(lower)
        b_ub.append(float(-prior[j]))

    a_eq = np.zeros((1, n_vars), dtype=float)
    a_eq[0, :n_products] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = (
        [(0.0, 1.0)] * n_products
        + [(0.0, None)] * n_obs
        + [(0.0, None)] * n_products
    )

    result = linprog(
        c,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"inverse master failed: {result.message}")
    theta = np.asarray(result.x[:n_products], dtype=float)
    slacks = np.asarray(result.x[slack_start:dev_start], dtype=float)
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
    if slack_penalty <= 0.0:
        raise ValueError("slack_penalty must be positive")
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
