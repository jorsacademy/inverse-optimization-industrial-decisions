from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import ProductionModel, solve_forward


@dataclass(frozen=True)
class Observation:
    capacities: np.ndarray
    decision: np.ndarray


def hidden_theta() -> np.ndarray:
    theta = np.array([0.34, 0.26, 0.23, 0.17], dtype=float)
    return theta / theta.sum()


def capacity_scenario(
    seed: int,
    model: ProductionModel,
    *,
    regime: str = "nominal",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if regime == "nominal":
        base = np.array([95.0, 88.0, 72.0], dtype=float)
        scale = np.array([12.0, 10.0, 8.0], dtype=float)
        floor = np.array([45.0, 40.0, 35.0], dtype=float)
    elif regime == "tight_capacity":
        base = np.array([76.0, 69.0, 57.0], dtype=float)
        scale = np.array([15.0, 13.0, 10.0], dtype=float)
        floor = np.array([32.0, 30.0, 26.0], dtype=float)
    else:
        raise ValueError(f"unknown capacity regime: {regime}")
    shock = rng.normal(0.0, scale)
    capacities = np.maximum(base + shock, floor)
    if capacities.shape != (model.n_resources,):
        raise ValueError("default capacity generator assumes three resources")
    return capacities


def build_observations(
    seeds: list[int],
    model: ProductionModel,
    theta: np.ndarray | None = None,
    *,
    decision_noise: float = 0.0,
    noise_seed: int = 0,
    capacity_regime: str = "nominal",
) -> list[Observation]:
    if not 0.0 <= decision_noise <= 1.0:
        raise ValueError("decision_noise must lie in [0, 1]")
    objective = hidden_theta() if theta is None else np.asarray(theta, dtype=float)
    rng = np.random.default_rng(noise_seed)
    rows = []
    for seed in seeds:
        capacities = capacity_scenario(seed, model, regime=capacity_regime)
        solution = solve_forward(model, capacities, objective)
        decision = np.asarray(solution["x"], dtype=float)
        if decision_noise > 0.0:
            challenger_theta = rng.dirichlet(np.ones(model.n_products))
            challenger = np.asarray(
                solve_forward(model, capacities, challenger_theta)["x"],
                dtype=float,
            )
            # A convex combination of feasible LP decisions remains feasible.
            decision = (1.0 - decision_noise) * decision + decision_noise * challenger
        rows.append(Observation(capacities=capacities, decision=decision))
    return rows
