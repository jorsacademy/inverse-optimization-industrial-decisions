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


def capacity_scenario(seed: int, model: ProductionModel) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([95.0, 88.0, 72.0], dtype=float)
    shock = rng.normal(0.0, [12.0, 10.0, 8.0])
    capacities = np.maximum(base + shock, np.array([45.0, 40.0, 35.0]))
    if capacities.shape != (model.n_resources,):
        raise ValueError("default capacity generator assumes three resources")
    return capacities


def build_observations(
    seeds: list[int],
    model: ProductionModel,
    theta: np.ndarray | None = None,
) -> list[Observation]:
    objective = hidden_theta() if theta is None else np.asarray(theta, dtype=float)
    rows = []
    for seed in seeds:
        capacities = capacity_scenario(seed, model)
        solution = solve_forward(model, capacities, objective)
        rows.append(Observation(capacities=capacities, decision=np.asarray(solution["x"])))
    return rows
