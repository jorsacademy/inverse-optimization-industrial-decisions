from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Observation, build_observations, capacity_scenario, hidden_theta
from .forward import ProductionModel, solve_forward
from .inverse import InverseResult, recover_objective


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    seed: int
    decision_regret: float
    decision_l1: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_theta(
    model: ProductionModel,
    theta_hat: np.ndarray,
    test_seeds: list[int],
) -> list[BenchmarkRow]:
    true_theta = hidden_theta()
    rows = []
    for seed in test_seeds:
        capacities = capacity_scenario(seed, model)
        truth = solve_forward(model, capacities, true_theta)
        predicted = solve_forward(model, capacities, theta_hat)
        true_x = np.asarray(truth["x"])
        predicted_x = np.asarray(predicted["x"])
        regret = float(true_theta @ true_x - true_theta @ predicted_x)
        rows.append(BenchmarkRow("inverse", seed, regret, float(np.sum(np.abs(true_x - predicted_x)))))
    return rows


def run_benchmark(
    model: ProductionModel,
    *,
    train_seeds: list[int],
    test_seeds: list[int],
) -> dict[str, object]:
    observations: list[Observation] = build_observations(train_seeds, model)
    inverse: InverseResult = recover_objective(model, observations)
    uniform = np.full(model.n_products, 1.0 / model.n_products)
    rng = np.random.default_rng(0)
    random_theta = rng.dirichlet(np.ones(model.n_products))
    true_theta = hidden_theta()

    methods = {"inverse": inverse.theta, "uniform": uniform, "random": random_theta}
    rows: list[BenchmarkRow] = []
    for method, theta in methods.items():
        method_rows = evaluate_theta(model, theta, test_seeds)
        rows.extend(BenchmarkRow(method, r.seed, r.decision_regret, r.decision_l1) for r in method_rows)

    return {
        "inverse_result": inverse,
        "true_theta": true_theta,
        "cosine_similarity": cosine_similarity(inverse.theta, true_theta),
        "l1_theta_error": float(np.sum(np.abs(inverse.theta - true_theta))),
        "rows": rows,
    }


def summarize(rows: list[BenchmarkRow]) -> list[dict[str, float | str]]:
    output = []
    for method in sorted({r.method for r in rows}):
        selected = [r for r in rows if r.method == method]
        output.append(
            {
                "method": method,
                "mean_decision_regret": float(np.mean([r.decision_regret for r in selected])),
                "mean_decision_l1": float(np.mean([r.decision_l1 for r in selected])),
            }
        )
    return output
