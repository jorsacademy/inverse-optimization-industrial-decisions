from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ProductionModel:
    resource_matrix: np.ndarray
    upper_bounds: np.ndarray

    @property
    def n_products(self) -> int:
        return int(self.resource_matrix.shape[1])

    @property
    def n_resources(self) -> int:
        return int(self.resource_matrix.shape[0])


def default_model() -> ProductionModel:
    return ProductionModel(
        resource_matrix=np.array(
            [
                [2.0, 1.0, 3.0, 2.5],
                [1.0, 2.5, 1.2, 2.0],
                [1.5, 1.0, 2.2, 0.8],
            ],
            dtype=float,
        ),
        upper_bounds=np.array([32.0, 28.0, 24.0, 30.0], dtype=float),
    )


def solve_forward(
    model: ProductionModel,
    capacities: np.ndarray,
    theta: np.ndarray,
) -> dict[str, np.ndarray | float]:
    capacities = np.asarray(capacities, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if capacities.shape != (model.n_resources,):
        raise ValueError("capacities have wrong shape")
    if theta.shape != (model.n_products,):
        raise ValueError("theta has wrong shape")
    if np.any(capacities < 0.0):
        raise ValueError("capacities must be nonnegative")
    result = linprog(
        -theta,
        A_ub=model.resource_matrix,
        b_ub=capacities,
        bounds=[(0.0, float(ub)) for ub in model.upper_bounds],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"forward LP failed: {result.message}")
    x = np.asarray(result.x, dtype=float)
    return {"x": x, "objective": float(theta @ x)}
