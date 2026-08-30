import numpy as np

from inverse_industrial.data import capacity_scenario, hidden_theta
from inverse_industrial.forward import default_model, solve_forward


def test_forward_solution_is_feasible():
    model = default_model()
    capacities = capacity_scenario(7, model)
    result = solve_forward(model, capacities, hidden_theta())
    x = np.asarray(result["x"])
    assert np.all(x >= -1e-9)
    assert np.all(x <= model.upper_bounds + 1e-9)
    assert np.all(model.resource_matrix @ x <= capacities + 1e-8)


def test_capacity_scenarios_are_reproducible():
    model = default_model()
    assert np.allclose(capacity_scenario(3, model), capacity_scenario(3, model))
