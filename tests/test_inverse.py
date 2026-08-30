import numpy as np

from inverse_industrial.benchmark import evaluate_theta
from inverse_industrial.data import build_observations, hidden_theta
from inverse_industrial.forward import default_model
from inverse_industrial.inverse import recover_objective


def test_inverse_solution_is_normalized_and_nonnegative():
    model = default_model()
    observations = build_observations(list(range(10)), model)
    result = recover_objective(model, observations)
    assert result.converged
    assert np.all(result.theta >= -1e-9)
    assert abs(float(np.sum(result.theta)) - 1.0) < 1e-8
    assert np.all(result.slacks >= -1e-9)


def test_inverse_objective_has_low_held_out_regret_on_clean_data():
    model = default_model()
    observations = build_observations(list(range(20)), model)
    result = recover_objective(model, observations)
    rows = evaluate_theta(model, result.theta, [100, 101, 102, 103])
    oracle_rows = evaluate_theta(model, hidden_theta(), [100, 101, 102, 103])
    assert np.mean([r.decision_regret for r in rows]) <= 2.0
    assert max(abs(r.decision_regret) for r in oracle_rows) < 1e-8
