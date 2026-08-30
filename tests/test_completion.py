import numpy as np

from inverse_industrial.benchmark import run_benchmark
from inverse_industrial.data import build_observations, capacity_scenario
from inverse_industrial.forward import default_model
from inverse_industrial.inverse import recover_objective
from inverse_industrial.statistics import (
    bootstrap_objective_recovery,
    paired_bootstrap_mean_difference,
)


def test_noisy_observations_remain_forward_feasible():
    model = default_model()
    observations = build_observations(
        [0, 1, 2],
        model,
        decision_noise=0.2,
        noise_seed=4,
    )
    for obs in observations:
        assert np.all(obs.decision >= -1e-10)
        assert np.all(obs.decision <= model.upper_bounds + 1e-10)
        assert np.all(model.resource_matrix @ obs.decision <= obs.capacities + 1e-8)


def test_inverse_solver_accepts_noisy_near_optimal_decisions():
    model = default_model()
    observations = build_observations(
        list(range(12)),
        model,
        decision_noise=0.1,
        noise_seed=2,
    )
    result = recover_objective(model, observations)
    assert result.converged
    assert np.all(result.theta >= -1e-9)
    assert abs(float(np.sum(result.theta)) - 1.0) < 1e-8
    assert np.all(result.slacks >= -1e-9)


def test_tight_capacity_regime_is_distinct_and_valid():
    model = default_model()
    nominal = capacity_scenario(8, model, regime="nominal")
    tight = capacity_scenario(8, model, regime="tight_capacity")
    assert nominal.shape == tight.shape == (model.n_resources,)
    assert np.all(tight > 0.0)
    assert not np.allclose(nominal, tight)


def test_final_benchmark_reports_capacity_regime_and_noise():
    model = default_model()
    result = run_benchmark(
        model,
        train_seeds=[0, 1, 2, 3],
        test_seeds=[100, 101],
        decision_noise=0.1,
        noise_seed=7,
        test_capacity_regime="tight_capacity",
    )
    assert result["decision_noise"] == 0.1
    assert result["test_capacity_regime"] == "tight_capacity"
    assert {row.regime for row in result["rows"]} == {"tight_capacity"}


def test_bootstrap_statistics_are_reproducible():
    candidate = np.array([0.0, 0.1, 0.2, 0.0])
    reference = np.array([0.5, 0.4, 0.3, 0.6])
    first = paired_bootstrap_mean_difference(candidate, reference, seed=5)
    second = paired_bootstrap_mean_difference(candidate, reference, seed=5)
    assert first == second

    model = default_model()
    observations = build_observations(list(range(8)), model)
    stability = bootstrap_objective_recovery(
        model,
        observations,
        n_bootstrap=10,
        seed=3,
    )
    assert stability["mean"].shape == (model.n_products,)
    assert stability["ci95_low"].shape == (model.n_products,)
    assert stability["ci95_high"].shape == (model.n_products,)
