from __future__ import annotations

import numpy as np

from .benchmark import BenchmarkRow, run_benchmark, summarize
from .data import build_observations
from .forward import default_model
from .statistics import bootstrap_objective_recovery, paired_decision_report


def _paired(rows: list[BenchmarkRow], candidate: str, reference: str) -> dict[str, float | int]:
    candidate_rows = sorted([r for r in rows if r.method == candidate], key=lambda r: r.seed)
    reference_rows = sorted([r for r in rows if r.method == reference], key=lambda r: r.seed)
    if [r.seed for r in candidate_rows] != [r.seed for r in reference_rows]:
        raise ValueError("paired methods must share identical test seeds")
    return paired_decision_report(
        np.asarray([r.decision_regret for r in candidate_rows]),
        np.asarray([r.decision_regret for r in reference_rows]),
        seed=17,
    )


def _print_result(name: str, result: dict[str, object]) -> None:
    inverse = result["inverse_result"]
    print(f"split={name}")
    print(f"theta_hat={inverse.theta.tolist()}")
    print(
        f"cosine={result['cosine_similarity']:.6f},"
        f"theta_l1={result['l1_theta_error']:.6f},"
        f"mean_slack={result['mean_training_slack']:.6f},"
        f"max_slack={result['max_training_slack']:.6f},"
        f"iterations={inverse.iterations},challengers={inverse.challenger_count}"
    )
    rows = result["rows"]
    for row in summarize(rows):
        print(
            f"{row['regime']},{row['method']},"
            f"regret={row['mean_decision_regret']:.6f},"
            f"decision_l1={row['mean_decision_l1']:.6f}"
        )
    for reference in ["uniform", "random"]:
        report = _paired(rows, "inverse", reference)
        print(
            f"paired,inverse-{reference},"
            f"mean_diff={report['mean_regret_difference']:.6f},"
            f"ci95=[{report['ci95_low']:.6f},{report['ci95_high']:.6f}],"
            f"win_rate={report['win_rate']:.3f},"
            f"p={report['sign_test_pvalue']:.4f}"
        )


def main() -> None:
    model = default_model()
    train_seeds = list(range(30))
    nominal_test = list(range(100, 112))
    ood_test = list(range(200, 212))

    clean_nominal = run_benchmark(
        model,
        train_seeds=train_seeds,
        test_seeds=nominal_test,
        decision_noise=0.0,
        test_capacity_regime="nominal",
    )
    noisy_nominal = run_benchmark(
        model,
        train_seeds=train_seeds,
        test_seeds=nominal_test,
        decision_noise=0.12,
        noise_seed=9,
        test_capacity_regime="nominal",
    )
    noisy_ood = run_benchmark(
        model,
        train_seeds=train_seeds,
        test_seeds=ood_test,
        decision_noise=0.12,
        noise_seed=9,
        test_capacity_regime="tight_capacity",
    )

    _print_result("clean_train_nominal_test", clean_nominal)
    _print_result("noisy_train_nominal_test", noisy_nominal)
    _print_result("noisy_train_tight_capacity_ood", noisy_ood)

    noisy_observations = build_observations(
        train_seeds,
        model,
        decision_noise=0.12,
        noise_seed=9,
    )
    stability = bootstrap_objective_recovery(
        model,
        noisy_observations,
        n_bootstrap=20,
        seed=33,
    )
    print(f"theta_bootstrap_mean={stability['mean'].tolist()}")
    print(f"theta_bootstrap_ci95_low={stability['ci95_low'].tolist()}")
    print(f"theta_bootstrap_ci95_high={stability['ci95_high'].tolist()}")


if __name__ == "__main__":
    main()
