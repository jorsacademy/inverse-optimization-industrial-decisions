from __future__ import annotations

from .benchmark import run_benchmark, summarize
from .forward import default_model


def main() -> None:
    result = run_benchmark(
        default_model(),
        train_seeds=list(range(20)),
        test_seeds=[100, 101, 102, 103, 104, 105],
    )
    inverse = result["inverse_result"]
    print(f"theta_hat={inverse.theta.tolist()}")
    print(f"iterations={inverse.iterations},challengers={inverse.challenger_count}")
    print(
        f"cosine_similarity={result['cosine_similarity']:.6f},"
        f"l1_theta_error={result['l1_theta_error']:.6f}"
    )
    for row in summarize(result["rows"]):
        print(
            f"{row['method']},decision_regret={row['mean_decision_regret']:.6f},"
            f"decision_l1={row['mean_decision_l1']:.6f}"
        )


if __name__ == "__main__":
    main()
