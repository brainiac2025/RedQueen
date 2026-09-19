"""One-command entry point for architecture §7a's headline experiment.

    python scripts/run_experiment.py --seeds 0 1 2 3 4 --n-steps 100000 \
        --out results/seeds0-4_n100000.json

Runs `redqueen.analysis.run_experiment` (independent evolutionary seeds,
each producing a fixed-reference competence series per species and a
realized catch/escape rate series, both put through the Hamed-Rao modified
Mann-Kendall trend test) and writes the per-seed series and trend results
to a JSON file, plus a plain-text verdict to stdout.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from redqueen.analysis import run_experiment
from redqueen.config import Config


def _trend_dict(trend) -> dict | None:
    return asdict(trend) if trend is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-steps", type=int, default=Config().n_steps)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default=Config().device)
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    parser.add_argument("--out", type=str, default="results/experiment.json")
    args = parser.parse_args()

    cfg = Config(device=args.device)

    t0 = time.time()
    result = run_experiment(
        cfg,
        seeds=args.seeds,
        n_steps=args.n_steps,
        log_interval=args.log_interval,
        compiled=not args.no_compile,
    )
    dt = time.time() - t0

    print(f"total experiment time: {dt:.1f}s ({dt / 60:.1f} min)")
    print()
    print(result.arms_race_verdict())

    out = {
        "seeds": args.seeds,
        "n_steps": args.n_steps,
        "total_time_s": dt,
        "arms_race_verdict": result.arms_race_verdict(),
        "results": [],
    }
    for sr in result.seed_results:
        out["results"].append(
            {
                "seed": sr.seed,
                "steps": sr.log.steps,
                "n_prey_series": sr.log.n_prey,
                "n_predators_series": sr.log.n_predators,
                "competence_steps": sr.log.competence_steps,
                "predator_competence_series": sr.predator_competence_series,
                "prey_competence_series": sr.prey_competence_series,
                "predator_competence_trend": _trend_dict(sr.predator_competence_trend),
                "prey_competence_trend": _trend_dict(sr.prey_competence_trend),
                "realized_rate_steps": sr.log.realized_rate_steps,
                "realized_predator_catch_rate": sr.log.realized_predator_catch_rate,
                "realized_prey_escape_rate": sr.log.realized_prey_escape_rate,
                "realized_predator_trend": _trend_dict(sr.realized_predator_trend),
                "realized_prey_trend": _trend_dict(sr.realized_prey_trend),
                "final_n_prey": sr.log.n_prey[-1],
                "final_n_predators": sr.log.n_predators[-1],
            }
        )

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
