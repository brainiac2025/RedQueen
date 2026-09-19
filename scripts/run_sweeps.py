"""Architecture §7b's three supporting sweeps.

    python scripts/run_sweeps.py --out-dir results/sweeps

Each sweep variant runs `scripts/run_experiment.py` as a **separate
subprocess** rather than looping in-process. This isn't just tidiness: an
earlier in-process version of this script hit a real torch.compile
correctness bug on this Windows + community-triton-windows setup, where
compiling a *different*-shaped Config earlier in the same process silently
corrupted results for a subsequently-repeated shape (verified: identical
seed + identical config gave wildly different trajectories depending on
what other shapes had been compiled first in that process; a completely
fresh process for the same config reproduced bit-identically). A fresh
subprocess per variant means a fresh torch.compile cache every time, which
sidesteps the bug entirely - at the cost of paying each shape's ~15-20s
compile warmup once per variant instead of once per unique shape.

1. Mutation rate: sigma_mutation in {low, med, high} - does escalation (or
   its absence) require a specific mutation-rate regime?
2. Population size: small/medium/large, scaling world_size and
   food_max_patches to hold density roughly constant, so only genetic
   diversity per generation varies, not crowding.
3. Sensing-range asymmetry: baseline (equal ranges) vs. a predator
   advantage vs. a prey advantage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from redqueen.config import Config

RUN_EXPERIMENT = str(Path(__file__).parent / "run_experiment.py")


def _run_variant(
    name: str,
    out_dir: Path,
    seeds: list[int],
    n_steps: int,
    log_interval: int,
    device: str,
    **cfg_overrides: float | int,
) -> dict:
    out_path = out_dir / f"{name}.json"
    cmd = [
        sys.executable,
        RUN_EXPERIMENT,
        "--seeds",
        *[str(s) for s in seeds],
        "--n-steps",
        str(n_steps),
        "--log-interval",
        str(log_interval),
        "--device",
        device,
        "--out",
        str(out_path),
    ]
    for key, value in cfg_overrides.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  [{name}] FAILED after {dt:.1f}s:\n{proc.stderr[-2000:]}")
        raise RuntimeError(f"variant {name} failed (see stderr above)")

    with open(out_path) as f:
        data = json.load(f)
    print(f"  [{name}] done in {dt:.1f}s: {data['arms_race_verdict']}")
    return data


def mutation_rate_sweep(out_dir, seeds, n_steps, log_interval, device) -> list[dict]:
    print("=== Sweep 1: mutation rate ===")
    variants = {"low_sigma_0.02": 0.02, "med_sigma_0.05": 0.05, "high_sigma_0.15": 0.15}
    return [
        _run_variant(name, out_dir, seeds, n_steps, log_interval, device, sigma_mutation=sigma)
        for name, sigma in variants.items()
    ]


def population_size_sweep(out_dir, seeds, n_steps, log_interval, device) -> list[dict]:
    print("=== Sweep 2: population size (density held ~constant) ===")
    variants = {
        "small_0.5x": (1000, 100, 70.7, 200),
        "medium_1x_baseline": (2000, 200, 100.0, 400),
        "large_2x": (4000, 400, 141.4, 800),
    }
    results = []
    for name, (max_prey, max_predators, world_size, food_max_patches) in variants.items():
        results.append(
            _run_variant(
                name, out_dir, seeds, n_steps, log_interval, device,
                max_prey=max_prey, max_predators=max_predators,
                world_size=world_size, food_max_patches=food_max_patches,
            )
        )
    return results


def sensing_range_asymmetry_sweep(out_dir, seeds, n_steps, log_interval, device) -> list[dict]:
    print("=== Sweep 3: sensing-range asymmetry ===")
    variants = {
        "baseline_equal_20_20": (20.0, 20.0),
        "predator_advantage_15prey_30pred": (15.0, 30.0),
        "prey_advantage_30prey_15pred": (30.0, 15.0),
    }
    results = []
    for name, (r_prey, r_pred) in variants.items():
        results.append(
            _run_variant(
                name, out_dir, seeds, n_steps, log_interval, device,
                sensing_range_prey=r_prey, sensing_range_predator=r_pred,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=Config().device)
    parser.add_argument("--out-dir", type=str, default="results/sweeps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-steps", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=250)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary = {
        "seeds": args.seeds,
        "n_steps": args.n_steps,
        "mutation_rate_sweep": mutation_rate_sweep(
            out_dir, args.seeds, args.n_steps, args.log_interval, args.device
        ),
        "population_size_sweep": population_size_sweep(
            out_dir, args.seeds, args.n_steps, args.log_interval, args.device
        ),
        "sensing_range_asymmetry_sweep": sensing_range_asymmetry_sweep(
            out_dir, args.seeds, args.n_steps, args.log_interval, args.device
        ),
    }
    dt = time.time() - t0
    summary["total_time_s"] = dt
    print(f"\ntotal sweep time: {dt:.1f}s ({dt / 60:.1f} min)")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved to {out_dir / 'summary.json'} (per-variant detail alongside it)")


if __name__ == "__main__":
    main()
