"""§7a: the headline experiment — is there a real, measurable arms race?

Runs many independent evolutionary seeds and applies a trend test to:
  1. each species' competence-vs-fixed-reference series (§6), and
  2. the realized catch/escape rate between the actual co-evolving
     populations (§7a point 2).

Architecture §5's definition requires *both*: competence trending up for
both species (real, sustained improvement) *and* realized outcomes staying
roughly flat (neither side actually winning). We use the same tool for
both questions — a trend test — since "is the realized rate stationary"
and "is there a significant trend" are, for this purpose, the same
question asked with the sign of the conclusion flipped: no significant
trend in realized catch/escape rate is the operational definition of
"stationary" used here, consistent with §5 defining the whole thing
operationally rather than relying on a vaguer statistical notion.

Trend test: Hamed & Rao (1998) modified Mann-Kendall, not vanilla
Mann-Kendall. Ordinary Mann-Kendall assumes independent samples; both
series here are adjacent snapshots of a continuously (co-)evolving
population, so they're serially correlated, and vanilla MK would inflate
the false-positive rate for "significant trend" — architecture §10's
"trend test is underpowered" risk cuts both ways: serial correlation can
also manufacture a trend that isn't really there.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pymannkendall as mk
import torch

from redqueen.config import Config
from redqueen.evolve import EvolutionLog, run_evolution

MIN_SERIES_LENGTH = 4  # pymannkendall's variance estimate is degenerate below this


@dataclass
class TrendResult:
    trend: str  # "increasing" / "decreasing" / "no trend"
    significant: bool
    p_value: float
    z: float
    tau: float
    slope: float
    n: int


def trend_test(series: list[float], alpha: float = 0.05) -> TrendResult | None:
    """Hamed-Rao modified Mann-Kendall trend test. Returns None if `series`
    is too short for the test's variance estimate to be meaningful."""
    x = np.asarray(series, dtype=float)
    if len(x) < MIN_SERIES_LENGTH or not np.all(np.isfinite(x)):
        return None
    trend, h, p, z, tau, s, var_s, slope, intercept = mk.hamed_rao_modification_test(x, alpha=alpha)
    if not np.isfinite(p):
        return None
    return TrendResult(
        trend=trend, significant=bool(h), p_value=float(p), z=float(z), tau=float(tau),
        slope=float(slope), n=len(x),
    )


@dataclass
class SeedResult:
    seed: int
    log: EvolutionLog
    predator_competence_series: list[float]
    prey_competence_series: list[float]
    predator_competence_trend: TrendResult | None
    prey_competence_trend: TrendResult | None
    realized_predator_trend: TrendResult | None
    realized_prey_trend: TrendResult | None


def run_seed(
    cfg: Config, seed: int, n_steps: int, log_interval: int = 100, compiled: bool = False
) -> SeedResult:
    cfg_seeded = replace(cfg, seed=seed)
    generator = torch.Generator(device=cfg.device).manual_seed(seed)
    _, log = run_evolution(
        cfg_seeded, generator, n_steps=n_steps, log_interval=log_interval, compiled=compiled
    )

    predator_competence_series = [float(cr.mean()) for cr in log.predator_catch_rate]
    prey_competence_series = [float(er.mean()) for er in log.prey_escape_rate]

    return SeedResult(
        seed=seed,
        log=log,
        predator_competence_series=predator_competence_series,
        prey_competence_series=prey_competence_series,
        predator_competence_trend=trend_test(predator_competence_series),
        prey_competence_trend=trend_test(prey_competence_series),
        realized_predator_trend=trend_test(log.realized_predator_catch_rate),
        realized_prey_trend=trend_test(log.realized_prey_escape_rate),
    )


@dataclass
class ExperimentResult:
    seeds: list[int]
    seed_results: list[SeedResult] = field(default_factory=list)

    def arms_race_verdict(self) -> str:
        """Architecture §5's definition, applied across seeds: an arms race
        is supported only if it holds up consistently, not in a cherry-picked
        seed. Reports what the data actually shows, including a null/mixed
        result — architecture §7a is explicit that this is a legitimate
        outcome, not a failure of the experiment."""
        n = len(self.seed_results)
        if n == 0:
            return "no seeds run"

        def _count(pred) -> int:
            return sum(1 for r in self.seed_results if pred(r))

        competence_both_up = _count(
            lambda r: r.predator_competence_trend is not None
            and r.predator_competence_trend.trend == "increasing"
            and r.prey_competence_trend is not None
            and r.prey_competence_trend.trend == "increasing"
        )
        realized_stationary = _count(
            lambda r: r.realized_predator_trend is not None
            and not r.realized_predator_trend.significant
            and r.realized_prey_trend is not None
            and not r.realized_prey_trend.significant
        )
        arms_race = _count(
            lambda r: r.predator_competence_trend is not None
            and r.predator_competence_trend.trend == "increasing"
            and r.prey_competence_trend is not None
            and r.prey_competence_trend.trend == "increasing"
            and r.realized_predator_trend is not None
            and not r.realized_predator_trend.significant
            and r.realized_prey_trend is not None
            and not r.realized_prey_trend.significant
        )
        return (
            f"{arms_race}/{n} seeds show the full §5 arms-race signature "
            f"(both competence series increasing AND realized rates stationary); "
            f"{competence_both_up}/{n} show both-competence-increasing alone; "
            f"{realized_stationary}/{n} show realized-rate stationarity alone"
        )


def run_experiment(
    cfg: Config, seeds: list[int], n_steps: int, log_interval: int = 100, compiled: bool = False
) -> ExperimentResult:
    result = ExperimentResult(seeds=list(seeds))
    for seed in seeds:
        result.seed_results.append(run_seed(cfg, seed, n_steps, log_interval, compiled=compiled))
    return result
