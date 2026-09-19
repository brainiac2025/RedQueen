"""The evolution loop (architecture §4) and its logging.

Runs `sim.step` for `n_steps`, periodically recording population-level
summaries (§4's "log per-timestep: population sizes, mean genome per
species, mean energy, birth/death counts") and, once a reference genome has
been frozen, periodic fixed-reference competence checkpoints (§6/§7a).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from redqueen.competence import measure_predator_competence, measure_prey_competence
from redqueen.config import Config
from redqueen.sim import step
from redqueen.world import PopulationState, WorldState, init_world


@dataclass
class EvolutionLog:
    steps: list[int] = field(default_factory=list)
    n_prey: list[int] = field(default_factory=list)
    n_predators: list[int] = field(default_factory=list)
    mean_prey_energy: list[float] = field(default_factory=list)
    mean_predator_energy: list[float] = field(default_factory=list)
    prey_births: list[int] = field(default_factory=list)
    prey_deaths: list[int] = field(default_factory=list)
    predator_births: list[int] = field(default_factory=list)
    predator_deaths: list[int] = field(default_factory=list)
    mean_genome_prey: list[torch.Tensor] = field(default_factory=list)
    mean_genome_predator: list[torch.Tensor] = field(default_factory=list)

    # §6/§7a: fixed-reference competence, sampled every `competence_sample_interval`
    competence_steps: list[int] = field(default_factory=list)
    predator_catch_rate: list[torch.Tensor] = field(default_factory=list)
    prey_escape_rate: list[torch.Tensor] = field(default_factory=list)

    # §7a point 2: realized catch/escape rate between the actual co-evolving
    # populations (not the fixed reference), windowed over each log_interval.
    realized_rate_steps: list[int] = field(default_factory=list)
    realized_predator_catch_rate: list[float] = field(default_factory=list)
    realized_prey_escape_rate: list[float] = field(default_factory=list)

    reference_prey_genome: torch.Tensor | None = None
    reference_predator_genome: torch.Tensor | None = None


def _mean_genome(pop: PopulationState) -> torch.Tensor:
    n_alive = pop.alive.sum().clamp(min=1)
    return (pop.genomes * pop.alive.unsqueeze(-1)).sum(0) / n_alive


def _sample_alive_genomes(
    pop: PopulationState, m: int, generator: torch.Generator
) -> torch.Tensor:
    """m genomes drawn from currently-alive individuals (without replacement
    if there are at least m of them, with replacement otherwise). This runs
    only at competence checkpoints (every `competence_sample_interval` steps,
    e.g. every 2000), not in the per-step hot loop, so the nonzero()-driven
    host sync here is fine — see sim._reproduce's docstring for why it
    wouldn't be fine every step."""
    device = pop.positions.device
    alive_idx = torch.nonzero(pop.alive, as_tuple=True)[0]
    n_alive = alive_idx.shape[0]
    if n_alive >= m:
        perm = torch.randperm(n_alive, device=device, generator=generator)[:m]
        idx = alive_idx[perm]
    else:
        idx = alive_idx[torch.randint(0, n_alive, (m,), device=device, generator=generator)]
    return pop.genomes[idx]


def freeze_reference_genome(pop: PopulationState, generator: torch.Generator) -> torch.Tensor:
    """A single random alive genome, detached from the live population it
    was sampled from (architecture §6: "the reference opponent is frozen")."""
    return _sample_alive_genomes(pop, 1, generator)[0].clone()


def run_evolution(
    cfg: Config,
    generator: torch.Generator,
    n_steps: int | None = None,
    log_interval: int = 100,
    device: str | torch.device | None = None,
    compiled: bool = False,
) -> tuple[WorldState, EvolutionLog]:
    device = cfg.device if device is None else device
    n_steps = cfg.n_steps if n_steps is None else n_steps

    # torch.compile trades ~10-20s upfront (re-)compilation for a ~2-3.5x
    # per-step speedup once warm (see README Status) — worth it once n_steps
    # is in the thousands, not for short test runs.
    step_fn = torch.compile(step) if compiled else step

    state = init_world(cfg, device, generator)
    log = EvolutionLog()
    catches_accum = torch.zeros((), device=device)

    for i in range(n_steps):
        state, stats = step_fn(state, cfg, generator)
        catches_accum = catches_accum + stats.catches

        # `.item()`/`.any()` truthiness checks below are host syncs (same
        # concern as sim._reproduce), but they're gated behind conditions
        # that are only true once (reference freeze) or once every
        # `competence_sample_interval` steps — not every step of the hot loop.
        if log.reference_prey_genome is None and state.step_count >= cfg.reference_burn_in_steps:
            if state.prey.alive.any() and state.predator.alive.any():
                log.reference_prey_genome = freeze_reference_genome(state.prey, generator)
                log.reference_predator_genome = freeze_reference_genome(state.predator, generator)

        if i % log_interval == 0:
            s = stats.item()
            log.steps.append(i)
            log.n_prey.append(s.n_prey)
            log.n_predators.append(s.n_predators)
            log.mean_prey_energy.append(s.mean_prey_energy)
            log.mean_predator_energy.append(s.mean_predator_energy)
            log.prey_births.append(s.prey_births)
            log.prey_deaths.append(s.prey_deaths)
            log.predator_births.append(s.predator_births)
            log.predator_deaths.append(s.predator_deaths)
            log.mean_genome_prey.append(_mean_genome(state.prey).cpu())
            log.mean_genome_predator.append(_mean_genome(state.predator).cpu())

            if i > 0:
                # Denominator uses the population size at the end of this
                # window as a stand-in for the window's average — fine given
                # populations move slowly relative to log_interval in
                # practice, and not worth an extra running-average tensor.
                window_catches = catches_accum.item()
                log.realized_rate_steps.append(i)
                log.realized_predator_catch_rate.append(
                    window_catches / (log_interval * max(s.n_predators, 1))
                )
                log.realized_prey_escape_rate.append(
                    1.0 - window_catches / (log_interval * max(s.n_prey, 1))
                )
            catches_accum = torch.zeros((), device=device)

        if (
            log.reference_prey_genome is not None
            and i > 0
            and i % cfg.competence_sample_interval == 0
            and state.prey.alive.any()
            and state.predator.alive.any()
        ):
            predator_sample = _sample_alive_genomes(
                state.predator, cfg.competence_n_trials, generator
            )
            prey_sample = _sample_alive_genomes(state.prey, cfg.competence_n_trials, generator)
            catch_rate = measure_predator_competence(
                predator_sample, log.reference_prey_genome, cfg, generator
            )
            escape_rate = measure_prey_competence(
                prey_sample, log.reference_predator_genome, cfg, generator
            )
            log.competence_steps.append(i)
            log.predator_catch_rate.append(catch_rate.cpu())
            log.prey_escape_rate.append(escape_rate.cpu())

    return state, log
