"""Batched simulation step (architecture §3).

Every function here operates on the *whole* population as tensor ops — no
Python loop over individual agents. Dead slots (`alive=False`) are carried
along and masked out rather than removed, so tensor shapes never change
across the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from redqueen.config import Config
from redqueen.controller import batched_forward
from redqueen.genome import genome_dim, mutate, unpack
from redqueen.world import PopulationState, WorldState


@dataclass
class StepStats:
    """Fields are 0-dim tensors, not Python scalars.

    Converting to Python (`.item()`) forces a GPU->CPU sync; doing that every
    step is exactly the throughput killer architecture §10 warns about ("no
    `.item()`/host syncs inside the stepped loop"). Callers running a tight
    simulation loop should ignore these until an actual logging point; call
    `.item()` on this object to get a plain-value copy for printing/asserting.
    """

    n_prey: torch.Tensor
    n_predators: torch.Tensor
    prey_births: torch.Tensor
    prey_deaths: torch.Tensor
    predator_births: torch.Tensor
    predator_deaths: torch.Tensor
    mean_prey_energy: torch.Tensor
    mean_predator_energy: torch.Tensor
    catches: torch.Tensor  # predation events this step (distinct from prey_deaths, which also counts starvation)

    def item(self) -> "StepStats":
        return StepStats(**{k: v.item() for k, v in vars(self).items()})


def torus_delta(a: torch.Tensor, b: torch.Tensor, world_size: float) -> torch.Tensor:
    """a: [Na,2], b: [Nb,2] -> minimum-image delta [Na,Nb,2] under wraparound."""
    delta = a.unsqueeze(1) - b.unsqueeze(0)
    return delta - world_size * torch.round(delta / world_size)


def topk_relative_features(
    self_pos: torch.Tensor,
    other_pos: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
    world_size: float,
    max_range: float | None = None,
) -> torch.Tensor:
    """Egocentric (dx, dy, dist) to the k nearest valid `other` entities, per row of `self_pos`.

    valid_mask: [Na, Nb] bool, e.g. alive & not-self & (for food) has-food.
    `max_range`: entities farther than this are treated the same as absent
    (zero-padded), i.e. this is a hard perceptual range, not just "closest k
    regardless of distance". None means unlimited range.
    Rows/entities with fewer than k valid neighbours are zero-padded.
    Returns [Na, k*3].

    Shared by `compute_sensory_inputs` (the live ecological sim) and
    `competence.py`'s fixed-reference trials, which need the same egocentric
    sensing but without a slot-array `alive` mask.
    """
    na = self_pos.shape[0]
    delta = torus_delta(self_pos, other_pos, world_size)  # [Na, Nb, 2]
    dist = torch.linalg.norm(delta, dim=-1)  # [Na, Nb]
    dist = dist.masked_fill(~valid_mask, float("inf"))
    if max_range is not None:
        dist = dist.masked_fill(dist > max_range, float("inf"))

    k = min(k, dist.shape[1])
    topk_dist, idx = torch.topk(dist, k, dim=-1, largest=False)  # [Na, k]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, 2)
    topk_delta = torch.gather(delta, 1, idx_exp)  # [Na, k, 2]

    valid = torch.isfinite(topk_dist)
    topk_delta = torch.where(valid.unsqueeze(-1), topk_delta, torch.zeros_like(topk_delta))
    topk_dist = torch.where(valid, topk_dist, torch.zeros_like(topk_dist))

    feat = torch.cat([topk_delta, topk_dist.unsqueeze(-1)], dim=-1)  # [Na, k, 3]
    return feat.reshape(na, k * 3)


def compute_sensory_inputs(
    state: WorldState, cfg: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (obs_prey [max_prey, obs_dim], obs_pred [max_predators, obs_dim])."""
    k = cfg.k_neighbours
    ws = cfg.world_size
    prey, pred = state.prey, state.predator

    n_prey, n_pred = prey.max_n, pred.max_n
    eye_prey = torch.eye(n_prey, dtype=torch.bool, device=prey.positions.device)
    eye_pred = torch.eye(n_pred, dtype=torch.bool, device=pred.positions.device)

    food_has = state.food_amount > 0

    r_prey = cfg.sensing_range_prey
    r_pred = cfg.sensing_range_predator

    # --- prey observations ---
    valid_pp = prey.alive[None, :] & prey.alive[:, None] & ~eye_prey
    feat_prey_same = topk_relative_features(prey.positions, prey.positions, valid_pp, k, ws, r_prey)

    valid_pd = prey.alive[:, None] & pred.alive[None, :]
    feat_prey_opp = topk_relative_features(prey.positions, pred.positions, valid_pd, k, ws, r_prey)

    valid_pf = prey.alive[:, None] & food_has[None, :]
    feat_prey_food = topk_relative_features(
        prey.positions, state.food_positions, valid_pf, k, ws, r_prey
    )

    energy_norm_prey = (prey.energy / cfg.reproduction_threshold).clamp(0, 1).unsqueeze(-1)
    obs_prey = torch.cat(
        [feat_prey_same, feat_prey_opp, feat_prey_food, energy_norm_prey], dim=-1
    )

    # --- predator observations ---
    valid_dd = pred.alive[None, :] & pred.alive[:, None] & ~eye_pred
    feat_pred_same = topk_relative_features(pred.positions, pred.positions, valid_dd, k, ws, r_pred)

    valid_dp = pred.alive[:, None] & prey.alive[None, :]
    feat_pred_opp = topk_relative_features(pred.positions, prey.positions, valid_dp, k, ws, r_pred)

    feat_pred_food = torch.zeros((n_pred, k * 3), device=pred.positions.device)  # no food access

    energy_norm_pred = (pred.energy / cfg.reproduction_threshold).clamp(0, 1).unsqueeze(-1)
    obs_pred = torch.cat(
        [feat_pred_same, feat_pred_opp, feat_pred_food, energy_norm_pred], dim=-1
    )

    return obs_prey, obs_pred


def step_controllers(obs: torch.Tensor, genomes: torch.Tensor, cfg: Config) -> torch.Tensor:
    """obs: [N, obs_dim], genomes: [N, genome_dim] -> actions [N, 2] (turn_rate in [-1,1], speed in [0,1])."""
    params = unpack(genomes, cfg)
    return batched_forward(params, obs)


def apply_movement(
    positions: torch.Tensor,
    headings: torch.Tensor,
    actions: torch.Tensor,
    max_speed: float,
    cfg: Config,
    alive: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """actions: [N,2] (turn_rate in [-1,1], speed in [0,1]) -> updated (positions, headings).

    `alive=None` moves every row unconditionally (used by competence.py's
    trial arenas, which have no slot-array death/birth to mask around).
    """
    turn = actions[:, 0] * cfg.max_turn_rate
    speed = actions[:, 1] * max_speed
    new_headings = headings + turn
    dx = torch.cos(new_headings) * speed
    dy = torch.sin(new_headings) * speed
    step_vec = torch.stack([dx, dy], dim=-1)
    new_positions = (positions + step_vec) % cfg.world_size

    if alive is not None:
        new_headings = torch.where(alive, new_headings, headings)
        new_positions = torch.where(alive.unsqueeze(-1), new_positions, positions)
    return new_positions, new_headings


def _reproduce(
    pop: PopulationState,
    cfg: Config,
    sigma: float,
    generator: torch.Generator,
    next_genome_id: torch.Tensor,
) -> tuple[PopulationState, torch.Tensor]:
    """Alive agents above the reproduction threshold spawn a mutated offspring
    into a free slot, splitting energy with the parent. When more agents
    qualify than there are free slots, which ones actually give birth this
    step is a uniform-random draw among the qualifiers (not fitness-directed
    — architecture §3's culling philosophy applied to the birth side of the
    same constraint).

    Deliberately avoids `nonzero`/boolean-mask indexing/`.item()` — those are
    the classic hidden GPU->CPU sync points (data-dependent output shape), and
    doing one every step is what turned a "few thousand steps/sec" simulation
    into ~160 steps/sec in practice. Everything below is a static-shape
    cumsum/argsort/gather/scatter over the full [max_n] slot array instead.
    """
    device = pop.positions.device
    n = pop.max_n

    repro_mask = pop.alive & (pop.energy > cfg.reproduction_threshold)
    free_mask = ~pop.alive

    # Random priority breaks ties among reproducers only, so scarce free
    # slots go to a uniform-random subset rather than "lowest index wins".
    priority = torch.rand(n, device=device, generator=generator)
    priority = torch.where(repro_mask, priority, torch.full_like(priority, 2.0))
    order = torch.argsort(priority)  # reproducers first (priority<1), random order
    rank_of_index = torch.empty(n, dtype=torch.long, device=device)
    rank_of_index[order] = torch.arange(n, device=device)
    # For a reproducer this is its 0-indexed rank among reproducers, since
    # reproducers occupy exactly the first len(reproducers) slots of `order`.

    total_repro = repro_mask.sum()

    dump = n  # scratch slot absorbing writes from non-reproducers
    buf = torch.zeros(n + 1, dtype=torch.long, device=device)
    target = torch.where(repro_mask, rank_of_index, torch.full_like(rank_of_index, dump))
    buf.scatter_(0, target, torch.arange(n, device=device))
    reproducer_index_by_rank = buf[:n]  # [r] = original index of the r-th reproducer

    rank_free = torch.cumsum(free_mask.long(), dim=0) - 1
    slot_gets_birth = free_mask & (rank_free >= 0) & (rank_free < total_repro)
    safe_rank_free = rank_free.clamp(min=0, max=n - 1)
    source_parent_idx = reproducer_index_by_rank[safe_rank_free]  # [n], gather

    child_energy = pop.energy[source_parent_idx] * cfg.reproduction_cost_fraction
    debit = torch.where(slot_gets_birth, -child_energy, torch.zeros_like(pop.energy))
    new_energy = pop.energy.scatter_add(0, source_parent_idx, debit)
    new_energy = torch.where(slot_gets_birth, child_energy, new_energy)

    new_positions = torch.where(
        slot_gets_birth.unsqueeze(-1), pop.positions[source_parent_idx], pop.positions
    )
    new_headings = torch.where(slot_gets_birth, pop.headings[source_parent_idx], pop.headings)
    mutated = mutate(pop.genomes[source_parent_idx], sigma, generator)
    new_genomes = torch.where(slot_gets_birth.unsqueeze(-1), mutated, pop.genomes)
    new_alive = pop.alive | slot_gets_birth

    birth_rank = torch.cumsum(slot_gets_birth.long(), dim=0) - 1
    new_id = next_genome_id + birth_rank
    new_genome_id = torch.where(slot_gets_birth, new_id, pop.genome_id)
    next_genome_id = next_genome_id + slot_gets_birth.sum()

    pop.positions = new_positions
    pop.headings = new_headings
    pop.energy = new_energy
    pop.genomes = new_genomes
    pop.alive = new_alive
    pop.genome_id = new_genome_id

    return pop, next_genome_id


def step_physics(
    state: WorldState,
    actions_prey: torch.Tensor,
    actions_pred: torch.Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> tuple[WorldState, StepStats]:
    prey, pred = state.prey, state.predator
    ws = cfg.world_size

    n_prey0 = prey.alive.sum()
    n_pred0 = pred.alive.sum()

    # --- movement ---
    for pop, actions, max_speed in (
        (prey, actions_prey, cfg.max_speed_prey),
        (pred, actions_pred, cfg.max_speed_predator),
    ):
        pop.positions, pop.headings = apply_movement(
            pop.positions, pop.headings, actions, max_speed, cfg, alive=pop.alive
        )

    # --- metabolism ---
    prey.energy = torch.where(prey.alive, prey.energy - cfg.metabolism_prey, prey.energy)
    pred.energy = torch.where(pred.alive, pred.energy - cfg.metabolism_predator, pred.energy)

    # --- prey eats food (nearest prey claims each patch within eat_radius) ---
    dist_pf = torch.linalg.norm(torus_delta(prey.positions, state.food_positions, ws), dim=-1)
    valid_pf = prey.alive[:, None] & (state.food_amount[None, :] > 0) & (dist_pf < cfg.eat_radius)
    dist_pf_masked = dist_pf.masked_fill(~valid_pf, float("inf"))
    min_dist_per_food, claimant_prey = dist_pf_masked.min(dim=0)  # [n_food]
    food_eaten = torch.isfinite(min_dist_per_food)
    consumed = torch.where(food_eaten, state.food_amount, torch.zeros_like(state.food_amount))
    # Always scatter (contribution 0 where nothing was eaten) rather than
    # guarding with `if food_eaten.any():` — that guard and boolean-mask
    # indexing are both host syncs; scatter_add with an all-zero contribution
    # to bogus indices is a correct, sync-free no-op.
    prey.energy = prey.energy.scatter_add(0, claimant_prey, consumed)
    state.food_amount = torch.where(food_eaten, torch.zeros_like(state.food_amount), state.food_amount)
    state.food_amount = (state.food_amount + cfg.food_regen_rate).clamp(max=cfg.food_energy)

    # --- predator eats prey (nearest predator claims each prey within eat_radius) ---
    dist_dp = torch.linalg.norm(torus_delta(pred.positions, prey.positions, ws), dim=-1)
    valid_dp = pred.alive[:, None] & prey.alive[None, :] & (dist_dp < cfg.eat_radius)
    dist_dp_masked = dist_dp.masked_fill(~valid_dp, float("inf"))
    min_dist_per_prey, claimant_pred = dist_dp_masked.min(dim=0)  # [n_prey]
    prey_eaten = torch.isfinite(min_dist_per_prey)
    catch_contribution = torch.where(
        prey_eaten,
        torch.full_like(min_dist_per_prey, cfg.predator_catch_energy),
        torch.zeros_like(min_dist_per_prey),
    )
    pred.energy = pred.energy.scatter_add(0, claimant_pred, catch_contribution)
    catches = prey_eaten.sum()  # genuine predation events, distinct from prey_deaths (also counts starvation)
    prey.alive = prey.alive & ~prey_eaten

    # Satiation cap: once a population sits at its slot cap, agents that clear
    # the reproduction threshold but find no free slot would otherwise keep
    # accumulating energy forever (observed empirically: mean predator energy
    # climbing unboundedly once max_predators was reached). Real organisms
    # can't store unlimited energy either.
    prey.energy = prey.energy.clamp(max=cfg.max_energy)
    pred.energy = pred.energy.clamp(max=cfg.max_energy)

    # --- death from starvation ---
    prey.alive = prey.alive & (prey.energy > 0)
    pred.alive = pred.alive & (pred.energy > 0)

    prey_deaths = n_prey0 - prey.alive.sum()
    pred_deaths = n_pred0 - pred.alive.sum()

    # --- reproduction ---
    n_prey_before_repro = prey.alive.sum()
    n_pred_before_repro = pred.alive.sum()
    state.prey, state.next_genome_id = _reproduce(
        prey, cfg, cfg.sigma_mutation, generator, state.next_genome_id
    )
    state.predator, state.next_genome_id = _reproduce(
        pred, cfg, cfg.sigma_mutation, generator, state.next_genome_id
    )

    prey_births = state.prey.alive.sum() - n_prey_before_repro
    pred_births = state.predator.alive.sum() - n_pred_before_repro

    state.step_count += 1

    n_prey_final = state.prey.alive.sum()
    n_pred_final = state.predator.alive.sum()
    mean_prey_energy = (state.prey.energy * state.prey.alive).sum() / n_prey_final.clamp(min=1)
    mean_predator_energy = (state.predator.energy * state.predator.alive).sum() / n_pred_final.clamp(
        min=1
    )
    stats = StepStats(
        n_prey=n_prey_final,
        n_predators=n_pred_final,
        prey_births=prey_births,
        prey_deaths=prey_deaths,
        predator_births=pred_births,
        predator_deaths=pred_deaths,
        mean_prey_energy=mean_prey_energy,
        mean_predator_energy=mean_predator_energy,
        catches=catches,
    )
    return state, stats


def step(state: WorldState, cfg: Config, generator: torch.Generator) -> tuple[WorldState, StepStats]:
    obs_prey, obs_pred = compute_sensory_inputs(state, cfg)
    actions_prey = step_controllers(obs_prey, state.prey.genomes, cfg)
    actions_pred = step_controllers(obs_pred, state.predator.genomes, cfg)
    return step_physics(state, actions_prey, actions_pred, cfg, generator)
