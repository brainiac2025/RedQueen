"""Fixed-reference competence trials (architecture §6).

Genome weights aren't interpretable, so "how good is this predator/prey"
is measured functionally: drop a sample of evolved genomes into a small,
standardized arena against a population of a single *frozen* reference
genome, and count catches over a fixed number of steps.

The reference genome is frozen once per run (see `evolve.freeze_reference`)
so that testing evolving predators against it isolates *predator*
improvement from any co-evolutionary shift in the prey they face
day-to-day — the whole point of doing this instead of just reading catch
rates off the live, co-evolving population.

This trial arena is deliberately not the full ecological simulation: no
food, no energy-driven death, no reproduction. Those would confound "is
this genome good at chasing/evading" with "did this genome get lucky with
the food supply this trial." The side playing prey respawns instantly at a
random position when caught, so a bad predator sample doesn't get starved
into an easy trial by depleting the prey it's being scored against.
"""

from __future__ import annotations

import torch

from redqueen.config import Config
from redqueen.controller import batched_forward
from redqueen.genome import unpack
from redqueen.sim import apply_movement, torus_delta, topk_relative_features


def _random_positions(
    n: int, world_size: float, device: str | torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.rand((n, 2), device=device, generator=generator) * world_size
    headings = torch.rand((n,), device=device, generator=generator) * 2 * torch.pi
    return positions, headings


def _trial_obs(
    pred_pos: torch.Tensor, prey_pos: torch.Tensor, cfg: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same egocentric sensing as `sim.compute_sensory_inputs`, but for a
    trial arena with no slot array / alive mask (everyone here is alive) and
    no food (both food feature blocks are zero-padded, same as how the main
    sim already zero-pads predators' food sensing)."""
    device = pred_pos.device
    k = cfg.k_neighbours
    ws = cfg.world_size
    r_prey = cfg.sensing_range_prey
    r_pred = cfg.sensing_range_predator
    n_pred, n_prey = pred_pos.shape[0], prey_pos.shape[0]

    eye_pred = torch.eye(n_pred, dtype=torch.bool, device=device)
    eye_dp = torch.ones((n_pred, n_prey), dtype=torch.bool, device=device)
    feat_pred_same = topk_relative_features(pred_pos, pred_pos, ~eye_pred, k, ws, r_pred)
    feat_pred_opp = topk_relative_features(pred_pos, prey_pos, eye_dp, k, ws, r_pred)
    feat_pred_food = torch.zeros((n_pred, k * 3), device=device)
    energy_pred = torch.full((n_pred, 1), 0.5, device=device)
    obs_pred = torch.cat([feat_pred_same, feat_pred_opp, feat_pred_food, energy_pred], dim=-1)

    eye_prey = torch.eye(n_prey, dtype=torch.bool, device=device)
    eye_pd = torch.ones((n_prey, n_pred), dtype=torch.bool, device=device)
    feat_prey_same = topk_relative_features(prey_pos, prey_pos, ~eye_prey, k, ws, r_prey)
    feat_prey_opp = topk_relative_features(prey_pos, pred_pos, eye_pd, k, ws, r_prey)
    feat_prey_food = torch.zeros((n_prey, k * 3), device=device)
    energy_prey = torch.full((n_prey, 1), 0.5, device=device)
    obs_prey = torch.cat([feat_prey_same, feat_prey_opp, feat_prey_food, energy_prey], dim=-1)

    return obs_pred, obs_prey


def _run_trial(
    test_genomes: torch.Tensor,
    reference_genome: torch.Tensor,
    test_is_predator: bool,
    cfg: Config,
    generator: torch.Generator,
) -> torch.Tensor:
    """Runs `test_genomes` (M individuals) against a same-size population of
    `reference_genome` clones for `cfg.competence_trial_steps` steps.

    Returns, per test individual: catches made (if `test_is_predator`) or
    catches suffered (if not) — always a length-M tensor of raw counts, since
    that's the one quantity both `measure_predator_competence` and
    `measure_prey_competence` derive their rate from.
    """
    device = test_genomes.device
    m = test_genomes.shape[0]
    reference_genomes = reference_genome.unsqueeze(0).expand(m, -1)

    test_pos, test_head = _random_positions(m, cfg.world_size, device, generator)
    ref_pos, ref_head = _random_positions(m, cfg.world_size, device, generator)

    if test_is_predator:
        pred_genomes, prey_genomes = test_genomes, reference_genomes
        pred_pos, prey_pos = test_pos, ref_pos
        pred_head, prey_head = test_head, ref_head
    else:
        pred_genomes, prey_genomes = reference_genomes, test_genomes
        pred_pos, prey_pos = ref_pos, test_pos
        pred_head, prey_head = ref_head, test_head

    n_pred, n_prey = pred_genomes.shape[0], prey_genomes.shape[0]
    catches_by_predator = torch.zeros(n_pred, device=device)
    catches_by_prey = torch.zeros(n_prey, device=device)  # catches suffered

    pred_params = unpack(pred_genomes, cfg)
    prey_params = unpack(prey_genomes, cfg)

    for _ in range(cfg.competence_trial_steps):
        obs_pred, obs_prey = _trial_obs(pred_pos, prey_pos, cfg)
        actions_pred = batched_forward(pred_params, obs_pred)
        actions_prey = batched_forward(prey_params, obs_prey)

        pred_pos, pred_head = apply_movement(
            pred_pos, pred_head, actions_pred, cfg.max_speed_predator, cfg
        )
        prey_pos, prey_head = apply_movement(
            prey_pos, prey_head, actions_prey, cfg.max_speed_prey, cfg
        )

        dist_dp = torch.linalg.norm(torus_delta(pred_pos, prey_pos, cfg.world_size), dim=-1)
        dist_dp_masked = dist_dp.masked_fill(dist_dp >= cfg.eat_radius, float("inf"))
        min_dist_per_prey, claimant_pred = dist_dp_masked.min(dim=0)  # [n_prey]
        prey_caught = torch.isfinite(min_dist_per_prey)

        catch_contribution = torch.where(
            prey_caught, torch.ones_like(min_dist_per_prey), torch.zeros_like(min_dist_per_prey)
        )
        catches_by_predator = catches_by_predator.scatter_add(0, claimant_pred, catch_contribution)
        catches_by_prey = catches_by_prey + prey_caught.float()

        respawn_pos, respawn_head = _random_positions(n_prey, cfg.world_size, device, generator)
        prey_pos = torch.where(prey_caught.unsqueeze(-1), respawn_pos, prey_pos)
        prey_head = torch.where(prey_caught, respawn_head, prey_head)

    return catches_by_predator if test_is_predator else catches_by_prey


def measure_predator_competence(
    predator_genomes_sample: torch.Tensor,
    reference_prey_genome: torch.Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> torch.Tensor:
    """[M, genome_dim] evolved predator genomes vs. a frozen reference prey
    genome -> [M] catch rate (catches per step) per sampled predator."""
    catches = _run_trial(predator_genomes_sample, reference_prey_genome, True, cfg, generator)
    return catches / cfg.competence_trial_steps


def measure_prey_competence(
    prey_genomes_sample: torch.Tensor,
    reference_predator_genome: torch.Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> torch.Tensor:
    """[M, genome_dim] evolved prey genomes vs. a frozen reference predator
    genome -> [M] escape rate (1 - catches suffered per step) per sampled prey."""
    catches_suffered = _run_trial(
        prey_genomes_sample, reference_predator_genome, False, cfg, generator
    )
    return 1.0 - catches_suffered / cfg.competence_trial_steps
