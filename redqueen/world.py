"""World state: the continuous 2D torus, food patches, and per-species agent
slot arrays (architecture §1).

Population is stored in fixed-capacity slot arrays (`max_prey`, `max_predators`)
with an `alive` mask, rather than growing/shrinking tensors, so every tensor
shape stays static across the run (required for vmap/GPU batching — a
Python-side resize on every birth/death would defeat the point of §3).
A dead slot is simply marked `alive=False`; a birth reuses the first free
slot. This trades a small amount of wasted compute on dead slots for shapes
that never change, which is the right trade at N in the low thousands.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from redqueen.config import Config
from redqueen.genome import genome_dim, init_genomes

PREY = 0
PREDATOR = 1


@dataclass
class PopulationState:
    positions: torch.Tensor  # [max_n, 2]
    headings: torch.Tensor  # [max_n]
    energy: torch.Tensor  # [max_n]
    genomes: torch.Tensor  # [max_n, genome_dim]
    alive: torch.Tensor  # [max_n] bool
    genome_id: torch.Tensor  # [max_n] long, unique lineage id (for logging only)

    @property
    def max_n(self) -> int:
        return self.positions.shape[0]


@dataclass
class WorldState:
    cfg: Config
    prey: PopulationState
    predator: PopulationState
    food_positions: torch.Tensor  # [n_food, 2]
    food_amount: torch.Tensor  # [n_food], in [0, food_energy]
    step_count: int
    next_genome_id: torch.Tensor  # 0-dim long tensor; kept on-device so
    # reproduction bookkeeping never has to sync to Python each step (see sim._reproduce)


def _init_population(
    max_n: int,
    n_alive: int,
    cfg: Config,
    device: str | torch.device,
    generator: torch.Generator,
    id_start: int,
) -> PopulationState:
    positions = torch.rand((max_n, 2), device=device, generator=generator) * cfg.world_size
    headings = torch.rand((max_n,), device=device, generator=generator) * 2 * torch.pi
    energy = torch.full((max_n,), cfg.initial_energy, device=device)
    genomes = torch.zeros((max_n, genome_dim(cfg)), device=device)
    genomes[:n_alive] = init_genomes(n_alive, cfg, device, generator)
    alive = torch.zeros((max_n,), dtype=torch.bool, device=device)
    alive[:n_alive] = True
    genome_id = torch.full((max_n,), -1, dtype=torch.long, device=device)
    genome_id[:n_alive] = torch.arange(id_start, id_start + n_alive, device=device)
    return PopulationState(positions, headings, energy, genomes, alive, genome_id)


def init_world(
    cfg: Config, device: str | torch.device, generator: torch.Generator
) -> WorldState:
    n_prey0 = cfg.max_prey // 4
    n_pred0 = cfg.max_predators // 4

    prey = _init_population(cfg.max_prey, n_prey0, cfg, device, generator, id_start=0)
    predator = _init_population(
        cfg.max_predators, n_pred0, cfg, device, generator, id_start=n_prey0
    )

    food_positions = (
        torch.rand((cfg.food_max_patches, 2), device=device, generator=generator) * cfg.world_size
    )
    food_amount = torch.full((cfg.food_max_patches,), cfg.food_energy, device=device)

    return WorldState(
        cfg=cfg,
        prey=prey,
        predator=predator,
        food_positions=food_positions,
        food_amount=food_amount,
        step_count=0,
        next_genome_id=torch.tensor(n_prey0 + n_pred0, dtype=torch.long, device=device),
    )
