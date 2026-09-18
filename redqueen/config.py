"""Single source of truth for simulation parameters (architecture §9)."""

from dataclasses import dataclass


@dataclass
class Config:
    # world
    world_size: float = 100.0
    max_prey: int = 2000
    max_predators: int = 200
    food_max_patches: int = 400
    food_regen_rate: float = 0.5
    food_energy: float = 1.0

    # agent
    k_neighbours: int = 5
    hidden_dim: int = 32
    metabolism_prey: float = 0.01
    metabolism_predator: float = 0.02
    reproduction_threshold: float = 2.0
    reproduction_cost_fraction: float = 0.5  # fraction of parent energy given to offspring
    max_energy: float = 5.0  # satiation cap: an agent stuck at the population cap can't store energy forever
    initial_energy: float = 1.0
    max_speed_prey: float = 1.0
    max_speed_predator: float = 1.2
    max_turn_rate: float = 0.7853981633974483  # pi/4 radians per step
    sensing_range: float = 20.0
    eat_radius: float = 1.0
    predator_catch_energy: float = 0.4
    sigma_mutation: float = 0.05

    # sim
    n_steps: int = 100_000
    n_parallel_worlds: int = 32
    competence_sample_interval: int = 2000
    competence_n_trials: int = 200
    device: str = "cuda"
    seed: int = 0

    @property
    def obs_dim(self) -> int:
        # k same-species neighbours (dx, dy, dist) + k opposite-species (dx, dy, dist)
        # + k food patches (dx, dy, dist) [prey only, zero-padded for predators] + own energy
        return self.k_neighbours * 3 * 3 + 1

    @property
    def action_dim(self) -> int:
        return 2  # (turn_rate, speed)
