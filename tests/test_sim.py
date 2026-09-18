import torch

from redqueen.config import Config
from redqueen.sim import compute_sensory_inputs, step, step_controllers
from redqueen.world import init_world


def _small_cfg(**overrides) -> Config:
    cfg = Config(
        world_size=50.0,
        max_prey=200,
        max_predators=60,
        food_max_patches=100,
        k_neighbours=3,
        hidden_dim=8,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_compute_sensory_inputs_shapes():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(0)
    state = init_world(cfg, "cpu", gen)

    obs_prey, obs_pred = compute_sensory_inputs(state, cfg)
    assert obs_prey.shape == (cfg.max_prey, cfg.obs_dim)
    assert obs_pred.shape == (cfg.max_predators, cfg.obs_dim)
    assert torch.isfinite(obs_prey).all()
    assert torch.isfinite(obs_pred).all()


def test_step_controllers_produces_valid_actions():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(0)
    state = init_world(cfg, "cpu", gen)

    obs_prey, _ = compute_sensory_inputs(state, cfg)
    actions = step_controllers(obs_prey, state.prey.genomes, cfg)
    assert actions.shape == (cfg.max_prey, cfg.action_dim)
    assert torch.isfinite(actions).all()


def test_population_survives_a_few_hundred_steps_without_nan_or_explosion():
    """Week-1 acceptance criterion (architecture §8): a small population should
    survive at least a few hundred steps without immediately dying out or
    producing NaNs/exploding tensor values."""
    cfg = _small_cfg(sigma_mutation=0.05)
    gen = torch.Generator().manual_seed(42)
    state = init_world(cfg, "cpu", gen)

    n_steps = 300
    alive_at_end = False
    for _ in range(n_steps):
        state, stats = step(state, cfg, gen)
        assert torch.isfinite(state.prey.positions).all()
        assert torch.isfinite(state.prey.energy).all()
        assert torch.isfinite(state.predator.positions).all()
        assert torch.isfinite(state.predator.energy).all()
        assert stats.n_prey <= cfg.max_prey
        assert stats.n_predators <= cfg.max_predators
        if stats.n_prey > 0 or stats.n_predators > 0:
            alive_at_end = True

    assert alive_at_end, "population went fully extinct within 300 steps"


def test_positions_stay_within_torus_bounds():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(7)
    state = init_world(cfg, "cpu", gen)

    for _ in range(50):
        state, _ = step(state, cfg, gen)

    assert torch.all(state.prey.positions >= 0) and torch.all(state.prey.positions < cfg.world_size)
    assert torch.all(state.predator.positions >= 0) and torch.all(
        state.predator.positions < cfg.world_size
    )


def test_dead_slots_can_be_reused_by_births():
    cfg = _small_cfg(reproduction_threshold=1.05, initial_energy=1.2, reproduction_cost_fraction=0.5)
    gen = torch.Generator().manual_seed(3)
    state = init_world(cfg, "cpu", gen)

    total_births = 0
    for _ in range(100):
        state, stats = step(state, cfg, gen)
        total_births += stats.prey_births + stats.predator_births

    assert total_births > 0
