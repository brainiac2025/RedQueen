import torch

from redqueen.config import Config
from redqueen.sim import compute_sensory_inputs, step, step_controllers, topk_relative_features
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


def test_topk_relative_features_max_range_zero_pads_out_of_range_entities():
    # Two agents 10 units apart on a large torus (no wraparound ambiguity).
    self_pos = torch.tensor([[0.0, 0.0]])
    other_pos = torch.tensor([[10.0, 0.0]])
    valid = torch.ones((1, 1), dtype=torch.bool)

    out_of_range = topk_relative_features(self_pos, other_pos, valid, k=1, world_size=1000.0, max_range=5.0)
    assert torch.allclose(out_of_range, torch.zeros_like(out_of_range))

    in_range = topk_relative_features(self_pos, other_pos, valid, k=1, world_size=1000.0, max_range=15.0)
    assert in_range[0, 2].item() == 10.0  # the distance component
    assert not torch.allclose(in_range, torch.zeros_like(in_range))


def test_sensing_range_is_species_specific_in_compute_sensory_inputs():
    """A prey with a short sensing range shouldn't see a same-species
    neighbour that a prey with a long range would see, at the same distance."""
    cfg_short = Config(
        world_size=1000.0, max_prey=2, max_predators=1, food_max_patches=1,
        k_neighbours=1, hidden_dim=4, sensing_range_prey=5.0, device="cpu",
    )
    cfg_long = Config(
        world_size=1000.0, max_prey=2, max_predators=1, food_max_patches=1,
        k_neighbours=1, hidden_dim=4, sensing_range_prey=50.0, device="cpu",
    )
    gen = torch.Generator().manual_seed(0)

    for cfg, expect_zero in ((cfg_short, True), (cfg_long, False)):
        state = init_world(cfg, "cpu", gen)
        state.prey.positions[0] = torch.tensor([0.0, 0.0])
        state.prey.positions[1] = torch.tensor([10.0, 0.0])
        state.prey.alive[:] = True
        state.predator.alive[:] = False  # isolate the same-species channel

        obs_prey, _ = compute_sensory_inputs(state, cfg)
        same_species_feat = obs_prey[0, :3]  # (dx, dy, dist) to nearest same-species neighbour
        if expect_zero:
            assert torch.allclose(same_species_feat, torch.zeros(3))
        else:
            assert same_species_feat[2].item() == 10.0


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
