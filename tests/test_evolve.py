import torch

from redqueen.config import Config
from redqueen.evolve import run_evolution


def _small_cfg(**overrides) -> Config:
    # Population caps/reproduction params mirror test_sim.py's
    # test_dead_slots_can_be_reused_by_births, which was verified stable
    # (no extinction) over 100 steps at this scale.
    cfg = Config(
        world_size=50.0,
        max_prey=200,
        max_predators=60,
        food_max_patches=100,
        k_neighbours=2,
        hidden_dim=4,
        eat_radius=1.5,
        reproduction_threshold=1.05,
        initial_energy=1.2,
        reference_burn_in_steps=20,
        competence_sample_interval=30,
        competence_n_trials=4,
        competence_trial_steps=20,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_run_evolution_logs_population_series():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(0)

    state, log = run_evolution(cfg, gen, n_steps=65, log_interval=10)

    assert log.steps == [0, 10, 20, 30, 40, 50, 60]
    assert len(log.n_prey) == len(log.steps)
    assert len(log.mean_genome_prey) == len(log.steps)

    from redqueen.genome import genome_dim

    assert log.mean_genome_prey[0].shape == (genome_dim(cfg),)
    assert log.mean_genome_predator[0].shape == (genome_dim(cfg),)
    assert all(n >= 0 for n in log.n_prey)
    assert torch.isfinite(state.prey.positions).all()


def test_run_evolution_freezes_reference_after_burn_in():
    cfg = _small_cfg(reference_burn_in_steps=15)
    gen = torch.Generator().manual_seed(1)

    _, log = run_evolution(cfg, gen, n_steps=65, log_interval=10)

    assert log.reference_prey_genome is not None
    assert log.reference_predator_genome is not None
    from redqueen.genome import genome_dim

    assert log.reference_prey_genome.shape == (genome_dim(cfg),)


def test_run_evolution_produces_competence_checkpoints():
    cfg = _small_cfg(reference_burn_in_steps=10, competence_sample_interval=20)
    gen = torch.Generator().manual_seed(2)

    _, log = run_evolution(cfg, gen, n_steps=65, log_interval=10)

    assert len(log.competence_steps) > 0
    assert len(log.predator_catch_rate) == len(log.competence_steps)
    assert len(log.prey_escape_rate) == len(log.competence_steps)
    for cr in log.predator_catch_rate:
        assert cr.shape == (cfg.competence_n_trials,)
