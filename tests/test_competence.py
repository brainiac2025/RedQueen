import torch

from redqueen.competence import measure_predator_competence, measure_prey_competence
from redqueen.config import Config
from redqueen.genome import init_genomes


def _small_cfg(**overrides) -> Config:
    cfg = Config(
        world_size=30.0,
        k_neighbours=2,
        hidden_dim=4,
        eat_radius=2.0,
        competence_trial_steps=150,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_predator_competence_shape_and_range():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(0)
    predator_genomes = init_genomes(6, cfg, "cpu", gen)
    reference_prey = init_genomes(1, cfg, "cpu", gen)[0]

    catch_rate = measure_predator_competence(predator_genomes, reference_prey, cfg, gen)
    assert catch_rate.shape == (6,)
    assert torch.all(catch_rate >= 0)
    assert torch.isfinite(catch_rate).all()


def test_prey_competence_shape_and_range():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(1)
    prey_genomes = init_genomes(6, cfg, "cpu", gen)
    reference_predator = init_genomes(1, cfg, "cpu", gen)[0]

    escape_rate = measure_prey_competence(prey_genomes, reference_predator, cfg, gen)
    assert escape_rate.shape == (6,)
    assert torch.all(escape_rate <= 1.0)
    assert torch.isfinite(escape_rate).all()


def test_predator_speed_advantage_increases_catch_rate():
    """Sanity check on the arena mechanics themselves (movement + catch
    detection), independent of genome quality: with the *same* genomes,
    giving predators a large speed advantage over prey should yield a
    higher catch rate than the reverse, since a faster predator can close
    distance a slower one can't."""
    gen = torch.Generator().manual_seed(2)
    genomes = init_genomes(10, _small_cfg(), "cpu", gen)
    reference = init_genomes(1, _small_cfg(), "cpu", gen)[0]

    fast_predator_cfg = _small_cfg(max_speed_predator=3.0, max_speed_prey=0.3)
    slow_predator_cfg = _small_cfg(max_speed_predator=0.3, max_speed_prey=3.0)

    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    fast_catch_rate = measure_predator_competence(genomes, reference, fast_predator_cfg, gen_a).mean()
    slow_catch_rate = measure_predator_competence(genomes, reference, slow_predator_cfg, gen_b).mean()

    assert fast_catch_rate > slow_catch_rate
