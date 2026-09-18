import torch

from redqueen.config import Config
from redqueen.genome import genome_dim, init_genomes, mutate, unpack


def test_genome_dim_matches_param_shapes():
    cfg = Config(hidden_dim=8, k_neighbours=3)
    dim = genome_dim(cfg)
    n = 4
    gen = torch.Generator().manual_seed(0)
    genomes = init_genomes(n, cfg, "cpu", gen)
    assert genomes.shape == (n, dim)


def test_unpack_shapes_and_roundtrip_size():
    cfg = Config(hidden_dim=8, k_neighbours=3)
    n = 5
    gen = torch.Generator().manual_seed(1)
    genomes = init_genomes(n, cfg, "cpu", gen)
    params = unpack(genomes, cfg)

    assert params["w1"].shape == (n, cfg.hidden_dim, cfg.obs_dim)
    assert params["b1"].shape == (n, cfg.hidden_dim)
    assert params["w2"].shape == (n, cfg.hidden_dim, cfg.hidden_dim)
    assert params["b2"].shape == (n, cfg.hidden_dim)
    assert params["w3"].shape == (n, cfg.action_dim, cfg.hidden_dim)
    assert params["b3"].shape == (n, cfg.action_dim)

    total = sum(v[0].numel() for v in params.values())
    assert total == genome_dim(cfg)


def test_mutate_changes_values_but_not_shape():
    cfg = Config(hidden_dim=8, k_neighbours=3)
    gen = torch.Generator().manual_seed(2)
    genomes = init_genomes(6, cfg, "cpu", gen)
    mutated = mutate(genomes, sigma=0.1, generator=gen)
    assert mutated.shape == genomes.shape
    assert not torch.allclose(mutated, genomes)


def test_mutate_zero_sigma_is_identity():
    cfg = Config(hidden_dim=8, k_neighbours=3)
    gen = torch.Generator().manual_seed(3)
    genomes = init_genomes(6, cfg, "cpu", gen)
    mutated = mutate(genomes, sigma=0.0, generator=gen)
    assert torch.allclose(mutated, genomes)
