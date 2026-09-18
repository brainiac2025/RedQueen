import torch

from redqueen.config import Config
from redqueen.controller import batched_forward, mlp_forward
from redqueen.genome import init_genomes, unpack


def test_batched_forward_matches_per_agent_loop():
    """The whole point of vmap here: batched_forward(params, obs) must equal
    calling mlp_forward once per agent with that agent's own (unbatched)
    params — i.e. vmap isn't silently sharing weights across the batch."""
    cfg = Config(hidden_dim=8, k_neighbours=3)
    n = 7
    gen = torch.Generator().manual_seed(0)
    genomes = init_genomes(n, cfg, "cpu", gen)
    params = unpack(genomes, cfg)
    obs = torch.randn(n, cfg.obs_dim, generator=gen)

    batched_out = batched_forward(params, obs)

    manual_out = torch.stack(
        [
            mlp_forward({k: v[i] for k, v in params.items()}, obs[i])
            for i in range(n)
        ]
    )

    assert torch.allclose(batched_out, manual_out, atol=1e-6)


def test_output_shape_and_ranges():
    cfg = Config(hidden_dim=8, k_neighbours=3)
    n = 10
    gen = torch.Generator().manual_seed(1)
    genomes = init_genomes(n, cfg, "cpu", gen)
    params = unpack(genomes, cfg)
    obs = torch.randn(n, cfg.obs_dim, generator=gen)

    out = batched_forward(params, obs)
    assert out.shape == (n, cfg.action_dim)
    turn_rate, speed = out[:, 0], out[:, 1]
    assert torch.all(turn_rate >= -1) and torch.all(turn_rate <= 1)
    assert torch.all(speed >= 0) and torch.all(speed <= 1)


def test_different_genomes_give_different_actions_for_same_obs():
    cfg = Config(hidden_dim=8, k_neighbours=3)
    n = 5
    gen = torch.Generator().manual_seed(2)
    genomes = init_genomes(n, cfg, "cpu", gen)
    params = unpack(genomes, cfg)
    obs = torch.ones(n, cfg.obs_dim)  # identical observation for every agent

    out = batched_forward(params, obs)
    # with distinct random genomes, actions should not all collapse to the same value
    assert not torch.allclose(out[0].expand_as(out), out)
