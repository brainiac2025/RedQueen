"""Genome representation: flattened MLP weight vectors (architecture §2).

A genome is a 1D float tensor of length `genome_dim`. It packs every weight
and bias of the per-agent controller MLP in a fixed order so it can be
reshaped into named parameter tensors for `controller.py`, and so mutation is
just elementwise Gaussian noise on the flat vector.
"""

from __future__ import annotations

import math

import torch

from redqueen.config import Config

# Ordered (name, shape) layout of the controller MLP's parameters.
# shape is (out_features, in_features) for weights, (out_features,) for biases.


def param_shapes(cfg: Config) -> list[tuple[str, tuple[int, ...]]]:
    h = cfg.hidden_dim
    o = cfg.obs_dim
    a = cfg.action_dim
    return [
        ("w1", (h, o)),
        ("b1", (h,)),
        ("w2", (h, h)),
        ("b2", (h,)),
        ("w3", (a, h)),
        ("b3", (a,)),
    ]


def genome_dim(cfg: Config) -> int:
    return sum(math.prod(shape) for _, shape in param_shapes(cfg))


def init_genomes(
    n: int, cfg: Config, device: str | torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Random initial genomes, Xavier-ish scaled so early-generation agents don't saturate tanh."""
    dim = genome_dim(cfg)
    genomes = torch.empty((n, dim), device=device)
    offset = 0
    for _, shape in param_shapes(cfg):
        size = math.prod(shape)
        fan_in = shape[1] if len(shape) == 2 else 1
        scale = fan_in**-0.5
        genomes[:, offset : offset + size] = (
            torch.randn((n, size), device=device, generator=generator) * scale
        )
        offset += size
    return genomes


def unpack(genomes: torch.Tensor, cfg: Config) -> dict[str, torch.Tensor]:
    """[N, genome_dim] -> dict of named per-agent parameter tensors [N, *shape]."""
    n = genomes.shape[0]
    params: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape in param_shapes(cfg):
        size = math.prod(shape)
        params[name] = genomes[:, offset : offset + size].reshape((n, *shape))
        offset += size
    return params


def mutate(
    genomes: torch.Tensor, sigma: float, generator: torch.Generator
) -> torch.Tensor:
    """Additive Gaussian mutation, elementwise on the flat genome vector (architecture §2)."""
    noise = torch.randn(genomes.shape, device=genomes.device, generator=generator) * sigma
    return genomes + noise
