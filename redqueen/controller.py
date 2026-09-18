"""Per-agent MLP controller, batched over heterogeneous per-agent weights via vmap.

Architecture §3: "N agents each with their own tiny network" must not become
N separate forward passes. `mlp_forward` is written for a *single* agent
(unbatched params, unbatched observation); `batched_forward` lifts it across
the whole population with `torch.func.vmap`, mapping over the leading
dimension of both the parameter pytree and the observation batch.
"""

from __future__ import annotations

import torch
from torch.func import vmap


def mlp_forward(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Single-agent forward pass. params values are unbatched: w1 [h,o], b1 [h], etc."""
    h = torch.tanh(params["w1"] @ x + params["b1"])
    h = torch.tanh(params["w2"] @ h + params["b2"])
    out = params["w3"] @ h + params["b3"]
    turn_rate = torch.tanh(out[0])  # in [-1, 1], scaled by species turn-rate constant by caller
    speed = torch.sigmoid(out[1])  # in [0, 1], scaled by species max-speed constant by caller
    return torch.stack([turn_rate, speed])


def batched_forward(params: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    """params: dict of [N, *shape] tensors, obs: [N, obs_dim] -> actions [N, 2].

    Every agent has its own weights, so this vmaps over per-sample parameters
    rather than sharing one weight set across the batch (the usual case).
    """
    return vmap(mlp_forward, in_dims=(0, 0))(params, obs)
