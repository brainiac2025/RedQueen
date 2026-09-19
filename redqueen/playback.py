"""Rendered playback of an evolutionary run (architecture deliverable 3, §8 week 4).

Records periodic snapshots of agent positions/food state while a run
progresses, then renders them as an animated GIF: prey (blue), predators
(red), food patches (green, fading with remaining amount).

Snapshotting uses boolean-mask indexing (`positions[alive]`) to drop dead
slots before rendering — a host sync/dynamic-shape op, same category as
`sim._reproduce`'s docstring warns about, but harmless here since it only
runs once per `sample_every` steps (e.g. every 20-50), not in the hot loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from redqueen.config import Config
from redqueen.sim import step
from redqueen.world import WorldState, init_world


@dataclass
class Frame:
    step: int
    prey_positions: np.ndarray  # [n_alive_prey, 2]
    predator_positions: np.ndarray  # [n_alive_predators, 2]
    food_positions: np.ndarray  # [n_food, 2]
    food_amount: np.ndarray  # [n_food]
    n_prey: int
    n_predators: int


def _snapshot(state: WorldState, step_idx: int) -> Frame:
    prey_alive = state.prey.alive
    pred_alive = state.predator.alive
    return Frame(
        step=step_idx,
        prey_positions=state.prey.positions[prey_alive].cpu().numpy(),
        predator_positions=state.predator.positions[pred_alive].cpu().numpy(),
        food_positions=state.food_positions.cpu().numpy(),
        food_amount=state.food_amount.cpu().numpy(),
        n_prey=int(prey_alive.sum().item()),
        n_predators=int(pred_alive.sum().item()),
    )


def record_playback(
    cfg: Config,
    generator: torch.Generator,
    n_steps: int,
    sample_every: int = 20,
    compiled: bool = False,
) -> list[Frame]:
    """Runs the simulation for `n_steps`, snapshotting agent/food state every
    `sample_every` steps (including step 0), for later rendering."""
    step_fn = torch.compile(step) if compiled else step

    state = init_world(cfg, cfg.device, generator)
    frames = [_snapshot(state, 0)]
    for i in range(1, n_steps + 1):
        state, _ = step_fn(state, cfg, generator)
        if i % sample_every == 0:
            frames.append(_snapshot(state, i))
    return frames


def render_gif(
    frames: list[Frame],
    cfg: Config,
    out_path: str,
    fps: int = 20,
    dpi: int = 100,
    figsize: float = 5.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")

    n_food = frames[0].food_positions.shape[0]
    food_scatter = ax.scatter(
        frames[0].food_positions[:, 0],
        frames[0].food_positions[:, 1],
        s=18,
        marker="s",
        facecolor=np.zeros((n_food, 4)),
        edgecolors="none",
    )
    prey_scatter = ax.scatter([], [], s=8, c="#4da6ff", marker="o", label="prey")
    pred_scatter = ax.scatter([], [], s=34, c="#ff4d4d", marker="^", label="predator")
    title = ax.set_title("", color="white", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="white", fontsize=8)

    def update(i: int):
        f = frames[i]
        colors = np.zeros((f.food_amount.shape[0], 4))
        colors[:, 1] = 0.75  # green channel
        colors[:, 3] = np.clip(f.food_amount / cfg.food_energy, 0.0, 1.0)  # alpha
        food_scatter.set_facecolor(colors)

        prey_scatter.set_offsets(f.prey_positions if f.prey_positions.size else np.empty((0, 2)))
        pred_scatter.set_offsets(
            f.predator_positions if f.predator_positions.size else np.empty((0, 2))
        )
        title.set_text(f"step {f.step:,}  |  prey={f.n_prey}  predators={f.n_predators}")
        return food_scatter, prey_scatter, pred_scatter, title

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
