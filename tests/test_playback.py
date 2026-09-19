import os

import torch

from redqueen.config import Config
from redqueen.playback import record_playback, render_gif


def _small_cfg(**overrides) -> Config:
    cfg = Config(
        world_size=40.0,
        max_prey=100,
        max_predators=30,
        food_max_patches=60,
        k_neighbours=2,
        hidden_dim=4,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_record_playback_frame_count_and_shapes():
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(0)

    frames = record_playback(cfg, gen, n_steps=100, sample_every=25)

    assert len(frames) == 5  # step 0, 25, 50, 75, 100
    assert [f.step for f in frames] == [0, 25, 50, 75, 100]
    for f in frames:
        assert f.food_positions.shape == (cfg.food_max_patches, 2)
        assert f.prey_positions.shape[1] == 2
        assert f.predator_positions.shape[1] == 2
        assert f.prey_positions.shape[0] == f.n_prey
        assert f.predator_positions.shape[0] == f.n_predators


def test_render_gif_produces_a_file(tmp_path):
    cfg = _small_cfg()
    gen = torch.Generator().manual_seed(0)
    frames = record_playback(cfg, gen, n_steps=50, sample_every=25)

    out_path = str(tmp_path / "test.gif")
    render_gif(frames, cfg, out_path, fps=10, dpi=50, figsize=2.0)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
