"""Deliverable 3: a rendered playback of an evolutionary run.

    python scripts/render_playback.py --n-steps 8000 --sample-every 20 \
        --seed 0 --out media/playback_demo.gif
"""

from __future__ import annotations

import argparse
import time

import torch

from redqueen.config import Config
from redqueen.playback import record_playback, render_gif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default=Config().device)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--out", type=str, default="media/playback_demo.gif")
    args = parser.parse_args()

    cfg = Config(device=args.device)
    generator = torch.Generator(device=cfg.device).manual_seed(args.seed)

    t0 = time.time()
    frames = record_playback(
        cfg, generator, n_steps=args.n_steps, sample_every=args.sample_every,
        compiled=not args.no_compile,
    )
    print(f"recorded {len(frames)} frames in {time.time() - t0:.1f}s")

    t0 = time.time()
    render_gif(frames, cfg, args.out, fps=args.fps)
    print(f"rendered to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
