"""Deliverable 3: a rendered playback of an evolutionary run.

    python scripts/render_playback.py --n-steps 8000 --sample-every 20 \
        --seed 0 --out media/playback_demo.gif

Or watch it happen live in a window instead of saving to a file:

    python scripts/render_playback.py --live --steps-per-frame 2
"""

from __future__ import annotations

import argparse
import time

import torch

from redqueen.config import Config
from redqueen.playback import live_view, record_playback, render_gif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default=Config().device)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--out", type=str, default="media/playback_demo.gif")
    parser.add_argument(
        "--live", action="store_true", help="open an interactive window instead of saving a GIF"
    )
    parser.add_argument(
        "--steps-per-frame", type=int, default=1, help="--live only: sim steps per rendered frame"
    )
    parser.add_argument(
        "--interval-ms", type=int, default=33, help="--live only: target ms between frames"
    )
    args = parser.parse_args()

    cfg = Config(device=args.device)
    generator = torch.Generator(device=cfg.device).manual_seed(args.seed)

    if args.live:
        print("opening live view - close the window to stop early")
        live_view(
            cfg, generator, n_steps=args.n_steps,
            steps_per_frame=args.steps_per_frame, interval_ms=args.interval_ms,
            compiled=not args.no_compile,
        )
        return

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
