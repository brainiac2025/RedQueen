# RedQueen

A predator-prey world where both species evolve their own tiny neural-network
brains from scratch. No reward function, no hand-tuned fitness score, just
energy and reproduction. The actual question I wanted answered: does a real
evolutionary arms race show up here, or does it just *look* like one? Full
answer, with the numbers, is in [writeup.md](writeup.md). Short version: no,
it doesn't. Predators keep getting better, prey don't keep up.

![Playback of a 10,000-step evolutionary run: prey (blue) cluster into
flocks, predators (red) hunt them, food (green) depletes locally in dense
clusters.](media/playback_demo.gif)

## Setup

```bash
pip install -e ".[dev]"
```

On Windows, plain `pip install torch` pulls a CPU-only wheel even on a CUDA
machine. Install the CUDA build first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
```

`torch.compile` roughly triples throughput on this sim but needs Triton,
which doesn't ship official Windows wheels. `pip install -e ".[compile]"`
pulls in the `triton-windows` community build if you want it.

## Tests

```bash
pytest
```

## Running the experiments

```bash
python scripts/run_experiment.py --seeds 0 1 2 3 4 --n-steps 100000 --out results/experiment.json
python scripts/run_sweeps.py --out-dir results/sweeps
python scripts/render_playback.py --n-steps 10000 --sample-every 40 --out media/playback_demo.gif

# or watch it happen live in a window instead of rendering to a file:
python scripts/render_playback.py --live --steps-per-frame 2
```

## What I found

Ran 5 seeds, 100,000 steps each. Predator competence against a frozen
benchmark opponent went up, significantly, in every single seed. Prey
competence against its own frozen benchmark never went up in a single one,
it was flat in most and actually got worse in two. Meanwhile, in the live
population, predators kept realizing a bigger catch advantage and prey kept
getting away less, with no sign of leveling off.

That's not an arms race. An arms race needs both sides genuinely improving
while neither one actually wins. Here only one side improves, and it's
winning. So what's actually going on is one-sided predator escalation, not
a Red Queen dynamic, and it held up across a whole sweep of different
mutation rates, population sizes, and sensing setups, not just the default
numbers. Full breakdown and the actual stats are in
[writeup.md](writeup.md).

## A few things worth knowing if you're poking around the code

`sim.py`'s reproduction and death bookkeeping deliberately avoids
`nonzero()`, `.item()`, and boolean-mask indexing, since each one forces a
GPU-to-CPU sync, and doing that every step is enough to tank a tight
simulation loop.

There's a real `torch.compile` bug in here I never fully root-caused:
compiling a differently-shaped config earlier in the same Python process
can silently corrupt results for a shape that was already compiled and is
being reused, even with a correctly-seeded generator. A fresh process gives
bit-identical results every time; a "dirty" process doesn't. `scripts/run_sweeps.py`
works around it by giving every config its own subprocess. Best guess is
it's specific to this Windows + community-Triton combo, not stock PyTorch.

Full `mode="reduce-overhead"` (CUDA graphs) would close most of the
remaining performance gap on top of what plain `torch.compile` already
gives, but it needs `step()` restructured around persistent pre-allocated
buffers instead of returning a fresh state object every call. Haven't
gotten to it.
