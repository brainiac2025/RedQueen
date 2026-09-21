# RedQueen

Open-ended predator-prey co-evolution with evolved neural controllers, and a
quantified test for whether an evolutionary arms race actually emerges. See
[redqueen-architecture.md](redqueen-architecture.md) for the full design and
**[writeup.md](writeup.md) for the actual findings** — spoiler: no arms race,
one-sided predator escalation instead.

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

For `torch.compile` (optional, meaningfully reduces per-step kernel-launch
overhead — see Status below): `pip install -e ".[compile]"`. Upstream Triton
has no Windows wheels, so this installs the `triton-windows` community build.

## Tests

```bash
pytest
```

## Running the experiments

```bash
python scripts/run_experiment.py --seeds 0 1 2 3 4 --n-steps 100000 --out results/experiment.json
python scripts/run_sweeps.py --out-dir results/sweeps
python scripts/render_playback.py --n-steps 10000 --sample-every 40 --out media/playback_demo.gif
```

## Status

**Weeks 1-4 (§8) substantially complete.** World/genome/controller/sim with
`torch.func.vmap` batching per-agent weights; the evolution loop (`evolve.py`)
and fixed-reference competence measurement (`competence.py`, §6); and the
trend/stationarity analysis (`analysis.py`, §7a) — Hamed-Rao modified
Mann-Kendall (not vanilla MK, which assumes independent samples these series
don't have) applied to both halves of §5's operational definition. §7b's three
sweeps are done too. Remaining: the §11 write-up.

**The headline result** (`results/seeds0-4_n100000.json`; 5 seeds, 100,000
steps each): **no arms race, by §5's definition.** Predator
competence-vs-fixed-reference increases significantly in 5/5 seeds; prey
competence never increases in any seed (significantly decreasing in 2/5, flat
in 3/5). Realized catch rate in the live co-evolving population rises
significantly in 4/5 seeds and prey escape rate falls significantly in 5/5.
§5 requires *both* species' competence trending up for it to count as an arms
race — here only the predator side improves, so what's actually happening is
one-sided predator escalation, not Red Queen dynamics.

**§7b's sweeps** (`results/sweeps/`; 9 variants × 3 seeds × 50,000 steps)
confirm this is **robust, not a fluke of the default config**: across the
full tested range of mutation rate (σ 0.02-0.15), population size (1,100-4,400
total agents, density held ~constant), and sensing-range asymmetry (predator-
or prey-favored ±50%), realized predator catch rate rises and prey escape
rate falls significantly in essentially every seed of every variant, and no
variant restores both species' competence improving together. One-sided
predator escalation held up under every manipulation tried.

**A real bug found along the way**, worth flagging for anyone extending this:
an earlier version of the sweep runner looped all 9 variants in one Python
process, and running a *different*-shaped `Config` through `torch.compile`
earlier in that process silently corrupted results for a subsequently-repeated
shape — same seed, same config, wildly different trajectory, verified
directly (see the "Fix torch.compile cross-shape contamination" commit).
Likely specific to this Windows + community `triton-windows` build. Fixed by
running each sweep variant in its own subprocess (`scripts/run_sweeps.py`);
the §7a headline experiment was unaffected since it never changes shape
mid-process. Takeaway for future work here: don't trust `torch.compile`
results across a shape change without a fresh process in between.

Performance notes: reproduction/death bookkeeping in `sim.py` is written
without `nonzero`/`.item()`/boolean-mask indexing (each is a hidden GPU->CPU
sync — architecture §10) so the hot loop doesn't stall waiting on the CPU
every step. At the default `Config` scale (2000 prey / 200 predators),
`torch.compile` gives ~105 steps/sec vs. ~30 eager — short of the architecture
doc's "thousands of steps/sec" aspiration, since raw GPU compute is only
~2-3ms/step and the rest is Python/kernel-launch overhead (worse under
Windows' WDDM driver than Linux). Full `mode="reduce-overhead"` (CUDA graphs)
would close most of the remaining gap but needs restructuring `step()` around
persistent pre-allocated buffers rather than returning a fresh `WorldState`
each call — not yet done.

Deliverable 3 (rendered playback, `redqueen/playback.py` /
`scripts/render_playback.py`) is done — see the GIF above; it also surfaced
an interesting emergent behaviour not visible in the aggregate statistics:
prey visibly self-organize into dense flocks as the population reaches its
cap, with food locally depleted inside clusters.

**All four deliverables are done**: the repo with tests, the quantified
arms-race detector and its result, the playback visualisation, and
[writeup.md](writeup.md). Honest open items: `mode="reduce-overhead"` CUDA
graphs (performance, not correctness), and the unrooted `torch.compile`
cross-shape bug's true cause (worked around, not fixed at the source).
