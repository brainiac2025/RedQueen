# RedQueen

Open-ended predator-prey co-evolution with evolved neural controllers, and a
quantified test for whether an evolutionary arms race actually emerges. See
[redqueen-architecture.md](redqueen-architecture.md) for the full design.

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

## Running the headline experiment

```bash
python scripts/run_experiment.py --seeds 0 1 2 3 4 --n-steps 100000 --out results/experiment.json
```

## Status

**Weeks 1-3 (§8) complete.** World/genome/controller/sim with `torch.func.vmap`
batching per-agent weights; the evolution loop (`evolve.py`) and fixed-reference
competence measurement (`competence.py`, §6); and the trend/stationarity
analysis (`analysis.py`, §7a) — Hamed-Rao modified Mann-Kendall (not vanilla
MK, which assumes independent samples these series don't have) applied to
both halves of §5's operational definition.

**The headline result** (`results/seeds0-4_n100000.json`; 5 seeds, 100,000
steps each, ~18 min total on an RTX 4090 Laptop GPU with `torch.compile`):
**no arms race, by §5's definition.** Predator competence-vs-fixed-reference
increases significantly in 5/5 seeds; prey competence never increases in any
seed (significantly decreasing in 2/5, flat in 3/5). Realized catch rate in
the live co-evolving population rises significantly in 4/5 seeds and prey
escape rate falls significantly in 5/5. §5 requires *both* species'
competence trending up for it to count as an arms race — here only the
predator side improves, so what's actually happening is one-sided predator
escalation, not Red Queen dynamics. This is consistent across all 5 seeds and
is exactly the kind of legitimate non-arms-race outcome architecture §0/§7a
call out as a real possibility, not a failure of the experiment.

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

Next: §7b's supporting sweeps (mutation rate, population size, sensing
asymmetry) — now with a concrete question to test them against: does any of
these restore genuine two-sided competence improvement, or is one-sided
predator escalation robust to them? Then the write-up (§11).
