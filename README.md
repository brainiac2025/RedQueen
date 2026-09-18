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

## Status

Week 1 (§8) complete: `world.py`/`genome.py`/`controller.py`/`sim.py`, with
`torch.func.vmap` batching the MLP forward pass over per-agent weights, and
reproduction/death bookkeeping written without `nonzero`/`.item()`/boolean-mask
indexing (each is a hidden GPU->CPU sync — see architecture §10) so a tight
simulation loop doesn't stall every step waiting on the CPU.

Verified on an RTX 4090 Laptop GPU at the default `Config` scale (2000 prey /
200 predators): population is stable over 5000+ steps (no NaNs, no explosion,
no extinction) with `torch.compile` giving ~105 steps/sec, vs. ~30 steps/sec
eager. That's short of the architecture doc's "thousands of steps/sec"
expectation; raw GPU compute is only ~2-3ms/step, so the gap is Python/kernel-
launch overhead (worse under Windows' WDDM driver than Linux). Full
`mode="reduce-overhead"` (CUDA graphs) would close most of the rest but needs
restructuring `step()` around persistent pre-allocated buffers rather than
returning a fresh `WorldState` each call — deferred to whenever the week 3/4
many-seeds runs make it worth the complexity.

Next: `evolve.py` (the evolution loop) and `competence.py` (§6), per §8 week 2.
