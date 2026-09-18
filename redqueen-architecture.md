# RedQueen — Open-Ended Predator-Prey Co-Evolution

A continuous 2D artificial-life world in which predator and prey populations are
controlled entirely by small evolved neural networks (genome = network weights),
reproducing and dying based on an energy/survival economy, with no explicit
fitness function beyond "acquire enough energy to reproduce before dying." The
central question: does the system spontaneously produce a measurable
evolutionary arms race (the "Red Queen" dynamic — predators and prey each
improving in alternating cycles, neither gaining lasting ground) rather than
just visually-plausible chasing behaviour? No dataset anywhere — the world,
agents, and evolutionary dynamics are entirely self-generated code. Target
hardware: Legion 7 Pro (16 GB GPU, 64 GB RAM).

**Deliverables**
1. Public repo with tests, reproducible evolution runs, pinned dependencies.
2. A quantified arms-race detector (§7) applied to the evolution run — the
   project's actual contribution, not the simulation itself.
3. A rendered playback/visualisation of a full evolutionary run.
4. 4–6 page write-up centred on whether an arms race is real and measurable.

---

## 0. What's already published, and what isn't

State this plainly. Open-ended multi-agent evolution and artificial life are
established fields — continuous cellular-automata life-forms (Lenia and its
descendants), large multi-agent ecosystem environments (Neural MMO-style
platforms), and evolved-neural-controller predator-prey simulations all exist,
and the Red Queen hypothesis itself is textbook evolutionary biology, studied
computationally in various forms for decades. **This project's base setup —
evolved neural controllers in a predator-prey world — is not new.** What's
underdone, and what this project targets: most artificial-life demos show a
chase-and-evade animation and call it an "arms race" without ever measuring
whether escalation is actually happening, whether it's cyclical, or whether it's
just noise in a system that's actually converged. §7 is the contribution —
turning "it looks like an arms race" into a number with a null-hypothesis test
behind it.

---

## 1. The world (`world.py`)

Continuous 2D torus (wraparound edges, no boundary artefacts), populated with:

- **Food** (static, regenerating resource patches — prey eat these for energy).
- **Prey** agents (eat food, evade predators).
- **Predator** agents (eat prey, no food access — pure predation for energy).

Each agent has position, heading, velocity, and an energy level that depletes
per step (metabolism) and increases on eating. Energy ≤ 0 → death. Energy above
a reproduction threshold → asexual reproduction with mutated genome (§2),
splitting energy between parent and offspring. This is the entire "fitness
function" — there is no reward signal beyond survival and reproduction
mechanics, which is what makes any resulting behaviour genuinely *evolved*
rather than trained toward a specified objective.

```python
state: positions [N,2], headings [N], energy [N], species [N] (0=prey, 1=pred),
       genome_id [N]   — all torch tensors on cuda, N up to several thousand
```

---

## 2. Agent controller and genome (`genome.py`, `controller.py`)

Each agent's behaviour is a tiny MLP (~2-3k params) mapping local sensory input
to a movement action, run identically for every agent of a species (weights
differ per individual — the genome *is* the weight vector):

```
sensory input (per agent, egocentric):
  - k nearest same-species neighbours: relative position, distance     (flocking cue)
  - k nearest opposite-species agents: relative position, distance     (predation/evasion cue)
  - k nearest food patches (prey only): relative position, distance
  - own energy level (normalised)
→ Linear 32 → tanh → Linear 32 → tanh → Linear → (turn_rate, speed)
```

Genome = flattened weight vector (~2-3k floats). Reproduction: child genome =
parent genome + Gaussian mutation noise (`σ_mutation`, swept in §7 experiments).
No crossover in the base version (asexual) — keeps the evolutionary dynamics
simpler to analyse; sexual reproduction with crossover is a stretch-goal
extension, not required.

**Species-level shared "meta-parameters"** (metabolism rate, sensing range,
max speed) are fixed constants per species, not evolved — only the *behaviour*
(the MLP weights) evolves. This is a deliberate scope decision: it isolates
behavioural co-evolution as the thing being studied, rather than also letting
morphology/metabolism drift, which would confound the arms-race measurement.

---

## 3. Batched simulation (`sim.py`)

The entire population — potentially thousands of agents across both species —
steps forward as GPU tensor operations, no Python loop over individual agents:

```python
def compute_sensory_inputs(state, k_neighbours) -> obs [N, obs_dim]
    # k-nearest-neighbour queries via pairwise distance matrix + topk,
    # batched across all agents simultaneously
def step_controllers(obs, genomes) -> actions [N, 2]
    # batched forward pass: every agent's genome is a different weight set,
    # so this is a batched MLP with per-sample weights — implement via
    # vmap (torch.func.vmap) over the population, NOT a Python loop
def step_physics(state, actions) -> state
    # move agents, resolve eating (predator-prey and prey-food proximity),
    # apply metabolism, check death, check reproduction
```

**`torch.func.vmap` over per-agent weights is the one non-obvious engineering
piece** — without it, "N agents each with their own tiny network" becomes N
separate forward passes and the GPU sits idle. Get this working and tested
early (week 1); it's the same batching discipline as every other project in
this portfolio, applied to heterogeneous per-agent parameters instead of a
shared model.

Population is capped (`MAX_PREY`, `MAX_PRED`) with random culling if exceeded,
to keep tensor shapes bounded — evolution still operates freely under the cap
since reproduction is stochastic and culling is uniform-random, not
fitness-directed (avoid accidentally injecting an extra selection pressure).

---

## 4. Running an evolutionary experiment (`evolve.py`)

```python
def run_evolution(cfg, gen, n_steps) -> EvolutionLog
    # initialise random population, run step() in a loop, log per-timestep:
    # population sizes, mean genome per species (for drift tracking),
    # mean energy, birth/death counts, and periodic genome snapshots
```

A single run is tens of thousands of simulation steps (not gradient-descent
steps — there's no gradient here, it's population dynamics). This is fast on
GPU since each step is a handful of tensor ops over a few thousand agents;
expect thousands of steps per second, so a full evolutionary run (enough steps
to see multiple population boom-bust cycles) completes in minutes, not hours.
This means the real experimental budget is **many independent runs** (different
seeds, different mutation rates) for statistical confidence in §7, not one long
run — batch multiple independent world instances on the GPU simultaneously the
same way the other portfolio projects batch games/scenarios.

---

## 5. What "arms race" would even mean, precisely

Before measuring anything, define it operationally, since "arms race" is
usually used loosely in artificial-life demos: an arms race is present if (a)
predator and prey **behavioural competence** (§6) both show a positive trend
over evolutionary time (both genuinely improving, not just one crashing to
extinction), and (b) neither population's *realized* success rate (catches per
predator-lifetime, escapes per prey-lifetime) shows a sustained trend — i.e.
both sides are getting better at their job, but relative outcomes stay roughly
stationary, which is the textbook signature of co-evolutionary escalation
rather than one-sided dominance.

---

## 6. Measuring behavioural competence (`competence.py`)

Competence can't be read off the raw genome (weights aren't interpretable), so
measure it **functionally**, via standardized trials against a fixed reference
opponent, repeated at intervals throughout the evolutionary run:

```python
def measure_predator_competence(predator_genomes_sample, reference_prey_genome, cfg) -> catch_rate
def measure_prey_competence(prey_genomes_sample, reference_predator_genome, cfg) -> escape_rate
```

The **reference opponent is frozen** (a fixed early-generation genome, saved
once at the start of the run) — this is what turns "look, it's chasing better"
into a real measurement: testing evolving predators against a fixed prey
benchmark isolates *predator* improvement from any co-evolutionary shift in the
prey they're actually facing day to day. Do this for both species, sampling
genomes at regular generation intervals throughout the run.

---

## 7. Experiments for the write-up

### 7a. Headline experiment — is there a real, statistically measurable arms race?

Run the full evolution (§4) for many independent seeds. At regular intervals,
sample current genomes and run the fixed-reference competence trials (§6) for
both species, producing two time series: predator-competence-vs-fixed-prey and
prey-competence-vs-fixed-predator, across evolutionary time. Test the §5
definition directly:

1. **Trend test** on each competence series (e.g. Mann-Kendall) — is there a
   statistically significant positive trend, not just visual upward drift?
2. **Stationarity test** on the *realized* catch/escape rate between
   co-evolving (not fixed-reference) predator and prey populations over the
   same period — should show no significant trend if it's a real arms race,
   versus a significant trend if one side is actually winning.
3. Report both series with confidence bands across seeds, and the test
   statistics plainly, including a **null result if that's what happens** —
   many artificial-life systems actually show one-sided extinction or early
   stagnation rather than a sustained arms race, and that's a legitimate,
   reportable finding given how the question was framed honestly in §5.

### 7b. Supporting experiments

1. **Mutation rate sweep**: `σ_mutation ∈ {low, med, high}` — does escalation
   require a specific mutation-rate regime, or is it robust across a range?
2. **Population size sweep**: does a larger population (more genetic diversity
   per generation) sustain an arms race longer than a small, drift-dominated one?
3. **Sensory range asymmetry**: give predators or prey an advantage in sensing
   range and see whether the arms-race signature (if present at baseline)
   survives an initial asymmetry, or whether one side permanently dominates.

---

## 8. Three-to-four-week plan

**Week 1 — World, controller, vmap batching.** `world.py`, `genome.py`,
`controller.py`, `sim.py` with `torch.func.vmap` working and tested (this is
the engineering risk of the project — get it solid first). Verify a small
population survives at least a few hundred steps without immediately dying out
or exploding — this is a real risk with an unshaped reward-free system.

**Week 2 — Full evolution loop, competence measurement.** `evolve.py`,
`competence.py`. Run several full evolutionary sessions, produce playback
visualisations, sanity-check that populations show boom-bust cycles rather than
flatlining (a flatlined population means the fitness landscape is too easy or
too hard — tune metabolism/reproduction thresholds).

**Week 3 — The contribution.** §7a across many seeds: the trend and
stationarity tests, confidence bands, the honest report of whatever the data
shows. This is the week not to compress.

**Week 4 — Supporting sweeps, write-up.** §7b's three sweeps, write the report
centred on §7a, populate README with playback GIFs, tag `v1.0`.

Git rule: push on day one; push every session; tag each week's milestone.

---

## 9. Config (single source of truth)

```python
@dataclass
class Config:
    # world
    world_size: float = 100.0
    max_prey: int = 2000
    max_predators: int = 500
    food_regen_rate: float = 0.5
    # agent
    k_neighbours: int = 5
    hidden_dim: int = 32
    metabolism_prey: float = 0.01
    metabolism_predator: float = 0.02
    reproduction_threshold: float = 2.0
    sigma_mutation: float = 0.05
    # sim
    n_steps: int = 100_000
    n_parallel_worlds: int = 32
    competence_sample_interval: int = 2000
    competence_n_trials: int = 200
    device: str = "cuda"
    seed: int = 0
```

---

## 10. Risks and their fixes

| Risk | Symptom | Fix |
|---|---|---|
| Population crashes immediately | all agents die within a few hundred steps | loosen metabolism/reproduction threshold; verify food regen keeps pace with prey population |
| One species goes extinct every run | no sustained co-evolution to measure | rebalance sensing range/speed asymmetry; check predator metabolism isn't too forgiving |
| `vmap` batching doesn't actually speed things up | GPU utilisation low despite vmap | profile; confirm no `.item()`/host syncs inside the stepped loop; check genome tensor layout is contiguous |
| Competence trend test is underpowered | trend never reaches significance either way | increase `competence_n_trials` and sampling frequency; report confidence intervals, not just a binary significant/not |
| "Arms race" claim isn't falsifiable as designed | every result gets called an arms race post hoc | §5's definition must be written and locked *before* running 7a — don't loosen the criteria after seeing the data |

---

## 11. Write-up outline (4–6 pages)

1. The Red Queen hypothesis and why artificial-life demos usually assert it
   rather than measure it (§0) — state the base setup is not novel, the
   measurement is.
2. Method: world, evolved controllers, the vmap-batched simulation, the
   operational definition of an arms race (§5) and how it's measured (§6).
3. **The contribution**: §7a's trend/stationarity results across seeds — report
   honestly whether an arms race is present, absent, or seed-dependent.
4. Supporting results: §7b's sweeps.
5. Limitations: fixed morphology/metabolism (only behaviour evolves), 2D toy
   physics, small population sizes relative to real ecological systems.
6. Reproducibility: one command, pinned seeds, hardware and wall-clock reported.
