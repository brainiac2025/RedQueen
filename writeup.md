# Is it actually an arms race? Measuring predator-prey co-evolution instead of asserting it

## The problem with most artificial-life demos

Evolve some predators and some prey, give them little neural network brains, let them chase and dodge each other for a few thousand generations, and you get a video that looks like an arms race. Predators get faster, prey get more evasive, the chase gets tighter, everyone nods and calls it a Red Queen dynamic. People have been making this kind of demo for decades, in one form or another (continuous-CA life forms, big multi-agent ecosystems, evolved predator-prey controllers), so none of that is new. The simulation in this repo isn't either: a 2D world, two species, tiny evolved MLP brains, energy-based reproduction. Fairly standard stuff.

What's missing from almost all of these demos is the part that would actually justify the word "arms race": a number, with a null hypothesis behind it. "It looks like they're escalating" is not a finding. It might be a real co-evolutionary dynamic where both sides are genuinely getting better and nobody gains lasting ground. It might just as easily be one side quietly winning while the loser hasn't gone extinct yet, or the system settling into a boring equilibrium and any wiggle you see in the chase behavior is noise. Those look identical in a rendered video. They are not identical scientifically, and the whole point of this project was to build the measurement, not the video.

The actual contribution here isn't the simulation itself. It's an operational definition of "arms race" that can be falsified, plus the statistics to test it honestly, including reporting a null or one-sided result if that's what the data shows. Which, as it turns out, is exactly what happened.

## What I built

The world is a continuous 100×100 toroidal plane (wraps at the edges, no boundary weirdness) with static food patches that regenerate over time, up to 2,000 prey and 200 predators, and nothing else. Each agent is a tiny MLP, about 2,600 parameters, that takes in its own energy level plus the relative position and distance of its nearest same-species neighbors, nearest opposite-species neighbors, and (for prey) nearest food, and outputs a turn rate and a speed. That's it. There's no reward function anywhere in this system. An agent survives by not running out of energy, and reproduces asexually with a mutated copy of its own weights once its energy crosses a threshold. Whatever behavior shows up, chasing, fleeing, flocking, has to come from that bare survival-and-reproduction loop, not from anything I told it to optimize for.

The one interesting engineering piece is that every agent in a species has a *different* set of weights, so "run the population's brains forward" isn't one matrix multiply, it's thousands of tiny ones. I used `torch.func.vmap` to batch that across the whole population as vectorized GPU ops instead of looping over agents in Python, which is the only way this runs at a reasonable speed.

Getting that batching to actually not stall the GPU took a couple of false starts. My first version of the reproduction bookkeeping used `nonzero()` and `.item()` in a few places, which quietly forces a GPU-to-CPU sync every single step. Cutting those out (replacing them with cumsum/argsort/scatter tricks that never leave the GPU) should have sped things up. Instead it got slower: about 160 steps/second before, about 30 after. Turned out kernel-launch overhead under Windows, not the syncs, was the real bottleneck. `torch.compile` clawed most of that back, landing around 105 steps/second.

None of this is the interesting part of the project. I mention it because how many agents fit in an evening's compute budget is a real constraint that shaped every experiment below. This is a laptop 4090, not a cluster.

### What "arms race" actually means here

Before running anything, I locked down a definition, because loosening the definition after seeing the data is how you end up calling everything an arms race. An arms race requires two things to both be true:

1. **Both species are genuinely getting better at their job over evolutionary time.** Not just "the chase looks different," but measured improvement in both predators and prey, independently.
2. **Neither side is actually winning.** How often predators actually catch something and how often prey actually get away, in the live co-evolving population, should stay roughly flat over time. If one side's real-world success rate is trending up, that's not a stalemate. That's someone winning.

The tricky part is measuring (1) in a way that isn't contaminated by (2). If I just watch the live population's catch rate go up and call that "predators improving," I can't tell whether predators got smarter or prey got dumber or the population composition shifted for some unrelated reason. So competence is measured separately, against a frozen opponent: at some point early in the run (I used step 2,000, not generation zero, since a genome that's barely finished initializing is a pointlessly easy or pointlessly random benchmark), I freeze one random living genome from each species and never let it evolve again. Every so often, every 2,000 steps, sampling 200 individuals from the current population, I drop a batch of current, evolving genomes into a small standalone arena against a population of clones of that frozen genome, with no food and no death economy, just movement and catch-counting for 200 steps, and measure how well the *evolving* side does against the *fixed* side. That isolates "did this species' behavior actually get better" from "did the population dynamics shift."

For the trend itself, I didn't use plain Mann-Kendall (the standard nonparametric trend test), because plain Mann-Kendall assumes the samples in your series are independent, and mine aren't. A competence measurement taken at generation N and one taken 2,000 steps later are looking at a population that only just changed a little, not two unrelated draws. Treating them as independent inflates your confidence in a trend that might just be autocorrelation. I used the Hamed-Rao modified Mann-Kendall test instead, via the `pymannkendall` package, which corrects the variance estimate for exactly this kind of serial correlation. It's a small detail that would be easy to skip, and it would make the headline result look more confident than it should.

## The result: no arms race. One-sided escalation instead.

I ran 5 independent seeds, 100,000 steps each, about 18 minutes total on the RTX 4090 with `torch.compile`. Here's what came out.

Predator competence against the frozen prey benchmark went up, and it went up in every single seed (p between 0.0005 and 0.0046, Kendall's τ between 0.28 and 0.41). No ambiguity there: predators get better at their job as the run goes on.

Prey competence against the frozen predator benchmark did not go up in a single seed. In two of five runs it actually got significantly worse (p = 0.0003 and p = 0.0004); the other three showed no trend either way. Zero out of five show prey getting better at evading.

And in the live, co-evolving population, the numbers you'd actually care about if you were a prey animal keep getting worse too: realized catch rate rose significantly in four of the five seeds, and realized escape rate fell significantly in all five. Predators are gaining ground, and it isn't leveling off.

So condition one fails (it's not a two-sided improvement, it's one-sided), and condition two fails too, since the realized outcome is trending, not flat. By the definition I locked in before running anything, this is not an arms race. What's actually happening is closer to one-sided predator escalation: predators keep getting measurably better at hunting, prey don't keep pace, and predators end up realizing a growing advantage that shows no sign of leveling off. That's still a real, reportable finding, and one the null-hypothesis framing was built to allow for. Plenty of real ecological and artificial-life systems look like this: stagnation on one side, not a sustained mutual escalation. Calling it an arms race anyway because that's the better story would defeat the entire point of measuring instead of assuming.

I don't have a settled explanation for why prey competence never improves while predator competence reliably does. But the playback gave me a hint, and it's speculation, not a result: watching the population over time, prey visibly clump into dense flocks as they approach the population cap. Nothing in the fitness function rewards this directly; it just falls out of the "sense your nearest same-species neighbors" input, which was there for a generic flocking cue, not to produce clustering. Clustering can be a good evasion strategy in general (safety in numbers, confusion effects), but it also means that once a predator finds a cluster, there's a lot of prey in one place, which could make catching something easier even if catching any particular prey isn't. If that's what's going on, prey competence looking flat while predator catch rate climbs isn't predators solving the problem so much as a side effect of an emergent spatial pattern working against the population that produced it. I haven't run the experiment that would actually confirm this, so take it as a guess, not a finding.

## Is this robust, or just this one config?

Before trusting a result like "no arms race" too much, the obvious next question is whether it's an artifact of the specific numbers I happened to pick: mutation rate, population size, how far agents can sense. Section 7b of the design asks exactly this, so I ran three sweeps, each 3 seeds × 50,000 steps, half the headline run length, since these are robustness checks rather than the primary claim:

- **Mutation rate** at σ = 0.02, 0.05, and 0.15 (0.05 is the default).
- **Population size** at roughly half, the baseline, and double, scaling the world size and food supply proportionally so density, not raw diversity, stays fixed across the comparison.
- **Sensing range asymmetry**: predators seeing 30 units against prey's 15, and the reverse, against a symmetric 20/20 baseline.

Across all nine configurations, the pattern held: realized predator catch rate rose and prey escape rate fell significantly in essentially every seed, and no configuration ever produced both species' competence improving together. Wider or narrower mutation, more or fewer agents, either species getting a sensing advantage: none of it flipped the outcome. Whatever is driving the one-sided escalation, it isn't a knife-edge sensitivity to these particular numbers.

Getting a trustworthy version of that sweep took a second attempt, and it's worth explaining why. My first pass ran all nine configurations back-to-back in a single Python process, to save on `torch.compile`'s warm-up cost, and it produced results that directly contradicted each other for configurations that were, on paper, identical: same population size, same mutation rate, same seeds, just reached via a different sweep. Chasing that down turned up a real bug. Compiling a *different*-shaped simulation earlier in the same process silently corrupted results for a shape that had already been compiled and was now being reused, even with an explicit, correctly-seeded random generator. A fresh process reproduced the same config bit-for-bit; a process that had compiled something else first did not. I don't have a root cause beyond "this is a community Triton build on Windows, not the officially supported combination," and I didn't chase it further than confirming the mechanism and finding a clean workaround: one subprocess per configuration, so every run gets a clean compile cache. It's the kind of bug that doesn't crash, it just gives you the wrong answer without complaining, and I'd rather write that down than pretend the first version of the sweep was fine.

## Limitations

A few things this result should not be read as saying. Only *behavior* evolves here: metabolism, speed, and sensing range are fixed constants per species, not something mutation can touch. That was a deliberate scope cut to keep the measurement clean, but it also means the population can't evolve its way out of the escalation by, say, getting faster or hungrier as a species. A real ecological system has that lever available and this one doesn't. The world itself is toy 2D physics: no obstacles, no third trophic level, no seasons, nothing that would make the food landscape genuinely non-stationary. Population sizes (a few thousand agents) are tiny next to anything in real ecology, which matters for how much genetic drift versus real selection is driving what I'm calling "competence" at any given moment. And a 100,000-step run, however you want to translate that into "generations" for an asexual, continuously-reproducing population, is short by the standards of the evolutionary-biology literature this whole framing borrows from. It's entirely possible a run ten times longer looks qualitatively different. I have no evidence either way, because I haven't run one.

## Reproducing this

```bash
pip install -e ".[dev]"
python scripts/run_experiment.py --seeds 0 1 2 3 4 --n-steps 100000 --out results/experiment.json
python scripts/run_sweeps.py --out-dir results/sweeps
python scripts/render_playback.py --n-steps 10000 --sample-every 40 --out media/playback_demo.gif
```

The headline experiment (5 seeds × 100,000 steps) took about 18 minutes; the full sweep (9 configurations × 3 seeds × 50,000 steps, one subprocess each) took about 72 minutes; the playback GIF a couple of minutes. All of it ran on a single RTX 4090 Laptop GPU (16 GB) with `torch.compile` enabled. Every run here used the seeds listed above, nothing was cherry-picked, and the raw per-seed trend statistics for both the headline experiment and every sweep configuration are checked into `results/` as JSON, not just the summary numbers quoted here.
