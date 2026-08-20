# napkin-dreams

Repo 4 of the **[napkin-gamemaster series](https://github.com/arose26/napkin-gamemaster)** (series home and index). Repos 1–3 were model-free. This one learns the game itself — a small world model trained on replayed experience — and trains the actor-critic **entirely inside its own dreams**. The real environment is touched only to collect data and to measure the truth.

> **How many real frames does a dream replace — and how long can you trust a dream before the policy starts exploiting its glitches?**

![dream](assets/dream.gif)

*Left: reality. Right: the model's dream, given the same start and the same 45 actions. The
visible degradation is not gif-compression — it IS the result (see the trust probe below).*

## The experiment

Same env as [napkin-replay](https://github.com/arose26/napkin-replay) (MinAtar Breakout, sticky actions), so its model-free DQN curve overlays directly on the same real-env-steps axis. The arms are **dream horizons**:

| arm | dream length | why |
|---|---|---|
| `h3` | 3 steps | barely more than TD — is imagination even needed? |
| `h15` | 15 steps | Dreamer's usual neighborhood |
| `h45` | 45 steps | long enough to hallucinate |

World model, deliberately deterministic (no VAE, no KL machinery): conv encoder → 128-d latent, GRU dynamics over (latent, action), deconv reconstruction (grounds the latent), reward + done heads. Trained on **5-step open-loop windows** — encode once, roll the dynamics, match every step — which is what gives the horizon question teeth. Actor-critic: REINFORCE + value baseline on λ-returns with soft continues, latents detached at the policy boundary.

That detachment buys the repo's signature `selfcheck`: the imagination loop is a function of an arbitrary `step_fn`, so the check drives it with the **real simulator** instead of the model and asserts the machinery reproduces the env's rewards and resets **bit for bit**. After that, the learned model is provably the only difference between dreaming and living.

150k real env steps per run, 5 seeds per arm, IQM + bootstrap CIs, ties reported as ties. Each run also measures **open-loop divergence**: per-step reconstruction BCE of a 45-step dream against held-out reality.

## Hypothesis (registered before the sweep ran)

1. **Dreams are more real-frame-efficient early**: `h15` sits above the DQN curve for the first ~50k real steps (each real frame is reused by ~every dream), and stays at or slightly above it at 150k.
2. **Horizon is an inverted U**: `h15` > `h3` (3 steps can't span a rally) and `h15` > `h45` (divergence eats the long dreams).
3. Divergence BCE grows monotonically in dream depth with a visible knee well before step 45 — and `h45`'s policy quality should track that knee, not the nominal horizon.
4. The gif will show the model hallucinating at long horizons (bricks healing, ball duplicating) — the glitches the policy learns to exploit.

## Results

![results](assets/results.png)

| arm | final IQM real return (95% CI) |
|---|---|
| `h3` | **0.56** (0.47 – 1.19) |
| `h15` | **0.50** (0.32 – 0.88) |
| `h45` | **0.49** (0.34 – 0.65) |
| model-free DQN, same env, same 150k real steps | **6.67** |

**The registered hypotheses were mostly wrong, and the plot says so.**

1. **Dreams were not more frame-efficient — not early, not ever.** The model-free DQN curve
   crosses the dream arms within the first ~20k real steps and ends **13× higher** at the same
   real-frame budget. At napkin scale (deterministic 128-d world model, REINFORCE-only actor),
   dreaming does not buy sample efficiency on MinAtar Breakout; it costs an order of magnitude.
2. **No inverted U detected.** h3 ≈ h15 ≈ h45: every CI overlaps every other at n=5. That is a
   failure to detect an ordering, not proof of equivalence — either way, no winner is claimed
   and the registered U did not show up.
3. **Divergence grows in dream depth as predicted** — but with a twist we didn't register:
   `h45`'s world model diverges *fastest* (BCE knee near step 10; 0.346 at step 20 against
   `h3`'s 0.161 and `h15`'s 0.186, all 5 seeds),
   not just its policy. The model recipe (architecture, schedule, hyperparameters) is identical
   across arms; what differs is the data each arm's own policy collects. So "dream longer"
   degraded the dream itself through the data it gathered, not through the training recipe.
4. One `h15` seed collapsed (seed 4: tail-mean return 0.25, the statistic the table uses; its
   last curve point is 0.35) — **included** in the return IQM above. Its divergence probe first
   came back **null**: bounded rejection sampling drew ~51k candidate starts and accepted 0 of
   the 256 windows needed. That null was a **false negative**, and this repo says so because it
   went back and checked. Rerunning the seed with the exact-enumeration fallback reproduced the
   training curve bit for bit (same 0.35 final) and *did* find usable windows — they were rare
   enough for 51k random draws to miss, not absent (see INSIGHTS #3). Every divergence curve
   below therefore averages **5 of 5 seeds**. The recovered seed is indeed h15's worst
   (BCE 0.339 at dream step 20 against the arm's 0.186 mean), which is exactly why leaving it
   out would have flattered h15. Two `h45` seeds had only 8–16 valid windows; their probes use
   exact enumeration of those.

### Where the failure actually lives: the trust probe

A skeptical eye on the gif (credit: a reviewer's five-second look) forced the question the
aggregate numbers dodge: is the *idea* of dreaming losing here, or is the dream itself too broken
to learn from? `probe` retrains the reference h15 run and scores the model **open-loop against
do-nothing baselines on the same held-out windows** (`assets/trust_probe.json`):

| what the dream predicts | beats its no-skill baseline until step |
|---|---|
| pixels (vs frozen first frame — "nothing ever moves") | **13** |
| reward (vs constant base rate) | 19 (first transient loss at 10) |
| episode end (vs constant base rate) | **5** — then confidently wrong (BCE ~7 vs 0.12 by step 20) |

The done row needs its direction stated, because the obvious reading is the wrong one. These
windows contain **no reset** except possibly at their final step, so the truth is "the episode
does not end here" at every step scored — the constant base rate gets BCE 0.124 by simply saying
so. A BCE near 7 means the model asserted the opposite with ~99.9% confidence: **it invents
episode ends that never happen**, rather than missing real ones.

The actor never sees pixels; it trains on latents through the reward and done heads. Hallucinated
episode ends **by dream step 5** poison the λ-returns of even the shortest useful dreams, and they
poison them in a specific way: soft continues multiply future credit by (1 − d̂), so a phantom
done at step 5 silently deletes the rest of the dream's return. That is the bottleneck, measured
at the exact interface the dream gradient consumes.

One boundary in the divergence plot, for the same reason: a window is allowed to end on a reset —
otherwise the done head would never see a positive example at all — so only a curve's final point
can be scored against a post-reset frame. `plot` draws the reset-free prefix; `assets/results.json`
keeps all 45 steps.

**The verdict, scoped to what was measured:** at napkin scale, *this* model class — deterministic
128-d latent, GRU dynamics, trained on 5-step windows — cannot dream well enough to train on, so
**napkin-gamemaster goes model-free**, citing these numbers. Nothing here indicts model-based RL
with a model good enough to dream past its horizon; Dreamer-scale machinery is exactly what this
repo deliberately excluded.

## Run it

```bash
pip install --target .deps "numpy<2" minatar
PYTHONPATH=.deps python3.10 napkin_dreams.py selfcheck   # ~4 min
PYTHONPATH=.deps python3.10 napkin_dreams.py sweep       # 3 arms x 5 seeds, overnight-able
PYTHONPATH=.deps python3.10 napkin_dreams.py plot
PYTHONPATH=.deps python3.10 napkin_dreams.py gif         # dream-vs-reality, side by side
```

`selfcheck` asserts: **imagination is exactly transparent to a real simulator** (60 steps, rewards and resets identical); λ-returns reduce to MC+bootstrap at λ=1 and one-step TD at λ=0 against hand recursions; a sure predicted done passes exactly `r` (no credit leaks across episode ends); replay windows never straddle a reset; the world model overfits one batch; the latent stays distinguishable while doing it (pairwise distance, not per-dim std — see INSIGHTS); dream training raises dream return end-to-end.

## What's deliberately not here

No stochastic latents, no KL balancing, no discrete codes, no straight-through gradients — Dreamer's machinery answers "how do I dream well at scale"; this repo asks the napkin question one level down. No dynamics-gradient actor (latents are detached; REINFORCE only), because that's what makes the transparency assert possible at all. Entropy bonus 1e-3 everywhere: unlike napkin-returns, dream training does thousands of updates on a narrow state distribution and collapses without it; it is a fixed background ingredient, not an arm.

## Model

Encoder conv 3×3 (16ch) → 128-d tanh latent; GRUCell dynamics; deconv decoder; 64-unit reward/done heads. Adam 3e-4 (model) / 1e-4 (AC), γ=0.99, λ=0.95, model batch 64×5-step windows, dream batch 256, one model batch + one AC batch every 8 env steps after 5k warmup. ~25 min per 150k-step run (h15) on an RTX 4050.
