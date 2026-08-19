# Insights from building napkin-dreams

Written in the order I hit them, same convention as the rest of the series.

## 1. "Collapse" is about distinguishability, not variance — aim the assert at the right quantity

The anti-collapse selfcheck asserted mean per-dimension latent std > 0.05 and failed — on a
provably healthy encoder. The tanh-squashed 128-d latent has per-dim std ≈ 0.021 *at
initialization*, and 16 near-identical Breakout frames (same bricks, ball a few pixels apart)
legitimately keep it there. Meanwhile mean pairwise latent distance was 1.35: distinct inputs
mapped to robustly distinct latents, which is the thing imagination actually needs.

The napkin-diffusion rule ("if an assert needs a suspiciously loose tolerance, it's aimed at the
wrong quantity") has a converse that bit here: if a healthy system fails a tight assert, don't
loosen it — re-derive what property you actually require. Collapse means "inputs become
indistinguishable", so the assert now measures pairwise distance.

## 2. Detaching the dream latents is what makes the machinery testable

Dreamer backpropagates the actor's objective through the learned dynamics. This repo doesn't —
latents are detached at the policy boundary, REINFORCE only. That costs gradient quality but buys
the series' favorite kind of assert: the imagination loop takes an arbitrary `step_fn`, so the
selfcheck can drive it with the *real simulator* and require rewards and resets to match the env
bit for bit. A dream loop that needed dynamics gradients could never be run against reality (the
real env has no gradients), and its bookkeeping would be untestable by substitution.

**Takeaway:** when choosing between two designs, "which one can be tested by swapping in the
ground truth" is a real design criterion, not an afterthought.

## 3. An unbounded rejection sampler is a livelock wearing a correctness proof

The divergence probe rejection-samples 45-step windows that don't cross an episode reset. The
loop was "obviously correct": keep drawing until you have 256 clean windows. Then one h15 seed
collapsed — final return 0.25, every episode shorter than 45 steps — and the probe didn't fail,
didn't error, didn't time out. It spun at 100% CPU forever, in numpy, invisible to any check
that only asks "is the process alive and busy?". It cost hours to notice precisely because the
failure mode looks identical to healthy training from the outside.

Two fixes, both now selfchecked. First: bound the loop and **fail loudly** — a raised error
naming found/needed counts turns an invisible hang into a one-line diagnosis. Second, and only
after the loud version revealed the true shape of the problem: two other seeds had *rare but
real* clean windows (8 and 16 of the needed 256) that bounded rejection would wrongly abandon,
so exhaustion now falls back to exact enumeration of valid starts — terminating, and null only
when null is true (that collapsed seed's divergence is reported as null, not fabricated).

**Takeaway:** any `while` loop whose exit depends on properties of the data, not of the code,
needs a bound and a loud exit. And the first loud failure is diagnostic gold: it told us the
difference between "no valid windows" and "rare valid windows", which the hang never could.

## 4. The long-horizon arm didn't just trust its model further — it had a worse model

The registered hypothesis blamed long dreams for *exploiting* a fixed-quality model. The
measurement says more: h45's open-loop BCE rises ~2× faster than h3/h15's (knee near dream step
10). The model's recipe — architecture, schedule, hyperparameters — is identical across arms;
what differs is the data, because each arm's own policy collects it. So the arm knob reached the
model through the buffer: "dream longer" degraded the dream itself via the experience it
gathered, not just the policy's use of it. Registered as a surprise, not an explanation.

## 5. The gif caught what the aggregates normalized away

The sweep numbers were consistent with two very different stories: "dream training doesn't pay at
this scale" and "this particular world model is too broken to dream with". The published verdict
initially leaned on the first. A reviewer looked at the hero gif for five seconds and called it:
the dream visibly falls apart mid-rollout, so blame the model, not the idea.

The follow-up probe made it quantitative, and at the right interface: the actor never sees
pixels — it trains on latents through the reward and done heads. Measured open-loop against
do-nothing baselines: pixels beat a frozen frame only to step 13, and done predictions are worse
than a constant base rate from step 5, then confidently wrong (BCE ~7 vs 0.12). With soft
continues feeding λ-returns, every dream longer than a few steps was training the policy on a
corrupted notion of when episodes end.

**Takeaway (two-part):** look at the artifact before you believe the aggregate — renderable
outputs disambiguate causal stories in seconds. And when a conclusion is negative, measure the
failing component against a no-skill baseline at the interface downstream code actually consumes,
then scope the claim to that component. "X fails" and "my 5MB instance of X's key module fails"
are different findings; only the second was earned here.
