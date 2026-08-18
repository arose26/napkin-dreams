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
