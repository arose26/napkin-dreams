"""napkin-dreams: train the policy inside a learned model of the game.

Repo 4 of the napkin-gamemaster series. Repos 1-3 were model-free; this one
learns the game itself -- a small world model trained on replayed experience --
and trains the actor-critic ENTIRELY on imagined rollouts. The real
environment is only ever touched to collect data and to measure the truth.

The napkin question:

    How many real frames does a dream replace -- and how long can you trust
    a dream before the policy starts exploiting its glitches?

Same env as napkin-replay (MinAtar Breakout), so the model-free DQN curve from
that repo overlays directly on the same axes. The arms are dream horizons:

    h3    3-step dreams   (barely more than TD)
    h15   15-step dreams  (Dreamer's usual neighborhood)
    h45   45-step dreams  (long enough to hallucinate)

World model (deliberately deterministic -- no VAE, no KL machinery):
    enc:  conv -> 128-d latent h
    f:    GRUCell(h, a) -> next latent
    dec:  deconv(h) -> 10x10x4 frame logits   (grounds the latent; BCE)
    r,d:  heads on f's output (reward after action, episode-done prob)
Trained on L=5-step open-loop windows: encode the first frame, roll f five
steps, match every step against the encoded truth + reconstruction + reward
+ done. Open-loop training is what gives the horizon question teeth.

Actor-critic in dreams: REINFORCE + learned value baseline on lambda-returns
with soft continues (gamma * (1 - d_hat)), latents detached -- exactly what a
real environment would provide, which is what makes the transparency
selfcheck possible: drive the SAME imagination code with the real simulator
instead of f and it must reproduce the real env's rewards and dones bit for
bit. The learned model is then the only difference between dreaming and
living.

Usage:
    PYTHONPATH=.deps python3.10 napkin_dreams.py selfcheck
    PYTHONPATH=.deps python3.10 napkin_dreams.py train --arm h15 --seed 0
    PYTHONPATH=.deps python3.10 napkin_dreams.py sweep [--shard i --nshards n]
    PYTHONPATH=.deps python3.10 napkin_dreams.py plot
    PYTHONPATH=.deps python3.10 napkin_dreams.py gif      # dream vs reality
"""
import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from minatar import Environment

OUT = Path(__file__).parent / "out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

GAME = "breakout"
GAMMA, LAMBDA = 0.99, 0.95
LATENT = 128
SEQ = 5                  # world-model open-loop training window
TOTAL_REAL = 150_000     # real env steps per run
WARMUP = 5_000           # random-policy prefill
TRAIN_EVERY = 8          # env steps between (1 model batch + 1 AC batch)
MODEL_BATCH = 64         # sequences per model batch
DREAM_BATCH = 256        # imagined rollouts per AC batch
BUFFER = TOTAL_REAL
LR_MODEL, LR_AC = 3e-4, 1e-4
ENTROPY = 1e-3           # constant across arms; dreams collapse without it
SEEDS = 5

ARMS = {"h3": 3, "h15": 15, "h45": 45}


# --------------------------------------------------------------------- models

class WorldModel(nn.Module):
    def __init__(self, channels, actions):
        super().__init__()
        self.actions = actions
        self.enc = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(16 * 8 * 8, LATENT), nn.Tanh())
        self.f = nn.GRUCell(actions, LATENT)
        self.dec = nn.Sequential(
            nn.Linear(LATENT, 16 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(16, channels, 3, 1))
        self.rhead = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(),
                                   nn.Linear(64, 1))
        self.dhead = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(),
                                   nn.Linear(64, 1))

    def step(self, h, a):
        """One imagined step: latent h [B,LATENT], action ids a [B].
        Returns (h', r_hat [B], done_prob [B])."""
        a1 = F.one_hot(a, self.actions).float()
        h2 = self.f(a1, h)
        return h2, self.rhead(h2).squeeze(-1), torch.sigmoid(self.dhead(h2)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(LATENT, 128), nn.ReLU(),
                                nn.Linear(128, actions))
        self.v = nn.Sequential(nn.Linear(LATENT, 128), nn.ReLU(),
                               nn.Linear(128, 1))

    def forward(self, h):
        return self.pi(h), self.v(h).squeeze(-1)


def obs(env):
    return env.state().astype(np.float32).transpose(2, 0, 1)


# --------------------------------------------------------------------- replay

class SeqReplay:
    """Flat ring buffer with contiguous-window sampling that never crosses an
    episode boundary except at a window's final step."""

    def __init__(self, cap, shape, seed):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.o = np.zeros((cap, *shape), np.uint8)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.n = 0

    def add(self, o, a, r, d):
        assert self.n < self.cap, "buffer sized to TOTAL_REAL; never wraps"
        self.o[self.n], self.a[self.n] = o, a
        self.r[self.n], self.d[self.n] = r, d
        self.n += 1

    def sample_windows(self, k, L, max_tries=200):
        """Windows [k, L+1] of obs and [k, L] of a/r/d. Rejection-samples
        starts whose first L-1 steps contain a done (crossing a reset).
        Fails loudly if the buffer has (almost) no done-free windows of
        length L -- an unbounded loop here livelocks on collapsed-policy
        seeds whose episodes are all shorter than L."""
        idx = np.zeros(k, np.int64)
        got = 0
        tries = 0
        while got < k:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"sample_windows: found {got}/{k} done-free windows of "
                    f"length {L} after {max_tries} tries; episodes in the "
                    f"buffer are too short for this window length")
            cand = self.rng.integers(0, self.n - L, size=k)
            ok = np.array([not self.d[c:c + L - 1].any() for c in cand])
            take = cand[ok][:k - got]
            idx[got:got + len(take)] = take
            got += len(take)
        o = np.stack([self.o[i:i + L + 1] for i in idx])
        a = np.stack([self.a[i:i + L] for i in idx])
        r = np.stack([self.r[i:i + L] for i in idx])
        d = np.stack([self.d[i:i + L] for i in idx])
        return o.astype(np.float32), a, r, d

    def sample_obs(self, k):
        idx = self.rng.integers(0, self.n, size=k)
        return self.o[idx].astype(np.float32)


# ----------------------------------------------------------- world model loss

def model_loss(wm, o, a, r, d):
    """Open-loop: encode frame 0, roll f for L steps, match everything.
    o [B,L+1,C,10,10], a/r/d [B,L]."""
    B, Lp1 = o.shape[:2]
    L = Lp1 - 1
    h = wm.enc(o[:, 0])
    losses = []
    for t in range(L):
        h, rhat_raw, _ = wm.step(h, a[:, t])
        dlogit = wm.dhead(h).squeeze(-1)
        rhat = wm.rhead(h).squeeze(-1)
        target_h = wm.enc(o[:, t + 1]).detach()
        losses.append(
            F.binary_cross_entropy_with_logits(wm.dec(h), o[:, t + 1])
            + F.mse_loss(h, target_h)
            + F.mse_loss(rhat, r[:, t])
            + F.binary_cross_entropy_with_logits(dlogit, d[:, t]))
    return sum(losses) / L


# ----------------------------------------------------------------- imagination

def dream_rollout(h0, ac, step_fn, H):
    """Roll H imagined steps from latents h0 [B,LATENT] using step_fn
    (the world model's step, or a real-simulator wrapper in selfcheck).
    Latents are DETACHED at the policy boundary -- the dream behaves like an
    environment, not like a differentiable graph."""
    h = h0.detach()
    logps, ents, vs, rs, ds = [], [], [], [], []
    for _ in range(H):
        logits, v = ac(h)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        vs.append(v)
        h, rhat, dprob = step_fn(h, a)
        h = h.detach()
        rs.append(rhat.detach())
        ds.append(dprob.detach())
    _, v_last = ac(h)
    return (torch.stack(logps), torch.stack(ents), torch.stack(vs),
            torch.stack(rs), torch.stack(ds), v_last)


def lambda_returns(rs, ds, v_last, vs, gamma=GAMMA, lam=LAMBDA):
    """G_t = r_t + gamma*c_t*[(1-lam)*V_{t+1} + lam*G_{t+1}], c = 1 - d_hat.
    rs/ds/vs [H,B], v_last [B]."""
    H = rs.shape[0]
    G = torch.zeros_like(rs)
    nxt = v_last
    for t in reversed(range(H)):
        c = 1.0 - ds[t]
        v_next = v_last if t == H - 1 else vs[t + 1]
        nxt = rs[t] + gamma * c * ((1 - lam) * v_next + lam * nxt)
        G[t] = nxt
    return G


def ac_update(ac, opt, wm, buf, H):
    h0 = wm.enc(torch.as_tensor(buf.sample_obs(DREAM_BATCH), device=DEV))
    logps, ents, vs, rs, ds, v_last = dream_rollout(h0, ac, wm.step, H)
    with torch.no_grad():
        G = lambda_returns(rs, ds, v_last.detach(), vs.detach())
        adv = G - vs.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # de-weight steps after a predicted episode end
        alive = torch.cumprod(torch.cat([torch.ones_like(ds[:1]),
                                         1 - ds[:-1]]), 0)
    loss = (alive * (-logps * adv - ENTROPY * ents)).mean() \
        + 0.5 * (alive * (vs - G) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()


# ---------------------------------------------------------------------- train

def train(arm, seed, total_real=TOTAL_REAL, quiet=False):
    H = ARMS[arm]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = Environment(GAME)
    env.seed(seed)
    env.reset()
    ch, acts = env.state_shape()[2], env.num_actions()
    wm = WorldModel(ch, acts).to(DEV)
    ac = ActorCritic(acts).to(DEV)
    opt_m = torch.optim.Adam(wm.parameters(), lr=LR_MODEL)
    opt_ac = torch.optim.Adam(ac.parameters(), lr=LR_AC)
    buf = SeqReplay(total_real + 8, (ch, 10, 10), seed)

    s = obs(env)
    ep_ret, returns = 0.0, deque(maxlen=20)
    curve = []
    for step in range(1, total_real + 1):
        if step <= WARMUP:
            a = int(rng.integers(acts))
        else:
            with torch.no_grad():
                h = wm.enc(torch.as_tensor(s[None], device=DEV))
                a = int(Categorical(logits=ac(h)[0]).sample())
        r, done = env.act(a)
        buf.add((s > 0).astype(np.uint8), a, r, float(done))
        ep_ret += r
        if done:
            returns.append(ep_ret)
            ep_ret = 0.0
            env.reset()
        s = obs(env)

        if step > WARMUP and step % TRAIN_EVERY == 0:
            o, aa, rr, dd = buf.sample_windows(MODEL_BATCH, SEQ)
            loss = model_loss(wm, torch.as_tensor(o, device=DEV),
                              torch.as_tensor(aa, device=DEV),
                              torch.as_tensor(rr, device=DEV),
                              torch.as_tensor(dd, device=DEV))
            opt_m.zero_grad(); loss.backward(); opt_m.step()
            ac_update(ac, opt_ac, wm, buf, H)

        if step % 5_000 == 0:
            avg = float(np.mean(returns)) if returns else 0.0
            curve.append((step, avg))
            if not quiet and step % 25_000 == 0:
                print(f"  {arm} seed {seed}  step {step:>7}  return {avg:6.2f}",
                      flush=True)
    return curve, wm, ac, buf


# ------------------------------------------------------------------ divergence

def divergence(wm, buf, horizon=45, k=256):
    """Open-loop dream vs recorded truth: per-step reconstruction BCE of the
    dream against the actual frames, from held-out windows."""
    o, a, _, _ = buf.sample_windows(k, horizon)
    o = torch.as_tensor(o, device=DEV)
    a = torch.as_tensor(a, device=DEV)
    with torch.no_grad():
        h = wm.enc(o[:, 0])
        per_step = []
        for t in range(horizon):
            h, _, _ = wm.step(h, a[:, t])
            bce = F.binary_cross_entropy_with_logits(
                wm.dec(h), o[:, t + 1], reduction="mean")
            per_step.append(float(bce))
    return per_step


# ---------------------------------------------------------------------- sweep

def run_sweep(total_real, seeds, shard=0, nshards=1):
    OUT.joinpath("sweep").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = [(arm, s) for arm in ARMS for s in range(seeds)][shard::nshards]
    for k, (arm, seed) in enumerate(todo):
        f = OUT / "sweep" / f"{arm}_{seed}.json"
        if f.exists():
            continue
        curve, wm, ac, buf = train(arm, seed, total_real, quiet=True)
        try:
            div = divergence(wm, buf)
        except RuntimeError as e:
            print(f"  {arm} seed {seed}: divergence skipped ({e})", flush=True)
            div = None
        f.write_text(json.dumps(dict(curve=curve, divergence=div)))
        print(f"[{k + 1:2}/{len(todo)}] {arm:4} seed {seed}  "
              f"final {curve[-1][1]:6.2f}  elapsed {(time.time() - t0) / 60:5.1f} min",
              flush=True)
    print("sweep done")


def load_sweep():
    runs = {}
    for f in (OUT / "sweep").glob("*.json"):
        arm, seed = f.stem.rsplit("_", 1)
        runs.setdefault(arm, {})[int(seed)] = json.loads(f.read_text())
    return runs


def iqm(x, axis=None):
    x = np.sort(np.asarray(x, np.float64), axis=axis)
    n = x.shape[-1] if axis in (None, -1) else x.shape[axis]
    lo, hi = n // 4, n - n // 4
    sl = [slice(None)] * x.ndim
    sl[-1 if axis in (None, -1) else axis] = slice(lo, hi)
    return x[tuple(sl)].mean(axis=axis)


def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_sweep()
    colors = dict(h3="#4477aa", h15="#228833", h45="#ee7733")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    results = {}
    for arm in ARMS:
        seeds = runs[arm]
        curves = np.array([seeds[s]["curve"] for s in sorted(seeds)])
        x = curves[0, :, 0]
        for c in curves:
            axes[0].plot(c[:, 0], c[:, 1], color=colors[arm], alpha=0.15, lw=0.6)
        axes[0].plot(x, iqm(curves[:, :, 1], axis=0), color=colors[arm],
                     lw=2.0, label=f"{arm} (dream horizon {ARMS[arm]})")
        divs = np.array([seeds[s]["divergence"] for s in sorted(seeds)
                         if seeds[s]["divergence"] is not None])
        if divs.size:
            axes[1].plot(range(1, divs.shape[1] + 1), divs.mean(0),
                         color=colors[arm], lw=2.0, label=arm)
        tail = curves[:, x >= 0.9 * x[-1], 1].mean(1)
        results[arm] = dict(iqm=float(iqm(tail)),
                            ci=[float(v) for v in bootstrap_ci(tail)],
                            seeds=[float(v) for v in tail])

    base = Path(__file__).parent / "assets" / "baseline_dqn.json"
    if base.exists():
        b = json.loads(base.read_text())
        axes[0].plot(b["steps"], b["iqm"], color="#333333", ls="--", lw=1.8,
                     label="model-free DQN (napkin-replay, full)")
    axes[0].set_title("real return vs REAL env steps -- dreams do the training")
    axes[0].set_xlabel("real env steps"); axes[0].set_ylabel("episode return")
    axes[0].legend(fontsize=8)
    axes[1].set_title("open-loop dream vs reality (held-out), per-step BCE")
    axes[1].set_xlabel("dream step"); axes[1].set_ylabel("reconstruction BCE")
    axes[1].axvline(3, color="#4477aa", ls=":", lw=0.8)
    axes[1].axvline(15, color="#228833", ls=":", lw=0.8)
    axes[1].axvline(45, color="#ee7733", ls=":", lw=0.8)
    axes[1].legend(fontsize=8)

    order = list(ARMS)
    vals = [results[a]["iqm"] for a in order]
    errs = np.array([[results[a]["iqm"] - results[a]["ci"][0],
                      results[a]["ci"][1] - results[a]["iqm"]] for a in order]).T
    axes[2].bar(range(len(order)), vals, yerr=errs, capsize=4,
                color=[colors[a] for a in order])
    axes[2].set_xticks(range(len(order)), order)
    axes[2].set_title("final IQM real return, 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.png'} and results.json")


# ------------------------------------------------------------------------ gif

PALETTE = np.array([[60, 60, 70], [200, 60, 40], [240, 200, 60], [80, 160, 220],
                    [120, 200, 120], [180, 120, 200]], np.uint8)


def frame_rgb(state_chw, scale=24):
    img = np.zeros((10, 10, 3), np.uint8) + 15
    for c in range(state_chw.shape[0]):
        img[state_chw[c] > 0.5] = PALETTE[c % len(PALETTE)]
    return np.kron(img, np.ones((scale, scale, 1), np.uint8))


def make_gif():
    from PIL import Image
    print("training h15 for the gif (full run)...")
    _, wm, ac, buf = train("h15", seed=0, quiet=True)
    # one real trajectory, then the dream re-lives it from the same start
    o, a, _, _ = buf.sample_windows(1, 45)
    o_t = torch.as_tensor(o, device=DEV)
    a_t = torch.as_tensor(a, device=DEV)
    with torch.no_grad():
        h = wm.enc(o_t[:, 0])
        frames = []
        for t in range(45):
            h, _, _ = wm.step(h, a_t[:, t])
            dream = torch.sigmoid(wm.dec(h))[0].cpu().numpy()
            real = o[0, t + 1]
            left = frame_rgb(real)
            right = frame_rgb(dream)
            sep = np.full((left.shape[0], 8, 3), 60, np.uint8)
            frames.append(Image.fromarray(
                np.concatenate([left, sep, right], 1)).convert("P"))
    frames[0].save(OUT / "dream.gif", save_all=True, append_images=frames[1:],
                   duration=180, loop=0)
    print(f"wrote {OUT / 'dream.gif'} (left: reality, right: the dream, "
          "same 45 actions)")


# ------------------------------------------------------------------ selfcheck

def selfcheck():
    env = Environment(GAME)
    env.reset()
    ch, acts = env.state_shape()[2], env.num_actions()

    # 1. THE transparency assert: drive dream_rollout with the real simulator
    #    instead of the world model. The imagination machinery (action loop,
    #    reward/done bookkeeping) must reproduce the real env bit for bit.
    torch.manual_seed(0)
    ac = ActorCritic(acts).to(DEV)
    sim = Environment(GAME)
    sim.seed(42)
    sim.reset()
    real_rs, real_ds = [], []

    def sim_step(h, a):
        r, done = sim.act(int(a[0]))
        real_rs.append(float(r)); real_ds.append(float(done))
        if done:
            sim.reset()
        return (torch.as_tensor(obs(sim).reshape(1, -1)[:, :LATENT], device=DEV),
                torch.tensor([float(r)], device=DEV),
                torch.tensor([float(done)], device=DEV))

    h0 = torch.zeros(1, LATENT, device=DEV)
    _, _, _, rs, ds, _ = dream_rollout(h0, ac, sim_step, H=60)
    assert rs.squeeze(1).tolist() == real_rs and ds.squeeze(1).tolist() == real_ds
    print("imagination machinery is exactly transparent to a real simulator "
          f"(60 steps, {sum(real_rs):.0f} reward, {int(sum(real_ds))} resets)")

    # 2. lambda-return identity: with d=0 and lam=1 it is the discounted MC sum
    #    with a bootstrap tail (hand recursion); with lam=0 it is one-step TD.
    rng = np.random.default_rng(0)
    H, B = 12, 4
    rs = torch.as_tensor(rng.normal(size=(H, B)).astype(np.float32))
    vs = torch.as_tensor(rng.normal(size=(H, B)).astype(np.float32))
    vl = torch.as_tensor(rng.normal(size=B).astype(np.float32))
    zeros = torch.zeros(H, B)
    G1 = lambda_returns(rs, zeros, vl, vs, lam=1.0)
    hand = vl.clone()
    for t in reversed(range(H)):
        hand = rs[t] + GAMMA * hand
        assert torch.allclose(G1[t], hand, atol=1e-5)
    G0 = lambda_returns(rs, zeros, vl, vs, lam=0.0)
    for t in range(H):
        v_next = vl if t == H - 1 else vs[t + 1]
        assert torch.allclose(G0[t], rs[t] + GAMMA * v_next, atol=1e-6)
    print("lambda-returns: lam=1 == MC+bootstrap, lam=0 == one-step TD")

    # 3. a predicted done gates the return: d=1 at step t kills credit beyond it.
    ds = torch.zeros(H, B); ds[5] = 1.0
    Gd = lambda_returns(rs, ds, vl, vs, lam=1.0)
    assert torch.allclose(Gd[5], rs[5], atol=1e-6)
    print("soft continues: G at a sure done is exactly r (no leakage)")

    # 4. replay windows never straddle a reset.
    buf = SeqReplay(2000, (ch, 10, 10), seed=1)
    e2 = Environment(GAME); e2.seed(1); e2.reset()
    rng2 = np.random.default_rng(1)
    for _ in range(1500):
        a = int(rng2.integers(acts))
        r, done = e2.act(a)
        buf.add((obs(e2) > 0).astype(np.uint8), a, r, float(done))
        if done:
            e2.reset()
    _, _, _, dw = buf.sample_windows(200, SEQ)
    assert not dw[:, :-1].any(), "window crosses an episode boundary"
    print("replay windows never straddle a reset (200 sampled)")

    # 4b. sampling fails loudly (no livelock) when every episode is shorter
    # than the requested window.
    short = SeqReplay(200, (ch, 10, 10), seed=1)
    for i in range(100):
        short.add(np.zeros((ch, 10, 10), np.uint8), 0, 0.0, 1.0)  # done every step
    try:
        short.sample_windows(4, 5, max_tries=20)
        raise AssertionError("sample_windows should have raised on all-done buffer")
    except RuntimeError:
        pass
    print("sample_windows raises (not livelocks) when windows are infeasible")

    # 5. world model overfits one batch (all four heads wired right).
    torch.manual_seed(1)
    wm = WorldModel(ch, acts).to(DEV)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    o, a, r, d = buf.sample_windows(16, SEQ)
    o, a, r, d = (torch.as_tensor(x, device=DEV) for x in (o, a, r, d))
    for i in range(400):
        loss = model_loss(wm, o, a, r, d)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.15, float(loss)
    print(f"world model overfits one batch: loss {float(loss):.3f}")

    # 6. the latent does not collapse while doing it. Collapse means distinct
    #    inputs stop mapping to distinct latents, so the test is pairwise
    #    distance, not per-dim std (which is small at init by tanh scaling and
    #    would fail a healthy encoder).
    with torch.no_grad():
        hs = wm.enc(o[:, 0])
        pd = torch.cdist(hs, hs)
        pd = pd[~torch.eye(len(hs), dtype=bool, device=DEV)]
    assert float(pd.mean()) > 0.3, f"latent collapse: mean pairwise {pd.mean()}"
    print(f"latent alive: mean pairwise distance {float(pd.mean()):.2f}")

    # 7. dreams improve a policy against that overfit model (end-to-end AC).
    ac = ActorCritic(acts).to(DEV)
    opt_ac = torch.optim.Adam(ac.parameters(), lr=1e-3)
    with torch.no_grad():
        h0 = wm.enc(o[:, 0])
        _, _, _, rs0, _, _ = dream_rollout(h0, ac, wm.step, 10)
    for _ in range(150):
        ac_update(ac, opt_ac, wm, buf, 10)
    with torch.no_grad():
        _, _, _, rs1, _, _ = dream_rollout(h0, ac, wm.step, 10)
    assert float(rs1.sum()) >= float(rs0.sum()) - 1e-3
    print(f"dream training raises dream return "
          f"{float(rs0.mean()):.4f} -> {float(rs1.mean()):.4f}")

    print("\nall selfchecks passed")


# ------------------------------------------------------------------------ cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=ARMS, default="h15")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=TOTAL_REAL)
    s = sub.add_parser("sweep")
    s.add_argument("--steps", type=int, default=TOTAL_REAL)
    s.add_argument("--seeds", type=int, default=SEEDS)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    sub.add_parser("plot")
    sub.add_parser("gif")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.cmd == "selfcheck":
        selfcheck()
    elif args.cmd == "train":
        t0 = time.time()
        curve, *_ = train(args.arm, args.seed, args.steps)
        print(f"final return {curve[-1][1]:.2f}  ({time.time() - t0:.0f}s)")
    elif args.cmd == "sweep":
        torch.set_num_threads(2)
        run_sweep(args.steps, args.seeds, args.shard, args.nshards)
    elif args.cmd == "plot":
        make_plots()
    elif args.cmd == "gif":
        make_gif()
