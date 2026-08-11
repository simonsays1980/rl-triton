"""ONE-OFF measurement: GAE's share of a real PPO update step, as a curve.

Purpose: pre-empt "does this speed up training?" by honestly measuring what
fraction of a full PPO update step GAE actually is. NOT a recurring benchmark --
run once, report the resulting numbers for the paper's evaluation section. Do
NOT wire this into bench_release.py, benchmarks.md, or README.md, and do not
draw a "we win/lose" conclusion here -- that is a STOP-and-report item for a
human to sign off on. A null result (GAE stays negligible everywhere) is a
publishable outcome and must be reported plainly if that is what comes out.

The original version of this measurement reported a single point in the design
space, and it was the point LEAST favourable to the kernel: a large policy net,
seq_len=128, and one GAE call amortized over 16 backward passes. This version
sweeps the three axes along which GAE's addressable share is expected to rise,
so the claim is measured rather than inferred:

  Axis A (net size):  hidden (64,64) ... (1024,1024) at fixed (num_envs, seq_len).
  Axis B (seq_len):   two ladders, see below.
  Axis C (reuse):     epochs x minibatches, i.e. how many backward passes one
                      GAE call is amortized over, plus a recompute-per-epoch
                      row where GAE is recomputed every epoch instead of once.

Axis B is run as TWO distinct ladders because they answer different questions
and neither alone is sufficient:

  "budget" ladder -- num_envs * seq_len held constant at 524288
      (4096,128) -> (1024,512) -> (256,2048) -> (128,4096).
      Total work, minibatch size, and backward cost are all constant, so a
      rising GAE share isolates the O(T) serial effect. This is the clean
      control.
  "envs" ladder -- num_envs held at 4096, seq_len grows
      (4096,128) -> (4096,512) -> (4096,2048) -> (4096,4096).
      Recognizable as a real deployment shape, but total work grows 32x with
      T, so it conflates "longer T" with "more work". This is the realistic
      case, not the control.
  They are labelled distinctly in the output. Do not merge them into one table.

METHODOLOGY NOTE 1 -- interleaving (fixed in an earlier version, still active).
Running the Triton-GAE arm and the torch.compile-GAE arm as separate timed
blocks made the "optimizer" stage come out 43% different between arms (0.294ms
vs 0.518ms) -- impossible from a GAE-only change, since GAE finishes well
before the optimizer step and has no way to affect Adam's cost. That is the
signature of the two arms not being measured identically (GPU clock state /
allocator warmth drifting between the two blocks). Fix: interleave both arms
in ONE process, same seeds, alternating which arm runs first every iteration,
so any slow drift affects both arms equally instead of contaminating whichever
arm ran second.

METHODOLOGY NOTE 2 -- CUDA events (fixed here).
The previous version timed every stage with `torch.cuda.synchronize()` +
`time.perf_counter()`, and computed the step total as the SUM of those stage
timers. That puts ~82 full device drains per iteration (5 stages x 16
minibatches, plus the pre-pass) inside the denominator. MEASURED (2026-08-11, --grid legacy --timing both): the effect is far larger
than the "<1% at large nets" this note originally estimated. At
hidden=(1024,1024) the true total is 166ms under sync timing vs 126ms under
event timing -- the drains inflated the denominator by ~24%. The per-stage
"gae" figure is inflated far worse: 0.30ms under sync vs 0.012ms under events,
a ~25x gap, because a device drain around a ~12us kernel is almost entirely
drain. A roofline check confirms the event number is the real one: at
4096x128, GAE moves ~8MB, which at H100 HBM bandwidth is ~3us -- so 12us is
plausible kernel time and 0.30ms is not.
Consequence for the previously published rows: they used the sync method, so
their denominators were ~24% too large AND their GAE numerators ~25x too
large. The numerator error dominates, so the published shares (0.53%, 0.13%)
are OVERSTATED, not understated. The corrected shares are ~0.03%.

Fix: per-stage timing now uses torch.cuda.Event(enable_timing=True) pairs
recorded on the stream, with a single synchronize() at the END of the whole
iteration. Events cost a stream marker rather than a device drain. In addition
one event pair wraps the ENTIRE update, giving a true measured total; the
report prints both that and the sum-of-stages, and their gap (`resid`) is a
direct readout of residual per-stage instrumentation overhead. GAE share and
speedup are computed against the TRUE total, never the sum.
`--timing sync` reproduces the OLD method on the same card in the same process
so the methodology shift can be sized with hardware held fixed; `--timing both`
runs each config under both and prints them side by side.

METHODOLOGY NOTE 3 -- the two arms do not do equal work, deliberately.
compute_gae() is called with truncateds=None and takes its no-truncation
kernel path. vectorized_gae() unconditionally allocates two [num_envs,
seq_len] zero tensors (`truncateds`, `full_bootstrap`) and dispatches to
vectorized_gae_with_truncations regardless. So the baseline arm pays two extra
full-size allocations plus the truncation-aware scan. That is a fair
representation of what a user calling each API actually gets, but it is NOT a
pure scan-vs-scan comparison, and at seq_len=4096 those extra allocations are
32x larger than at seq_len=128 -- large enough that it stops being a footnote
and starts being a visible fraction of the baseline arm's GAE time. To make
the asymmetry visible rather than argued, the report shows GAE DEVICE time
(kernel time alone, via events around the gae call only) alongside the
full-call time for both arms.

METHODOLOGY NOTE 4 -- TF32 and the launch-bound regime at small nets.
torch.set_float32_matmul_precision("high") is set deliberately (see comment at
the setting). That rationale is size-dependent: at hidden=(64,64) the GEMMs
are [B,60]x[60,64] -- K=60 is not a multiple of 8, N=64 is small, and the
matmul is bandwidth-bound on the activation -- so TF32 buys close to nothing
there. The setting is kept for consistency with the larger sizes; it simply
becomes inert at the small end rather than distorting it.
The more important small-net effect is that the step becomes LAUNCH-BOUND: 16
minibatches x ~10 small kernels each, none of which saturate the GPU. If GAE's
share rises at small hidden sizes, that is therefore only partly Amdahl's law
on a shrinking net -- it is also "the whole step became overhead-dominated".
The honest claim is narrower than "the kernel matters more": it is "credit
assignment stops being rounding error, in a regime where the entire step is
overhead-dominated". `--probe-launch-bound` instruments this directly (see
_probe_launch_bound) so it is reported as a measurement, not an argument.

Setup: synthetic rollout buffer (no real env -- this measures update-step cost,
not env-interaction cost), Isaac-Gym-Ant-like MLP policy+value net (obs_dim=60,
action_dim=8, continuous Gaussian policy). Full PPO update per measured
iteration: GAE once per rollout (or once per epoch, on recompute rows), then
n_epochs x n_minibatches of (forward, clipped policy loss + value loss,
backward, Adam step). Both eager and torch.compile'd policy/value nets.

Per-iteration times are retained (not accumulated into a running sum), so the
report gives median and IQR per stage, not a bare mean. The paper's claim is
that end-to-end speedup is "within noise of 1x"; that sentence needs an actual
noise estimate behind it.

Memory: the (4096,4096) config in the "envs" ladder allocates a ~4.8GB rollout
buffer, and each minibatch gather materializes ~1.0GB more, with both arms
live simultaneously. Peak is roughly 12-18GB. That fits an H100 80GB / H200,
and does NOT fit a 24GB card. Configs are sized for H100 80GB. Nothing here
silently rescales num_envs -- if a config does not fit, it OOMs and is
reported as skipped.

Usage:
    python benchmarks/ppo_e2e_measurement.py                  # full sweep
    python benchmarks/ppo_e2e_measurement.py --grid axis-a    # net-size curve only
    python benchmarks/ppo_e2e_measurement.py --grid legacy --timing both
                                                              # old-vs-new methodology
    python benchmarks/ppo_e2e_measurement.py --probe-launch-bound
"""
import argparse
import datetime
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Silence torch.compile/dynamo symbolic-shapes warnings (e.g. "q1 is not in
# var_ranges, defaulting to unknown range") that otherwise spam stdout on
# every torch.compile call in this sweep -- same setting, same reason as
# tests/bench_release.py. setdefault, so an explicit caller override still
# wins. Must be set before `import torch`.
os.environ.setdefault("TORCH_LOGS", "-dynamic")

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_gae import vectorized_gae, reference_gae
from rl_triton.ops.gae import compute_gae

torch._dynamo.config.cache_size_limit = 64  # avoid the silent eager-fallback bug from Experiment 1
# TF32 tensor cores for fp32 matmul: a real production PPO deployment on Ampere/Hopper would
# normally enable this. Leaving it off (the default) makes forward/backward artificially slow,
# which understates GAE's percentage share of the step relative to a realistically-tuned
# production config (same absolute GAE-vs-baseline gap, smaller denominator with TF32 on).
# NOTE: this reasoning is size-dependent -- see METHODOLOGY NOTE 4. At hidden=(64,64) the
# GEMMs are too small and too badly shaped (K=60) for tensor cores to help, so the setting
# is inert there rather than shrinking the denominator. Kept for consistency across sizes.
torch.set_float32_matmul_precision("high")

OBS_DIM = 60
ACTION_DIM = 8
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
N_WARMUP = 10
N_ITERS = 30
SEED = 0

# "gather" is the per-minibatch advanced-indexing copy (flat_obs[idx] etc.)
# plus the reshape/detach bookkeeping; "perm" is the per-epoch
# torch.randperm(batch_size), which at batch_size=524288 is a real sort
# kernel run once per epoch. Both were untimed originally, which made
# sum-of-stages fall BELOW the true total (negative `resid` -- the fix was
# staged: adding "gather" alone closed only about half the gap, because
# randperm sits per-EPOCH, outside the per-minibatch loop). They are broken
# out rather than folded into "forward" because at long seq_len the gather
# materializes ~1GB per minibatch and becomes a story of its own.
#
# `resid` (sum-of-stages minus true total) is the correctness check on this
# accounting: it should be small and slightly POSITIVE (stages nest inside
# the total and each carries a little event overhead). A clearly negative
# resid means some real work is still untimed -- do not trust per-stage
# percentages until it is near zero.
STAGES = ("forward", "gather", "perm", "gae", "loss", "backward", "optimizer")


@dataclass(frozen=True)
class Config:
    """One measured point. Everything that used to be a module-level constant
    or a hardcoded loop bound in main() lives here, so all three axes (net
    size, sequence shape, GAE reuse) can vary independently."""
    num_envs: int
    seq_len: int
    hidden: tuple
    compiled: bool
    n_epochs: int = 4
    n_minibatches: int = 4
    recompute_gae_per_epoch: bool = False
    ladder: str = ""          # "budget" / "envs" / "" -- kept distinct in output
    n_iters: int = N_ITERS

    @property
    def label(self):
        base = (f"envs={self.num_envs} seq={self.seq_len} hidden={self.hidden} "
                f"{'compile' if self.compiled else 'eager'}")
        if self.n_epochs != 4 or self.n_minibatches != 4:
            base += f" {self.n_epochs}x{self.n_minibatches}"
        if self.recompute_gae_per_epoch:
            base += " recompute/epoch"
        if self.ladder:
            base += f" [{self.ladder}]"
        return base

    @property
    def gae_calls(self):
        return self.n_epochs if self.recompute_gae_per_epoch else 1


class ActorCritic(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(h2, ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))
        self.critic = nn.Linear(h2, 1)

    def forward(self, obs):
        z = self.trunk(obs)
        mean = self.actor_mean(z)
        value = self.critic(z).squeeze(-1)
        return mean, self.log_std, value


def _make_rollout(cfg, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    n, t = cfg.num_envs, cfg.seq_len
    obs = torch.randn(n, t, OBS_DIM, device=device, generator=g)
    actions = torch.randn(n, t, ACTION_DIM, device=device, generator=g)
    old_log_probs = torch.randn(n, t, device=device, generator=g) * 0.1 - 5.0
    rewards = torch.randn(n, t, device=device, generator=g)
    dones = (torch.rand(n, t, device=device, generator=g) < 0.02).float()
    old_values = torch.randn(n, t, device=device, generator=g)
    return obs, actions, old_log_probs, rewards, dones, old_values


def _gaussian_log_prob(mean, log_std, actions):
    std = log_std.exp()
    var = std * std
    return (-0.5 * ((actions - mean) ** 2) / var - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)


class _EventTimer:
    """Per-stage timing via CUDA events. Records stream markers during the
    step and resolves them to milliseconds only after ONE synchronize() at the
    end of the iteration -- so the measured region does not include a device
    drain per stage (see METHODOLOGY NOTE 2)."""

    def __init__(self):
        self._pairs = []   # (stage, start_event, end_event)
        self._open = None

    def start(self, stage):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._open = (stage, ev)

    def stop(self):
        stage, start_ev = self._open
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        self._pairs.append((stage, start_ev, end_ev))
        self._open = None

    def resolve(self):
        """Single sync for the whole iteration, then read every pair."""
        torch.cuda.synchronize()
        out = {s: 0.0 for s in STAGES}
        out["_total"] = 0.0
        for stage, a, b in self._pairs:
            out[stage] = out.get(stage, 0.0) + a.elapsed_time(b)
        self._pairs.clear()
        return out


class _SyncTimer:
    """The OLD method: synchronize() + perf_counter() around every stage.
    Retained ONLY so the methodology shift can be measured on the same card in
    the same process (--timing sync/both). Not the default."""

    def __init__(self):
        self._acc = {s: 0.0 for s in STAGES}
        self._acc["_total"] = 0.0
        self._open = None

    def start(self, stage):
        torch.cuda.synchronize()
        self._open = (stage, time.perf_counter())

    def stop(self):
        stage, t0 = self._open
        torch.cuda.synchronize()
        self._acc[stage] += (time.perf_counter() - t0) * 1000
        self._open = None

    def resolve(self):
        out = dict(self._acc)
        self._acc = {s: 0.0 for s in STAGES}
        self._acc["_total"] = 0.0
        return out


def _run_ppo_update(cfg, net, optimizer, rollout, gae_fn, timer, total_timer):
    """One full PPO update, timed with `timer` (event- or sync-based).

    `total_timer` wraps the ENTIRE update to give a true measured total, as
    opposed to the sum of stage timers (which carries instrumentation cost).
    Also returns the GAE device-only time so the two arms' unequal work
    (METHODOLOGY NOTE 3) is visible in the report.
    """
    obs, actions, old_log_probs, rewards, dones, old_values = rollout
    n, t = cfg.num_envs, cfg.seq_len

    gae_device_ms = 0.0
    total_timer.start("_total")

    timer.start("forward")
    with torch.no_grad():
        flat_obs = obs.reshape(-1, OBS_DIM)
        _, _, values_flat = net(flat_obs)
        values = values_flat.reshape(n, t)
    timer.stop()

    def _do_gae():
        """GAE call, timed both as part of the step and in isolation."""
        nonlocal gae_device_ms
        dev_a = torch.cuda.Event(enable_timing=True)
        dev_b = torch.cuda.Event(enable_timing=True)
        timer.start("gae")
        dev_a.record()
        adv = gae_fn(rewards, values, dones, gamma=GAMMA, lambda_=LAMBDA)
        ret = adv + values
        dev_b.record()
        timer.stop()
        return adv, ret, (dev_a, dev_b)

    advantages, returns, gae_events = _do_gae()
    pending_gae_events = [gae_events]

    timer.start("gather")
    flat_actions = actions.reshape(-1, ACTION_DIM)
    flat_old_log_probs = old_log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1).detach()
    flat_returns = returns.reshape(-1).detach()
    timer.stop()
    batch_size = n * t
    minibatch_size = batch_size // cfg.n_minibatches

    for epoch in range(cfg.n_epochs):
        # Axis C: recompute advantages every epoch instead of once per rollout.
        # This is a real PPO variant (values drift as the net updates), and it
        # is the third mechanism by which GAE's addressable share rises.
        if cfg.recompute_gae_per_epoch and epoch > 0:
            timer.start("forward")
            with torch.no_grad():
                _, _, v_flat = net(flat_obs)
                values = v_flat.reshape(n, t)
            timer.stop()
            advantages, returns, ev = _do_gae()
            pending_gae_events.append(ev)
            timer.start("gather")
            flat_advantages = advantages.reshape(-1).detach()
            flat_returns = returns.reshape(-1).detach()
            timer.stop()

        timer.start("perm")
        perm = torch.randperm(batch_size, device=obs.device)
        timer.stop()
        for mb in range(cfg.n_minibatches):
            timer.start("gather")
            idx = perm[mb * minibatch_size:(mb + 1) * minibatch_size]
            mb_obs, mb_actions = flat_obs[idx], flat_actions[idx]
            mb_old_lp, mb_adv, mb_ret = flat_old_log_probs[idx], flat_advantages[idx], flat_returns[idx]
            timer.stop()

            timer.start("forward")
            mean, log_std, value = net(mb_obs)
            timer.stop()

            timer.start("loss")
            log_prob = _gaussian_log_prob(mean, log_std, mb_actions)
            ratio = (log_prob - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
            loss = policy_loss + value_loss
            timer.stop()

            timer.start("backward")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            timer.stop()

            timer.start("optimizer")
            optimizer.step()
            timer.stop()

    total_timer.stop()
    totals = total_timer.resolve()
    stages = timer.resolve()
    torch.cuda.synchronize()
    for a, b in pending_gae_events:
        gae_device_ms += a.elapsed_time(b)
    stages["_total"] = totals["_total"]
    stages["_gae_device"] = gae_device_ms
    return stages


def _agg(samples):
    """median + IQR across iterations. The previous version accumulated a
    running sum and divided, discarding the distribution -- so 'within noise
    of 1x' had no noise estimate behind it."""
    s = sorted(samples)
    if not s:
        return 0.0, 0.0
    med = statistics.median(s)
    if len(s) < 4:
        return med, 0.0
    q1 = statistics.median(s[: len(s) // 2])
    q3 = statistics.median(s[(len(s) + 1) // 2:])
    return med, q3 - q1


def _loop_gae(rewards, values, terminateds, gamma, lambda_):
    """The sequential backward loop most RL libraries actually ship: vectorized
    over envs, but a Python-level `for t in reversed(range(T))` over the time
    axis, launching O(T) tiny kernels. This is `reference_gae` from the test
    suite, wrapped to match the call signature of the other arms.

    Included because the paper's headline table reports a Loop baseline
    (2.5-2.8x at 4096x128) but tab:ppo-e2e did not -- so the end-to-end table
    compared the kernel only against the STRONG compiled-scan baseline, which
    is not what most users run. Measuring it here says what GAE's share is for
    a library that never adopted a scan at all.
    """
    return reference_gae(rewards, values, terminateds, gamma, lambda_)


# name -> (callable factory, needs_compile). Factories are called once per
# measure() so torch.compile wrappers are per-config, not shared across shapes.
_ARMS = {
    "triton":     (lambda: compute_gae, False),
    "scan":       (lambda: torch.compile(vectorized_gae), False),
    "loop":       (lambda: _loop_gae, False),
    "loop_compiled": (lambda: torch.compile(_loop_gae), False),
}
DEFAULT_ARMS = ("triton", "scan", "loop", "loop_compiled")


def measure(cfg, timing="event", arms=DEFAULT_ARMS, device="cuda"):
    """Interleaved N-way A/B across GAE implementations.

    Each arm gets its OWN net and optimizer, all initialized identically, so
    the only difference between arms is the GAE call. Arm order is ROTATED
    every iteration (not merely swapped, which only works for two arms) so
    slow GPU clock/allocator drift lands on every arm equally instead of
    contaminating whichever ran last -- the generalization of the original
    two-arm interleaving fix.
    """
    torch.manual_seed(SEED)
    ref_net = ActorCritic(cfg.hidden).to(device)
    state = ref_net.state_dict()

    nets, opts, fns = {}, {}, {}
    for name in arms:
        net = ActorCritic(cfg.hidden).to(device)
        net.load_state_dict(state)  # identical init -- isolates the GAE-arm difference
        nets[name] = torch.compile(net) if cfg.compiled else net
        opts[name] = torch.optim.Adam(net.parameters(), lr=3e-4)
        fns[name] = _ARMS[name][0]()

    mk = _EventTimer if timing == "event" else _SyncTimer
    samples = {n: {k: [] for k in (*STAGES, "_total", "_gae_device")} for n in arms}

    def run(name, rollout, record):
        # The true-total wrapper is ALWAYS an _EventTimer, even under
        # --timing sync, so the total is measured the same way in both modes
        # and the two are directly comparable.
        #
        # Note what this means under --timing sync: the event pair spans the
        # whole update, so the true total there legitimately INCLUDES the
        # per-stage device drains. That is the point -- it is what the old
        # method's denominator actually was. The event-vs-sync difference in
        # `_total` is therefore the size of the methodology shift itself, with
        # hardware, seeds, and net held fixed.
        res = _run_ppo_update(cfg, nets[name], opts[name], rollout, fns[name],
                              mk(), _EventTimer())
        if record:
            for k, v in res.items():
                samples[name][k].append(v)

    for i in range(N_WARMUP):
        rollout = _make_rollout(cfg, seed=1000 + i, device=device)
        for name in arms:
            run(name, rollout, record=False)
        del rollout

    for i in range(cfg.n_iters):
        rollout = _make_rollout(cfg, seed=2000 + i, device=device)
        order = arms[i % len(arms):] + arms[:i % len(arms)]  # rotate
        for name in order:
            run(name, rollout, record=True)
        del rollout

    return samples


def _probe_launch_bound(cfg, device="cuda"):
    """Is the step compute-bound or launch-bound at this net size?

    Measures the net's forward+backward on one minibatch (a) normally and
    (b) with the same work submitted back-to-back without any host-side gap,
    then compares against a CUDA-graph replay of the same region. A step whose
    graph replay is much faster than eager submission is launch-bound: the GPU
    was idle waiting on the host, not busy computing. Reported as a ratio, so
    the small-net rows can be read as "share rose because GAE grew relative to
    real work" vs "share rose because the whole step became overhead".
    """
    torch.manual_seed(SEED)
    net = ActorCritic(cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    mb = (cfg.num_envs * cfg.seq_len) // cfg.n_minibatches
    obs = torch.randn(mb, OBS_DIM, device=device)
    tgt = torch.randn(mb, device=device)

    def one():
        opt.zero_grad(set_to_none=True)
        _, _, v = net(obs)
        ((v - tgt) ** 2).mean().backward()
        opt.step()

    for _ in range(10):
        one()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        one()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) * 1000 / 50

    graph_ms = float("nan")
    try:
        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        static = torch.cuda.graphs.graph_pool_handle()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, pool=static):
            one()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) * 1000 / 50
    except Exception as exc:  # capture can fail (Adam/dynamo interactions)
        print(f"    [launch-probe] graph capture unavailable: {type(exc).__name__}: {exc}")

    return eager_ms, graph_ms


# ---------------------------------------------------------------- grids

_HIDDENS = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
_BUDGET_LADDER = [(4096, 128), (1024, 512), (256, 2048), (128, 4096)]
_ENVS_LADDER = [(4096, 128), (4096, 512), (4096, 2048), (4096, 4096)]


def build_grid(which):
    """Axis A sweeps net size at the original (4096,128) shape. Axis B is the
    two seq_len ladders, held at hidden=(256,256) eager so the only thing
    moving is the sequence shape. Axis C contrasts GAE reuse."""
    axis_a = [Config(4096, 128, h, c) for h in _HIDDENS for c in (False, True)]
    axis_b = (
        [Config(n, t, (256, 256), False, ladder="budget") for n, t in _BUDGET_LADDER]
        + [Config(n, t, (256, 256), False, ladder="envs") for n, t in _ENVS_LADDER]
    )
    axis_c = [
        Config(4096, 128, (256, 256), False, n_epochs=4, n_minibatches=4),
        Config(4096, 128, (256, 256), False, n_epochs=4, n_minibatches=4,
               recompute_gae_per_epoch=True),
        Config(4096, 128, (256, 256), False, n_epochs=1, n_minibatches=1),
        Config(4096, 128, (64, 64), False, n_epochs=4, n_minibatches=4,
               recompute_gae_per_epoch=True),
    ]
    legacy = [Config(4096, 128, h, c) for h in ((256, 256), (1024, 1024)) for c in (False, True)]
    return {
        "axis-a": axis_a,
        "axis-b": axis_b,
        "axis-c": axis_c,
        "legacy": legacy,
        "all": axis_a + axis_b + axis_c,
    }[which]


def _row(cfg, samples, arms, ref="triton"):
    """Collapse one config's samples into per-arm numbers.

    Speedup is reported as (this arm's total) / (ref arm's total) computed
    PER ITERATION and then aggregated, so the IQR is a real noise band on the
    ratio rather than a ratio of two independently-noisy medians.
    """
    out = {}
    for name in arms:
        s = samples[name]
        tot, tot_iqr = _agg(s["_total"])
        gae, _ = _agg(s["gae"])
        dev, _ = _agg(s["_gae_device"])
        stage_sum = sum(_agg(s[st])[0] for st in STAGES)
        # this_arm / ref: >1 means the ref arm (Triton) finished the step faster.
        ratios = [t / r for t, r in zip(s["_total"], samples[ref]["_total"]) if r > 0]
        sp, sp_iqr = _agg(ratios)
        out[name] = {
            "total": tot, "total_iqr": tot_iqr,
            "sum": stage_sum, "resid": stage_sum - tot,
            "gae": gae, "dev": dev,
            "share": gae / tot * 100 if tot else 0.0,
            "speedup": sp, "speedup_iqr": sp_iqr,
            "stages": {st: _agg(s[st])[0] for st in STAGES},
        }

    # NON-GAE contamination check. A GAE-only change cannot alter forward,
    # gather, perm, backward or optimizer cost -- those stages run after GAE
    # has finished and share no state with it. If they differ between arms,
    # the arm totals are not comparable and any end-to-end ratio built from
    # them is an artifact, not a result. This exists because the four-arm run
    # showed loop_compiled finishing the whole step ~2.5ms FASTER than the
    # Triton arm at (256,256) -- 100x more than GAE's entire 0.019ms budget,
    # so it cannot have come from the GAE call.
    # `nongae_vs_ref` is (this arm's non-GAE time) / (ref's non-GAE time);
    # anything more than ~1% from 1.0 means the comparison is contaminated.
    ref_nongae = sum(out[ref]["stages"][st] for st in STAGES if st != "gae")
    for name in arms:
        nongae = sum(out[name]["stages"][st] for st in STAGES if st != "gae")
        out[name]["nongae"] = nongae
        out[name]["nongae_vs_ref"] = nongae / ref_nongae if ref_nongae else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="all",
                        choices=["all", "axis-a", "axis-b", "axis-c", "legacy"])
    parser.add_argument("--timing", default="event", choices=["event", "sync", "both"],
                        help="'both' runs each config under old (sync) and new (event) "
                             "timing on the same card, to size the methodology shift.")
    parser.add_argument("--probe-launch-bound", action="store_true",
                        help="measure whether small nets are launch-bound rather than "
                             "compute-bound (see METHODOLOGY NOTE 4).")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help=f"comma-separated GAE arms from {sorted(_ARMS)}. "
                             "'triton' must be present -- it is the speedup reference. "
                             "Drop 'loop' at long seq_len if runtime becomes a problem.")
    args = parser.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in arms if a not in _ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; choose from {sorted(_ARMS)}")
    if "triton" not in arms:
        parser.error("'triton' must be among --arms (it is the speedup reference)")

    if not torch.cuda.is_available():
        print("CUDA not available -- skipping.")
        return

    gpu = torch.cuda.get_device_name(0)
    grid = build_grid(args.grid)
    if args.iters:
        grid = [Config(**{**c.__dict__, "n_iters": args.iters}) for c in grid]
    timings = ["sync", "event"] if args.timing == "both" else [args.timing]

    print(f"GPU: {gpu}  torch: {torch.__version__}")
    print(f"grid={args.grid} ({len(grid)} configs)  timing={args.timing}")
    print("GAE share and speedup are computed against the TRUE measured total "
          "(one event pair around the whole step), not the sum of stage timers.\n")

    report = [
        f"# PPO end-to-end measurement -- {datetime.date.today().isoformat()}",
        "",
        "ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table.",
        f"GPU: {gpu} · torch {torch.__version__}",
        f"Grid `{args.grid}` ({len(grid)} configs), timing mode `{args.timing}`, "
        f"arms `{','.join(arms)}`, interleaved with rotating arm order, "
        f"median (IQR) across iterations.",
        "",
        "GAE share and end-to-end speedup are computed against the **true measured total** "
        "(a single CUDA-event pair around the entire update), not the sum of per-stage "
        "timers. `resid` = sum-of-stages minus true total; it should be small and slightly "
        "positive. A clearly negative resid means real work is still untimed and per-stage "
        "percentages should not be trusted.",
        "",
        "The arms do not do identical work: the Triton arm takes its no-truncation path, "
        "while `scan` allocates two extra [num_envs, seq_len] tensors and runs the "
        "truncation-aware scan regardless. `GAE dev` gives GAE device time alone so this "
        "asymmetry is visible rather than argued.",
        "",
    ]

    results = []
    for cfg in grid:
        print(f"=== {cfg.label} ===")
        per_timing = {}
        for tm in timings:
            try:
                samples = measure(cfg, timing=tm, arms=arms)
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at {cfg.label} (timing={tm}) -- skipped, shape NOT rescaled.")
                torch.cuda.empty_cache()
                per_timing[tm] = None
                continue
            r = _row(cfg, samples, arms)
            per_timing[tm] = r
            for name in arms:
                a = r[name]
                flag = "" if abs(a["nongae_vs_ref"] - 1.0) < 0.01 else "  <-- CONTAMINATED"
                print(f"  [{tm:5s}] {name:<14} total {a['total']:>9.3f}ms "
                      f"(IQR {a['total_iqr']:.3f})  GAE {a['gae']:.4f}ms "
                      f"(dev {a['dev']:.4f})  share {a['share']:.3f}%  "
                      f"vs-triton {a['speedup']:.4f}x (IQR {a['speedup_iqr']:.4f})  "
                      f"resid {a['resid']:+.3f}")
                print(f"           {'':<14} non-GAE {a['nongae']:>7.3f}ms "
                      f"({a['nongae_vs_ref']:.4f}x ref)  "
                      + " ".join(f"{st}={a['stages'][st]:.2f}" for st in STAGES)
                      + flag)
        results.append((cfg, per_timing))
        if args.probe_launch_bound:
            e, g = _probe_launch_bound(cfg)
            ratio = e / g if g == g and g > 0 else float("nan")
            print(f"  [launch] eager {e:.4f}ms vs graph-replay {g:.4f}ms -> {ratio:.2f}x "
                  f"({'launch-bound' if ratio > 1.3 else 'compute-bound'})")
        print()

    report += [
        "| config | ladder | timing | arm | total | GAE | GAE dev | share | "
        "vs-triton | resid |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg, per_timing in results:
        for tm, r in per_timing.items():
            if r is None:
                report.append(f"| {cfg.label} | {cfg.ladder or '-'} | {tm} | "
                              "OOM | - | - | - | - | - | - |")
                continue
            for name in arms:
                a = r[name]
                report.append(
                    f"| {cfg.label} | {cfg.ladder or '-'} | {tm} | {name} | "
                    f"{a['total']:.3f} ({a['total_iqr']:.3f}) | "
                    f"{a['gae']:.4f} | {a['dev']:.4f} | {a['share']:.3f}% | "
                    f"{a['speedup']:.4f} ({a['speedup_iqr']:.4f}) | "
                    f"{a['resid']:+.3f} |"
                )

    report += [
        "",
        "## Per-stage breakdown (contamination check)",
        "",
        "A GAE-only change cannot alter forward, gather, perm, backward or optimizer "
        "cost -- those stages run after GAE has finished and share no state with it. "
        "`non-GAE vs ref` is this arm's total non-GAE stage time divided by the Triton "
        "arm's. Anything more than ~1% from 1.000 means the arms' step totals are not "
        "comparable and any end-to-end ratio built from them is an artifact.",
        "",
        "| config | timing | arm | " + " | ".join(STAGES) + " | non-GAE | non-GAE vs ref |",
        "|---|---|---|" + "---|" * (len(STAGES) + 2),
    ]
    for cfg, per_timing in results:
        for tm, r in per_timing.items():
            if r is None:
                continue
            for name in arms:
                a = r[name]
                cells = " | ".join(f"{a['stages'][st]:.3f}" for st in STAGES)
                mark = "" if abs(a["nongae_vs_ref"] - 1.0) < 0.01 else " **!**"
                report.append(
                    f"| {cfg.label} | {tm} | {name} | {cells} | "
                    f"{a['nongae']:.3f} | {a['nongae_vs_ref']:.4f}{mark} |"
                )

    report += [
        "",
        "All times in ms, median with IQR in parentheses. `vs-triton` is the median of "
        "per-iteration (this arm's total / Triton arm's total); >1 means the Triton arm "
        "finished the step faster. Its IQR is the noise band -- a ratio whose IQR spans "
        "1.0 is not distinguishable from no difference.",
        "",
        "Arms: `triton` = rl-triton kernel; `scan` = torch.compile'd vectorized "
        "doubling scan (the strong baseline); `loop` = the sequential backward Python "
        "loop most RL libraries ship (vectorized over envs, O(T) kernel launches); "
        "`loop_compiled` = the same loop under torch.compile.",
        "",
    ]

    out_dir = Path(__file__).parent.parent / "docs" / "benchmark-history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ppo-e2e-curve-{args.grid}-{datetime.date.today().isoformat()}.md"
    out_path.write_text("\n".join(report))
    print(f"Wrote {out_path} (reference only -- not part of benchmarks.md/README's recurring tables).")


if __name__ == "__main__":
    main()
