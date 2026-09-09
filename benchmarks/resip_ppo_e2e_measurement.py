"""ONE-OFF measurement: learner-side PPO update cost in a ResiP-INSPIRED
long-horizon, massively-parallel residual-RL regime.

Purpose: the paper's existing `ppo_e2e_measurement.py` sweeps net size and
seq_len around a generic Isaac-Gym-Ant-like policy. It never asks the
question a workshop reviewer working on residual/specialist robot policies
will ask first: "what about MY regime -- long horizon (700-1000 steps),
1024 parallel envs, a genuinely compact actor, PPO with many update epochs
per rollout?" This script answers exactly that, using a configuration
lifted from the published ResiP setting (Ankile et al., "From Imitation to
Refinement -- Residual RL for Precise Assembly", arXiv:2407.16677):

    num_envs        = 1024
    seq_len (T)     = 700, 1000          (published task horizon limits)
    gamma           = 0.999
    lambda_ (GAE)   = 0.95
    n_minibatches   = 1                  (published: one minibatch)
    n_epochs        = 1, 4, 10, 50       (50 = published ResiP setting)
    action_dim      = 10
    obs_dim (proprio) = 16
    critic          = 2 hidden layers, hidden size 256, ReLU
    actor           = 2 hidden layers, hidden size 256, ReLU, Gaussian head
                       (see ResidualActorCritic below)
    reward          = sparse (Bernoulli, ~1 nonzero step per episode)

IMPORTANT SCOPING, read before citing any number from this script:

  * This is a "ResiP-INSPIRED" workload, NOT a reproduction. ResiP trains
    inside Isaac Gym with a frozen chunked diffusion/BC base policy and a
    residual policy on top; this script does not run Isaac Gym and does not
    load or execute any ResiP code. The residual actor/critic ARCHITECTURE
    (input width, hidden sizes, layer count, activation) and the
    BASE-POLICY FORWARD-PASS SCOPING below were all verified directly
    against the published source (github.com/ankile/robust-rearrangement),
    not assumed:
      - src/models/residual.py's `ResidualPolicy`: input is
        `nobs = concat([state, base_action])`
        (`obs_dim = prod(obs_shape) + prod(action_shape)`).
      - src/config/actor/residual_mlp.yaml and residual_diffusion.yaml (both
        base-policy variants agree): `actor_hidden_size: 256`,
        `actor_num_layers: 2`, `actor_activation: ReLU`,
        `critic_hidden_size: 256`, `critic_num_layers: 2`,
        `critic_activation: ReLU`, `action_scale: 0.1`. This script matches
        all of these exactly rather than guessing at "compact."
      - src/config/experiment/rl/residual_mlp_ppo.yaml (via
        base_residual_rl.yaml) confirms num_envs=1024, update_epochs=50,
        num_env_steps=700, num_minibatches=1, discount=0.999,
        gae_lambda=0.95 for the published one_leg/700-step setting.
      - `init_logstd: -1.5` for the residual_mlp base policy variant (the
        residual_diffusion variant instead uses -1.0); this script uses
        -1.5, matching the residual_mlp_ppo.yaml experiment config it
        otherwise follows for episode length and update_epochs.
    What is NOT reproduced: ResiP's `layer_init` uses Kaiming-normal
    initialization for ReLU layers with a small-std output head
    (`action_head_std: 0.0`) and a biased/scaled critic output head
    (`critic_last_layer_bias_const: 0.25`, `critic_last_layer_std: 0.25`);
    this script uses PyTorch's default `nn.Linear` initialization
    throughout, since initialization does not affect steady-state forward/
    backward timing (the only thing measured here) and reproducing training
    dynamics is explicitly out of scope.
  * BASE-POLICY SCOPING (verified, not assumed): in ResiP's actual source,
    `agent.base_action_normalized(next_obs)` is called ONCE per rollout step
    during collection, concatenated into `next_residual_nobs =
    cat([next_nobs, base_naction])`, and CACHED into the stored rollout
    buffer (residual_ppo.py: `obs[step] = next_residual_nobs`). The
    `update_epochs` loop then reads `mb_obs` straight from that buffer and
    never calls the base policy again -- so the base policy's forward pass
    is correctly OUT OF SCOPE for a learner-side (post-rollout) PPO-update
    benchmark: it is a one-time-per-rollout-step cost paid during rollout
    collection, not a per-epoch cost inside the timed update. This script
    therefore does NOT execute a base policy, but DOES model its structural
    footprint: `ResidualActorCritic.forward(obs, base_action)` concatenates
    a same-shaped stand-in `base_action` tensor (itself generated once and
    cached alongside `obs`, exactly like ResiP's cached `base_naction`
    column) before the actor/critic trunks, so the widened first linear
    layer matches `ResidualPolicy.obs_dim = prod(obs_shape) +
    prod(action_shape)` from the real implementation. This adds zero extra
    forward passes to the timed epoch loop -- only wider input layers.
  * Rollout tensors (obs, base_action, actions, rewards, dones, values) are
    SYNTHETIC (i.i.d. random, matching the paper's existing PPO benchmark
    methodology). No simulator, no real robot, no real assembly task.
  * This measures ONLY the learner-side PPO update step: GAE + n_epochs of
    (forward, PPO loss, backward, Adam). It excludes simulator stepping and
    rollout collection (including the base policy's one-time-per-step
    forward pass during that collection, per the verified scoping above).
    Do NOT interpret any number here as a total-training-time or
    wall-clock-to-convergence speedup for ResiP or for robot residual RL in
    general. Call this a "learner-side PPO update benchmark", not an
    end-to-end training result.

Methodology reused from ppo_e2e_measurement.py in this directory (see that
file's docstring for the full derivation of each choice):
  - CUDA-event timing with a single synchronize() per iteration, not
    sync+perf_counter per stage (avoids ~24x GAE-time inflation from device
    drains around a microsecond-scale kernel).
  - N-way interleaved A/B/C measurement in one process with rotating arm
    order, so slow clock/allocator drift lands on every arm equally.
  - Every arm's advantages are checked against the Triton kernel on
    identical inputs before any timing is trusted (correctness gate).
  - A non-GAE-stage contamination check: forward/loss/backward/optimizer
    cost cannot legitimately differ between arms, since those stages run
    strictly after GAE has produced advantages/returns. If it does, the
    comparison is flagged and excluded from citable rows.
  - Median + IQR across iterations (not a bare mean), because "the gain is
    within noise" needs an actual noise estimate.

Usage:
    python benchmarks/resip_ppo_e2e_measurement.py                 # full sweep (T x epochs)
    python benchmarks/resip_ppo_e2e_measurement.py --seq-lens 700  # one horizon only
    python benchmarks/resip_ppo_e2e_measurement.py --epochs 1,50   # boundary points only
    python benchmarks/resip_ppo_e2e_measurement.py --arms triton,scan,loop,pufferlib
                                                                    # include PufferLib CUDA arm
    python benchmarks/resip_ppo_e2e_measurement.py --iters 15      # shorter run
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("TORCH_LOGS", "-dynamic")

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_gae import vectorized_gae, reference_gae
from rl_triton.ops.gae import compute_gae

torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------- ResiP-inspired config
NUM_ENVS = 1024
OBS_DIM = 16          # proprioceptive state dimension (published ResiP setting)
ACTION_DIM = 10       # published ResiP setting
GAMMA = 0.999         # published ResiP setting
LAMBDA = 0.95         # published GAE lambda
CLIP_EPS = 0.2
CRITIC_HIDDEN = (256, 256)   # "critic: 2 hidden layers, hidden size 256"
ACTOR_HIDDEN = (256, 256)    # compact residual actor -- see module docstring
SPARSE_REWARD_P = 1.0 / 350  # ~1 nonzero reward per 350 steps -> ~2-3 nonzero
                              # rewards across a 700-1000 step episode, matching
                              # "one reward for one_leg, two for lamp/round_table"
N_WARMUP = 8
N_ITERS = 20
SEED = 0

STAGES = ("forward", "gather", "gae", "loss", "backward", "optimizer")


@dataclass(frozen=True)
class Config:
    seq_len: int
    n_epochs: int
    n_minibatches: int = 1        # published ResiP setting: one minibatch
    n_iters: int = N_ITERS
    n_warmup: int = N_WARMUP

    @property
    def label(self):
        return f"T={self.seq_len} epochs={self.n_epochs} mb={self.n_minibatches}"


class ResidualActorCritic(nn.Module):
    """Residual Gaussian actor + separate critic, matching ResiP's verified
    architecture: 2 hidden layers of width 256, ReLU, for both actor and
    critic (src/config/actor/residual_mlp.yaml and residual_diffusion.yaml
    -- see module docstring for the full verified-vs-not-reproduced list).

    ResiP's residual policy outputs a small corrective action added to the
    frozen base (BC/diffusion) policy's proposal. Verified against the
    published source (ankile/robust-rearrangement, src/models/residual.py and
    src/train/residual_ppo.py): the actor and critic both take
    `nobs = concat([state, base_action])` as input
    (`ResidualPolicy.obs_dim = prod(obs_shape) + prod(action_shape)`), and
    `base_naction` is computed ONCE per rollout step during collection
    (`agent.base_action_normalized(next_obs)`) and cached into the stored
    observation buffer (`obs[step] = next_residual_nobs`) -- the
    `update_epochs` loop (residual_ppo.py) reads `mb_obs` straight from that
    buffer and never calls the base policy again. So the base policy's
    forward pass is correctly OUT of scope for this learner-side benchmark
    (it is a one-time-per-rollout-step cost, not a per-epoch one), but the
    actor/critic input width must include the concatenated base action to
    match the real architecture. `base_action` below is therefore a
    same-shaped stand-in tensor (also cached once, exactly like ResiP's
    `base_naction`) concatenated inside `forward`, adding zero extra forward
    passes to the timed epoch loop -- only a wider first linear layer.
    """

    def __init__(self, actor_hidden=ACTOR_HIDDEN, critic_hidden=CRITIC_HIDDEN):
        super().__init__()
        nobs_dim = OBS_DIM + ACTION_DIM  # state + base_action, per ResiP's nobs
        a1, a2 = actor_hidden
        # ReLU, not Tanh: verified against src/config/actor/residual_mlp.yaml
        # and residual_diffusion.yaml (both set actor_activation: ReLU,
        # critic_activation: ReLU).
        self.actor_trunk = nn.Sequential(
            nn.Linear(nobs_dim, a1), nn.ReLU(),
            nn.Linear(a1, a2), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(a2, ACTION_DIM)
        # init_logstd=-1.5 (residual_mlp.yaml's value for the one_leg/700-step
        # published setting this script targets; residual_diffusion.yaml uses
        # -1.0 for the diffusion base policy instead).
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM) - 1.5)

        c1, c2 = critic_hidden
        self.critic_trunk = nn.Sequential(
            nn.Linear(nobs_dim, c1), nn.ReLU(),
            nn.Linear(c1, c2), nn.ReLU(),
        )
        self.critic_head = nn.Linear(c2, 1)

    def forward(self, obs, base_action):
        nobs = torch.cat([obs, base_action], dim=-1)
        za = self.actor_trunk(nobs)
        mean = self.actor_mean(za)
        zc = self.critic_trunk(nobs)
        value = self.critic_head(zc).squeeze(-1)
        return mean, self.log_std, value


def _make_rollout(cfg, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    n, t = NUM_ENVS, cfg.seq_len
    obs = torch.randn(n, t, OBS_DIM, device=device, generator=g)
    # Stand-in for ResiP's `base_naction`: computed once per rollout step by
    # the frozen base policy and cached into the observation buffer (see
    # ResidualActorCritic's docstring). Generated once here, alongside obs,
    # and never recomputed -- exactly like the real cached buffer column.
    base_actions = torch.randn(n, t, ACTION_DIM, device=device, generator=g) * 0.1
    actions = torch.randn(n, t, ACTION_DIM, device=device, generator=g) * 0.1
    old_log_probs = torch.randn(n, t, device=device, generator=g) * 0.1 - 8.0
    # sparse task reward: mostly zero, occasional +1 (assembly-subtask-style)
    rewards = (torch.rand(n, t, device=device, generator=g) < SPARSE_REWARD_P).float()
    # long-horizon episodic tasks: terminate only at (or rarely before) the
    # horizon limit, matching "episode limits of 700/1000 steps" rather than
    # the short-episode churn used in the paper's other synthetic benchmark.
    dones = torch.zeros(n, t, device=device)
    dones[:, -1] = 1.0
    early_term = torch.rand(n, t, device=device, generator=g) < (1.0 / (t * 20))
    dones = torch.clamp(dones + early_term.float(), max=1.0)
    old_values = torch.randn(n, t, device=device, generator=g) * 0.1
    return obs, base_actions, actions, old_log_probs, rewards, dones, old_values


def _gaussian_log_prob(mean, log_std, actions):
    std = log_std.exp()
    var = std * std
    return (-0.5 * ((actions - mean) ** 2) / var - log_std
            - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=mean.device))).sum(-1)


class _EventTimer:
    """Per-stage timing via pooled CUDA events, one synchronize() per iteration.
    See ppo_e2e_measurement.py METHODOLOGY NOTES 2 and 5 for the full derivation."""

    def __init__(self):
        self._pairs = []
        self._open = None
        self._pool = []
        self._pool_idx = 0

    def _get_event(self):
        if self._pool_idx < len(self._pool):
            ev = self._pool[self._pool_idx]
        else:
            ev = torch.cuda.Event(enable_timing=True)
            self._pool.append(ev)
        self._pool_idx += 1
        return ev

    def start(self, stage):
        ev = self._get_event()
        ev.record()
        self._open = (stage, ev)

    def stop(self):
        stage, start_ev = self._open
        end_ev = self._get_event()
        end_ev.record()
        self._pairs.append((stage, start_ev, end_ev))
        self._open = None

    def resolve(self):
        torch.cuda.synchronize()
        out = {s: 0.0 for s in STAGES}
        out["_total"] = 0.0
        for stage, a, b in self._pairs:
            out[stage] = out.get(stage, 0.0) + a.elapsed_time(b)
        self._pairs.clear()
        self._pool_idx = 0
        return out


def _loop_gae(rewards, values, terminateds, gamma, lambda_):
    """Sequential O(T) backward loop, vectorized over envs -- what CleanRL/
    RLlib/Sample Factory ship. See ppo_e2e_measurement.py for provenance."""
    return reference_gae(rewards, values, terminateds, gamma, lambda_)


def _make_pufferlib_arm():
    """Optional PufferLib CUDA GAE arm, only constructed if requested and if
    the vendored extension in benchmarks/pufferlib_ext builds successfully.
    Not part of DEFAULT_ARMS: this is the 'if easy and semantically
    compatible' optional comparison from the experiment brief, and the
    vendored build has real environment dependencies (nvcc, matching torch
    ABI) that make it reasonable to skip by default.
    """
    ext_dir = Path(__file__).parent / "pufferlib_ext"
    sys.path.insert(0, str(ext_dir))
    from build import get_puffer_extension  # vendored, SHA256-pinned source

    ext = get_puffer_extension()

    def puffer_gae(rewards, values, terminateds, gamma, lambda_):
        # PufferLib's index convention: PR[1:]=R[:-1], PD[1:]=D[:-1] (see
        # benchmark_gae_vs_pufferlib.py's STEP 0 for the full derivation).
        # Row T-1 is structurally unwritten by puff_advantage_row_cuda; we
        # zero-pad it the same way a real PufferLib caller's output buffer
        # would (torch.zeros(...) before the kernel call).
        n, t = rewards.shape
        pr = torch.zeros_like(rewards)
        pd = torch.zeros_like(terminateds)
        pr[:, 1:] = rewards[:, :-1]
        pd[:, 1:] = terminateds[:, :-1]
        adv = torch.zeros_like(rewards)
        ext.compute_puff_advantage(pr, pd, values, adv, gamma, lambda_, 1.0, 1.0)
        return adv

    return puffer_gae


_ARMS = {
    "triton": (lambda: compute_gae, False),
    "scan": (lambda: torch.compile(vectorized_gae), False),
    "loop": (lambda: _loop_gae, False),
}
DEFAULT_ARMS = ("triton", "scan", "loop")


def _agg(samples):
    s = sorted(samples)
    if not s:
        return 0.0, 0.0
    med = statistics.median(s)
    if len(s) < 4:
        return med, 0.0
    q1 = statistics.median(s[: len(s) // 2])
    q3 = statistics.median(s[(len(s) + 1) // 2:])
    return med, q3 - q1


def correctness_gate(cfg, arms, device="cuda"):
    torch.manual_seed(SEED)
    rollout = _make_rollout(cfg, seed=42, device=device)
    _, _, _, _, rewards, dones, old_values = rollout

    ref_fn = _ARMS["triton"][0]() if "triton" in _ARMS else compute_gae
    ref_adv = compute_gae(rewards, old_values, dones, gamma=GAMMA, lambda_=LAMBDA)

    print(f"  [gate] correctness vs triton, atol=1e-4 rtol=1e-4:")
    results = {}
    for name in arms:
        if name == "triton":
            continue
        fn = _ARMS[name][0]()
        adv = fn(rewards, old_values, dones, gamma=GAMMA, lambda_=LAMBDA)
        diff = (adv - ref_adv).abs()
        max_abs = diff.max().item()
        denom = ref_adv.abs().clamp_min(1e-12)
        max_rel = (diff / denom).max().item()
        ok = torch.allclose(adv, ref_adv, atol=1e-4, rtol=1e-4)
        results[name] = (ok, max_abs, max_rel)
        status = "PASS" if ok else "FAIL"
        print(f"         {name:<12} {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
        if not ok:
            raise AssertionError(
                f"correctness gate failed for arm '{name}' at {cfg.label}: "
                f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
            )
    return results


def _run_ppo_update(cfg, net, optimizer, rollout, gae_fn, timer, total_timer):
    obs, base_actions, actions, old_log_probs, rewards, dones, old_values = rollout
    n, t = NUM_ENVS, cfg.seq_len

    gae_device_ms = 0.0
    total_timer.start("_total")

    timer.start("forward")
    with torch.no_grad():
        flat_obs = obs.reshape(-1, OBS_DIM)
        flat_base_actions = base_actions.reshape(-1, ACTION_DIM)
        _, _, values_flat = net(flat_obs, flat_base_actions)
        values = values_flat.reshape(n, t)
    timer.stop()

    dev_a = torch.cuda.Event(enable_timing=True)
    dev_b = torch.cuda.Event(enable_timing=True)
    timer.start("gae")
    dev_a.record()
    advantages = gae_fn(rewards, values, dones, gamma=GAMMA, lambda_=LAMBDA)
    returns = advantages + values
    dev_b.record()
    timer.stop()

    timer.start("gather")
    flat_actions = actions.reshape(-1, ACTION_DIM)
    flat_old_log_probs = old_log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1).detach()
    flat_returns = returns.reshape(-1).detach()
    timer.stop()

    batch_size = n * t
    minibatch_size = batch_size // cfg.n_minibatches

    for epoch in range(cfg.n_epochs):
        if cfg.n_minibatches == 1:
            idx = slice(None)
            mb_obs, mb_base_actions, mb_actions = flat_obs, flat_base_actions, flat_actions
            mb_old_lp, mb_adv, mb_ret = flat_old_log_probs, flat_advantages, flat_returns
        else:
            timer.start("gather")
            perm = torch.randperm(batch_size, device=obs.device)
            idx = perm[:minibatch_size]
            mb_obs, mb_base_actions, mb_actions = (
                flat_obs[idx], flat_base_actions[idx], flat_actions[idx])
            mb_old_lp, mb_adv, mb_ret = flat_old_log_probs[idx], flat_advantages[idx], flat_returns[idx]
            timer.stop()

        timer.start("forward")
        mean, log_std, value = net(mb_obs, mb_base_actions)
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
    gae_device_ms = dev_a.elapsed_time(dev_b)
    stages["_total"] = totals["_total"]
    stages["_gae_device"] = gae_device_ms
    return stages


def measure(cfg, arms, device="cuda"):
    torch.manual_seed(SEED)
    ref_net = ResidualActorCritic().to(device)
    state = ref_net.state_dict()

    nets, opts, fns = {}, {}, {}
    for name in arms:
        net = ResidualActorCritic().to(device)
        net.load_state_dict(state)
        nets[name] = net
        opts[name] = torch.optim.Adam(net.parameters(), lr=3e-4)
        fns[name] = _ARMS[name][0]()

    samples = {n: {k: [] for k in (*STAGES, "_total", "_gae_device")} for n in arms}
    stage_timers = {name: _EventTimer() for name in arms}
    total_timers = {name: _EventTimer() for name in arms}

    def run(name, rollout, record):
        res = _run_ppo_update(cfg, nets[name], opts[name], rollout, fns[name],
                               stage_timers[name], total_timers[name])
        if record:
            for k, v in res.items():
                samples[name][k].append(v)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for i in range(cfg.n_warmup):
        rollout = _make_rollout(cfg, seed=1000 + i, device=device)
        for name in arms:
            run(name, rollout, record=False)
        del rollout

    for i in range(cfg.n_iters):
        rollout = _make_rollout(cfg, seed=2000 + i, device=device)
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        for name in order:
            run(name, rollout, record=True)
        del rollout

    return samples


def _row(samples, arms, ref="triton"):
    n_iters = len(samples[ref]["_total"])
    out = {}
    for name in arms:
        s = samples[name]
        tot, tot_iqr = _agg(s["_total"])
        gae, _ = _agg(s["gae"])
        dev, _ = _agg(s["_gae_device"])
        per_iter_resid = [sum(s[st][i] for st in STAGES) - s["_total"][i]
                           for i in range(n_iters)]
        resid, resid_iqr = _agg(per_iter_resid)
        ratios = [t / r for t, r in zip(s["_total"], samples[ref]["_total"]) if r > 0]
        sp, sp_iqr = _agg(ratios)
        out[name] = {
            "total": tot, "total_iqr": tot_iqr,
            "resid": resid, "resid_iqr": resid_iqr,
            "gae": gae, "dev": dev,
            "share": gae / tot * 100 if tot else 0.0,
            "speedup_vs_triton": sp, "speedup_vs_triton_iqr": sp_iqr,
            "stages": {st: _agg(s[st])[0] for st in STAGES},
        }

    ref_s = samples[ref]
    for name in arms:
        s = samples[name]
        per_iter_nongae = [sum(s[st][i] for st in STAGES if st != "gae")
                            for i in range(n_iters)]
        per_iter_ref_nongae = [sum(ref_s[st][i] for st in STAGES if st != "gae")
                                for i in range(n_iters)]
        nongae, _ = _agg(per_iter_nongae)
        nongae_ratios = [a / b for a, b in zip(per_iter_nongae, per_iter_ref_nongae) if b > 0]
        nongae_vs_ref, _ = _agg(nongae_ratios)
        out[name]["nongae"] = nongae
        out[name]["nongae_vs_ref"] = nongae_vs_ref
    return out


_CONTAMINATION_THRESHOLD = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="700,1000")
    parser.add_argument("--epochs", default="1,4,10,50")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--iters", type=int, default=None)
    args = parser.parse_args()

    arms = list(a.strip() for a in args.arms.split(",") if a.strip())
    if "pufferlib" in arms:
        try:
            _ARMS["pufferlib"] = (_make_pufferlib_arm, False)
        except Exception as exc:
            print(f"PufferLib arm unavailable ({type(exc).__name__}: {exc}) -- dropping.")
            arms.remove("pufferlib")
    unknown = [a for a in arms if a not in _ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}")
    if "triton" not in arms:
        parser.error("'triton' must be among --arms (speedup reference)")

    if not torch.cuda.is_available():
        print("CUDA not available -- this script requires a GPU (RunPod H100/A100 "
              "as specified in the experiment design). Skipping.")
        return

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    epochs_list = [int(x) for x in args.epochs.split(",")]
    grid = [Config(seq_len=t, n_epochs=e,
                   n_iters=args.iters or N_ITERS)
            for t in seq_lens for e in epochs_list]

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch: {torch.__version__}")
    print(f"ResiP-inspired learner-side PPO benchmark. num_envs={NUM_ENVS}, "
          f"gamma={GAMMA}, lambda={LAMBDA}, action_dim={ACTION_DIM}, obs_dim={OBS_DIM}, "
          f"n_minibatches=1, arms={arms}")
    print("This is a LEARNER-SIDE benchmark: no simulator, no base-policy inference, "
          "synthetic rollout tensors. See module docstring for full scoping.\n")

    raw_results = []
    gate_results = {}
    for cfg in grid:
        print(f"=== {cfg.label} ===")
        gate_results[cfg.label] = correctness_gate(cfg, arms)
        try:
            samples = measure(cfg, arms)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at {cfg.label} -- skipped.")
            torch.cuda.empty_cache()
            continue
        r = _row(samples, arms)
        for name in arms:
            a = r[name]
            flag = "" if abs(a["nongae_vs_ref"] - 1.0) < _CONTAMINATION_THRESHOLD else "  <-- CONTAMINATED"
            print(f"  {name:<10} total {a['total']:>10.3f}ms (IQR {a['total_iqr']:.3f})  "
                  f"GAE {a['gae']:.4f}ms (dev {a['dev']:.4f})  share {a['share']:.4f}%  "
                  f"vs-triton {a['speedup_vs_triton']:.4f}x (IQR {a['speedup_vs_triton_iqr']:.4f})  "
                  f"resid {a['resid']:+.3f}{flag}")
        raw_results.append({
            "config": asdict(cfg),
            "gpu": gpu,
            "torch_version": torch.__version__,
            "gate": {k: {"pass": v[0], "max_abs": v[1], "max_rel": v[2]}
                     for k, v in gate_results[cfg.label].items()},
            "rows": r,
        })
        print()

    ts = datetime.date.today().isoformat()
    out_dir = Path(__file__).parent
    json_path = out_dir / f"resip_ppo_e2e-{ts}.json"
    json_path.write_text(json.dumps(raw_results, indent=2))
    print(f"Wrote {json_path}")

    csv_path = out_dir / f"resip_ppo_e2e-{ts}.csv"
    with open(csv_path, "w") as f:
        f.write("seq_len,n_epochs,arm,total_ms,total_iqr_ms,gae_ms,gae_device_ms,"
                "gae_share_pct,speedup_vs_triton,speedup_vs_triton_iqr,"
                "nongae_vs_ref,resid_ms,citable\n")
        for entry in raw_results:
            cfg = entry["config"]
            for name, a in entry["rows"].items():
                citable = abs(a["nongae_vs_ref"] - 1.0) < _CONTAMINATION_THRESHOLD
                f.write(f"{cfg['seq_len']},{cfg['n_epochs']},{name},"
                        f"{a['total']:.4f},{a['total_iqr']:.4f},{a['gae']:.5f},"
                        f"{a['dev']:.5f},{a['share']:.5f},"
                        f"{a['speedup_vs_triton']:.5f},{a['speedup_vs_triton_iqr']:.5f},"
                        f"{a['nongae_vs_ref']:.5f},{a['resid']:.4f},{citable}\n")
    print(f"Wrote {csv_path}")
    print("\nNOTE: raw output files are gitignored (paper-specific dated output, "
          "matching this repo's convention for ppo_e2e_measurement.py -- see "
          "benchmarks/README.md). The script itself is the tracked artifact.")


if __name__ == "__main__":
    main()
