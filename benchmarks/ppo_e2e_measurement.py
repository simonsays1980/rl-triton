"""ONE-OFF measurement: GAE's share of a real PPO update step.

Purpose: pre-empt "does this speed up training?" by honestly measuring what
fraction of a full PPO update step GAE actually is. NOT a recurring benchmark —
run once, report the resulting number as a single sentence for the paper's
evaluation section. Do NOT wire this into bench_release.py, benchmarks.md, or
README.md, and do not draw a "we win/lose" conclusion here — that is a
STOP-and-report item for a human to sign off on.

Fixes a methodology bug in the prior measurement (tests/h100_short_horizon_l2_
retrace_ppo_report.md, Experiment 4): that report ran the Triton-GAE arm and
the torch.compile-GAE arm as separate timed blocks, and its own "optimizer"
stage came out 43% different between arms (0.294ms vs 0.518ms) — impossible
from a GAE-only change, since GAE finishes well before the optimizer step and
has no way to affect Adam's cost. That is the signature of the two arms not
being measured identically (GPU clock state / allocator warmth drifting
between the two blocks). Fix here: interleave both arms in ONE process, same
seeds, alternating which arm runs first every iteration, so any slow drift
affects both arms equally instead of contaminating whichever arm ran second.

Setup: synthetic rollout buffer (no real env — this measures update-step cost,
not env-interaction cost), num_envs=4096, seq_len=128, Isaac-Gym-Ant-like MLP
policy+value net (obs_dim=60, action_dim=8, continuous Gaussian policy).
Full PPO update per measured iteration: GAE once per rollout, then 4 epochs x
4 minibatches of (forward, clipped policy loss + value loss, backward, Adam
step) = 16 backward/optimizer passes. Both eager and torch.compile'd
policy/value net are measured, at hidden sizes (256,256) and (1024,1024).

Usage:
    python benchmarks/ppo_e2e_measurement.py
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_gae import vectorized_gae
from rl_triton.ops.gae import compute_gae

torch._dynamo.config.cache_size_limit = 64  # avoid the silent eager-fallback bug from Experiment 1
# TF32 tensor cores for fp32 matmul: a real production PPO deployment on Ampere/Hopper would
# normally enable this. Leaving it off (the default) makes forward/backward artificially slow,
# which understates GAE's percentage share of the step relative to a realistically-tuned
# production config (same absolute GAE-vs-baseline gap, smaller denominator with TF32 on).
torch.set_float32_matmul_precision("high")

NUM_ENVS = 4096
SEQ_LEN = 128
OBS_DIM = 60
ACTION_DIM = 8
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
N_EPOCHS = 4
N_MINIBATCHES = 4
N_WARMUP = 10
N_ITERS = 30
SEED = 0


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


def _make_rollout(seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    obs = torch.randn(NUM_ENVS, SEQ_LEN, OBS_DIM, device=device, generator=g)
    actions = torch.randn(NUM_ENVS, SEQ_LEN, ACTION_DIM, device=device, generator=g)
    old_log_probs = torch.randn(NUM_ENVS, SEQ_LEN, device=device, generator=g) * 0.1 - 5.0
    rewards = torch.randn(NUM_ENVS, SEQ_LEN, device=device, generator=g)
    dones = (torch.rand(NUM_ENVS, SEQ_LEN, device=device, generator=g) < 0.02).float()
    old_values = torch.randn(NUM_ENVS, SEQ_LEN, device=device, generator=g)
    return obs, actions, old_log_probs, rewards, dones, old_values


def _gaussian_log_prob(mean, log_std, actions):
    std = log_std.exp()
    var = std * std
    return (-0.5 * ((actions - mean) ** 2) / var - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)


def _run_ppo_update(net, optimizer, rollout, gae_fn, timers):
    """One full PPO update. Returns nothing; accumulates elapsed times (ms)
    into `timers` dict (keys: forward, gae, loss, backward, optimizer)."""
    obs, actions, old_log_probs, rewards, dones, old_values = rollout

    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        flat_obs = obs.reshape(-1, OBS_DIM)
        _, _, values_flat = net(flat_obs)
        values = values_flat.reshape(NUM_ENVS, SEQ_LEN)
    torch.cuda.synchronize(); timers["forward"] += (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize(); t0 = time.perf_counter()
    advantages = gae_fn(rewards, values, dones, gamma=GAMMA, lambda_=LAMBDA)
    returns = advantages + values
    torch.cuda.synchronize(); timers["gae"] += (time.perf_counter() - t0) * 1000

    flat_actions = actions.reshape(-1, ACTION_DIM)
    flat_old_log_probs = old_log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1).detach()
    flat_returns = returns.reshape(-1).detach()
    batch_size = NUM_ENVS * SEQ_LEN
    minibatch_size = batch_size // N_MINIBATCHES

    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(batch_size, device=obs.device)
        for mb in range(N_MINIBATCHES):
            idx = perm[mb * minibatch_size:(mb + 1) * minibatch_size]
            mb_obs, mb_actions = flat_obs[idx], flat_actions[idx]
            mb_old_lp, mb_adv, mb_ret = flat_old_log_probs[idx], flat_advantages[idx], flat_returns[idx]

            torch.cuda.synchronize(); t0 = time.perf_counter()
            mean, log_std, value = net(mb_obs)
            torch.cuda.synchronize(); timers["forward"] += (time.perf_counter() - t0) * 1000

            torch.cuda.synchronize(); t0 = time.perf_counter()
            log_prob = _gaussian_log_prob(mean, log_std, mb_actions)
            ratio = (log_prob - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
            loss = policy_loss + value_loss
            torch.cuda.synchronize(); timers["loss"] += (time.perf_counter() - t0) * 1000

            torch.cuda.synchronize(); t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize(); timers["backward"] += (time.perf_counter() - t0) * 1000

            torch.cuda.synchronize(); t0 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(); timers["optimizer"] += (time.perf_counter() - t0) * 1000


def measure(hidden, compiled, device="cuda"):
    """Interleaved A/B: arm 'triton' (compute_gae) vs arm 'baseline'
    (torch.compile(vectorized_gae)), alternating which arm runs first every
    iteration, same rollout/seed for both arms within an iteration."""
    torch.manual_seed(SEED)
    net_a = ActorCritic(hidden).to(device)
    net_b = ActorCritic(hidden).to(device)
    net_b.load_state_dict(net_a.state_dict())  # identical init — isolates the GAE-arm difference

    if compiled:
        net_a_fwd = torch.compile(net_a)
        net_b_fwd = torch.compile(net_b)
    else:
        net_a_fwd, net_b_fwd = net_a, net_b

    opt_a = torch.optim.Adam(net_a.parameters(), lr=3e-4)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=3e-4)

    gae_baseline = torch.compile(vectorized_gae)

    def run_arm(net_fwd, optimizer, gae_fn, rollout, timers):
        _run_ppo_update(net_fwd, optimizer, rollout, gae_fn, timers)

    timers_a = {"forward": 0.0, "gae": 0.0, "loss": 0.0, "backward": 0.0, "optimizer": 0.0}
    timers_b = {"forward": 0.0, "gae": 0.0, "loss": 0.0, "backward": 0.0, "optimizer": 0.0}

    for i in range(N_WARMUP):
        rollout = _make_rollout(seed=1000 + i, device=device)
        run_arm(net_a_fwd, opt_a, compute_gae, rollout, {"forward": 0, "gae": 0, "loss": 0, "backward": 0, "optimizer": 0})
        run_arm(net_b_fwd, opt_b, gae_baseline, rollout, {"forward": 0, "gae": 0, "loss": 0, "backward": 0, "optimizer": 0})

    for i in range(N_ITERS):
        rollout = _make_rollout(seed=2000 + i, device=device)
        if i % 2 == 0:
            run_arm(net_a_fwd, opt_a, compute_gae, rollout, timers_a)
            run_arm(net_b_fwd, opt_b, gae_baseline, rollout, timers_b)
        else:
            run_arm(net_b_fwd, opt_b, gae_baseline, rollout, timers_b)
            run_arm(net_a_fwd, opt_a, compute_gae, rollout, timers_a)

    for timers in (timers_a, timers_b):
        for k in timers:
            timers[k] /= N_ITERS

    return timers_a, timers_b


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available — skipping.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")
    print(f"num_envs={NUM_ENVS} seq_len={SEQ_LEN} epochs={N_EPOCHS} minibatches={N_MINIBATCHES}")
    print(f"Interleaved A/B, {N_ITERS} timed iterations ({N_WARMUP} warmup), same seeds per iteration.\n")

    report_lines = [
        f"# PPO end-to-end measurement — {datetime.date.today().isoformat()}",
        "",
        "ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table.",
        f"GPU: {torch.cuda.get_device_name(0)} · torch {torch.__version__}",
        f"num_envs={NUM_ENVS}, seq_len={SEQ_LEN}, {N_EPOCHS} epochs x {N_MINIBATCHES} minibatches, "
        f"{N_ITERS} interleaved-A/B iterations ({N_WARMUP} warmup).",
        "",
    ]

    for hidden in [(256, 256), (1024, 1024)]:
        for compiled in [False, True]:
            mode = "torch.compile" if compiled else "eager"
            print(f"=== hidden={hidden}  net mode={mode} ===")
            timers_a, timers_b = measure(hidden, compiled)
            total_a = sum(timers_a.values())
            total_b = sum(timers_b.values())
            print(f"{'stage':<12} {'triton GAE':>14} {'baseline GAE':>16}")
            for stage in ("forward", "gae", "loss", "backward", "optimizer"):
                pa = timers_a[stage] / total_a * 100
                pb = timers_b[stage] / total_b * 100
                print(f"{stage:<12} {timers_a[stage]:>10.4f}ms ({pa:>5.1f}%) {timers_b[stage]:>12.4f}ms ({pb:>5.1f}%)")
            print(f"{'total':<12} {total_a:>10.4f}ms {'':>7} {total_b:>12.4f}ms")
            speedup = total_b / total_a
            gae_pct_a = timers_a["gae"] / total_a * 100
            gae_pct_b = timers_b["gae"] / total_b * 100
            print(f"GAE share of step: {gae_pct_a:.2f}% (triton arm) / {gae_pct_b:.2f}% (baseline arm)")
            print(f"End-to-end speedup (total_b / total_a): {speedup:.3f}x\n")

            report_lines += [
                f"### hidden={hidden}, net mode={mode}",
                "",
                "| stage | Triton GAE | baseline (torch.compile vectorized) GAE |",
                "|---|---|---|",
            ]
            for stage in ("forward", "gae", "loss", "backward", "optimizer"):
                pa = timers_a[stage] / total_a * 100
                pb = timers_b[stage] / total_b * 100
                report_lines.append(f"| {stage} | {timers_a[stage]:.4f} ms ({pa:.1f}%) | {timers_b[stage]:.4f} ms ({pb:.1f}%) |")
            report_lines += [
                f"| **total** | **{total_a:.4f} ms** | **{total_b:.4f} ms** |",
                "",
                f"GAE share of step: {gae_pct_a:.2f}% (triton arm) / {gae_pct_b:.2f}% (baseline arm). "
                f"End-to-end speedup: {speedup:.3f}x.",
                "",
            ]

    out_dir = Path(__file__).parent.parent / "docs" / "benchmark-history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ppo-e2e-measurement-{datetime.date.today().isoformat()}.md"
    out_path.write_text("\n".join(report_lines))
    print(f"Wrote {out_path} (reference only — not part of benchmarks.md/README's recurring tables).")


if __name__ == "__main__":
    main()
