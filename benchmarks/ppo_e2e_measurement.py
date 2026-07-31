"""ONE-OFF measurement: GAE's share of a real PPO update step.

Purpose: pre-empt "does this speed up training?" by honestly measuring what
fraction of a full PPO update step GAE actually is. NOT a recurring benchmark --
run once, report the resulting number as a single sentence for the paper's
evaluation section. Do NOT wire this into bench_release.py, benchmarks.md, or
README.md, and do not draw a "we win/lose" conclusion here -- that is a
STOP-and-report item for a human to sign off on.

Fixes a methodology bug in an earlier version of this measurement: running
the Triton-GAE arm and the torch.compile-GAE arm as separate timed blocks
made the "optimizer" stage come out 43% different between arms (0.294ms vs
0.518ms) -- impossible
from a GAE-only change, since GAE finishes well before the optimizer step and
has no way to affect Adam's cost. That is the signature of the arms not being
measured identically (GPU clock state / allocator warmth drifting between
separate blocks). Fix here: interleave all arms in ONE process, same seeds,
cycling through every ordering of the arms across iterations (round-robin
over itertools.permutations), so any slow drift affects all arms equally
instead of consistently favoring/penalizing whichever arm runs in a given
position.

Three arms: the Triton kernel (compute_gae), a torch.compile'd fully
vectorized parallel-scan baseline (vectorized_gae, log2(T)-doubling, no
Python loop), and a torch.compile'd pure-Python sequential backward-scan
baseline (reference_gae, one CUDA op per timestep from Python -- the
"naive" reference implementation someone would actually write by hand).

Setup: synthetic rollout buffer (no real env -- this measures update-step cost,
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
import itertools
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_gae import reference_gae, vectorized_gae
from rl_triton.ops.gae import compute_gae

ARMS = [
    {"key": "triton", "label": "Triton GAE", "make_gae_fn": lambda: compute_gae},
    {
        "key": "vectorized",
        "label": "baseline (torch.compile vectorized) GAE",
        "make_gae_fn": lambda: torch.compile(vectorized_gae),
    },
    {
        "key": "loop",
        "label": "baseline (torch.compile sequential loop) GAE",
        "make_gae_fn": lambda: torch.compile(reference_gae),
    },
]

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
    """Interleaved N-arm: one net+optimizer+gae_fn triple per entry in ARMS,
    all with identical init, cycling through every ordering of the arms
    (round-robin over itertools.permutations) so slow drift affects all arms
    equally instead of consistently favoring/penalizing one position."""
    torch.manual_seed(SEED)
    n_arms = len(ARMS)
    nets = [ActorCritic(hidden).to(device) for _ in ARMS]
    for net in nets[1:]:
        net.load_state_dict(nets[0].state_dict())  # identical init -- isolates the GAE-arm difference

    if compiled:
        net_fwds = [torch.compile(net) for net in nets]
    else:
        net_fwds = list(nets)

    opts = [torch.optim.Adam(net.parameters(), lr=3e-4) for net in nets]
    gae_fns = [arm["make_gae_fn"]() for arm in ARMS]

    def run_arm(idx, rollout, timers):
        _run_ppo_update(net_fwds[idx], opts[idx], rollout, gae_fns[idx], timers)

    timers_list = [{"forward": 0.0, "gae": 0.0, "loss": 0.0, "backward": 0.0, "optimizer": 0.0} for _ in ARMS]
    orderings = list(itertools.permutations(range(n_arms)))

    for i in range(N_WARMUP):
        rollout = _make_rollout(seed=1000 + i, device=device)
        for idx in range(n_arms):
            run_arm(idx, rollout, {"forward": 0, "gae": 0, "loss": 0, "backward": 0, "optimizer": 0})

    for i in range(N_ITERS):
        rollout = _make_rollout(seed=2000 + i, device=device)
        for idx in orderings[i % len(orderings)]:
            run_arm(idx, rollout, timers_list[idx])

    for timers in timers_list:
        for k in timers:
            timers[k] /= N_ITERS

    return timers_list


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available -- skipping.")
        return

    arm_keys = [arm["key"] for arm in ARMS]
    arm_labels = [arm["label"] for arm in ARMS]

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")
    print(f"num_envs={NUM_ENVS} seq_len={SEQ_LEN} epochs={N_EPOCHS} minibatches={N_MINIBATCHES}")
    print(f"Arms: {', '.join(arm_keys)}")
    print(f"Interleaved (round-robin over all {len(ARMS)}!={len(list(itertools.permutations(arm_keys)))} orderings), "
          f"{N_ITERS} timed iterations ({N_WARMUP} warmup), same seeds per iteration.\n")

    report_lines = [
        f"# PPO end-to-end measurement -- {datetime.date.today().isoformat()}",
        "",
        "ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table.",
        f"GPU: {torch.cuda.get_device_name(0)} · torch {torch.__version__}",
        f"Arms: {', '.join(arm_labels)}.",
        f"num_envs={NUM_ENVS}, seq_len={SEQ_LEN}, {N_EPOCHS} epochs x {N_MINIBATCHES} minibatches, "
        f"{N_ITERS} interleaved iterations ({N_WARMUP} warmup), round-robin over all arm orderings.",
        "",
    ]

    for hidden in [(256, 256), (1024, 1024)]:
        for compiled in [False, True]:
            mode = "torch.compile" if compiled else "eager"
            print(f"=== hidden={hidden}  net mode={mode} ===")
            timers_list = measure(hidden, compiled)
            totals = [sum(t.values()) for t in timers_list]

            header = "".join(f"{label:>28}" for label in arm_labels)
            print(f"{'stage':<12}{header}")
            for stage in ("forward", "gae", "loss", "backward", "optimizer"):
                row = "".join(
                    f"{t[stage]:>14.4f}ms ({t[stage] / total * 100:>5.1f}%)"
                    for t, total in zip(timers_list, totals)
                )
                print(f"{stage:<12}{row}")
            totals_row = "".join(f"{total:>21.4f}ms" for total in totals)
            print(f"{'total':<12}{totals_row}")

            gae_pcts = [t["gae"] / total * 100 for t, total in zip(timers_list, totals)]
            gae_share_str = ", ".join(f"{pct:.2f}% ({key})" for pct, key in zip(gae_pcts, arm_keys))
            print(f"GAE share of step: {gae_share_str}")
            speedups = [totals[i] / totals[0] for i in range(1, len(ARMS))]
            speedup_str = ", ".join(
                f"{s:.3f}x ({arm_keys[i + 1]} / {arm_keys[0]})" for i, s in enumerate(speedups)
            )
            print(f"End-to-end speedup vs. triton arm: {speedup_str}\n")

            report_lines += [
                f"### hidden={hidden}, net mode={mode}",
                "",
                "| stage | " + " | ".join(arm_labels) + " |",
                "|---|" + "---|" * len(ARMS),
            ]
            for stage in ("forward", "gae", "loss", "backward", "optimizer"):
                cells = " | ".join(
                    f"{t[stage]:.4f} ms ({t[stage] / total * 100:.1f}%)"
                    for t, total in zip(timers_list, totals)
                )
                report_lines.append(f"| {stage} | {cells} |")
            total_cells = " | ".join(f"**{total:.4f} ms**" for total in totals)
            report_lines += [
                f"| **total** | {total_cells} |",
                "",
                f"GAE share of step: {gae_share_str}. "
                f"End-to-end speedup vs. triton arm: {speedup_str}.",
                "",
            ]

    out_dir = Path(__file__).parent.parent / "docs" / "benchmark-history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ppo-e2e-measurement-{datetime.date.today().isoformat()}.md"
    out_path.write_text("\n".join(report_lines))
    print(f"Wrote {out_path} (reference only -- not part of benchmarks.md/README's recurring tables).")


if __name__ == "__main__":
    main()
