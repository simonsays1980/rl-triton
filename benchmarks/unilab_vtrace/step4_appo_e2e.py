"""Step 4: real APPOLearner training loop, numpy V-trace vs. Triton V-trace.

No env, worker, or collector pipeline: uni_rl's own
tests/algos/test_appo_learner_metrics.py drives APPOLearner.process_batch()/
update() directly with a synthetic batch_dict (no EnvProtocol implementation
needed at all) -- that pattern is reused and extended here (its minimal
actor/critic stub only covers process_batch(); update()'s
_actor_mean_std/_critic_value need actor.mlp/obs_normalizer/distribution too,
so BenchActor/BenchCritic below add those, kept duck-typed and NOT the real
rsl_rl.models.MLPModel, staying faithful to the test suite's own minimal-stub
convention rather than pulling in rsl_rl's distribution_cfg/resolve_class
config system).

learner.py is NEVER modified. The only difference between the "numpy" and
"triton" arms is which function is bound to
uni_rl.algos.appo.learner.vtrace_advantages (module-level monkeypatch,
restored after this arm's iterations) -- everything else (weights, batches,
PPO/APPO loss, clipping, minibatching, learning rate) is identical.

This is a wall-clock/throughput check ONLY. 8 iterations proves nothing about
learning quality -- no sample-efficiency or convergence claim is made or
implied.
"""
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import capture_metadata, today, unique_path  # noqa: E402

import uni_rl.algos.appo.learner as learner_mod  # noqa: E402
from uni_rl.algos.appo.learner import APPOLearner, vtrace_advantages as unilab_vtrace_advantages  # noqa: E402
from rl_triton.ops.vtrace import compute_vtrace  # noqa: E402

UNILAB_REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path(__file__).parent.parent

T, N = 24, 1024  # confirmed against unilab_rl tests/algos/test_rsl_rl_ppo.py:69
OBS_DIM, ACTION_DIM = 16, 10  # matches this repo's own resip_ppo_e2e_measurement.py convention
N_ITERS = 8
DEVICE = "cuda"
CONTAMINATION_THRESHOLD_PCT = 1.0


def triton_vtrace_advantages(behavior_log_probs, target_log_probs, rewards, values,
                              bootstrap_values, dones, gamma=0.99, clip_rho=1.0, clip_c=1.0):
    """Drop-in replacement for uni_rl's vtrace_advantages with the identical
    signature/shape contract ([T, N] in, [T, N] out) -- verified correct
    against UniLab's real formula in Step 1."""
    t = lambda x: x.transpose(0, 1).contiguous().float()
    vs, adv = compute_vtrace(
        log_pi_target=t(target_log_probs), log_pi_behavior=t(behavior_log_probs),
        values=t(values), rewards=t(rewards), terminateds=t(dones), truncateds=None,
        gamma=gamma, rho_bar=clip_rho, c_bar=clip_c, last_value=bootstrap_values.float(),
    )
    return vs.transpose(0, 1), adv.transpose(0, 1)


class _DistStub:
    """Minimal stand-in for rsl_rl's Distribution -- only the two attributes
    APPOLearner._distribution_std() reads."""
    def __init__(self, log_std_param):
        self.std_type = "log"
        self.log_std_param = log_std_param


class BenchActor(nn.Module):
    """Extends test_appo_learner_metrics.py's _Actor stub with .mlp/.obs_normalizer/
    .distribution so update()'s _actor_mean_std path works too (that test only
    exercises process_batch())."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))
        self.distribution = _DistStub(self.log_std_param)
        self.obs_normalizer = nn.Identity()
        self.output_mean = torch.zeros(1, action_dim)
        self.output_std = torch.ones(1, action_dim)

    @property
    def mlp(self):
        return self.linear

    def forward(self, obs, stochastic_output: bool = False):
        del stochastic_output
        mean = self.linear(self.obs_normalizer(obs["policy"]))
        std = torch.exp(self.log_std_param).expand_as(mean)
        self.output_mean = mean
        self.output_std = std
        return mean

    def get_output_log_prob(self, actions):
        dist = torch.distributions.Normal(self.output_mean, self.output_std)
        return dist.log_prob(actions).sum(dim=-1)


class BenchCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.linear = nn.Linear(obs_dim, 1)
        self.obs_normalizer = nn.Identity()

    @property
    def mlp(self):
        return self.linear

    def forward(self, obs):
        return self.linear(self.obs_normalizer(obs["policy"]))


def make_batch(seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return {
        "observations": torch.randn(T, N, OBS_DIM, generator=g, device=DEVICE),
        "actions": torch.randn(T, N, ACTION_DIM, generator=g, device=DEVICE),
        "actions_log_prob": -torch.rand(T, N, generator=g, device=DEVICE),
        "rewards": torch.randn(T, N, generator=g, device=DEVICE),
        "dones": (torch.rand(T, N, generator=g, device=DEVICE) < 0.02).float(),
        "last_obs": torch.randn(N, OBS_DIM, generator=g, device=DEVICE),
    }


class StageTimer:
    def __init__(self):
        self.ms = {}

    def add(self, stage, ms):
        self.ms[stage] = self.ms.get(stage, 0.0) + ms


def cuda_timed(fn, timer, stage):
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        timer.add(stage, start.elapsed_time(end))
        return out
    return wrapped


def wall_timed(fn, timer, stage):
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timer.add(stage, (t1 - t0) * 1000.0)
        return out
    return wrapped


def vtrace_timed_wrapper(fn, timer):
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timer.add("vtrace_wrapped", (t1 - t0) * 1000.0)
        return out
    return wrapped


def main():
    metadata = capture_metadata(UNILAB_REPO_DIR)

    # Same input batches fed to BOTH arms (pre-generated once, fixed seed).
    batches = [make_batch(seed=1000 + i) for i in range(N_ITERS)]

    # vtrace_ms is captured via a per-iteration wrapper (needs access to that
    # iteration's StageTimer, hence the timer_holder indirection).
    def make_numpy_vtrace(timer_holder):
        def fn(*a, **kw):
            return vtrace_timed_wrapper(unilab_vtrace_advantages, timer_holder[0])(*a, **kw)
        return fn

    def make_triton_vtrace(timer_holder):
        def fn(*a, **kw):
            return vtrace_timed_wrapper(triton_vtrace_advantages, timer_holder[0])(*a, **kw)
        return fn

    results = {}
    for arm_name, vtrace_impl in [("numpy", unilab_vtrace_advantages), ("triton", triton_vtrace_advantages)]:
        torch.manual_seed(42)
        actor = BenchActor(OBS_DIM, ACTION_DIM).to(DEVICE)
        critic = BenchCritic(OBS_DIM).to(DEVICE)
        if arm_name == "numpy":
            actor_seed_state = copy.deepcopy(actor.state_dict())
            critic_seed_state = copy.deepcopy(critic.state_dict())
        else:
            actor.load_state_dict(actor_seed_state)
            critic.load_state_dict(critic_seed_state)

        learner = APPOLearner(actor=actor, critic=critic, device=DEVICE)
        orig_vtrace = learner_mod.vtrace_advantages
        orig_critic_forward = learner.critic.forward
        orig_target_actor_forward = learner.target_actor.forward
        orig_optimizer_step = learner.optimizer.step

        # Untimed warmup: one full process_batch()+update() call to absorb
        # lazy cuBLAS/cuDNN init (hits whichever arm runs first) and, for the
        # triton arm, the Triton kernel's own JIT compile/autotune -- without
        # this, "iteration 0" below measures one-time compile cost, not
        # steady-state per-iteration training time (confirmed empirically:
        # iteration 0 was 568-664ms vs. ~100-120ms steady state before this
        # warmup was added).
        learner_mod.vtrace_advantages = vtrace_impl
        warmup_batch = dict(batches[0])
        learner.process_batch(warmup_batch)
        learner.update(warmup_batch)
        torch.cuda.synchronize()

        per_iter = []
        for i, batch in enumerate(batches):
            timer = StageTimer()
            timer_holder = [timer]
            learner_mod.vtrace_advantages = (
                make_numpy_vtrace(timer_holder) if arm_name == "numpy" else make_triton_vtrace(timer_holder)
            )
            learner.critic.forward = cuda_timed(orig_critic_forward, timer, "critic_forward")
            learner.target_actor.forward = cuda_timed(orig_target_actor_forward, timer, "actor_forward")
            learner.optimizer.step = cuda_timed(orig_optimizer_step, timer, "optimizer")

            batch_copy = dict(batch)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wall_timed(learner.process_batch, timer, "process_batch_total")(batch_copy)
            wall_timed(learner.update, timer, "update_total")(batch_copy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            learner.critic.forward = orig_critic_forward
            learner.target_actor.forward = orig_target_actor_forward
            learner.optimizer.step = orig_optimizer_step

            total_ms = (t1 - t0) * 1000.0
            vtrace_ms = timer.ms.get("vtrace_wrapped", 0.0)
            critic_forward_ms = timer.ms.get("critic_forward", 0.0)
            actor_forward_ms = timer.ms.get("actor_forward", 0.0)
            pb_total_ms = timer.ms.get("process_batch_total", 0.0)
            rest_of_pb_ms = pb_total_ms - critic_forward_ms - actor_forward_ms - vtrace_ms
            update_total_ms = timer.ms.get("update_total", 0.0)
            optimizer_ms = timer.ms.get("optimizer", 0.0)
            update_minibatch_loop_ms = update_total_ms - optimizer_ms

            per_iter.append({
                "iter": i, "total_ms": total_ms,
                "critic_forward_ms": critic_forward_ms, "actor_forward_ms": actor_forward_ms,
                "vtrace_ms": vtrace_ms, "rest_of_process_batch_ms": rest_of_pb_ms,
                "update_minibatch_loop_ms": update_minibatch_loop_ms, "optimizer_ms": optimizer_ms,
                "env_steps_per_sec": (T * N) / (total_ms / 1000.0),
            })
            print(f"[{arm_name}] iter {i}: total={total_ms:8.2f}ms  vtrace={vtrace_ms:7.3f}ms  "
                  f"critic_fwd={critic_forward_ms:6.3f}ms  actor_fwd={actor_forward_ms:6.3f}ms  "
                  f"update_loop={update_minibatch_loop_ms:7.3f}ms  optimizer={optimizer_ms:6.3f}ms  "
                  f"env_steps/s={(T*N)/(total_ms/1000.0):.1f}")

        learner_mod.vtrace_advantages = orig_vtrace
        results[arm_name] = per_iter

    # Contamination check: do the non-vtrace stages differ by more than the threshold?
    non_vtrace_stages = ["critic_forward_ms", "actor_forward_ms", "rest_of_process_batch_ms",
                          "update_minibatch_loop_ms", "optimizer_ms"]
    max_rel_diff_pct = 0.0
    worst_stage = None
    for stage in non_vtrace_stages:
        numpy_mean = sum(r[stage] for r in results["numpy"]) / N_ITERS
        triton_mean = sum(r[stage] for r in results["triton"]) / N_ITERS
        if max(numpy_mean, triton_mean) < 1e-6:
            continue
        rel_diff_pct = 100.0 * abs(numpy_mean - triton_mean) / max(numpy_mean, triton_mean)
        if rel_diff_pct > max_rel_diff_pct:
            max_rel_diff_pct = rel_diff_pct
            worst_stage = stage
    contamination_citable = max_rel_diff_pct <= CONTAMINATION_THRESHOLD_PCT

    print(f"\nContamination check (non-vtrace stages): max relative diff = {max_rel_diff_pct:.2f}% "
          f"(stage: {worst_stage}), threshold={CONTAMINATION_THRESHOLD_PCT}%, "
          f"{'PASS' if contamination_citable else 'FAIL -- CONTAMINATED'}")

    out = {
        "scope": f"learner-side wall-clock/throughput only; NOT a sample-efficiency or "
                 f"convergence result; N={N_ITERS} iterations",
        "metadata": metadata,
        "config": {"T": T, "N": N, "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
                   "num_learning_epochs": "class default (5)", "num_mini_batches": "class default (4)",
                   "n_iters": N_ITERS, "seed": 42,
                   "note": "learner.py NOT modified -- vtrace_advantages module-level name "
                           "monkeypatched from this script only, restored after each arm; "
                           "actor/critic weights identical across arms via state_dict copy "
                           "before either learner is constructed; same pre-generated batches "
                           "fed to both arms; ONE untimed warmup process_batch()+update() call "
                           "per arm before the N_ITERS measured iterations below, to absorb "
                           "lazy cuBLAS init and (triton arm only) Triton JIT compile -- both "
                           "arms' post-warmup weights differ slightly from pristine init as a "
                           "result, symmetrically for both arms"},
        "contamination_check": {
            "max_rel_diff_pct": max_rel_diff_pct, "worst_stage": worst_stage,
            "threshold_pct": CONTAMINATION_THRESHOLD_PCT, "passed": contamination_citable,
        },
        "citable": contamination_citable,
        "results": results,
    }
    out_path = unique_path(OUT_DIR / f"unilab_appo_learner_e2e-{today()}.json")
    out_path.write_text(json.dumps(out, indent=2))
    csv_path = out_path.with_suffix(".csv")
    cols = ["arm", "iter", "total_ms", "critic_forward_ms", "actor_forward_ms", "vtrace_ms",
            "rest_of_process_batch_ms", "update_minibatch_loop_ms", "optimizer_ms", "env_steps_per_sec"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for arm_name in ["numpy", "triton"]:
            for r in results[arm_name]:
                f.write(",".join(str(r.get(c, arm_name) if c != "arm" else arm_name) for c in cols) + "\n")
    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
