"""Step 5 / Part A: does aggregated batch size (multi-worker RolloutStagingPool
output) change V-trace's standalone speedup or its share of a learner iteration?

Follow-up to step2_standalone_timing.py (single-worker N=512/1024 only) and
step4_appo_e2e.py, whose derived-residual stage fields (rest_of_process_batch_ms,
update_minibatch_loop_ms -- both computed as wall-clock-minus-CUDA-event
subtractions) were identified as the likely source of that script's noisy,
uncitable e2e result. This script avoids that pattern entirely: every reported
stage is DIRECTLY measured (never subtracted), and all stage timings used for
the vtrace-share-of-iteration computation share ONE clock (wall-clock-with-sync)
so nothing is ever a difference of two different clock domains.

No stated typical worker count W exists anywhere in unilab_rl (searched
configs/docstrings/comments/tests; the only related default found,
runner.py's replay_queue_size=3 -> RolloutStagingPool.capacity, is a bounded
queue depth, not documented anywhere as a worker count) -- so W is swept,
not guessed.

learner.py is never modified -- same module-level vtrace_advantages
monkeypatch technique as step4_appo_e2e.py, restored after each arm.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    assert_finite_and_close, bench_gpu_cuda_events, bench_gpu_wall_clock,
    capture_metadata, today, unique_path, warmup_gpu,
)
from step1_correctness import run_unilab, run_triton  # noqa: E402
from step4_appo_e2e import BenchActor, BenchCritic, triton_vtrace_advantages  # noqa: E402

import uni_rl.algos.appo.learner as learner_mod  # noqa: E402
from uni_rl.algos.appo.learner import APPOLearner, vtrace_advantages as unilab_vtrace_advantages  # noqa: E402
from uni_rl.algos.appo.staging import RolloutStagingPool  # noqa: E402

UNILAB_REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path(__file__).parent.parent
ATOL = RTOL = 1e-4
N_WARMUP, N_TRIALS, N_ITER = 20, 5, 100
DEVICE = "cuda"

T = 24
NUM_ENVS_PER_WORKER = 512
W_SWEEP = [1, 4, 8, 16, 32]  # aggregated N spans 512 .. 16384
OBS_DIM, ACTION_DIM = 16, 10
GAMMA, CLIP_RHO, CLIP_C = 0.99, 1.0, 1.0
DECISION_THRESHOLD_PCT = 2.5  # "under ~2-3%" per task


def build_aggregated_batch(W, seed):
    """Stage W independent synthetic per-worker rollouts through the REAL
    RolloutStagingPool, exactly as a real deployment would."""
    slot_shapes = {
        "obs": (NUM_ENVS_PER_WORKER, T, OBS_DIM),
        "actions": (NUM_ENVS_PER_WORKER, T, ACTION_DIM),
        "log_probs": (NUM_ENVS_PER_WORKER, T),
        "rewards": (NUM_ENVS_PER_WORKER, T),
        "dones": (NUM_ENVS_PER_WORKER, T),
        "last_obs": (NUM_ENVS_PER_WORKER, OBS_DIM),
    }
    pool = RolloutStagingPool(capacity=W, num_envs=NUM_ENVS_PER_WORKER, slot_shapes=slot_shapes, device=DEVICE)
    rng = np.random.default_rng(seed)
    for _ in range(W):
        raw = {
            "obs": rng.standard_normal((NUM_ENVS_PER_WORKER, T, OBS_DIM)).astype(np.float32),
            "actions": rng.standard_normal((NUM_ENVS_PER_WORKER, T, ACTION_DIM)).astype(np.float32),
            "log_probs": (-rng.random((NUM_ENVS_PER_WORKER, T))).astype(np.float32),
            "rewards": rng.standard_normal((NUM_ENVS_PER_WORKER, T)).astype(np.float32),
            "dones": (rng.random((NUM_ENVS_PER_WORKER, T)) < 0.02).astype(np.float32),
            "last_obs": rng.standard_normal((NUM_ENVS_PER_WORKER, OBS_DIM)).astype(np.float32),
        }
        pool.stage_numpy_views(raw)
    return pool.batch()


def cuda_timed(fn, timer, stage):
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        timer[stage] = timer.get(stage, 0.0) + start.elapsed_time(end)
        return out
    return wrapped


def wall_timed(fn, timer, stage):
    import time
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timer[stage] = timer.get(stage, 0.0) + (t1 - t0) * 1000.0
        return out
    return wrapped


def run_arm_at_shape(arm_name, vtrace_impl, batch, actor_state, critic_state):
    """Run one warmup + one measured process_batch()+update() pass with a real
    APPOLearner, all stage timing on wall-clock-with-sync (uniform clock,
    nothing derived by subtraction). Returns stage_ms dict and the extracted
    (behavior_log_probs, target_log_probs, rewards, values, bootstrap_values,
    dones) tensors for the separate standalone vtrace-only comparison below."""
    actor = BenchActor(OBS_DIM, ACTION_DIM).to(DEVICE)
    critic = BenchCritic(OBS_DIM).to(DEVICE)
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    learner = APPOLearner(actor=actor, critic=critic, device=DEVICE)

    orig_vtrace = learner_mod.vtrace_advantages
    orig_critic_forward = learner.critic.forward
    orig_target_actor_forward = learner.target_actor.forward

    # Untimed warmup (absorbs cuBLAS lazy-init / Triton JIT compile -- see
    # step4_appo_e2e.py's run notes for why this matters).
    learner_mod.vtrace_advantages = vtrace_impl
    warmup_batch = dict(batch)
    learner.process_batch(warmup_batch)
    learner.update(warmup_batch)
    torch.cuda.synchronize()

    timer = {}
    learner_mod.vtrace_advantages = wall_timed(vtrace_impl, timer, "vtrace")
    learner.critic.forward = wall_timed(orig_critic_forward, timer, "critic_forward")
    learner.target_actor.forward = wall_timed(orig_target_actor_forward, timer, "actor_forward")

    measured_batch = dict(batch)
    wall_timed(learner.process_batch, timer, "process_batch_total")(measured_batch)
    wall_timed(learner.update, timer, "update_total")(measured_batch)

    learner_mod.vtrace_advantages = orig_vtrace
    learner.critic.forward = orig_critic_forward
    learner.target_actor.forward = orig_target_actor_forward

    dones_f = measured_batch["dones"].float() if measured_batch["dones"].dtype != torch.float32 else measured_batch["dones"]
    stage_ms = {
        "critic_forward_ms": timer.get("critic_forward", 0.0),
        "actor_forward_ms": timer.get("actor_forward", 0.0),
        "vtrace_ms": timer.get("vtrace", 0.0),
        "update_total_ms": timer.get("update_total", 0.0),
    }
    extracted = {
        "behavior_log_probs": measured_batch["actions_log_prob"],
        "target_log_probs": measured_batch["target_log_probs"],
        "rewards": measured_batch["rewards"],
        "values": measured_batch["values"],
        "dones": dones_f,
    }
    return stage_ms, extracted


def main():
    metadata = capture_metadata(UNILAB_REPO_DIR)
    torch.manual_seed(0)
    seed_actor = BenchActor(OBS_DIM, ACTION_DIM).to(DEVICE)
    seed_critic = BenchCritic(OBS_DIM).to(DEVICE)
    actor_state = copy.deepcopy(seed_actor.state_dict())
    critic_state = copy.deepcopy(seed_critic.state_dict())

    rows = []
    for W in W_SWEEP:
        N_agg = W * NUM_ENVS_PER_WORKER
        batch = build_aggregated_batch(W, seed=W)

        for arm_name, vtrace_impl in [("unilab_numpy", unilab_vtrace_advantages),
                                       ("triton", triton_vtrace_advantages)]:
            stage_ms, extracted = run_arm_at_shape(arm_name, vtrace_impl, batch, actor_state, critic_state)
            share_denom = (stage_ms["critic_forward_ms"] + stage_ms["actor_forward_ms"]
                           + stage_ms["vtrace_ms"] + stage_ms["update_total_ms"])
            vtrace_share_pct = 100.0 * stage_ms["vtrace_ms"] / share_denom if share_denom > 0 else float("nan")

            # bootstrap_values: recompute directly (process_batch doesn't
            # store last_values in batch_dict, only uses it internally).
            with torch.inference_mode():
                bootstrap_values = seed_critic({"policy": batch["last_obs"]}).squeeze(-1).detach()

            # Correctness gate + standalone comparison at this shape, using
            # the REAL values/target_log_probs this arm's own process_batch()
            # just computed (not independently-generated random tensors).
            vs_unilab, adv_unilab = run_unilab(
                extracted["behavior_log_probs"], extracted["target_log_probs"], extracted["rewards"],
                extracted["values"], bootstrap_values, extracted["dones"], GAMMA, CLIP_RHO, CLIP_C,
            )
            vs_triton, adv_triton = run_triton(
                extracted["behavior_log_probs"], extracted["target_log_probs"], extracted["rewards"],
                extracted["values"], bootstrap_values, extracted["dones"], GAMMA, CLIP_RHO, CLIP_C,
            )
            vs_pass, vs_max, _, _ = assert_finite_and_close(vs_triton, vs_unilab, "vs", ATOL, RTOL)
            adv_pass, adv_max, _, _ = assert_finite_and_close(adv_triton, adv_unilab, "adv", ATOL, RTOL)
            citable = vs_pass and adv_pass

            def unilab_fn():
                return run_unilab(extracted["behavior_log_probs"], extracted["target_log_probs"],
                                   extracted["rewards"], extracted["values"], bootstrap_values,
                                   extracted["dones"], GAMMA, CLIP_RHO, CLIP_C)
            def triton_fn():
                return run_triton(extracted["behavior_log_probs"], extracted["target_log_probs"],
                                   extracted["rewards"], extracted["values"], bootstrap_values,
                                   extracted["dones"], GAMMA, CLIP_RHO, CLIP_C)
            standalone_fn = unilab_fn if arm_name == "unilab_numpy" else triton_fn
            warmup_gpu(standalone_fn, n_warmup=N_WARMUP)
            cuda_ms, cuda_trials = bench_gpu_cuda_events(standalone_fn, n_iter=N_ITER, n_trials=N_TRIALS)
            wall_ms, wall_trials = bench_gpu_wall_clock(standalone_fn, n_iter=N_ITER, n_trials=N_TRIALS)

            row = {
                "W": W, "N_aggregated": N_agg, "num_transitions": T * N_agg, "implementation": arm_name,
                "standalone_cuda_latency_us": cuda_ms * 1000.0,
                "standalone_wall_latency_us": wall_ms * 1000.0,
                "critic_forward_ms": stage_ms["critic_forward_ms"],
                "actor_forward_ms": stage_ms["actor_forward_ms"],
                "vtrace_ms": stage_ms["vtrace_ms"],
                "update_total_ms": stage_ms["update_total_ms"],
                "vtrace_share_of_iteration_pct": vtrace_share_pct,
                "vs_max_abs_err": vs_max, "adv_max_abs_err": adv_max, "citable": citable,
                "trial_medians_cuda_ms": cuda_trials, "trial_medians_wall_ms": wall_trials,
            }
            rows.append(row)
            print(f"W={W:3d} N_agg={N_agg:6d} {arm_name:13s}  standalone_cuda={cuda_ms*1000:9.2f}us  "
                  f"critic_fwd={stage_ms['critic_forward_ms']:7.3f}ms  actor_fwd={stage_ms['actor_forward_ms']:7.3f}ms  "
                  f"vtrace={stage_ms['vtrace_ms']:7.3f}ms  update={stage_ms['update_total_ms']:8.3f}ms  "
                  f"vtrace_share={vtrace_share_pct:.3f}%  citable={citable}")

        # standalone speedup line for this W (paired unilab/triton rows just appended)
        numpy_row, triton_row = rows[-2], rows[-1]
        speedup_cuda = numpy_row["standalone_cuda_latency_us"] / triton_row["standalone_cuda_latency_us"]
        print(f"  -> W={W} standalone CUDA-event speedup: {speedup_cuda:.3f}x")

    max_share = max(r["vtrace_share_of_iteration_pct"] for r in rows)
    proceed_to_part_b = max_share > DECISION_THRESHOLD_PCT
    print(f"\nMax vtrace share observed across all W: {max_share:.3f}% "
          f"(threshold {DECISION_THRESHOLD_PCT}%) -> "
          f"{'PROCEED to Part B' if proceed_to_part_b else 'STOP -- do not proceed to Part B'}")

    out = {
        "metadata": metadata,
        "config": {"T": T, "num_envs_per_worker": NUM_ENVS_PER_WORKER, "W_sweep": W_SWEEP,
                   "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
                   "worker_count_note": "No stated typical/default worker count found in unilab_rl "
                                        "(searched configs/docstrings/tests). runner.py's "
                                        "replay_queue_size defaults to 3 and maps directly to "
                                        "RolloutStagingPool.capacity, but that is a bounded queue "
                                        "depth, not documented anywhere as a worker count -- W was "
                                        "swept rather than guessed, per task instructions.",
                   "note": "learner.py NOT modified; batches built via the REAL RolloutStagingPool; "
                           "values/target_log_probs used in the correctness gate and standalone "
                           "comparison are the REAL outputs of this arm's own process_batch() call, "
                           "not independently-generated random tensors; one untimed warmup pass per "
                           "arm/W before the measured pass"},
        "protocol": {"standalone_n_warmup": N_WARMUP, "standalone_n_trials": N_TRIALS,
                     "standalone_n_iter": N_ITER, "standalone_reducer": "min-of-trial-medians",
                     "share_clock": "wall-clock-with-sync, uniform across critic_forward/actor_forward/"
                                    "vtrace/update_total -- all four DIRECTLY measured, none derived "
                                    "by subtraction"},
        "decision": {"max_vtrace_share_pct": max_share, "threshold_pct": DECISION_THRESHOLD_PCT,
                     "proceed_to_part_b": proceed_to_part_b},
        "rows": rows,
    }
    out_path = unique_path(OUT_DIR / f"unilab_vtrace_aggregated-{today()}.json")
    out_path.write_text(json.dumps(out, indent=2))
    csv_path = out_path.with_suffix(".csv")
    cols = ["W", "N_aggregated", "num_transitions", "implementation", "standalone_cuda_latency_us",
            "standalone_wall_latency_us", "critic_forward_ms", "actor_forward_ms", "vtrace_ms",
            "update_total_ms", "vtrace_share_of_iteration_pct", "vs_max_abs_err", "adv_max_abs_err", "citable"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")
    return proceed_to_part_b


if __name__ == "__main__":
    main()
