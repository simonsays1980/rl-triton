"""Step 6 / Part B: real async multiprocessing pipeline check.

Only run because Part A (step5_aggregated_batch.py) showed vtrace's share of
a learner iteration exceeding the 2.5% decision threshold at large aggregated
N (9.08% at W=32).

Architectural note found while reading uni_rl.ipc.rollout_ring_buffer before
writing any code: RolloutRingBuffer.signal_write_done() unconditionally
increments write_ptr -- there is no wait-for-space / blocking-on-full check.
A reader that falls behind is clamped forward by
_clamp_read_ptr_to_valid_window() (skips unread slots) rather than blocking
the writer. This means, BY DESIGN, the collector (ring-buffer writer) cannot
be blocked by the learner (reader) via ring-buffer backpressure -- there is
no such mechanism in this architecture. The only cross-process coupling is
the weight-sync read, which the collector performs WITHOUT a lock (see
worker.py's SharedWeightSync(..., create=False, shm_name=...) call, no
lock= passed -> self._lock=None on the subprocess side) -- a deliberate
accepted-tearing-risk design, not a blocking one either. This predicts NO
learner-side stall mechanism reaches the collector through these primitives;
verified empirically below rather than assumed.

Real multiprocessing (mp.get_context("spawn")), genuinely two OS processes,
SAME CUDA device string for both (single H100, no multi-GPU). Uses the real
RolloutRingBuffer/SharedWeightSync/RolloutStagingPool classes unmodified.
learner.py is never modified -- vtrace_advantages monkeypatched exactly as
step4_appo_e2e.py did.
"""
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")

from common import capture_metadata, today, unique_path  # noqa: E402
from step4_appo_e2e import BenchActor, BenchCritic, triton_vtrace_advantages  # noqa: E402
from synthetic_env import synthetic_env_factory  # noqa: E402
from async_collector import collector_fn  # noqa: E402

import uni_rl.algos.appo.learner as learner_mod  # noqa: E402
from uni_rl.algos.appo.learner import APPOLearner, vtrace_advantages as unilab_vtrace_advantages  # noqa: E402
from uni_rl.algos.appo.staging import RolloutStagingPool  # noqa: E402
from uni_rl.ipc.rollout_ring_buffer import RolloutRingBuffer  # noqa: E402
from uni_rl.ipc.weight_sync import SharedWeightSync  # noqa: E402

UNILAB_REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path(__file__).parent.parent
DEVICE = "cuda:0"  # same string for learner and collector -- single H100, no multi-GPU
NUM_ENVS = 1024
STEPS_PER_ENV = 24
OBS_DIM, ACTION_DIM = 16, 10
NUM_SLOTS = 4
N_ITERS_TARGET = 40
MAX_WALL_SECONDS_PER_ARM = 90.0  # give up and report partial if this is exceeded
CONTAMINATION_THRESHOLD_PCT = 2.0  # non-vtrace-adjacent stage drift; justified below


def run_arm(arm_name, vtrace_impl, actor_state, critic_state):
    ctx = mp.get_context("spawn")

    ring_buffer = RolloutRingBuffer(
        num_envs=NUM_ENVS, num_steps=STEPS_PER_ENV, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        critic_dim=0, num_slots=NUM_SLOTS, create=True,
    )  # create=True already allocates _write_ptr/_read_ptr as spawn-context Values

    actor = BenchActor(OBS_DIM, ACTION_DIM).to(DEVICE)
    critic = BenchCritic(OBS_DIM).to(DEVICE)
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    learner = APPOLearner(actor=actor, critic=critic, device=DEVICE)

    actor_weight_sync = SharedWeightSync.from_state_dict(learner.actor.state_dict(), create=True)
    actor_weight_param_shapes = {name: p.shape for name, p in learner.actor.state_dict().items()}

    metrics_queue = ctx.Queue(maxsize=1000)
    stop_event = ctx.Event()

    proc = ctx.Process(
        target=collector_fn,
        kwargs=dict(
            stop_event=stop_event, env_factory=synthetic_env_factory, num_envs=NUM_ENVS,
            steps_per_env=STEPS_PER_ENV, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            ring_buffer_shm_names=ring_buffer.name, write_ptr=ring_buffer._write_ptr,
            read_ptr=ring_buffer._read_ptr, actor_weight_sync_name=actor_weight_sync.name,
            actor_weight_param_shapes=actor_weight_param_shapes, device=DEVICE,
            metrics_queue=metrics_queue, max_rollouts=N_ITERS_TARGET * NUM_SLOTS * 4,
        ),
    )
    proc.start()

    staging_pool = RolloutStagingPool(
        capacity=NUM_SLOTS, num_envs=NUM_ENVS, slot_shapes=ring_buffer.slot_shapes, device=DEVICE,
    )

    orig_vtrace = learner_mod.vtrace_advantages
    learner_mod.vtrace_advantages = vtrace_impl

    per_iter = []
    collector_samples = []
    run_start = time.perf_counter()
    iteration = 0
    try:
        while iteration < N_ITERS_TARGET and (time.perf_counter() - run_start) < MAX_WALL_SECONDS_PER_ARM:
            iteration_start = time.perf_counter()
            wait_start = time.perf_counter()
            if not ring_buffer.wait_for_data(timeout=10.0):
                if not proc.is_alive():
                    raise RuntimeError(f"[{arm_name}] collector died before producing data")
                continue
            wait_time = time.perf_counter() - wait_start

            while not metrics_queue.empty():
                try:
                    collector_samples.append(metrics_queue.get_nowait())
                except Exception:
                    break

            num_new = ring_buffer.available()
            stage_start = time.perf_counter()
            for _ in range(num_new):
                staging_pool.stage_numpy_views(ring_buffer.read_numpy_views())
                ring_buffer.advance_read()
            stage_time = time.perf_counter() - stage_start

            sample_start = time.perf_counter()
            combined = staging_pool.batch()
            sample_time = time.perf_counter() - sample_start

            train_start = time.perf_counter()
            learner.process_batch(combined)
            learner.update(combined)
            train_time = time.perf_counter() - train_start

            weight_sync_start = time.perf_counter()
            actor_weight_sync.write_weights(learner.actor.state_dict())
            weight_sync_time = time.perf_counter() - weight_sync_start

            iteration_time = time.perf_counter() - iteration_start
            iteration += 1
            per_iter.append({
                "iter": iteration, "collector_wait_time_ms": wait_time * 1000.0,
                "stage_time_ms": stage_time * 1000.0, "sample_time_ms": sample_time * 1000.0,
                "train_time_ms": train_time * 1000.0, "weight_sync_time_ms": weight_sync_time * 1000.0,
                "iteration_time_ms": iteration_time * 1000.0, "rollouts_consumed": num_new,
            })
    finally:
        stop_event.set()
        proc.join(timeout=15.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        learner_mod.vtrace_advantages = orig_vtrace
        while not metrics_queue.empty():
            try:
                collector_samples.append(metrics_queue.get_nowait())
            except Exception:
                break
        ring_buffer.cleanup()
        actor_weight_sync.cleanup()

    # Collector throughput from the collector's OWN perspective: total env
    # steps produced / wall time spanned by its own timestamped samples.
    collector_throughput = None
    if len(collector_samples) >= 2:
        t0, t1 = collector_samples[0]["wall_time"], collector_samples[-1]["wall_time"]
        steps0, steps1 = collector_samples[0]["total_steps"], collector_samples[-1]["total_steps"]
        if t1 > t0:
            collector_throughput = (steps1 - steps0) / (t1 - t0)

    return {
        "arm": arm_name, "n_iters_completed": iteration,
        "wall_time_budget_exceeded": (time.perf_counter() - run_start) >= MAX_WALL_SECONDS_PER_ARM,
        "per_iter": per_iter, "collector_samples": collector_samples,
        "collector_throughput_env_steps_per_sec": collector_throughput,
    }


def main():
    metadata = capture_metadata(UNILAB_REPO_DIR)
    torch.manual_seed(0)
    seed_actor = BenchActor(OBS_DIM, ACTION_DIM)
    seed_critic = BenchCritic(OBS_DIM)
    actor_state = copy.deepcopy(seed_actor.state_dict())
    critic_state = copy.deepcopy(seed_critic.state_dict())

    results = {}
    for arm_name, vtrace_impl in [("numpy", unilab_vtrace_advantages), ("triton", triton_vtrace_advantages)]:
        print(f"=== Running arm: {arm_name} ===")
        results[arm_name] = run_arm(arm_name, vtrace_impl, actor_state, critic_state)
        n = results[arm_name]["n_iters_completed"]
        exceeded = results[arm_name]["wall_time_budget_exceeded"]
        thr = results[arm_name]["collector_throughput_env_steps_per_sec"]
        print(f"  completed {n} iterations{' (WALL TIME BUDGET EXCEEDED, partial)' if exceeded else ''}, "
              f"collector throughput={thr}")

    # Contamination-style check: do non-vtrace stages differ beyond threshold?
    non_vtrace_stages = ["collector_wait_time_ms", "stage_time_ms", "sample_time_ms", "weight_sync_time_ms"]
    max_rel_diff_pct, worst_stage = 0.0, None
    for stage in non_vtrace_stages:
        n_iters = min(len(results["numpy"]["per_iter"]), len(results["triton"]["per_iter"]))
        if n_iters == 0:
            continue
        numpy_mean = sum(r[stage] for r in results["numpy"]["per_iter"][:n_iters]) / n_iters
        triton_mean = sum(r[stage] for r in results["triton"]["per_iter"][:n_iters]) / n_iters
        if max(numpy_mean, triton_mean) < 1e-6:
            continue
        rel = 100.0 * abs(numpy_mean - triton_mean) / max(numpy_mean, triton_mean)
        if rel > max_rel_diff_pct:
            max_rel_diff_pct, worst_stage = rel, stage
    contamination_citable = max_rel_diff_pct <= CONTAMINATION_THRESHOLD_PCT
    print(f"\nContamination check (non-vtrace-adjacent stages, threshold {CONTAMINATION_THRESHOLD_PCT}% "
          f"-- chosen slightly looser than step4's 1% since these stages cross real IPC/process boundaries "
          f"with inherently more OS-scheduling jitter than a single-process synchronous call): "
          f"max diff {max_rel_diff_pct:.2f}% ({worst_stage}) -> {'PASS' if contamination_citable else 'FAIL'}")

    numpy_total = [r["iteration_time_ms"] for r in results["numpy"]["per_iter"]]
    triton_total = [r["iteration_time_ms"] for r in results["triton"]["per_iter"]]
    numpy_mean_total = sum(numpy_total) / len(numpy_total) if numpy_total else float("nan")
    triton_mean_total = sum(triton_total) / len(triton_total) if triton_total else float("nan")

    out = {
        "metadata": metadata,
        "architectural_note": "RolloutRingBuffer.signal_write_done() has no wait-for-space check; "
                               "a lagging reader is clamped forward, not blocking, by "
                               "_clamp_read_ptr_to_valid_window(). Collector-side SharedWeightSync reads "
                               "are unlocked. Predicts no ring-buffer-mediated stall path exists from "
                               "learner to collector; verified empirically below.",
        "config": {"device": DEVICE, "num_envs": NUM_ENVS, "steps_per_env": STEPS_PER_ENV,
                   "obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "num_slots": NUM_SLOTS,
                   "n_iters_target": N_ITERS_TARGET, "max_wall_seconds_per_arm": MAX_WALL_SECONDS_PER_ARM,
                   "note": "learner.py NOT modified; real RolloutRingBuffer/SharedWeightSync/"
                           "RolloutStagingPool used unmodified; collector runs as a genuine spawned "
                           "OS subprocess targeting the same CUDA device string as the learner"},
        "summary": {
            "numpy_mean_iteration_ms": numpy_mean_total, "triton_mean_iteration_ms": triton_mean_total,
            "numpy_collector_throughput_env_steps_per_sec": results["numpy"]["collector_throughput_env_steps_per_sec"],
            "triton_collector_throughput_env_steps_per_sec": results["triton"]["collector_throughput_env_steps_per_sec"],
            "numpy_n_iters": results["numpy"]["n_iters_completed"], "triton_n_iters": results["triton"]["n_iters_completed"],
        },
        "contamination_check": {"max_rel_diff_pct": max_rel_diff_pct, "worst_stage": worst_stage,
                                 "threshold_pct": CONTAMINATION_THRESHOLD_PCT, "passed": contamination_citable},
        "citable": contamination_citable,
        "results": results,
    }
    out_path = unique_path(OUT_DIR / f"unilab_appo_async_pipeline-{today()}.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
