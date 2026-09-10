"""Step 2: standalone V-trace timing, UniLab's real vtrace_advantages() vs. our
Triton kernel, at UniLab's real production shapes.

unilab_numpy arm calls UniLab's actual vtrace_advantages() as-is (not a
reimplementation) -- GPU-resident inputs in, internal .cpu().numpy() backward
loop, GPU tensor out. triton arm goes through the same [T,N]->[N,T] transpose
wrapper verified correct in Step 1. Transpose cost is reported as its own
line item, not hidden inside either arm.

Protocol: 20 warmup, 5 trials x 100 iters, min-of-trial-medians (matches
resip_ppo_e2e_measurement.py's methodology). Both CUDA-event and synchronized
wall-clock (perf_counter) timings are reported -- never blended into one
speedup number, since the CPU round-trip arm may behave differently under
the two clocks.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    bench_gpu_cuda_events, bench_gpu_wall_clock, capture_metadata,
    today, unique_path, warmup_gpu,
)
from step1_correctness import make_inputs, run_unilab, run_triton  # noqa: E402

from rl_triton.ops.vtrace import compute_vtrace  # noqa: E402

UNILAB_REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path(__file__).parent.parent  # benchmarks/
ATOL = RTOL = 1e-4
N_WARMUP, N_TRIALS, N_ITER = 20, 5, 100

# (T, N): two real production configs + 2 longer-T scaling points at the same N.
SHAPES = [(24, 512), (24, 1024), (64, 512), (128, 512)]


def transpose_wrapper_cost(behavior_log_probs, target_log_probs, rewards, values, dones):
    """Isolated cost of the [T,N] -> [N,T] transpose+contiguous+float the Triton
    wrapper does on every call, measured as its own line item."""
    def _do():
        t = lambda x: x.transpose(0, 1).contiguous().float()
        return t(target_log_probs), t(behavior_log_probs), t(values), t(rewards), t(dones)
    return _do


def main():
    metadata = capture_metadata(UNILAB_REPO_DIR)
    gamma, clip_rho, clip_c = 0.99, 1.0, 1.0
    rows = []

    # Which shapes are citable, per Step 1's gate. Multiple runs may exist
    # (unique_path appends -run2, -run3, ...) -- use whichever candidate
    # actually covers all of this script's SHAPES, preferring the most
    # recently modified one if several do.
    candidates = sorted(
        OUT_DIR.glob(f"unilab_vtrace_correctness-{today()}*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    correctness_path = None
    passed_shapes = set()
    for cand in candidates:
        data = json.loads(cand.read_text())
        shapes_in_file = {(r["T"], r["N"]) for r in data["records"]}
        if set(SHAPES).issubset(shapes_in_file):
            correctness_path = cand
            passed_shapes = {(r["T"], r["N"]) for r in data["records"] if r["pass"]}
            break
    if correctness_path is None:
        raise RuntimeError(
            f"No unilab_vtrace_correctness-{today()}*.json in {OUT_DIR} covers all of "
            f"SHAPES={SHAPES}. Run step1_correctness.py first with these shapes included."
        )
    print(f"Loaded correctness gate from {correctness_path.name}: {len(passed_shapes)} passing shapes")

    for (T, N) in SHAPES:
        behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones = \
            make_inputs(T, N, gamma, clip_rho, clip_c)

        # Correctness at this exact shape (re-derive max/mean/rmse for the row,
        # gate boolean comes from Step 1's persisted record).
        vs_unilab, adv_unilab = run_unilab(
            behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
            gamma, clip_rho, clip_c,
        )
        vs_triton, adv_triton = run_triton(
            behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
            gamma, clip_rho, clip_c,
        )
        from common import assert_finite_and_close
        vs_pass, vs_max, vs_mean, vs_rmse = assert_finite_and_close(vs_triton, vs_unilab, "vs", ATOL, RTOL)
        adv_pass, adv_max, adv_mean, adv_rmse = assert_finite_and_close(adv_triton, adv_unilab, "adv", ATOL, RTOL)
        shape_citable = (T, N) in passed_shapes and vs_pass and adv_pass

        # --- timing: unilab_numpy arm ---
        def unilab_fn():
            return run_unilab(
                behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
                gamma, clip_rho, clip_c,
            )
        warmup_gpu(unilab_fn, n_warmup=N_WARMUP)
        unilab_cuda_ms, unilab_cuda_trials = bench_gpu_cuda_events(unilab_fn, n_iter=N_ITER, n_trials=N_TRIALS)
        unilab_wall_ms, unilab_wall_trials = bench_gpu_wall_clock(unilab_fn, n_iter=N_ITER, n_trials=N_TRIALS)

        # --- timing: triton arm ---
        def triton_fn():
            return run_triton(
                behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
                gamma, clip_rho, clip_c,
            )
        warmup_gpu(triton_fn, n_warmup=N_WARMUP)
        triton_cuda_ms, triton_cuda_trials = bench_gpu_cuda_events(triton_fn, n_iter=N_ITER, n_trials=N_TRIALS)
        triton_wall_ms, triton_wall_trials = bench_gpu_wall_clock(triton_fn, n_iter=N_ITER, n_trials=N_TRIALS)

        # --- timing: transpose cost alone ---
        transpose_fn = transpose_wrapper_cost(behavior_log_probs, target_log_probs, rewards, values, dones)
        warmup_gpu(transpose_fn, n_warmup=N_WARMUP)
        transpose_cuda_ms, transpose_trials = bench_gpu_cuda_events(transpose_fn, n_iter=N_ITER, n_trials=N_TRIALS)

        num_transitions = T * N
        for impl, cuda_ms, cuda_trials, wall_ms, wall_trials in [
            ("unilab_numpy", unilab_cuda_ms, unilab_cuda_trials, unilab_wall_ms, unilab_wall_trials),
            ("triton", triton_cuda_ms, triton_cuda_trials, triton_wall_ms, triton_wall_trials),
        ]:
            cuda_speedup = unilab_cuda_ms / cuda_ms
            wall_speedup = unilab_wall_ms / wall_ms
            row = {
                "T": T, "N": N, "num_transitions": num_transitions, "implementation": impl,
                "cuda_latency_us": cuda_ms * 1000.0,
                "cuda_speedup_vs_unilab": cuda_speedup,
                "cuda_transitions_per_sec": num_transitions / (cuda_ms / 1000.0),
                "wall_latency_us": wall_ms * 1000.0,
                "wall_speedup_vs_unilab": wall_speedup,
                "wall_transitions_per_sec": num_transitions / (wall_ms / 1000.0),
                "transpose_cost_us": transpose_cuda_ms * 1000.0,
                "vs_max_abs_err": vs_max, "vs_mean_abs_err": vs_mean, "vs_rmse": vs_rmse,
                "adv_max_abs_err": adv_max, "adv_mean_abs_err": adv_mean, "adv_rmse": adv_rmse,
                "citable": shape_citable,
                "trial_medians_cuda_ms": cuda_trials,
                "trial_medians_wall_ms": wall_trials,
            }
            rows.append(row)
            print(f"T={T:4d} N={N:4d} {impl:13s}  cuda={cuda_ms*1000:9.2f}us (x{cuda_speedup:.3f})  "
                  f"wall={wall_ms*1000:9.2f}us (x{wall_speedup:.3f})  "
                  f"trans/s={num_transitions/(cuda_ms/1000.0):.3e}  citable={shape_citable}")
        print(f"  transpose_cost: {transpose_cuda_ms*1000:.2f}us")

    out = {
        "metadata": metadata,
        "gamma": gamma, "clip_rho": clip_rho, "clip_c": clip_c,
        "protocol": {"n_warmup": N_WARMUP, "n_trials": N_TRIALS, "n_iter": N_ITER,
                     "reducer": "min-of-trial-medians"},
        "rows": rows,
        "concatenation_ablation": {"skipped": True,
                                    "reason": "populated by step3_ablation.py if run"},
    }
    out_path = unique_path(OUT_DIR / f"unilab_vtrace_standalone-{today()}.json")
    out_path.write_text(json.dumps(out, indent=2))
    csv_path = out_path.with_suffix(".csv")
    cols = ["T", "N", "num_transitions", "implementation", "cuda_latency_us", "cuda_speedup_vs_unilab",
            "cuda_transitions_per_sec", "wall_latency_us", "wall_speedup_vs_unilab", "wall_transitions_per_sec",
            "transpose_cost_us", "vs_max_abs_err", "vs_mean_abs_err", "vs_rmse",
            "adv_max_abs_err", "adv_mean_abs_err", "adv_rmse", "citable"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
