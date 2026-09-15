"""Phase 2 of 3 (separate JAX venv, NO torch import): reads the inputs
benchmark_gae_vs_jax_scan_triton.py wrote, runs+times a JAX associative-scan
GAE implementation, writes JAX results as .npz.

See benchmark_gae_vs_jax_scan_triton.py's module docstring for why this
benchmark is split into three phases across two venvs (torch's exact
`nvidia-cudnn-cu12==9.1.0.70` pin is incompatible with the cuDNN version
jax[cuda12]'s GPU plugin requires -- confirmed on a real pod: sharing one
venv makes XLA's GPU compiler hard-crash with a null cuDNN handle on ANY
GPU program, not just cuDNN-using ones).

This script deliberately does NOT import torch, matplotlib, or triton --
only numpy (for .npz I/O) and jax. Run it in a venv with ONLY `jax[cuda12]`
installed (a genuinely separate virtualenv, not this repo's `pip install
-e ".[jax]"` alongside torch -- installing jax[cuda12] into the same venv as
this repo's pinned torch==2.4.1 is exactly the conflict this three-phase
split exists to avoid). numpy is required (`pip install numpy`) since jax's
own install does not always guarantee a compatible numpy is present.

Timing methodology differs from the Triton phase's CUDA-event-based
_bench_gpu: torch.cuda.Event/torch.profiler both require torch, which this
script cannot import. Instead this uses wall-clock time.perf_counter() around
`jax_call().block_until_ready()`, same min-of-medians-across-trials structure
as the Triton side's _bench_gpu, just host-side timing instead of device
events -- slightly less precise (includes some Python/dispatch overhead) but
needs no extra dependency, and JAX's own device work still dominates at
these problem sizes. Device-only kernel time and launch counts (the Triton
phase's torch.profiler-based _device_profile) have NO equivalent here: JAX
has no simple "give me total device microseconds" API comparable to
torch.profiler.key_averages() (jax.profiler.trace() writes a raw
TensorBoard-format trace that would need parsing, meaningfully more code and
more fragile across JAX versions) -- so this phase reports wall-clock timing
only. The report phase's markdown table and analysis reflect this: JAX's
dev_us/launches columns are omitted, not synthesized.

================================================================================
Usage:
  # in the JAX venv:
  pip install "jax[cuda12]" numpy
  python benchmarks/benchmark_gae_vs_jax_scan_jax.py [--results-dir DIR]
(--results-dir defaults to benchmarks/jax_gae_results/ and must match the
Triton phase's --results-dir, and match what you later pass to
benchmark_gae_vs_jax_scan_report.py.)
================================================================================
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # this dir -- for jax_gae_ext
from jax_gae_ext.gae_scan import generalised_advantages
from jax_gae_ext.config import (
    GAMMA, LAMBDA, N_ITER, N_TRIALS, N_AMORTIZED_CALLS,
    EQUIVALENCE_CASES, cell_key,
)

import jax
import jax.numpy as jnp

DEFAULT_RESULTS_DIR = Path(__file__).parent / "jax_gae_results"


def _to_jax_inputs(rewards, values, terminateds):
    """Append the length-(T+1) bootstrap slot and shift terminated/truncated
    by +1 (see benchmark_gae_vs_jax_scan_triton.py's module docstring for the
    index-mapping derivation). numpy arrays in, jax arrays out -- no torch,
    no dlpack (the two phases exchange data via .npz on disk, not in-process)."""
    num_envs, seq_len = rewards.shape
    zeros_col = np.zeros((num_envs, 1), dtype=rewards.dtype)
    reward_j = np.concatenate([rewards, zeros_col], axis=1)
    value_j = np.concatenate([values, zeros_col], axis=1)
    term_j = np.concatenate([zeros_col, terminateds], axis=1)
    trunc_j = np.zeros((num_envs, seq_len + 1), dtype=rewards.dtype)
    return (
        jnp.asarray(reward_j), jnp.asarray(value_j),
        jnp.asarray(term_j), jnp.asarray(trunc_j),
    )


def _bench_jax(fn, n_iter=N_ITER, n_trials=N_TRIALS):
    """Wall-clock timing, min-of-medians across n_trials. See module
    docstring for why this differs from the Triton phase's CUDA-event timing."""
    trial_medians = []
    for _ in range(n_trials):
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            fn().block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        trial_medians.append(times[len(times) // 2])
    return min(trial_medians)


def _bench_jax_amortized(fn, n_calls=N_AMORTIZED_CALLS, n_trials=N_TRIALS):
    """N calls dispatched back-to-back, only the LAST result blocked-on --
    XLA calls dispatch asynchronously, so this measures true dispatch
    throughput, not N independent round trips. Per-call ms, min across trials."""
    per_trial = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = None
        for _ in range(n_calls):
            result = fn()
        result.block_until_ready()
        per_trial.append((time.perf_counter() - t0) * 1000 / n_calls)
    return min(per_trial)


def _warmup(fn, n_warmup=15):
    for _ in range(n_warmup):
        fn().block_until_ready()


def run_jax_equivalence_gate(results_dir, jax_gae_fn):
    """Reads each equivalence-gate case's inputs + rl-triton output (written
    by phase 1's run_triton_equivalence_gate), runs the SAME inputs through
    JAX, and writes JAX's output back to results_dir/equivalence/ for phase 3
    to complete the comparison (JAX vs. independent reference, and the direct
    Triton-vs-JAX diff -- neither needs jax at all, so they run in phase 3)."""
    print("=" * 88)
    print("STEP 0b (JAX half): running JAX on the SAME equivalence-gate inputs phase 1 used")
    print("=" * 88)
    print("The actual pass/fail check happens in phase 3 (no torch/triton needed there,")
    print("no jax needed here beyond running the function) -- this just produces JAX's")
    print("output against phase 1's exact inputs so phase 3 has both sides to compare.")
    print()

    equivalence_dir = results_dir / "equivalence"
    if not equivalence_dir.exists():
        print(f"ERROR: {equivalence_dir}/ not found. Run benchmark_gae_vs_jax_scan_triton.py "
              "first, with the same --results-dir.")
        sys.exit(1)

    for num_envs, seq_len, seed in EQUIVALENCE_CASES:
        key = cell_key(num_envs, seq_len) + f"_eq{seed}"
        triton_path = equivalence_dir / f"{key}_triton.npz"
        if not triton_path.exists():
            print(f"ERROR: missing {triton_path}. Run benchmark_gae_vs_jax_scan_triton.py "
                  "first, with the same --results-dir.")
            sys.exit(1)

        data = np.load(triton_path)
        rewards, values, terminateds = data["rewards"], data["values"], data["terminateds"]
        rj, vj, tj, trj = _to_jax_inputs(rewards, values, terminateds)
        jax_adv = np.asarray(jax_gae_fn(rj, vj, tj, trj))

        np.savez(
            equivalence_dir / f"{key}_jax.npz",
            advantages=jax_adv, num_envs=num_envs, seq_len=seq_len, seed=seed,
        )
        print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  JAX output written")
    print()


def run_sweep(results_dir, jax_gae_fn):
    inputs_dir = results_dir / "inputs"
    jax_dir = results_dir / "jax"
    jax_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(inputs_dir.glob("*.npz"))
    if not input_files:
        print(f"ERROR: no input files found in {inputs_dir}/. Run "
              f"benchmark_gae_vs_jax_scan_triton.py first (same --results-dir).")
        sys.exit(1)

    print("=" * 88)
    print(f"SWEEP (JAX side): reading {len(input_files)} input cells from {inputs_dir}/")
    print("=" * 88)

    for input_path in input_files:
        data = np.load(input_path)
        rewards, values, terminateds = data["rewards"], data["values"], data["terminateds"]
        num_envs, seq_len = int(data["num_envs"]), int(data["seq_len"])

        rj, vj, tj, trj = _to_jax_inputs(rewards, values, terminateds)

        def jax_call(rj=rj, vj=vj, tj=tj, trj=trj):
            return jax_gae_fn(rj, vj, tj, trj)

        _warmup(jax_call, n_warmup=15)

        jax_ms = _bench_jax(jax_call, n_iter=N_ITER, n_trials=N_TRIALS)
        jax_ms_amort = _bench_jax_amortized(jax_call)
        jax_out = np.asarray(jax_call())

        # reward, value (both T+1 slots) + terminated, truncated (T+1 slots,
        # though truncated is all-zero -- still materialized/read) reads +
        # length-T output write. Conservatively counts the full T+1 buffers
        # actually passed in, not just the T "used" elements.
        jax_bytes = 4 * num_envs * (seq_len + 1) * 4 + num_envs * seq_len * 4
        jax_gbps = jax_bytes / (jax_ms / 1000) / 1e9

        np.savez(
            jax_dir / input_path.name,
            advantages=jax_out,
            jax_ms=jax_ms, jax_ms_amort=jax_ms_amort, jax_gbps=jax_gbps,
            num_envs=num_envs, seq_len=seq_len,
        )
        print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  "
              f"jax={jax_ms:>8.4f}ms ({jax_gbps:>7.1f} GB/s)")
    print()
    print(f"Wrote JAX results to {jax_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    jax_devices = jax.devices()
    jax_gpu = [d for d in jax_devices if d.platform == "gpu"]
    if not jax_gpu:
        print("ERROR: jax.devices() found no GPU backend "
              f"(jax.devices()={jax_devices}, jax.default_backend()={jax.default_backend()}).")
        print("This benchmark compares GPU kernels -- benchmarking JAX-on-CPU against")
        print("Triton-on-GPU would not be a meaningful comparison, so it refuses to run.")
        print('Install a CUDA-enabled jaxlib in THIS venv, e.g.: pip install -U "jax[cuda12]"')
        sys.exit(1)

    results_dir = args.results_dir
    inputs_dir = results_dir / "inputs"
    if not inputs_dir.exists():
        print(f"ERROR: {inputs_dir}/ does not exist. Run "
              f"benchmark_gae_vs_jax_scan_triton.py first, with the same --results-dir.")
        sys.exit(1)

    jax_gae_fn = jax.jit(
        lambda r, v, te, tr: generalised_advantages(r, v, te, tr, GAMMA, LAMBDA, axis=1)
    )

    print("=" * 88)
    print("PHASE 2/3: JAX associative-scan GAE (separate JAX venv, no torch)")
    print("=" * 88)
    print(f"jax:            {jax.__version__}  (backend: {jax.default_backend()}, device: {jax_gpu[0]})")
    print(f"dtype:          float32")
    print(f"gamma={GAMMA}  lambda={LAMBDA}")
    print(f"results dir:    {results_dir}")
    print()

    run_jax_equivalence_gate(results_dir, jax_gae_fn)
    run_sweep(results_dir, jax_gae_fn)

    np.savez(
        results_dir / "jax_meta.npz",
        jax_version=jax.__version__, jax_backend=jax.default_backend(),
        jax_device=str(jax_gpu[0]),
    )

    print("=" * 88)
    print(f"PHASE 2 COMPLETE. Next: switch back to the torch/triton venv and run")
    print(f"benchmark_gae_vs_jax_scan_report.py, pointed at the same --results-dir "
          f"({results_dir}).")
    print("=" * 88)


if __name__ == "__main__":
    main()
