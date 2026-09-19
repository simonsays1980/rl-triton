"""Phase 1 of 3 (torch/triton venv): rl-triton GAE kernel vs. a JAX
associative-scan GAE implementation. Thanks to Sasha Abramowitz for sharing
the JAX code -- see jax_gae_ext/gae_scan.py for the verbatim source.

================================================================================
WHY THIS IS SPLIT INTO THREE PHASES ACROSS TWO VENVS
================================================================================

This benchmark originally ran both kernels in one process (rl-triton via
torch/Triton, the JAX baseline moved on/off-device via dlpack, zero-copy).
That does not work in one environment: torch==2.4.1 (this repo's pinned,
validated version -- every existing report in this repo is stamped
`torch 2.4.1+cu124`) hard-pins `nvidia-cudnn-cu12==9.1.0.70` (exact, not a
floor), while jax[cuda12]'s GPU plugin (checked at 0.9.2 AND 0.10.2) requires
cuDNN >=9.8.0. Installing both in one venv leaves ONE of them with the wrong
cuDNN loaded. This is not cosmetic: when JAX gets the too-old cuDNN, XLA's
GPU compiler hard-crashes on ANY GPU program --

  jax.errors.JaxRuntimeError: INTERNAL: RET_CHECK failure
  (external/xla/xla/service/gpu/gpu_compiler.cc:2798) dnn_support != nullptr

-- even though this benchmark's actual computation (elementwise ops +
jax.lax.associative_scan) never calls a cuDNN op. Confirmed on a real pod:
XLA apparently initializes a cuDNN handle unconditionally while compiling any
GPU program, so the crash is unavoidable in a shared env regardless of
whether cuDNN is ever actually dispatched to. There is no jax[cuda12] release
compiled against cuDNN as old as 9.1.0.70 to route around this with a pin.

The fix: two separate venvs, one per framework, exchanging data as .npz
files under a shared results directory (default: benchmarks/jax_gae_results/).
Three phases, run in this order:

  1. THIS SCRIPT (torch/triton venv, `pip install -e ".[dev]"`): generates
     the test tensors, runs+times rl-triton's compute_gae, runs the
     Triton-side half of the equivalence gate (vs. an independent reference),
     and writes inputs (for phase 2 to consume) + Triton results (for phase 3
     to report) as .npz files.
  2. benchmark_gae_vs_jax_scan_jax.py (separate JAX venv, `pip install
     "jax[cuda12]"` -- NOT this repo's [jax] extra if that ever gets
     re-added to the main venv; a genuinely separate venv/virtualenv, not an
     extra alongside torch): reads the inputs THIS script wrote, runs+times
     the JAX associative-scan GAE, writes JAX results as .npz. Never imports
     torch.
  3. benchmark_gae_vs_jax_scan_report.py (back in the torch/triton venv):
     reads both phases' .npz outputs, runs the cross-implementation
     equivalence check, prints the monotonicity gate/markdown tables/
     crossover analysis, saves plots, writes the dated report -- all the
     same output this benchmark always produced, just assembled from two
     phases' saved results instead of computed live in one process.

================================================================================
STEP 0 -- SEMANTIC EQUIVALENCE (read this before trusting any number)
================================================================================

Both sides compute the SAME standard backward GAE recurrence
  A[t] = delta[t] + decay[t]*A[t+1]
but over different buffer conventions:

  rl-triton (compute_gae, no truncations, default zero bootstrap), length-T
  buffers, axis=1 ([num_envs, seq_len]):
    delta[t] = R[t] + gamma*(1 - terminated[t])*V[t+1] - V[t]   (V[T] := last_value, default 0)
    decay[t] = gamma*lambda*(1 - terminated[t])
    A[t] = delta[t] + decay[t]*A[t+1],  A[T] := 0

  JAX baseline (generalised_advantages), length-(T+1) buffers, axis=1:
    leaving  = slice[0:T]      (the "departing" slot -- reward/value consumed as-is)
    landing  = slice[1:T+1]    (the "arriving" slot -- next value + this step's flags)
    bootstrap[t] = 1 - terminated[t+1]
    continues[t] = bootstrap[t] * (1 - truncated[t+1])
    residual[t]  = reward[t] + gamma*bootstrap[t]*value[t+1] - value[t]
    decay[t]     = gamma*lambda*continues[t]
    A[t] = residual[t] + decay[t]*A[t+1]   for t in [0, T-1)  -- output length T

Index mapping (derived by reading both implementations, then verified
numerically against an independent third reference -- see
`_ref_gae_matched_window` below): feed the JAX function T+1 slots per side,
where slots 0..T-1 are rl-triton's native length-T buffers verbatim, and
slot T is the explicit bootstrap slot:

  reward_jax[0:T]    = rewards            reward_jax[T]    = 0  (unused: no T-th residual)
  value_jax[0:T]     = values             value_jax[T]     = last_value (default 0)
  terminated_jax[0]  = 0 (unused)         terminated_jax[1:T+1] = terminateds[0:T]
  truncated_jax[0]   = 0 (unused)         truncated_jax[1:T+1]  = truncateds[0:T]  (all-zero if no truncations)

This is the SAME "+1 shift" shape as PufferLib's own buffer convention (see
benchmark_gae_vs_pufferlib.py's capability-difference #1). Unlike PufferLib,
JAX's length-(T+1) buffer means BOTH sides produce a genuine length-T output
here -- no PufferLib-style "row T-1 is structurally uncomputable" capability
gap.

Truncation / bootstrap_values are NOT exercised by this benchmark's
equivalence gate, despite JAX's `generalised_advantages` taking `truncated`
as a first-class argument: JAX has no `bootstrap_values` injection mechanism
at all -- a truncated (or simply boundary) step's landing value is always
`value[t+1]` verbatim (0 for the appended slot), with no way to override it
with a true continuation value. This means ANY nonzero bootstrap value
anywhere -- even just rl-triton's ordinary default boundary bootstrap at
t=T-1 -- changes rl-triton's last column relative to JAX's, and that
difference then propagates backward through EVERY earlier column via the
scan's carry (verified empirically while building this benchmark: masking
out only the directly-affected column is not sufficient -- the whole trace
diverges). There is therefore no shape/config where a partial-column
comparison would be meaningful once bootstrap_values is nonzero anywhere, so
this benchmark -- like the PufferLib comparison, which excludes rl-triton's
truncation path for a different structural reason -- covers only rl-triton's
default `HAS_TRUNCATIONS=False`, zero-bootstrap path (`last_value` and
`bootstrap_values` both `None`), where both sides' inputs are equivalent
under the shift above and produce a genuine, directly-comparable length-T
output.

Numerical agreement check: `_ref_gae_matched_window`, a reference
implementation independent of both (plain PyTorch, same delta/decay formula,
zero carry seeded at A[T]=0), verified against rl-triton directly (bit
identical convention, no shift needed) here in phase 1; the JAX side of the
same check (JAX's output vs. the T+1-shifted inputs this phase writes, plus
the direct Triton-vs-JAX diff) happens in phase 3, once phase 2's results
exist to compare.

================================================================================
TWO REGIMES (same shapes as benchmark_gae_vs_pufferlib.py, for comparability)
================================================================================

  1. "Production" regime: seq_len in [128..4096], num_envs in [128..8192].
  2. "Massively-parallel-sim" regime: seq_len in [8..128], num_envs in
     [4096..32768] -- Isaac Gym/Isaac Lab-style aspect ratio.

================================================================================
Usage:
  pip install -e ".[dev]"
  python benchmarks/benchmark_gae_vs_jax_scan_triton.py [--results-dir DIR]
Then, in the separate JAX venv:
  python benchmarks/benchmark_gae_vs_jax_scan_jax.py [--results-dir DIR]
Then, back in this venv:
  python benchmarks/benchmark_gae_vs_jax_scan_report.py [--results-dir DIR]
(--results-dir defaults to benchmarks/jax_gae_results/ and must match across
all three invocations.)
================================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))  # this dir -- for jax_gae_ext
from jax_gae_ext.config import (
    GAMMA, LAMBDA, TERM_PROB, SEED, EQUIVALENCE_CASES,
    N_ITER, N_TRIALS, N_AMORTIZED_CALLS, N_PROFILE_ITER,
    all_sweep_cells, cell_key,
)
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))  # for bench_utils
from bench_utils import _warmup_gpu

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rl_triton.ops.gae import compute_gae

import triton
from torch.profiler import ProfilerActivity, profile

DEFAULT_RESULTS_DIR = Path(__file__).parent / "jax_gae_results"

# Exact versions this benchmark has actually been run against (matches
# pyproject.toml's exact-pinned dependencies -- torch==2.4.1, triton==3.0.0
# -- and every report file's own provenance stamp). A fresh
# `pip install -e ".[dev]"` should land here exactly; if it doesn't, either
# pyproject.toml has been updated without this constant (update both
# together) or something upgraded torch/triton out from under the venv
# (see NOTES.md's "torch==2.4.1 and jax[cuda12] cannot share one venv"
# section for a real instance of this happening via pip's resolver).
# CUDA build tag (+cu121, +cu124, ...) is NOT pinned here -- confirmed
# backward-compatible across recent driver versions (e.g. a +cu121 torch
# wheel ran fine against a CUDA 13.2 driver on a real pod), so only the
# torch/triton version numbers themselves are checked, not the CUDA suffix.
KNOWN_GOOD_TORCH_VERSION = "2.4.1"
KNOWN_GOOD_TRITON_VERSION = "3.0.0"


def _check_known_good_versions():
    torch_base = torch.__version__.split("+")[0]
    if torch_base != KNOWN_GOOD_TORCH_VERSION:
        print(f"WARNING: torch {torch.__version__} != known-good "
              f"{KNOWN_GOOD_TORCH_VERSION} -- this benchmark's numbers, and every "
              f"other report in this repo, were produced against torch=="
              f"{KNOWN_GOOD_TORCH_VERSION}. Results may still be valid but are "
              f"unverified at this version.")
    if triton.__version__ != KNOWN_GOOD_TRITON_VERSION:
        print(f"WARNING: triton {triton.__version__} != known-good "
              f"{KNOWN_GOOD_TRITON_VERSION} -- same caveat as above.")


# ------------------------------------------------------------------------
# Shared input generation (also mirrored, numpy-only, in the JAX-phase
# script -- kept in sync by both reading the same seeds/shapes from
# jax_gae_ext.config, not by sharing code across venvs).
# ------------------------------------------------------------------------

def _make_rl_triton_inputs(num_envs, seq_len, device, seed=SEED):
    g = torch.Generator(device=device).manual_seed(seed)
    rewards = torch.randn(num_envs, seq_len, device=device, generator=g).contiguous()
    values = torch.randn(num_envs, seq_len, device=device, generator=g).contiguous()
    terminateds = (torch.rand(num_envs, seq_len, device=device, generator=g) < TERM_PROB).float().contiguous()
    return rewards, values, terminateds


def _ref_gae_matched_window(rewards, values, terminateds, gamma, lambda_):
    """Independent (non-Triton, non-JAX) reference: standard GAE delta/decay,
    zero carry seeded at A[T]=0 -- rl-triton's own native convention, so no
    index shift is needed to compare against rl-triton directly."""
    num_envs, seq_len = rewards.shape
    device = rewards.device
    out = torch.zeros(num_envs, seq_len, device=device, dtype=rewards.dtype)
    carry = torch.zeros(num_envs, device=device, dtype=rewards.dtype)
    for t in reversed(range(seq_len)):
        v_next = values[:, t + 1] if t < seq_len - 1 else torch.zeros(num_envs, device=device, dtype=rewards.dtype)
        not_term = 1.0 - terminateds[:, t]
        delta = rewards[:, t] + gamma * not_term * v_next - values[:, t]
        carry = delta + gamma * lambda_ * not_term * carry
        out[:, t] = carry
    return out


def run_triton_equivalence_gate(results_dir, device="cuda"):
    print("=" * 88)
    print("STEP 0a: TRITON-SIDE EQUIVALENCE GATE (rl-triton vs. independent reference)")
    print("=" * 88)
    print("Checks rl-triton's compute_gae against a from-scratch PyTorch reference using")
    print("the identical delta/decay formula, seeded independently of both. This is only")
    print("half the equivalence proof -- the JAX-side half (JAX vs. the same T+1-shifted")
    print("reference, plus the direct Triton-vs-JAX diff) runs in phase 3, once phase 2's")
    print("JAX results exist. This phase writes each case's inputs + rl-triton's own output")
    print("to results_dir/equivalence/ so phase 2 can compute JAX's output against the SAME")
    print("inputs, and phase 3 can complete the comparison. See this script's module")
    print("docstring for the full index-mapping derivation and why the check is split this way.")
    print()

    equivalence_dir = results_dir / "equivalence"
    equivalence_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for num_envs, seq_len, seed in EQUIVALENCE_CASES:
        rewards, values, terminateds = _make_rl_triton_inputs(num_envs, seq_len, device, seed)
        ref = _ref_gae_matched_window(rewards, values, terminateds, GAMMA, LAMBDA)
        full = compute_gae(rewards, values, terminateds, gamma=GAMMA, lambda_=LAMBDA)
        max_diff = (ref - full).abs().max().item()
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        if max_diff >= 1e-3:
            all_ok = False
        print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  "
              f"max|Triton - ref| = {max_diff:.3e}   [{status}]")

        key = cell_key(num_envs, seq_len) + f"_eq{seed}"
        np.savez(
            equivalence_dir / f"{key}_triton.npz",
            rewards=rewards.cpu().numpy(), values=values.cpu().numpy(),
            terminateds=terminateds.cpu().numpy(), advantages=full.cpu().numpy(),
            num_envs=num_envs, seq_len=seq_len, seed=seed,
        )

    print()
    if all_ok:
        print("RESULT: rl-triton's output matches the independent reference within float32 "
              "tolerance on every case.")
    else:
        print("RESULT: FAIL. Do not trust the timing comparison below.")
        sys.exit(1)
    print()
    return all_ok


# ------------------------------------------------------------------------
# Timing harness (unchanged from the single-process version)
# ------------------------------------------------------------------------

def _bench_gpu(fn, n_iter=N_ITER, n_trials=N_TRIALS):
    trial_medians = []
    for _ in range(n_trials):
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        trial_medians.append(times[len(times) // 2])
    return min(trial_medians)


def _bench_gpu_amortized(fn, n_calls=N_AMORTIZED_CALLS, n_trials=N_TRIALS):
    per_trial = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_calls):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_trial.append(start.elapsed_time(end) / n_calls)
    return min(per_trial)


def _device_profile(fn, n_iter=N_PROFILE_ITER):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_device_us = sum(e.self_device_time_total for e in events)
    launches = sum(e.count for e in events)
    return total_device_us / n_iter, launches / n_iter


# ------------------------------------------------------------------------
# Sweep: run+time Triton, write inputs + Triton results to disk
# ------------------------------------------------------------------------

def run_sweep(results_dir, device="cuda"):
    inputs_dir = results_dir / "inputs"
    triton_dir = results_dir / "triton"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("SWEEP (Triton side): writing inputs + Triton results to disk")
    print("=" * 88)

    for num_envs, seq_len, regime in all_sweep_cells():
        rewards, values, terminateds = _make_rl_triton_inputs(num_envs, seq_len, device)

        def triton_call(rewards=rewards, values=values, terminateds=terminateds):
            return compute_gae(rewards, values, terminateds, gamma=GAMMA, lambda_=LAMBDA)

        _warmup_gpu(triton_call, n_warmup=15)

        triton_ms = _bench_gpu(triton_call, n_iter=N_ITER, n_trials=N_TRIALS)
        triton_ms_amort = _bench_gpu_amortized(triton_call)
        triton_dev_us, triton_launches = _device_profile(triton_call)
        triton_out = triton_call()

        triton_bytes = 4 * num_envs * seq_len * 4  # rewards, values, terminateds reads + out write
        triton_gbps = triton_bytes / (triton_ms / 1000) / 1e9

        key = cell_key(num_envs, seq_len)
        np.savez(
            inputs_dir / f"{key}.npz",
            rewards=rewards.cpu().numpy(),
            values=values.cpu().numpy(),
            terminateds=terminateds.cpu().numpy(),
            num_envs=num_envs, seq_len=seq_len,
        )
        np.savez(
            triton_dir / f"{key}.npz",
            advantages=triton_out.cpu().numpy(),
            triton_ms=triton_ms, triton_ms_amort=triton_ms_amort,
            triton_dev_us=triton_dev_us, triton_launches=triton_launches,
            triton_gbps=triton_gbps, num_envs=num_envs, seq_len=seq_len,
        )
        print(f"  [{regime}] num_envs={num_envs:>5} seq_len={seq_len:>5}  "
              f"triton={triton_ms:>8.4f}ms ({triton_gbps:>7.1f} GB/s)")
    print()
    print(f"Wrote inputs to {inputs_dir}/ and Triton results to {triton_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 88)
    print("PHASE 1/3: rl-triton GAE kernel (torch/triton venv)")
    print("=" * 88)
    print(f"GPU:            {gpu_name}")
    print(f"torch:          {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"triton:         {triton.__version__}")
    print(f"dtype:          float32")
    print(f"gamma={GAMMA}  lambda={LAMBDA}  termination_prob={TERM_PROB}")
    print(f"results dir:    {args.results_dir}")
    print()
    _check_known_good_versions()
    print()

    run_triton_equivalence_gate(args.results_dir, device)
    run_sweep(args.results_dir, device)

    meta_dir = args.results_dir
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        meta_dir / "triton_meta.npz",
        gpu_name=gpu_name, torch_version=torch.__version__,
        torch_cuda=torch.version.cuda or "", triton_version=triton.__version__,
    )

    print("=" * 88)
    print(f"PHASE 1 COMPLETE. Next: run benchmark_gae_vs_jax_scan_jax.py in the JAX venv,")
    print(f"pointed at the same --results-dir ({args.results_dir}).")
    print("=" * 88)


if __name__ == "__main__":
    main()
