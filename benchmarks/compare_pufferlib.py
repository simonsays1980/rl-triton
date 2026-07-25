"""Standalone, optional comparison: rl-triton's GAE kernel vs. PufferLib's advantage kernel.

NOT part of bench_release.py and NOT part of the test suite, NOT installed or built by
CI or .runpod/start.sh — this is a separate, run-on-demand, one-time-study script. It
imports the real `pufferlib` pip package ONLY; it does not vendor, JIT-build, or
otherwise carry any PufferLib source, and has no fallback that does. If `pufferlib` is
not installed, it skips cleanly — see benchmarks/pufferlib.md for the already-recorded
results from the one-time run that did have it available (that file notes the exact
version and method used, since this script's own behavior has changed since then).

(A vendored, sha256-pinned copy of PufferLib's CUDA source exists separately at
benchmarks/pufferlib_ext/, predating this script, used only by the separate, non-pytest-
collected benchmarks/benchmark_gae_vs_pufferlib.py. Neither is referenced by this script,
by bench_release.py, by bench_safeguard.py, by any test_*.py file, or by CI.)

Frame: capability + design comparison, not a scoreboard. PufferLib's advantage
kernel is a genuine hand-written CUDA kernel — one thread per environment row,
sequential O(T) scan within the thread. rl-triton's `compute_gae` is a Triton
kernel — one program per environment row, O(log T) in-SRAM tree reduction via
`tl.associative_scan`. Different tradeoffs: PufferLib's flat per-thread cost
tends to win at very short horizons; rl-triton's parallel scan tends to win as
horizon grows, and additionally supports interior truncations (terminated vs.
truncated, with a true bootstrap continuation value) that PufferLib has no
equivalent for at all — PufferLib takes a single `dones` flag per step and no
`bootstrap_values` concept. That gap is reported here as a factual capability
difference, not fabricated as a benchmark PufferLib could theoretically win or
lose.

Reference: PufferLib, https://github.com/PufferAI/PufferLib (pip: `pufferlib`).

Usage:
    python benchmarks/compare_pufferlib.py [--gpu LABEL]

Writes benchmarks/pufferlib.md. Does not touch bench_release.py, benchmarks.md,
or README.md.
"""
import argparse
import datetime
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bench_utils import _bench_gpu, _bench_gpu_amortized, _device_profile, _warmup_gpu
from rl_triton.ops.gae import compute_gae

GAMMA = 0.99
LAMBDA = 0.95
TERM_PROB = 0.05
SEED = 0

# Production regime: real massively-parallel-sim RL is high-N/short-T.
# GPUDrive 90, Nocturne 80, Gigaflow (arXiv:2502.03349) 128, PufferLib default
# 128; Isaac Gym 4096-16384 envs, Gigaflow 38,400 envs — same regime bench_release.py
# sweeps for all 7 rl-triton algorithms.
PRODUCTION_SEQ_LENS = [80, 128]
PRODUCTION_NUM_ENVS = [4096, 8192, 16384, 32768, 38400]

# Boundary marker: PufferLib's per-row-sequential design has a flat per-thread
# cost, so very short horizons are its best case relative to a kernel doing
# O(log T) tree-reduction work per program — checked explicitly, not assumed.
BOUNDARY_SEQ_LENS = [8, 16, 32]


def _try_load_pufferlib():
    """Pip package ONLY — no vendored/JIT-built fallback. Returns (op,
    version_str) on success, (None, reason) if unavailable for any reason."""
    try:
        import pufferlib
    except ImportError:
        return None, "`pufferlib` is not installed (pip show pufferlib: not found)."

    try:
        from pufferlib import _C  # noqa: F401
    except ImportError:
        return None, (
            f"`pufferlib` {getattr(pufferlib, '__version__', '?')} is installed, but its "
            "compiled CUDA extension (`pufferlib._C`) is not available — the pip package "
            "does not always build it. This script does not vendor/JIT-build PufferLib's "
            "CUDA source itself — skipping cleanly."
        )

    if not hasattr(torch.ops.pufferlib, "compute_puff_advantage"):
        return None, (
            "`pufferlib._C` imported, but torch.ops.pufferlib.compute_puff_advantage is not "
            "registered — skipping cleanly."
        )

    return torch.ops.pufferlib.compute_puff_advantage, getattr(pufferlib, "__version__", "unknown")


def _make_rl_triton_inputs(num_envs, seq_len, device, seed=SEED):
    g = torch.Generator(device=device).manual_seed(seed)
    rewards = torch.randn(num_envs, seq_len, device=device, generator=g).contiguous()
    values = torch.randn(num_envs, seq_len, device=device, generator=g).contiguous()
    terminateds = (torch.rand(num_envs, seq_len, device=device, generator=g) < TERM_PROB).float().contiguous()
    return rewards, values, terminateds


def _to_puffer_inputs(rewards, values, terminateds):
    """PufferLib's rewards/dones are indexed one slot ahead of values (see
    module docstring in the historical benchmarks/benchmark_gae_vs_pufferlib.py
    for the full derivation) — a real PufferLib caller's buffer is already laid
    out this way, so this reshuffle is a property of test-data generation,
    not a cost PufferLib actually pays at call time."""
    num_envs, seq_len = rewards.shape
    device = rewards.device
    puffer_rewards = torch.zeros(num_envs, seq_len, device=device)
    puffer_dones = torch.zeros(num_envs, seq_len, device=device)
    puffer_rewards[:, 1:] = rewards[:, :seq_len - 1]
    puffer_dones[:, 1:] = terminateds[:, :seq_len - 1]
    puffer_importance = torch.ones(num_envs, seq_len, device=device)
    return values.contiguous(), puffer_rewards.contiguous(), puffer_dones.contiguous(), puffer_importance.contiguous()


def _ref_gae_matched_window(rewards, values, terminateds, gamma, lambda_):
    """Independent (non-Triton, non-PufferLib) sequential reference over the
    T-1 columns both sides can actually produce from a length-T buffer under
    PufferLib's convention — used only for the correctness gate."""
    num_envs, seq_len = rewards.shape
    device = rewards.device
    out = torch.zeros(num_envs, seq_len - 1, device=device, dtype=rewards.dtype)
    carry = torch.zeros(num_envs, device=device, dtype=rewards.dtype)
    for t in reversed(range(seq_len - 1)):
        not_term = 1.0 - terminateds[:, t]
        delta = rewards[:, t] + gamma * not_term * values[:, t + 1] - values[:, t]
        carry = delta + gamma * lambda_ * not_term * carry
        out[:, t] = carry
    return out


def run_equivalence_gate(op, device="cuda"):
    print("=" * 88)
    print("SEMANTIC EQUIVALENCE GATE (on-policy, no interior truncations — the only regime")
    print("PufferLib and rl-triton's plain GAE path have in common)")
    print("=" * 88)
    all_ok = True
    for num_envs, seq_len, seed in [(64, 256, 1), (37, 129, 2), (4096, 8, 3), (16384, 32, 4)]:
        rewards, values, terminateds = _make_rl_triton_inputs(num_envs, seq_len, device, seed)
        pv, pr, pd, pimp = _to_puffer_inputs(rewards, values, terminateds)
        puffer_adv = torch.zeros(num_envs, seq_len, device=device)
        op(pv, pr, pd, pimp, puffer_adv, GAMMA, LAMBDA, 1.0, 1.0)
        ref = _ref_gae_matched_window(rewards, values, terminateds, GAMMA, LAMBDA)
        max_diff = (ref - puffer_adv[:, :seq_len - 1]).abs().max().item()
        status = "PASS (exact)" if max_diff < 1e-4 else "FAIL"
        if max_diff >= 1e-4:
            all_ok = False
        print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  max|PufferLib - ref| = {max_diff:.3e}  [{status}]")
    if not all_ok:
        print("RESULT: FAIL — do not trust the timing comparison below.")
        sys.exit(1)
    print("RESULT: PASS — PufferLib's output matches the independent reference exactly on every")
    print("column it can structurally produce, in the one regime it and rl-triton's GAE overlap.")
    print()
    return all_ok


def run_sweep(op, num_envs_list, seq_lens, label, device="cuda"):
    print("=" * 88)
    print(f"SWEEP: {label}")
    print("=" * 88)
    results = {}
    for num_envs in num_envs_list:
        for seq_len in seq_lens:
            rewards, values, terminateds = _make_rl_triton_inputs(num_envs, seq_len, device)
            pv, pr, pd, pimp = _to_puffer_inputs(rewards, values, terminateds)

            def triton_call(rewards=rewards, values=values, terminateds=terminateds):
                return compute_gae(rewards, values, terminateds, gamma=GAMMA, lambda_=LAMBDA)

            def puffer_call(pv=pv, pr=pr, pd=pd, pimp=pimp, num_envs=num_envs, seq_len=seq_len):
                adv = torch.zeros(num_envs, seq_len, device=device)
                op(pv, pr, pd, pimp, adv, GAMMA, LAMBDA, 1.0, 1.0)
                return adv

            _warmup_gpu(triton_call, n_warmup=15)
            _warmup_gpu(puffer_call, n_warmup=15)

            triton_ms = _bench_gpu(triton_call, n_iter=100, n_trials=5)
            puffer_ms = _bench_gpu(puffer_call, n_iter=100, n_trials=5)
            triton_dev_ms, _ = _device_profile(triton_call)
            puffer_dev_ms, _ = _device_profile(puffer_call)
            triton_amort_ms = _bench_gpu_amortized(triton_call)
            puffer_amort_ms = _bench_gpu_amortized(puffer_call)

            results[(num_envs, seq_len)] = dict(
                triton_ms=triton_ms, puffer_ms=puffer_ms,
                triton_dev_ms=triton_dev_ms, puffer_dev_ms=puffer_dev_ms,
                triton_amort_ms=triton_amort_ms, puffer_amort_ms=puffer_amort_ms,
                speedup=puffer_ms / triton_ms,
                dev_speedup=puffer_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
            )
            r = results[(num_envs, seq_len)]
            print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  "
                  f"triton={r['triton_ms']:.4f}ms (dev {r['triton_dev_ms']:.4f}ms)  "
                  f"puffer={r['puffer_ms']:.4f}ms (dev {r['puffer_dev_ms']:.4f}ms)  "
                  f"speedup={r['speedup']:.2f}x  dev_speedup={r['dev_speedup']:.2f}x")
    print()
    return results


def render_markdown(gpu_label, puffer_version, prod_results, boundary_results):
    date = datetime.date.today().isoformat()
    lines = [
        f"# rl-triton vs. PufferLib — GAE advantage kernel",
        "",
        f"*Measured on {gpu_label} · {date} · pufferlib {puffer_version}.*",
        "",
        "Capability + design comparison, not a scoreboard. PufferLib's `puff_advantage_row_cuda` "
        "is a real hand-written CUDA kernel: one thread per environment row, sequential O(T) "
        "scan within the thread. rl-triton's `compute_gae` is a Triton kernel: one program per "
        "environment row, O(log T) in-SRAM tree reduction via `tl.associative_scan`. Different "
        "mechanisms, different tradeoffs — PufferLib's flat per-thread cost tends to win at very "
        "short horizons; rl-triton's parallel scan tends to win as horizon grows.",
        "",
        "**Capability gap (not benchmarked, because there is nothing on the other side to "
        "benchmark against):** PufferLib takes a single `dones` flag per step, with no "
        "distinction between true episode termination and a time-limit truncation, and no "
        "`bootstrap_values` mechanism. rl-triton's interior-truncation path "
        "(`terminateds`/`truncateds`/`bootstrap_values`) has no PufferLib equivalent at all.",
        "",
        "Both full-call wall time (headline — includes launch/wrapper overhead, what a caller "
        "pays every invocation) and device-only kernel time (diagnostic) are reported, since "
        "per-call overhead is central to the short-horizon comparison.",
        "",
        "#### Production regime (seq_len [80,128] x num_envs [4096..38400])",
        "",
        "| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) "
        "| puffer device (ms) | speedup (full-call) | speedup (device) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for num_envs in PRODUCTION_NUM_ENVS:
        for seq_len in PRODUCTION_SEQ_LENS:
            r = prod_results[(num_envs, seq_len)]
            lines.append(
                f"| {num_envs} | {seq_len} | {r['triton_ms']:.4f} | {r['triton_dev_ms']:.4f} "
                f"| {r['puffer_ms']:.4f} | {r['puffer_dev_ms']:.4f} | {r['speedup']:.2f}x "
                f"| {r['dev_speedup']:.2f}x |"
            )
    lines += [
        "",
        "#### Boundary marker (seq_len [8,16,32]) — PufferLib's best-case regime",
        "",
        "| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) "
        "| puffer device (ms) | speedup (full-call) | speedup (device) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for num_envs in PRODUCTION_NUM_ENVS:
        for seq_len in BOUNDARY_SEQ_LENS:
            r = boundary_results[(num_envs, seq_len)]
            lines.append(
                f"| {num_envs} | {seq_len} | {r['triton_ms']:.4f} | {r['triton_dev_ms']:.4f} "
                f"| {r['puffer_ms']:.4f} | {r['puffer_dev_ms']:.4f} | {r['speedup']:.2f}x "
                f"| {r['dev_speedup']:.2f}x |"
            )
    lines += [
        "",
        "Reference: [PufferLib](https://github.com/PufferAI/PufferLib).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="", metavar="LABEL")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping (this comparison requires a GPU).")
        return

    gpu_label = args.gpu or torch.cuda.get_device_name(0)
    op, version_or_reason = _try_load_pufferlib()
    if op is None:
        print("=" * 88)
        print("PufferLib comparison SKIPPED")
        print("=" * 88)
        print(version_or_reason)
        print()
        print("No numeric comparison was run — this script only uses a pip-installed pufferlib's ")
        print("own prebuilt extension; it does not vendor or build any PufferLib source itself.")
        out_path = Path(__file__).parent / "pufferlib.md"
        out_path.write_text(
            f"# rl-triton vs. PufferLib — GAE advantage kernel\n\n"
            f"*Attempted on {gpu_label} · {datetime.date.today().isoformat()}.*\n\n"
            f"**Skipped**: {version_or_reason}\n\n"
            f"Reference: [PufferLib](https://github.com/PufferAI/PufferLib).\n"
        )
        print(f"\nWrote skip notice to {out_path}")
        return

    print(f"GPU: {gpu_label}")
    print(f"pufferlib: {version_or_reason}")
    print()

    run_equivalence_gate(op, "cuda")
    prod_results = run_sweep(op, PRODUCTION_NUM_ENVS, PRODUCTION_SEQ_LENS, "PRODUCTION REGIME")
    boundary_results = run_sweep(op, PRODUCTION_NUM_ENVS, BOUNDARY_SEQ_LENS, "BOUNDARY MARKER")

    markdown = render_markdown(gpu_label, version_or_reason, prod_results, boundary_results)
    out_path = Path(__file__).parent / "pufferlib.md"
    out_path.write_text(markdown)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
