"""Ad hoc num_envs sweep – NOT part of the official release benchmark.

Investigates whether Triton's wall-clock speedup over torch.compile's
vectorized baseline (which collapsed from 11.3x on RTX 2000 Ada to 1.6x on
H100 at (num_envs=512, seq_len=4096) for compute_eligibility_traces, per
benchmarks.md "v0.x.0 - 2026-07-21 - H100 SXM") recovers as num_envs grows
into the thousands, at a fixed seq_len.

compute_eligibility_traces / compute_gae / compute_vtrace_fused /
compute_retrace_fused all launch one Triton program per environment
(grid=(num_envs,)) — num_envs controls how many independent programs run in
parallel across the GPU's SMs, a genuinely different axis from seq_len (which
controls per-program register/compute footprint). At num_envs=512 the grid
may be too small to saturate an H100's 132 SMs across the internal waves the
_bench_gpu harness times, in which case fixed per-launch dispatch overhead
would dominate wall-clock regardless of kernel quality — same mechanism
identified for the seq_len axis in the prior profiling pass.

This script does NOT touch tests/bench_release.py's CONFIGS list, and does
NOT write to tests/benchmarks.md — it is a standalone, throwaway diagnostic.
Reports both:
  - wall-clock ms via the existing _bench_gpu/_warmup_gpu/_n_iter_gpu helpers
    (same CUDA-event methodology as the official suite, for comparability)
  - device-only kernel time via torch.profiler CUDA activity (steady-state),
    since ncu/nsys are unusable in this container (RmProfilingAdminOnly=1 +
    missing CAP_SYS_ADMIN/CAP_PERFMON — see prior profiling report)

Usage:
    python tests/bench_num_envs_sweep.py
"""
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parent))
from bench_utils import _bench_gpu, _n_iter_gpu, _warmup_gpu

from rl_triton.ops.gae import compute_gae
from rl_triton.ops.retrace_fused import compute_retrace_fused
from rl_triton.ops.returns import compute_eligibility_traces
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

from test_gae import vectorized_gae
from test_retrace import vectorized_retrace
from test_returns import vectorized_eligibility_traces
from test_vtrace import vectorized_vtrace

# Fixed seq_len=2048 (moderate, within the 2k-8k on-policy rollout range per
# NOTES.md), num_envs sweeping from the existing suite's max (512) up into
# the low thousands. 512 is included as a baseline anchor for continuity
# with the official benchmarks.md numbers.
CONFIGS_NUM_ENVS = [512, 1024, 4096, 8192]
SEQ_LEN = 2048


def _make_gae(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    d = device
    return (
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),
    )


def _make_returns(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    d = device
    rewards = torch.randn(num_envs, seq_len, device=d)
    dones = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    return rewards, dones


def _make_vtrace(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    d = device
    return (
        -torch.rand(num_envs, seq_len, device=d),
        -torch.rand(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),
    )


def _make_retrace(num_envs, seq_len, num_actions=4, device="cuda"):
    torch.manual_seed(0)
    d = device
    apt = torch.softmax(torch.randn(num_envs, seq_len, num_actions, device=d), dim=-1)
    apb = torch.rand(num_envs, seq_len, device=d) * 0.8 + 0.1
    q_values = torch.randn(num_envs, seq_len, device=d)
    next_q_all = torch.randn(num_envs, seq_len, num_actions, device=d)
    actions = torch.randint(0, num_actions, (num_envs, seq_len), device=d)
    rewards = torch.randn(num_envs, seq_len, device=d)
    terminated = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    truncated = torch.zeros(num_envs, seq_len, device=d)
    return apt, apb, q_values, next_q_all, actions, rewards, truncated, terminated


def _device_time_us(fn, kernel_name_substr, *args, n_iter=20, **kwargs):
    """Steady-state device-only self time via torch.profiler, us/call."""
    for _ in range(5):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
    total = sum(
        e.self_device_time_total for e in prof.key_averages()
        if kernel_name_substr in e.key
    )
    return total / n_iter


def _row(label, num_envs, seq_len, triton_ms, vec_ms, triton_us, vec_us):
    speedup_wall = vec_ms / triton_ms
    speedup_dev = vec_us / triton_us if triton_us else float("nan")
    print(
        f"{label:<20} {num_envs:>8} {seq_len:>8}  "
        f"{triton_ms:>10.4f}ms {vec_ms:>10.4f}ms {speedup_wall:>8.2f}x   "
        f"{triton_us:>10.2f}us {vec_us:>10.2f}us {speedup_dev:>8.2f}x"
    )


def sweep_eligibility_traces():
    print("\n### compute_eligibility_traces ###")
    compiled_vec = torch.compile(vectorized_eligibility_traces)
    r0, d0 = _make_returns(64, 512)
    compiled_vec(r0, d0, gamma=0.99, lambda_=0.95)
    torch.cuda.synchronize()

    header = (
        f"{'algo':<20} {'num_envs':>8} {'seq_len':>8}  "
        f"{'triton(wall)':>12} {'vec(wall)':>12} {'wall x':>8}   "
        f"{'triton(dev)':>12} {'vec(dev)':>12} {'dev x':>8}"
    )
    print(header)
    print("-" * len(header))
    for num_envs in CONFIGS_NUM_ENVS:
        rewards, dones = _make_returns(num_envs, SEQ_LEN)
        ni = _n_iter_gpu(SEQ_LEN, num_envs)
        _warmup_gpu(compute_eligibility_traces, rewards, dones, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_vec, rewards, dones, gamma=0.99, lambda_=0.95)

        triton_ms = _bench_gpu(compute_eligibility_traces, rewards, dones,
                                gamma=0.99, lambda_=0.95, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec, rewards, dones,
                             gamma=0.99, lambda_=0.95, n_iter=ni)
        triton_us = _device_time_us(compute_eligibility_traces, "eligibility_traces_fused_kernel",
                                     rewards, dones, gamma=0.99, lambda_=0.95)
        vec_us = _device_time_us(compiled_vec, "triton_", rewards, dones,
                                  gamma=0.99, lambda_=0.95)
        _row("eligibility-traces", num_envs, SEQ_LEN, triton_ms, vec_ms, triton_us, vec_us)


def sweep_gae():
    print("\n### compute_gae ###")
    compiled_vec = torch.compile(vectorized_gae)
    r0, v0, d0 = _make_gae(64, 512)
    compiled_vec(r0, v0, d0, gamma=0.99, lambda_=0.95)
    torch.cuda.synchronize()

    header = (
        f"{'algo':<20} {'num_envs':>8} {'seq_len':>8}  "
        f"{'triton(wall)':>12} {'vec(wall)':>12} {'wall x':>8}   "
        f"{'triton(dev)':>12} {'vec(dev)':>12} {'dev x':>8}"
    )
    print(header)
    print("-" * len(header))
    for num_envs in CONFIGS_NUM_ENVS:
        rewards, values, dones = _make_gae(num_envs, SEQ_LEN)
        ni = _n_iter_gpu(SEQ_LEN, num_envs)
        _warmup_gpu(compute_gae, rewards, values, dones, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_vec, rewards, values, dones, gamma=0.99, lambda_=0.95)

        triton_ms = _bench_gpu(compute_gae, rewards, values, dones,
                                gamma=0.99, lambda_=0.95, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec, rewards, values, dones,
                             gamma=0.99, lambda_=0.95, n_iter=ni)
        triton_us = _device_time_us(compute_gae, "gae_fused_kernel",
                                     rewards, values, dones, gamma=0.99, lambda_=0.95)
        vec_us = _device_time_us(compiled_vec, "triton_", rewards, values, dones,
                                  gamma=0.99, lambda_=0.95)
        _row("gae", num_envs, SEQ_LEN, triton_ms, vec_ms, triton_us, vec_us)


def sweep_vtrace_fused():
    print("\n### compute_vtrace_fused ###")
    compiled_vec = torch.compile(vectorized_vtrace)
    args0 = _make_vtrace(64, 512)
    compiled_vec(*args0, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"{'algo':<20} {'num_envs':>8} {'seq_len':>8}  "
        f"{'triton(wall)':>12} {'vec(wall)':>12} {'wall x':>8}   "
        f"{'triton(dev)':>12} {'vec(dev)':>12} {'dev x':>8}"
    )
    print(header)
    print("-" * len(header))
    for num_envs in CONFIGS_NUM_ENVS:
        args = _make_vtrace(num_envs, SEQ_LEN)
        ni = _n_iter_gpu(SEQ_LEN, num_envs)
        _warmup_gpu(compute_vtrace_fused, *args, gamma=0.99)
        _warmup_gpu(compiled_vec, *args, gamma=0.99)

        triton_ms = _bench_gpu(compute_vtrace_fused, *args, gamma=0.99, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec, *args, gamma=0.99, n_iter=ni)
        triton_us = _device_time_us(compute_vtrace_fused, "vtrace_fused_kernel", *args, gamma=0.99)
        vec_us = _device_time_us(compiled_vec, "triton_", *args, gamma=0.99)
        _row("vtrace-fused", num_envs, SEQ_LEN, triton_ms, vec_ms, triton_us, vec_us)


def sweep_retrace_fused():
    print("\n### compute_retrace_fused ###")
    compiled_vec = torch.compile(vectorized_retrace)
    apt0, apb0, q0, nq0, act0, r0, trunc0, term0 = _make_retrace(64, 512)
    compiled_vec(apt0, apb0, q0, nq0, act0, r0, term0, trunc0, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"{'algo':<20} {'num_envs':>8} {'seq_len':>8}  "
        f"{'triton(wall)':>12} {'vec(wall)':>12} {'wall x':>8}   "
        f"{'triton(dev)':>12} {'vec(dev)':>12} {'dev x':>8}"
    )
    print(header)
    print("-" * len(header))
    for num_envs in CONFIGS_NUM_ENVS:
        apt, apb, q_values, next_q_all, actions, rewards, truncated, terminated = \
            _make_retrace(num_envs, SEQ_LEN)
        ni = _n_iter_gpu(SEQ_LEN, num_envs)
        _warmup_gpu(compute_retrace_fused, apt, apb, q_values, next_q_all, actions,
                    rewards, truncated, terminated, gamma=0.99)
        _warmup_gpu(compiled_vec, apt, apb, q_values, next_q_all, actions, rewards,
                    terminated, truncated, gamma=0.99)

        triton_ms = _bench_gpu(compute_retrace_fused, apt, apb, q_values, next_q_all,
                                actions, rewards, truncated, terminated, gamma=0.99, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec, apt, apb, q_values, next_q_all, actions,
                             rewards, terminated, truncated, gamma=0.99, n_iter=ni)
        triton_us = _device_time_us(compute_retrace_fused, "retrace_fused_kernel",
                                     apt, apb, q_values, next_q_all, actions, rewards,
                                     truncated, terminated, gamma=0.99)
        vec_us = _device_time_us(compiled_vec, "triton_", apt, apb, q_values, next_q_all,
                                  actions, rewards, terminated, truncated, gamma=0.99)
        _row("retrace-fused", num_envs, SEQ_LEN, triton_ms, vec_ms, triton_us, vec_us)


if __name__ == "__main__":
    print(f"num_envs sweep at fixed seq_len={SEQ_LEN}, num_envs in {CONFIGS_NUM_ENVS}")
    print("(ad hoc diagnostic — does not touch bench_release.py CONFIGS or benchmarks.md)")
    sweep_eligibility_traces()
    sweep_gae()
    sweep_vtrace_fused()
    sweep_retrace_fused()
