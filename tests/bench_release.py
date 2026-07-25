"""
Release benchmark – full sweep across all algorithms and configurations.

Compares rl-triton Triton kernels against torch.compile on both a naive
per-timestep Python loop and the fastest hand-vectorized PyTorch equivalent,
against pure NumPy CPU loops, and against a NumPy→GPU→NumPy adoption path.
Algorithms that support truncated episodes (GAE, V-Trace, λ-returns,
discounted returns) also report a third baseline – the same associative-scan
implementation used for the truncation path, called with zero truncations –
plus an additional table benchmarking the truncation path itself against
that vectorized baseline.

After running, updates the <!-- BENCH_START --> ... <!-- BENCH_END --> section
in README.md with the latest results.

Usage:
    python tests/bench_release.py              # run and update README.md
    python tests/bench_release.py --no-update  # print only, do not touch README
    python tests/bench_release.py --gpu RTX4090  # label the GPU in the table header
"""
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import triton

# Ensure the tests/ directory is on sys.path so bench_utils is importable.
sys.path.insert(0, str(Path(__file__).parent))
from bench_utils import (
    _bench_cpu,
    _bench_gpu,
    _bench_gpu_amortized,
    _device_profile,
    _n_iter_gpu,
    _warmup_gpu,
    assert_correctness,
    assert_deterministic,
    check_monotonic_grid,
    print_environment_header,
)

from rl_triton.ops.gae import compute_gae
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum
from rl_triton.ops.retrace import compute_retrace
from rl_triton.ops.returns import (
    compute_discounted_returns,
    compute_eligibility_traces,
    compute_lambda_returns,
)
from rl_triton.ops.vtrace import compute_vtrace
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

from test_gae import vectorized_gae, vectorized_gae_with_truncations
from test_returns import (
    _ref_discounted_sequential,
    _ref_lambda_sequential,
    vectorized_discounted_returns,
    vectorized_discounted_returns_with_truncations,
    vectorized_eligibility_traces,
    vectorized_lambda_returns,
    vectorized_lambda_returns_with_truncations,
)
from test_vtrace import (
    _ref_vtrace_sequential,
    vectorized_vtrace,
    vectorized_vtrace_with_truncations,
)


# ---------------------------------------------------------------------------
# Reference implementations (PyTorch loops – ground truth & compiled baselines)
# ---------------------------------------------------------------------------

def _ref_gae(rewards, values, dones, gamma, lambda_):
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = 0.0
    for t in reversed(range(T)):
        not_done  = 1.0 - dones[:, t]
        delta     = rewards[:, t] + gamma * not_done * next_values[:, t] - values[:, t]
        carry     = delta + gamma * lambda_ * not_done * carry
        out[:, t] = carry
    return out


def _ref_gae_trunc(rewards, values, terminateds, truncateds, bootstrap_values, gamma, lambda_):
    """Vectorized-over-N (loop over T only) ground truth with interior
    truncations — same recurrence as test_gae.py's O(N*T) _ref_gae_sequential,
    but fast enough to run at every swept config (production regime included).
    """
    N, T = rewards.shape
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    out   = torch.zeros_like(rewards)
    carry = bootstrap_values[:, T - 1].clone()
    for t in reversed(range(T)):
        if t == T - 1:
            v_next = bootstrap_values[:, t]
        else:
            v_next = torch.where(truncateds[:, t].bool(), bootstrap_values[:, t], next_values[:, t])
        not_terminated = 1.0 - terminateds[:, t]
        done  = (terminateds[:, t] + truncateds[:, t]).clamp(max=1.0)
        delta = rewards[:, t] + gamma * not_terminated * v_next - values[:, t]
        carry = delta + gamma * lambda_ * (1.0 - done) * carry
        out[:, t] = carry
    return out


def _ref_vtrace(log_pi_t, log_pi_b, values, rewards, dones, gamma,
                rho_bar=1.0, c_bar=1.0, bootstrap_values=None):
    num_envs, T = rewards.shape
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = 0.0 if bootstrap_values is None else bootstrap_values
    rho = torch.clamp(torch.exp(log_pi_t - log_pi_b), max=rho_bar)
    c   = torch.clamp(torch.exp(log_pi_t - log_pi_b), max=c_bar)
    deltas = rho * (rewards + gamma * next_values * (1.0 - dones) - values)
    decays = gamma * c * (1.0 - dones)
    out   = torch.zeros_like(rewards)
    carry = (bootstrap_values.clone() if bootstrap_values is not None
             else torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype))
    for t in reversed(range(T)):
        carry     = deltas[:, t] + decays[:, t] * carry
        out[:, t] = carry
    targets = out + values
    next_t  = torch.empty_like(targets)
    next_t[:, :-1] = targets[:, 1:]
    next_t[:, -1]  = 0.0 if bootstrap_values is None else bootstrap_values
    advantages = rho * (rewards + gamma * next_t * (1.0 - dones) - values)
    return targets, advantages


def _ref_retrace(rewards, dones, values, next_q_all, q_values, actions,
                 action_probs_target, action_probs_behavior, gamma,
                 lambda_=1.0, c_bar=1.0):
    num_envs, T = rewards.shape
    pi_a   = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c      = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    exp_nq = (action_probs_target * next_q_all).sum(dim=-1)
    u      = rewards + gamma * exp_nq * (1.0 - dones) - q_values
    c_next = torch.empty_like(c)
    c_next[:, :-1] = c[:, 1:]
    c_next[:, -1]  = 0.0
    v     = gamma * c_next * (1.0 - dones)
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        carry     = u[:, t] + v[:, t] * carry
        out[:, t] = carry
    return out + q_values


def _vec_retrace(rewards, dones, values, next_q_all, q_values, actions,
                 action_probs_target, action_probs_behavior, gamma,
                 lambda_=1.0, c_bar=1.0):
    """
    Fully vectorized Retrace(λ) via log-space suffix cumsum – strong compiled baseline.

    Replaces the Python backward loop in _ref_retrace with a vectorized equivalent.
    The backward scan Δ[t] = u[t] + v[t]*Δ[t+1] is a weighted sum where each
    weight is the suffix product of decays v from t to T-1, computed in log-space
    to avoid underflow.  Timed with CUDA events (no Python loop).
    """
    pi_a   = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c      = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    exp_nq = (action_probs_target * next_q_all).sum(dim=-1)
    u      = rewards + gamma * exp_nq * (1.0 - dones) - q_values
    c_next          = torch.empty_like(c)
    c_next[:, :-1]  = c[:, 1:]
    c_next[:, -1]   = 0.0
    v = gamma * c_next * (1.0 - dones)

    log_suffix = torch.flip(
        torch.cumsum(torch.flip(torch.log(v.clamp(min=1e-38)), [1]), dim=1), [1]
    )
    weights  = torch.exp(log_suffix)
    q_deltas = torch.flip(
        torch.cumsum(torch.flip(u * weights, [1]), dim=1), [1]
    ) / weights

    return q_deltas + q_values


def _ref_lambda(rewards, next_values, dones, gamma, lambda_, bootstrap=None):
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = (bootstrap.clone() if bootstrap is not None
             else torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype))
    for t in reversed(range(T)):
        not_done = 1.0 - dones[:, t]
        carry = (rewards[:, t]
                 + gamma * (1.0 - lambda_) * not_done * next_values[:, t]
                 + gamma * lambda_ * not_done * carry)
        out[:, t] = carry
    return out


def _ref_disc(rewards, dones, gamma, bootstrap=None):
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = (bootstrap.clone() if bootstrap is not None
             else torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype))
    for t in reversed(range(T)):
        carry     = rewards[:, t] + gamma * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


def _ref_traces(features, dones, gamma, lambda_, seed=None):
    T     = features.shape[1]
    out   = torch.zeros_like(features)
    carry = (seed.clone() if seed is not None
             else torch.zeros(features.shape[0], device=features.device, dtype=features.dtype))
    for t in range(T):
        carry     = features[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


# ---------------------------------------------------------------------------
# NumPy CPU baselines and NumPy→GPU→NumPy adoption paths (GAE and V-Trace)
# ---------------------------------------------------------------------------

def numpy_gae_cpu(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """CPU GAE backward loop – moves GPU tensors to CPU and runs a plain Python loop."""
    rewards, values, dones = rewards.cpu(), values.cpu(), dones.cpu()
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = 0.0
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0])
    for t in reversed(range(T)):
        not_done  = 1.0 - dones[:, t]
        delta     = rewards[:, t] + gamma * not_done * next_values[:, t] - values[:, t]
        carry     = delta + gamma * lambda_ * not_done * carry
        out[:, t] = carry
    return out


def numpy_gae_np_to_triton(
    rewards_np: np.ndarray,
    values_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """NumPy → GPU Triton → NumPy end-to-end adoption path for GAE."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.float32)
    out = compute_gae(
        to_gpu(rewards_np), to_gpu(values_np), to_gpu(dones_np),
        gamma=gamma, lambda_=lambda_,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def numpy_vtrace_cpu(log_pi_t, log_pi_b, values, rewards, dones,
                     gamma, rho_bar=1.0, c_bar=1.0):
    """CPU V-Trace backward loop on CPU tensors."""
    cpu = lambda t: t.cpu().float()
    lpt, lpb, v, r, d = map(cpu, [log_pi_t, log_pi_b, values, rewards, dones])
    nv = torch.empty_like(v)
    nv[:, :-1] = v[:, 1:]
    nv[:, -1]  = 0.0
    num_envs, T = r.shape
    rho = torch.clamp(torch.exp(lpt - lpb), max=rho_bar)
    c   = torch.clamp(torch.exp(lpt - lpb), max=c_bar)
    disc = gamma * (1.0 - d)
    deltas = rho * (r + disc * nv - v)
    out   = torch.zeros_like(r)
    carry = torch.zeros(num_envs)
    for t in reversed(range(T)):
        carry     = deltas[:, t] + disc[:, t] * c[:, t] * carry
        out[:, t] = carry
    targets = out + v
    next_t  = torch.empty_like(targets)
    next_t[:, :-1] = targets[:, 1:]
    next_t[:, -1]  = nv[:, -1]
    advantages = rho * (r + disc * next_t - v)
    return targets, advantages


def numpy_vtrace_np_to_triton(log_pi_t_np, log_pi_b_np, values_np,
                               rewards_np, dones_np, gamma, rho_bar=1.0, c_bar=1.0):
    """NumPy → GPU Triton → NumPy end-to-end adoption path for V-Trace."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.float32)
    targets, advantages = compute_vtrace(
        to_gpu(log_pi_t_np), to_gpu(log_pi_b_np), to_gpu(values_np),
        to_gpu(rewards_np), to_gpu(dones_np),
        gamma=gamma, rho_bar=rho_bar, c_bar=c_bar,
    )
    torch.cuda.synchronize()
    return targets.cpu().numpy(), advantages.cpu().numpy()


def numpy_retrace_np_to_triton(rewards_np, dones_np, q_values_np, next_q_all_np,
                               actions_np, action_probs_target_np, action_probs_behavior_np,
                               gamma, lambda_=1.0, c_bar=1.0):
    """NumPy → GPU Triton → NumPy end-to-end adoption path for Retrace(λ)."""
    to_gpu_f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.float32)
    to_gpu_i = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.int64)
    apt  = to_gpu_f(action_probs_target_np)
    apb  = to_gpu_f(action_probs_behavior_np)
    q    = to_gpu_f(q_values_np)
    nqa  = to_gpu_f(next_q_all_np)
    acts = to_gpu_i(actions_np)
    r    = to_gpu_f(rewards_np)
    d    = to_gpu_f(dones_np)
    truncateds = torch.zeros_like(d)
    retrace_targets, _ = compute_retrace(apt, apb, q, nqa, acts, r, d, truncateds,
                                         gamma=gamma, lambda_=lambda_, c_bar=c_bar)
    torch.cuda.synchronize()
    return retrace_targets.cpu().numpy()


def numpy_retrace_cpu(rewards_np, dones_np, q_values_np, next_q_all_np,
                      actions_np, action_probs_target_np, action_probs_behavior_np,
                      gamma, lambda_=1.0, c_bar=1.0):
    """Pure NumPy backward scan for Retrace(λ) on CPU."""
    num_envs, seq_len = rewards_np.shape
    pi_a   = action_probs_target_np[np.arange(num_envs)[:, None],
                                    np.arange(seq_len)[None, :],
                                    actions_np]
    c      = lambda_ * np.minimum(pi_a / action_probs_behavior_np, c_bar)
    exp_nq = (action_probs_target_np * next_q_all_np).sum(axis=-1)
    u      = rewards_np + gamma * exp_nq * (1.0 - dones_np) - q_values_np
    c_next              = np.empty_like(c)
    c_next[:, :-1]      = c[:, 1:]
    c_next[:, -1]       = 0.0
    v = gamma * c_next * (1.0 - dones_np)
    out = np.empty_like(rewards_np)
    for i in range(num_envs):
        last = 0.0
        for t in reversed(range(seq_len)):
            last      = u[i, t] + v[i, t] * last
            out[i, t] = last
    return out + q_values_np


def numpy_lambda_returns_cpu(rewards: torch.Tensor, next_values: torch.Tensor,
                             dones: torch.Tensor, gamma: float, lambda_: float) -> np.ndarray:
    """Pure NumPy backward scan for λ-returns on CPU."""
    r  = rewards.cpu().numpy()
    nv = next_values.cpu().numpy()
    d  = dones.cpu().numpy()
    T  = r.shape[1]
    out   = np.empty_like(r)
    carry = np.zeros(r.shape[0], dtype=np.float32)
    for t in reversed(range(T)):
        not_done = 1.0 - d[:, t]
        carry = (r[:, t]
                 + gamma * (1.0 - lambda_) * not_done * nv[:, t]
                 + gamma * lambda_ * not_done * carry)
        out[:, t] = carry
    return out


def numpy_discounted_returns_cpu(rewards: torch.Tensor, dones: torch.Tensor,
                                  gamma: float) -> np.ndarray:
    """Pure NumPy backward scan for discounted returns on CPU."""
    r = rewards.cpu().numpy()
    d = dones.cpu().numpy()
    T = r.shape[1]
    out   = np.empty_like(r)
    carry = np.zeros(r.shape[0], dtype=np.float32)
    for t in reversed(range(T)):
        carry    = r[:, t] + gamma * (1.0 - d[:, t]) * carry
        out[:, t] = carry
    return out


def numpy_eligibility_traces_cpu(features: torch.Tensor, dones: torch.Tensor,
                                  gamma: float, lambda_: float) -> np.ndarray:
    """Pure NumPy forward scan for eligibility traces on CPU."""
    x = features.cpu().numpy()
    d = dones.cpu().numpy()
    T = x.shape[1]
    out   = np.empty_like(x)
    carry = np.zeros(x.shape[0], dtype=np.float32)
    for t in range(T):
        carry    = x[:, t] + gamma * lambda_ * (1.0 - d[:, t]) * carry
        out[:, t] = carry
    return out


# ---------------------------------------------------------------------------
# Input factories
# ---------------------------------------------------------------------------

def _make_trunc_extras(num_envs, seq_len, terminateds, device="cuda"):
    """~5% truncated steps (mutually exclusive with terminateds), with
    bootstrap_values populated at truncated steps and the final boundary.

    Shared across GAE, V-Trace, λ-returns, and discounted returns – the same
    pattern used by their respective test_*_truncation_performance tests.
    """
    trunc_cand = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    truncateds = trunc_cand * (1.0 - terminateds)
    bootstrap_values = torch.zeros(num_envs, seq_len, device=device)
    n = int(truncateds.sum().item())
    if n:
        bootstrap_values[truncateds.bool()] = torch.rand(n, device=device)
    bootstrap_values[:, -1] = torch.rand(num_envs, device=device)
    return truncateds, bootstrap_values


def _make_gae(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    d = device
    return (
        torch.randn(num_envs, seq_len, device=d),   # rewards
        torch.randn(num_envs, seq_len, device=d),   # values
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),  # dones
    )


def _make_vtrace(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    d = device
    return (
        -torch.rand(num_envs, seq_len, device=d),   # log_pi_target
        -torch.rand(num_envs, seq_len, device=d),   # log_pi_behavior
        torch.randn(num_envs, seq_len, device=d),   # values
        torch.randn(num_envs, seq_len, device=d),   # rewards
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),  # dones
    )


def _make_retrace(num_envs, seq_len, num_actions=4, device="cuda"):
    torch.manual_seed(0)
    d = device
    return (
        torch.randn(num_envs, seq_len, device=d),
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, num_actions, device=d),
        torch.randn(num_envs, seq_len, device=d),
        torch.randint(0, num_actions, (num_envs, seq_len), device=d),
        torch.softmax(torch.randn(num_envs, seq_len, num_actions, device=d), dim=-1),
        torch.rand(num_envs, seq_len, device=d) * 0.8 + 0.1,
    )


def _make_returns(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    d = device
    rewards     = torch.randn(num_envs, seq_len, device=d)
    next_values = torch.randn(num_envs, seq_len, device=d)
    dones       = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    return rewards, next_values, dones


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

CONFIGS = [
    # Original moderate-N/high-T sweep (kept for continuity / parity baseline).
    (64,  512),
    (128, 1024),
    (256, 1024),
    (512, 2048),
    (512, 4096),
    # Main grid addition — massively-parallel-sim regime (high-N/short-T):
    # rollout lengths GPUDrive 90 / Nocturne 80 / Gigaflow 128 / PufferLib
    # default 128; env counts Isaac Gym 4096-16384, Gigaflow 38,400.
    (512,   128),
    (512,   512),
    (4096,  128),
    (4096,  512),
    (4096,  2048),
    (16384, 128),
    (16384, 512),
]

# ~4 representative sizes per algorithm for benchmarks.md headline tables
# (small/parity, mid, main-grid-large, production-adjacent-large). The full
# CONFIGS grid above remains reproducible via `python tests/bench_release.py`.
HEADLINE_CONFIGS = [(64, 512), (256, 1024), (512, 4096), (16384, 512)]

# Production regime: real massively-parallel-sim RL is high-N/short-T.
# GPUDrive 90, Nocturne 80, Gigaflow (arXiv:2502.03349) 128, PufferLib
# default 128; Isaac Gym 4096-16384 envs, Gigaflow 38,400 envs.
PRODUCTION_SEQ_LENS  = [80, 128]
PRODUCTION_NUM_ENVS  = [4096, 8192, 16384, 32768, 38400]
PRODUCTION_CONFIGS   = [
    (num_envs, seq_len) for seq_len in PRODUCTION_SEQ_LENS for num_envs in PRODUCTION_NUM_ENVS
]

# Boundary marker (limitations): one-program-per-env kernels pay more grid
# waves per SM as num_envs grows while per-program work shrinks with seq_len;
# prior work (tests/h100_short_horizon_l2_retrace_ppo_report.md) found
# device-only time can invert below 1x at very short T / very high num_envs.
BOUNDARY_CONFIG = (16384, 16)

ALL_ALGOS = [
    "gae", "vtrace", "retrace", "lambda_returns",
    "discounted_returns", "eligibility_traces", "prefix_sum",
]


def _bench_production_regime(algo_label, triton_fn, compiled_fn, ref_fn, make_inputs_fn, kwargs):
    """Shared production-regime + boundary-marker sweep for algorithms whose
    triton/compiled/reference callables all accept the SAME positional args
    (everything except Retrace, whose kernel needs a reordered arg tuple —
    see its own inline block in bench_retrace).

    Runs PRODUCTION_CONFIGS (seq_len [80,128] x num_envs [4096..38400]) plus
    the single BOUNDARY_CONFIG marker row, reporting full-call ms (headline),
    device-only ms (diagnostic), and the amortized (N-calls-per-region)
    variant — the latter specifically to separate harness per-call sync
    overhead from genuine per-call cost at these short seq_lens. Asserts
    tolerance-based correctness (atol=1e-4) at every config before trusting
    the timing.
    """
    rows = []
    for num_envs, seq_len in PRODUCTION_CONFIGS + [BOUNDARY_CONFIG]:
        is_boundary = (num_envs, seq_len) == BOUNDARY_CONFIG
        args_gpu = make_inputs_fn(num_envs, seq_len)
        ni = _n_iter_gpu(seq_len, num_envs)

        triton_out = triton_fn(*args_gpu, **kwargs)
        ref_out    = ref_fn(*args_gpu, **kwargs)
        assert_correctness(triton_out, ref_out, f"{algo_label}[production,{num_envs}x{seq_len}]")

        _warmup_gpu(triton_fn,   *args_gpu, **kwargs)
        _warmup_gpu(compiled_fn, *args_gpu, **kwargs)

        triton_ms = _bench_gpu(triton_fn,   *args_gpu, n_iter=ni, **kwargs)
        vec_ms    = _bench_gpu(compiled_fn, *args_gpu, n_iter=ni, **kwargs)
        triton_dev_ms, _ = _device_profile(triton_fn,   *args_gpu, **kwargs)
        vec_dev_ms, _    = _device_profile(compiled_fn, *args_gpu, **kwargs)
        triton_amort_ms  = _bench_gpu_amortized(triton_fn,   *args_gpu, **kwargs)
        vec_amort_ms     = _bench_gpu_amortized(compiled_fn, *args_gpu, **kwargs)

        row = {
            "algo": algo_label, "num_envs": num_envs, "seq_len": seq_len, "is_boundary": is_boundary,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms, "triton_amort_ms": triton_amort_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms, "vec_amort_ms": vec_amort_ms,
            "su_vec": vec_ms / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        }
        rows.append(row)
        tag = " [BOUNDARY]" if is_boundary else ""
        print(f"  [{algo_label} production {num_envs}x{seq_len}]{tag} "
              f"triton={triton_ms:.4f}ms dev={triton_dev_ms:.4f}ms amort={triton_amort_ms:.4f}ms  "
              f"vec={vec_ms:.4f}ms  su={row['su_vec']:.2f}x su_dev={row['su_vec_dev']:.2f}x",
              flush=True)
    return rows


def _precompile_triton_kernels():
    """Trigger LLVM compilation for every Triton kernel variant before timing.

    Each unique (kernel, HAS_TRUNCATIONS, HAS_BOOTSTRAP) specialisation compiles
    separately. Running them all upfront on the smallest config concentrates the
    silent LLVM phase into one visible block so the timed sweep has no cold-start
    pauses. On a warm Triton cache this takes < 1 s total.
    """
    ne, sl = 64, 512
    r, v, d = _make_gae(ne, sl)

    def _step(label, fn, *args, **kwargs):
        print(f"  {label} …", end="", flush=True)
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        print(" ✓", flush=True)

    _step("compute_gae (no-trunc)",
          compute_gae, r, v, d, gamma=0.99, lambda_=0.95)
    trunc, bsv = _make_trunc_extras(ne, sl, d)
    _step("compute_gae (with-trunc)",
          compute_gae, r, v, d, trunc, gamma=0.99, lambda_=0.95, bootstrap_values=bsv)

    args_vt = _make_vtrace(ne, sl)
    _step("compute_vtrace_fused (no-trunc)",
          compute_vtrace_fused, *args_vt, gamma=0.99)
    trunc_vt, bsv_vt = _make_trunc_extras(ne, sl, args_vt[4])
    _step("compute_vtrace_fused (with-trunc)",
          compute_vtrace_fused, *args_vt, truncateds=trunc_vt,
          gamma=0.99, bootstrap_values=bsv_vt)

    args_re = _make_retrace(ne, sl)
    _step("compute_retrace",
          compute_retrace, *_retrace_kernel_args(args_re), gamma=0.99)

    r2, nv, d2 = _make_returns(ne, sl)
    _step("compute_lambda_returns (no-trunc)",
          compute_lambda_returns, r2, nv, d2, gamma=0.99, lambda_=0.95)
    trunc2, bsv2 = _make_trunc_extras(ne, sl, d2)
    _step("compute_lambda_returns (with-trunc)",
          compute_lambda_returns, r2, nv, d2, gamma=0.99, lambda_=0.95,
          truncateds=trunc2, bootstrap_values=bsv2)

    _step("compute_discounted_returns (no-trunc)",
          compute_discounted_returns, r2, d2, gamma=0.99)
    _step("compute_discounted_returns (with-trunc)",
          compute_discounted_returns, r2, d2, gamma=0.99,
          truncateds=trunc2, bootstrap_values=bsv2)

    _step("compute_eligibility_traces",
          compute_eligibility_traces, r2, d2, gamma=0.99, lambda_=0.9)

    _step("compute_episodic_prefix_sum",
          compute_episodic_prefix_sum, r2, d2)


# ---------------------------------------------------------------------------
# Per-algorithm benchmark runners – return list of result dicts
# ---------------------------------------------------------------------------

def bench_gae():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec  = torch.compile(vectorized_gae)
    compiled_assoc = torch.compile(vectorized_gae_with_truncations)
    args_warmup = _make_gae(64, 512)
    compiled_vec(*args_warmup, gamma=0.99, lambda_=0.95); torch.cuda.synchronize()
    _n0, _t0 = args_warmup[0].shape
    _trunc0_w = torch.zeros(_n0, _t0, device="cuda")
    _bsv0_w   = torch.zeros(_n0, _t0, device="cuda")
    _bsv0_w[:, -1] = torch.rand(_n0, device="cuda")
    compiled_assoc(args_warmup[0], args_warmup[1], args_warmup[2], _trunc0_w, _bsv0_w,
                   0.99, 0.95); torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'vs vec':>8} {'vs vec(dev)':>12} "
        f"{'compile(assoc)':>15} {'vs assoc':>10} "
        f"{'loop(gpu)':>11} {'vs loop':>9} {'numpy(cpu)':>12} {'vs numpy':>10} "
        f"{'np->tri->np':>13} {'e2e vs np':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        args_gpu = _make_gae(num_envs, seq_len)
        args_np  = tuple(t.cpu().numpy() for t in args_gpu)
        ni       = _n_iter_gpu(seq_len, num_envs)

        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0[:, -1] = torch.rand(num_envs, device="cuda")

        triton_out = compute_gae(*args_gpu, gamma=0.99, lambda_=0.95)
        ref_out    = _ref_gae(*args_gpu, gamma=0.99, lambda_=0.95)
        assert_correctness(triton_out, ref_out, f"gae[{num_envs}x{seq_len}]")

        _warmup_gpu(compute_gae, *args_gpu, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_vec, *args_gpu, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_assoc, args_gpu[0], args_gpu[1], args_gpu[2], trunc0, bsv0, 0.99, 0.95)

        triton_ms = _bench_gpu(compute_gae,    *args_gpu, gamma=0.99, lambda_=0.95, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,   *args_gpu, gamma=0.99, lambda_=0.95, n_iter=ni)
        assoc_ms  = _bench_gpu(compiled_assoc,
                                args_gpu[0], args_gpu[1], args_gpu[2], trunc0, bsv0, 0.99, 0.95,
                                n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_gae, *args_gpu, gamma=0.99, lambda_=0.95)
        vec_dev_ms, _    = _device_profile(compiled_vec, *args_gpu, gamma=0.99, lambda_=0.95)
        loop_ms   = _bench_cpu(_ref_gae,               *args_gpu, gamma=0.99, lambda_=0.95)
        numpy_ms  = _bench_cpu(numpy_gae_cpu,          *args_gpu, gamma=0.99, lambda_=0.95)
        e2e_ms    = _bench_cpu(numpy_gae_np_to_triton, *args_np,  gamma=0.99, lambda_=0.95)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "assoc_ms": assoc_ms, "loop_ms": loop_ms,
            "numpy_ms": numpy_ms, "e2e_ms": e2e_ms,
            "su_vec":   vec_ms   / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
            "su_assoc": assoc_ms / triton_ms,
            "su_loop":  loop_ms  / triton_ms,
            "su_numpy": numpy_ms / triton_ms,
            "su_e2e":   numpy_ms / e2e_ms,
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>14} {f'{vec_ms/triton_ms:.2f}x':>8} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>12} "
            f"{f'{assoc_ms:.3f}ms':>15} {f'{assoc_ms/triton_ms:.2f}x':>10} "
            f"{f'{loop_ms:.3f}ms':>11} {f'{loop_ms/triton_ms:.1f}x':>9} "
            f"{f'{numpy_ms:.3f}ms':>12} {f'{numpy_ms/triton_ms:.1f}x':>10} "
            f"{f'{e2e_ms:.3f}ms':>13} {f'{numpy_ms/e2e_ms:.1f}x':>11}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print(f"  MONOTONICITY GATE FAILED (gae, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (gae, triton_ms, 2% band)", flush=True)

    production_rows = _bench_production_regime(
        "GAE", compute_gae, compiled_vec, _ref_gae, _make_gae,
        kwargs={"gamma": 0.99, "lambda_": 0.95},
    )
    prod_violations = check_monotonic_grid(
        [r for r in production_rows if not r["is_boundary"]], ms_key="triton_ms")
    if prod_violations:
        print("  MONOTONICITY GATE FAILED (gae, production regime, 2% band):", flush=True)
        for v in prod_violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (gae, production regime, 2% band)", flush=True)

    return rows, production_rows, violations + prod_violations


def bench_gae_truncation():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec_trunc = torch.compile(vectorized_gae_with_truncations)
    terminateds_warmup = _make_gae(64, 512)[2]
    truncateds_warmup, bootstrap_warmup = _make_trunc_extras(64, 512, terminateds_warmup)
    rewards_w, values_w, _ = _make_gae(64, 512)
    compiled_vec_trunc(rewards_w, values_w, terminateds_warmup, truncateds_warmup,
                        bootstrap_warmup, 0.99, 0.95)
    compute_gae(rewards_w, values_w, terminateds_warmup, truncateds_warmup,
                gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_warmup)
    torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec_trunc)':>20} {'speedup':>9} {'speedup(dev)':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, values, terminateds = _make_gae(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_gae_trunc(rewards, values, terminateds, truncateds, bootstrap_values,
                                  gamma=0.99, lambda_=0.95)
        triton_out = compute_gae(rewards, values, terminateds, truncateds,
                                  gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        vec_out = vectorized_gae_with_truncations(rewards, values, terminateds, truncateds,
                                                    bootstrap_values, 0.99, 0.95)
        assert_correctness(triton_out, ref_out, f"gae_trunc[{num_envs}x{seq_len}] (triton vs ref)")
        assert_correctness(vec_out, ref_out, f"gae_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

        _warmup_gpu(compute_gae, rewards, values, terminateds, truncateds,
                    gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        _warmup_gpu(compiled_vec_trunc, rewards, values, terminateds, truncateds,
                    bootstrap_values, gamma=0.99, lambda_=0.95)

        triton_ms = _bench_gpu(compute_gae, rewards, values, terminateds, truncateds,
                                gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values,
                                n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec_trunc, rewards, values, terminateds, truncateds,
                             bootstrap_values, gamma=0.99, lambda_=0.95, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_gae, rewards, values, terminateds, truncateds,
                                            gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        vec_dev_ms, _    = _device_profile(compiled_vec_trunc, rewards, values, terminateds,
                                            truncateds, bootstrap_values, gamma=0.99, lambda_=0.95)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "su_vec": vec_ms / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>19} {f'{vec_ms/triton_ms:.2f}x':>9} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>13}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (gae_trunc, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (gae_trunc, triton_ms, 2% band)", flush=True)
    return rows, violations


def bench_vtrace():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec  = torch.compile(vectorized_vtrace)
    compiled_assoc = torch.compile(vectorized_vtrace_with_truncations)
    args = _make_vtrace(64, 512)
    compiled_vec(*args, gamma=0.99); torch.cuda.synchronize()
    _log_pi_t0, _log_pi_b0, _values0, _rewards0, _terminateds0 = args
    _n0, _t0 = _rewards0.shape
    _trunc0_w = torch.zeros(_n0, _t0, device="cuda")
    _bsv0_w   = torch.zeros(_n0, _t0, device="cuda")
    _bsv0_w[:, -1] = torch.rand(_n0, device="cuda")
    compiled_assoc(_log_pi_t0, _log_pi_b0, _values0, _rewards0, _terminateds0,
                   _trunc0_w, _bsv0_w, gamma=0.99); torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'vs vec':>8} {'vs vec(dev)':>12} "
        f"{'compile(assoc)':>15} {'vs assoc':>10} "
        f"{'loop(gpu)':>11} {'vs loop':>9} {'numpy(cpu)':>12} {'vs numpy':>10} "
        f"{'np->tri->np':>13} {'e2e vs np':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        args_gpu = _make_vtrace(num_envs, seq_len)
        args_np  = tuple(t.cpu().numpy() for t in args_gpu)
        ni       = _n_iter_gpu(seq_len, num_envs)
        log_pi_t, log_pi_b, values, rewards, terminateds = args_gpu

        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0[:, -1] = torch.rand(num_envs, device="cuda")

        triton_out = compute_vtrace_fused(*args_gpu, gamma=0.99)
        ref_out    = _ref_vtrace(*args_gpu, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"vtrace[{num_envs}x{seq_len}]")

        _warmup_gpu(compute_vtrace_fused, *args_gpu, gamma=0.99)
        _warmup_gpu(compiled_vec,         *args_gpu, gamma=0.99)
        _warmup_gpu(compiled_assoc, log_pi_t, log_pi_b, values, rewards, terminateds,
                    trunc0, bsv0, gamma=0.99)
        triton_ms = _bench_gpu(compute_vtrace_fused, *args_gpu, gamma=0.99, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,         *args_gpu, gamma=0.99, n_iter=ni)
        assoc_ms  = _bench_gpu(compiled_assoc, log_pi_t, log_pi_b, values, rewards, terminateds,
                                trunc0, bsv0, gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_vtrace_fused, *args_gpu, gamma=0.99)
        vec_dev_ms, _    = _device_profile(compiled_vec,         *args_gpu, gamma=0.99)
        loop_ms   = _bench_cpu(_ref_vtrace,               *args_gpu, gamma=0.99)
        numpy_ms  = _bench_cpu(numpy_vtrace_cpu,          *args_gpu, gamma=0.99)
        e2e_ms    = _bench_cpu(numpy_vtrace_np_to_triton, *args_np,  gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "assoc_ms": assoc_ms, "loop_ms": loop_ms,
            "numpy_ms": numpy_ms, "e2e_ms": e2e_ms,
            "su_vec":   vec_ms   / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
            "su_assoc": assoc_ms / triton_ms,
            "su_loop":  loop_ms  / triton_ms,
            "su_numpy": numpy_ms / triton_ms,
            "su_e2e":   numpy_ms / e2e_ms,
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>14} {f'{vec_ms/triton_ms:.2f}x':>8} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>12} "
            f"{f'{assoc_ms:.3f}ms':>15} {f'{assoc_ms/triton_ms:.2f}x':>10} "
            f"{f'{loop_ms:.3f}ms':>11} {f'{loop_ms/triton_ms:.1f}x':>9} "
            f"{f'{numpy_ms:.3f}ms':>12} {f'{numpy_ms/triton_ms:.1f}x':>10} "
            f"{f'{e2e_ms:.3f}ms':>13} {f'{numpy_ms/e2e_ms:.1f}x':>11}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (vtrace, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (vtrace, triton_ms, 2% band)", flush=True)

    production_rows = _bench_production_regime(
        "V-Trace", compute_vtrace_fused, compiled_vec, _ref_vtrace, _make_vtrace,
        kwargs={"gamma": 0.99},
    )
    prod_violations = check_monotonic_grid(
        [r for r in production_rows if not r["is_boundary"]], ms_key="triton_ms")
    if prod_violations:
        print("  MONOTONICITY GATE FAILED (vtrace, production regime, 2% band):", flush=True)
        for v in prod_violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (vtrace, production regime, 2% band)", flush=True)

    return rows, production_rows, violations + prod_violations


def bench_vtrace_truncation():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec_trunc = torch.compile(vectorized_vtrace_with_truncations)

    log_pi_t_w, log_pi_b_w, values_w, rewards_w, terminateds_w = _make_vtrace(64, 512)
    truncateds_w, bootstrap_w = _make_trunc_extras(64, 512, terminateds_w)
    compiled_vec_trunc(log_pi_t_w, log_pi_b_w, values_w, rewards_w, terminateds_w,
                        truncateds_w, bootstrap_w, gamma=0.99)
    compute_vtrace_fused(log_pi_t_w, log_pi_b_w, values_w, rewards_w, terminateds_w,
                         truncateds=truncateds_w, gamma=0.99, bootstrap_values=bootstrap_w)
    torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec_trunc)':>20} {'speedup':>9} {'speedup(dev)':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        log_pi_t, log_pi_b, values, rewards, terminateds = _make_vtrace(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_vtrace_sequential(log_pi_t, log_pi_b, values, rewards, terminateds,
                                          truncateds, bootstrap_values, gamma=0.99)
        triton_out = compute_vtrace_fused(log_pi_t, log_pi_b, values, rewards, terminateds,
                                           truncateds=truncateds, gamma=0.99,
                                           bootstrap_values=bootstrap_values)
        vec_out = vectorized_vtrace_with_truncations(log_pi_t, log_pi_b, values, rewards,
                                                      terminateds, truncateds, bootstrap_values,
                                                      gamma=0.99)
        assert_correctness(triton_out, ref_out, f"vtrace_trunc[{num_envs}x{seq_len}] (triton vs ref)")
        assert_correctness(vec_out, ref_out, f"vtrace_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

        _warmup_gpu(compute_vtrace_fused, log_pi_t, log_pi_b, values, rewards, terminateds,
                    truncateds=truncateds, gamma=0.99, bootstrap_values=bootstrap_values)
        _warmup_gpu(compiled_vec_trunc, log_pi_t, log_pi_b, values, rewards, terminateds,
                    truncateds, bootstrap_values, gamma=0.99)

        triton_ms = _bench_gpu(compute_vtrace_fused, log_pi_t, log_pi_b, values, rewards, terminateds,
                                truncateds=truncateds, gamma=0.99, bootstrap_values=bootstrap_values,
                                n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec_trunc, log_pi_t, log_pi_b, values, rewards, terminateds,
                             truncateds, bootstrap_values, gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_vtrace_fused, log_pi_t, log_pi_b, values, rewards,
                                            terminateds, truncateds=truncateds, gamma=0.99,
                                            bootstrap_values=bootstrap_values)
        vec_dev_ms, _    = _device_profile(compiled_vec_trunc, log_pi_t, log_pi_b, values, rewards,
                                            terminateds, truncateds, bootstrap_values, gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "su_vec": vec_ms / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>19} {f'{vec_ms/triton_ms:.2f}x':>9} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>13}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (vtrace_trunc, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (vtrace_trunc, triton_ms, 2% band)", flush=True)
    return rows, violations


def _retrace_kernel_args(args_gpu):
    """Reorder _make_retrace's tuple into compute_retrace's positional order
    and add a zero truncateds tensor (no truncation by default).

    _make_retrace order: rewards, dones, values, next_q_all, q_values, actions, apt, apb
    compute_retrace order: action_probs_target, action_probs_behavior, q_values,
                            next_q_values_all, actions, rewards, terminateds, truncateds
    """
    rewards, dones, _values, next_q_all, q_values, actions, apt, apb = args_gpu
    truncateds = torch.zeros_like(dones)
    return apt, apb, q_values, next_q_all, actions, rewards, dones, truncateds


def bench_retrace():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec = torch.compile(_vec_retrace)
    args = _make_retrace(64, 512)
    compiled_vec(*args, gamma=0.99); torch.cuda.synchronize()
    compute_retrace(*_retrace_kernel_args(args), gamma=0.99); torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'vs vec':>8} {'vs vec(dev)':>12} "
        f"{'loop(gpu)':>11} {'vs loop':>9} "
        f"{'numpy(cpu)':>12} {'vs numpy':>10} {'np->tri->np':>13} {'e2e vs np':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        args_gpu = _make_retrace(num_envs, seq_len)
        rn, dn, qn, nqn, _qn2, an, aptn, apbn = tuple(t.cpu().numpy() for t in args_gpu)
        ni = _n_iter_gpu(seq_len, num_envs)

        retrace_kernel_args = _retrace_kernel_args(args_gpu)
        triton_out, _ = compute_retrace(*retrace_kernel_args, gamma=0.99)
        ref_out       = _ref_retrace(*args_gpu, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"retrace[{num_envs}x{seq_len}]")

        _warmup_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99)
        _warmup_gpu(compiled_vec,    *args_gpu, gamma=0.99)
        triton_ms = _bench_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,    *args_gpu,            gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_retrace, *retrace_kernel_args, gamma=0.99)
        vec_dev_ms, _    = _device_profile(compiled_vec,    *args_gpu,            gamma=0.99)
        loop_ms   = _bench_cpu(_ref_retrace,               *args_gpu, gamma=0.99)
        numpy_ms  = _bench_cpu(numpy_retrace_cpu,          rn, dn, qn, nqn, an, aptn, apbn, gamma=0.99)
        e2e_ms    = _bench_cpu(numpy_retrace_np_to_triton, rn, dn, qn, nqn, an, aptn, apbn, gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "loop_ms": loop_ms, "numpy_ms": numpy_ms, "e2e_ms": e2e_ms,
            "su_vec":   vec_ms   / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
            "su_loop":  loop_ms  / triton_ms,
            "su_numpy": numpy_ms / triton_ms,
            "su_e2e":   numpy_ms / e2e_ms,
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>14} {f'{vec_ms/triton_ms:.2f}x':>8} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>12} "
            f"{f'{loop_ms:.3f}ms':>11} {f'{loop_ms/triton_ms:.1f}x':>9} "
            f"{f'{numpy_ms:.3f}ms':>12} {f'{numpy_ms/triton_ms:.1f}x':>10} "
            f"{f'{e2e_ms:.3f}ms':>13} {f'{numpy_ms/e2e_ms:.1f}x':>11}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (retrace, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (retrace, triton_ms, 2% band)", flush=True)

    # Retrace's kernel args need reordering vs. the vec/ref call signature
    # (see _retrace_kernel_args), so it can't use the shared
    # _bench_production_regime helper — inline production sweep instead.
    production_rows = []
    for num_envs, seq_len in PRODUCTION_CONFIGS + [BOUNDARY_CONFIG]:
        is_boundary = (num_envs, seq_len) == BOUNDARY_CONFIG
        args_gpu = _make_retrace(num_envs, seq_len)
        retrace_kernel_args = _retrace_kernel_args(args_gpu)
        ni = _n_iter_gpu(seq_len, num_envs)

        triton_out, _ = compute_retrace(*retrace_kernel_args, gamma=0.99)
        ref_out       = _ref_retrace(*args_gpu, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"retrace[production,{num_envs}x{seq_len}]")

        _warmup_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99)
        _warmup_gpu(compiled_vec,    *args_gpu, gamma=0.99)
        triton_ms = _bench_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,    *args_gpu,            gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_retrace, *retrace_kernel_args, gamma=0.99)
        vec_dev_ms, _    = _device_profile(compiled_vec,    *args_gpu,            gamma=0.99)
        triton_amort_ms  = _bench_gpu_amortized(compute_retrace, *retrace_kernel_args, gamma=0.99)
        vec_amort_ms     = _bench_gpu_amortized(compiled_vec,    *args_gpu,            gamma=0.99)

        row = {
            "algo": "Retrace", "num_envs": num_envs, "seq_len": seq_len, "is_boundary": is_boundary,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms, "triton_amort_ms": triton_amort_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms, "vec_amort_ms": vec_amort_ms,
            "su_vec": vec_ms / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        }
        production_rows.append(row)
        tag = " [BOUNDARY]" if is_boundary else ""
        print(f"  [Retrace production {num_envs}x{seq_len}]{tag} "
              f"triton={triton_ms:.4f}ms dev={triton_dev_ms:.4f}ms amort={triton_amort_ms:.4f}ms  "
              f"vec={vec_ms:.4f}ms  su={row['su_vec']:.2f}x su_dev={row['su_vec_dev']:.2f}x",
              flush=True)

    prod_violations = check_monotonic_grid(
        [r for r in production_rows if not r["is_boundary"]], ms_key="triton_ms")
    if prod_violations:
        print("  MONOTONICITY GATE FAILED (retrace, production regime, 2% band):", flush=True)
        for v in prod_violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (retrace, production regime, 2% band)", flush=True)

    return rows, production_rows, violations + prod_violations


def bench_returns():
    print("  compiling torch.compile baselines …", end="", flush=True)
    c_lambda_vec  = torch.compile(vectorized_lambda_returns)
    c_disc_vec    = torch.compile(vectorized_discounted_returns)
    c_traces_vec  = torch.compile(vectorized_eligibility_traces)
    c_lambda_assoc = torch.compile(vectorized_lambda_returns_with_truncations)
    c_disc_assoc   = torch.compile(vectorized_discounted_returns_with_truncations)
    r, nv, d = _make_returns(64, 512)
    c_lambda_vec(r, nv, d, gamma=0.99, lambda_=0.95); torch.cuda.synchronize()
    c_disc_vec(r, d, gamma=0.99);                     torch.cuda.synchronize()
    c_traces_vec(r, d, gamma=0.99, lambda_=0.95);     torch.cuda.synchronize()
    _n0, _t0 = r.shape
    _trunc0_w = torch.zeros(_n0, _t0, device="cuda")
    _bsv0_w   = torch.zeros(_n0, _t0, device="cuda")
    _bsv0_w[:, -1] = torch.rand(_n0, device="cuda")
    c_lambda_assoc(r, nv, d, _trunc0_w, _bsv0_w, gamma=0.99, lambda_=0.95); torch.cuda.synchronize()
    c_disc_assoc(r, d, _trunc0_w, _bsv0_w, gamma=0.99);                     torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'algo':>20} {'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'compile(assoc)':>15} {'loop(gpu)':>11} "
        f"{'numpy(cpu)':>12} {'vs vec':>8} {'vs vec(dev)':>12} {'vs assoc':>10} {'vs loop':>9} {'vs numpy':>10}"
    )
    print(header)
    print("-" * len(header))

    def _print_sub_row(name, num_envs, seq_len, triton_ms, triton_dev_ms, vec_ms, vec_dev_ms,
                        loop_ms, numpy_ms, assoc_ms=None):
        assoc_str = f"{assoc_ms:.3f}ms" if assoc_ms is not None else "n/a"
        assoc_su  = f"{assoc_ms/triton_ms:.2f}x" if assoc_ms is not None else "n/a"
        print(
            f"{name:>20} {num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>14} {assoc_str:>15} "
            f"{f'{loop_ms:.3f}ms':>11} {f'{numpy_ms:.3f}ms':>12} "
            f"{f'{vec_ms/triton_ms:.2f}x':>8} {f'{vec_dev_ms/triton_dev_ms:.2f}x':>12} {assoc_su:>10} "
            f"{f'{loop_ms/triton_ms:.1f}x':>9} {f'{numpy_ms/triton_ms:.1f}x':>10}",
            flush=True,
        )

    rows_lambda, rows_disc, rows_traces = [], [], []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, next_values, dones = _make_returns(num_envs, seq_len)
        ni = _n_iter_gpu(seq_len, num_envs)

        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0[:, -1] = torch.rand(num_envs, device="cuda")

        lam_out  = compute_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        disc_out = compute_discounted_returns(rewards, dones, gamma=0.99)
        trc_out  = compute_eligibility_traces(rewards, dones, gamma=0.99, lambda_=0.9)
        assert_correctness(lam_out,  _ref_lambda(rewards, next_values, dones, gamma=0.99, lambda_=0.95),
                            f"lambda_returns[{num_envs}x{seq_len}]")
        assert_correctness(disc_out, _ref_disc(rewards, dones, gamma=0.99),
                            f"discounted_returns[{num_envs}x{seq_len}]")
        assert_correctness(trc_out,  _ref_traces(rewards, dones, gamma=0.99, lambda_=0.9),
                            f"eligibility_traces[{num_envs}x{seq_len}]")

        _warmup_gpu(compute_lambda_returns,     rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compute_discounted_returns,  rewards, dones, gamma=0.99)
        _warmup_gpu(compute_eligibility_traces,  rewards, dones, gamma=0.99, lambda_=0.9)
        _warmup_gpu(c_lambda_vec,  rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        _warmup_gpu(c_disc_vec,    rewards, dones, gamma=0.99)
        _warmup_gpu(c_traces_vec,  rewards, dones, gamma=0.99, lambda_=0.9)
        _warmup_gpu(c_lambda_assoc, rewards, next_values, dones, trunc0, bsv0, gamma=0.99, lambda_=0.95)
        _warmup_gpu(c_disc_assoc,   rewards, dones, trunc0, bsv0, gamma=0.99)

        lam_ms  = _bench_gpu(compute_lambda_returns,    rewards, next_values, dones,
                              gamma=0.99, lambda_=0.95, n_iter=ni)
        disc_ms = _bench_gpu(compute_discounted_returns, rewards, dones, gamma=0.99, n_iter=ni)
        trc_ms  = _bench_gpu(compute_eligibility_traces, rewards, dones,
                              gamma=0.99, lambda_=0.9, n_iter=ni)
        lam_vec_ms    = _bench_gpu(c_lambda_vec,   rewards, next_values, dones,
                                   gamma=0.99, lambda_=0.95, n_iter=ni)
        disc_vec_ms   = _bench_gpu(c_disc_vec,     rewards, dones, gamma=0.99, n_iter=ni)
        trc_vec_ms    = _bench_gpu(c_traces_vec,   rewards, dones, gamma=0.99, lambda_=0.9, n_iter=ni)
        lam_assoc_ms  = _bench_gpu(c_lambda_assoc, rewards, next_values, dones, trunc0, bsv0,
                                   gamma=0.99, lambda_=0.95, n_iter=ni)
        disc_assoc_ms = _bench_gpu(c_disc_assoc,   rewards, dones, trunc0, bsv0, gamma=0.99, n_iter=ni)
        lam_dev_ms, _  = _device_profile(compute_lambda_returns, rewards, next_values, dones,
                                          gamma=0.99, lambda_=0.95)
        disc_dev_ms, _ = _device_profile(compute_discounted_returns, rewards, dones, gamma=0.99)
        trc_dev_ms, _  = _device_profile(compute_eligibility_traces, rewards, dones,
                                          gamma=0.99, lambda_=0.9)
        lam_vec_dev_ms, _  = _device_profile(c_lambda_vec, rewards, next_values, dones,
                                              gamma=0.99, lambda_=0.95)
        disc_vec_dev_ms, _ = _device_profile(c_disc_vec, rewards, dones, gamma=0.99)
        trc_vec_dev_ms, _  = _device_profile(c_traces_vec, rewards, dones, gamma=0.99, lambda_=0.9)
        lam_loop_ms  = _bench_cpu(_ref_lambda, rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        disc_loop_ms = _bench_cpu(_ref_disc,   rewards, dones, gamma=0.99)
        trc_loop_ms  = _bench_cpu(_ref_traces, rewards, dones, gamma=0.99, lambda_=0.9)
        lam_np_ms    = _bench_cpu(numpy_lambda_returns_cpu,     rewards, next_values, dones,
                                   gamma=0.99, lambda_=0.95)
        disc_np_ms   = _bench_cpu(numpy_discounted_returns_cpu, rewards, dones, gamma=0.99)
        trc_np_ms    = _bench_cpu(numpy_eligibility_traces_cpu, rewards, dones,
                                   gamma=0.99, lambda_=0.9)

        base = {"num_envs": num_envs, "seq_len": seq_len}
        rows_lambda.append({
            **base, "triton_ms": lam_ms, "triton_dev_ms": lam_dev_ms,
            "vec_ms": lam_vec_ms, "vec_dev_ms": lam_vec_dev_ms, "assoc_ms": lam_assoc_ms,
            "loop_ms": lam_loop_ms, "numpy_ms": lam_np_ms,
            "su_vec": lam_vec_ms / lam_ms,
            "su_vec_dev": lam_vec_dev_ms / lam_dev_ms if lam_dev_ms else float("nan"),
            "su_assoc": lam_assoc_ms / lam_ms,
            "su_loop": lam_loop_ms / lam_ms, "su_numpy": lam_np_ms / lam_ms,
        })
        rows_disc.append({
            **base, "triton_ms": disc_ms, "triton_dev_ms": disc_dev_ms,
            "vec_ms": disc_vec_ms, "vec_dev_ms": disc_vec_dev_ms, "assoc_ms": disc_assoc_ms,
            "loop_ms": disc_loop_ms, "numpy_ms": disc_np_ms,
            "su_vec": disc_vec_ms / disc_ms,
            "su_vec_dev": disc_vec_dev_ms / disc_dev_ms if disc_dev_ms else float("nan"),
            "su_assoc": disc_assoc_ms / disc_ms,
            "su_loop": disc_loop_ms / disc_ms, "su_numpy": disc_np_ms / disc_ms,
        })
        rows_traces.append({
            **base, "triton_ms": trc_ms, "triton_dev_ms": trc_dev_ms,
            "vec_ms": trc_vec_ms, "vec_dev_ms": trc_vec_dev_ms,
            "loop_ms": trc_loop_ms, "numpy_ms": trc_np_ms,
            "su_vec": trc_vec_ms / trc_ms,
            "su_vec_dev": trc_vec_dev_ms / trc_dev_ms if trc_dev_ms else float("nan"),
            "su_loop": trc_loop_ms / trc_ms, "su_numpy": trc_np_ms / trc_ms,
        })
        _print_sub_row("lambda-returns",     num_envs, seq_len, lam_ms,  lam_dev_ms,  lam_vec_ms,  lam_vec_dev_ms,  lam_loop_ms,  lam_np_ms,  lam_assoc_ms)
        _print_sub_row("discounted-returns", num_envs, seq_len, disc_ms, disc_dev_ms, disc_vec_ms, disc_vec_dev_ms, disc_loop_ms, disc_np_ms, disc_assoc_ms)
        _print_sub_row("eligibility-traces", num_envs, seq_len, trc_ms,  trc_dev_ms,  trc_vec_ms,  trc_vec_dev_ms,  trc_loop_ms,  trc_np_ms)

    violations = []
    for label, rows, ms_key in [("lambda_returns", rows_lambda, "triton_ms"),
                                 ("discounted_returns", rows_disc, "triton_ms"),
                                 ("eligibility_traces", rows_traces, "triton_ms")]:
        v = check_monotonic_grid(rows, ms_key=ms_key)
        if v:
            print(f"  MONOTONICITY GATE FAILED ({label}, triton_ms, 2% band):", flush=True)
            for line in v:
                print(f"    - {line}", flush=True)
        else:
            print(f"  monotonicity gate: PASSED ({label}, triton_ms, 2% band)", flush=True)
        violations += v

    def _make_returns_2(num_envs, seq_len, device="cuda"):
        r, _nv, d = _make_returns(num_envs, seq_len, device)
        return r, d

    production_rows = []
    production_rows += _bench_production_regime(
        "lambda-returns", compute_lambda_returns, c_lambda_vec, _ref_lambda, _make_returns,
        kwargs={"gamma": 0.99, "lambda_": 0.95},
    )
    production_rows += _bench_production_regime(
        "discounted-returns", compute_discounted_returns, c_disc_vec, _ref_disc, _make_returns_2,
        kwargs={"gamma": 0.99},
    )
    production_rows += _bench_production_regime(
        "eligibility-traces", compute_eligibility_traces, c_traces_vec, _ref_traces, _make_returns_2,
        kwargs={"gamma": 0.99, "lambda_": 0.9},
    )
    for label in ("lambda-returns", "discounted-returns", "eligibility-traces"):
        algo_rows = [r for r in production_rows if r["algo"] == label and not r["is_boundary"]]
        v = check_monotonic_grid(algo_rows, ms_key="triton_ms")
        if v:
            print(f"  MONOTONICITY GATE FAILED ({label}, production regime, 2% band):", flush=True)
            for line in v:
                print(f"    - {line}", flush=True)
        else:
            print(f"  monotonicity gate: PASSED ({label}, production regime, 2% band)", flush=True)
        violations += v

    return rows_lambda, rows_disc, rows_traces, production_rows, violations


def bench_lambda_returns_truncation():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec_trunc = torch.compile(vectorized_lambda_returns_with_truncations)

    rewards_w, next_values_w, terminateds_w = _make_returns(64, 512)
    truncateds_w, bootstrap_w = _make_trunc_extras(64, 512, terminateds_w)
    compiled_vec_trunc(rewards_w, next_values_w, terminateds_w, truncateds_w, bootstrap_w,
                        gamma=0.99, lambda_=0.95)
    compute_lambda_returns(rewards_w, next_values_w, terminateds_w, truncateds=truncateds_w,
                            gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_w)
    torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec_trunc)':>20} {'speedup':>9} {'speedup(dev)':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, next_values, terminateds = _make_returns(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_lambda_sequential(rewards, next_values, terminateds, truncateds,
                                          bootstrap_values, gamma=0.99, lambda_=0.95)
        triton_out = compute_lambda_returns(rewards, next_values, terminateds, truncateds=truncateds,
                                             gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        vec_out = vectorized_lambda_returns_with_truncations(rewards, next_values, terminateds,
                                                              truncateds, bootstrap_values,
                                                              gamma=0.99, lambda_=0.95)
        assert_correctness(triton_out, ref_out, f"lambda_trunc[{num_envs}x{seq_len}] (triton vs ref)")
        assert_correctness(vec_out, ref_out, f"lambda_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

        _warmup_gpu(compute_lambda_returns, rewards, next_values, terminateds,
                    truncateds=truncateds, gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        _warmup_gpu(compiled_vec_trunc, rewards, next_values, terminateds, truncateds,
                    bootstrap_values, gamma=0.99, lambda_=0.95)

        triton_ms = _bench_gpu(compute_lambda_returns, rewards, next_values, terminateds,
                                truncateds=truncateds, gamma=0.99, lambda_=0.95,
                                bootstrap_values=bootstrap_values, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec_trunc, rewards, next_values, terminateds, truncateds,
                             bootstrap_values, gamma=0.99, lambda_=0.95, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_lambda_returns, rewards, next_values, terminateds,
                                            truncateds=truncateds, gamma=0.99, lambda_=0.95,
                                            bootstrap_values=bootstrap_values)
        vec_dev_ms, _    = _device_profile(compiled_vec_trunc, rewards, next_values, terminateds,
                                            truncateds, bootstrap_values, gamma=0.99, lambda_=0.95)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "su_vec": vec_ms / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>19} {f'{vec_ms/triton_ms:.2f}x':>9} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>13}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (lambda_trunc, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (lambda_trunc, triton_ms, 2% band)", flush=True)
    return rows, violations


def bench_discounted_returns_truncation():
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec_trunc = torch.compile(vectorized_discounted_returns_with_truncations)

    rewards_w, _, terminateds_w = _make_returns(64, 512)
    truncateds_w, bootstrap_w = _make_trunc_extras(64, 512, terminateds_w)
    compiled_vec_trunc(rewards_w, terminateds_w, truncateds_w, bootstrap_w, gamma=0.99)
    compute_discounted_returns(rewards_w, terminateds_w, truncateds=truncateds_w,
                                gamma=0.99, bootstrap_values=bootstrap_w)
    torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec_trunc)':>20} {'speedup':>9} {'speedup(dev)':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, _, terminateds = _make_returns(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_discounted_sequential(rewards, terminateds, truncateds, bootstrap_values,
                                              gamma=0.99)
        triton_out = compute_discounted_returns(rewards, terminateds, truncateds=truncateds,
                                                  gamma=0.99, bootstrap_values=bootstrap_values)
        vec_out = vectorized_discounted_returns_with_truncations(rewards, terminateds, truncateds,
                                                                   bootstrap_values, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"disc_trunc[{num_envs}x{seq_len}] (triton vs ref)")
        assert_correctness(vec_out, ref_out, f"disc_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

        _warmup_gpu(compute_discounted_returns, rewards, terminateds,
                    truncateds=truncateds, gamma=0.99, bootstrap_values=bootstrap_values)
        _warmup_gpu(compiled_vec_trunc, rewards, terminateds, truncateds, bootstrap_values, gamma=0.99)

        triton_ms = _bench_gpu(compute_discounted_returns, rewards, terminateds,
                                truncateds=truncateds, gamma=0.99,
                                bootstrap_values=bootstrap_values, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec_trunc, rewards, terminateds, truncateds, bootstrap_values,
                             gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_discounted_returns, rewards, terminateds,
                                            truncateds=truncateds, gamma=0.99,
                                            bootstrap_values=bootstrap_values)
        vec_dev_ms, _    = _device_profile(compiled_vec_trunc, rewards, terminateds, truncateds,
                                            bootstrap_values, gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "su_vec": vec_ms / triton_ms,
            "su_vec_dev": vec_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>19} {f'{vec_ms/triton_ms:.2f}x':>9} "
            f"{f'{vec_dev_ms/triton_dev_ms:.2f}x':>13}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (disc_trunc, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (disc_trunc, triton_ms, 2% band)", flush=True)
    return rows, violations


def _make_prefix_sum(num_envs, seq_len, device="cuda"):
    torch.manual_seed(0)
    inputs = torch.randn(num_envs, seq_len, device=device)
    dones  = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return inputs, dones


def bench_prefix_sum():
    from test_prefix_sum import reference_episodic_prefix_sum, vectorized_episodic_prefix_sum
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled = torch.compile(vectorized_episodic_prefix_sum)
    r, d = _make_prefix_sum(64, 512)
    compiled(r, d); torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>10} {'dev':>8} {'compile(vec)':>14} {'vs vec':>8} {'vs vec(dev)':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        inputs, dones = _make_prefix_sum(num_envs, seq_len)
        ni = _n_iter_gpu(seq_len, num_envs)

        triton_out = compute_episodic_prefix_sum(inputs, dones)
        ref_out    = reference_episodic_prefix_sum(inputs, dones)
        assert_correctness(triton_out, ref_out, f"prefix_sum[{num_envs}x{seq_len}]")

        _warmup_gpu(compute_episodic_prefix_sum, inputs, dones)
        _warmup_gpu(compiled, inputs, dones)
        # NOTE: the compiled baseline is timed with _bench_gpu (CUDA events,
        # explicit sync), not _bench_cpu — _bench_cpu never syncs the GPU, so
        # timing an async-dispatching compiled fn with it would silently
        # measure only CPU-side launch time and understate the baseline.
        triton_ms   = _bench_gpu(compute_episodic_prefix_sum, inputs, dones, n_iter=ni)
        compiled_ms = _bench_gpu(compiled, inputs, dones, n_iter=ni)
        triton_dev_ms, _   = _device_profile(compute_episodic_prefix_sum, inputs, dones)
        compiled_dev_ms, _ = _device_profile(compiled, inputs, dones)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "compiled_ms": compiled_ms, "compiled_dev_ms": compiled_dev_ms,
            "su_compile": compiled_ms / triton_ms,
            "su_compile_dev": compiled_dev_ms / triton_dev_ms if triton_dev_ms else float("nan"),
        })
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>10} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{compiled_ms:.3f}ms':>14} {f'{compiled_ms/triton_ms:.2f}x':>8} "
            f"{f'{compiled_dev_ms/triton_dev_ms:.2f}x':>12}",
            flush=True,
        )

    violations = check_monotonic_grid(rows, ms_key="triton_ms")
    if violations:
        print("  MONOTONICITY GATE FAILED (prefix_sum, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (prefix_sum, triton_ms, 2% band)", flush=True)

    production_rows = _bench_production_regime(
        "prefix-sum", compute_episodic_prefix_sum, compiled, reference_episodic_prefix_sum,
        _make_prefix_sum, kwargs={},
    )
    # _bench_production_regime's generic field names are "vec_ms" etc.; rename
    # for prefix-sum's single-baseline table via the su_compile key at format time.
    prod_violations = check_monotonic_grid(
        [r for r in production_rows if not r["is_boundary"]], ms_key="triton_ms")
    if prod_violations:
        print("  MONOTONICITY GATE FAILED (prefix_sum, production regime, 2% band):", flush=True)
        for v in prod_violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (prefix_sum, production regime, 2% band)", flush=True)

    return rows, production_rows, violations + prod_violations


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def _fmt_row_numpy(r, include_assoc=False):
    """Row for GAE / V-Trace: triton | dev | vec | vs vec | vs vec(dev) | [assoc | vs assoc] | loop | vs loop | numpy | vs numpy | e2e | e2e vs numpy."""
    base = (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['vec_ms']:>13.3f} | {r['su_vec']:>6.1f}x | {r['su_vec_dev']:>6.1f}x |")
    if include_assoc:
        base += f" {r['assoc_ms']:>14.3f} | {r['su_assoc']:>8.1f}x |"
    base += (f" {r['loop_ms']:>10.3f} | {r['su_loop']:>7.1f}x |"
             f" {r['numpy_ms']:>10.3f} | {r['su_numpy']:>8.1f}x |"
             f" {r['e2e_ms']:>14.3f} | {r['su_e2e']:>7.1f}x |")
    return base


def _fmt_row_simple(r, include_assoc=False):
    """Row for λ-returns / discounted-returns / eligibility-traces: triton | dev | vec | vs vec | vs vec(dev) | [assoc] | loop | numpy."""
    base = (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['vec_ms']:>13.3f} | {r['su_vec']:>6.1f}x | {r['su_vec_dev']:>6.1f}x |")
    if include_assoc:
        base += f" {r['assoc_ms']:>14.3f} | {r['su_assoc']:>8.1f}x |"
    base += f" {r['loop_ms']:>10.3f} | {r['su_loop']:>7.1f}x | {r['numpy_ms']:>10.3f} | {r['su_numpy']:>8.1f}x |"
    return base


def _fmt_row_retrace(r):
    """Row for Retrace: triton | dev | vec | vs vec | vs vec(dev) | loop | vs loop | numpy | vs numpy | e2e | e2e vs numpy."""
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['vec_ms']:>13.3f} | {r['su_vec']:>6.1f}x | {r['su_vec_dev']:>6.1f}x |"
            f" {r['loop_ms']:>10.3f} | {r['su_loop']:>7.1f}x |"
            f" {r['numpy_ms']:>10.3f} | {r['su_numpy']:>8.1f}x |"
            f" {r['e2e_ms']:>14.3f} | {r['su_e2e']:>7.1f}x |")


def _fmt_row_prefix(r):
    """Row for episodic prefix sum: triton | dev | compile(vec) | vs vec | vs vec(dev)."""
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['compiled_ms']:>13.3f} | {r['su_compile']:>6.1f}x | {r['su_compile_dev']:>6.1f}x |")


def _table_numpy(title, rows, include_assoc=False):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | vs vec (full-call) | vs vec (device) |")
    sep_cols    = "|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|"
    if include_assoc:
        header_cols += " compile(assoc) (ms) | vs assoc |"
        sep_cols    += ":-------------------:|:--------:|"
    header_cols += " loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |"
    sep_cols    += ":-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|"
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_numpy(r, include_assoc=include_assoc) for r in rows)
    return header + "\n" + body


def _table_simple(title, rows, include_assoc=False):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | vs vec (full-call) | vs vec (device) |")
    sep_cols    = "|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|"
    if include_assoc:
        header_cols += " compile(assoc) (ms) | vs assoc |"
        sep_cols    += ":-------------------:|:--------:|"
    header_cols += " loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |"
    sep_cols    += ":-------------:|:-------:|:--------------:|:--------:|"
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_simple(r, include_assoc=include_assoc) for r in rows)
    return header + "\n" + body


def _table_retrace(title, rows):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | vs vec (full-call) | vs vec (device) |"
                   " loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |")
    sep_cols    = ("|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|"
                   ":-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|")
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_retrace(r) for r in rows)
    return header + "\n" + body


def _table_prefix_sum(title, rows):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | vs vec (full-call) | vs vec (device) |")
    sep_cols    = "|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|"
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_prefix(r) for r in rows)
    return header + "\n" + body


def _fmt_trunc_row(r):
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} | {r['vec_ms']:>17.3f} "
            f"| {r['su_vec']:>9.1f}x | {r['su_vec_dev']:>9.1f}x |")


def _table_truncation(title, rows):
    header = (
        f"#### {title}\n\n"
        "| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |\n"
        "|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------:|:----------------------:|"
    )
    body = "\n".join(_fmt_trunc_row(r) for r in rows)
    return header + "\n" + body


def _fmt_row_production(r):
    marker = " ⚠️" if r["is_boundary"] else ""
    return (f"| {r['algo']:<20} | {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.4f} | {r['triton_dev_ms']:>10.4f} | {r['triton_amort_ms']:>10.4f} "
            f"| {r['vec_ms']:>13.4f} | {r['su_vec']:>6.2f}x | {r['su_vec_dev']:>6.2f}x |{marker}")


def _table_production(title, rows):
    """ONE combined table across all 7 algorithms (algo is a row field, not a
    separate table per algorithm) for the production regime (seq_len [80,128]
    x num_envs [4096,8192,16384,32768,38400]) plus the boundary-marker row
    (num_envs=16384, seq_len=16, flagged with a warning marker)."""
    header_cols = (
        "| algo | num_envs | seq_len | triton full-call (ms) | triton device (ms) | "
        "triton amortized (ms) | compile(vec) full-call (ms) | vs vec (full-call) | vs vec (device) |"
    )
    sep_cols = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_production(r) for r in rows)
    footnote = (
        "\n\n*⚠️ marks the boundary-marker row (num_envs=16384, seq_len=16). "
        "**vs vec (full-call)** is the headline ratio — the complete "
        "`compute_*(tensors) -> tensors` call including launch/wrapper overhead, "
        "which a caller pays every invocation. **vs vec (device)** is a diagnostic "
        "showing the same ratio for CUDA-kernel-only time; where full-call and device "
        "speedups diverge, the gap is launch + wrapper overhead. **triton amortized** "
        "is N calls timed inside one region (separates harness per-call sync overhead "
        "from genuine per-call cost) — reported alongside, not used for any ratio.*"
    )
    return header + "\n" + body + footnote


def _headline(rows):
    """Filter a rows list down to HEADLINE_CONFIGS for benchmarks.md's headline
    tables (~4 representative sizes/algorithm) — the full CONFIGS grid stays
    reproducible via `python tests/bench_release.py` and is not duplicated here."""
    wanted = set(HEADLINE_CONFIGS)
    return [r for r in rows if (r["num_envs"], r["seq_len"]) in wanted]


def _methodology_text(gpu_label: str) -> str:
    """Shared methodology header — used by both _section() (README/console)
    and update_benchmarks_md() (benchmarks.md), so benchmarks.md carries the
    full dtype/gamma-lambda/truncation-density/harness explanation too, not
    just the README's copy of it."""
    date = datetime.date.today().isoformat()
    gpu  = gpu_label or _detect_gpu()
    return (
        f"*Measured on {gpu} · {date} · "
        f"[`triton`](https://github.com/openai/triton) kernels vs `torch.compile` baselines and NumPy CPU.*\n\n"
        f"**Configuration.**  "
        f"dtype float32 (all kernels require it; see NOTES.md on bf16 and autocast).  "
        f"gamma=0.99, lambda=0.95 (lambda=0.9 for eligibility traces).  "
        f"Termination probability ~5% per step; truncation-path tables additionally inject "
        f"~5% interior truncated steps (mutually exclusive with terminations) with populated "
        f"`bootstrap_values`.\n\n"
        f"**Methodology.**  "
        f"All GPU full-call timings use CUDA events (start/stop around the complete "
        f"`compute_*(tensors) -> tensors` call, explicit sync immediately before start); "
        f"reported value is the min-of-medians across 5 independent trials to filter clock-state "
        f"noise. Every config is warmed up at its exact shape (20 untimed calls) before any timed "
        f"call, so `torch.compile` JIT/autotuning and Triton kernel compilation never land in the "
        f"timed region. A tolerance-based correctness gate (atol=rtol=1e-4 vs. a sequential "
        f"reference implementation) runs before every timed config — not bit-identical, since "
        f"`tl.associative_scan` reorders float ops depending on num_warps/block layout, so "
        f"cross-config last-bit differences are legitimate. A monotonicity gate (2% band) then "
        f"asserts a larger problem never measures faster than a smaller one along either swept "
        f"axis. CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.\n\n"
        f"**Two timing granularities.**  "
        f"**triton** (headline) is full-call wall time — what a caller pays every invocation, "
        f"including launch overhead and wrapper setup (HAS_TRUNCATIONS/HAS_BOOTSTRAP dispatch, "
        f"allocation, layout). All speedup ratios are computed from this number. **dev** is "
        f"device-only CUDA time (`torch.profiler` CUDA activity around steady-state calls, "
        f"ncu/nsys being unavailable in typical containerized GPU environments) — a diagnostic "
        f"showing pure kernel execution time; where dev is much smaller than the full-call number, "
        f"the gap is launch + wrapper overhead the caller still pays. The production-regime table "
        f"additionally reports an **amortized** variant (N calls in one timed region) for its "
        f"short-seq_len rows, to separate harness per-call sync overhead from genuine per-call "
        f"cost — the single-call full-call number remains the ratio basis throughout.\n\n"
        f"**Columns.**  "
        f"**triton**: full-call wall time, headline (CUDA events).  "
        f"**dev**: device-only kernel time, diagnostic (see above).  "
        f"**compile(vec)**: `torch.compile` applied to the fastest hand-vectorized PyTorch "
        f"equivalent – cumsum / suffix-product implementations with no Python loop; "
        f"this is the strongest GPU baseline a competent engineer would write.  "
        f"**compile(assoc)**: `torch.compile` applied to the unified associative-scan "
        f"implementation that also handles truncations – called here with zero truncations "
        f"to show the cost of a single regime-agnostic implementation vs the specialized "
        f"no-truncation path (GAE, V-Trace, λ-returns, discounted returns only).  "
        f"**compile(vec-trunc)**: `torch.compile` of the vectorized truncation baseline, "
        f"used in the with-truncations tables (itself asserted correct against the sequential "
        f"truncation reference before being trusted as a baseline).  "
        f"**loop (gpu)**: uncompiled sequential Python loop dispatching GPU ops – "
        f"the pattern used by CleanRL, RLlib, and most RL codebases today; "
        f"no `torch.compile`, no vectorization; wall-clock timing.  "
        f"**np→triton→np**: end-to-end wall-clock for the NumPy adoption path "
        f"(CPU → GPU transfer, kernel, GPU → CPU transfer).  "
        f"**numpy cpu**: sequential NumPy loop on CPU – same algorithm as the kernel, "
        f"no GPU; establishes the CPU reference for each algorithm.  "
        f"Headline tables below show {len(HEADLINE_CONFIGS)} representative sizes per algorithm "
        f"(small/parity, mid, main-grid-large, production-adjacent-large); the full CONFIGS grid "
        f"is reproducible via `python tests/bench_release.py`.\n"
    )


def _section(gpu_label: str, tables: list[str]) -> str:
    header = f"<!-- BENCH_START -->\n## Performance\n\n" + _methodology_text(gpu_label)
    return header + "\n\n".join(tables) + "\n<!-- BENCH_END -->"


def _detect_gpu() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown GPU"


# ---------------------------------------------------------------------------
# README updater
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).parent.parent
README        = REPO_ROOT / "README.md"
BENCHMARKS_MD = REPO_ROOT / "benchmarks.md"

_BENCH_RE = re.compile(
    r"<!-- BENCH_START -->.*?<!-- BENCH_END -->",
    re.DOTALL,
)


def update_readme(section: str) -> None:
    text = README.read_text()
    if _BENCH_RE.search(text):
        new_text = _BENCH_RE.sub(section, text)
    else:
        new_text = text.rstrip() + "\n\n" + section + "\n"
    README.write_text(new_text)
    print(f"\nREADME.md updated ({README})")


BENCHMARK_HISTORY_DIR = REPO_ROOT / "docs" / "benchmark-history"

_RELEASE_SECTION_RE = re.compile(r"## [^\n]*\n.*?(?=\n## |\Z)", re.DOTALL)


def update_benchmarks_md(version: str, gpu_label: str, tables: list[str]) -> None:
    """REPLACE the current-release section in benchmarks.md with this run's
    results; archive whatever release section was there before to
    docs/benchmark-history/<version>.md instead of prepending in-file.

    benchmarks.md holds ONLY the latest release, in full — it must not bloat
    across releases. Prior releases remain reachable via the archive
    directory rather than accumulating in one ever-growing file.
    """
    date = datetime.date.today().isoformat()
    gpu  = gpu_label or _detect_gpu()
    heading = f"## {version} – {date} – {gpu}\n\n" + _methodology_text(gpu_label)
    body    = "\n\n".join(tables) + "\n"
    new_section = heading + body

    if BENCHMARKS_MD.exists():
        existing = BENCHMARKS_MD.read_text()
    else:
        existing = "# Benchmarks\n\nLatest release only — see docs/benchmark-history/ for prior releases.\n\n"

    # Archive whatever release section currently occupies the file (if any)
    # before overwriting it. The fixed preamble is rewritten from scratch each
    # time (not preserved from `existing`), so any stale leftover content
    # before the first "## " heading — e.g. a "no benchmarks recorded yet"
    # placeholder — does not linger once a real release lands; benchmarks.md
    # holds ONLY the latest release, in full, nothing else.
    preamble = "# Benchmarks\n\nLatest release only — see docs/benchmark-history/ for prior releases.\n\n"
    prior_match = _RELEASE_SECTION_RE.search(existing)
    if prior_match:
        prior_text = prior_match.group(0)
        prior_version_match = re.match(r"## ([^\s]+)", prior_text)
        prior_version = prior_version_match.group(1) if prior_version_match else "unknown"
        BENCHMARK_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = BENCHMARK_HISTORY_DIR / f"{prior_version}.md"
        archive_path.write_text(f"# Benchmarks archive: {prior_version}\n\n" + prior_text)
        print(f"  (archived prior release '{prior_version}' -> {archive_path})")
    updated = preamble + new_section

    BENCHMARKS_MD.write_text(updated)
    print(f"\nbenchmarks.md updated ({BENCHMARKS_MD}) — replaced current-release section only")


def render_readme_table_draft(gpu_label: str, production_rows: list[dict]) -> str:
    """Render the README's ~4-row H100 production-relevant summary table WITHOUT
    touching README.md — README prose edits are a STOP-and-report item under the
    autonomy boundary. Caller should write this to a draft file for human review
    and manual inclusion in README.md, never call update_readme() with it directly
    during an unattended run.

    All rows use the SAME (num_envs, seq_len) so the table is a genuine
    apples-to-apples comparison across algorithms — mixing different sizes per
    row (an earlier version of this function did) makes speedups incomparable
    and risks reading as cherry-picked even when it isn't.
    """
    gpu = gpu_label or _detect_gpu()
    date = datetime.date.today().isoformat()
    fixed_num_envs, fixed_seq_len = 4096, 128  # PufferLib/Gigaflow-default rollout size
    algos = ["GAE", "V-Trace", "Retrace", "lambda-returns"]
    lines = [
        f"<!-- README_BENCH_DRAFT: NOT auto-applied. Prepared {date} on {gpu}. -->",
        "",
        f"All rows at num_envs={fixed_num_envs}, seq_len={fixed_seq_len} (PufferLib/Gigaflow-default "
        f"rollout size) for a genuine apples-to-apples comparison across algorithms.",
        "",
        "| algorithm | speedup vs torch.compile (full-call) |",
        "|:---|:---:|",
    ]
    by_key = {(r["algo"], r["num_envs"], r["seq_len"]): r for r in production_rows}
    for algo in algos:
        r = by_key.get((algo, fixed_num_envs, fixed_seq_len))
        if r is None:
            continue
        lines.append(f"| {algo} | {r['su_vec']:.1f}× |")
    lines.append("")
    lines.append("See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-update", action="store_true",
                        help="Print results but do not modify README.md or benchmarks.md")
    parser.add_argument("--skip-readme", action="store_true",
                        help="Update benchmarks.md but do not touch README.md "
                             "(README prose is a STOP-and-report item; use this for "
                             "unattended runs that should still archive data/tables).")
    parser.add_argument("--gpu", default="", metavar="LABEL",
                        help="GPU label to embed in the table header (default: auto-detect)")
    parser.add_argument("--version", default="", metavar="TAG",
                        help="Release version tag (e.g. v0.1.0) written into benchmarks.md")
    parser.add_argument("--algos", default="all", metavar="LIST",
                        help="Comma-separated subset of ALL_ALGOS to run (default: all). "
                             f"Choices: {','.join(ALL_ALGOS)}. Used for the GAE-only smoke "
                             "test before committing to the full unattended sweep.")
    args = parser.parse_args()

    selected_algos = ALL_ALGOS if args.algos == "all" else [a.strip() for a in args.algos.split(",")]
    unknown = set(selected_algos) - set(ALL_ALGOS)
    if unknown:
        print(f"Unknown --algos entries: {unknown}. Choices: {ALL_ALGOS}")
        sys.exit(1)

    print_environment_header("bench_release.py")

    if not torch.cuda.is_available() and not args.no_update:
        print("CUDA not available – aborting.")
        sys.exit(1)

    if args.no_update and not torch.cuda.is_available():
        # Dry-run without GPU: just render dummy tables to verify formatting.
        print("CUDA not available – rendering dummy output for format verification.")
        dummy = {"num_envs": 128, "seq_len": 1024, "triton_ms": 1.234, "triton_dev_ms": 0.987,
                 "compiled_ms": 2.345, "compiled_dev_ms": 1.987, "su_compile": 1.9, "su_compile_dev": 2.0,
                 "e2e_ms": 5.678, "numpy_ms": 8.901, "su_e2e": 1.6, "su_numpy": 7.2,
                 "vec_ms": 2.100, "vec_dev_ms": 1.765, "su_vec": 1.7, "su_vec_dev": 1.79,
                 "assoc_ms": 2.500, "su_assoc": 1.4,
                 "loop_ms": 45.678, "su_loop": 37.0}
        dummy_trunc = {"num_envs": 128, "seq_len": 1024, "triton_ms": 1.234, "triton_dev_ms": 0.987,
                       "vec_ms": 1.890, "vec_dev_ms": 1.654, "su_vec": 1.5, "su_vec_dev": 1.68}
        dummy_prod = {"algo": "GAE", "num_envs": 4096, "seq_len": 128, "is_boundary": False,
                      "triton_ms": 0.234, "triton_dev_ms": 0.187, "triton_amort_ms": 0.201,
                      "vec_ms": 0.410, "vec_dev_ms": 0.365, "su_vec": 1.75, "su_vec_dev": 1.95}
        tables = [
            _table_numpy("GAE (`compute_gae`)", [dummy], include_assoc=True),
            _table_truncation("GAE – with truncations (`compute_gae`)", [dummy_trunc]),
            _table_numpy("V-Trace (`compute_vtrace`)", [dummy], include_assoc=True),
            _table_truncation("V-Trace – with truncations (`compute_vtrace`)", [dummy_trunc]),
            _table_retrace("Retrace(λ) (`compute_retrace`)", [dummy]),
            _table_simple("λ-returns (`compute_lambda_returns`)", [dummy], include_assoc=True),
            _table_truncation("λ-returns – with truncations (`compute_lambda_returns`)", [dummy_trunc]),
            _table_simple("Discounted returns (`compute_discounted_returns`)", [dummy], include_assoc=True),
            _table_truncation("Discounted returns – with truncations (`compute_discounted_returns`)", [dummy_trunc]),
            _table_simple("Eligibility traces (`compute_eligibility_traces`)", [dummy]),
            _table_prefix_sum("Episodic prefix sum (`compute_episodic_prefix_sum`)", [dummy]),
            _table_production("Production regime (dummy)", [dummy_prod]),
        ]
        print(_section(args.gpu or "dry-run GPU", tables))
        return

    print("Compiling Triton kernels (first run may take several minutes on a cold cache) …",
          flush=True)
    _precompile_triton_kernels()
    print("All kernels compiled.\n", flush=True)

    all_violations = []
    production_rows_all = []
    full_tables, headline_tables = [], []

    if "gae" in selected_algos:
        print("Running GAE benchmark …", flush=True)
        gae_rows, gae_prod, v = bench_gae()
        all_violations += v
        production_rows_all += gae_prod
        print("Running GAE truncation-path benchmark …", flush=True)
        gae_trunc_rows, v = bench_gae_truncation()
        all_violations += v
        full_tables.append(_table_numpy("GAE (`compute_gae`)", gae_rows, include_assoc=True))
        headline_tables.append(_table_numpy("GAE (`compute_gae`)", _headline(gae_rows), include_assoc=True))
        full_tables.append(_table_truncation("GAE – with truncations (`compute_gae`)", gae_trunc_rows))
        headline_tables.append(_table_truncation("GAE – with truncations (`compute_gae`)", _headline(gae_trunc_rows)))

    if "vtrace" in selected_algos:
        print("Running V-Trace benchmark …", flush=True)
        vtrace_rows, vtrace_prod, v = bench_vtrace()
        all_violations += v
        production_rows_all += vtrace_prod
        print("Running V-Trace truncation-path benchmark …", flush=True)
        vtrace_trunc_rows, v = bench_vtrace_truncation()
        all_violations += v
        full_tables.append(_table_numpy("V-Trace (`compute_vtrace`)", vtrace_rows, include_assoc=True))
        headline_tables.append(_table_numpy("V-Trace (`compute_vtrace`)", _headline(vtrace_rows), include_assoc=True))
        full_tables.append(_table_truncation("V-Trace – with truncations (`compute_vtrace`)", vtrace_trunc_rows))
        headline_tables.append(_table_truncation("V-Trace – with truncations (`compute_vtrace`)", _headline(vtrace_trunc_rows)))

    if "retrace" in selected_algos:
        print("Running Retrace(λ) benchmark …", flush=True)
        retrace_rows, retrace_prod, v = bench_retrace()
        all_violations += v
        production_rows_all += retrace_prod
        full_tables.append(_table_retrace("Retrace(λ) (`compute_retrace`)", retrace_rows))
        headline_tables.append(_table_retrace("Retrace(λ) (`compute_retrace`)", _headline(retrace_rows)))

    if {"lambda_returns", "discounted_returns", "eligibility_traces"} & set(selected_algos):
        print("Running returns / eligibility-traces benchmark …", flush=True)
        lambda_rows, disc_rows, traces_rows, returns_prod, v = bench_returns()
        all_violations += v
        production_rows_all += returns_prod
        if "lambda_returns" in selected_algos:
            print("Running λ-returns truncation-path benchmark …", flush=True)
            lambda_trunc_rows, v = bench_lambda_returns_truncation()
            all_violations += v
            full_tables.append(_table_simple("λ-returns (`compute_lambda_returns`)", lambda_rows, include_assoc=True))
            headline_tables.append(_table_simple("λ-returns (`compute_lambda_returns`)", _headline(lambda_rows), include_assoc=True))
            full_tables.append(_table_truncation("λ-returns – with truncations (`compute_lambda_returns`)", lambda_trunc_rows))
            headline_tables.append(_table_truncation("λ-returns – with truncations (`compute_lambda_returns`)", _headline(lambda_trunc_rows)))
        if "discounted_returns" in selected_algos:
            print("Running discounted-returns truncation-path benchmark …", flush=True)
            disc_trunc_rows, v = bench_discounted_returns_truncation()
            all_violations += v
            full_tables.append(_table_simple("Discounted returns (`compute_discounted_returns`)", disc_rows, include_assoc=True))
            headline_tables.append(_table_simple("Discounted returns (`compute_discounted_returns`)", _headline(disc_rows), include_assoc=True))
            full_tables.append(_table_truncation("Discounted returns – with truncations (`compute_discounted_returns`)", disc_trunc_rows))
            headline_tables.append(_table_truncation("Discounted returns – with truncations (`compute_discounted_returns`)", _headline(disc_trunc_rows)))
        if "eligibility_traces" in selected_algos:
            full_tables.append(_table_simple("Eligibility traces (`compute_eligibility_traces`)", traces_rows))
            headline_tables.append(_table_simple("Eligibility traces (`compute_eligibility_traces`)", _headline(traces_rows)))

    if "prefix_sum" in selected_algos:
        print("Running episodic prefix sum benchmark …", flush=True)
        prefix_sum_rows, prefix_prod, v = bench_prefix_sum()
        all_violations += v
        production_rows_all += prefix_prod
        full_tables.append(_table_prefix_sum("Episodic prefix sum (`compute_episodic_prefix_sum`)", prefix_sum_rows))
        headline_tables.append(_table_prefix_sum("Episodic prefix sum (`compute_episodic_prefix_sum`)", _headline(prefix_sum_rows)))

    production_table = _table_production(
        "Production regime — seq_len [80,128] × num_envs [4096..38400], all algorithms "
        "(plus one boundary-marker row, num_envs=16384/seq_len=16)",
        production_rows_all,
    )
    full_tables.append(production_table)
    headline_tables.append(production_table)

    section = _section(args.gpu, headline_tables)

    print("\n" + section)

    print(f"\n{'=' * 88}\nGATE SUMMARY\n{'=' * 88}")
    if all_violations:
        print(f"MONOTONICITY GATE: FAILED — {len(all_violations)} violation(s) across the run:")
        for v in all_violations:
            print(f"  - {v}")
    else:
        print("MONOTONICITY GATE: PASSED across the entire run.")
    print("CORRECTNESS GATE (atol=rtol=1e-4 vs sequential reference): PASSED for every config "
          "reached (assert_correctness raises immediately on failure, so reaching this line "
          "means every gate up to here held).")

    if not args.no_update:
        version = args.version or "unreleased"
        update_benchmarks_md(version, args.gpu, full_tables)

        readme_draft = render_readme_table_draft(args.gpu, production_rows_all)
        draft_path = REPO_ROOT / "benchmarks" / "readme_table_draft.md"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(readme_draft)
        print(f"\nREADME summary table drafted to {draft_path}.")

        if args.skip_readme:
            print("--skip-readme passed: README.md left untouched — review the draft above "
                  "and paste manually (README prose edits are a STOP-and-report item).")
        else:
            update_readme(section)

    if all_violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
