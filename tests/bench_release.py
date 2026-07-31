"""
Release benchmark – full sweep across all algorithms and configurations.

Compares rl-triton Triton kernels against torch.compile on both a naive
per-timestep Python loop and the fastest hand-vectorized PyTorch equivalent,
against pure NumPy CPU loops, and against a NumPy→GPU→NumPy adoption path.
Algorithms that support truncated episodes (GAE, V-Trace, Retrace, λ-returns,
discounted returns) also report a third baseline – the same associative-scan
implementation used for the truncation path, called with zero truncations –
plus an additional table benchmarking the truncation path itself against
that vectorized baseline.

By default, stages the result in docs/benchmark-history/unreleased.md and drafts
benchmarks/readme_table_draft.md for manual review -- it never writes benchmarks.md
or README.md directly. Use --promote --version <tag> in a separate invocation to
move a staged candidate into benchmarks.md as an actual release; README.md itself
is always a manual, human paste from the draft (see benchmarks/README.md's
placement policy).

Usage:
    python tests/bench_release.py                # run and stage the candidate
    python tests/bench_release.py --no-update    # print only, stage nothing
    python tests/bench_release.py --gpu RTX4090  # label the GPU in the table header
    python tests/bench_release.py --promote --version v0.1.1  # promote the staged candidate
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Silence torch.compile/dynamo symbolic-shapes warnings (e.g. "q1 is not in
# var_ranges, defaulting to unknown range") that otherwise spam stdout on
# every torch.compile call in this sweep. .runpod/start.sh sets this at the
# shell level for the CI runner pod; set it here too (setdefault, so an
# explicit caller override still wins) so a bare `python tests/bench_release.py`
# run outside that pod is quiet as well. Must be set before `import torch`.
os.environ.setdefault("TORCH_LOGS", "-dynamic")

import numpy as np
import torch
import torch._dynamo
import triton

# This sweep compiles ~15 distinct torch.compile(...) wrapper objects, several
# reused across every shape in CONFIGS (12) plus PRODUCTION_CONFIGS+BOUNDARY_
# CONFIG (11) -- ~200 (function, shape) compiles total across the process.
# Raising Dynamo's cache limits well above that avoids one confirmed failure
# mode (recompiles past the default cache_size_limit=8 silently falling back
# instead of compiling) even though it turned out NOT to be the cause of the
# cross-shape wrong-output bug (that one is a genuine Inductor buffer-reuse
# bug around parallel_suffix_scan's internal padding branch -- confirmed
# still present with these limits raised; see _needs_pad()'s comment and the
# targeted torch._dynamo.reset() at the actual transition points). Keeping
# both limits high is still worthwhile: it avoids silent eager fallback
# (slow, not wrong, but undermines the whole point of the sweep) and costs
# nothing.
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 512

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
from test_retrace import reference_retrace, vectorized_retrace
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
    truncations -- same recurrence as test_gae.py's O(N*T) _ref_gae_sequential,
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


# _vec_retrace (a local log-space-suffix-cumsum duplicate of test_retrace.py's
# vectorized_retrace) used to live here. It was never migrated to the
# parallel_suffix_scan fix applied to every other algorithm's baseline --
# 97-99% non-finite output at every CONFIGS/production shape, silently timed
# because bench_retrace() was the only bench_* function with no
# assert_correctness check on its vec baseline. Removed; bench_retrace() now
# imports vectorized_retrace directly, the same already-fixed implementation
# tests/bench_safeguard.py uses.


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
    # Main grid addition -- massively-parallel-sim regime (high-N/short-T):
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
# prior work found device-only time can invert below 1x at very short T /
# very high num_envs (see NOTES.md's PufferLib design-tradeoff note for the
# grid-wave mechanism).
BOUNDARY_CONFIG = (16384, 16)

ALL_ALGOS = [
    "gae", "vtrace", "retrace", "lambda_returns",
    "discounted_returns", "eligibility_traces", "prefix_sum",
]


def _needs_pad(seq_len):
    """Whether parallel_suffix_scan/parallel_prefix_scan will pad seq_len up
    to the next power of 2 (its doubling scan requires T_pad a power of 2).

    Confirmed root cause of the cross-shape torch.compile correctness bug
    (see the comment at the first use below): the padding branch allocates a
    zero tensor INSIDE the compiled region (torch.cat([x, torch.zeros(...)]))
    -- the same class of Inductor buffer-reuse/aliasing bug as the
    always-zero-argument bug fixed elsewhere in this file, just triggered
    internally rather than by a caller-supplied zero tensor. Only PRODUCTION_
    SEQ_LENS' seq_len=80 hits this branch anywhere in this sweep; every
    CONFIGS/HEADLINE_CONFIGS/BOUNDARY_CONFIG seq_len is already a power of 2.
    """
    t_pad = 1
    while t_pad < seq_len:
        t_pad *= 2
    return t_pad != seq_len


def _bench_production_regime(algo_label, triton_fn, compiled_fn, ref_fn, make_inputs_fn, kwargs):
    """Shared production-regime + boundary-marker sweep for algorithms whose
    triton/compiled/reference callables all accept the SAME positional args
    (everything except Retrace, whose kernel needs a reordered arg tuple --
    see its own inline block in bench_retrace).

    Runs PRODUCTION_CONFIGS (seq_len [80,128] x num_envs [4096..38400]) plus
    the single BOUNDARY_CONFIG marker row, reporting full-call ms (headline),
    device-only ms (diagnostic), and the amortized (N-calls-per-region)
    variant -- the latter specifically to separate harness per-call sync
    overhead from genuine per-call cost at these short seq_lens. Asserts
    tolerance-based correctness (atol=1e-4) at every config before trusting
    the timing.
    """
    rows = []
    # compiled_fn was already used across CONFIGS beforehand in every current
    # caller, and every CONFIGS seq_len is a power of 2 (no padding) -- so the
    # object arrives here having never taken the padding branch.
    prev_needs_pad = False
    for num_envs, seq_len in PRODUCTION_CONFIGS + [BOUNDARY_CONFIG]:
        is_boundary = (num_envs, seq_len) == BOUNDARY_CONFIG
        args_gpu = make_inputs_fn(num_envs, seq_len)
        ni = _n_iter_gpu(seq_len, num_envs)

        # Reusing one torch.compile(...)-wrapped object across a transition
        # into/out of parallel_suffix_scan's internal padding branch (only
        # seq_len=80 here, needing pad to 128) silently produces WRONG numeric
        # output (not NaN) -- confirmed via bisection to be independent of
        # cache_size_limit/accumulated_cache_size_limit (raising both to 128/
        # 512 does not fix it) and independent of torch._dynamo.mark_dynamic
        # (does not fix it either); a genuinely fresh torch.compile(...)
        # object in the same process is STILL wrong, so it's process-level
        # Dynamo/Inductor state, not object identity. torch._dynamo.reset()
        # right at the transition does fix it, and padding depends only on
        # seq_len (not num_envs), so only the ~2 seq_len-parity transitions
        # per compiled object need it -- resetting before every shape (the
        # previous approach) also "fixed" this but accumulates enough resets
        # across a full sweep to itself corrupt CUDA state (illegal memory
        # access late in the run); resetting only at these rare transitions
        # stays far below that threshold.
        needs_pad = _needs_pad(seq_len)
        if needs_pad != prev_needs_pad:
            torch._dynamo.reset()
        prev_needs_pad = needs_pad

        # Reusing one torch.compile(...)-wrapped object across many distinct
        # shapes (e.g. seq_len=80 needs parallel_suffix_scan to pad to 128,
        # seq_len=128 doesn't) used to silently produce WRONG numeric output
        # (not NaN) at some shapes once >cache_size_limit (default 8) distinct
        # shapes had compiled against it in this process -- Dynamo evicts/
        # reuses a stale guard instead of recompiling. torch._dynamo.reset()
        # per shape "fixed" this but defeats the compile cache and, called
        # enough times in one process, corrupts CUDA state (illegal memory
        # access crashes late in the full sweep). Real fix: raise
        # torch._dynamo.config.cache_size_limit / accumulated_cache_size_limit
        # at startup (see module top) so every distinct shape in the sweep
        # gets its own cache entry with no eviction and no reset needed.
        triton_out = triton_fn(*args_gpu, **kwargs)
        ref_out    = ref_fn(*args_gpu, **kwargs)
        assert_correctness(triton_out, ref_out, f"{algo_label}[production,{num_envs}x{seq_len}]")

        vec_out = compiled_fn(*args_gpu, **kwargs)
        assert_correctness(vec_out, ref_out, f"{algo_label}[production,{num_envs}x{seq_len}] (vec baseline)")

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
    # compile(vec) and compile(assoc) used to be two different algorithms (a
    # log-space cumsum vs. the associative doubling scan) -- now that
    # vectorized_gae is a thin wrapper around vectorized_gae_with_truncations
    # (see test_gae.py), they're the same computation, so the "compile(assoc)"
    # column was dropped: it no longer demonstrates anything vec doesn't.
    #
    # We compile vectorized_gae_with_truncations DIRECTLY here rather than the
    # vectorized_gae wrapper -- torch.compile(vectorized_gae) was found to
    # silently produce WRONG numeric output (up to 17% of elements, not
    # NaN/inf) at some shapes. Bisected to: allocating torch.zeros_like(...)
    # INSIDE a torch.compiled function, then feeding it into
    # parallel_suffix_scan, corrupts results (likely an Inductor buffer-
    # reuse/aliasing bug around the always-zero tensor); allocating the same
    # zeros in eager code and passing them in as plain arguments is correct.
    # So truncateds/bootstrap below are always constructed in eager code.
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec  = torch.compile(vectorized_gae_with_truncations)
    args_warmup = _make_gae(64, 512)
    trunc_w = torch.zeros(64, 512, device="cuda")
    bsv_w   = torch.zeros(64, 512, device="cuda")
    compiled_vec(*args_warmup, trunc_w, bsv_w, 0.99, 0.95); torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'vs vec':>8} {'vs vec(dev)':>12} "
        f"{'loop(gpu)':>11} {'vs loop':>9} {'numpy(cpu)':>12} {'vs numpy':>10} "
        f"{'np->tri->np':>13} {'e2e vs np':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        args_gpu = _make_gae(num_envs, seq_len)
        args_np  = tuple(t.cpu().numpy() for t in args_gpu)
        ni       = _n_iter_gpu(seq_len, num_envs)
        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")

        triton_out = compute_gae(*args_gpu, gamma=0.99, lambda_=0.95)
        ref_out    = _ref_gae(*args_gpu, gamma=0.99, lambda_=0.95)
        assert_correctness(triton_out, ref_out, f"gae[{num_envs}x{seq_len}]")

        vec_out = compiled_vec(*args_gpu, trunc0, bsv0, 0.99, 0.95)
        assert_correctness(vec_out, ref_out, f"gae[{num_envs}x{seq_len}] (vec baseline)")

        _warmup_gpu(compute_gae, *args_gpu, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_vec, *args_gpu, trunc0, bsv0, 0.99, 0.95)

        triton_ms = _bench_gpu(compute_gae,    *args_gpu, gamma=0.99, lambda_=0.95, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,   *args_gpu, trunc0, bsv0, 0.99, 0.95, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_gae, *args_gpu, gamma=0.99, lambda_=0.95)
        vec_dev_ms, _    = _device_profile(compiled_vec, *args_gpu, trunc0, bsv0, 0.99, 0.95)
        loop_ms   = _bench_cpu(_ref_gae,               *args_gpu, gamma=0.99, lambda_=0.95)
        numpy_ms  = _bench_cpu(numpy_gae_cpu,          *args_gpu, gamma=0.99, lambda_=0.95)
        e2e_ms    = _bench_cpu(numpy_gae_np_to_triton, *args_np,  gamma=0.99, lambda_=0.95)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "loop_ms": loop_ms,
            "numpy_ms": numpy_ms, "e2e_ms": e2e_ms,
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
        print(f"  MONOTONICITY GATE FAILED (gae, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (gae, triton_ms, 2% band)", flush=True)

    # Eager (uncompiled) shim matching compute_gae's signature -- constructs
    # truncateds/bootstrap in eager code per call (never inside the compiled
    # region) before delegating to the already-compiled compiled_vec, per the
    # torch.compile buffer-corruption note above.
    def _compiled_vec_prod(rewards, values, terminateds, gamma, lambda_):
        trunc = torch.zeros_like(terminateds)
        bsv   = torch.zeros_like(rewards)
        return compiled_vec(rewards, values, terminateds, trunc, bsv, gamma, lambda_)

    production_rows = _bench_production_regime(
        "GAE", compute_gae, _compiled_vec_prod, _ref_gae, _make_gae,
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
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, values, terminateds = _make_gae(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_gae_trunc(rewards, values, terminateds, truncateds, bootstrap_values,
                                  gamma=0.99, lambda_=0.95)
        triton_out = compute_gae(rewards, values, terminateds, truncateds,
                                  gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        # Check compiled_vec_trunc itself here, not the eager
        # vectorized_gae_with_truncations it wraps: compiled_vec_trunc is the
        # object actually timed below, and it's the one exposed to the
        # cross-shape torch.compile corruption documented in NOTES.md -- the
        # eager function is never at risk of that bug, so checking it alone
        # would silently pass even if this shape's compile came back wrong.
        vec_out = compiled_vec_trunc(rewards, values, terminateds, truncateds,
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
    # compile(assoc) column dropped -- see bench_gae()'s comment; vectorized_vtrace
    # is now a thin wrapper around vectorized_vtrace_with_truncations, so the two
    # baselines are the same computation. Compiling vectorized_vtrace_with_truncations
    # directly (not the wrapper) -- see bench_gae()'s torch.compile buffer-
    # corruption note; truncateds/bootstrap always built in eager code.
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec  = torch.compile(vectorized_vtrace_with_truncations)
    args = _make_vtrace(64, 512)
    trunc_w = torch.zeros(64, 512, device="cuda")
    bsv_w   = torch.zeros(64, 512, device="cuda")
    compiled_vec(*args, trunc_w, bsv_w, gamma=0.99); torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'vs vec':>8} {'vs vec(dev)':>12} "
        f"{'loop(gpu)':>11} {'vs loop':>9} {'numpy(cpu)':>12} {'vs numpy':>10} "
        f"{'np->tri->np':>13} {'e2e vs np':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for num_envs, seq_len in CONFIGS:
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        args_gpu = _make_vtrace(num_envs, seq_len)
        args_np  = tuple(t.cpu().numpy() for t in args_gpu)
        ni       = _n_iter_gpu(seq_len, num_envs)
        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")

        triton_out = compute_vtrace_fused(*args_gpu, gamma=0.99)
        ref_out    = _ref_vtrace(*args_gpu, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"vtrace[{num_envs}x{seq_len}]")

        vec_out = compiled_vec(*args_gpu, trunc0, bsv0, gamma=0.99)
        assert_correctness(vec_out, ref_out, f"vtrace[{num_envs}x{seq_len}] (vec baseline)")

        _warmup_gpu(compute_vtrace_fused, *args_gpu, gamma=0.99)
        _warmup_gpu(compiled_vec,         *args_gpu, trunc0, bsv0, gamma=0.99)
        triton_ms = _bench_gpu(compute_vtrace_fused, *args_gpu, gamma=0.99, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,         *args_gpu, trunc0, bsv0, gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_vtrace_fused, *args_gpu, gamma=0.99)
        vec_dev_ms, _    = _device_profile(compiled_vec,         *args_gpu, trunc0, bsv0, gamma=0.99)
        loop_ms   = _bench_cpu(_ref_vtrace,               *args_gpu, gamma=0.99)
        numpy_ms  = _bench_cpu(numpy_vtrace_cpu,          *args_gpu, gamma=0.99)
        e2e_ms    = _bench_cpu(numpy_vtrace_np_to_triton, *args_np,  gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "triton_dev_ms": triton_dev_ms,
            "vec_ms": vec_ms, "vec_dev_ms": vec_dev_ms,
            "loop_ms": loop_ms,
            "numpy_ms": numpy_ms, "e2e_ms": e2e_ms,
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
        print("  MONOTONICITY GATE FAILED (vtrace, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (vtrace, triton_ms, 2% band)", flush=True)

    def _compiled_vec_prod(log_pi_target, log_pi_behavior, values, rewards, terminateds, gamma):
        trunc = torch.zeros_like(terminateds)
        bsv   = torch.zeros_like(rewards)
        return compiled_vec(log_pi_target, log_pi_behavior, values, rewards, terminateds,
                             trunc, bsv, gamma=gamma)

    production_rows = _bench_production_regime(
        "V-Trace", compute_vtrace_fused, _compiled_vec_prod, _ref_vtrace, _make_vtrace,
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
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        log_pi_t, log_pi_b, values, rewards, terminateds = _make_vtrace(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_vtrace_sequential(log_pi_t, log_pi_b, values, rewards, terminateds,
                                          truncateds, bootstrap_values, gamma=0.99)
        triton_out = compute_vtrace_fused(log_pi_t, log_pi_b, values, rewards, terminateds,
                                           truncateds=truncateds, gamma=0.99,
                                           bootstrap_values=bootstrap_values)
        # Check compiled_vec_trunc itself, not the eager function it wraps --
        # see bench_gae_truncation's matching comment.
        vec_out = compiled_vec_trunc(log_pi_t, log_pi_b, values, rewards,
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
    compiled_vec = torch.compile(vectorized_retrace)
    warmup_args = _retrace_kernel_args(_make_retrace(64, 512))
    compiled_vec(*warmup_args, gamma=0.99); torch.cuda.synchronize()
    compute_retrace(*warmup_args, gamma=0.99); torch.cuda.synchronize()
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
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        args_gpu = _make_retrace(num_envs, seq_len)
        rn, dn, qn, nqn, _qn2, an, aptn, apbn = tuple(t.cpu().numpy() for t in args_gpu)
        ni = _n_iter_gpu(seq_len, num_envs)

        retrace_kernel_args = _retrace_kernel_args(args_gpu)
        triton_out, _ = compute_retrace(*retrace_kernel_args, gamma=0.99)
        ref_out       = _ref_retrace(*args_gpu, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"retrace[{num_envs}x{seq_len}]")

        # Check compiled_vec itself, not the eager vectorized_retrace it wraps
        # -- see bench_gae_truncation's matching comment.
        vec_out, _ = compiled_vec(*retrace_kernel_args, gamma=0.99)
        assert_correctness(vec_out, ref_out, f"retrace[{num_envs}x{seq_len}] (vec baseline)")

        _warmup_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99)
        _warmup_gpu(compiled_vec,    *retrace_kernel_args, gamma=0.99)
        triton_ms = _bench_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,    *retrace_kernel_args, gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_retrace, *retrace_kernel_args, gamma=0.99)
        vec_dev_ms, _    = _device_profile(compiled_vec,    *retrace_kernel_args, gamma=0.99)
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
    # _bench_production_regime helper -- inline production sweep instead.
    production_rows = []
    # See _bench_production_regime's matching comment: reset only at a
    # padding-regime transition (compiled_vec already only saw non-padded
    # CONFIGS shapes above), not on every shape.
    prev_needs_pad = False
    for num_envs, seq_len in PRODUCTION_CONFIGS + [BOUNDARY_CONFIG]:
        is_boundary = (num_envs, seq_len) == BOUNDARY_CONFIG
        needs_pad = _needs_pad(seq_len)
        if needs_pad != prev_needs_pad:
            torch._dynamo.reset()
        prev_needs_pad = needs_pad
        args_gpu = _make_retrace(num_envs, seq_len)
        retrace_kernel_args = _retrace_kernel_args(args_gpu)
        ni = _n_iter_gpu(seq_len, num_envs)

        triton_out, _ = compute_retrace(*retrace_kernel_args, gamma=0.99)
        ref_out       = _ref_retrace(*args_gpu, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"retrace[production,{num_envs}x{seq_len}]")

        # Check compiled_vec itself, not the eager vectorized_retrace it wraps
        # -- see bench_gae_truncation's matching comment.
        vec_out, _ = compiled_vec(*retrace_kernel_args, gamma=0.99)
        assert_correctness(vec_out, ref_out, f"retrace[production,{num_envs}x{seq_len}] (vec baseline)")

        _warmup_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99)
        _warmup_gpu(compiled_vec,    *retrace_kernel_args, gamma=0.99)
        triton_ms = _bench_gpu(compute_retrace, *retrace_kernel_args, gamma=0.99, n_iter=ni)
        vec_ms    = _bench_gpu(compiled_vec,    *retrace_kernel_args, gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_retrace, *retrace_kernel_args, gamma=0.99)
        vec_dev_ms, _    = _device_profile(compiled_vec,    *retrace_kernel_args, gamma=0.99)
        triton_amort_ms  = _bench_gpu_amortized(compute_retrace, *retrace_kernel_args, gamma=0.99)
        vec_amort_ms     = _bench_gpu_amortized(compiled_vec,    *retrace_kernel_args, gamma=0.99)

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


def bench_retrace_truncation():
    """Retrace's truncation path, benchmarked against its own with-truncations
    vectorized baseline -- the counterpart to bench_gae_truncation() /
    bench_vtrace_truncation() / bench_lambda_returns_truncation() /
    bench_discounted_returns_truncation() for Retrace.

    Unlike those four, Retrace has no bootstrap_values concept -- the
    continuation value is already folded into next_q_values_all every step
    (docs/kernels/retrace.md §4), so only a non-zero truncateds is generated
    here (mutually exclusive with terminateds, same ~5% rate _make_trunc_
    extras uses for the other four), and reference_retrace (test_retrace.py)
    is used as ground truth rather than bench_release.py's own _ref_retrace --
    that one takes a single combined `dones` flag and cannot distinguish
    terminated (zero the Q-bootstrap) from truncated (keep the bootstrap,
    sever the trace); it predates the truncation feature and is only valid
    for bench_retrace()'s existing always-zero-truncateds usage.

    bench_retrace() elsewhere in this file always passes truncateds=torch.
    zeros_like(dones) via _retrace_kernel_args, so this is the first place
    Retrace's truncation path is exercised across the full CONFIGS grid.
    """
    print("  compiling torch.compile baselines …", end="", flush=True)
    compiled_vec_trunc = torch.compile(vectorized_retrace)
    (rewards_w, terminateds_w, _values_w, next_q_all_w,
     q_values_w, actions_w, apt_w, apb_w) = _make_retrace(64, 512)
    truncateds_w = (torch.rand(64, 512, device=rewards_w.device) < 0.05).float() * (1.0 - terminateds_w)
    compiled_vec_trunc(apt_w, apb_w, q_values_w, next_q_all_w, actions_w, rewards_w,
                        terminateds_w, truncateds_w, gamma=0.99)
    compute_retrace(apt_w, apb_w, q_values_w, next_q_all_w, actions_w, rewards_w,
                     terminateds_w, truncateds_w, gamma=0.99)
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
        # Reset before every shape -- see bench_gae_truncation's matching
        # comment: reusing one torch.compile(...) object across distinct
        # shapes without this can silently give wrong output at some
        # transition.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        (rewards, terminateds, _values, next_q_all,
         q_values, actions, apt, apb) = _make_retrace(num_envs, seq_len)
        truncateds = (torch.rand(num_envs, seq_len, device=rewards.device) < 0.05).float() * (1.0 - terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out, _ = reference_retrace(apt, apb, q_values, next_q_all, actions, rewards,
                                        terminateds, truncateds, gamma=0.99)
        triton_out, _ = compute_retrace(apt, apb, q_values, next_q_all, actions, rewards,
                                         terminateds, truncateds, gamma=0.99)
        assert_correctness(triton_out, ref_out, f"retrace_trunc[{num_envs}x{seq_len}] (triton vs ref)")

        # Check compiled_vec_trunc itself here, not the eager vectorized_retrace
        # it wraps -- see bench_gae_truncation's matching comment.
        vec_out, _ = compiled_vec_trunc(apt, apb, q_values, next_q_all, actions, rewards,
                                         terminateds, truncateds, gamma=0.99)
        assert_correctness(vec_out, ref_out, f"retrace_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

        _warmup_gpu(compute_retrace, apt, apb, q_values, next_q_all, actions, rewards,
                    terminateds, truncateds, gamma=0.99)
        _warmup_gpu(compiled_vec_trunc, apt, apb, q_values, next_q_all, actions, rewards,
                    terminateds, truncateds, gamma=0.99)

        triton_ms = _bench_gpu(compute_retrace, apt, apb, q_values, next_q_all, actions, rewards,
                                terminateds, truncateds, gamma=0.99, n_iter=ni)
        vec_ms = _bench_gpu(compiled_vec_trunc, apt, apb, q_values, next_q_all, actions, rewards,
                             terminateds, truncateds, gamma=0.99, n_iter=ni)
        triton_dev_ms, _ = _device_profile(compute_retrace, apt, apb, q_values, next_q_all, actions,
                                            rewards, terminateds, truncateds, gamma=0.99)
        vec_dev_ms, _ = _device_profile(compiled_vec_trunc, apt, apb, q_values, next_q_all, actions,
                                         rewards, terminateds, truncateds, gamma=0.99)

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
        print("  MONOTONICITY GATE FAILED (retrace_trunc, triton_ms, 2% band):", flush=True)
        for v in violations:
            print(f"    - {v}", flush=True)
    else:
        print("  monotonicity gate: PASSED (retrace_trunc, triton_ms, 2% band)", flush=True)
    return rows, violations


def bench_returns(selected=frozenset({"lambda_returns", "discounted_returns", "eligibility_traces"})):
    # compile(assoc) columns dropped for lambda-returns/discounted-returns --
    # see bench_gae()'s comment; both vec baselines are now thin wrappers
    # around their _with_truncations siblings, same computation either way.
    # c_lambda_vec/c_disc_vec compile the *_with_truncations siblings directly
    # (not the thin wrappers) -- see bench_gae()'s torch.compile buffer-
    # corruption note; truncateds/bootstrap always built in eager code.
    # c_traces_vec is safe to compile directly: vectorized_eligibility_traces
    # never allocates an unconditional all-zero buffer fed into the scan the
    # way the other four did (its decay tensor is always genuinely computed),
    # confirmed bug-free under torch.compile before shipping this.
    #
    # `selected` gates which of the three objects actually get built/compiled/
    # exercised below -- NOT just which tables the caller keeps afterward.
    # This used to unconditionally compile and interleave all three regardless
    # of which algorithms were actually requested; that interleaved-multi-
    # object-per-shape pattern turned out to be exactly what triggers the
    # cross-compile wrong-output bug (see NOTES.md), and it reproduced even
    # inside a solo `--algos lambda_returns` subprocess because this function
    # still built and interleaved all three objects regardless. Skipping the
    # non-selected objects entirely (not compiling them at all) is the actual
    # fix -- a solo run now only ever has ONE compiled object in play, same
    # shape as gae/vtrace/retrace's already-reliable solo subprocesses.
    want_lambda = "lambda_returns" in selected
    want_disc   = "discounted_returns" in selected
    want_traces = "eligibility_traces" in selected

    print("  compiling torch.compile baselines …", end="", flush=True)
    c_lambda_vec = torch.compile(vectorized_lambda_returns_with_truncations) if want_lambda else None
    c_disc_vec   = torch.compile(vectorized_discounted_returns_with_truncations) if want_disc else None
    c_traces_vec = torch.compile(vectorized_eligibility_traces) if want_traces else None
    r, nv, d = _make_returns(64, 512)
    trunc_w = torch.zeros(64, 512, device="cuda")
    bsv_w   = torch.zeros(64, 512, device="cuda")
    if want_lambda:
        c_lambda_vec(r, nv, d, trunc_w, bsv_w, gamma=0.99, lambda_=0.95); torch.cuda.synchronize()
    if want_disc:
        c_disc_vec(r, d, trunc_w, bsv_w, gamma=0.99);                     torch.cuda.synchronize()
    if want_traces:
        c_traces_vec(r, d, gamma=0.99, lambda_=0.95);                     torch.cuda.synchronize()
    print(" done.", flush=True)

    header = (
        f"\n{'algo':>20} {'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'dev':>8} {'compile(vec)':>14} {'loop(gpu)':>11} "
        f"{'numpy(cpu)':>12} {'vs vec':>8} {'vs vec(dev)':>12} {'vs loop':>9} {'vs numpy':>10}"
    )
    print(header)
    print("-" * len(header))

    def _print_sub_row(name, num_envs, seq_len, triton_ms, triton_dev_ms, vec_ms, vec_dev_ms,
                        loop_ms, numpy_ms):
        print(
            f"{name:>20} {num_envs:>10} {seq_len:>8} "
            f"{f'{triton_ms:.3f}ms':>8} {f'{triton_dev_ms:.3f}ms':>8} "
            f"{f'{vec_ms:.3f}ms':>14} "
            f"{f'{loop_ms:.3f}ms':>11} {f'{numpy_ms:.3f}ms':>12} "
            f"{f'{vec_ms/triton_ms:.2f}x':>8} {f'{vec_dev_ms/triton_dev_ms:.2f}x':>12} "
            f"{f'{loop_ms/triton_ms:.1f}x':>9} {f'{numpy_ms/triton_ms:.1f}x':>10}",
            flush=True,
        )

    rows_lambda, rows_disc, rows_traces = [], [], []
    for num_envs, seq_len in CONFIGS:
        # Reset before every shape in this loop (not just padding transitions
        # like _bench_production_regime's _needs_pad guard): direct isolation
        # testing found vectorized_discounted_returns_with_truncations gives
        # silently wrong output (~22.5% of elements) the first time its
        # compiled object recompiles for a NEW shape after already being
        # compiled+warmed at a different one -- reproduces even for two
        # power-of-2 shapes (64x512 -> 128x1024, neither needs padding), so
        # this is a distinct bug from the padding-transition one, and a fresh
        # object's first-ever compile at either shape alone is always
        # correct. Total resets here is bounded by len(CONFIGS) (12) per
        # subprocess -- far under the threshold that was found to corrupt
        # CUDA state when resets accumulate into the dozens across a whole
        # unbounded sweep.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, next_values, dones = _make_returns(num_envs, seq_len)
        ni = _n_iter_gpu(seq_len, num_envs)
        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")

        if want_lambda:
            lam_out = compute_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=0.95)
            lam_ref = _ref_lambda(rewards, next_values, dones, gamma=0.99, lambda_=0.95)
            assert_correctness(lam_out, lam_ref, f"lambda_returns[{num_envs}x{seq_len}]")
            lam_vec_out = c_lambda_vec(rewards, next_values, dones, trunc0, bsv0, gamma=0.99, lambda_=0.95)
            assert_correctness(lam_vec_out, lam_ref, f"lambda_returns[{num_envs}x{seq_len}] (vec baseline)")
        if want_disc:
            disc_out = compute_discounted_returns(rewards, dones, gamma=0.99)
            disc_ref = _ref_disc(rewards, dones, gamma=0.99)
            assert_correctness(disc_out, disc_ref, f"discounted_returns[{num_envs}x{seq_len}]")
            disc_vec_out = c_disc_vec(rewards, dones, trunc0, bsv0, gamma=0.99)
            assert_correctness(disc_vec_out, disc_ref, f"discounted_returns[{num_envs}x{seq_len}] (vec baseline)")
        if want_traces:
            trc_out = compute_eligibility_traces(rewards, dones, gamma=0.99, lambda_=0.9)
            trc_ref = _ref_traces(rewards, dones, gamma=0.99, lambda_=0.9)
            assert_correctness(trc_out, trc_ref, f"eligibility_traces[{num_envs}x{seq_len}]")
            trc_vec_out = c_traces_vec(rewards, dones, gamma=0.99, lambda_=0.9)
            assert_correctness(trc_vec_out, trc_ref, f"eligibility_traces[{num_envs}x{seq_len}] (vec baseline)")

        if want_lambda:
            _warmup_gpu(compute_lambda_returns, rewards, next_values, dones, gamma=0.99, lambda_=0.95)
            _warmup_gpu(c_lambda_vec, rewards, next_values, dones, trunc0, bsv0, gamma=0.99, lambda_=0.95)
        if want_disc:
            _warmup_gpu(compute_discounted_returns, rewards, dones, gamma=0.99)
            _warmup_gpu(c_disc_vec, rewards, dones, trunc0, bsv0, gamma=0.99)
        if want_traces:
            _warmup_gpu(compute_eligibility_traces, rewards, dones, gamma=0.99, lambda_=0.9)
            _warmup_gpu(c_traces_vec, rewards, dones, gamma=0.99, lambda_=0.9)

        if want_lambda:
            lam_ms = _bench_gpu(compute_lambda_returns, rewards, next_values, dones,
                                 gamma=0.99, lambda_=0.95, n_iter=ni)
            lam_vec_ms = _bench_gpu(c_lambda_vec, rewards, next_values, dones, trunc0, bsv0,
                                     gamma=0.99, lambda_=0.95, n_iter=ni)
            lam_dev_ms, _ = _device_profile(compute_lambda_returns, rewards, next_values, dones,
                                             gamma=0.99, lambda_=0.95)
            lam_vec_dev_ms, _ = _device_profile(c_lambda_vec, rewards, next_values, dones, trunc0, bsv0,
                                                 gamma=0.99, lambda_=0.95)
            lam_loop_ms = _bench_cpu(_ref_lambda, rewards, next_values, dones, gamma=0.99, lambda_=0.95)
            lam_np_ms = _bench_cpu(numpy_lambda_returns_cpu, rewards, next_values, dones,
                                    gamma=0.99, lambda_=0.95)
        if want_disc:
            disc_ms = _bench_gpu(compute_discounted_returns, rewards, dones, gamma=0.99, n_iter=ni)
            disc_vec_ms = _bench_gpu(c_disc_vec, rewards, dones, trunc0, bsv0, gamma=0.99, n_iter=ni)
            disc_dev_ms, _ = _device_profile(compute_discounted_returns, rewards, dones, gamma=0.99)
            disc_vec_dev_ms, _ = _device_profile(c_disc_vec, rewards, dones, trunc0, bsv0, gamma=0.99)
            disc_loop_ms = _bench_cpu(_ref_disc, rewards, dones, gamma=0.99)
            disc_np_ms = _bench_cpu(numpy_discounted_returns_cpu, rewards, dones, gamma=0.99)
        if want_traces:
            trc_ms = _bench_gpu(compute_eligibility_traces, rewards, dones,
                                 gamma=0.99, lambda_=0.9, n_iter=ni)
            trc_vec_ms = _bench_gpu(c_traces_vec, rewards, dones, gamma=0.99, lambda_=0.9, n_iter=ni)
            trc_dev_ms, _ = _device_profile(compute_eligibility_traces, rewards, dones,
                                             gamma=0.99, lambda_=0.9)
            trc_vec_dev_ms, _ = _device_profile(c_traces_vec, rewards, dones, gamma=0.99, lambda_=0.9)
            trc_loop_ms = _bench_cpu(_ref_traces, rewards, dones, gamma=0.99, lambda_=0.9)
            trc_np_ms = _bench_cpu(numpy_eligibility_traces_cpu, rewards, dones, gamma=0.99, lambda_=0.9)

        base = {"num_envs": num_envs, "seq_len": seq_len}
        if want_lambda:
            rows_lambda.append({
                **base, "triton_ms": lam_ms, "triton_dev_ms": lam_dev_ms,
                "vec_ms": lam_vec_ms, "vec_dev_ms": lam_vec_dev_ms,
                "loop_ms": lam_loop_ms, "numpy_ms": lam_np_ms,
                "su_vec": lam_vec_ms / lam_ms,
                "su_vec_dev": lam_vec_dev_ms / lam_dev_ms if lam_dev_ms else float("nan"),
                "su_loop": lam_loop_ms / lam_ms, "su_numpy": lam_np_ms / lam_ms,
            })
            _print_sub_row("lambda-returns", num_envs, seq_len, lam_ms, lam_dev_ms, lam_vec_ms, lam_vec_dev_ms, lam_loop_ms, lam_np_ms)
        if want_disc:
            rows_disc.append({
                **base, "triton_ms": disc_ms, "triton_dev_ms": disc_dev_ms,
                "vec_ms": disc_vec_ms, "vec_dev_ms": disc_vec_dev_ms,
                "loop_ms": disc_loop_ms, "numpy_ms": disc_np_ms,
                "su_vec": disc_vec_ms / disc_ms,
                "su_vec_dev": disc_vec_dev_ms / disc_dev_ms if disc_dev_ms else float("nan"),
                "su_loop": disc_loop_ms / disc_ms, "su_numpy": disc_np_ms / disc_ms,
            })
            _print_sub_row("discounted-returns", num_envs, seq_len, disc_ms, disc_dev_ms, disc_vec_ms, disc_vec_dev_ms, disc_loop_ms, disc_np_ms)
        if want_traces:
            rows_traces.append({
                **base, "triton_ms": trc_ms, "triton_dev_ms": trc_dev_ms,
                "vec_ms": trc_vec_ms, "vec_dev_ms": trc_vec_dev_ms,
                "loop_ms": trc_loop_ms, "numpy_ms": trc_np_ms,
                "su_vec": trc_vec_ms / trc_ms,
                "su_vec_dev": trc_vec_dev_ms / trc_dev_ms if trc_dev_ms else float("nan"),
                "su_loop": trc_loop_ms / trc_ms, "su_numpy": trc_np_ms / trc_ms,
            })
            _print_sub_row("eligibility-traces", num_envs, seq_len, trc_ms, trc_dev_ms, trc_vec_ms, trc_vec_dev_ms, trc_loop_ms, trc_np_ms)

    violations = []
    for label, rows, ms_key, want in [("lambda_returns", rows_lambda, "triton_ms", want_lambda),
                                       ("discounted_returns", rows_disc, "triton_ms", want_disc),
                                       ("eligibility_traces", rows_traces, "triton_ms", want_traces)]:
        if not want:
            continue
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

    def _c_lambda_vec_prod(rewards, next_values, terminateds, gamma, lambda_):
        trunc = torch.zeros_like(terminateds)
        bsv   = torch.zeros_like(rewards)
        return c_lambda_vec(rewards, next_values, terminateds, trunc, bsv, gamma=gamma, lambda_=lambda_)

    def _c_disc_vec_prod(rewards, terminateds, gamma):
        trunc = torch.zeros_like(terminateds)
        bsv   = torch.zeros_like(rewards)
        return c_disc_vec(rewards, terminateds, trunc, bsv, gamma=gamma)

    production_rows = []
    if want_lambda:
        production_rows += _bench_production_regime(
            "lambda-returns", compute_lambda_returns, _c_lambda_vec_prod, _ref_lambda, _make_returns,
            kwargs={"gamma": 0.99, "lambda_": 0.95},
        )
    if want_disc:
        production_rows += _bench_production_regime(
            "discounted-returns", compute_discounted_returns, _c_disc_vec_prod, _ref_disc, _make_returns_2,
            kwargs={"gamma": 0.99},
        )
    if want_traces:
        production_rows += _bench_production_regime(
            "eligibility-traces", compute_eligibility_traces, c_traces_vec, _ref_traces, _make_returns_2,
            kwargs={"gamma": 0.99, "lambda_": 0.9},
        )
    _wanted_labels = {
        "lambda-returns": want_lambda, "discounted-returns": want_disc, "eligibility-traces": want_traces,
    }
    for label in ("lambda-returns", "discounted-returns", "eligibility-traces"):
        if not _wanted_labels[label]:
            continue
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
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, next_values, terminateds = _make_returns(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_lambda_sequential(rewards, next_values, terminateds, truncateds,
                                          bootstrap_values, gamma=0.99, lambda_=0.95)
        triton_out = compute_lambda_returns(rewards, next_values, terminateds, truncateds=truncateds,
                                             gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)
        # Check compiled_vec_trunc itself, not the eager function it wraps --
        # see bench_gae_truncation's matching comment.
        vec_out = compiled_vec_trunc(rewards, next_values, terminateds,
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
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        rewards, _, terminateds = _make_returns(num_envs, seq_len)
        truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
        ni = _n_iter_gpu(seq_len, num_envs)

        ref_out = _ref_discounted_sequential(rewards, terminateds, truncateds, bootstrap_values,
                                              gamma=0.99)
        triton_out = compute_discounted_returns(rewards, terminateds, truncateds=truncateds,
                                                  gamma=0.99, bootstrap_values=bootstrap_values)
        # Check compiled_vec_trunc itself, not the eager function it wraps --
        # see bench_gae_truncation's matching comment.
        vec_out = compiled_vec_trunc(rewards, terminateds, truncateds,
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
        # Reset before every shape -- Bug 2 (see NOTES.md): reusing one
        # torch.compile(...) object across distinct shapes without this can
        # silently give wrong output at some transition. bench_returns()
        # already carried this reset (see its matching comment, where the
        # bug was originally found); this loop shared the identical
        # structure -- one compiled object across the same CONFIGS grid --
        # but had never been given the same protection until directly
        # exercising it on its own (subprocess-isolated from every other
        # algorithm) surfaced a failure here too.
        torch._dynamo.reset()
        print(f"  [{num_envs}×{seq_len}] …", end="", flush=True)
        inputs, dones = _make_prefix_sum(num_envs, seq_len)
        ni = _n_iter_gpu(seq_len, num_envs)

        triton_out = compute_episodic_prefix_sum(inputs, dones)
        ref_out    = reference_episodic_prefix_sum(inputs, dones)
        assert_correctness(triton_out, ref_out, f"prefix_sum[{num_envs}x{seq_len}]")

        # Check the compiled object itself, not the eager
        # vectorized_episodic_prefix_sum it wraps -- see
        # bench_gae_truncation's matching comment.
        vec_out = compiled(inputs, dones)
        assert_correctness(vec_out, ref_out, f"prefix_sum[{num_envs}x{seq_len}] (vec baseline)")

        _warmup_gpu(compute_episodic_prefix_sum, inputs, dones)
        _warmup_gpu(compiled, inputs, dones)
        # NOTE: the compiled baseline is timed with _bench_gpu (CUDA events,
        # explicit sync), not _bench_cpu -- _bench_cpu never syncs the GPU, so
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

def _fmt_row_numpy(r):
    """Row for GAE / V-Trace: triton | dev | vec | vec dev | vs vec | vs vec(dev) | loop | vs loop | numpy | vs numpy | e2e | e2e vs numpy."""
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['vec_ms']:>13.3f} | {r['vec_dev_ms']:>13.3f} | {r['su_vec']:>6.1f}x | {r['su_vec_dev']:>6.1f}x |"
            f" {r['loop_ms']:>10.3f} | {r['su_loop']:>7.1f}x |"
            f" {r['numpy_ms']:>10.3f} | {r['su_numpy']:>8.1f}x |"
            f" {r['e2e_ms']:>14.3f} | {r['su_e2e']:>7.1f}x |")


def _fmt_row_simple(r):
    """Row for λ-returns / discounted-returns / eligibility-traces: triton | dev | vec | vec dev | vs vec | vs vec(dev) | loop | numpy."""
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['vec_ms']:>13.3f} | {r['vec_dev_ms']:>13.3f} | {r['su_vec']:>6.1f}x | {r['su_vec_dev']:>6.1f}x |"
            f" {r['loop_ms']:>10.3f} | {r['su_loop']:>7.1f}x | {r['numpy_ms']:>10.3f} | {r['su_numpy']:>8.1f}x |")


def _fmt_row_retrace(r):
    """Row for Retrace: triton | dev | vec | vec dev | vs vec | vs vec(dev) | loop | vs loop | numpy | vs numpy | e2e | e2e vs numpy."""
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['vec_ms']:>13.3f} | {r['vec_dev_ms']:>13.3f} | {r['su_vec']:>6.1f}x | {r['su_vec_dev']:>6.1f}x |"
            f" {r['loop_ms']:>10.3f} | {r['su_loop']:>7.1f}x |"
            f" {r['numpy_ms']:>10.3f} | {r['su_numpy']:>8.1f}x |"
            f" {r['e2e_ms']:>14.3f} | {r['su_e2e']:>7.1f}x |")


def _fmt_row_prefix(r):
    """Row for episodic prefix sum: triton | dev | compile(vec) | compile(vec) dev | vs vec | vs vec(dev)."""
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} "
            f"| {r['compiled_ms']:>13.3f} | {r['compiled_dev_ms']:>13.3f} | {r['su_compile']:>6.1f}x | {r['su_compile_dev']:>6.1f}x |")


def _table_numpy(title, rows):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |"
                   " loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |")
    sep_cols    = ("|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|"
                   ":-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|")
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_numpy(r) for r in rows)
    return header + "\n" + body


def _table_simple(title, rows):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |"
                   " loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |")
    sep_cols    = ("|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|"
                   ":-------------:|:-------:|:--------------:|:--------:|")
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_simple(r) for r in rows)
    return header + "\n" + body


def _table_retrace(title, rows):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |"
                   " loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |")
    sep_cols    = ("|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|"
                   ":-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|")
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_retrace(r) for r in rows)
    return header + "\n" + body


def _table_prefix_sum(title, rows):
    header_cols = ("| num_envs | seq_len | triton full-call (ms) | triton device (ms) "
                   "| compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |")
    sep_cols    = "|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|"
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_prefix(r) for r in rows)
    return header + "\n" + body


def _fmt_trunc_row(r):
    return (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['triton_dev_ms']:>9.3f} | {r['vec_ms']:>17.3f} "
            f"| {r['vec_dev_ms']:>17.3f} | {r['su_vec']:>9.1f}x | {r['su_vec_dev']:>9.1f}x |")


def _table_truncation(title, rows):
    header = (
        f"#### {title}\n\n"
        "| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |\n"
        "|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|"
    )
    body = "\n".join(_fmt_trunc_row(r) for r in rows)
    return header + "\n" + body


def _fmt_row_production(r):
    marker = " ⚠️" if r["is_boundary"] else ""
    return (f"| {r['algo']:<20} | {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.4f} | {r['triton_dev_ms']:>10.4f} | {r['triton_amort_ms']:>10.4f} "
            f"| {r['vec_ms']:>13.4f} | {r['vec_dev_ms']:>13.4f} | {r['su_vec']:>6.2f}x | {r['su_vec_dev']:>6.2f}x |{marker}")


def _table_production(title, rows):
    """ONE combined table across all 7 algorithms (algo is a row field, not a
    separate table per algorithm) for the production regime (seq_len [80,128]
    x num_envs [4096,8192,16384,32768,38400]) plus the boundary-marker row
    (num_envs=16384, seq_len=16, flagged with a warning marker)."""
    header_cols = (
        "| algo | num_envs | seq_len | triton full-call (ms) | triton device (ms) | "
        "triton amortized (ms) | compile(vec) full-call (ms) | compile(vec) device (ms) | "
        "vs vec (full-call) | vs vec (device) |"
    )
    sep_cols = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    header = f"#### {title}\n\n{header_cols}\n{sep_cols}"
    body = "\n".join(_fmt_row_production(r) for r in rows)
    footnote = (
        "\n\n*⚠️ marks the boundary-marker row (num_envs=16384, seq_len=16). "
        "**vs vec (full-call)** is the headline ratio -- the complete "
        "`compute_*(tensors) -> tensors` call including launch/wrapper overhead, "
        "which a caller pays every invocation. **vs vec (device)** is a diagnostic "
        "showing the same ratio for CUDA-kernel-only time; where full-call and device "
        "speedups diverge, the gap is launch + wrapper overhead. **triton amortized** "
        "is N calls timed inside one region (separates harness per-call sync overhead "
        "from genuine per-call cost) -- reported alongside, not used for any ratio.*"
    )
    return header + "\n" + body + footnote


def _headline(rows):
    """Filter a rows list down to HEADLINE_CONFIGS for benchmarks.md's headline
    tables (~4 representative sizes/algorithm) -- the full CONFIGS grid stays
    reproducible via `python tests/bench_release.py` and is not duplicated here."""
    wanted = set(HEADLINE_CONFIGS)
    return [r for r in rows if (r["num_envs"], r["seq_len"]) in wanted]


def _methodology_text(gpu_label: str) -> str:
    """Shared methodology header -- used by both _section() (console/staged
    candidate) and promote() (benchmarks.md carries it forward unchanged from
    whatever was staged), so benchmarks.md always has the full dtype/
    gamma-lambda/truncation-density/harness explanation, not just a summary."""
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
        f"reference implementation) runs before every timed config -- not bit-identical, since "
        f"`tl.associative_scan` reorders float ops depending on num_warps/block layout, so "
        f"cross-config last-bit differences are legitimate. A monotonicity gate (2% band) then "
        f"asserts a larger problem never measures faster than a smaller one along either swept "
        f"axis. CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.\n\n"
        f"**Two timing granularities.**  "
        f"**triton** (headline) is full-call wall time -- what a caller pays every invocation, "
        f"including launch overhead and wrapper setup (HAS_TRUNCATIONS/HAS_BOOTSTRAP dispatch, "
        f"allocation, layout). All speedup ratios are computed from this number. **dev** is "
        f"device-only CUDA time (`torch.profiler` CUDA activity around steady-state calls, "
        f"ncu/nsys being unavailable in typical containerized GPU environments) -- a diagnostic "
        f"showing pure kernel execution time; where dev is much smaller than the full-call number, "
        f"the gap is launch + wrapper overhead the caller still pays. The production-regime table "
        f"additionally reports an **amortized** variant (N calls in one timed region) for its "
        f"short-seq_len rows, to separate harness per-call sync overhead from genuine per-call "
        f"cost -- the single-call full-call number remains the ratio basis throughout.\n\n"
        f"**Columns.**  "
        f"**triton**: full-call wall time, headline (CUDA events).  "
        f"**dev**: device-only kernel time, diagnostic (see above).  "
        f"**compile(vec)**: `torch.compile` applied to the strongest *correct* vectorized "
        f"PyTorch equivalent found so far – a log2(T)-doubling associative scan "
        f"(`parallel_suffix_scan`/`parallel_prefix_scan`, no Python loop, no log-space); the "
        f"same implementation used for the with-truncations tables (called here with "
        f"truncateds=0) – an earlier log-space cumsum version of this baseline silently "
        f"underflowed to inf/nan at every size in this table and was replaced (see NOTES.md's "
        f"log-space-underflow note for the investigation); there is no longer a separate "
        f"specialized no-truncation baseline to compare against, so the prior compile(assoc) "
        f"column has been dropped as redundant. This is not necessarily the fastest possible "
        f"correct baseline – it pays 6-12 kernel launches per call (one per doubling step) "
        f"where the Triton kernel pays 1-2, and a numerically-stable non-log-space cumsum "
        f"formulation may exist and would be faster; see NOTES.md for that caveat in full.  "
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
# benchmarks.md / unreleased.md path constants
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).parent.parent
BENCHMARKS_MD = REPO_ROOT / "benchmarks.md"

# benchmarks.md holds ONLY the latest release's "## " section(s) -- this fixed
# preamble is rewritten from scratch on every write (never preserved from
# existing content) so stale leftover text before the first "## " heading never
# lingers. Shared by promote() and (as a fallback, when benchmarks.md doesn't
# exist yet) its own "nothing to archive" branch.
_BENCHMARKS_MD_PREAMBLE = (
    "# Benchmarks\n\nLatest release only -- see docs/benchmark-history/ for prior releases.\n\n"
)

BENCHMARK_HISTORY_DIR = REPO_ROOT / "docs" / "benchmark-history"

_RELEASE_SECTION_RE = re.compile(r"## [^\n]*\n.*?(?=\n## |\Z)", re.DOTALL)

UNRELEASED_MD = BENCHMARK_HISTORY_DIR / "unreleased.md"


_SECTION_GPU_RE = re.compile(r"^## unreleased – [^–]+ – (.+)$", re.MULTILINE)


def stage_unreleased_md(gpu_label: str, tables: list[str]) -> None:
    """Write this run's results as the staged release candidate --
    docs/benchmark-history/unreleased.md -- the DEFAULT output target.

    Per this repo's benchmark-placement policy (see benchmarks/README.md):
    a manual sweep run is a CANDIDATE, not a release. This is an UPSERT KEYED
    BY GPU, not a wholesale overwrite of the whole file: a real release can
    require sweeps from more than one card staged together before a single
    promotion (see benchmarks/README.md's placement policy) -- e.g. an H100
    sweep staged, then an RTX 2000 Ada sweep staged afterward, both still
    present when --promote runs. Re-staging the SAME gpu (e.g. rerunning
    while iterating on a fix) replaces that gpu's own section in place;
    staging a DIFFERENT gpu appends alongside whatever else is already
    staged, leaving other gpus' sections untouched. Never archived (there is
    nothing to archive; a discarded candidate section just disappears).
    This function never reads or writes benchmarks.md or README.md.
    Promotion (writing benchmarks.md + a version-tagged archive file, moving
    every staged gpu section across at once) is promote()'s job.
    """
    date = datetime.date.today().isoformat()
    gpu  = gpu_label or _detect_gpu()
    heading = f"## unreleased – {date} – {gpu}\n\n" + _methodology_text(gpu_label)
    body    = "\n\n".join(tables) + "\n"
    new_section = heading + body

    existing = UNRELEASED_MD.read_text() if UNRELEASED_MD.exists() else ""
    prior_sections = _RELEASE_SECTION_RE.findall(existing)
    kept_sections = []
    for sec in prior_sections:
        m = _SECTION_GPU_RE.match(sec)
        if m is None or m.group(1) != gpu:
            kept_sections.append(sec)
    replaced_existing = len(kept_sections) < len(prior_sections)
    all_sections = kept_sections + [new_section]

    BENCHMARK_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    UNRELEASED_MD.write_text("# Benchmarks archive: unreleased\n\n" + "\n\n".join(all_sections))
    other_gpus = len(all_sections) - 1
    if replaced_existing:
        note = f"replaced this gpu's existing staged section"
    elif other_gpus:
        note = f"added alongside {other_gpus} other staged gpu section(s)"
    else:
        note = "first staged section in this candidate"
    print(f"\nStaged candidate written to {UNRELEASED_MD} ({note}) -- NOT a release; "
          f"benchmarks.md is untouched. Promote by hand, or rerun with "
          f"`--promote --version <tag>` to cut an actual release.")


_UNRELEASED_STUB = (
    "# Benchmarks archive: unreleased\n\n"
    "No candidate currently staged. Under this repo's benchmark-placement policy (see\n"
    "`benchmarks/README.md`), this file holds only the most recent not-yet-promoted release\n"
    "candidate; it resets to this empty state immediately after that candidate is promoted to\n"
    "`benchmarks.md` + a version-tagged archive file on release. It is overwritten wholesale by\n"
    "each new candidate, not appended to -- do not treat it as a running log.\n"
)

# A real release tag, e.g. v0.1.1 -- deliberately stricter than "any non-whitespace
# token" so a heading that merely *looks* like it has a version (typos, "latest",
# "unreleased" itself) is refused rather than silently treated as one.
_VERSION_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")

_STAGED_HEADING_RE = re.compile(r"^## unreleased – .+$", re.MULTILINE)


def promote(version: str) -> None:
    """Promote the currently staged candidate (docs/benchmark-history/unreleased.md)
    to benchmarks.md as an actual release, tagged --version.

    Purely mechanical relocation of already-rendered markdown -- runs no benchmarks,
    computes no numbers, needs no GPU. unreleased.md may hold more than one "## "
    section (one per gpu staged via stage_unreleased_md's upsert-by-gpu -- e.g. an
    H100 sweep and an RTX 2000 Ada sweep staged separately before a release that
    needs both); every staged section moves across together, all tagged the same
    --version. Three steps, each gated so a failure never leaves files partially
    mutated:

      1. Find EVERY "## " section currently in benchmarks.md and parse the first
         one's version from ITS OWN heading (never from --version, never guessed;
         all sections of one release share one tag by construction, see step 2) --
         archive all of them together to docs/benchmark-history/<that-version>.md.
         If benchmarks.md has no release section at all (nothing has ever been
         legitimately promoted), there is nothing to archive and this step is
         skipped, not refused -- that is the expected shape of the very first
         promotion.
      2. Rewrite every staged section's heading from "## unreleased -- ..." to
         "## <version> -- ..." (date and GPU label are left exactly as the sweep
         recorded them -- they describe when/on what hardware the numbers were
         actually measured, not when promotion happened) and write all of them,
         in the order staged, as the new benchmarks.md.
      3. Reset unreleased.md to its stub -- every staged section has been consumed.

    Refuses outright, before touching any file, if:
      - unreleased.md doesn't exist or has no staged candidate (still its stub) --
        nothing to promote.
      - benchmarks.md DOES have a "## " release section but its (first) heading
        token doesn't parse as a real version tag (e.g. it literally reads
        "unreleased" -- meaning some prior run bypassed --promote and wrote a
        staged-shaped section directly into benchmarks.md). Archiving that
        under a guessed name, or under the literal string "unreleased", would
        collide with the real staging file and corrupt history -- refuse and let a
        human sort out how benchmarks.md got into that state instead.
      - the archive destination for benchmarks.md's outgoing section already exists
        (release history is immutable -- never silently overwrite a prior archive).

    Running this twice in a row is safe: the second run finds unreleased.md already
    reset to its stub (from step 3 of the first run) and refuses at the first check,
    rather than re-promoting stale content or double-archiving.
    """
    if not _VERSION_TAG_RE.match(version):
        print(f"--promote --version {version!r} doesn't look like a release tag "
              f"(expected e.g. v0.1.1) -- refusing.")
        sys.exit(1)

    if not UNRELEASED_MD.exists():
        print(f"{UNRELEASED_MD} does not exist -- nothing staged to promote.")
        sys.exit(1)
    unreleased_text = UNRELEASED_MD.read_text()
    if _STAGED_HEADING_RE.search(unreleased_text) is None:
        print(f"{UNRELEASED_MD} has no staged candidate (still its reset stub) -- "
              f"nothing to promote. Run a sweep first (plain `python tests/bench_release.py`, "
              f"the staging default) to produce one.")
        sys.exit(1)
    staged_sections = _RELEASE_SECTION_RE.findall(unreleased_text)
    if not staged_sections:
        # Unreachable if the heading check above passed (the heading regex is a
        # subset of what _RELEASE_SECTION_RE matches) -- guard anyway rather than
        # ever promoting something half-parsed.
        print(f"{UNRELEASED_MD} has an 'unreleased' heading but its section body "
              f"didn't parse -- refusing rather than guessing.")
        sys.exit(1)

    # Step 1: archive benchmarks.md's own current release (every "## " section
    # in it, together), named from its FIRST section's own header.
    existing = BENCHMARKS_MD.read_text() if BENCHMARKS_MD.exists() else ""
    prior_sections = _RELEASE_SECTION_RE.findall(existing)
    if prior_sections:
        prior_version_match = re.match(r"## (\S+)", prior_sections[0])
        prior_version = prior_version_match.group(1) if prior_version_match else None
        if prior_version is None or not _VERSION_TAG_RE.match(prior_version):
            print(f"benchmarks.md's current release header ({prior_version!r}) doesn't "
                  f"parse as a release tag -- refusing to archive it under a guessed or "
                  f"colliding name. (If this reads 'unreleased', benchmarks.md was written "
                  f"directly by some path that bypassed --promote; fix that by hand -- e.g. "
                  f"relocate it to docs/benchmark-history/unreleased.md -- before promoting.)")
            sys.exit(1)
        archive_path = BENCHMARK_HISTORY_DIR / f"{prior_version}.md"
        if archive_path.exists():
            print(f"{archive_path} already exists -- refusing to overwrite release "
                  f"history. (Has {prior_version} already been archived? Check whether "
                  f"promotion already happened before re-running.)")
            sys.exit(1)
        BENCHMARK_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(f"# Benchmarks archive: {prior_version}\n\n" + "\n\n".join(prior_sections))
        n = len(prior_sections)
        print(f"  (archived outgoing release '{prior_version}' ({n} gpu section(s)) -> {archive_path})")
    else:
        print("  (benchmarks.md has no current release section -- nothing to archive; "
              "this is the first promotion.)")

    # Step 2: move every staged section into benchmarks.md, rewriting each
    # heading from "unreleased" to the promoted tag.
    promoted_sections = [
        sec.replace("## unreleased –", f"## {version} –", 1) for sec in staged_sections
    ]
    preamble = _BENCHMARKS_MD_PREAMBLE
    BENCHMARKS_MD.write_text(preamble + "\n\n".join(promoted_sections))
    print(f"benchmarks.md promoted to {version} ({len(promoted_sections)} gpu section(s)) "
          f"({BENCHMARKS_MD})")

    # Step 3: reset unreleased.md to its stub -- the candidate has been consumed.
    UNRELEASED_MD.write_text(_UNRELEASED_STUB)
    print(f"{UNRELEASED_MD} reset to its stub -- staged candidate consumed by promotion.")


def _bench_truncation_headline(num_envs=4096, seq_len=128):
    """Truncation-path full-call speedup at the SAME (num_envs, seq_len) as the
    README draft's main table, for the 5 algorithms that have a vectorized
    with-truncations baseline to compare against (GAE, V-Trace, lambda-returns,
    discounted-returns, Retrace). The other 2 algorithms in ALL_ALGOS are
    absent from this dict because they have no truncation-path concept at
    all -- render_readme_table_draft() states this inline rather than
    silently dropping the rows, see that function: compute_eligibility_traces
    and compute_episodic_prefix_sum each take only a single `dones` flag with
    no terminated/truncated distinction and no bootstrap_values parameter
    (see their signatures in rl_triton/ops/returns.py and
    rl_triton/ops/prefix_sum.py) -- there is no "_with_truncations" variant
    to build because the kernels themselves have nothing to distinguish.

    Retrace was previously omitted here even though its kernel supports
    truncations and a valid baseline (vectorized_retrace, test_retrace.py)
    already exists -- nobody had wired it into this specific headline table.
    Now wired in below. Note this required its own sequential reference:
    bench_release.py's other _ref_retrace (used by bench_retrace()'s main
    sweep) takes a single combined `dones` flag and cannot distinguish
    terminated (zero the Q-bootstrap) from truncated (keep the bootstrap,
    sever the trace) -- it predates the truncation feature and is only valid
    for bench_retrace()'s existing always-zero-truncateds usage. The Retrace
    block below imports test_retrace.py's reference_retrace instead, the
    real truncation-aware ground truth already used by
    test_retrace_truncated_keeps_bootstrap and friends.

    Existing truncation-path tables in benchmarks.md run at the full CONFIGS
    grid (64x512 .. 16384x512), which already includes (4096,128) -- but
    those tables are keyed by algorithm and rendered straight to markdown, not
    exposed as a flat structure this function could reuse without threading
    new plumbing through --parent-sweep's JSON payload. Recomputing directly
    here (correctness-gated the same way) is the smaller change; it also
    means this headline never silently drifts from a table row it wasn't
    actually re-derived from.

    Unlike every bench_*() function above, this one used to compute all three
    ratios with NO correctness check anywhere -- neither the Triton output nor
    either compiled baseline was ever verified against a reference before its
    timing was trusted here, the widest instance of the gap described in
    NOTES.md/the Retrace-baseline fix. Fixed: each block now asserts both
    sides against the same sequential reference the corresponding full-sweep
    bench_*_truncation() function uses, before any _bench_gpu call.
    """
    results = {}
    ni = _n_iter_gpu(seq_len, num_envs)

    terminateds = _make_gae(num_envs, seq_len)[2]
    truncateds, bootstrap_values = _make_trunc_extras(num_envs, seq_len, terminateds)
    rewards, values, _ = _make_gae(num_envs, seq_len)
    compiled = torch.compile(vectorized_gae_with_truncations)
    kwargs_triton = dict(gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values)

    ref_out = _ref_gae_trunc(rewards, values, terminateds, truncateds, bootstrap_values,
                              gamma=0.99, lambda_=0.95)
    triton_out = compute_gae(rewards, values, terminateds, truncateds, **kwargs_triton)
    assert_correctness(triton_out, ref_out, f"headline_gae_trunc[{num_envs}x{seq_len}] (triton vs ref)")
    vec_out = compiled(rewards, values, terminateds, truncateds, bootstrap_values, 0.99, 0.95)
    assert_correctness(vec_out, ref_out, f"headline_gae_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

    _warmup_gpu(compute_gae, rewards, values, terminateds, truncateds, **kwargs_triton)
    _warmup_gpu(compiled, rewards, values, terminateds, truncateds, bootstrap_values, 0.99, 0.95)
    triton_ms = _bench_gpu(compute_gae, rewards, values, terminateds, truncateds, n_iter=ni, **kwargs_triton)
    vec_ms = _bench_gpu(compiled, rewards, values, terminateds, truncateds, bootstrap_values, 0.99, 0.95, n_iter=ni)
    results["GAE"] = vec_ms / triton_ms

    log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt = _make_vtrace(num_envs, seq_len)
    truncateds_vt, bootstrap_vt = _make_trunc_extras(num_envs, seq_len, terminateds_vt)
    compiled_vt = torch.compile(vectorized_vtrace_with_truncations)
    kwargs_vt = dict(truncateds=truncateds_vt, gamma=0.99, bootstrap_values=bootstrap_vt)

    ref_out_vt = _ref_vtrace_sequential(log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt,
                                         truncateds_vt, bootstrap_vt, gamma=0.99)
    triton_out_vt = compute_vtrace_fused(log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt, **kwargs_vt)
    assert_correctness(triton_out_vt, ref_out_vt, f"headline_vtrace_trunc[{num_envs}x{seq_len}] (triton vs ref)")
    vec_out_vt = compiled_vt(log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt,
                              truncateds_vt, bootstrap_vt, gamma=0.99)
    assert_correctness(vec_out_vt, ref_out_vt, f"headline_vtrace_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

    _warmup_gpu(compute_vtrace_fused, log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt, **kwargs_vt)
    _warmup_gpu(compiled_vt, log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt,
                truncateds_vt, bootstrap_vt, gamma=0.99)
    triton_ms = _bench_gpu(compute_vtrace_fused, log_pi_t, log_pi_b, values_vt, rewards_vt,
                            terminateds_vt, n_iter=ni, **kwargs_vt)
    vec_ms = _bench_gpu(compiled_vt, log_pi_t, log_pi_b, values_vt, rewards_vt, terminateds_vt,
                         truncateds_vt, bootstrap_vt, gamma=0.99, n_iter=ni)
    results["V-Trace"] = vec_ms / triton_ms

    rewards_lr, next_values_lr, terminateds_lr = _make_returns(num_envs, seq_len)
    truncateds_lr, bootstrap_lr = _make_trunc_extras(num_envs, seq_len, terminateds_lr)
    compiled_lr = torch.compile(vectorized_lambda_returns_with_truncations)
    kwargs_lr = dict(truncateds=truncateds_lr, gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_lr)

    ref_out_lr = _ref_lambda_sequential(rewards_lr, next_values_lr, terminateds_lr, truncateds_lr,
                                         bootstrap_lr, gamma=0.99, lambda_=0.95)
    triton_out_lr = compute_lambda_returns(rewards_lr, next_values_lr, terminateds_lr, **kwargs_lr)
    assert_correctness(triton_out_lr, ref_out_lr, f"headline_lambda_trunc[{num_envs}x{seq_len}] (triton vs ref)")
    vec_out_lr = compiled_lr(rewards_lr, next_values_lr, terminateds_lr, truncateds_lr, bootstrap_lr,
                              gamma=0.99, lambda_=0.95)
    assert_correctness(vec_out_lr, ref_out_lr, f"headline_lambda_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

    _warmup_gpu(compute_lambda_returns, rewards_lr, next_values_lr, terminateds_lr, **kwargs_lr)
    _warmup_gpu(compiled_lr, rewards_lr, next_values_lr, terminateds_lr, truncateds_lr, bootstrap_lr,
                gamma=0.99, lambda_=0.95)
    triton_ms = _bench_gpu(compute_lambda_returns, rewards_lr, next_values_lr, terminateds_lr,
                            n_iter=ni, **kwargs_lr)
    vec_ms = _bench_gpu(compiled_lr, rewards_lr, next_values_lr, terminateds_lr, truncateds_lr,
                         bootstrap_lr, gamma=0.99, lambda_=0.95, n_iter=ni)
    results["lambda-returns"] = vec_ms / triton_ms

    rewards_dr, _, terminateds_dr = _make_returns(num_envs, seq_len)
    truncateds_dr, bootstrap_dr = _make_trunc_extras(num_envs, seq_len, terminateds_dr)
    compiled_dr = torch.compile(vectorized_discounted_returns_with_truncations)

    ref_out_dr = _ref_discounted_sequential(rewards_dr, terminateds_dr, truncateds_dr,
                                             bootstrap_dr, gamma=0.99)
    triton_out_dr = compute_discounted_returns(rewards_dr, terminateds_dr, truncateds=truncateds_dr,
                                                gamma=0.99, bootstrap_values=bootstrap_dr)
    assert_correctness(triton_out_dr, ref_out_dr, f"headline_disc_trunc[{num_envs}x{seq_len}] (triton vs ref)")
    vec_out_dr = compiled_dr(rewards_dr, terminateds_dr, truncateds_dr, bootstrap_dr, gamma=0.99)
    assert_correctness(vec_out_dr, ref_out_dr, f"headline_disc_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

    _warmup_gpu(compute_discounted_returns, rewards_dr, terminateds_dr, truncateds=truncateds_dr,
                gamma=0.99, bootstrap_values=bootstrap_dr)
    _warmup_gpu(compiled_dr, rewards_dr, terminateds_dr, truncateds_dr, bootstrap_dr, gamma=0.99)
    triton_ms = _bench_gpu(compute_discounted_returns, rewards_dr, terminateds_dr,
                            truncateds=truncateds_dr, gamma=0.99,
                            bootstrap_values=bootstrap_dr, n_iter=ni)
    vec_ms = _bench_gpu(compiled_dr, rewards_dr, terminateds_dr, truncateds_dr, bootstrap_dr,
                         gamma=0.99, n_iter=ni)
    results["discounted-returns"] = vec_ms / triton_ms

    # Retrace has no no-truncation specialization to begin with -- both
    # compute_retrace and vectorized_retrace take terminateds/truncateds as
    # two mandatory, distinct arguments always (see compute_retrace's
    # docstring: the Q-bootstrap is folded into next_q_values_all, so there
    # is no separate bootstrap_values parameter to wire up here, unlike
    # GAE/V-Trace/lambda-returns/discounted-returns above). bench_retrace()
    # elsewhere in this file always passes truncateds=torch.zeros_like(dones)
    # via _retrace_kernel_args, so this is the first place Retrace's
    # truncation path is actually exercised with a non-zero truncateds at
    # this headline config.
    args_gpu_rt = _make_retrace(num_envs, seq_len)
    rewards_rt, terminateds_rt, values_rt, next_q_all_rt, q_values_rt, actions_rt, apt_rt, apb_rt = args_gpu_rt
    truncateds_rt = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float() * (1.0 - terminateds_rt)
    compiled_rt = torch.compile(vectorized_retrace)

    # bench_release.py's own _ref_retrace is stale here: it takes a single
    # combined `dones` flag and has no way to distinguish terminated (zero
    # the Q-bootstrap) from truncated (keep the bootstrap, sever the trace)
    # -- it predates the truncation feature and is only valid for
    # bench_retrace()'s existing always-zero-truncateds usage elsewhere in
    # this file. reference_retrace (test_retrace.py) is the real
    # truncation-aware ground truth already used by
    # test_retrace_truncated_keeps_bootstrap and friends.
    ref_out_rt, _ = reference_retrace(apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                                       rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99)
    triton_out_rt, _ = compute_retrace(apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                                        rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99)
    assert_correctness(triton_out_rt, ref_out_rt, f"headline_retrace_trunc[{num_envs}x{seq_len}] (triton vs ref)")
    # Check compiled_rt itself, not the eager vectorized_retrace it wraps --
    # see bench_gae_truncation's matching comment.
    vec_out_rt, _ = compiled_rt(apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                                 rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99)
    assert_correctness(vec_out_rt, ref_out_rt, f"headline_retrace_trunc[{num_envs}x{seq_len}] (vectorized baseline vs ref)")

    _warmup_gpu(compute_retrace, apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99)
    _warmup_gpu(compiled_rt, apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99)
    triton_ms = _bench_gpu(compute_retrace, apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                            rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99, n_iter=ni)
    vec_ms = _bench_gpu(compiled_rt, apt_rt, apb_rt, q_values_rt, next_q_all_rt, actions_rt,
                         rewards_rt, terminateds_rt, truncateds_rt, gamma=0.99, n_iter=ni)
    results["Retrace"] = vec_ms / triton_ms

    return results


def render_readme_table_draft(gpu_label: str, production_rows: list[dict],
                               truncation_headline: dict | None = None) -> str:
    """Render the README's production-relevant summary table WITHOUT touching
    README.md -- README prose edits are a STOP-and-report item under the
    autonomy boundary, and this script has no code path that writes README.md at
    all. Caller should write this to a draft file for human review and manual
    inclusion in README.md.

    All rows use the SAME (num_envs, seq_len) so the table is a genuine
    apples-to-apples comparison across algorithms -- mixing different sizes per
    row (an earlier version of this function did) makes speedups incomparable
    and risks reading as cherry-picked even when it isn't.

    Includes all 7 of ALL_ALGOS in the plain table (all 7 always have a
    production row -- see bench_release.py's per-algorithm bench_*() functions,
    each of which contributes to production_rows_all regardless of which
    algorithms were run). An earlier version of this function hardcoded a
    4-algorithm subset here with no stated reason for the other 3 -- traced
    back to this function's very first version (commit bac9325), never
    revisited. Silently dropping algorithms from the front-page table is not
    acceptable; if an algorithm's production row is genuinely missing (e.g. a
    partial --algos run), it is just absent from `production_rows` and this
    loop naturally skips it -- that is a caller's explicit selection, not a
    hardcoded omission baked into this function.

    The truncation-path table covers the 5 algorithms that have a genuine
    vectorized with-truncations baseline (GAE, V-Trace, Retrace, lambda-returns,
    discounted-returns -- see _bench_truncation_headline). The other 2
    (eligibility-traces, prefix-sum) are stated inline with the actual reason
    -- a structural non-applicability, since those kernels have no
    terminated/truncated distinction to begin with -- never silently omitted.
    """
    gpu = gpu_label or _detect_gpu()
    date = datetime.date.today().isoformat()
    fixed_num_envs, fixed_seq_len = 4096, 128  # PufferLib/Gigaflow-default rollout size
    algos = ["GAE", "V-Trace", "Retrace", "lambda-returns", "discounted-returns",
             "eligibility-traces", "prefix-sum"]
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

    if truncation_headline:
        lines.append("")
        lines.append(
            "With truncations (terminations + time-limit truncations + bootstrap values), "
            "same config. Eligibility-traces and episodic-prefix-sum have no row here, for a "
            "structural reason, not an unwired gap: both kernels take only a single `dones` flag "
            "with no terminated/truncated distinction and no bootstrap values, so there is no "
            "truncation-path baseline to compare against for either. Every other algorithm, "
            "including Retrace (terminated/truncated are both mandatory, distinct arguments, and "
            "no separate bootstrap_values parameter -- the continuation value is folded into "
            "next_q_values_all every step, see docs/kernels/retrace.md §4), has a row below."
        )
        lines.append("")
        lines.append("| algorithm | speedup vs torch.compile, with truncations (full-call) |")
        lines.append("|:---|:---:|")
        for algo in ["GAE", "V-Trace", "Retrace", "lambda-returns", "discounted-returns"]:
            if algo in truncation_headline:
                lines.append(f"| {algo} | {truncation_headline[algo]:.1f}× |")

    lines.append("")
    lines.append("See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-update", action="store_true",
                        help="Print results but do not modify benchmarks.md or "
                             "docs/benchmark-history/unreleased.md")
    parser.add_argument("--promote", action="store_true",
                        help="Promote the currently staged candidate "
                             "(docs/benchmark-history/unreleased.md) to benchmarks.md as an "
                             "actual release, archiving benchmarks.md's own outgoing release "
                             "first (named from ITS OWN header, not --version). Requires "
                             "--version -- the tag to promote the staged candidate AS; "
                             "unreleased.md itself never carries a version, it is always "
                             "headed 'unreleased' until promoted. Purely mechanical: runs no "
                             "sweep, needs no GPU/CUDA. This is the ONLY way to write "
                             "benchmarks.md -- there is no direct-from-sweep path anymore (see "
                             "benchmarks/README.md's placement policy).")
    parser.add_argument("--gpu", default="", metavar="LABEL",
                        help="GPU label to embed in the table header (default: auto-detect)")
    parser.add_argument("--version", default="", metavar="TAG",
                        help="Release version tag (e.g. v0.1.0), required with --promote. "
                             "Ignored otherwise -- a staged candidate is always headed "
                             "'unreleased' until promoted.")
    parser.add_argument("--algos", default="all", metavar="LIST",
                        help="Comma-separated subset of ALL_ALGOS to run (default: all). "
                             f"Choices: {','.join(ALL_ALGOS)}. Used for the GAE-only smoke "
                             "test before committing to the full unattended sweep.")
    parser.add_argument("--variant", default="all", choices=["all", "plain", "truncation"],
                        help="For the five algorithms with both a plain and a truncation-path "
                             "table (gae, vtrace, retrace, lambda_returns, discounted_returns): "
                             "which to run. 'all' (default) runs both -- the normal manual-run "
                             "and --algos-smoke-test behavior. 'plain'/'truncation' run only that "
                             "table; used internally by --parent-sweep to put the two tables in "
                             "separate subprocesses, since each builds its own "
                             "torch.compile(...) wrapper of the same underlying vectorized "
                             "function (IDENTICAL for gae/vtrace/lambda_returns/discounted_"
                             "returns' with_truncations variant; retrace reuses vectorized_"
                             "retrace itself for both tables, since it has no separate "
                             "with-truncations variant) -- two such wrappers in one process can "
                             "silently miscompile one of them once either has compiled at 2+ "
                             "distinct shapes (see NOTES.md). Has no effect on eligibility_traces "
                             "or prefix_sum -- each has only one table.")
    parser.add_argument("--output-json", default="", metavar="PATH",
                        help="Dump this invocation's tables/rows/violations as JSON to PATH "
                             "instead of staging a candidate. Used internally by "
                             "--parent-sweep's per-algorithm/variant subprocesses.")
    parser.add_argument("--parent-sweep", action="store_true",
                        help="Run each (algorithm, variant) group in its OWN subprocess and "
                             "merge their --output-json results here before staging the "
                             "combined candidate -- see _PARENT_SWEEP_GROUPS for the exact list. "
                             "Two isolation concerns are handled this way, both via the same "
                             "fresh-CUDA-context mechanism: (1) a single process running the "
                             "full multi-algorithm sweep hits a torch.compile/CUDA illegal-"
                             "memory-access crash from accumulated Dynamo/Inductor state -- "
                             "confirmed independent of torch._dynamo.reset() usage; (2) two "
                             "separate torch.compile(...) wrappers of the SAME underlying "
                             "vectorized function in one process can silently miscompile one of "
                             "them (see NOTES.md) -- confirmed ALSO independent of reset() usage, "
                             "which is why the plain and truncation tables for gae/vtrace/retrace/"
                             "lambda_returns/discounted_returns are split into separate "
                             "subprocesses via --variant, not just separate algorithms.")
    args = parser.parse_args()

    if args.promote:
        if not args.version:
            print("--promote requires --version (e.g. --version v0.1.1) -- refusing to "
                  "promote the staged candidate to an unnamed release.")
            sys.exit(1)
        promote(args.version)
        return

    selected_algos = ALL_ALGOS if args.algos == "all" else [a.strip() for a in args.algos.split(",")]
    unknown = set(selected_algos) - set(ALL_ALGOS)
    if unknown:
        print(f"Unknown --algos entries: {unknown}. Choices: {ALL_ALGOS}")
        sys.exit(1)

    print_environment_header("bench_release.py")

    if args.parent_sweep:
        _run_parent_sweep(selected_algos, args)
        return

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
                 "loop_ms": 45.678, "su_loop": 37.0}
        dummy_trunc = {"num_envs": 128, "seq_len": 1024, "triton_ms": 1.234, "triton_dev_ms": 0.987,
                       "vec_ms": 1.890, "vec_dev_ms": 1.654, "su_vec": 1.5, "su_vec_dev": 1.68}
        dummy_prod = {"algo": "GAE", "num_envs": 4096, "seq_len": 128, "is_boundary": False,
                      "triton_ms": 0.234, "triton_dev_ms": 0.187, "triton_amort_ms": 0.201,
                      "vec_ms": 0.410, "vec_dev_ms": 0.365, "su_vec": 1.75, "su_vec_dev": 1.95}
        tables = [
            _table_numpy("GAE (`compute_gae`)", [dummy]),
            _table_truncation("GAE – with truncations (`compute_gae`)", [dummy_trunc]),
            _table_numpy("V-Trace (`compute_vtrace`)", [dummy]),
            _table_truncation("V-Trace – with truncations (`compute_vtrace`)", [dummy_trunc]),
            _table_retrace("Retrace(λ) (`compute_retrace`)", [dummy]),
            _table_truncation("Retrace(λ) – with truncations (`compute_retrace`)", [dummy_trunc]),
            _table_simple("λ-returns (`compute_lambda_returns`)", [dummy]),
            _table_truncation("λ-returns – with truncations (`compute_lambda_returns`)", [dummy_trunc]),
            _table_simple("Discounted returns (`compute_discounted_returns`)", [dummy]),
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

    # For gae/vtrace/lambda_returns/discounted_returns, --variant selects which of
    # the two tables (each backed by its OWN torch.compile(...) wrapper of the same
    # underlying vectorized_*_with_truncations function) this invocation computes --
    # see --variant's own help and NOTES.md for why the two must never share a
    # process. retrace/eligibility_traces/prefix_sum have only one table each and
    # are unaffected by either flag.
    run_plain = args.variant in ("all", "plain")
    run_trunc = args.variant in ("all", "truncation")

    if "gae" in selected_algos:
        if run_plain:
            print("Running GAE benchmark …", flush=True)
            gae_rows, gae_prod, v = bench_gae()
            all_violations += v
            production_rows_all += gae_prod
            full_tables.append(_table_numpy("GAE (`compute_gae`)", gae_rows))
            headline_tables.append(_table_numpy("GAE (`compute_gae`)", _headline(gae_rows)))
        if run_trunc:
            print("Running GAE truncation-path benchmark …", flush=True)
            gae_trunc_rows, v = bench_gae_truncation()
            all_violations += v
            full_tables.append(_table_truncation("GAE – with truncations (`compute_gae`)", gae_trunc_rows))
            headline_tables.append(_table_truncation("GAE – with truncations (`compute_gae`)", _headline(gae_trunc_rows)))

    if "vtrace" in selected_algos:
        if run_plain:
            print("Running V-Trace benchmark …", flush=True)
            vtrace_rows, vtrace_prod, v = bench_vtrace()
            all_violations += v
            production_rows_all += vtrace_prod
            full_tables.append(_table_numpy("V-Trace (`compute_vtrace`)", vtrace_rows))
            headline_tables.append(_table_numpy("V-Trace (`compute_vtrace`)", _headline(vtrace_rows)))
        if run_trunc:
            print("Running V-Trace truncation-path benchmark …", flush=True)
            vtrace_trunc_rows, v = bench_vtrace_truncation()
            all_violations += v
            full_tables.append(_table_truncation("V-Trace – with truncations (`compute_vtrace`)", vtrace_trunc_rows))
            headline_tables.append(_table_truncation("V-Trace – with truncations (`compute_vtrace`)", _headline(vtrace_trunc_rows)))

    if "retrace" in selected_algos:
        if run_plain:
            print("Running Retrace(λ) benchmark …", flush=True)
            retrace_rows, retrace_prod, v = bench_retrace()
            all_violations += v
            production_rows_all += retrace_prod
            full_tables.append(_table_retrace("Retrace(λ) (`compute_retrace`)", retrace_rows))
            headline_tables.append(_table_retrace("Retrace(λ) (`compute_retrace`)", _headline(retrace_rows)))
        if run_trunc:
            print("Running Retrace(λ) truncation-path benchmark …", flush=True)
            retrace_trunc_rows, v = bench_retrace_truncation()
            all_violations += v
            full_tables.append(_table_truncation("Retrace(λ) – with truncations (`compute_retrace`)", retrace_trunc_rows))
            headline_tables.append(_table_truncation("Retrace(λ) – with truncations (`compute_retrace`)", _headline(retrace_trunc_rows)))

    # bench_returns() alone produces: the "plain" table for lambda_returns/
    # discounted_returns (only when run_plain), and eligibility_traces' ONLY table
    # (it has no separate truncation path, so it's always wanted whenever selected,
    # regardless of --variant -- a truncation-only subprocess that happens to also
    # select eligibility_traces would otherwise silently drop it).
    _returns_trio = {"lambda_returns", "discounted_returns", "eligibility_traces"}
    _trio_selected = _returns_trio & set(selected_algos)
    _want_plain_in_trio = {a for a in _trio_selected if a == "eligibility_traces" or run_plain}
    if _want_plain_in_trio:
        print("Running returns / eligibility-traces benchmark …", flush=True)
        lambda_rows, disc_rows, traces_rows, returns_prod, v = bench_returns(_want_plain_in_trio)
        all_violations += v
        production_rows_all += returns_prod
        if "lambda_returns" in _want_plain_in_trio:
            full_tables.append(_table_simple("λ-returns (`compute_lambda_returns`)", lambda_rows))
            headline_tables.append(_table_simple("λ-returns (`compute_lambda_returns`)", _headline(lambda_rows)))
        if "discounted_returns" in _want_plain_in_trio:
            full_tables.append(_table_simple("Discounted returns (`compute_discounted_returns`)", disc_rows))
            headline_tables.append(_table_simple("Discounted returns (`compute_discounted_returns`)", _headline(disc_rows)))
        if "eligibility_traces" in _want_plain_in_trio:
            full_tables.append(_table_simple("Eligibility traces (`compute_eligibility_traces`)", traces_rows))
            headline_tables.append(_table_simple("Eligibility traces (`compute_eligibility_traces`)", _headline(traces_rows)))

    if "lambda_returns" in selected_algos and run_trunc:
        print("Running λ-returns truncation-path benchmark …", flush=True)
        lambda_trunc_rows, v = bench_lambda_returns_truncation()
        all_violations += v
        full_tables.append(_table_truncation("λ-returns – with truncations (`compute_lambda_returns`)", lambda_trunc_rows))
        headline_tables.append(_table_truncation("λ-returns – with truncations (`compute_lambda_returns`)", _headline(lambda_trunc_rows)))

    if "discounted_returns" in selected_algos and run_trunc:
        print("Running discounted-returns truncation-path benchmark …", flush=True)
        disc_trunc_rows, v = bench_discounted_returns_truncation()
        all_violations += v
        full_tables.append(_table_truncation("Discounted returns – with truncations (`compute_discounted_returns`)", disc_trunc_rows))
        headline_tables.append(_table_truncation("Discounted returns – with truncations (`compute_discounted_returns`)", _headline(disc_trunc_rows)))

    if "prefix_sum" in selected_algos:
        print("Running episodic prefix sum benchmark …", flush=True)
        prefix_sum_rows, prefix_prod, v = bench_prefix_sum()
        all_violations += v
        production_rows_all += prefix_prod
        full_tables.append(_table_prefix_sum("Episodic prefix sum (`compute_episodic_prefix_sum`)", prefix_sum_rows))
        headline_tables.append(_table_prefix_sum("Episodic prefix sum (`compute_episodic_prefix_sum`)", _headline(prefix_sum_rows)))

    # --parent-sweep's per-group subprocesses stop here: dump raw tables/rows
    # instead of building the (cross-group) production table or touching
    # benchmarks.md/README.md -- the parent does that once, after merging
    # every group's output, so there's exactly one combined production table
    # rather than one fragment per subprocess.
    if args.output_json:
        # bench_returns() unconditionally computes+returns production rows for
        # ALL THREE of lambda_returns/discounted_returns/eligibility_traces
        # whenever ANY of them is selected (they share one bench_returns() call
        # -- see its call site's comment). Now that _PARENT_SWEEP_GROUPS runs
        # each of those three in its OWN subprocess (see that constant's
        # comment for why), every such subprocess's production_rows_all still
        # contains all three algos' rows -- filter down to just the ones this
        # invocation actually selected, or the parent's merge across 3
        # subprocesses would triple lambda-returns/discounted-returns/
        # eligibility-traces production rows in the final combined table.
        _ALGO_TO_PROD_LABEL = {
            "gae": "GAE", "vtrace": "V-Trace", "retrace": "Retrace",
            "lambda_returns": "lambda-returns", "discounted_returns": "discounted-returns",
            "eligibility_traces": "eligibility-traces", "prefix_sum": "prefix-sum",
        }
        selected_prod_labels = {_ALGO_TO_PROD_LABEL[a] for a in selected_algos}
        production_rows_all = [r for r in production_rows_all if r["algo"] in selected_prod_labels]
        payload = {
            "full_tables": full_tables,
            "headline_tables": headline_tables,
            "production_rows_all": production_rows_all,
            "all_violations": all_violations,
        }
        Path(args.output_json).write_text(json.dumps(payload))
        print(f"\nWrote partial results to {args.output_json}", flush=True)
        # Do NOT sys.exit(1) here for monotonicity violations alone: the
        # monotonicity gate is a soft, noise-tolerant check (2% band; ~2-4%
        # GPU timing jitter is expected and documented, not a correctness
        # failure -- see check_monotonic_grid's docstring). violations are
        # already serialized into the payload above and get merged into
        # all_violations by the parent, which reports them and sets the
        # FINAL exit code via _finalize -- exactly matching the single-
        # process (non---parent-sweep) path's behavior, where monotonicity
        # noise in one algorithm never prevents the rest of the sweep from
        # running. A real failure (assert_correctness raising) is an
        # uncaught exception, which already exits non-zero on its own and
        # is correctly treated as fatal by _run_parent_sweep's returncode
        # check below -- this change does not weaken that.
        return

    _finalize(full_tables, headline_tables, production_rows_all, all_violations, args)


def _finalize(full_tables, headline_tables, production_rows_all, all_violations, args):
    """Build the combined production table, print the gate summary, and (unless
    --no-update) stage the candidate and write the README draft. Shared by the normal
    single-process path and --parent-sweep (after merging every subprocess's
    --output-json output) so there is exactly one code path for this tail,
    regardless of how full_tables/production_rows_all were assembled.
    """
    production_table = _table_production(
        "Production regime -- seq_len [80,128] × num_envs [4096..38400], all algorithms "
        "(plus one boundary-marker row, num_envs=16384/seq_len=16)",
        production_rows_all,
    )
    full_tables = full_tables + [production_table]
    headline_tables = headline_tables + [production_table]

    section = _section(args.gpu, headline_tables)

    print("\n" + section)

    print(f"\n{'=' * 88}\nGATE SUMMARY\n{'=' * 88}")
    if all_violations:
        print(f"MONOTONICITY GATE: FAILED -- {len(all_violations)} violation(s) across the run:")
        for v in all_violations:
            print(f"  - {v}")
    else:
        print("MONOTONICITY GATE: PASSED across the entire run.")
    print("CORRECTNESS GATE (atol=rtol=1e-4 vs sequential reference): PASSED for every config "
          "reached (assert_correctness raises immediately on failure, so reaching this line "
          "means every gate up to here held).")

    if not args.no_update:
        stage_unreleased_md(args.gpu, full_tables)

        print("Running truncation-path headline (same config as README draft) …", flush=True)
        truncation_headline = _bench_truncation_headline()
        readme_draft = render_readme_table_draft(args.gpu, production_rows_all, truncation_headline)
        draft_path = REPO_ROOT / "benchmarks" / "readme_table_draft.md"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(readme_draft)
        print(f"\nREADME summary table drafted to {draft_path}.")

        # README.md is always a STOP-and-report item for a human: this script
        # never writes it. Review the draft above and paste manually. Promote
        # the staged candidate to benchmarks.md separately with
        # --promote --version <tag> when actually cutting a release.
        print("README.md left untouched -- it is never written by this script; review the "
              "draft above and paste manually.")

    if all_violations:
        sys.exit(1)


# Algorithm/variant groups for --parent-sweep: each group runs in its own
# subprocess. Two DISTINCT isolation concerns are both handled by this same
# fresh-subprocess-per-group mechanism, and it is worth keeping them
# conceptually separate:
#
# (1) Algorithm-level isolation (why lambda_returns/discounted_returns/
# eligibility_traces are three separate groups rather than one "returns"
# group, even though they share one bench_returns() call). These used to be
# bundled into one subprocess on the theory that splitting them would only
# triple redundant input construction for no isolation benefit, since the
# crash this guards against was believed to be about TOTAL accumulated
# process state, not which algorithm is running. That theory was tested
# directly and falsified: a full-sweep run hit a wrong-output bug INSIDE the
# bundled "returns" subprocess itself even though that subprocess started
# with a fresh CUDA/Dynamo context -- three compiled objects x 12 CONFIGS
# shapes x (base + truncation + production variants) in ONE subprocess was
# already enough accumulated compile state to reproduce it. One subprocess
# per algorithm (each still internally computes only its own objects via
# bench_returns()'s `selected` argument -- see that function's own comment)
# keeps each subprocess's total compile count in the same range as gae/
# vtrace/retrace's solo subprocesses, which do not reproduce that crash.
#
# (2) Variant-level isolation (why gae/vtrace/retrace/lambda_returns/
# discounted_returns are further split into "_plain" and "_truncation"
# groups). Distinct from (1): a LATER bisection found that TWO SEPARATE
# torch.compile(...) wrappers of the IDENTICAL vectorized function, in one
# process, can silently give wrong (finite, plausible-looking) output from
# one of them once either wrapper has compiled at 2+ distinct shapes --
# confirmed independent of torch._dynamo.reset() usage, confirmed specific to
# discounted-returns among the original four algorithms with this
# two-wrapper structure (see NOTES.md for the full characterization; GAE/
# V-Trace/lambda-returns were tested under the identical pattern and did not
# reproduce it, but WHY they don't is not understood, so isolation is applied
# to all of them rather than relying on that gap). retrace fits the same
# two-wrapper shape even though it has no separate with_truncations variant:
# bench_retrace() and bench_retrace_truncation() each build their own
# torch.compile(vectorized_retrace) wrapper of the SAME underlying function,
# so it is split by --variant here too rather than assumed safe.
# eligibility_traces and prefix_sum have only one torch.compile(...) wrapper
# of their function each and need no variant split.
_PARENT_SWEEP_GROUPS = [
    ("gae_plain",                   ["gae"],                "plain"),
    ("gae_truncation",              ["gae"],                "truncation"),
    ("vtrace_plain",                ["vtrace"],              "plain"),
    ("vtrace_truncation",           ["vtrace"],              "truncation"),
    ("retrace_plain",               ["retrace"],             "plain"),
    ("retrace_truncation",          ["retrace"],             "truncation"),
    ("lambda_returns_plain",        ["lambda_returns"],      "plain"),
    ("lambda_returns_truncation",   ["lambda_returns"],      "truncation"),
    ("discounted_returns_plain",    ["discounted_returns"],  "plain"),
    ("discounted_returns_truncation", ["discounted_returns"], "truncation"),
    ("eligibility_traces",          ["eligibility_traces"],  "all"),
    ("prefix_sum",                  ["prefix_sum"],          "all"),
]


def _run_parent_sweep(selected_algos, args):
    full_tables, headline_tables, production_rows_all, all_violations = [], [], [], []
    script = str(Path(__file__).resolve())

    with tempfile.TemporaryDirectory() as tmpdir:
        for label, group_algos, variant in _PARENT_SWEEP_GROUPS:
            wanted = [a for a in group_algos if a in selected_algos]
            if not wanted:
                continue
            out_path = Path(tmpdir) / f"{label}.json"
            cmd = [
                sys.executable, script,
                "--algos", ",".join(wanted),
                "--variant", variant,
                "--gpu", args.gpu,
                "--no-update",
                "--output-json", str(out_path),
            ]
            print(f"\n{'=' * 88}\nSubprocess for group '{label}' ({','.join(wanted)}, "
                  f"variant={variant}) -- fresh CUDA/Dynamo process\n{'=' * 88}", flush=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\nSubprocess for group '{label}' FAILED (exit {result.returncode}) -- "
                      f"aborting the parent sweep. Nothing partial gets written to "
                      f"benchmarks.md; see the subprocess's own output above for the traceback.",
                      flush=True)
                sys.exit(result.returncode)
            payload = json.loads(out_path.read_text())
            full_tables += payload["full_tables"]
            headline_tables += payload["headline_tables"]
            production_rows_all += payload["production_rows_all"]
            all_violations += payload["all_violations"]

    _finalize(full_tables, headline_tables, production_rows_all, all_violations, args)


if __name__ == "__main__":
    main()
