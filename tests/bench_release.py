"""
Release benchmark — full sweep across all algorithms and configurations.

Compares rl-triton Triton kernels against:
  - torch.compile on equivalent PyTorch reference loops (all algorithms)
  - Pure NumPy CPU loop (GAE, V-Trace, Retrace)
  - NumPy→GPU→NumPy adoption path (GAE, V-Trace, Retrace)

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

# Ensure the tests/ directory is on sys.path so bench_utils is importable.
sys.path.insert(0, str(Path(__file__).parent))
from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu

from rl_triton.ops.gae import compute_gae_triton
from rl_triton.ops.retrace import compute_retrace_triton
from rl_triton.ops.returns import (
    compute_discounted_returns,
    compute_eligibility_traces,
    compute_lambda_returns,
)
from rl_triton.ops.vtrace import compute_vtrace_triton
from rl_triton.ops.vtrace_fused import compute_vtrace_fused


# ---------------------------------------------------------------------------
# Reference implementations (PyTorch loops — ground truth & compiled baselines)
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


def _ref_vtrace(log_pi_t, log_pi_b, values, next_values, rewards, dones, gamma,
                rho_bar=1.0, c_bar=1.0, bootstrap_values=None):
    num_envs, T = rewards.shape
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
    next_t[:, -1]  = next_values[:, -1]
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
    Fully vectorized Retrace(λ) via log-space suffix cumsum — strong compiled baseline.

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
    """CPU GAE backward loop — moves GPU tensors to CPU and runs a plain Python loop."""
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
    out = compute_gae_triton(
        to_gpu(rewards_np), to_gpu(values_np), to_gpu(dones_np),
        gamma=gamma, lambda_=lambda_,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def numpy_vtrace_cpu(log_pi_t, log_pi_b, values, next_values, rewards, dones,
                     gamma, rho_bar=1.0, c_bar=1.0):
    """CPU V-Trace backward loop on CPU tensors."""
    cpu     = lambda t: t.cpu().float()
    lpt, lpb, v, nv, r, d = map(cpu, [log_pi_t, log_pi_b, values, next_values, rewards, dones])
    num_envs, T = r.shape
    rho = torch.clamp(torch.exp(lpt - lpb), max=rho_bar)
    c   = torch.clamp(torch.exp(lpt - lpb), max=c_bar)
    disc = gamma * (1.0 - d)
    deltas = rho * (r + disc * nv - v)
    out   = torch.zeros_like(r)
    carry = torch.zeros(num_envs)
    for t in reversed(range(T)):
        carry      = deltas[:, t] + disc[:, t] * c[:, t] * carry
        out[:, t]  = carry
    targets = out + v
    next_t  = torch.empty_like(targets)
    next_t[:, :-1] = targets[:, 1:]
    next_t[:, -1]  = nv[:, -1]
    advantages = rho * (r + disc * next_t - v)
    return targets, advantages


def numpy_vtrace_np_to_triton(log_pi_t_np, log_pi_b_np, values_np, next_values_np,
                               rewards_np, dones_np, gamma, rho_bar=1.0, c_bar=1.0):
    """NumPy → GPU Triton → NumPy end-to-end adoption path for V-Trace."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.float32)
    args = map(to_gpu, [log_pi_t_np, log_pi_b_np, values_np, next_values_np,
                        rewards_np, dones_np])
    targets, advantages = compute_vtrace_triton(*args, gamma=gamma, rho_bar=rho_bar, c_bar=c_bar)
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
    out  = compute_retrace_triton(apt, apb, q, nqa, acts, r, d,
                                  gamma=gamma, lambda_=lambda_, c_bar=c_bar)
    torch.cuda.synchronize()
    return out.cpu().numpy()


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


# ---------------------------------------------------------------------------
# Input factories
# ---------------------------------------------------------------------------

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
        -torch.rand(num_envs, seq_len, device=d),
        -torch.rand(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),
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
    (64,  512),
    (128, 1024),
    (256, 1024),
    (512, 2048),
    (512, 4096),
]


# ---------------------------------------------------------------------------
# Per-algorithm benchmark runners — return list of result dicts
# ---------------------------------------------------------------------------

def bench_gae():
    compiled = torch.compile(_ref_gae)
    args_warmup = _make_gae(64, 512)
    compiled(*args_warmup, gamma=0.99, lambda_=0.95); torch.cuda.synchronize()

    rows = []
    for num_envs, seq_len in CONFIGS:
        args_gpu = _make_gae(num_envs, seq_len)
        args_np  = tuple(t.cpu().numpy() for t in args_gpu)
        nw, ni   = _n_iter_gpu(seq_len, num_envs)

        triton_ms   = _bench_gpu(compute_gae_triton,     *args_gpu, gamma=0.99, lambda_=0.95, n_warmup=nw, n_iter=ni)
        compiled_ms = _bench_cpu(compiled,               *args_gpu, gamma=0.99, lambda_=0.95)
        e2e_ms      = _bench_cpu(numpy_gae_np_to_triton, *args_np,  gamma=0.99, lambda_=0.95)
        numpy_ms    = _bench_cpu(numpy_gae_cpu,          *args_gpu, gamma=0.99, lambda_=0.95)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "compiled_ms": compiled_ms,
            "e2e_ms": e2e_ms, "numpy_ms": numpy_ms,
            "su_compile": compiled_ms / triton_ms,
            "su_e2e":     numpy_ms    / e2e_ms,
            "su_numpy":   numpy_ms    / triton_ms,
        })
    return rows


def bench_vtrace():
    compiled = torch.compile(_ref_vtrace)
    args = _make_vtrace(64, 512)
    compiled(*args, gamma=0.99); torch.cuda.synchronize()

    rows = []
    for num_envs, seq_len in CONFIGS:
        args_gpu = _make_vtrace(num_envs, seq_len)
        args_np  = tuple(t.cpu().numpy() for t in args_gpu)
        nw, ni   = _n_iter_gpu(seq_len, num_envs)

        triton_ms   = _bench_gpu(compute_vtrace_fused,          *args_gpu, gamma=0.99, n_warmup=nw, n_iter=ni)
        compiled_ms = _bench_cpu(compiled,                      *args_gpu, gamma=0.99)
        e2e_ms      = _bench_cpu(numpy_vtrace_np_to_triton,     *args_np,  gamma=0.99)
        numpy_ms    = _bench_cpu(numpy_vtrace_cpu,              *args_gpu, gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "compiled_ms": compiled_ms,
            "e2e_ms": e2e_ms, "numpy_ms": numpy_ms,
            "su_compile": compiled_ms / triton_ms,
            "su_e2e":     numpy_ms    / e2e_ms,
            "su_numpy":   numpy_ms    / triton_ms,
        })
    return rows


def bench_retrace():
    compiled_vec  = torch.compile(_vec_retrace)
    compiled_loop = torch.compile(_ref_retrace)
    args = _make_retrace(64, 512)
    compiled_vec(*args, gamma=0.99);  torch.cuda.synchronize()
    compiled_loop(*args, gamma=0.99); torch.cuda.synchronize()

    rows = []
    for num_envs, seq_len in CONFIGS:
        args_gpu = _make_retrace(num_envs, seq_len)
        # _make_retrace order: rewards, dones, values, next_q_all, q_values, actions, apt, apb
        # numpy_retrace_cpu order: rewards, dones, q_values, next_q_all, actions, apt, apb
        # (index 2 is "values" used as q_values in _ref_retrace; index 4 is a duplicate — skip it)
        rn, dn, qn, nqn, _qn2, an, aptn, apbn = tuple(t.cpu().numpy() for t in args_gpu)
        nw, ni = _n_iter_gpu(seq_len, num_envs)

        triton_ms   = _bench_gpu(compute_retrace_triton,       *args_gpu, gamma=0.99, n_warmup=nw, n_iter=ni)
        vec_ms      = _bench_gpu(compiled_vec,                  *args_gpu, gamma=0.99, n_warmup=nw, n_iter=ni)
        compiled_ms = _bench_cpu(compiled_loop,                 *args_gpu, gamma=0.99)
        e2e_ms      = _bench_cpu(numpy_retrace_np_to_triton,    rn, dn, qn, nqn, an, aptn, apbn, gamma=0.99)
        numpy_ms    = _bench_cpu(numpy_retrace_cpu,             rn, dn, qn, nqn, an, aptn, apbn, gamma=0.99)

        rows.append({
            "num_envs": num_envs, "seq_len": seq_len,
            "triton_ms": triton_ms, "compiled_ms": compiled_ms,
            "vec_ms": vec_ms, "e2e_ms": e2e_ms, "numpy_ms": numpy_ms,
            "su_compile": compiled_ms / triton_ms,
            "su_vec":     vec_ms      / triton_ms,
            "su_e2e":     numpy_ms    / e2e_ms,
            "su_numpy":   numpy_ms    / triton_ms,
        })
    return rows


def bench_returns():
    c_lambda = torch.compile(_ref_lambda)
    c_disc   = torch.compile(_ref_disc)
    c_traces = torch.compile(_ref_traces)
    r, nv, d = _make_returns(64, 512)
    c_lambda(r, nv, d, gamma=0.99, lambda_=0.95); torch.cuda.synchronize()
    c_disc(r, d, gamma=0.99);                     torch.cuda.synchronize()
    c_traces(r, d, gamma=0.99, lambda_=0.95);     torch.cuda.synchronize()

    rows_lambda, rows_disc, rows_traces = [], [], []
    for num_envs, seq_len in CONFIGS:
        rewards, next_values, dones = _make_returns(num_envs, seq_len)
        nw, ni = _n_iter_gpu(seq_len, num_envs)

        lam_ms  = _bench_gpu(compute_lambda_returns,    rewards, next_values, dones,
                             gamma=0.99, lambda_=0.95, n_warmup=nw, n_iter=ni)
        disc_ms = _bench_gpu(compute_discounted_returns, rewards, dones,
                             gamma=0.99, n_warmup=nw, n_iter=ni)
        trc_ms  = _bench_gpu(compute_eligibility_traces, rewards, dones,
                             gamma=0.99, lambda_=0.9, n_warmup=nw, n_iter=ni)
        clam_ms  = _bench_cpu(c_lambda, rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        cdisc_ms = _bench_cpu(c_disc,   rewards, dones, gamma=0.99)
        ctrc_ms  = _bench_cpu(c_traces, rewards, dones, gamma=0.99, lambda_=0.9)

        base = {"num_envs": num_envs, "seq_len": seq_len}
        rows_lambda.append({**base, "triton_ms": lam_ms,  "compiled_ms": clam_ms,  "su_compile": clam_ms  / lam_ms})
        rows_disc.append(  {**base, "triton_ms": disc_ms, "compiled_ms": cdisc_ms, "su_compile": cdisc_ms / disc_ms})
        rows_traces.append({**base, "triton_ms": trc_ms,  "compiled_ms": ctrc_ms,  "su_compile": ctrc_ms  / trc_ms})
    return rows_lambda, rows_disc, rows_traces


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def _fmt_row(r, include_numpy=False, include_vec=False):
    base = (f"| {r['num_envs']:>8} | {r['seq_len']:>7} "
            f"| {r['triton_ms']:>10.3f} | {r['compiled_ms']:>11.3f} "
            f"| {r['su_compile']:>8.1f}x |")
    if include_numpy:
        base += (f" {r['e2e_ms']:>14.3f} | {r['numpy_ms']:>10.3f} "
                 f"| {r['su_e2e']:>7.1f}x |")
    if include_vec:
        base += (f" {r['vec_ms']:>13.3f} | {r['e2e_ms']:>17.3f} | {r['numpy_ms']:>10.3f} "
                 f"| {r['su_vec']:>6.1f}x | {r['su_e2e']:>7.1f}x | {r['su_numpy']:>8.1f}x |")
    return base


def _table_numpy(title, rows):
    header = (
        f"#### {title}\n\n"
        "| num_envs | seq_len | triton (ms) | pt.compile (ms) | vs compile |"
        " np→triton→np (ms) | numpy cpu (ms) | np→tri→np vs numpy |\n"
        "|:--------:|:-------:|:-----------:|:---------------:|:----------:|"
        ":------------------:|:--------------:|:------------------:|"
    )
    body = "\n".join(_fmt_row(r, include_numpy=True) for r in rows)
    return header + "\n" + body


def _table_simple(title, rows):
    header = (
        f"#### {title}\n\n"
        "| num_envs | seq_len | triton (ms) | pt.compile (ms) | vs compile |\n"
        "|:--------:|:-------:|:-----------:|:---------------:|:----------:|"
    )
    body = "\n".join(_fmt_row(r, include_numpy=False) for r in rows)
    return header + "\n" + body


def _table_retrace(title, rows):
    header = (
        f"#### {title}\n\n"
        "| num_envs | seq_len | triton (ms) | pt.compile (ms) | vs compile |"
        " compile(vec) (ms) | np→triton→np (ms) | numpy cpu (ms) | vs vec | np→tri→np vs numpy | vs numpy |\n"
        "|:--------:|:-------:|:-----------:|:---------------:|:----------:|"
        ":------------------:|:-----------------:|:--------------:|:------:|:-------------------:|:--------:|"
    )
    body = "\n".join(_fmt_row(r, include_vec=True) for r in rows)
    return header + "\n" + body


def _section(gpu_label: str, tables: list[str]) -> str:
    date = datetime.date.today().isoformat()
    gpu  = gpu_label or _detect_gpu()
    header = (
        f"<!-- BENCH_START -->\n"
        f"## Performance\n\n"
        f"*Measured on {gpu} · {date} · "
        f"[`triton`](https://github.com/openai/triton) kernels vs `torch.compile` and NumPy CPU.*\n\n"
        f"**triton**: CUDA events — pure kernel time.  "
        f"**pt.compile**: wall-clock (dispatches one CUDA op per timestep from Python).  "
        f"**np→triton→np**: NumPy → GPU → NumPy adoption path.  "
        f"**numpy cpu**: pure NumPy backward loop on CPU.\n"
    )
    return header + "\n\n".join(tables) + "\n<!-- BENCH_END -->"


def _detect_gpu() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown GPU"


# ---------------------------------------------------------------------------
# README updater
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
README    = REPO_ROOT / "README.md"

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-update", action="store_true",
                        help="Print results but do not modify README.md")
    parser.add_argument("--gpu", default="", metavar="LABEL",
                        help="GPU label to embed in the table header (default: auto-detect)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — aborting.")
        sys.exit(1)

    print("Running GAE benchmark …")
    gae_rows = bench_gae()
    print("Running V-Trace benchmark …")
    vtrace_rows = bench_vtrace()
    print("Running Retrace(λ) benchmark …")
    retrace_rows = bench_retrace()
    print("Running returns / eligibility-traces benchmark …")
    lambda_rows, disc_rows, traces_rows = bench_returns()

    tables = [
        _table_numpy("GAE (`compute_gae_triton`)", gae_rows),
        _table_numpy("V-Trace (`compute_vtrace_triton`)", vtrace_rows),
        _table_retrace("Retrace(λ) (`compute_retrace_triton`)", retrace_rows),
        _table_simple("λ-returns (`compute_lambda_returns`)", lambda_rows),
        _table_simple("Discounted returns (`compute_discounted_returns`)", disc_rows),
        _table_simple("Eligibility traces (`compute_eligibility_traces`)", traces_rows),
    ]

    section = _section(args.gpu, tables)

    print("\n" + section)

    if not args.no_update:
        update_readme(section)


if __name__ == "__main__":
    main()
