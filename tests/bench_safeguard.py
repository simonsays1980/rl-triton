"""
PR safeguard performance tests.

Each test covers one algorithm at a single representative config (128 envs x 1024
steps) and asserts the Triton kernel is >=1.5x faster than torch.compile on the
same reference loop.  They are fast (~5-10s total on GPU) and are intended to run
on every PR to catch regressions before they ship.

Run with:
    pytest tests/bench_safeguard.py -v
or select in CI:
    pytest -m perf
"""
import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu
from rl_triton.ops.gae import compute_gae_triton
from rl_triton.ops.retrace import compute_retrace_triton
from rl_triton.ops.returns import (
    compute_discounted_returns,
    compute_eligibility_traces,
    compute_lambda_returns,
)
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_NUM_ENVS = 128
_SEQ_LEN  = 1024
_SPEEDUP_FLOOR = 1.5


# ---------------------------------------------------------------------------
# Minimal reference implementations (used only to build the compiled baseline)
# ---------------------------------------------------------------------------

def _ref_gae(deltas, decays, bootstrap=None):
    T     = deltas.shape[1]
    out   = torch.zeros_like(deltas)
    carry = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    if bootstrap is not None:
        carry = bootstrap.clone()
    for t in reversed(range(T)):
        carry     = deltas[:, t] + decays[:, t] * carry
        out[:, t] = carry
    return out


def _ref_vtrace(log_pi_t, log_pi_b, values, next_values, rewards, dones, gamma,
                rho_bar=1.0, c_bar=1.0, bootstrap_values=None):
    num_envs, T = rewards.shape
    rho = torch.clamp(torch.exp(log_pi_t - log_pi_b), max=rho_bar)
    c   = torch.clamp(torch.exp(log_pi_t - log_pi_b), max=c_bar)
    deltas = rho * (rewards + gamma * next_values * (1.0 - dones) - values)
    decays = gamma * c * (1.0 - dones)
    value_deltas = torch.zeros_like(rewards)
    carry = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry = deltas[:, t] + decays[:, t] * carry
        value_deltas[:, t] = carry
    targets = value_deltas + values
    next_t = torch.empty_like(targets)
    next_t[:, :-1] = targets[:, 1:]
    next_t[:, -1]  = next_values[:, -1]
    advantages = rho * (rewards + gamma * next_t * (1.0 - dones) - values)
    return targets, advantages


def _ref_retrace(rewards, dones, values, next_q_all, q_values, actions,
                 action_probs_target, action_probs_behavior, gamma, lambda_=1.0,
                 c_bar=1.0, bootstrap_values=None):
    num_envs, T = rewards.shape
    pi_a    = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c       = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    exp_nq  = (action_probs_target * next_q_all).sum(dim=-1)
    u       = rewards + gamma * exp_nq * (1.0 - dones) - q_values
    c_next  = torch.empty_like(c)
    c_next[:, :-1] = c[:, 1:]
    c_next[:, -1]  = 0.0
    v = gamma * c_next * (1.0 - dones)
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry     = u[:, t] + v[:, t] * carry
        out[:, t] = carry
    return out + q_values


def _ref_lambda(rewards, next_values, dones, gamma, lambda_, bootstrap=None):
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    if bootstrap is not None:
        carry = bootstrap.clone()
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
    carry = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    if bootstrap is not None:
        carry = bootstrap.clone()
    for t in reversed(range(T)):
        carry = rewards[:, t] + gamma * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


def _ref_traces(features, dones, gamma, lambda_, seed=None):
    T     = features.shape[1]
    out   = torch.zeros_like(features)
    carry = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
    if seed is not None:
        carry = seed.clone()
    for t in range(T):
        carry     = features[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


# ---------------------------------------------------------------------------
# Helpers to build inputs
# ---------------------------------------------------------------------------

def _gae_inputs():
    torch.manual_seed(0)
    d = "cuda"
    deltas = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    decays = torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) * 0.99
    return deltas, decays


def _vtrace_inputs():
    torch.manual_seed(0)
    d = "cuda"
    return (
        -torch.rand(_NUM_ENVS, _SEQ_LEN, device=d),   # log_pi_target
        -torch.rand(_NUM_ENVS, _SEQ_LEN, device=d),   # log_pi_behavior
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # values
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # next_values
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # rewards
        (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float(),  # dones
    )


def _retrace_inputs(num_actions=4):
    torch.manual_seed(0)
    d = "cuda"
    return (
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),           # rewards
        (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float(),  # dones
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),           # values (unused by retrace but kept for ref)
        torch.randn(_NUM_ENVS, _SEQ_LEN, num_actions, device=d),     # next_q_all
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),           # q_values
        torch.randint(0, num_actions, (_NUM_ENVS, _SEQ_LEN), device=d),  # actions
        torch.softmax(torch.randn(_NUM_ENVS, _SEQ_LEN, num_actions, device=d), dim=-1),  # probs_target
        torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) * 0.8 + 0.1,      # probs_behavior
    )


def _returns_inputs():
    torch.manual_seed(0)
    d = "cuda"
    rewards     = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    next_values = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones       = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return rewards, next_values, dones


def _warmup_compile(fn, *args, **kwargs):
    fn(*args, **kwargs)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Safeguard tests — one per algorithm
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.perf
def test_perf_gae():
    deltas, decays = _gae_inputs()
    compiled = torch.compile(_ref_gae)
    _warmup_compile(compiled, deltas, decays)

    n_warmup, n_iter = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms   = _bench_gpu(compute_gae_triton, deltas, decays, n_warmup=n_warmup, n_iter=n_iter)
    compiled_ms = _bench_cpu(compiled, deltas, decays)
    speedup = compiled_ms / triton_ms

    print(f"\nGAE  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile={compiled_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"GAE Triton {speedup:.2f}x vs torch.compile — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_vtrace():
    args = _vtrace_inputs()
    compiled = torch.compile(_ref_vtrace)
    _warmup_compile(compiled, *args, gamma=0.99)

    n_warmup, n_iter = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms   = _bench_gpu(compute_vtrace_fused, *args, gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
    compiled_ms = _bench_cpu(compiled, *args, gamma=0.99)
    speedup = compiled_ms / triton_ms

    print(f"\nV-Trace  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile={compiled_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"V-Trace Triton {speedup:.2f}x vs torch.compile — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_retrace():
    args = _retrace_inputs()
    compiled = torch.compile(_ref_retrace)
    _warmup_compile(compiled, *args, gamma=0.99)

    n_warmup, n_iter = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms   = _bench_gpu(compute_retrace_triton, *args, gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
    compiled_ms = _bench_cpu(compiled, *args, gamma=0.99)
    speedup = compiled_ms / triton_ms

    print(f"\nRetrace  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile={compiled_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"Retrace Triton {speedup:.2f}x vs torch.compile — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_lambda_returns():
    rewards, next_values, dones = _returns_inputs()
    compiled = torch.compile(_ref_lambda)
    _warmup_compile(compiled, rewards, next_values, dones, gamma=0.99, lambda_=0.95)

    n_warmup, n_iter = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms   = _bench_gpu(compute_lambda_returns, rewards, next_values, dones,
                             gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
    compiled_ms = _bench_cpu(compiled, rewards, next_values, dones, gamma=0.99, lambda_=0.95)
    speedup = compiled_ms / triton_ms

    print(f"\nλ-returns  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile={compiled_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"λ-returns Triton {speedup:.2f}x vs torch.compile — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_discounted_returns():
    rewards, _, dones = _returns_inputs()
    compiled = torch.compile(_ref_disc)
    _warmup_compile(compiled, rewards, dones, gamma=0.99)

    n_warmup, n_iter = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms   = _bench_gpu(compute_discounted_returns, rewards, dones,
                             gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
    compiled_ms = _bench_cpu(compiled, rewards, dones, gamma=0.99)
    speedup = compiled_ms / triton_ms

    print(f"\nDisc-returns  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile={compiled_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"Discounted-returns Triton {speedup:.2f}x vs torch.compile — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_eligibility_traces():
    rewards, _, dones = _returns_inputs()
    compiled = torch.compile(_ref_traces)
    _warmup_compile(compiled, rewards, dones, gamma=0.99, lambda_=0.9)

    n_warmup, n_iter = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms   = _bench_gpu(compute_eligibility_traces, rewards, dones,
                             gamma=0.99, lambda_=0.9, n_warmup=n_warmup, n_iter=n_iter)
    compiled_ms = _bench_cpu(compiled, rewards, dones, gamma=0.99, lambda_=0.9)
    speedup = compiled_ms / triton_ms

    print(f"\nElig-traces  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile={compiled_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"Eligibility-traces Triton {speedup:.2f}x vs torch.compile — below {_SPEEDUP_FLOOR}x floor"
    )
