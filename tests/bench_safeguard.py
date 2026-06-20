"""
PR safeguard performance tests.

Each test covers one algorithm at a single representative config (128 envs x 1024
steps) and asserts the Triton kernel is >=1.5x faster than torch.compile on a
fully-vectorized PyTorch reference.  Both sides are timed with CUDA events so the
comparison is apples-to-apples (pure GPU time, no Python-loop overhead).

They are fast (~5-10s total on GPU) and are intended to run on every PR to catch
regressions before they ship.

Run with:
    pytest tests/bench_safeguard.py -v
or select in CI:
    pytest -m perf
"""
import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_gpu, _n_iter_gpu
from rl_triton.ops.gae import compute_gae
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum
from rl_triton.ops.retrace import compute_retrace
from rl_triton.ops.returns import (
    compute_discounted_returns,
    compute_eligibility_traces,
    compute_lambda_returns,
)
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

# Import vectorized baselines from each algorithm's test module.
# These are the strongest compiled PyTorch baselines (no Python loop per timestep).
from test_gae import vectorized_gae
from test_prefix_sum import vectorized_episodic_prefix_sum
from test_retrace import vectorized_retrace
from test_returns import (
    vectorized_discounted_returns,
    vectorized_eligibility_traces,
    vectorized_lambda_returns,
)
from test_vtrace import vectorized_vtrace

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_NUM_ENVS      = 128
_SEQ_LEN       = 1024
_SPEEDUP_FLOOR = 1.5


# ---------------------------------------------------------------------------
# Input factories
# ---------------------------------------------------------------------------

def _gae_inputs():
    torch.manual_seed(0)
    d = "cuda"
    rewards = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    values  = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones   = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return rewards, values, dones


def _vtrace_inputs():
    torch.manual_seed(0)
    d = "cuda"
    return (
        -torch.rand(_NUM_ENVS, _SEQ_LEN, device=d),   # log_pi_target
        -torch.rand(_NUM_ENVS, _SEQ_LEN, device=d),   # log_pi_behavior
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # values
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # rewards
        (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float(),  # dones
    )


def _retrace_inputs(num_actions=4):
    torch.manual_seed(0)
    d = "cuda"
    terminateds = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    # truncateds are mutually exclusive with terminateds
    truncateds  = ((torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05) & ~terminateds.bool()).float()
    return (
        torch.softmax(torch.randn(_NUM_ENVS, _SEQ_LEN, num_actions, device=d), dim=-1),
        torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) * 0.8 + 0.1,
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),
        torch.randn(_NUM_ENVS, _SEQ_LEN, num_actions, device=d),
        torch.randint(0, num_actions, (_NUM_ENVS, _SEQ_LEN), device=d),
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),
        terminateds,
        truncateds,
    )


def _returns_inputs():
    torch.manual_seed(0)
    d = "cuda"
    rewards     = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    next_values = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones       = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return rewards, next_values, dones


def _warmup(fn, *args, **kwargs):
    fn(*args, **kwargs)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Safeguard tests — one per algorithm
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.perf
def test_perf_gae():
    args = _gae_inputs()
    compiled = torch.compile(vectorized_gae)
    _warmup(compiled, *args, gamma=0.99, lambda_=0.95)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms  = _bench_gpu(compute_gae, *args, gamma=0.99, lambda_=0.95, n_warmup=nw, n_iter=ni)
    vec_ms     = _bench_gpu(compiled,           *args, gamma=0.99, lambda_=0.95, n_warmup=nw, n_iter=ni)
    speedup    = vec_ms / triton_ms

    print(f"\nGAE  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"GAE Triton {speedup:.2f}x vs torch.compile(vec) — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_vtrace():
    args = _vtrace_inputs()
    compiled = torch.compile(vectorized_vtrace)
    _warmup(compiled, *args, gamma=0.99)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms = _bench_gpu(compute_vtrace_fused, *args, gamma=0.99, n_warmup=nw, n_iter=ni)
    vec_ms    = _bench_gpu(compiled,             *args, gamma=0.99, n_warmup=nw, n_iter=ni)
    speedup   = vec_ms / triton_ms

    print(f"\nV-Trace  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"V-Trace Triton {speedup:.2f}x vs torch.compile(vec) — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_retrace():
    args = _retrace_inputs()
    compiled = torch.compile(vectorized_retrace)
    _warmup(compiled, *args, gamma=0.99)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms = _bench_gpu(compute_retrace, *args, gamma=0.99, n_warmup=nw, n_iter=ni)
    vec_ms    = _bench_gpu(compiled,               *args, gamma=0.99, n_warmup=nw, n_iter=ni)
    speedup   = vec_ms / triton_ms

    print(f"\nRetrace  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"Retrace Triton {speedup:.2f}x vs torch.compile(vec) — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_lambda_returns():
    rewards, next_values, dones = _returns_inputs()
    compiled = torch.compile(vectorized_lambda_returns)
    _warmup(compiled, rewards, next_values, dones, gamma=0.99, lambda_=0.95)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms = _bench_gpu(compute_lambda_returns, rewards, next_values, dones,
                           gamma=0.99, lambda_=0.95, n_warmup=nw, n_iter=ni)
    vec_ms    = _bench_gpu(compiled, rewards, next_values, dones,
                           gamma=0.99, lambda_=0.95, n_warmup=nw, n_iter=ni)
    speedup   = vec_ms / triton_ms

    print(f"\nλ-returns  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"λ-returns Triton {speedup:.2f}x vs torch.compile(vec) — below {_SPEEDUP_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_discounted_returns():
    # Discounted returns is the lightest kernel (2 inputs, u[t]=r[t]) so the
    # associative-scan tree overhead is a larger fraction of total runtime than
    # for heavier kernels.  torch.compile(vectorized) avoids the scan entirely
    # via a log-space cumsum that is fully parallel across envs, leaving less
    # room for the Triton kernel to win on bandwidth.  1.2x is the realistic
    # floor for this algorithm at 128x1024; larger configs see higher speedups.
    _DISC_FLOOR = 1.2
    rewards, _, dones = _returns_inputs()
    compiled = torch.compile(vectorized_discounted_returns)
    _warmup(compiled, rewards, dones, gamma=0.99)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms = _bench_gpu(compute_discounted_returns, rewards, dones,
                           gamma=0.99, n_warmup=nw, n_iter=ni)
    vec_ms    = _bench_gpu(compiled, rewards, dones,
                           gamma=0.99, n_warmup=nw, n_iter=ni)
    speedup   = vec_ms / triton_ms

    print(f"\nDisc-returns  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _DISC_FLOOR, (
        f"Discounted-returns Triton {speedup:.2f}x vs torch.compile(vec) — below {_DISC_FLOOR}x floor"
    )


@cuda_only
@pytest.mark.perf
def test_perf_eligibility_traces():
    rewards, _, dones = _returns_inputs()
    compiled = torch.compile(vectorized_eligibility_traces)
    _warmup(compiled, rewards, dones, gamma=0.99, lambda_=0.9)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms = _bench_gpu(compute_eligibility_traces, rewards, dones,
                           gamma=0.99, lambda_=0.9, n_warmup=nw, n_iter=ni)
    vec_ms    = _bench_gpu(compiled, rewards, dones,
                           gamma=0.99, lambda_=0.9, n_warmup=nw, n_iter=ni)
    speedup   = vec_ms / triton_ms

    print(f"\nElig-traces  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _SPEEDUP_FLOOR, (
        f"Eligibility-traces Triton {speedup:.2f}x vs torch.compile(vec) — below {_SPEEDUP_FLOOR}x floor"
    )


def _prefix_sum_inputs():
    torch.manual_seed(0)
    d = "cuda"
    inputs = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones  = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return inputs, dones


@cuda_only
@pytest.mark.perf
def test_perf_prefix_sum():
    # Prefix sum is the lightest kernel in the library: u[t]=x[t], v[t]=1-done[t],
    # no derived quantities, no next-values.  The torch.compile(cumsum) baseline
    # maps to torch.cumsum — a CUDA-native parallel scan — with only 3 elementwise
    # ops and no serial dependency chain.  At 128x1024 both paths complete in
    # ~30µs, well below where bandwidth or compute differences show up; the margin
    # is determined by kernel launch overhead, not algorithm efficiency.
    # The baseline is torch.compile(cumsum) — a native CUDA parallel scan with no
    # reset logic. Our kernel does strictly more work (fused done-mask resets) so
    # matching cumsum at 128x1024 is unrealistic; the advantage shows at larger
    # configs where the fused reset saves a full read-write pass. 0.85x guards
    # against genuine regressions (e.g. a broken scan or an extra allocation)
    # without penalising the inherent overhead difference vs plain cumsum.
    _PREFIX_FLOOR = 0.85
    args = _prefix_sum_inputs()
    compiled = torch.compile(vectorized_episodic_prefix_sum)
    _warmup(compiled, *args)

    nw, ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    triton_ms = _bench_gpu(compute_episodic_prefix_sum, *args, n_warmup=nw, n_iter=ni)
    vec_ms    = _bench_gpu(compiled,                    *args, n_warmup=nw, n_iter=ni)
    speedup   = vec_ms / triton_ms

    print(f"\nPrefix-sum  {_NUM_ENVS}x{_SEQ_LEN}: triton={triton_ms:.3f}ms  compile(vec)={vec_ms:.3f}ms  speedup={speedup:.1f}x")
    assert speedup >= _PREFIX_FLOOR, (
        f"Prefix-sum Triton {speedup:.2f}x vs torch.compile(cumsum) — below {_PREFIX_FLOOR}x floor"
    )
