import time

import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops.returns import compute_discounted_returns, compute_lambda_returns

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def reference_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch TD(λ) backward scan — ground truth for correctness tests."""
    T       = rewards.shape[1]
    out     = torch.zeros_like(rewards)
    carry   = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        not_done  = 1.0 - dones[:, t]
        carry     = (rewards[:, t]
                     + gamma * (1.0 - lambda_) * not_done * next_values[:, t]
                     + gamma * lambda_ * not_done * carry)
        out[:, t] = carry
    return out


def reference_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch discounted-return backward scan — ground truth."""
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry     = rewards[:, t] + gamma * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


def _make_inputs(num_envs, seq_len, device="cuda", seed=0):
    torch.manual_seed(seed)
    rewards     = torch.randn(num_envs, seq_len, device=device)
    next_values = torch.randn(num_envs, seq_len, device=device)
    dones       = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return rewards, next_values, dones


# ---------------------------------------------------------------------------
# Ground-truth value tests — compute_lambda_returns
# ---------------------------------------------------------------------------

@cuda_only
def test_lambda_returns_known_values_lambda0():
    # lambda=0: G[t] = r[t] + gamma*(1-d[t])*V(s_{t+1})  (one-step TD target).
    # No carry from the future at all.
    # r=[1,2], V'=[3,4], gamma=1, no dones.
    # G[0] = 1 + 1*3 = 4,  G[1] = 2 + 1*4 = 6
    rewards     = torch.tensor([[1.0, 2.0]], device="cuda")
    next_values = torch.tensor([[3.0, 4.0]], device="cuda")
    dones       = torch.zeros(1, 2, device="cuda")
    out = compute_lambda_returns(rewards, next_values, dones, gamma=1.0, lambda_=0.0)
    torch.testing.assert_close(out, torch.tensor([[4.0, 6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_lambda_returns_known_values_lambda1():
    # lambda=1: reduces to discounted returns — V(s_{t+1}) drops out of u.
    # G[1] = 2 + 1*0 = 2  (bootstrap=0, no next G)
    # G[0] = 1 + 1*2 = 3
    rewards     = torch.tensor([[1.0, 2.0]], device="cuda")
    next_values = torch.tensor([[99.0, 99.0]], device="cuda")   # irrelevant at lambda=1
    dones       = torch.zeros(1, 2, device="cuda")
    out = compute_lambda_returns(rewards, next_values, dones, gamma=1.0, lambda_=1.0)
    torch.testing.assert_close(out, torch.tensor([[3.0, 2.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_lambda_returns_known_values_intermediate():
    # lambda=0.5, gamma=1, seq_len=2, no dones, bootstrap=0.
    # u[t] = r[t] + 1*0.5*(1)*V'[t],  v[t] = 1*0.5*1 = 0.5
    # r=[1,2], V'=[3,4]
    # u[0] = 1 + 0.5*3 = 2.5,  u[1] = 2 + 0.5*4 = 4.0
    # G[1] = u[1] + 0.5*0 = 4.0
    # G[0] = u[0] + 0.5*4.0 = 2.5 + 2.0 = 4.5
    rewards     = torch.tensor([[1.0, 2.0]], device="cuda")
    next_values = torch.tensor([[3.0, 4.0]], device="cuda")
    dones       = torch.zeros(1, 2, device="cuda")
    out = compute_lambda_returns(rewards, next_values, dones, gamma=1.0, lambda_=0.5)
    torch.testing.assert_close(out, torch.tensor([[4.5, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_lambda_returns_known_values_done():
    # done[0]=1 cuts the trace: not_done[0]=0 -> u[0]=r[0], v[0]=0.
    # r=[1,2], V'=[3,4], lambda=0.5, gamma=1.
    # u[1]=2+0.5*4=4, v[1]=0.5;  u[0]=1+0=1, v[0]=0
    # G[1] = 4 + 0.5*0 = 4
    # G[0] = 1 + 0*4   = 1
    rewards     = torch.tensor([[1.0, 2.0]], device="cuda")
    next_values = torch.tensor([[3.0, 4.0]], device="cuda")
    dones       = torch.tensor([[1.0, 0.0]], device="cuda")
    out = compute_lambda_returns(rewards, next_values, dones, gamma=1.0, lambda_=0.5)
    torch.testing.assert_close(out, torch.tensor([[1.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_lambda_returns_known_values_bootstrap():
    # lambda=1, gamma=1, bootstrap=5.0: reduces to discounted returns with G[T]=5.
    # G[1] = 2 + 1*5 = 7
    # G[0] = 1 + 1*7 = 8
    rewards     = torch.tensor([[1.0, 2.0]], device="cuda")
    next_values = torch.tensor([[99.0, 99.0]], device="cuda")
    dones       = torch.zeros(1, 2, device="cuda")
    bootstrap   = torch.tensor([5.0], device="cuda")
    out = compute_lambda_returns(rewards, next_values, dones, gamma=1.0, lambda_=1.0,
                                  bootstrap_values=bootstrap)
    torch.testing.assert_close(out, torch.tensor([[8.0, 7.0]], device="cuda"), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Correctness vs reference — compute_lambda_returns
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len,lambda_", [
    (1,   1,    0.0),
    (1,   7,    1.0),
    (4,   128,  0.5),
    (32,  333,  0.95),
    (128, 1024, 0.9),
])
def test_lambda_returns_correctness(num_envs, seq_len, lambda_):
    rewards, next_values, dones = _make_inputs(num_envs, seq_len, seed=42)
    expected = reference_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=lambda_)
    actual   = compute_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=lambda_)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_lambda_returns_correctness_bootstrap():
    rewards, next_values, dones = _make_inputs(32, 512, seed=3)
    bootstrap = torch.rand(32, device="cuda")
    expected  = reference_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=0.95,
                                          bootstrap_values=bootstrap)
    actual    = compute_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=0.95,
                                        bootstrap_values=bootstrap)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_lambda_returns_lambda1_matches_discounted_returns():
    """lambda=1 must produce identical output to compute_discounted_returns."""
    rewards, next_values, dones = _make_inputs(32, 512, seed=4)
    bootstrap = torch.rand(32, device="cuda")
    lambda1 = compute_lambda_returns(rewards, next_values, dones, gamma=0.99, lambda_=1.0,
                                      bootstrap_values=bootstrap)
    disc    = compute_discounted_returns(rewards, dones, gamma=0.99,
                                         bootstrap_values=bootstrap)
    torch.testing.assert_close(lambda1, disc, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Ground-truth value tests — compute_discounted_returns
# ---------------------------------------------------------------------------

@cuda_only
def test_discounted_returns_known_values():
    # G[1] = 2 + 0.9*0 = 2,  G[0] = 1 + 0.9*2 = 2.8
    rewards = torch.tensor([[1.0, 2.0]], device="cuda")
    dones   = torch.zeros(1, 2, device="cuda")
    out = compute_discounted_returns(rewards, dones, gamma=0.9)
    torch.testing.assert_close(out, torch.tensor([[2.8, 2.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_discounted_returns_known_values_done():
    # done[0]=1: G[0] = r[0] + gamma*(1-1)*G[1] = 1.0
    rewards = torch.tensor([[1.0, 2.0]], device="cuda")
    dones   = torch.tensor([[1.0, 0.0]], device="cuda")
    out = compute_discounted_returns(rewards, dones, gamma=0.9)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1, 1), (1, 7), (4, 128), (32, 512), (128, 1024),
])
def test_discounted_returns_correctness(num_envs, seq_len):
    rewards, _, dones = _make_inputs(num_envs, seq_len, seed=10)
    expected = reference_discounted_returns(rewards, dones, gamma=0.99)
    actual   = compute_discounted_returns(rewards, dones, gamma=0.99)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_gpu(fn, *args, n_warmup: int = 25, n_iter: int = 100, **kwargs) -> float:
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def _bench_cpu(fn, *args, n_warmup: int = 3, target_s: float = 0.5, **kwargs) -> float:
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    elapsed = 0.0
    n = 0
    t0 = time.perf_counter()
    while elapsed < target_s or n < 5:
        fn(*args, **kwargs)
        n += 1
        elapsed = time.perf_counter() - t0
    return elapsed / n * 1000.0


def _n_iter_gpu(seq_len: int, num_envs: int) -> tuple[int, int]:
    elements = seq_len * num_envs
    n_warmup = max(5,  min(25,  5_000_000 // elements))
    n_iter   = max(20, min(200, 20_000_000 // elements))
    return n_warmup, n_iter


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

BENCH_CONFIGS = [
    (64,  512),
    (128, 1024),
    (256, 1024),
    (512, 2048),
    (512, 4096),
]


@cuda_only
@pytest.mark.slow
def test_returns_performance():
    """
    Benchmark compute_lambda_returns and compute_discounted_returns against
    torch.compile on the reference loops.

    Both Triton kernels use CUDA events (pure kernel time).
    pt.compile uses wall-clock — it dispatches one CUDA op per timestep from
    Python; CUDA events would miss that CPU stall.

    Assertion: both kernels must be >=1.5x faster than torch.compile.
    """
    compiled_lambda = torch.compile(reference_lambda_returns)
    compiled_disc   = torch.compile(reference_discounted_returns)

    _r, _nv, _d = _make_inputs(64, 512)
    compiled_lambda(_r, _nv, _d, gamma=0.99, lambda_=0.95)
    compiled_disc(_r, _d, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'lambda(tri)':>12} {'disc(tri)':>10} {'pt.compile λ':>14} {'pt.compile G':>14} "
        f"{'vs λ compile':>14} {'vs G compile':>14}"
    )
    print(header)
    print("-" * len(header))

    all_speedups = []

    for num_envs, seq_len in BENCH_CONFIGS:
        rewards, next_values, dones = _make_inputs(num_envs, seq_len)
        n_warmup, n_iter = _n_iter_gpu(seq_len, num_envs)

        lambda_ms       = _bench_gpu(compute_lambda_returns,  rewards, next_values, dones,
                                     gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
        disc_ms         = _bench_gpu(compute_discounted_returns, rewards, dones,
                                     gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
        compiled_lam_ms = _bench_cpu(compiled_lambda, rewards, next_values, dones,
                                     gamma=0.99, lambda_=0.95)
        compiled_disc_ms= _bench_cpu(compiled_disc,   rewards, dones, gamma=0.99)

        speedup_lam  = compiled_lam_ms  / lambda_ms
        speedup_disc = compiled_disc_ms / disc_ms
        all_speedups.extend([speedup_lam, speedup_disc])

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{lambda_ms:>11.3f}ms {disc_ms:>9.3f}ms "
            f"{compiled_lam_ms:>13.3f}ms {compiled_disc_ms:>13.3f}ms "
            f"{speedup_lam:>12.1f}x  {speedup_disc:>12.1f}x"
        )

    print(
        "\nlambda(tri)/disc(tri): CUDA events — pure kernel time."
        "\npt.compile:            wall-clock — dispatches one CUDA op per timestep from Python."
        "\nspeedups are relative to each kernel's pt.compile baseline."
    )

    assert min(all_speedups) >= 1.5, (
        f"Expected >=1.5x speedup over pt.compile, worst was {min(all_speedups):.2f}x"
    )
