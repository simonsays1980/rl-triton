import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu
from rl_triton.ops.returns import compute_discounted_returns, compute_eligibility_traces, compute_lambda_returns

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


def _make_inputs_np(num_envs, seq_len, seed=0):
    """Return the same random inputs as _make_inputs but as NumPy arrays (CPU)."""
    args_cpu = _make_inputs(num_envs, seq_len, device="cpu", seed=seed)
    return tuple(t.numpy() for t in args_cpu)


# ---------------------------------------------------------------------------
# numpy baselines and np->triton->np adoption paths (benchmark only)
# ---------------------------------------------------------------------------

def numpy_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """CPU TD(λ) backward loop — moves GPU tensors to CPU and runs a plain Python loop."""
    rewards, next_values, dones = rewards.cpu(), next_values.cpu(), dones.cpu()
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0])
    for t in reversed(range(T)):
        not_done  = 1.0 - dones[:, t]
        carry     = (rewards[:, t]
                     + gamma * (1.0 - lambda_) * not_done * next_values[:, t]
                     + gamma * lambda_ * not_done * carry)
        out[:, t] = carry
    return out


def numpy_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """CPU discounted-return backward loop — moves GPU tensors to CPU and runs a plain Python loop."""
    rewards, dones = rewards.cpu(), dones.cpu()
    T     = rewards.shape[1]
    out   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0])
    for t in reversed(range(T)):
        carry     = rewards[:, t] + gamma * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


def numpy_eligibility_traces(
    gradients: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """CPU eligibility-trace forward loop — moves GPU tensors to CPU and runs a plain Python loop."""
    gradients, dones = gradients.cpu(), dones.cpu()
    T     = gradients.shape[1]
    out   = torch.zeros_like(gradients)
    carry = torch.zeros(gradients.shape[0])
    for t in range(T):
        carry     = gradients[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


def np_to_triton_to_np_lambda(
    rewards_np: np.ndarray,
    next_values_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """NumPy → GPU Triton kernel → NumPy end-to-end adoption path for TD(λ) returns."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch.float32)
    return compute_lambda_returns(
        to_gpu(rewards_np), to_gpu(next_values_np), to_gpu(dones_np),
        gamma=gamma, lambda_=lambda_,
    ).cpu().numpy()


def np_to_triton_to_np_discounted(
    rewards_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """NumPy → GPU Triton kernel → NumPy end-to-end adoption path for discounted returns."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch.float32)
    return compute_discounted_returns(
        to_gpu(rewards_np), to_gpu(dones_np), gamma=gamma,
    ).cpu().numpy()


def np_to_triton_to_np_eligibility(
    gradients_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """NumPy → GPU Triton kernel → NumPy end-to-end adoption path for eligibility traces."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch.float32)
    return compute_eligibility_traces(
        to_gpu(gradients_np), to_gpu(dones_np), gamma=gamma, lambda_=lambda_,
    ).cpu().numpy()


# ---------------------------------------------------------------------------
# Ground-truth value tests — compute_lambda_returns
# ---------------------------------------------------------------------------

@cuda_only
def test_lambda_returns_known_values_lambda0():
    # lambda=0: G[t] = r[t] + gamma*(1-d[t])*V(s_{t+1})  (one-step TD target).
    # No carry from the future at all.
    # rewards=[1,2], next_values=[3,4], gamma=1, no dones.
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
    # The recurrence G[t] = r[t] + gamma*(1-lambda)*next_values[t] + gamma*lambda*G[t+1]
    # maps to G[t] = u[t] + decay*G[t+1] with:
    #   u[t] = r[t] + (1-lambda)*next_values[t] = r[t] + 0.5*next_values[t]
    #   decay = lambda = 0.5
    # rewards=[1,2], next_values=[3,4]
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
    # rewards=[1,2], next_values=[3,4], lambda=0.5, gamma=1.
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
# Ground-truth value tests — compute_eligibility_traces
# ---------------------------------------------------------------------------

@cuda_only
def test_eligibility_traces_known_values_basic():
    # gamma=1, lambda=1, no dones, seed=0.
    # e[0] = 1 + 1*0 = 1
    # e[1] = 2 + 1*1 = 3
    # e[2] = 3 + 1*3 = 6
    gradients = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones    = torch.zeros(1, 3, device="cuda")
    out = compute_eligibility_traces(gradients, dones, gamma=1.0, lambda_=1.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, 3.0, 6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_known_values_decay():
    # gamma=1, lambda=0.5, no dones, seed=0.
    # e[0] = 1 + 0.5*0 = 1
    # e[1] = 2 + 0.5*1 = 2.5
    gradients = torch.tensor([[1.0, 2.0]], device="cuda")
    dones    = torch.zeros(1, 2, device="cuda")
    out = compute_eligibility_traces(gradients, dones, gamma=1.0, lambda_=0.5)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.5]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_known_values_done():
    # done[1]=1 resets the trace: e[1] = gradients[1] + gamma*lambda*(1-done[1])*e[0] = gradients[1] + 0.
    # e[0] = 1,  e[1] = 2 + 0.9*0 = 2,  e[2] = 3 + 0.9*2 = 4.8
    gradients = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones    = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    out = compute_eligibility_traces(gradients, dones, gamma=1.0, lambda_=0.9)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 4.8]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_known_values_seed():
    # Non-zero seed: e[-1]=2, gamma=1, lambda=1, no dones.
    # e[0] = 1 + 1*2 = 3
    # e[1] = 2 + 1*3 = 5
    gradients = torch.tensor([[1.0, 2.0]], device="cuda")
    dones    = torch.zeros(1, 2, device="cuda")
    seed     = torch.tensor([2.0], device="cuda")
    out = compute_eligibility_traces(gradients, dones, gamma=1.0, lambda_=1.0, seed_values=seed)
    torch.testing.assert_close(out, torch.tensor([[3.0, 5.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_lambda0():
    """lambda=0: trace equals the gradient at every step (no accumulation)."""
    gradients = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones    = torch.zeros(1, 3, device="cuda")
    out = compute_eligibility_traces(gradients, dones, gamma=0.99, lambda_=0.0)
    torch.testing.assert_close(out, gradients, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Correctness vs reference — compute_eligibility_traces
# ---------------------------------------------------------------------------

def reference_eligibility_traces(
    gradients: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch forward scan — ground truth."""
    T     = gradients.shape[1]
    out   = torch.zeros_like(gradients)
    carry = torch.zeros(gradients.shape[0], device=gradients.device, dtype=gradients.dtype)
    if seed_values is not None:
        carry = seed_values.clone()
    for t in range(T):
        carry     = gradients[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len,lambda_", [
    (1,   1,    0.0),
    (1,   7,    1.0),
    (4,   128,  0.5),
    (32,  333,  0.9),
    (128, 1024, 0.95),
])
def test_eligibility_traces_correctness(num_envs, seq_len, lambda_):
    rewards, _, dones = _make_inputs(num_envs, seq_len, seed=50)
    gradients = rewards   # any float tensor works as gradients
    expected = reference_eligibility_traces(gradients, dones, gamma=0.99, lambda_=lambda_)
    actual   = compute_eligibility_traces(gradients, dones, gamma=0.99, lambda_=lambda_)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_eligibility_traces_correctness_seed():
    rewards, _, dones = _make_inputs(32, 512, seed=51)
    seed     = torch.rand(32, device="cuda")
    expected = reference_eligibility_traces(rewards, dones, gamma=0.99, lambda_=0.9,
                                             seed_values=seed)
    actual   = compute_eligibility_traces(rewards, dones, gamma=0.99, lambda_=0.9,
                                           seed_values=seed)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


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
# ---------------------------------------------------------------------------
# Compiled vectorized baselines (benchmark only)
# ---------------------------------------------------------------------------

def vectorized_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """
    TD(λ) returns via flipped cumsum — vectorized compiled baseline.

    The recurrence G[t] = u[t] + (γλ(1-d[t])) * G[t+1] with
    u[t] = r[t] + γ(1-λ)(1-d[t])*V(s_{t+1}) has the same backward weighted-
    cumsum structure as discounted returns, with per-step additive term u[t]
    and per-step decay γλ(1-d[t]). The log-cumsum trick handles the
    episode-boundary resets. Not production-hardened; only used for benchmarking.
    """
    not_done = 1.0 - dones
    u        = rewards + gamma * (1.0 - lambda_) * not_done * next_values
    decay    = gamma * lambda_ * not_done
    log_dec  = torch.log(decay.clamp(min=1e-38))
    # Suffix log-product of decay factors: log_w[t] = sum_{k=t}^{T-1} log_dec[k]
    log_w    = torch.flip(torch.cumsum(torch.flip(log_dec, [1]), dim=1), [1])
    weights  = torch.exp(log_w)
    # Weighted backward cumsum: sum_{k=t}^{T-1} (prod_{j=t}^{k-1} decay[j]) * u[k]
    scaled   = u * weights
    running  = torch.flip(torch.cumsum(torch.flip(scaled, [1]), dim=1), [1])
    return running / weights


def vectorized_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Discounted returns via flipped cumsum — vectorized compiled baseline.

    Reverses the sequence, applies the cumsum correction trick with geometric
    discounting via log-space, then flips back. Not production-hardened;
    only used for benchmarking.
    """
    not_done  = 1.0 - dones
    log_gamma = torch.full_like(rewards, gamma).log() * not_done
    log_disc  = torch.flip(torch.cumsum(torch.flip(log_gamma, [1]), dim=1), [1])
    discounted = rewards * log_disc.exp()
    running   = torch.flip(torch.cumsum(torch.flip(discounted, [1]), dim=1), [1])
    return running / log_disc.exp()


def vectorized_eligibility_traces(
    gradients: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """
    Eligibility traces via cumsum correction trick — vectorized compiled baseline.

    Same segment-correction approach as cumsum_discounted_returns, extended
    with per-step geometric decay gamma*lambda*(1-done).  Not production-
    hardened; only used for benchmarking.
    """
    decay    = gamma * lambda_ * (1.0 - dones)
    log_acc  = torch.cumsum(torch.log(decay.clamp(min=1e-38)), dim=1)
    weights  = torch.exp(log_acc)
    scaled   = gradients * weights
    running  = torch.cumsum(scaled, dim=1)
    boundary = running * dones
    offset   = torch.cumsum(boundary, dim=1) - boundary
    return (running - offset) / weights


# ---------------------------------------------------------------------------
# Performance benchmarks (one per function)
# ---------------------------------------------------------------------------

BENCH_CONFIGS = [
    (64,  512),
    (128, 1024),
    (256, 1024),
    (512, 2048),
    (512, 4096),
]

SPEEDUP_THRESHOLD = 1.5


@cuda_only
@pytest.mark.slow
def test_lambda_returns_performance():
    """
    Benchmark compute_lambda_returns against torch.compile baselines.

    Baselines: (1) compiled sequential reference loop (wall-clock),
               (2) compiled vectorized cumsum (CUDA events),
               (3) numpy CPU loop (wall-clock).
    Also times the NumPy→GPU→NumPy adoption path end-to-end.
    Triton kernel uses CUDA events (pure kernel time).
    Assertions:
      - triton >=1.5x faster than max(loop, vec).
      - np->triton->np >=1.5x faster than numpy(cpu).
    """
    compiled_loop = torch.compile(reference_lambda_returns)
    compiled_vec  = torch.compile(vectorized_lambda_returns)

    _r, _nv, _d = _make_inputs(64, 512)
    compiled_loop(_r, _nv, _d, gamma=0.99, lambda_=0.95)
    compiled_vec(_r, _nv, _d, gamma=0.99, lambda_=0.95)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'compile(vec)':>14} {'compile(loop)':>15} "
        f"{'np->tri->np':>13} {'numpy(cpu)':>12} "
        f"{'vs vec':>8} {'vs loop':>9} {'np->tri->np vs numpy':>22}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_vec = []
    all_speedups_e2e = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_np  = _make_inputs_np(num_envs, seq_len)
        n_warmup, n_iter = _n_iter_gpu(seq_len, num_envs)

        rewards, next_values, dones = args_gpu
        rewards_np, next_values_np, dones_np = args_np

        tri_ms    = _bench_gpu(compute_lambda_returns, rewards, next_values, dones,
                                gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
        vec_ms    = _bench_gpu(compiled_vec,  rewards, next_values, dones,
                                gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
        loop_ms   = _bench_cpu(compiled_loop, rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        np_tri_ms = _bench_cpu(np_to_triton_to_np_lambda, rewards_np, next_values_np, dones_np,
                                gamma=0.99, lambda_=0.95)
        numpy_ms  = _bench_cpu(numpy_lambda_returns, rewards, next_values, dones,
                                gamma=0.99, lambda_=0.95)

        su_vec = vec_ms  / tri_ms
        su_loop = loop_ms / tri_ms
        su_e2e = numpy_ms / np_tri_ms
        all_speedups_vec.append(su_vec)
        all_speedups_e2e.append(su_e2e)
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{tri_ms:>8.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms "
            f"{np_tri_ms:>12.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_e2e:>20.1f}x"
        )

    print(
        "\ntriton      : CUDA events — pure kernel time."
        "\ncompile(vec): CUDA events — vectorized backward cumsum, no Python loop."
        "\ncompile(loop): wall-clock — one CUDA op per timestep from Python;"
        "\n               CUDA events would miss the CPU stall."
        "\nnp->tri->np : wall-clock — NumPy→GPU→NumPy, realistic adoption path."
        "\nnumpy(cpu)  : wall-clock — plain Python loop on CPU tensors."
        "\nspeedups vs triton kernel."
    )
    assert min(all_speedups_vec) >= SPEEDUP_THRESHOLD, (
        f"Expected >={SPEEDUP_THRESHOLD}x speedup over pt.compile(vec), worst was {min(all_speedups_vec):.2f}x"
    )
    assert min(all_speedups_e2e) >= SPEEDUP_THRESHOLD, (
        f"Expected >={SPEEDUP_THRESHOLD}x end-to-end speedup (np->triton->np vs numpy(cpu)), "
        f"worst was {min(all_speedups_e2e):.2f}x"
    )


@cuda_only
@pytest.mark.slow
def test_discounted_returns_performance():
    """
    Benchmark compute_discounted_returns against torch.compile baselines.

    Baselines: (1) compiled sequential reference loop (wall-clock),
               (2) compiled vectorized cumsum (CUDA events),
               (3) numpy CPU loop (wall-clock).
    Also times the NumPy→GPU→NumPy adoption path end-to-end.
    Triton kernel uses CUDA events (pure kernel time).
    Assertions:
      - triton >=1.5x faster than max(loop, vec).
      - np->triton->np >=1.5x faster than numpy(cpu).
    """
    compiled_loop = torch.compile(reference_discounted_returns)
    compiled_vec  = torch.compile(vectorized_discounted_returns)

    _r, _, _d = _make_inputs(64, 512)
    compiled_loop(_r, _d, gamma=0.99)
    compiled_vec(_r, _d, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'compile(vec)':>14} {'compile(loop)':>15} "
        f"{'np->tri->np':>13} {'numpy(cpu)':>12} "
        f"{'vs vec':>8} {'vs loop':>9} {'np->tri->np vs numpy':>22}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_vec = []
    all_speedups_e2e = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_np  = _make_inputs_np(num_envs, seq_len)
        n_warmup, n_iter = _n_iter_gpu(seq_len, num_envs)

        rewards, _, dones = args_gpu
        rewards_np, _, dones_np = args_np

        tri_ms    = _bench_gpu(compute_discounted_returns, rewards, dones,
                                gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
        vec_ms    = _bench_gpu(compiled_vec,  rewards, dones,
                                gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
        loop_ms   = _bench_cpu(compiled_loop, rewards, dones, gamma=0.99)
        np_tri_ms = _bench_cpu(np_to_triton_to_np_discounted, rewards_np, dones_np, gamma=0.99)
        numpy_ms  = _bench_cpu(numpy_discounted_returns, rewards, dones, gamma=0.99)

        su_vec  = vec_ms  / tri_ms
        su_loop = loop_ms / tri_ms
        su_e2e  = numpy_ms / np_tri_ms
        all_speedups_vec.append(su_vec)
        all_speedups_e2e.append(su_e2e)
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{tri_ms:>7.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms "
            f"{np_tri_ms:>12.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_e2e:>20.1f}x"
        )

    print(
        "\ntriton       : CUDA events — pure kernel time."
        "\ncompile(vec) : CUDA events — vectorized backward cumsum, no Python loop."
        "\ncompile(loop): wall-clock — one CUDA op per timestep from Python;"
        "\n               CUDA events would miss the CPU stall."
        "\nnp->tri->np  : wall-clock — NumPy→GPU→NumPy, realistic adoption path."
        "\nnumpy(cpu)   : wall-clock — plain Python loop on CPU tensors."
        "\nspeedups vs triton kernel."
    )
    assert min(all_speedups_vec) >= SPEEDUP_THRESHOLD, (
        f"Expected >={SPEEDUP_THRESHOLD}x speedup over pt.compile(vec), worst was {min(all_speedups_vec):.2f}x"
    )
    assert min(all_speedups_e2e) >= SPEEDUP_THRESHOLD, (
        f"Expected >={SPEEDUP_THRESHOLD}x end-to-end speedup (np->triton->np vs numpy(cpu)), "
        f"worst was {min(all_speedups_e2e):.2f}x"
    )


@cuda_only
@pytest.mark.slow
def test_eligibility_traces_performance():
    """
    Benchmark compute_eligibility_traces against torch.compile baselines.

    Baselines: (1) compiled sequential reference loop (wall-clock),
               (2) compiled vectorized cumsum (CUDA events),
               (3) numpy CPU loop (wall-clock).
    Also times the NumPy→GPU→NumPy adoption path end-to-end.
    Triton kernel uses CUDA events (pure kernel time).
    Assertions:
      - triton >=1.5x faster than max(loop, vec).
      - np->triton->np >=1.5x faster than numpy(cpu).
    """
    compiled_loop = torch.compile(reference_eligibility_traces)
    compiled_vec  = torch.compile(vectorized_eligibility_traces)

    _r, _, _d = _make_inputs(64, 512)
    compiled_loop(_r, _d, gamma=0.99, lambda_=0.95)
    compiled_vec(_r, _d, gamma=0.99, lambda_=0.95)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'compile(vec)':>14} {'compile(loop)':>15} "
        f"{'np->tri->np':>13} {'numpy(cpu)':>12} "
        f"{'vs vec':>8} {'vs loop':>9} {'np->tri->np vs numpy':>22}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_vec = []
    all_speedups_e2e = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_np  = _make_inputs_np(num_envs, seq_len)
        n_warmup, n_iter = _n_iter_gpu(seq_len, num_envs)

        gradients, _, dones = args_gpu
        gradients_np, _, dones_np = args_np

        tri_ms    = _bench_gpu(compute_eligibility_traces, gradients, dones,
                                gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
        vec_ms    = _bench_gpu(compiled_vec,  gradients, dones,
                                gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
        loop_ms   = _bench_cpu(compiled_loop, gradients, dones, gamma=0.99, lambda_=0.95)
        np_tri_ms = _bench_cpu(np_to_triton_to_np_eligibility, gradients_np, dones_np,
                                gamma=0.99, lambda_=0.95)
        numpy_ms  = _bench_cpu(numpy_eligibility_traces, gradients, dones,
                                gamma=0.99, lambda_=0.95)

        su_vec  = vec_ms  / tri_ms
        su_loop = loop_ms / tri_ms
        su_e2e  = numpy_ms / np_tri_ms
        all_speedups_vec.append(su_vec)
        all_speedups_e2e.append(su_e2e)
        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{tri_ms:>7.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms "
            f"{np_tri_ms:>12.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_e2e:>20.1f}x"
        )

    print(
        "\ntriton       : CUDA events — pure kernel time."
        "\ncompile(vec) : CUDA events — vectorized forward cumsum, no Python loop."
        "\ncompile(loop): wall-clock — one CUDA op per timestep from Python;"
        "\n               CUDA events would miss the CPU stall."
        "\nnp->tri->np  : wall-clock — NumPy→GPU→NumPy, realistic adoption path."
        "\nnumpy(cpu)   : wall-clock — plain Python loop on CPU tensors."
        "\nspeedups vs triton kernel."
    )
    assert min(all_speedups_vec) >= SPEEDUP_THRESHOLD, (
        f"Expected >={SPEEDUP_THRESHOLD}x speedup over pt.compile(vec), worst was {min(all_speedups_vec):.2f}x"
    )
    assert min(all_speedups_e2e) >= SPEEDUP_THRESHOLD, (
        f"Expected >={SPEEDUP_THRESHOLD}x end-to-end speedup (np->triton->np vs numpy(cpu)), "
        f"worst was {min(all_speedups_e2e):.2f}x"
    )
