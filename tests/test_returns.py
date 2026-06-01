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
    features = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones    = torch.zeros(1, 3, device="cuda")
    out = compute_eligibility_traces(features, dones, gamma=1.0, lambda_=1.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, 3.0, 6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_known_values_decay():
    # gamma=1, lambda=0.5, no dones, seed=0.
    # e[0] = 1 + 0.5*0 = 1
    # e[1] = 2 + 0.5*1 = 2.5
    features = torch.tensor([[1.0, 2.0]], device="cuda")
    dones    = torch.zeros(1, 2, device="cuda")
    out = compute_eligibility_traces(features, dones, gamma=1.0, lambda_=0.5)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.5]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_known_values_done():
    # done[1]=1 resets the trace: e[1] = features[1] + gamma*lambda*(1-done[1])*e[0] = features[1] + 0.
    # e[0] = 1,  e[1] = 2 + 0.9*0 = 2,  e[2] = 3 + 0.9*2 = 4.8
    features = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones    = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    out = compute_eligibility_traces(features, dones, gamma=1.0, lambda_=0.9)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 4.8]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_known_values_seed():
    # Non-zero seed: e[-1]=2, gamma=1, lambda=1, no dones.
    # e[0] = 1 + 1*2 = 3
    # e[1] = 2 + 1*3 = 5
    features = torch.tensor([[1.0, 2.0]], device="cuda")
    dones    = torch.zeros(1, 2, device="cuda")
    seed     = torch.tensor([2.0], device="cuda")
    out = compute_eligibility_traces(features, dones, gamma=1.0, lambda_=1.0, seed_values=seed)
    torch.testing.assert_close(out, torch.tensor([[3.0, 5.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_eligibility_traces_lambda0():
    """lambda=0: trace equals the feature at every step (no accumulation)."""
    features = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones    = torch.zeros(1, 3, device="cuda")
    out = compute_eligibility_traces(features, dones, gamma=0.99, lambda_=0.0)
    torch.testing.assert_close(out, features, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Correctness vs reference — compute_eligibility_traces
# ---------------------------------------------------------------------------

def reference_eligibility_traces(
    features: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch forward scan — ground truth."""
    T     = features.shape[1]
    out   = torch.zeros_like(features)
    carry = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
    if seed_values is not None:
        carry = seed_values.clone()
    for t in range(T):
        carry     = features[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * carry
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
    features = rewards   # any float tensor works as features
    expected = reference_eligibility_traces(features, dones, gamma=0.99, lambda_=lambda_)
    actual   = compute_eligibility_traces(features, dones, gamma=0.99, lambda_=lambda_)
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
# Compiled cumsum baselines (benchmark only)
# ---------------------------------------------------------------------------

def cumsum_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Discounted returns via flipped cumsum — strong compiled baseline.

    Reverses the sequence, applies the cumsum correction trick with geometric
    discounting via log-space, then flips back.  Not production-hardened;
    only used for benchmarking.
    """
    not_done  = 1.0 - dones
    # Build per-step discount factors and compute log-cumsum for geometric weights.
    log_gamma = torch.full_like(rewards, gamma).log() * not_done
    log_disc  = torch.flip(torch.cumsum(torch.flip(log_gamma, [1]), dim=1), [1])
    discounted = rewards * log_disc.exp()
    running   = torch.flip(torch.cumsum(torch.flip(discounted, [1]), dim=1), [1])
    return running / log_disc.exp()


def cumsum_eligibility_traces(
    features: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """
    Eligibility traces via cumsum correction trick — strong compiled baseline.

    Same segment-correction approach as cumsum_episodic_prefix_sum, extended
    with per-step geometric decay gamma*lambda*(1-done).  Not production-
    hardened; only used for benchmarking.
    """
    decay    = gamma * lambda_ * (1.0 - dones)
    log_acc  = torch.cumsum(torch.log(decay.clamp(min=1e-38)), dim=1)
    weights  = torch.exp(log_acc)
    scaled   = features * weights
    running  = torch.cumsum(scaled, dim=1)
    boundary = running * dones
    offset   = torch.cumsum(boundary, dim=1) - boundary
    return (running - offset) / weights


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
    compiled_lam_loop  = torch.compile(reference_lambda_returns)
    compiled_disc_loop = torch.compile(reference_discounted_returns)
    compiled_trc_loop  = torch.compile(reference_eligibility_traces)
    compiled_disc_vec  = torch.compile(cumsum_discounted_returns)
    compiled_trc_vec   = torch.compile(cumsum_eligibility_traces)

    _r, _nv, _d = _make_inputs(64, 512)
    compiled_lam_loop(_r, _nv, _d, gamma=0.99, lambda_=0.95)
    compiled_disc_loop(_r, _d, gamma=0.99)
    compiled_trc_loop(_r, _d, gamma=0.99, lambda_=0.95)
    compiled_disc_vec(_r, _d, gamma=0.99)
    compiled_trc_vec(_r, _d, gamma=0.99, lambda_=0.95)
    torch.cuda.synchronize()

    # lambda returns have no clean vectorized cumsum equivalent — loop is the baseline.
    # discounted returns and eligibility traces have a vectorized cumsum baseline.
    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'λ(tri)':>8} {'G(tri)':>8} {'e(tri)':>8} "
        f"{'λ loop':>8} {'G loop':>8} {'e loop':>8} "
        f"{'G vec':>7} {'e vec':>7} "
        f"{'numpy λ':>9} {'numpy G':>9} {'numpy e':>9} "
        f"{'vs λ':>6} {'vs G':>6} {'vs e':>6}"
    )
    print(header)
    print("-" * len(header))

    all_speedups = []

    for num_envs, seq_len in BENCH_CONFIGS:
        rewards, next_values, dones = _make_inputs(num_envs, seq_len)
        rewards_cpu, next_values_cpu, dones_cpu = _make_inputs(num_envs, seq_len, device="cpu")
        n_warmup, n_iter = _n_iter_gpu(seq_len, num_envs)

        lam_ms      = _bench_gpu(compute_lambda_returns,    rewards, next_values, dones,
                                  gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)
        disc_ms     = _bench_gpu(compute_discounted_returns, rewards, dones,
                                  gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
        traces_ms   = _bench_gpu(compute_eligibility_traces, rewards, dones,
                                  gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)

        lam_loop_ms  = _bench_cpu(compiled_lam_loop,  rewards, next_values, dones, gamma=0.99, lambda_=0.95)
        disc_loop_ms = _bench_cpu(compiled_disc_loop,  rewards, dones, gamma=0.99)
        trc_loop_ms  = _bench_cpu(compiled_trc_loop,   rewards, dones, gamma=0.99, lambda_=0.95)

        disc_vec_ms  = _bench_gpu(compiled_disc_vec,   rewards, dones,
                                  gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)
        trc_vec_ms   = _bench_gpu(compiled_trc_vec,    rewards, dones,
                                  gamma=0.99, lambda_=0.95, n_warmup=n_warmup, n_iter=n_iter)

        numpy_lam_ms  = _bench_cpu(reference_lambda_returns,    rewards_cpu, next_values_cpu, dones_cpu, gamma=0.99, lambda_=0.95)
        numpy_disc_ms = _bench_cpu(reference_discounted_returns, rewards_cpu, dones_cpu, gamma=0.99)
        numpy_trc_ms  = _bench_cpu(reference_eligibility_traces, rewards_cpu, dones_cpu, gamma=0.99, lambda_=0.95)

        # Assert against strongest compiled baseline per kernel.
        su_lam  = lam_loop_ms  / lam_ms
        su_disc = disc_vec_ms  / disc_ms
        su_trc  = trc_vec_ms   / traces_ms
        all_speedups.extend([su_lam, su_disc, su_trc])

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{lam_ms:>7.3f}ms {disc_ms:>7.3f}ms {traces_ms:>7.3f}ms "
            f"{lam_loop_ms:>7.3f}ms {disc_loop_ms:>7.3f}ms {trc_loop_ms:>7.3f}ms "
            f"{disc_vec_ms:>6.3f}ms {trc_vec_ms:>6.3f}ms "
            f"{numpy_lam_ms:>8.3f}ms {numpy_disc_ms:>8.3f}ms {numpy_trc_ms:>8.3f}ms "
            f"{su_lam:>5.1f}x {su_disc:>5.1f}x {su_trc:>5.1f}x"
        )

    print(
        "\nλ/G/e (tri)   : CUDA events — pure kernel time, no CPU overhead."
        "\nloop baselines : wall-clock — one CUDA op per timestep from Python."
        "\nvec baselines  : CUDA events — vectorized cumsum (no Python loop); "
        "\n                 not available for λ-returns (mixed V term prevents clean cumsum)."
        "\nnumpy          : wall-clock — reference loop on CPU tensors."
        "\nspeedups vs strongest compiled baseline: λ vs loop, G vs vec, e vs vec."
    )

    assert min(all_speedups) >= 1.5, (
        f"Expected >=1.5x speedup over strongest compiled baseline, worst was {min(all_speedups):.2f}x"
    )
