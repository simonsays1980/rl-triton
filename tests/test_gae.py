# numpy is used intentionally: the CPU baseline (numpy_gae) mirrors how RL frameworks
# compute GAE on CPU — a plain backward loop over NumPy arrays. The end-to-end
# adoption path (numpy->GPU->numpy) is benchmarked in test_gae_performance.
import numpy as np
import pytest
import torch

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu, parallel_suffix_scan

triton = pytest.importorskip("triton")

from rl_triton.ops.gae import compute_gae

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _make_next_values(values: torch.Tensor, bootstrap_values: torch.Tensor | None) -> torch.Tensor:
    """Construct next_values from values by shifting: next_values[t] = values[t+1]."""
    nv = torch.empty_like(values)
    nv[:, :-1] = values[:, 1:]
    if bootstrap_values is None:
        nv[:, -1] = 0.0
    else:
        nv[:, -1] = bootstrap_values
    return nv


def reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch backward scan — ground truth for correctness tests.

    Derives next_values as values[:, t+1] with bootstrap_values at the boundary,
    matching exactly what the Triton kernel computes internally.
    """
    next_values = _make_next_values(values, bootstrap_values)
    T       = rewards.shape[1]
    adv     = torch.zeros_like(rewards)
    carry   = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        not_done  = 1.0 - terminateds[:, t]
        delta     = rewards[:, t] + gamma * not_done * next_values[:, t] - values[:, t]
        carry     = delta + gamma * lambda_ * not_done * carry
        adv[:, t] = carry
    return adv


def _make_inputs(num_envs, seq_len, device="cuda", seed=0):
    torch.manual_seed(seed)
    rewards = torch.randn(num_envs, seq_len, device=device)
    values  = torch.randn(num_envs, seq_len, device=device)
    terminateds   = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return rewards, values, terminateds


def _make_inputs_np(num_envs, seq_len, seed=0):
    """Return the same random inputs as _make_inputs but as NumPy arrays (CPU)."""
    args_cpu = _make_inputs(num_envs, seq_len, device="cpu", seed=seed)
    return tuple(t.numpy() for t in args_cpu)


def numpy_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """CPU GAE backward loop — moves GPU tensors to CPU and runs a plain Python loop."""
    rewards, values, terminateds = rewards.cpu(), values.cpu(), terminateds.cpu()
    next_values = _make_next_values(values, None)
    T     = rewards.shape[1]
    adv   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0])
    for t in reversed(range(T)):
        not_done  = 1.0 - terminateds[:, t]
        delta     = rewards[:, t] + gamma * not_done * next_values[:, t] - values[:, t]
        carry     = delta + gamma * lambda_ * not_done * carry
        adv[:, t] = carry
    return adv


def numpy_to_triton_to_numpy(
    rewards_np: np.ndarray,
    values_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """NumPy → GPU Triton kernel → NumPy end-to-end adoption path."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch.float32)
    result = compute_gae(
        to_gpu(rewards_np), to_gpu(values_np), to_gpu(dones_np),
        gamma=gamma, lambda_=lambda_,
    )
    torch.cuda.synchronize()
    return result.cpu().numpy()


# ---------------------------------------------------------------------------
# Ground-truth value tests
# ---------------------------------------------------------------------------
#
# Hand-computed from the recurrence
#   delta[t] = r[t] + gamma*(1-d[t])*V(s_{t+1}) - V(s_t)
#   A[t]     = delta[t] + gamma*lambda*(1-d[t])*A[t+1],  A[T] = bootstrap
#
# All tests use values=zeros so V(s_{t+1})=0 and delta[t]=r[t], making
# the arithmetic easy to verify by hand.

@cuda_only
def test_gae_known_values_single_env():
    # gamma=1, lambda=1, no terminateds, bootstrap=0, V=0 -> delta[t]=r[t].
    # A[2] = 3 + 0 = 3
    # A[1] = 2 + 3 = 5
    # A[0] = 1 + 5 = 6
    rewards  = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    values   = torch.zeros(1, 3, device="cuda")
    terminateds    = torch.zeros(1, 3, device="cuda")
    expected = torch.tensor([[6.0, 5.0, 3.0]], device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds, gamma=1.0, lambda_=1.0),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_delta():
    # gamma=1, lambda=1, no terminateds, bootstrap=0.
    # values=[1,1,1] -> next_values=[1,1,0(bootstrap)] -> delta[t]=r[t]+1-1=r[t]
    # Same result as above — verifies V shifts cancel correctly.
    rewards  = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    values   = torch.ones(1, 3, device="cuda")
    terminateds    = torch.zeros(1, 3, device="cuda")
    # next_values = [1, 1, 0(bootstrap)] -> delta = [1+1-1, 2+1-1, 3+0-1] = [1, 2, 2]
    # A[2] = 2
    # A[1] = 2 + 2 = 4
    # A[0] = 1 + 4 = 5
    expected = torch.tensor([[5.0, 4.0, 2.0]], device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds, gamma=1.0, lambda_=1.0),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_gamma():
    # gamma=0.9, lambda=1, no terminateds, bootstrap=0, V=0.
    # next_values = [0, 0(bootstrap)] -> delta[t]=r[t]
    # A[1] = 2 + 0.9*0 = 2
    # A[0] = 1 + 0.9*2 = 2.8
    rewards  = torch.tensor([[1.0, 2.0]], device="cuda")
    values   = torch.zeros(1, 2, device="cuda")
    terminateds    = torch.zeros(1, 2, device="cuda")
    expected = torch.tensor([[2.8, 2.0]], device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds, gamma=0.9, lambda_=1.0),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_lambda():
    # gamma=1, lambda=0, no terminateds, V=0 -> A[t] = delta[t] = r[t] (one-step TD).
    rewards  = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    values   = torch.zeros(1, 3, device="cuda")
    terminateds    = torch.zeros(1, 3, device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds, gamma=1.0, lambda_=0.0),
        rewards, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_truncated():
    # gamma=1, lambda=1, no terminateds, bootstrap=2.0, V=0.
    # bootstrap serves as both A[T] and V(s_T) (standard GAE semantics).
    # v_next = [values[1], values[2], bootstrap] = [0, 0, 2]
    # delta  = [1+0-0, 2+0-0, 3+2-0] = [1, 2, 5]
    # A[2] = delta[2] + 1*A[T] = 5 + 2 = 7
    # A[1] = delta[1] + 1*A[2] = 2 + 7 = 9
    # A[0] = delta[0] + 1*A[1] = 1 + 9 = 10
    rewards    = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    values     = torch.zeros(1, 3, device="cuda")
    terminateds      = torch.zeros(1, 3, device="cuda")
    bootstrap  = torch.tensor([2.0], device="cuda")
    expected   = torch.tensor([[10.0, 9.0, 7.0]], device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds,
                           gamma=1.0, lambda_=1.0, last_value=bootstrap),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_mixed_termination():
    # Two envs: env 0 terminated (bootstrap=0), env 1 truncated (bootstrap=5).
    # gamma=1, lambda=1, no mid-sequence terminateds, V=0.
    # Env 0: next_values=[0,0], delta=r, A[1]=2, A[0]=3
    # Env 1: next_values=[0,5], delta=[1,2+5]=[1,7]... wait:
    #   delta[1] = r[1] + gamma*next_v[1] - v[1] = 2 + 1*5 - 0 = 7
    #   delta[0] = r[0] + gamma*next_v[0] - v[0] = 1 + 1*0 - 0 = 1
    #   A[1] = delta[1] + decay*bootstrap = 7 + 1*5 = 12... no wait
    #   A[T] = bootstrap = 5 (carry)
    #   A[1] = delta[1] + gamma*lambda*(1-done)*A[T] = 7 + 1*5 = 12
    #   Hmm, that changes the hand calc. Let me use reference_gae as oracle.
    rewards   = torch.tensor([[1.0, 2.0], [1.0, 2.0]], device="cuda")
    values    = torch.zeros(2, 2, device="cuda")
    terminateds     = torch.zeros(2, 2, device="cuda")
    bootstrap = torch.tensor([0.0, 5.0], device="cuda")
    expected  = reference_gae(rewards, values, terminateds,
                               gamma=1.0, lambda_=1.0, bootstrap_values=bootstrap)
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds,
                           gamma=1.0, lambda_=1.0, last_value=bootstrap),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_episode_boundary():
    # done[1]=1 resets the carry: decay[1] = gamma*lambda*(1-1) = 0.
    # gamma=1, lambda=1, V=0 -> delta[t]=r[t] (next_values=[0,0,0(boot)])
    # A[2] = 3 + 1*0 = 3
    # A[1] = 2 + 0*3 = 2   <- boundary resets carry
    # A[0] = 1 + 1*2 = 3
    rewards  = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    values   = torch.zeros(1, 3, device="cuda")
    terminateds    = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    expected = torch.tensor([[3.0, 2.0, 3.0]], device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds, gamma=1.0, lambda_=1.0),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_batch():
    # Two envs, gamma=1, lambda=1, no terminateds, V=0.
    # next_values = zeros (bootstrap=0) -> delta=r
    # Env 0: A[1]=2, A[0]=1+2=3
    # Env 1: A[1]=4, A[0]=3+4=7
    rewards  = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    values   = torch.zeros(2, 2, device="cuda")
    terminateds    = torch.zeros(2, 2, device="cuda")
    expected = torch.tensor([[3.0, 2.0], [7.0, 4.0]], device="cuda")
    torch.testing.assert_close(
        compute_gae(rewards, values, terminateds, gamma=1.0, lambda_=1.0),
        expected, atol=1e-5, rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Correctness vs reference
# ---------------------------------------------------------------------------

@cuda_only
def test_gae_correctness_basic():
    rewards, values, terminateds = _make_inputs(64, 512, seed=0)
    expected = reference_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
    actual   = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),
    (4,   128),
    (32,  333),
    (128, 1024),
    (256, 2048),
])
def test_gae_correctness_shapes(num_envs, seq_len):
    rewards, values, terminateds = _make_inputs(num_envs, seq_len, seed=42)
    expected = reference_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
    actual   = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_gae_correctness_bootstrap():
    rewards, values, terminateds = _make_inputs(32, 512, seed=3)
    bootstrap = torch.rand(32, device="cuda")
    expected  = reference_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95,
                               bootstrap_values=bootstrap)
    actual    = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95,
                            last_value=bootstrap)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_gae_lambda0():
    """lambda=0: advantage equals the one-step TD error at every step."""
    rewards, values, terminateds = _make_inputs(8, 64, seed=1)
    # next_values = values shifted by 1, with 0 at end
    next_values = _make_next_values(values, None)
    not_done = 1.0 - terminateds
    expected = rewards + 0.99 * not_done * next_values - values
    actual   = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Sequential reference for full [num_envs, seq_len] bootstrap_values
# ---------------------------------------------------------------------------

def _ref_gae_sequential(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Pure step-by-step Python loop — ground truth for per-step bootstrap_values.

    Handles interior truncated steps (bootstrap_values[n, t] used as v_next when
    truncateds[n, t]=1) in addition to the window boundary (t=T-1).
    """
    N, T = rewards.shape
    out = torch.zeros_like(rewards)
    for n in range(N):
        carry = bootstrap_values[n, T - 1].item()   # A[T] = boundary bootstrap
        for t in reversed(range(T)):
            if t == T - 1 or truncateds[n, t].item() == 1.0:
                v_next = bootstrap_values[n, t].item()
            else:
                v_next = values[n, t + 1].item()
            not_terminated = 1.0 - terminateds[n, t].item()
            done = min(1.0, terminateds[n, t].item() + truncateds[n, t].item())
            delta = rewards[n, t].item() + gamma * not_terminated * v_next - values[n, t].item()
            carry = delta + gamma * lambda_ * (1.0 - done) * carry
            out[n, t] = carry
    return out


# ---------------------------------------------------------------------------
# bootstrap_values / last_value API tests
# ---------------------------------------------------------------------------

@cuda_only
def test_gae_two_interior_truncations():
    """Two interior truncated episodes + continuing window boundary.

    This is the case CleanRL/SB3-style single-bootstrap approaches get wrong.
    Verified against a pure sequential Python reference loop.
    """
    torch.manual_seed(99)
    N, T = 2, 12
    rewards     = torch.rand(N, T, device="cuda")
    values      = torch.rand(N, T, device="cuda")
    terminateds = torch.zeros(N, T, device="cuda")
    truncateds  = torch.zeros(N, T, device="cuda")

    # env 0: truncations at t=3 and t=7; window continues past t=11
    truncateds[0, 3]  = 1.0
    truncateds[0, 7]  = 1.0
    # env 1: single truncation at t=5; window terminates at t=11
    truncateds[1, 5]  = 1.0

    bootstrap_values = torch.zeros(N, T, device="cuda")
    bootstrap_values[0, 3]  = 2.5    # V(s_{t=4}^true) for env 0, first truncation
    bootstrap_values[0, 7]  = 1.8    # V(s_{t=8}^true) for env 0, second truncation
    bootstrap_values[0, 11] = 3.2    # V(s_T), env 0 window continues
    bootstrap_values[1, 5]  = 0.9    # V(s_{t=6}^true) for env 1
    # bootstrap_values[1, 11] stays 0: env 1 terminates at t=11

    expected = _ref_gae_sequential(
        rewards.cpu(), values.cpu(), terminateds.cpu(), truncateds.cpu(),
        bootstrap_values.cpu(), gamma=0.99, lambda_=0.95,
    ).cuda()
    actual = compute_gae(
        rewards, values, terminateds, truncateds,
        gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@cuda_only
def test_gae_bootstrap_values_full_tensor():
    """Passing a full [num_envs, seq_len] bootstrap_values tensor (no last_value).

    Exercises the public bootstrap_values API directly — values at truncated steps
    plus a non-zero boundary column for a continuing episode.
    """
    torch.manual_seed(42)
    N, T = 8, 64
    rewards, values, terminateds = _make_inputs(N, T, seed=42)
    truncateds = (torch.rand(N, T, device="cuda") < 0.05).float()
    # Ensure terminated and truncated are mutually exclusive
    truncateds = truncateds * (1.0 - terminateds)

    bootstrap_values = torch.zeros(N, T, device="cuda")
    # Set bootstrap at every truncated step
    bootstrap_values[truncateds.bool()] = torch.rand(int(truncateds.sum().item()), device="cuda")
    # Set boundary continuation value
    bootstrap_values[:, -1] = torch.rand(N, device="cuda")

    expected = _ref_gae_sequential(
        rewards.cpu(), values.cpu(), terminateds.cpu(), truncateds.cpu(),
        bootstrap_values.cpu(), gamma=0.99, lambda_=0.95,
    ).cuda()
    actual = compute_gae(
        rewards, values, terminateds, truncateds,
        gamma=0.99, lambda_=0.95, bootstrap_values=bootstrap_values,
    )
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_gae_last_value_mutual_exclusion():
    """Passing both last_value and bootstrap_values must raise."""
    rewards     = torch.rand(4, 16, device="cuda")
    values      = torch.rand(4, 16, device="cuda")
    terminateds = torch.zeros(4, 16, device="cuda")
    bv = torch.zeros(4, 16, device="cuda")
    lv = torch.rand(4, device="cuda")
    with pytest.raises(AssertionError, match="not both"):
        compute_gae(rewards, values, terminateds,
                    bootstrap_values=bv, last_value=lv)


@cuda_only
def test_gae_last_value_equivalence():
    """last_value=[num_envs] produces identical output to the equivalent
    hand-built bootstrap_values[:, -1] tensor."""
    torch.manual_seed(7)
    N, T = 8, 64
    rewards, values, terminateds = _make_inputs(N, T, seed=7)
    lv = torch.rand(N, device="cuda")

    actual = compute_gae(rewards, values, terminateds,
                         gamma=0.99, lambda_=0.95, last_value=lv)

    bv = torch.zeros(N, T, device="cuda")
    bv[:, -1] = lv
    expected = compute_gae(rewards, values, terminateds,
                           gamma=0.99, lambda_=0.95, bootstrap_values=bv)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@cuda_only
def test_gae_non_contiguous_input():
    """Wrapper must handle non-contiguous inputs via .contiguous()."""
    base    = torch.randn(64, 512, 2, device="cuda")
    rewards = base[..., 0]
    values  = torch.randn(64, 512, 2, device="cuda")[..., 0]
    terminateds   = torch.zeros(64, 512, 2, device="cuda")[..., 0]

    expected = reference_gae(
        rewards.contiguous(), values.contiguous(), terminateds.contiguous(),
        gamma=0.99, lambda_=0.95,
    )
    actual = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


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
def test_gae_performance():
    """
    Sweep over (num_envs, seq_len) configs comparing:

      triton           — Triton kernel  (CUDA events)
      pt.compile(vec)  — torch.compile on vectorized_gae  (CUDA events)
      pt.compile(loop) — torch.compile on reference_gae loop  (wall-clock)
      np->triton->np   — NumPy→GPU→NumPy adoption path  (wall-clock)
      numpy(cpu)       — CPU Python loop  (wall-clock)

    Scope: NO-TRUNCATION PATH ONLY.  _make_inputs produces no truncateds, so
    the kernel dispatches to HAS_TRUNCATIONS=False.  vectorized_gae also takes
    no truncateds.  Both sides solve the same termination-only problem — the
    comparison is apples-to-apples for this path.  For the truncation-path
    speedup (HAS_TRUNCATIONS=True, the feature that distinguishes this library
    from single-bootstrap approaches) see test_gae_truncation_performance.

    Assertions:
      - triton >=1.5x faster than pt.compile(vec).
      - np->triton->np >=1.5x faster than numpy(cpu).
    """
    compiled_vec  = torch.compile(vectorized_gae)
    compiled_loop = torch.compile(reference_gae)

    _args = _make_inputs(64, 512)
    compiled_vec(*_args, gamma=0.99, lambda_=0.95)
    compiled_loop(*_args, gamma=0.99, lambda_=0.95)
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
        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)

        tri_ms    = _bench_gpu(compute_gae, *args_gpu,
                                gamma=0.99, lambda_=0.95, n_warmup=gpu_warmup, n_iter=gpu_iter)
        vec_ms    = _bench_gpu(compiled_vec,  *args_gpu,
                                gamma=0.99, lambda_=0.95, n_warmup=gpu_warmup, n_iter=gpu_iter)
        loop_ms   = _bench_cpu(compiled_loop, *args_gpu, gamma=0.99, lambda_=0.95)
        np_tri_ms = _bench_cpu(numpy_to_triton_to_numpy, *args_np, gamma=0.99, lambda_=0.95)
        numpy_ms  = _bench_cpu(numpy_gae, *args_gpu, gamma=0.99, lambda_=0.95)

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
        "\ncompile(vec) : CUDA events — vectorized log-space cumsum, no Python loop."
        "\ncompile(loop): wall-clock — one CUDA op per timestep from Python;"
        "\n               CUDA events would miss the CPU stall."
        "\nnp->tri->np  : wall-clock — NumPy→GPU→NumPy, realistic adoption path."
        "\nnumpy(cpu)   : wall-clock — plain Python loop on CPU tensors."
        "\nspeedups vs triton kernel."
    )

    assert min(all_speedups_vec) >= 1.5, (
        f"Expected >=1.5x speedup over pt.compile(vec) across all configs, "
        f"worst was {min(all_speedups_vec):.2f}x"
    )
    assert min(all_speedups_e2e) >= 1.5, (
        f"Expected >=1.5x end-to-end speedup (np->triton->np vs numpy(cpu)) across all configs, "
        f"worst was {min(all_speedups_e2e):.2f}x"
    )


# ---------------------------------------------------------------------------
# Vectorized PyTorch baseline (benchmark only)
# ---------------------------------------------------------------------------

def vectorized_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fully vectorized GAE via log-space suffix cumsum — strong compiled baseline.

    Replaces the Python backward loop with a vectorized equivalent using the same
    log-space suffix-product trick as vectorized_vtrace. Not production-hardened;
    only used for benchmarking.
    """
    not_done    = 1.0 - terminateds
    next_values = _make_next_values(values, bootstrap_values)
    deltas      = rewards + gamma * not_done * next_values - values
    decays      = gamma * lambda_ * not_done

    log_suffix = torch.flip(
        torch.cumsum(torch.flip(torch.log(decays.clamp(min=1e-38)), [1]), dim=1), [1]
    )
    weights = torch.exp(log_suffix)
    adv     = torch.flip(
        torch.cumsum(torch.flip(deltas * weights, [1]), dim=1), [1]
    ) / weights

    if bootstrap_values is not None:
        adv = adv + weights * bootstrap_values.unsqueeze(1)

    return adv



def vectorized_gae_with_truncations(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """
    Vectorized GAE with truncation support — log-depth parallel scan baseline.

    Uses the linear-space associative scan combine operator
      (a1, b1) ∘ (a2, b2) = (a2 + b2*a1, b2*b1)
    which is associative and handles decay=0 at boundaries exactly: b=0
    severs the carry (right element wins), so no boundary-specific logic
    is needed.  This is the same algorithm tl.associative_scan uses in the
    Triton kernel.  Log2(T) doubling steps, each fully vectorized over N and T.

    The log-space cumsum trick fails here because decay[t]=0 at truncated/
    terminated steps gives log(0)=-inf which contaminates the suffix cumsum.
    The linear-space associative scan handles decay=0 directly.

    Matches the kernel semantics exactly:
      - next_values[t] = values[t+1] at interior non-truncated steps,
        bootstrap_values[t] at truncated steps and the boundary.
      - delta[t] = r[t] + gamma*(1-terminated[t])*next_values[t] - V(s_t)
      - decay[t] = gamma*lambda*(1 - clamp(terminated[t]+truncated[t], max=1))
      - carry seeded from bootstrap_values[:, -1].
    """
    not_terminated = 1.0 - terminateds
    not_done       = 1.0 - (terminateds + truncateds).clamp(max=1.0)

    # next_values[t]: values[t+1] at interior non-truncated steps, bootstrap elsewhere.
    v_next_raw         = torch.empty_like(values)
    v_next_raw[:, :-1] = values[:, 1:] * (1.0 - truncateds[:, :-1])
    v_next_raw[:, -1]  = 0.0
    next_values        = v_next_raw + bootstrap_values

    deltas = rewards + gamma * not_terminated * next_values - values
    decays = gamma * lambda_ * not_done

    # Incorporate boundary bootstrap into the initial 'a' values:
    # The suffix scan computes G[t] assuming G[T]=0.  We need G[T-1] to include
    # the carry from bootstrap_values[:, -1].  Inject it by treating the boundary
    # as delta[T-1] += decay[T-1] * bootstrap (which is 0 since T-1 may not be a
    # boundary, but the last step always contributes bootstrap).
    # Simpler: append a sentinel step at position T with a=bootstrap, b=0, then scan.
    N = rewards.shape[0]
    bootstrap = bootstrap_values[:, -1].unsqueeze(1)   # [N, 1]
    a = torch.cat([deltas, bootstrap], dim=1)           # [N, T+1]
    b = torch.cat([decays, torch.zeros(N, 1, device=decays.device, dtype=decays.dtype)], dim=1)
    result = parallel_suffix_scan(a, b)                 # [N, T+1]
    return result[:, :rewards.shape[1]]                 # [N, T]


def test_vectorized_gae_with_truncations_correctness():
    """Verify vectorized_gae_with_truncations matches _ref_gae_sequential.

    Uses the same two-interior-truncation fixture as test_gae_two_interior_truncations
    so correctness of the new baseline is confirmed against the sequential reference
    before it is used in the benchmark.
    """
    torch.manual_seed(99)
    N, T = 2, 12
    rewards     = torch.rand(N, T)
    values      = torch.rand(N, T)
    terminateds = torch.zeros(N, T)
    truncateds  = torch.zeros(N, T)

    truncateds[0, 3] = 1.0
    truncateds[0, 7] = 1.0
    truncateds[1, 5] = 1.0

    bootstrap_values = torch.zeros(N, T)
    bootstrap_values[0, 3]  = 2.5
    bootstrap_values[0, 7]  = 1.8
    bootstrap_values[0, 11] = 3.2
    bootstrap_values[1, 5]  = 0.9

    expected = _ref_gae_sequential(
        rewards, values, terminateds, truncateds, bootstrap_values,
        gamma=0.99, lambda_=0.95,
    )
    actual = vectorized_gae_with_truncations(
        rewards, values, terminateds, truncateds, bootstrap_values,
        gamma=0.99, lambda_=0.95,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@cuda_only
@pytest.mark.slow
def test_gae_truncation_performance():
    """
    Truncation-path performance: HAS_TRUNCATIONS=True kernel vs
    torch.compile(vectorized_gae_with_truncations).

    vectorized_gae_with_truncations uses parallel_suffix_scan: a log-depth
    parallel associative scan with no Python time loop, compiles cleanly.
    Uses the same combine operator as tl.associative_scan in the kernel.

    Inputs have ~5% truncated steps (mutually exclusive with terminated),
    so the kernel dispatches to HAS_TRUNCATIONS=True.  Both sides compute
    full per-step bootstrap support.  This path reads 7 full-width tensors
    vs 5 for the no-truncation path, so expect a lower speedup number.

    No assertion floor is imposed here — the truncation path is a correctness
    feature, not a performance regression from the no-truncation baseline.
    This test exists to make the truncation-path speedup visible and tracked.
    """
    compiled_vec_trunc = torch.compile(vectorized_gae_with_truncations)

    def _make_trunc_inputs(num_envs, seq_len, seed=0):
        torch.manual_seed(seed)
        rewards     = torch.randn(num_envs, seq_len, device="cuda")
        values      = torch.randn(num_envs, seq_len, device="cuda")
        terminateds = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
        # truncateds mutually exclusive with terminateds, ~5% rate
        trunc_cand  = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
        truncateds  = trunc_cand * (1.0 - terminateds)
        bootstrap_values = torch.zeros(num_envs, seq_len, device="cuda")
        bootstrap_values[truncateds.bool()] = torch.rand(
            int(truncateds.sum().item()), device="cuda"
        )
        bootstrap_values[:, -1] = torch.rand(num_envs, device="cuda")
        return rewards, values, terminateds, truncateds, bootstrap_values

    # Warmup — compiles compiled_vec_trunc.
    _wa = _make_trunc_inputs(64, 512, seed=1)
    compiled_vec_trunc(*_wa, gamma=0.99, lambda_=0.95)
    compute_gae(_wa[0], _wa[1], _wa[2], _wa[3], gamma=0.99, lambda_=0.95,
                bootstrap_values=_wa[4])
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'compile(vec_trunc)':>20} {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))

    all_speedups = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args = _make_trunc_inputs(num_envs, seq_len)
        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)

        tri_ms = _bench_gpu(
            compute_gae, args[0], args[1], args[2], args[3],
            gamma=0.99, lambda_=0.95, bootstrap_values=args[4],
            n_warmup=gpu_warmup, n_iter=gpu_iter,
        )
        vec_ms = _bench_gpu(
            compiled_vec_trunc, *args,
            gamma=0.99, lambda_=0.95,
            n_warmup=gpu_warmup, n_iter=gpu_iter,
        )

        su = vec_ms / tri_ms
        all_speedups.append(su)
        print(f"{num_envs:>10} {seq_len:>8} {tri_ms:>7.3f}ms {vec_ms:>19.3f}ms {su:>8.2f}x")

    print(
        f"\nMin speedup: {min(all_speedups):.2f}x  "
        f"Max: {max(all_speedups):.2f}x  "
        f"(HAS_TRUNCATIONS=True path; 7 full-width reads vs 5 for no-truncation path)"
    )
