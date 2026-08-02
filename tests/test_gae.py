# numpy is used intentionally: the CPU baseline (numpy_gae) mirrors how RL frameworks
# compute GAE on CPU -- a plain backward loop over NumPy arrays. The end-to-end
# adoption path (numpy->GPU->numpy) is benchmarked in test_gae_performance.
import numpy as np
import pytest
import torch

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu, _warmup_gpu, parallel_suffix_scan

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
    """Pure-PyTorch backward scan -- ground truth for correctness tests.

    Derives next_values as values[:, t+1] with bootstrap_values at the boundary,
    matching exactly what the Triton kernel computes internally.

    The scan carry A[T] is always 0: bootstrap_values[:, -1] already enters
    the recurrence once, inside delta[T-1] via next_values (weight 1).
    Seeding the carry with it too would double-count it.
    """
    next_values = _make_next_values(values, bootstrap_values)
    T       = rewards.shape[1]
    adv     = torch.zeros_like(rewards)
    carry   = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
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
    """CPU GAE backward loop -- moves GPU tensors to CPU and runs a plain Python loop."""
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
    # Same result as above -- verifies V shifts cancel correctly.
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
    # bootstrap feeds delta[T-1] as V(s_T) only; the additive boundary carry
    # A[T] is always 0 (see kernels/gae.py's module docstring).
    # v_next = [values[1], values[2], bootstrap] = [0, 0, 2]
    # delta  = [1+0-0, 2+0-0, 3+2-0] = [1, 2, 5]
    # A[2] = delta[2] + 1*A[T] = 5 + 1*0 = 5
    # A[1] = delta[1] + 1*A[2] = 2 + 5 = 7
    # A[0] = delta[0] + 1*A[1] = 1 + 7 = 8
    rewards    = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    values     = torch.zeros(1, 3, device="cuda")
    terminateds      = torch.zeros(1, 3, device="cuda")
    bootstrap  = torch.tensor([2.0], device="cuda")
    expected   = torch.tensor([[8.0, 7.0, 5.0]], device="cuda")
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
    # Env 1: next_values=[0,5], delta=[1, 2+5]=[1,7].
    #   Additive boundary carry A[T]=0 (see kernels/gae.py's module docstring):
    #   A[1] = delta[1] + gamma*lambda*(1-done)*A[T] = 7 + 1*0 = 7
    #   A[0] = delta[0] + gamma*lambda*(1-done)*A[1] = 1 + 1*7 = 8
    # Checked against reference_gae as oracle (also fixed to seed A[T]=0).
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
# lambda=1 Monte-Carlo identity -- regression test for the window-boundary
# bootstrap double-counting bug.
#
# At lambda=1, GAE must telescope exactly to the Monte-Carlo advantage
#   A_t = G_t - V(s_t),   G_t = discounted return with bootstrap tail.
# This is an implementation-independent identity, not a property of any
# particular oracle -- it holds regardless of how reference_gae or
# _ref_gae_sequential happen to be written, which is why this test computes
# G_t by hand from rewards/gamma/bootstrap directly and never calls either
# of them.  A previous version of compute_gae seeded the scan's additive
# boundary carry with the window bootstrap V(s_T) in addition to using it
# inside delta[T-1], double-counting it by exactly
# (gamma*lambda)^(T-t) * V(s_T).  That error is zero whenever V(s_T)=0 (the
# vanishing case most fixtures exercise, e.g. an episode that terminates
# exactly at the window edge), which is why the rest of the suite didn't
# catch it -- this test uses a NONZERO bootstrap on a continuing episode
# (no terminated/truncated flags at all) specifically to make the error
# visible.
# ---------------------------------------------------------------------------

@cuda_only
def test_gae_lambda1_monte_carlo_identity():
    """lambda=1 => A_t = G_t - V(s_t), with a nonzero window bootstrap.

    G_t is computed here by a direct hand-rolled backward recurrence over
    rewards/gamma/bootstrap -- NOT via reference_gae or _ref_gae_sequential,
    which is the point: those oracles encoded this exact bug until they were
    fixed alongside the kernel, so validating against them would have been
    circular. The window continues past the buffer (no terminated flags),
    so V(s_T)=last_value is nonzero and the bug -- had it still been
    present -- would show up as a large, non-vanishing error.
    """
    torch.manual_seed(123)
    num_envs, seq_len = 16, 37
    gamma = 0.97

    rewards     = torch.randn(num_envs, seq_len, device="cuda")
    values      = torch.randn(num_envs, seq_len, device="cuda")
    terminateds = torch.zeros(num_envs, seq_len, device="cuda")  # continuing episode
    last_value  = torch.randn(num_envs, device="cuda")           # V(s_T), nonzero

    # G_t = r_t + gamma*G_{t+1},  G_T = last_value  (independent hand-rolled MC return)
    returns = torch.zeros(num_envs, seq_len, device="cuda")
    carry = last_value.clone()
    for t in reversed(range(seq_len)):
        carry = rewards[:, t] + gamma * carry
        returns[:, t] = carry
    expected_advantages = returns - values

    actual = compute_gae(rewards, values, terminateds,
                          gamma=gamma, lambda_=1.0, last_value=last_value)
    torch.testing.assert_close(actual, expected_advantages, atol=1e-4, rtol=1e-4)


@cuda_only
def test_gae_lambda1_monte_carlo_identity_interior_truncation():
    """Same identity, but with an interior truncation -- pins that path unchanged.

    Interior truncations were already correct (bootstrap enters delta only,
    never the scan carry) and must stay that way. The episode is split into
    two independent segments by a truncation at t=2; the hand-rolled MC
    return restarts from the truncation's bootstrap value there, and the
    window still continues past the buffer with a second, different nonzero
    bootstrap at t=T-1.
    """
    torch.manual_seed(124)
    num_envs, seq_len = 8, 6
    gamma = 0.95

    rewards     = torch.randn(num_envs, seq_len, device="cuda")
    values      = torch.randn(num_envs, seq_len, device="cuda")
    terminateds = torch.zeros(num_envs, seq_len, device="cuda")
    truncateds  = torch.zeros(num_envs, seq_len, device="cuda")
    truncateds[:, 2] = 1.0

    bootstrap_values = torch.zeros(num_envs, seq_len, device="cuda")
    bootstrap_values[:, 2]  = torch.randn(num_envs, device="cuda")   # V(s_3^true) at the truncation
    bootstrap_values[:, -1] = torch.randn(num_envs, device="cuda")   # V(s_T), window continues

    # Hand-rolled MC return, restarting the carry at the truncation boundary.
    returns = torch.zeros(num_envs, seq_len, device="cuda")
    carry = bootstrap_values[:, -1].clone()
    for t in reversed(range(seq_len)):
        if t == 2:
            carry = rewards[:, t] + gamma * bootstrap_values[:, t]
        else:
            carry = rewards[:, t] + gamma * carry
        returns[:, t] = carry
    expected_advantages = returns - values

    actual = compute_gae(rewards, values, terminateds, truncateds,
                          gamma=gamma, lambda_=1.0, bootstrap_values=bootstrap_values)
    torch.testing.assert_close(actual, expected_advantages, atol=1e-4, rtol=1e-4)


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
    """Pure step-by-step Python loop -- ground truth for per-step bootstrap_values.

    Handles interior truncated steps (bootstrap_values[n, t] used as v_next when
    truncateds[n, t]=1) in addition to the window boundary (t=T-1).

    A[T] = 0: bootstrap_values[n, T-1] already enters the recurrence once,
    inside delta[T-1] as v_next (weight 1). Seeding the carry with it too
    would double-count it.
    """
    N, T = rewards.shape
    out = torch.zeros_like(rewards)
    for n in range(N):
        carry = 0.0   # A[T] = 0
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

    Exercises the public bootstrap_values API directly -- values at truncated steps
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

      triton              -- Triton kernel  (CUDA events)
      compile(cumsum)     -- torch.compile on vectorized_gae (log-space cumsum)  (CUDA events)
      compile(assoc_scan) -- torch.compile on vectorized_gae_with_truncations
                            called with zero truncateds  (CUDA events)
      compile(loop)       -- torch.compile on reference_gae loop  (wall-clock)
      np->triton->np      -- NumPy→GPU→NumPy adoption path  (wall-clock)
      numpy(cpu)          -- CPU Python loop  (wall-clock)

    Scope: NO-TRUNCATION PATH ONLY.  _make_inputs produces no truncateds, so
    the kernel dispatches to HAS_TRUNCATIONS=False.

    Two baselines are reported for the no-truncation path:
      (a) compile(cumsum):     the specialized fast baseline that only works
          when there are no truncations (log(0)=-inf breaks it otherwise).
      (b) compile(assoc_scan): the general-purpose baseline -- the same
          vectorized_gae_with_truncations function used in the truncation
          benchmark, called with zero truncateds and bootstrap_values nonzero
          only at the boundary column.  A real engineer supporting both
          truncated and non-truncated episodes would maintain one
          implementation, not two.  This baseline uses the same associative
          scan for both regimes, making the no-truncation and truncation
          benchmark methodologies consistent.

    Assertions:
      - triton >=1.5x faster than compile(cumsum).
      - np->triton->np >=1.5x faster than numpy(cpu).
    """
    compiled_cumsum     = torch.compile(vectorized_gae)
    compiled_assoc_scan = torch.compile(vectorized_gae_with_truncations)
    compiled_loop       = torch.compile(reference_gae)

    _args = _make_inputs(64, 512)
    _N, _T = _args[0].shape
    _trunc0 = torch.zeros(_N, _T, device="cuda")
    _bsv0   = torch.zeros(_N, _T, device="cuda")
    _bsv0[:, -1] = torch.rand(_N, device="cuda")

    # Trigger torch.compile tracing at one shape; per-config warmup below
    # handles each distinct BLOCK_SIZE (seq_len → power-of-2) before timing.
    compiled_cumsum(*_args, gamma=0.99, lambda_=0.95)
    compiled_assoc_scan(_args[0], _args[1], _args[2], _trunc0, _bsv0, 0.99, 0.95)
    compiled_loop(*_args, gamma=0.99, lambda_=0.95)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'compile(cumsum)':>16} {'compile(assoc)':>15} {'compile(loop)':>15} "
        f"{'np->tri->np':>13} {'numpy(cpu)':>12} "
        f"{'vs cumsum':>11} {'vs assoc':>10} {'vs loop':>9} {'np->tri->np vs numpy':>22}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_cumsum = []
    all_speedups_e2e    = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_np  = _make_inputs_np(num_envs, seq_len)
        n_iter   = _n_iter_gpu(seq_len, num_envs)

        # Build zero-truncation inputs for the assoc-scan baseline.
        trunc0 = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0   = torch.zeros(num_envs, seq_len, device="cuda")
        bsv0[:, -1] = torch.rand(num_envs, device="cuda")

        # Per-config warmup at the exact shape being timed -- each distinct
        # seq_len triggers a fresh Triton compile (new BLOCK_SIZE power-of-2).
        _warmup_gpu(compute_gae, *args_gpu, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_cumsum, *args_gpu, gamma=0.99, lambda_=0.95)
        _warmup_gpu(compiled_assoc_scan,
                    args_gpu[0], args_gpu[1], args_gpu[2], trunc0, bsv0, 0.99, 0.95)

        tri_ms    = _bench_gpu(compute_gae, *args_gpu,
                               gamma=0.99, lambda_=0.95, n_iter=n_iter)
        cum_ms    = _bench_gpu(compiled_cumsum, *args_gpu,
                               gamma=0.99, lambda_=0.95, n_iter=n_iter)
        asc_ms    = _bench_gpu(compiled_assoc_scan,
                               args_gpu[0], args_gpu[1], args_gpu[2], trunc0, bsv0, 0.99, 0.95,
                               n_iter=n_iter)
        loop_ms   = _bench_cpu(compiled_loop, *args_gpu, gamma=0.99, lambda_=0.95)
        np_tri_ms = _bench_cpu(numpy_to_triton_to_numpy, *args_np, gamma=0.99, lambda_=0.95)
        numpy_ms  = _bench_cpu(numpy_gae, *args_gpu, gamma=0.99, lambda_=0.95)

        su_cumsum = cum_ms  / tri_ms
        su_assoc  = asc_ms  / tri_ms
        su_loop   = loop_ms / tri_ms
        su_e2e    = numpy_ms / np_tri_ms
        all_speedups_cumsum.append(su_cumsum)
        all_speedups_e2e.append(su_e2e)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{tri_ms:>7.3f}ms {cum_ms:>15.3f}ms {asc_ms:>14.3f}ms {loop_ms:>14.3f}ms "
            f"{np_tri_ms:>12.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_cumsum:>9.2f}x {su_assoc:>9.2f}x {su_loop:>7.1f}x {su_e2e:>20.1f}x"
        )

    print(
        "\ntriton          : CUDA events -- pure kernel time."
        "\ncompile(cumsum) : CUDA events -- log-space suffix cumsum; specialized for"
        "\n                  no-truncation only (log(0)=-inf breaks it with truncateds)."
        "\ncompile(assoc)  : CUDA events -- vectorized_gae_with_truncations, zero truncateds;"
        "\n                  same function as the truncation benchmark -- consistent baseline."
        "\ncompile(loop)   : wall-clock -- one CUDA op per timestep from Python;"
        "\n                  CUDA events would miss the CPU stall."
        "\nnp->tri->np     : wall-clock -- NumPy→GPU→NumPy, realistic adoption path."
        "\nnumpy(cpu)      : wall-clock -- plain Python loop on CPU tensors."
        "\nspeedups are baseline_ms / triton_ms (higher = triton is faster)."
    )

    assert min(all_speedups_cumsum) >= 1.5, (
        f"Expected >=1.5x speedup over compile(cumsum) across all configs, "
        f"worst was {min(all_speedups_cumsum):.2f}x"
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
    Fully vectorized GAE -- strong compiled baseline. Thin wrapper around
    vectorized_gae_with_truncations (truncateds=0), which uses parallel_suffix_scan
    (linear-space, log2(T)-doubling, no log/exp) instead of a log-space suffix
    cumsum.

    This function used to compute a log-space suffix product directly
    (log(decay) suffix-summed, then exp()'d) and was BROKEN: at this project's
    real termination rate (~5%/step), any window with 2+ termination events --
    which is nearly guaranteed at every seq_len actually benchmarked, not just
    long ones -- pushes the suffix log-sum past float32's underflow floor
    (each termination contributes log(1e-38)~=-87.5 via the done-boundary
    clamp; two of them already exceed the ~-103 underflow threshold). Measured
    directly: 90-99% of output elements were inf/nan at every size in
    benchmarks.md (64x512 through 16384x512), silently, because this baseline's
    own output was never correctness/finiteness-checked anywhere -- only timed.
    vectorized_gae_with_truncations doesn't have this failure mode (verified
    finite up to seq_len=524288) and, called with truncateds=0, matches the
    sequential reference to float32 precision (~2e-6 max abs diff) -- it is the
    correct baseline for both regimes, so there is no reason to maintain two
    implementations. bootstrap_values here accepts the old [num_envs]-shaped
    "final column only" convenience form (unused by any caller in this repo,
    kept for signature compatibility) and is expanded into the [num_envs,
    seq_len] boundary-column form vectorized_gae_with_truncations expects.
    """
    truncateds = torch.zeros_like(terminateds)
    full_bootstrap = torch.zeros_like(rewards)
    if bootstrap_values is not None:
        full_bootstrap[:, -1] = bootstrap_values
    return vectorized_gae_with_truncations(
        rewards, values, terminateds, truncateds, full_bootstrap, gamma, lambda_
    )



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
    Vectorized GAE with truncation support -- log-depth parallel scan baseline.

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
      - beta[t] = gamma*lambda*(1 - clamp(terminated[t]+truncated[t], max=1))
        (decay[t] below -- the scan's multiplicative decay coefficient)
      - additive boundary carry A[T] = 0 (see kernels/gae.py's module
        docstring: bootstrap_values[:, -1] already enters delta[T-1] above
        at weight 1 via next_values; seeding the scan's boundary carry with
        it too would double-count it). beta[t] itself is unaffected by this.
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

    # parallel_suffix_scan assumes A[T]=0, which is exactly what we want here --
    # no sentinel/boundary injection needed.
    return parallel_suffix_scan(deltas, decays)


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

    No assertion floor is imposed here -- the truncation path is a correctness
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

    # Initial compile trigger at one shape; per-config warmup below handles
    # each distinct BLOCK_SIZE before timing.
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
        args   = _make_trunc_inputs(num_envs, seq_len)
        n_iter = _n_iter_gpu(seq_len, num_envs)

        # Per-config warmup at the exact shape being timed.
        _warmup_gpu(compute_gae, args[0], args[1], args[2], args[3],
                    gamma=0.99, lambda_=0.95, bootstrap_values=args[4])
        _warmup_gpu(compiled_vec_trunc, *args, gamma=0.99, lambda_=0.95)

        tri_ms = _bench_gpu(
            compute_gae, args[0], args[1], args[2], args[3],
            gamma=0.99, lambda_=0.95, bootstrap_values=args[4],
            n_iter=n_iter,
        )
        vec_ms = _bench_gpu(
            compiled_vec_trunc, *args,
            gamma=0.99, lambda_=0.95,
            n_iter=n_iter,
        )

        su = vec_ms / tri_ms
        all_speedups.append(su)
        print(f"{num_envs:>10} {seq_len:>8} {tri_ms:>7.3f}ms {vec_ms:>19.3f}ms {su:>8.2f}x")

    print(
        f"\nMin speedup: {min(all_speedups):.2f}x  "
        f"Max: {max(all_speedups):.2f}x  "
        f"(HAS_TRUNCATIONS=True path; 7 full-width reads vs 5 for no-truncation path)"
    )
