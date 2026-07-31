import numpy as np
import pytest
import torch

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu, _warmup_gpu, parallel_suffix_scan

# numpy is used intentionally: episode data is typically stored as NumPy arrays,
# so numpy_to_triton_to_numpy mirrors the real adoption path (np -> GPU -> np)
# a user would take to swap in the Triton kernel.

triton = pytest.importorskip("triton")

from rl_triton.ops.vtrace import compute_vtrace
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_next_values(
    values: torch.Tensor,
    bootstrap_values: torch.Tensor | None,
) -> torch.Tensor:
    """Construct next_values from values by shifting: next_values[t] = values[t+1]."""
    nv = torch.empty_like(values)
    nv[:, :-1] = values[:, 1:]
    nv[:, -1]  = 0.0 if bootstrap_values is None else bootstrap_values
    return nv


def _make_inputs(num_envs, seq_len, device="cuda", seed=0):
    torch.manual_seed(seed)
    log_pi_target   = -torch.rand(num_envs, seq_len, device=device)
    log_pi_behavior = -torch.rand(num_envs, seq_len, device=device)
    values          = torch.randn(num_envs, seq_len, device=device)
    rewards         = torch.randn(num_envs, seq_len, device=device)
    terminateds           = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return log_pi_target, log_pi_behavior, values, rewards, terminateds


def _make_inputs_np(num_envs, seq_len, seed=0):
    """Return the same random inputs as _make_inputs but as NumPy arrays (CPU)."""
    args_gpu = _make_inputs(num_envs, seq_len, device="cpu", seed=seed)
    return tuple(t.numpy() for t in args_gpu)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def reference_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch V-Trace backward scan -- ground truth for correctness tests.

    Derives next_values as values[:, t+1] with bootstrap_values at the boundary,
    matching exactly what the Triton kernel computes internally.
    """
    next_values = _make_next_values(values, bootstrap_values)
    num_envs, T = rewards.shape

    is_ratios = torch.exp(log_pi_target - log_pi_behavior)
    rho = torch.clamp(is_ratios, max=rho_bar)
    c   = torch.clamp(is_ratios, max=c_bar)

    deltas = rho * (rewards + gamma * next_values * (1.0 - terminateds) - values)
    decays = gamma * c * (1.0 - terminateds)

    value_deltas = torch.zeros_like(rewards)
    carry = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry = deltas[:, t] + decays[:, t] * carry
        value_deltas[:, t] = carry

    vtrace_targets = value_deltas + values

    # next_vtrace_target[t] = targets[t+1]; at t=T-1 use bootstrap as V(s_T).
    next_vtrace_targets = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1] = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]  = 0.0 if bootstrap_values is None else bootstrap_values

    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * (1.0 - terminateds) - values)
    return vtrace_targets, vtrace_advantages


# ---------------------------------------------------------------------------
# CPU numpy baseline and end-to-end adoption path
# ---------------------------------------------------------------------------

def numpy_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU V-Trace backward loop -- plain Python loop on CPU tensors."""
    cpu = lambda t: t.cpu().float()
    log_pi_target, log_pi_behavior = cpu(log_pi_target), cpu(log_pi_behavior)
    values, rewards, terminateds = cpu(values), cpu(rewards), cpu(terminateds)

    next_values = _make_next_values(values, None)
    num_envs, T = rewards.shape

    is_ratios = torch.exp(log_pi_target - log_pi_behavior)
    rho       = torch.clamp(is_ratios, max=rho_bar)
    c         = torch.clamp(is_ratios, max=c_bar)
    discounts = gamma * (1.0 - terminateds)

    deltas = rho * (rewards + discounts * next_values - values)

    value_deltas = torch.zeros_like(rewards)
    carry        = torch.zeros(num_envs)
    for t in reversed(range(T)):
        carry              = deltas[:, t] + discounts[:, t] * c[:, t] * carry
        value_deltas[:, t] = carry

    vtrace_targets = value_deltas + values

    next_vtrace_targets          = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1]  = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]   = next_values[:, -1]   # = 0 (bootstrap=None)

    vtrace_advantages = rho * (rewards + discounts * next_vtrace_targets - values)
    return vtrace_targets, vtrace_advantages


def numpy_to_triton_to_numpy(
    log_pi_target_np: np.ndarray,
    log_pi_behavior_np: np.ndarray,
    values_np: np.ndarray,
    rewards_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy → GPU Triton kernel → NumPy end-to-end adoption path."""
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch.float32)
    vtrace_targets, vtrace_advantages = compute_vtrace(
        to_gpu(log_pi_target_np), to_gpu(log_pi_behavior_np),
        to_gpu(values_np), to_gpu(rewards_np), to_gpu(dones_np),
        gamma=gamma, rho_bar=rho_bar, c_bar=c_bar,
    )
    torch.cuda.synchronize()
    return vtrace_targets.cpu().numpy(), vtrace_advantages.cpu().numpy()


# ---------------------------------------------------------------------------
# Ground-truth value tests
# ---------------------------------------------------------------------------
#
# Hand-computed with rho_bar=c_bar=100 (no clipping) so IS ratios pass through.
# All tests use values=zeros so V(s_{t+1}) = values[t+1] = 0 for t < T-1, and
# bootstrap (default 0) for t = T-1.

@cuda_only
def test_vtrace_known_values_single_env():
    # log_pi equal -> is_ratio=1, rho=c=1. gamma=1, V=0, r=1, no terminateds.
    # next_values = [values[1], bootstrap] = [0, 0]
    # delta[t] = 1*(1 + 1*0 - 0) = 1,  decay[t] = 1
    # Δ[1] = 1,  Δ[0] = 1 + 1*1 = 2
    # targets = [2, 1]
    # next_target = [targets[1], bootstrap] = [1, 0]
    # adv[0] = 1*(1 + 1*1 - 0) = 2
    # adv[1] = 1*(1 + 1*0 - 0) = 1
    log_pi  = torch.zeros(1, 2, device="cuda")
    values  = torch.zeros(1, 2, device="cuda")
    rewards = torch.ones(1, 2, device="cuda")
    terminateds   = torch.zeros(1, 2, device="cuda")

    targets, advantages = compute_vtrace(
        log_pi, log_pi, values, rewards, terminateds,
        gamma=1.0, rho_bar=100.0, c_bar=100.0,
    )
    torch.testing.assert_close(targets,    torch.tensor([[2.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(advantages, torch.tensor([[2.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_known_values_truncated():
    # Same but bootstrap=0.5. V=0, r=1, no terminateds, rho=c=1, gamma=1.
    # next_values = [values[1], bootstrap] = [0, 0.5]
    # delta[0] = 1*(1 + 0 - 0) = 1
    # delta[1] = 1*(1 + 1*0.5 - 0) = 1.5
    # Δ[1] = 1.5 + 1*0.5 = 2.0
    # Δ[0] = 1.0 + 1*2.0 = 3.0
    # targets = [3.0, 2.0]
    # next_target = [targets[1], bootstrap] = [2.0, 0.5]
    # adv[0] = 1*(1 + 1*2.0 - 0) = 3.0
    # adv[1] = 1*(1 + 1*0.5 - 0) = 1.5
    log_pi    = torch.zeros(1, 2, device="cuda")
    values    = torch.zeros(1, 2, device="cuda")
    rewards   = torch.ones(1, 2, device="cuda")
    terminateds     = torch.zeros(1, 2, device="cuda")
    bootstrap = torch.tensor([0.5], device="cuda")

    targets, advantages = compute_vtrace(
        log_pi, log_pi, values, rewards, terminateds,
        gamma=1.0, rho_bar=100.0, c_bar=100.0,
        last_value=bootstrap,
    )
    torch.testing.assert_close(targets,    torch.tensor([[3.0, 2.0]], device="cuda"), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(advantages, torch.tensor([[3.0, 1.5]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_known_values_done():
    # done[0]=1: not_done[0]=0 zeros out delta[0]'s next_value term and decay[0].
    # rho=c=1, gamma=1, V=0, r=1.
    # next_values = [values[1], bootstrap] = [0, 0]
    # delta[0] = 1*(1 + 1*0*(1-1) - 0) = 1   (next_value term killed by not_done)
    # delta[1] = 1*(1 + 1*0*(1-0) - 0) = 1
    # decay[0] = 1*1*(1-1) = 0                (carry reset at boundary)
    # decay[1] = 1*1*(1-0) = 1
    # Δ[1] = 1 + 1*0 = 1
    # Δ[0] = 1 + 0*1 = 1
    # targets = [1, 1]
    log_pi  = torch.zeros(1, 2, device="cuda")
    values  = torch.zeros(1, 2, device="cuda")
    rewards = torch.ones(1, 2, device="cuda")
    terminateds   = torch.tensor([[1.0, 0.0]], device="cuda")

    targets, _ = compute_vtrace(
        log_pi, log_pi, values, rewards, terminateds,
        gamma=1.0, rho_bar=100.0, c_bar=100.0,
    )
    torch.testing.assert_close(targets, torch.tensor([[1.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_known_values_clipping():
    # IS ratio = exp(0 - log(2)) = 0.5, rho_bar=c_bar=1 -> rho=c=0.5.
    # gamma=1, V=0, r=1, no terminateds. next_values = [0, 0].
    # delta[t] = 0.5*(1+0-0) = 0.5,  decay[t] = 1*0.5*1 = 0.5
    # Δ[1] = 0.5,  Δ[0] = 0.5 + 0.5*0.5 = 0.75
    # targets = [0.75, 0.5]
    log_pi_t = torch.zeros(1, 2, device="cuda")
    log_pi_b = torch.full((1, 2), torch.log(torch.tensor(2.0)).item(), device="cuda")
    values   = torch.zeros(1, 2, device="cuda")
    rewards  = torch.ones(1, 2, device="cuda")
    terminateds    = torch.zeros(1, 2, device="cuda")

    targets, _ = compute_vtrace(
        log_pi_t, log_pi_b, values, rewards, terminateds,
        gamma=1.0, rho_bar=1.0, c_bar=1.0,
    )
    torch.testing.assert_close(targets, torch.tensor([[0.75, 0.5]], device="cuda"), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Correctness vs reference
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),
    (4,   128),
    (32,  333),
    (128, 1024),
    (256, 2048),
])
def test_vtrace_correctness_shapes(num_envs, seq_len):
    args = _make_inputs(num_envs, seq_len, seed=42)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_correctness_bootstrap():
    args = _make_inputs(32, 512, seed=3)
    bootstrap = torch.rand(32, device="cuda")
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    act_t, act_a = compute_vtrace(*args, gamma=0.99, last_value=bootstrap)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_correctness_mixed_termination():
    args = _make_inputs(16, 256, seed=4)
    bootstrap = torch.cat([
        torch.zeros(8, device="cuda"),
        torch.rand(8, device="cuda"),
    ])
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    act_t, act_a = compute_vtrace(*args, gamma=0.99, last_value=bootstrap)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_non_contiguous_input():
    torch.manual_seed(5)
    base    = torch.randn(32, 512, 2, device="cuda")
    log_pi_t = base[..., 0]
    log_pi_b = base[..., 1]
    values   = torch.randn(32, 512, 2, device="cuda")[..., 0]
    rewards  = torch.randn(32, 512, 2, device="cuda")[..., 0]
    terminateds    = torch.zeros(32, 512, 2, device="cuda")[..., 0]

    exp_t, exp_a = reference_vtrace(
        log_pi_t.contiguous(), log_pi_b.contiguous(),
        values.contiguous(), rewards.contiguous(), terminateds.contiguous(),
        gamma=0.99,
    )
    act_t, act_a = compute_vtrace(
        log_pi_t, log_pi_b, values, rewards, terminateds, gamma=0.99,
    )
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fused kernel correctness
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),
    (4,   128),
    (32,  512),
    (128, 1024),
])
def test_vtrace_fused_correctness(num_envs, seq_len):
    args = _make_inputs(num_envs, seq_len, seed=20)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace_fused(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_fused_bootstrap():
    args = _make_inputs(32, 512, seed=21)
    bootstrap = torch.rand(32, device="cuda")                               # shape [num_envs]
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    act_t, act_a = compute_vtrace_fused(*args, gamma=0.99, last_value=bootstrap)  # [num_envs] → last_value
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Sequential reference for full [num_envs, seq_len] bootstrap_values
# ---------------------------------------------------------------------------

def _ref_vtrace_sequential(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure step-by-step Python loop -- ground truth for per-step bootstrap_values.

    Handles interior truncated steps (bootstrap_values[n, t] used as v_next when
    truncateds[n, t]=1) in addition to the window boundary (t=T-1).
    """
    N, T = rewards.shape
    is_ratios = torch.exp(log_pi_target - log_pi_behavior)
    rho = torch.clamp(is_ratios, max=rho_bar)
    c   = torch.clamp(is_ratios, max=c_bar)

    value_deltas = torch.zeros_like(rewards)
    carry = bootstrap_values[:, T - 1].clone()   # Δ[T] = boundary bootstrap

    for t in reversed(range(T)):
        v_next = torch.where(
            (truncateds[:, t] == 1.0) | torch.tensor(t == T - 1),
            bootstrap_values[:, t],
            values[:, t + 1] if t < T - 1 else bootstrap_values[:, t],
        )
        not_terminated = 1.0 - terminateds[:, t]
        done = (terminateds[:, t] + truncateds[:, t]).clamp(max=1.0)
        delta = rho[:, t] * (rewards[:, t] + gamma * v_next * not_terminated - values[:, t])
        carry = delta + gamma * c[:, t] * (1.0 - done) * carry
        value_deltas[:, t] = carry

    vtrace_targets = value_deltas + values

    next_vtrace_targets = torch.empty_like(vtrace_targets)
    for t in range(T):
        if t == T - 1:
            next_vtrace_targets[:, t] = bootstrap_values[:, t]
        else:
            next_vtrace_targets[:, t] = torch.where(
                truncateds[:, t] == 1.0,
                bootstrap_values[:, t],
                vtrace_targets[:, t + 1],
            )

    not_terminated = 1.0 - terminateds
    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * not_terminated - values)
    return vtrace_targets, vtrace_advantages


# ---------------------------------------------------------------------------
# bootstrap_values / last_value API tests
# ---------------------------------------------------------------------------

@cuda_only
def test_vtrace_two_interior_truncations():
    """Two interior truncated episodes + continuing window boundary.

    Verified against a pure sequential Python reference loop.
    Uses the full bootstrap_values=[num_envs, seq_len] API directly.
    """
    torch.manual_seed(99)
    N, T = 2, 12
    log_pi_target   = -torch.rand(N, T, device="cuda")
    log_pi_behavior = -torch.rand(N, T, device="cuda")
    values      = torch.rand(N, T, device="cuda")
    rewards     = torch.rand(N, T, device="cuda")
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

    exp_t, exp_a = _ref_vtrace_sequential(
        log_pi_target.cpu(), log_pi_behavior.cpu(),
        values.cpu(), rewards.cpu(), terminateds.cpu(), truncateds.cpu(),
        bootstrap_values.cpu(), gamma=0.99,
    )
    act_t, act_a = compute_vtrace(
        log_pi_target, log_pi_behavior, values, rewards, terminateds, truncateds,
        gamma=0.99, bootstrap_values=bootstrap_values,
    )
    torch.testing.assert_close(act_t, exp_t.cuda(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(act_a, exp_a.cuda(), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_fused_two_interior_truncations():
    """HAS_TRUNCATIONS=True kernel path: two interior truncated episodes + boundary.

    Exercises compute_vtrace_fused directly with truncateds and a full
    [num_envs, seq_len] bootstrap_values tensor.  T=12, truncations at
    t=3 and t=7 in env 0, t=5 in env 1.  Boundary continues (non-zero
    bootstrap at t=11 for env 0).  Compared against _ref_vtrace_sequential.
    """
    torch.manual_seed(77)
    N, T = 2, 12
    log_pi_target   = -torch.rand(N, T, device="cuda")
    log_pi_behavior = -torch.rand(N, T, device="cuda")
    values      = torch.rand(N, T, device="cuda")
    rewards     = torch.rand(N, T, device="cuda")
    terminateds = torch.zeros(N, T, device="cuda")
    truncateds  = torch.zeros(N, T, device="cuda")

    # env 0: truncations at t=3 and t=7; window continues past t=11
    truncateds[0, 3] = 1.0
    truncateds[0, 7] = 1.0
    # env 1: single truncation at t=5; window terminates at t=11
    truncateds[1, 5] = 1.0

    bootstrap_values = torch.zeros(N, T, device="cuda")
    bootstrap_values[0, 3]  = 2.5   # V(s_{t=4}^true) for env 0
    bootstrap_values[0, 7]  = 1.8   # V(s_{t=8}^true) for env 0
    bootstrap_values[0, 11] = 3.2   # V(s_T), env 0 window continues
    bootstrap_values[1, 5]  = 0.9   # V(s_{t=6}^true) for env 1
    # bootstrap_values[1, 11] stays 0: env 1 terminates at t=11

    exp_t, exp_a = _ref_vtrace_sequential(
        log_pi_target.cpu(), log_pi_behavior.cpu(),
        values.cpu(), rewards.cpu(), terminateds.cpu(), truncateds.cpu(),
        bootstrap_values.cpu(), gamma=0.99,
    )
    act_t, act_a = compute_vtrace_fused(
        log_pi_target, log_pi_behavior, values, rewards, terminateds,
        truncateds=truncateds, gamma=0.99, bootstrap_values=bootstrap_values,
    )
    torch.testing.assert_close(act_t, exp_t.cuda(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cuda(), atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_bootstrap_values_full_tensor():
    """Passing a full [num_envs, seq_len] bootstrap_values tensor (no last_value).

    Exercises the public bootstrap_values API directly -- values at truncated steps
    plus a non-zero boundary column for a continuing episode.
    """
    torch.manual_seed(42)
    N, T = 8, 64
    args = _make_inputs(N, T, seed=42)
    log_pi_target, log_pi_behavior, values, rewards, terminateds = args
    truncateds = (torch.rand(N, T, device="cuda") < 0.05).float()
    truncateds = truncateds * (1.0 - terminateds)  # mutually exclusive

    bootstrap_values = torch.zeros(N, T, device="cuda")
    bootstrap_values[truncateds.bool()] = torch.rand(int(truncateds.sum().item()), device="cuda")
    bootstrap_values[:, -1] = torch.rand(N, device="cuda")

    exp_t, exp_a = _ref_vtrace_sequential(
        log_pi_target.cpu(), log_pi_behavior.cpu(),
        values.cpu(), rewards.cpu(), terminateds.cpu(), truncateds.cpu(),
        bootstrap_values.cpu(), gamma=0.99,
    )
    act_t, act_a = compute_vtrace(
        log_pi_target, log_pi_behavior, values, rewards, terminateds, truncateds,
        gamma=0.99, bootstrap_values=bootstrap_values,
    )
    torch.testing.assert_close(act_t, exp_t.cuda(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cuda(), atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_last_value_mutual_exclusion():
    """Passing both last_value and bootstrap_values must raise."""
    N, T = 4, 16
    log_pi = -torch.rand(N, T, device="cuda")
    values  = torch.rand(N, T, device="cuda")
    rewards = torch.rand(N, T, device="cuda")
    terminateds = torch.zeros(N, T, device="cuda")
    bv = torch.zeros(N, T, device="cuda")
    lv = torch.rand(N, device="cuda")
    with pytest.raises(AssertionError, match="not both"):
        compute_vtrace(log_pi, log_pi, values, rewards, terminateds,
                       bootstrap_values=bv, last_value=lv)


@cuda_only
def test_vtrace_last_value_equivalence():
    """last_value=[num_envs] produces identical output to the equivalent
    hand-built bootstrap_values[:, -1] tensor."""
    torch.manual_seed(7)
    N, T = 8, 64
    args = _make_inputs(N, T, seed=7)
    log_pi_target, log_pi_behavior, values, rewards, terminateds = args
    lv = torch.rand(N, device="cuda")

    act_t, act_a = compute_vtrace(
        log_pi_target, log_pi_behavior, values, rewards, terminateds,
        gamma=0.99, last_value=lv,
    )
    bv = torch.zeros(N, T, device="cuda")
    bv[:, -1] = lv
    exp_t, exp_a = compute_vtrace(
        log_pi_target, log_pi_behavior, values, rewards, terminateds,
        gamma=0.99, bootstrap_values=bv,
    )
    torch.testing.assert_close(act_t, exp_t, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(act_a, exp_a, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# numpy baseline correctness
# ---------------------------------------------------------------------------

@cuda_only
def test_numpy_vtrace_matches_reference():
    args = _make_inputs(32, 256, device="cpu", seed=10)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = numpy_vtrace(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cpu(), atol=1e-4, rtol=1e-4)


@cuda_only
def test_numpy_to_triton_to_numpy_matches_reference():
    args_np  = _make_inputs_np(32, 256, seed=11)
    args_gpu = tuple(torch.from_numpy(a).cuda() for a in args_np)
    exp_t, exp_a = reference_vtrace(*args_gpu, gamma=0.99)

    act_t_np, act_a_np = numpy_to_triton_to_numpy(*args_np, gamma=0.99)
    torch.testing.assert_close(torch.from_numpy(act_t_np), exp_t.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.from_numpy(act_a_np), exp_a.cpu(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Vectorized PyTorch baseline (benchmark only)
# ---------------------------------------------------------------------------

def vectorized_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully vectorized V-Trace -- strong compiled baseline. Thin wrapper around
    vectorized_vtrace_with_truncations (truncateds=0) -- see vectorized_gae's
    docstring (test_gae.py) for why: the log-space suffix-cumsum formula this
    used to compute directly was broken (90%+ non-finite output at every size
    actually benchmarked, never checked), and parallel_suffix_scan isn't.
    bootstrap_values here keeps the old [num_envs]-shaped "final column only"
    convenience form for signature compatibility.
    """
    truncateds = torch.zeros_like(terminateds)
    full_bootstrap = torch.zeros_like(rewards)
    if bootstrap_values is not None:
        full_bootstrap[:, -1] = bootstrap_values
    return vectorized_vtrace_with_truncations(
        log_pi_target, log_pi_behavior, values, rewards, terminateds, truncateds,
        full_bootstrap, gamma, rho_bar=rho_bar, c_bar=c_bar,
    )


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
def test_vtrace_performance():
    """
    Sweep over (num_envs, seq_len) configs comparing:

      fused              -- single Triton kernel  (CUDA events)
      pt.compile(vec)    -- torch.compile on vectorized_vtrace  (CUDA events)
      pt.compile(loop)   -- torch.compile on reference_vtrace loop  (wall-clock)
      np->triton->np     -- NumPy->GPU->NumPy adoption path  (wall-clock)
      numpy(cpu)         -- CPU Python loop  (wall-clock)

    Assertions:
      - fused must be >=1.5x faster than pt.compile(vec).
      - np->triton->np must be >=1.5x faster than numpy(cpu).
    """
    compiled_vec  = torch.compile(vectorized_vtrace)
    compiled_loop = torch.compile(reference_vtrace)

    _args = _make_inputs(64, 512)
    compiled_vec(*_args, gamma=0.99)
    compiled_loop(*_args, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'fused':>8} {'compile(vec)':>14} {'compile(loop)':>15} "
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
        n_iter   = _n_iter_gpu(seq_len, num_envs)

        # Per-config warmup at the exact shape being timed.
        _warmup_gpu(compute_vtrace_fused, *args_gpu, gamma=0.99)
        _warmup_gpu(compiled_vec,         *args_gpu, gamma=0.99)

        fused_ms  = _bench_gpu(compute_vtrace_fused,     *args_gpu, gamma=0.99, n_iter=n_iter)
        vec_ms    = _bench_gpu(compiled_vec,             *args_gpu, gamma=0.99, n_iter=n_iter)
        loop_ms   = _bench_cpu(compiled_loop,            *args_gpu, gamma=0.99)
        np_tri_ms = _bench_cpu(numpy_to_triton_to_numpy, *args_np,  gamma=0.99)
        numpy_ms  = _bench_cpu(numpy_vtrace,             *args_gpu, gamma=0.99)

        su_vec  = vec_ms   / fused_ms
        su_loop = loop_ms  / fused_ms
        su_e2e  = numpy_ms / np_tri_ms
        all_speedups_vec.append(su_vec)
        all_speedups_e2e.append(su_e2e)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{fused_ms:>7.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms "
            f"{np_tri_ms:>12.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_e2e:>20.1f}x"
        )

    print(
        "\nfused            : CUDA events -- single kernel, IS ratios + scan + targets fused."
        "\ncompile(vec)     : CUDA events -- vectorized log-space cumsum, no Python loop."
        "\ncompile(loop)    : wall-clock  -- one CUDA op per timestep from Python;"
        "\n                   CUDA events would miss the CPU stall."
        "\nnp->tri->np      : wall-clock  -- NumPy->GPU->NumPy, realistic adoption path."
        "\nnumpy(cpu)       : wall-clock  -- pure CPU Python loop."
        "\nspeedups vs fused kernel."
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
# Vectorized baseline with truncation support (benchmark only)
# ---------------------------------------------------------------------------

def vectorized_vtrace_with_truncations(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized V-Trace with truncation support -- log-depth parallel scan baseline.

    Uses parallel_suffix_scan (the same associative operator as tl.associative_scan
    in the kernel) to compute the value-delta suffix scan in O(log2(T)) passes,
    fully vectorized.  No Python time loop; torch.compile works cleanly.

    Matches the kernel semantics exactly:
      - v_next[t] = values[t+1] at interior non-truncated steps,
        bootstrap_values[t] at truncated steps and the boundary.
      - not_terminated[t] gates the one-step TD: gamma * v_next * (1 - terminated)
      - not_done[t] = 1 - clamp(terminated + truncated, max=1) gates trace decay
      - carry seeded from bootstrap_values[:, -1] (boundary value).
    """
    is_ratios      = torch.exp(log_pi_target - log_pi_behavior)
    rho            = torch.clamp(is_ratios, max=rho_bar)
    c              = torch.clamp(is_ratios, max=c_bar)
    not_terminated = 1.0 - terminateds
    not_done       = 1.0 - (terminateds + truncateds).clamp(max=1.0)

    # v_next[t]: values[t+1] at interior non-truncated steps, bootstrap elsewhere.
    v_next_raw         = torch.empty_like(values)
    v_next_raw[:, :-1] = values[:, 1:] * (1.0 - truncateds[:, :-1])
    v_next_raw[:, -1]  = 0.0
    v_next             = v_next_raw + bootstrap_values

    deltas = rho * (rewards + gamma * not_terminated * v_next - values)
    decays = gamma * c * not_done

    # Append sentinel at T: a=bootstrap_values[:,-1], b=0 to seed the boundary carry.
    N        = rewards.shape[0]
    sentinel_a = bootstrap_values[:, -1].unsqueeze(1)
    sentinel_b = torch.zeros(N, 1, device=decays.device, dtype=decays.dtype)
    a = torch.cat([deltas, sentinel_a], dim=1)
    b = torch.cat([decays, sentinel_b], dim=1)
    value_deltas = parallel_suffix_scan(a, b)[:, :rewards.shape[1]]

    vtrace_targets = value_deltas + values

    next_vtrace_targets         = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1] = torch.where(
        truncateds[:, :-1].bool(),
        bootstrap_values[:, :-1],
        vtrace_targets[:, 1:],
    )
    next_vtrace_targets[:, -1] = bootstrap_values[:, -1]

    vtrace_advantages = rho * (rewards + gamma * not_terminated * next_vtrace_targets - values)
    return vtrace_targets, vtrace_advantages


def test_vectorized_vtrace_with_truncations_correctness():
    """Verify vectorized_vtrace_with_truncations matches _ref_vtrace_sequential.

    Uses the same two-interior-truncation fixture as test_vtrace_fused_two_interior_truncations
    so correctness of the new baseline is confirmed before it is used in the benchmark.
    """
    torch.manual_seed(77)
    N, T = 2, 12
    log_pi_target   = -torch.rand(N, T)
    log_pi_behavior = -torch.rand(N, T)
    values      = torch.rand(N, T)
    rewards     = torch.rand(N, T)
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

    exp_t, exp_a = _ref_vtrace_sequential(
        log_pi_target, log_pi_behavior, values, rewards, terminateds, truncateds,
        bootstrap_values, gamma=0.99,
    )
    act_t, act_a = vectorized_vtrace_with_truncations(
        log_pi_target, log_pi_behavior, values, rewards, terminateds, truncateds,
        bootstrap_values, gamma=0.99,
    )
    torch.testing.assert_close(act_t, exp_t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(act_a, exp_a, atol=1e-5, rtol=1e-5)


@cuda_only
@pytest.mark.slow
def test_vtrace_truncation_performance():
    """
    Truncation-path performance: HAS_TRUNCATIONS=True kernel vs
    torch.compile(vectorized_vtrace_with_truncations).

    vectorized_vtrace_with_truncations uses parallel_suffix_scan -- a log-depth
    parallel associative scan with no Python time loop, compiles cleanly.

    Inputs have ~5% truncated steps (mutually exclusive with terminated),
    so the kernel dispatches HAS_TRUNCATIONS=True (7 full-width reads).
    No floor is asserted -- the truncation path is a correctness feature.
    This test makes the speedup visible and tracked.
    """
    compiled_vec_trunc = torch.compile(vectorized_vtrace_with_truncations)

    def _make_trunc_inputs(num_envs, seq_len, seed=0):
        torch.manual_seed(seed)
        log_pi_target   = -torch.rand(num_envs, seq_len, device="cuda")
        log_pi_behavior = -torch.rand(num_envs, seq_len, device="cuda")
        values      = torch.randn(num_envs, seq_len, device="cuda")
        rewards     = torch.randn(num_envs, seq_len, device="cuda")
        terminateds = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
        trunc_cand  = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
        truncateds  = trunc_cand * (1.0 - terminateds)
        bootstrap_values = torch.zeros(num_envs, seq_len, device="cuda")
        bootstrap_values[truncateds.bool()] = torch.rand(
            int(truncateds.sum().item()), device="cuda"
        )
        bootstrap_values[:, -1] = torch.rand(num_envs, device="cuda")
        return log_pi_target, log_pi_behavior, values, rewards, terminateds, truncateds, bootstrap_values

    _wa = _make_trunc_inputs(64, 512, seed=1)
    compiled_vec_trunc(*_wa, gamma=0.99)
    compute_vtrace_fused(
        _wa[0], _wa[1], _wa[2], _wa[3], _wa[4],
        truncateds=_wa[5], gamma=0.99, bootstrap_values=_wa[6],
    )
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
        _warmup_gpu(compute_vtrace_fused,
                    args[0], args[1], args[2], args[3], args[4],
                    truncateds=args[5], gamma=0.99, bootstrap_values=args[6])
        _warmup_gpu(compiled_vec_trunc, *args, gamma=0.99)

        tri_ms = _bench_gpu(
            compute_vtrace_fused,
            args[0], args[1], args[2], args[3], args[4],
            truncateds=args[5], gamma=0.99, bootstrap_values=args[6],
            n_iter=n_iter,
        )
        vec_ms = _bench_gpu(
            compiled_vec_trunc, *args,
            gamma=0.99,
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
