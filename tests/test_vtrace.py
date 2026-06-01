import numpy as np
import pytest
import torch

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu

# numpy is used intentionally: episode data is typically stored as NumPy arrays,
# so numpy_to_triton_to_numpy mirrors the real adoption path (np -> GPU -> np)
# a user would take to swap in the Triton kernel.

triton = pytest.importorskip("triton")

from rl_triton.ops.vtrace import compute_vtrace_triton
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def reference_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch V-Trace backward scan — ground truth for correctness tests only."""
    num_envs, T = rewards.shape

    is_ratios = torch.exp(log_pi_target - log_pi_behavior)
    rho = torch.clamp(is_ratios, max=rho_bar)
    c   = torch.clamp(is_ratios, max=c_bar)

    deltas = rho * (rewards + gamma * next_values * (1.0 - dones) - values)
    decays = gamma * c * (1.0 - dones)

    # Backward scan for value_deltas: Δ[t] = δ[t] + decay[t] * Δ[t+1]
    value_deltas = torch.zeros_like(rewards)
    carry = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry = deltas[:, t] + decays[:, t] * carry
        value_deltas[:, t] = carry

    vtrace_targets = value_deltas + values

    next_vtrace_targets = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1] = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]  = next_values[:, -1]

    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * (1.0 - dones) - values)

    return vtrace_targets, vtrace_advantages


def _make_inputs(num_envs, seq_len, device="cuda", seed=0):
    torch.manual_seed(seed)
    log_pi_target   = -torch.rand(num_envs, seq_len, device=device)
    log_pi_behavior = -torch.rand(num_envs, seq_len, device=device)
    values          = torch.randn(num_envs, seq_len, device=device)
    next_values     = torch.randn(num_envs, seq_len, device=device)
    rewards         = torch.randn(num_envs, seq_len, device=device)
    dones           = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return log_pi_target, log_pi_behavior, values, next_values, rewards, dones


def _make_inputs_np(num_envs, seq_len, seed=0):
    """Return the same random inputs as _make_inputs but as NumPy arrays (CPU)."""
    args_gpu = _make_inputs(num_envs, seq_len, device="cpu", seed=seed)
    return tuple(t.numpy() for t in args_gpu)


# ---------------------------------------------------------------------------
# CPU numpy baseline and end-to-end adoption path
# ---------------------------------------------------------------------------
#
# numpy_vtrace mirrors how a CPU-based framework computes V-Trace:
# a plain Python backward loop on CPU tensors.
# These adapters expose the same interface as the public API (num_envs, seq_len)
# so they can be benchmarked head-to-head.

def numpy_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU V-Trace backward loop — plain Python loop on CPU tensors.

    Accepts and returns [num_envs, seq_len] tensors (CPU) to match our public
    API.  Runs entirely on CPU — no GPU involved.
    """
    # Move inputs to CPU — this function runs entirely on CPU with no GPU involvement.
    cpu = lambda t: t.cpu().float()
    log_pi_target   = cpu(log_pi_target)
    log_pi_behavior = cpu(log_pi_behavior)
    values          = cpu(values)
    next_values     = cpu(next_values)
    rewards         = cpu(rewards)
    dones           = cpu(dones)

    num_envs, T = rewards.shape

    is_ratios  = torch.exp(log_pi_target - log_pi_behavior)
    rho        = torch.clamp(is_ratios, max=rho_bar)
    c          = torch.clamp(is_ratios, max=c_bar)
    discounts  = gamma * (1.0 - dones)            # [num_envs, T]

    deltas     = rho * (rewards + discounts * next_values - values)

    # Backward scan: same recurrence as reference_vtrace, factorized as
    # decay[t] = discounts[t]*c[t] to avoid recomputing gamma*(1-done) twice.
    value_deltas = torch.zeros_like(rewards)
    carry        = torch.zeros(num_envs)
    for t in reversed(range(T)):
        carry                = deltas[:, t] + discounts[:, t] * c[:, t] * carry
        value_deltas[:, t]   = carry

    vtrace_targets = value_deltas + values

    next_vtrace_targets            = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1]    = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]     = next_values[:, -1]

    vtrace_advantages = rho * (rewards + discounts * next_vtrace_targets - values)
    return vtrace_targets, vtrace_advantages


def numpy_to_triton_to_numpy(
    log_pi_target_np: np.ndarray,
    log_pi_behavior_np: np.ndarray,
    values_np: np.ndarray,
    next_values_np: np.ndarray,
    rewards_np: np.ndarray,
    dones_np: np.ndarray,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NumPy → GPU Triton kernel → NumPy end-to-end adoption path.

    Mirrors how a user would replace numpy_vtrace with the Triton kernel:
    read NumPy arrays, move to GPU, compute, move back.
    """
    to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch.float32)
    log_pi_target   = to_gpu(log_pi_target_np)
    log_pi_behavior = to_gpu(log_pi_behavior_np)
    values          = to_gpu(values_np)
    next_values     = to_gpu(next_values_np)
    rewards         = to_gpu(rewards_np)
    dones           = to_gpu(dones_np)

    vtrace_targets, vtrace_advantages = compute_vtrace_triton(
        log_pi_target, log_pi_behavior,
        values, next_values, rewards, dones,
        gamma=gamma, rho_bar=rho_bar, c_bar=c_bar,
    )
    return vtrace_targets.cpu().numpy(), vtrace_advantages.cpu().numpy()


# ---------------------------------------------------------------------------
# Ground-truth value tests
# ---------------------------------------------------------------------------
#
# Hand-computed from the recurrence with rho_bar=c_bar=inf (no clipping) so
# IS ratios pass through, making the arithmetic easy to verify by hand.

@cuda_only
def test_vtrace_known_values_single_env():
    # No clipping (rho_bar=c_bar=100), no dones, gamma=1.
    # log_pi_target=log_pi_behavior=0 -> is_ratio=exp(0)=1, rho=c=1.
    # rewards=[1,1], values=[0,0], next_values=[0,0].
    # delta[t] = rho*(r[t] + gamma*next_values[t]*(1-done[t]) - values[t])
    #          = 1*(1 + 1*0 - 0) = 1
    # decay[t] = gamma*c*(1-done[t]) = 1
    # Δ[1] = 1 + 1*0 = 1
    # Δ[0] = 1 + 1*1 = 2
    # targets = Δ + values = [2, 1]
    # next_vtrace = [targets[1], next_values[:,-1]] = [1, 0]
    # adv[t] = rho*(r[t] + gamma*next_vtrace[t]*(1-done[t]) - values[t])
    # adv[0] = 1*(1 + 1*1 - 0) = 2
    # adv[1] = 1*(1 + 1*0 - 0) = 1  <- next_vtrace[1]=next_values[:,-1]=0
    log_pi  = torch.zeros(1, 2, device="cuda")
    values  = torch.zeros(1, 2, device="cuda")
    nvalues = torch.zeros(1, 2, device="cuda")
    rewards = torch.ones(1, 2, device="cuda")
    dones   = torch.zeros(1, 2, device="cuda")

    targets, advantages = compute_vtrace_triton(
        log_pi, log_pi, values, nvalues, rewards, dones,
        gamma=1.0, rho_bar=100.0, c_bar=100.0,
    )
    torch.testing.assert_close(targets,    torch.tensor([[2.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(advantages, torch.tensor([[2.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_known_values_truncated():
    # Same setup but bootstrap=0.5 — Δ[T]=0.5 propagates back.
    # V=0, V'=0 (next_values=zeros), r=1, log_pi equal so rho=c=1, gamma=1, no dones.
    # delta=[1,1], decay=[1,1], bootstrap=0.5
    # Δ[1] = 1 + 1*0.5 = 1.5
    # Δ[0] = 1 + 1*1.5 = 2.5
    # targets = [2.5, 1.5]
    # next_vtrace = [targets[1], next_values[:,-1]] = [1.5, 0]
    #   next_vtrace[t] = targets[t+1] for t<T; falls back to next_values[:,-1]=0 at t=T-1
    #   (no corrected target exists beyond the end of the sequence)
    # adv[t] = rho*(r + gamma*next_vtrace[t] - V)
    # adv[0] = 1*(1 + 1*1.5 - 0) = 2.5
    # adv[1] = 1*(1 + 1*0.0 - 0) = 1.0  <- next_vtrace[1]=0 because next_values[:,-1]=0
    log_pi     = torch.zeros(1, 2, device="cuda")
    values     = torch.zeros(1, 2, device="cuda")
    nvalues    = torch.zeros(1, 2, device="cuda")
    rewards    = torch.ones(1, 2, device="cuda")
    dones      = torch.zeros(1, 2, device="cuda")
    bootstrap  = torch.tensor([0.5], device="cuda")

    targets, advantages = compute_vtrace_triton(
        log_pi, log_pi, values, nvalues, rewards, dones,
        gamma=1.0, rho_bar=100.0, c_bar=100.0,
        bootstrap_values=bootstrap,
    )
    torch.testing.assert_close(targets,    torch.tensor([[2.5, 1.5]], device="cuda"), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(advantages, torch.tensor([[2.5, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_known_values_done():
    # done[0]=1: episode ends at t=0, so next_values is cut off in delta[0].
    # log_pi equal -> is_ratio=1, rho=c=1. gamma=1, values=[0,0], next_values=[2,2], rewards=[1,1].
    # delta[t] = rho*(r[t] + gamma*next_values[t]*(1-done[t]) - values[t])
    # delta[0] = 1*(1 + 1*2*(1-1) - 0) = 1*(1+0-0) = 1
    # delta[1] = 1*(1 + 1*2*(1-0) - 0) = 3
    # decay[t] = gamma*c*(1-done[t])
    # decay[0] = 1*1*(1-1) = 0
    # decay[1] = 1*1*(1-0) = 1
    # Δ[1] = 3 + 1*0 = 3
    # Δ[0] = 1 + 0*3 = 1   <- decay=0 cuts the carry at the episode boundary
    # targets = Δ + values = [1, 3]
    log_pi  = torch.zeros(1, 2, device="cuda")
    values  = torch.zeros(1, 2, device="cuda")
    nvalues = torch.full((1, 2), 2.0, device="cuda")
    rewards = torch.ones(1, 2, device="cuda")
    dones   = torch.tensor([[1.0, 0.0]], device="cuda")

    targets, _ = compute_vtrace_triton(
        log_pi, log_pi, values, nvalues, rewards, dones,
        gamma=1.0, rho_bar=100.0, c_bar=100.0,
    )
    torch.testing.assert_close(targets, torch.tensor([[1.0, 3.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_vtrace_known_values_clipping():
    # IS ratio = exp(0 - log(2)) = 0.5, rho_bar=c_bar=1 -> rho=c=0.5.
    # gamma=1, values=[0,0], next_values=[0,0], rewards=[1,1], dones=[0,0].
    # delta[t] = 0.5*(1+0-0) = 0.5
    # decay[t] = 1*0.5*1 = 0.5
    # Δ[1] = 0.5
    # Δ[0] = 0.5 + 0.5*0.5 = 0.75
    # targets = [0.75, 0.5]
    log_pi_t = torch.zeros(1, 2, device="cuda")
    log_pi_b = torch.full((1, 2), torch.log(torch.tensor(2.0)).item(), device="cuda")
    values   = torch.zeros(1, 2, device="cuda")
    nvalues  = torch.zeros(1, 2, device="cuda")
    rewards  = torch.ones(1, 2, device="cuda")
    dones    = torch.zeros(1, 2, device="cuda")

    targets, _ = compute_vtrace_triton(
        log_pi_t, log_pi_b, values, nvalues, rewards, dones,
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
    act_t, act_a = compute_vtrace_triton(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_correctness_bootstrap():
    """Truncated episodes: non-zero bootstrap propagates correctly."""
    torch.manual_seed(3)
    num_envs, seq_len = 32, 512
    args = _make_inputs(num_envs, seq_len, seed=3)
    bootstrap = torch.rand(num_envs, device="cuda")

    exp_t, exp_a = reference_vtrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    act_t, act_a = compute_vtrace_triton(*args, gamma=0.99, bootstrap_values=bootstrap)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_correctness_mixed_termination():
    """Batch with some terminated (bootstrap=0) and some truncated (bootstrap>0) envs."""
    torch.manual_seed(4)
    num_envs, seq_len = 16, 256
    args = _make_inputs(num_envs, seq_len, seed=4)
    bootstrap = torch.cat([
        torch.zeros(num_envs // 2, device="cuda"),
        torch.rand(num_envs // 2, device="cuda"),
    ])

    exp_t, exp_a = reference_vtrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    act_t, act_a = compute_vtrace_triton(*args, gamma=0.99, bootstrap_values=bootstrap)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_non_contiguous_input():
    """Wrapper must handle non-contiguous inputs via .contiguous()."""
    torch.manual_seed(5)
    base = torch.randn(32, 512, 2, device="cuda")
    log_pi_t = base[..., 0]
    log_pi_b = base[..., 1]
    values   = torch.randn(32, 512, 2, device="cuda")[..., 0]
    nvalues  = torch.randn(32, 512, 2, device="cuda")[..., 0]
    rewards  = torch.randn(32, 512, 2, device="cuda")[..., 0]
    dones    = torch.zeros(32, 512, 2, device="cuda")[..., 0]

    exp_t, exp_a = reference_vtrace(
        log_pi_t.contiguous(), log_pi_b.contiguous(),
        values.contiguous(), nvalues.contiguous(),
        rewards.contiguous(), dones.contiguous(),
        gamma=0.99,
    )
    act_t, act_a = compute_vtrace_triton(
        log_pi_t, log_pi_b, values, nvalues, rewards, dones, gamma=0.99,
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
    """Fused kernel must match reference_vtrace."""
    args = _make_inputs(num_envs, seq_len, seed=20)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace_fused(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_fused_bootstrap():
    """Bootstrap propagates correctly through the fused kernel."""
    args = _make_inputs(32, 512, seed=21)
    bootstrap = torch.rand(32, device="cuda")
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    act_t, act_a = compute_vtrace_fused(*args, gamma=0.99, bootstrap_values=bootstrap)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# numpy baseline correctness
# ---------------------------------------------------------------------------

@cuda_only
def test_numpy_vtrace_matches_reference():
    """numpy_vtrace (CPU loop) must agree with reference_vtrace within float32 tolerance."""
    args = _make_inputs(32, 256, device="cpu", seed=10)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = numpy_vtrace(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cpu(), atol=1e-4, rtol=1e-4)


@cuda_only
def test_numpy_to_triton_to_numpy_matches_reference():
    """NumPy→GPU→NumPy path must agree with reference_vtrace."""
    args_np = _make_inputs_np(32, 256, seed=11)
    args_gpu = tuple(torch.from_numpy(a).cuda() for a in args_np)
    exp_t, exp_a = reference_vtrace(*args_gpu, gamma=0.99)

    act_t_np, act_a_np = numpy_to_triton_to_numpy(*args_np, gamma=0.99)
    act_t = torch.from_numpy(act_t_np)
    act_a = torch.from_numpy(act_a_np)
    torch.testing.assert_close(act_t, exp_t.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cpu(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Vectorized PyTorch baseline (benchmark only)
# ---------------------------------------------------------------------------

def vectorized_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully vectorized V-Trace via log-space suffix cumsum — strong compiled baseline.

    Replaces the Python backward loop in reference_vtrace with a vectorized
    equivalent.  The backward scan Δ[t] = δ[t] + decay[t]*Δ[t+1] is a weighted
    sum where each weight is the suffix product of decays from t to T-1.  These
    suffix products are computed in log-space to avoid underflow over long sequences.

    Not production-hardened (log of zero decay requires clamping); only used
    for benchmarking as the strongest fully-vectorized PyTorch baseline.
    Timed with CUDA events (no Python loop, so wall-clock is not needed).
    """
    is_ratios = torch.exp(log_pi_target - log_pi_behavior)
    rho    = torch.clamp(is_ratios, max=rho_bar)
    c      = torch.clamp(is_ratios, max=c_bar)
    deltas = rho * (rewards + gamma * next_values * (1.0 - dones) - values)
    decays = gamma * c * (1.0 - dones)

    # Suffix product of decays via log-space cumsum, then exponentiate.
    # log_suffix[t] = log(decay[t]) + log(decay[t+1]) + ... + log(decay[T-1])
    log_suffix = torch.flip(
        torch.cumsum(torch.flip(torch.log(decays.clamp(min=1e-38)), [1]), dim=1), [1]
    )
    weights      = torch.exp(log_suffix)
    value_deltas = torch.flip(
        torch.cumsum(torch.flip(deltas * weights, [1]), dim=1), [1]
    ) / weights

    if bootstrap_values is not None:
        value_deltas = value_deltas + weights * bootstrap_values.unsqueeze(1)

    vtrace_targets               = value_deltas + values
    next_vtrace_targets          = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1]  = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]   = next_values[:, -1]
    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * (1.0 - dones) - values)
    return vtrace_targets, vtrace_advantages


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

      fused              — single Triton kernel  (CUDA events)
      pt.compile(vec)    — torch.compile on vectorized_vtrace  (CUDA events)
      pt.compile(loop)   — torch.compile on reference_vtrace loop  (wall-clock)
      np->triton->np     — NumPy->GPU->NumPy adoption path  (wall-clock)
      numpy(cpu)         — CPU Python loop  (wall-clock)

    fused and pt.compile(vec) are fully vectorized — no Python loop — so CUDA
    events are appropriate for both.  pt.compile(loop) dispatches one CUDA op
    per timestep from Python; wall-clock captures that CPU stall.

    Assertions:
      - fused must be >=1.5x faster than pt.compile(vec) — the strongest
        fully-vectorized PyTorch baseline.
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

    all_speedups_vec  = []
    all_speedups_e2e  = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_np  = _make_inputs_np(num_envs, seq_len)
        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)

        fused_ms    = _bench_gpu(compute_vtrace_fused,     *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
        vec_ms      = _bench_gpu(compiled_vec,             *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
        loop_ms     = _bench_cpu(compiled_loop,            *args_gpu, gamma=0.99)
        np_tri_ms   = _bench_cpu(numpy_to_triton_to_numpy, *args_np,  gamma=0.99)
        numpy_ms    = _bench_cpu(numpy_vtrace,             *args_gpu, gamma=0.99)

        su_vec   = vec_ms  / fused_ms
        su_loop  = loop_ms / fused_ms
        su_e2e   = numpy_ms / np_tri_ms
        all_speedups_vec.append(su_vec)
        all_speedups_e2e.append(su_e2e)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{fused_ms:>7.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms "
            f"{np_tri_ms:>12.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_e2e:>20.1f}x"
        )

    print(
        "\nfused            : CUDA events — single kernel, IS ratios + scan + targets fused."
        "\ncompile(vec)     : CUDA events — vectorized log-space cumsum, no Python loop."
        "\ncompile(loop)    : wall-clock  — one CUDA op per timestep from Python;"
        "\n                   CUDA events would miss the CPU stall."
        "\nnp->tri->np      : wall-clock  — NumPy->GPU->NumPy, realistic adoption path."
        "\nnumpy(cpu)       : wall-clock  — pure CPU Python loop."
        "\nspeedups vs fused kernel."
    )

    assert min(all_speedups_vec) >= 1.5, (
        f"Expected >=1.5x speedup over pt.compile(vec) across all configs, "
        f"worst was {min(all_speedups_vec):.2f}x"
    )

    min_e2e_speedup = min(all_speedups_e2e)
    assert min_e2e_speedup >= 1.5, (
        f"Expected >=1.5x end-to-end speedup (np->triton->np vs numpy(cpu)) across all configs — "
        f"below this the kernel gain is likely lost to inter-worker communication overhead in a "
        f"distributed setup. Worst was {min_e2e_speedup:.2f}x"
    )
