import time

import numpy as np
import pytest
import torch

# numpy is used intentionally: RLlib v2 stores episode data as NumPy arrays in
# SingleAgentEpisode, so rllib_vtrace_triton mirrors the real adoption path
# (np -> GPU -> np) a user would take to swap in the Triton kernel.

triton = pytest.importorskip("triton")

from rl_triton.ops.vtrace import compute_vtrace_triton, _FLAT_MAX_SEQ_LEN

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
# RLlib-style reference (CPU, mirrors ray.rllib.utils.vtrace_torch)
# ---------------------------------------------------------------------------
#
# RLlib's from_importance_weights works on [T, B]-shaped tensors and receives
# `discounts` (γ*(1-done), pre-multiplied) rather than separate gamma/dones.
# It also hard-clips c at 1.0 and runs the backward loop on CPU.
# These adapters expose the same interface as the public API (num_envs, seq_len)
# so they can be benchmarked head-to-head.

def rllib_vtrace(
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
    CPU V-Trace backward loop matching RLlib's from_importance_weights logic.

    Mirrors the [T, B] + discounts convention of RLlib internally but accepts
    and returns [num_envs, seq_len] tensors (CPU) to match our public API.
    Runs entirely on CPU — no GPU involved.
    """
    # Move to CPU for an apples-to-apples comparison with the RLlib baseline.
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
    discounts  = gamma * (1.0 - dones)           # [num_envs, T], matches RLlib's discounts

    deltas     = rho * (rewards + discounts * next_values - values)

    # Backward scan — identical to RLlib's Python loop (transposed to [B, T]).
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


def rllib_vtrace_triton(
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
    NumPy → GPU Triton kernel → NumPy adoption path.

    Mirrors how an RLlib user would replace rllib_vtrace with the Triton kernel:
    read NumPy arrays from SingleAgentEpisode, move to GPU, compute, move back.
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
    # is_ratio = exp(log_target - log_behavior) = exp(0) = 1  for all t.
    # delta[t] = 1 * (r[t] + 1 * V'[t] - V[t])
    # decay[t] = 1 * 1 * 1 = 1
    #
    # r=[1,1], V=[0,0], V'=[0,0]  => delta=[1,1], decay=[1,1]
    # Δ[1] = 1 + 1*0 = 1
    # Δ[0] = 1 + 1*1 = 2
    # targets = Δ + V = [2, 1]
    # next_vtrace = [targets[1], V'[-1]] = [1, 0]
    # adv[t] = rho*(r + gamma*next_vtrace*(1-done) - V)
    # adv[0] = 1*(1 + 1*1 - 0) = 2
    # adv[1] = 1*(1 + 1*0 - 0) = 1
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
    # delta=[1,1], decay=[1,1], bootstrap=0.5
    # Δ[1] = 1 + 1*0.5 = 1.5
    # Δ[0] = 1 + 1*1.5 = 2.5
    # targets = [2.5, 1.5]
    # next_vtrace = [1.5, 0]  (next_values[:,-1]=0)
    # adv[0] = 1*(1+1*1.5-0) = 2.5
    # adv[1] = 1*(1+1*0-0)   = 1.0
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
    # Episode ends at t=0 (done[0]=1): V(s_1) is cut off for delta[0].
    # is_ratio=1, gamma=1, V=0, V'=2.
    # delta[0] = 1*(1 + 1*2*(1-1) - 0) = 1*(1+0-0) = 1
    # delta[1] = 1*(1 + 1*2*(1-0) - 0) = 3
    # decay[0] = 1*1*(1-1) = 0
    # decay[1] = 1*1*(1-0) = 1
    # Δ[1] = 3 + 1*0 = 3
    # Δ[0] = 1 + 0*3 = 1   <- done cuts the carry
    # targets = [1, 3]
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
    # gamma=1, V=0, V'=0, r=1, done=0.
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
def test_vtrace_correctness_basic():
    args = _make_inputs(64, 512)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace_triton(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


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
# RLlib adapter correctness
# ---------------------------------------------------------------------------

@cuda_only
def test_rllib_vtrace_matches_reference():
    """rllib_vtrace (CPU loop) must agree with reference_vtrace within float32 tolerance."""
    args = _make_inputs(32, 256, device="cpu", seed=10)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = rllib_vtrace(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cpu(), atol=1e-4, rtol=1e-4)


@cuda_only
def test_rllib_vtrace_triton_matches_reference():
    """NumPy→GPU→NumPy path must agree with reference_vtrace."""
    args_np  = _make_inputs_np(32, 256, seed=11)
    args_gpu = _make_inputs(32, 256, seed=11)
    exp_t, exp_a = reference_vtrace(*args_gpu, gamma=0.99)

    act_t_np, act_a_np = rllib_vtrace_triton(*args_np, gamma=0.99)
    act_t = torch.from_numpy(act_t_np)
    act_a = torch.from_numpy(act_a_np)
    torch.testing.assert_close(act_t, exp_t.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a.cpu(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_gpu(fn, *args, n_warmup: int = 25, n_iter: int = 100) -> float:
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def _bench_cpu(fn, *args, n_warmup: int = 5, n_iter: int = 20) -> float:
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    return (time.perf_counter() - t0) / n_iter * 1000.0


def _n_iter_gpu(seq_len: int, num_envs: int = 1) -> tuple[int, int]:
    elements = seq_len * num_envs
    n_warmup = max(5,  min(25,  5_000_000 // elements))
    n_iter   = max(20, min(200, 20_000_000 // elements))
    return n_warmup, n_iter


def _n_iter_cpu(seq_len: int, num_envs: int = 1) -> tuple[int, int]:
    work = seq_len * num_envs
    n_warmup = max(1, min(5,  500_000 // (2 * work)))
    n_iter   = max(1, min(20, 500_000 // work))
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
def test_vtrace_performance():
    """
    Sweep over (num_envs, seq_len) configs comparing:
      - triton:       GPU Triton kernel  (CUDA events)
      - pt.compile:   torch.compile on the reference loop  (wall-clock)
      - rllib(cpu):   CPU Python loop matching RLlib's from_importance_weights  (wall-clock)
      - np→triton→np: NumPy→GPU→NumPy adoption path  (wall-clock, includes transfer overhead)

    triton uses CUDA events (pure kernel time). All others use wall-clock because
    torch.compile dispatches one CUDA op per timestep from Python — CUDA events would
    miss that CPU stall and make it look unrealistically fast.

    Assertion: Triton must be >=1.5x faster than torch.compile (wall-clock vs CUDA events).
    """
    compiled_vtrace = torch.compile(reference_vtrace)

    _args = _make_inputs(64, 512)
    compiled_vtrace(*_args, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>10} {'compiled':>10} {'rllib(cpu)':>12} {'np→triton→np':>14} {'vs compile':>12}"
    )
    print(header)
    print("-" * len(header))

    all_speedups = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_np  = _make_inputs_np(num_envs, seq_len)
        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)
        cpu_warmup, cpu_iter = _n_iter_cpu(seq_len, num_envs)

        triton_ms    = _bench_gpu(compute_vtrace_triton, *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
        compiled_ms  = _bench_cpu(compiled_vtrace, *args_gpu, gamma=0.99, n_warmup=cpu_warmup, n_iter=cpu_iter)
        rllib_ms     = _bench_cpu(rllib_vtrace, *args_gpu, gamma=0.99, n_warmup=cpu_warmup, n_iter=cpu_iter)
        np_triton_ms = _bench_cpu(rllib_vtrace_triton, *args_np, gamma=0.99, n_warmup=cpu_warmup, n_iter=cpu_iter)
        speedup      = compiled_ms / triton_ms
        all_speedups.append(speedup)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>9.3f}ms {compiled_ms:>9.3f}ms "
            f"{rllib_ms:>11.3f}ms {np_triton_ms:>13.3f}ms "
            f"{speedup:>10.1f}x"
        )

    print(
        "\ntriton:       CUDA events — pure kernel time."
        "\npt.compile:   wall-clock — dispatches one CUDA op per timestep from Python;"
        "\n              CUDA events would miss that CPU stall and make it look unrealistically fast."
        "\nrllib(cpu):   wall-clock — CPU Python loop matching RLlib's from_importance_weights."
        "\nnp→triton→np: wall-clock — NumPy→GPU→NumPy path replacing rllib_vtrace in an RLlib worker."
    )

    assert min(all_speedups) >= 1.5, (
        f"Expected >=1.5x speedup over torch.compile, worst was {min(all_speedups):.2f}x"
    )
