import pytest
import torch
import numpy as np

triton = pytest.importorskip("triton")

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu
from rl_triton.ops.retrace import compute_retrace

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def reference_retrace(
    action_probs_target: torch.Tensor,
    action_probs_behavior: torch.Tensor,
    q_values: torch.Tensor,
    next_q_values_all: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    truncateds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch Retrace(λ) backward scan — ground truth for correctness tests only."""
    num_envs, T = rewards.shape

    # terminated[t]=1 for true episode ends only; truncations keep the bootstrap.
    if truncateds is not None:
        terminated = (dones - truncateds).clamp(min=0.0)
    else:
        terminated = dones

    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    deltas = rewards + gamma * expected_next_q * (1.0 - terminated) - q_values

    pi_a = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c    = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)

    q_deltas = torch.zeros_like(rewards)
    carry    = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        # decay at t uses c[t+1]; c[t+1] is out-of-bounds at t=T-1 so treat as 0.
        # dones (terminated OR truncated) stops trace propagation at any boundary.
        c_next = c[:, t + 1] if t + 1 < T else torch.zeros(num_envs, device=rewards.device)
        decay  = gamma * c_next * (1.0 - dones[:, t])
        carry          = deltas[:, t] + decay * carry
        q_deltas[:, t] = carry

    return q_deltas + q_values


def vectorized_retrace(
    action_probs_target: torch.Tensor,
    action_probs_behavior: torch.Tensor,
    q_values: torch.Tensor,
    next_q_values_all: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    truncateds: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fully vectorized Retrace(λ) via log-space suffix cumsum — strong compiled baseline.

    Replaces the Python backward loop in reference_retrace with a vectorized
    equivalent.  The backward scan Δ[t] = u[t] + v[t]*Δ[t+1] is a weighted sum
    where each weight is the suffix product of decays v from t+1 to T-1.  These
    suffix products are computed in log-space to avoid underflow over long sequences.

    Note: Retrace uses c[t+1] as the decay coefficient, so the suffix product for
    Δ[t] starts at v[t] = γ·c[t+1]·(1-d[t]), exactly matching the recurrence
    derived in docs/retrace.md.  v[T-1]=0 because c[T] is out-of-bounds; Δ[T]=0.

    Not production-hardened (log of zero decay requires clamping); used only
    for benchmarking as the strongest fully-vectorized PyTorch baseline.
    Timed with CUDA events (no Python loop, so wall-clock is not needed).
    """
    if truncateds is not None:
        terminated = (dones - truncateds).clamp(min=0.0)
    else:
        terminated = dones

    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    u = rewards + gamma * expected_next_q * (1.0 - terminated) - q_values

    pi_a = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c    = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    c_next          = torch.empty_like(c)
    c_next[:, :-1]  = c[:, 1:]
    c_next[:, -1]   = 0.0
    v = gamma * c_next * (1.0 - dones)

    # Suffix product of decays via log-space cumsum, then exponentiate.
    # log_suffix[t] = log(v[t]) + log(v[t+1]) + ... + log(v[T-1])
    log_suffix   = torch.flip(
        torch.cumsum(torch.flip(torch.log(v.clamp(min=1e-38)), [1]), dim=1), [1]
    )
    weights      = torch.exp(log_suffix)
    q_deltas     = torch.flip(
        torch.cumsum(torch.flip(u * weights, [1]), dim=1), [1]
    ) / weights

    return q_deltas + q_values


def _make_inputs(num_envs, seq_len, num_actions=4, device="cuda", seed=0):
    torch.manual_seed(seed)
    # Target policy: valid probability distribution over actions.
    action_probs_target   = torch.softmax(torch.randn(num_envs, seq_len, num_actions, device=device), dim=-1)
    # Behavior policy probability of the taken action: scalar per step.
    action_probs_behavior = torch.rand(num_envs, seq_len, device=device) * 0.8 + 0.1
    q_values              = torch.randn(num_envs, seq_len, device=device)
    next_q_values_all     = torch.randn(num_envs, seq_len, num_actions, device=device)
    actions               = torch.randint(0, num_actions, (num_envs, seq_len), device=device)
    rewards               = torch.randn(num_envs, seq_len, device=device)
    dones                 = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return action_probs_target, action_probs_behavior, q_values, next_q_values_all, actions, rewards, dones


# ---------------------------------------------------------------------------
# Ground-truth value tests
# ---------------------------------------------------------------------------
#
# Hand-computed with num_actions=1, on-policy (π=μ so c=1), no dones, gamma=1,
# lambda=1.  With a single action, E_π[Q(s_{t+1},·)] = Q_next[t] exactly.
# The recurrence reduces to:
#   Δ[t] = δ[t] + Δ[t+1],  δ[t] = r[t] + Q_next[t] - Q[t]

@cuda_only
def test_retrace_known_values_single_env():
    # seq_len=2, num_actions=1, gamma=1, lambda=1, no dones.
    # pi=mu=1.0, c=1.0, E_pi[Q_next]=Q_next (single action).
    # delta[0] = 1 + 1*2 - 0 = 3,  delta[1] = 1 + 1*3 - 0 = 4
    # decay[0] = 1*c[1]*(1-0) = 1
    # decay[1] = 1*c[2]*(1-0) = 0  — c[2] is out of bounds (no action beyond the
    #                                 trajectory end), so c_next[:,-1] is forced to
    #                                 0 by the implementation; there is no c[T].
    # Delta[1] = 4 + 0*0 = 4
    # Delta[0] = 3 + 1*4 = 7
    # Q_ret = Delta + Q = [7, 4]
    probs_t   = torch.ones(1, 2, 1, device="cuda")
    probs_b   = torch.ones(1, 2, device="cuda")
    q         = torch.zeros(1, 2, device="cuda")
    q_next    = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions   = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards   = torch.ones(1, 2, device="cuda")
    dones     = torch.zeros(1, 2, device="cuda")

    out = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([[7.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_values_done():
    # done[0]=1 cuts the trace at t=0: decay[0]=gamma*c[1]*(1-done[0])=0.
    # delta[0] = 1 + 1*2*(1-1) - 0 = 1
    # delta[1] = 1 + 1*3*(1-0) - 0 = 4
    # Delta[1] = 4
    # Delta[0] = 1 + 0*4 = 1
    # Q_ret = [1, 4]
    probs_t   = torch.ones(1, 2, 1, device="cuda")
    probs_b   = torch.ones(1, 2, device="cuda")
    q         = torch.zeros(1, 2, device="cuda")
    q_next    = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions   = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards   = torch.ones(1, 2, device="cuda")
    dones     = torch.tensor([[1.0, 0.0]], device="cuda")

    out = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_values_clipping():
    # pi_a=1.0 (single action, on-policy), mu=0.5 -> IS ratio=2.0, clipped to c_bar=1.0 -> c=1.0.
    # E_pi[Q_next] = 1.0 * q_next = q_next (single action, probs_t=1.0).
    # seq_len=2, gamma=1, no dones, q_next=0.5.
    # delta[0] = 1 + 1*0.5 - 0 = 1.5,  delta[1] = 1 + 1*0.5 - 0 = 1.5
    # c=min(1.0, 2.0)=1.0 everywhere; decay[0]=1*c[1]=1.0, decay[1]=0 (c[T]=0)
    # Delta[1] = 1.5
    # Delta[0] = 1.5 + 1.0*1.5 = 3.0 (delta[0] + decay[0]*Delta[1])
    # Q_ret = [3.0, 1.5]
    probs_t   = torch.ones(1, 2, 1, device="cuda")           # pi_a = 1.0
    probs_b   = torch.full((1, 2), 0.5, device="cuda")       # mu = 0.5 -> IS ratio=2, clipped to 1
    q         = torch.zeros(1, 2, device="cuda")
    q_next    = torch.full((1, 2, 1), 0.5, device="cuda")    # E_pi[Q_next] = 0.5
    actions   = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards   = torch.ones(1, 2, device="cuda")
    dones     = torch.zeros(1, 2, device="cuda")

    out = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([[3.0, 1.5]], device="cuda"), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Terminal vs. truncated boundary tests
# ---------------------------------------------------------------------------
#
# These three tests verify the key distinction:
#   terminated: s_{t+1} is a reset state — bootstrap must be zero.
#   truncated:  window ended but episode continues — bootstrap must be kept.
#
# Setup: seq_len=1, num_actions=1, gamma=1, pi=mu=1 (c=1), Q=0.
#   delta[0] = r + gamma * E_pi[Q_next] * (1-terminated) - Q
#            = r + 1 * q_next * (1-terminated)
#   decay[0] = gamma * c[1] * (1-done) = 0  (c[1] out-of-bounds)
#   Delta[0] = delta[0]
#   Q_ret[0] = Delta[0] + Q = delta[0] (Q=0 )

@cuda_only
def test_retrace_terminal_no_bootstrap():
    # True termination: done=1, no truncateds supplied.
    # terminated=1 -> bootstrap term zeroed.
    # delta[0] = 1 + 1*5*(1-1) - 0 = 1
    # Q_ret = [1]
    probs_t = torch.ones(1, 1, 1, device="cuda")
    probs_b = torch.ones(1, 1, device="cuda")
    q       = torch.zeros(1, 1, device="cuda")
    q_next  = torch.full((1, 1, 1), 5.0, device="cuda")
    actions = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
    rewards = torch.ones(1, 1, device="cuda")
    dones   = torch.ones(1, 1, device="cuda")   # terminated

    out = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([[1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_truncated_keeps_bootstrap():
    # Time-limit truncation: done=1, truncated=1.
    # terminated = done - truncated = 0 -> bootstrap kept.
    # delta[0] = 1 + 1*5*(1-0) - 0 = 6
    # Q_ret = [6]
    probs_t    = torch.ones(1, 1, 1, device="cuda")
    probs_b    = torch.ones(1, 1, device="cuda")
    q          = torch.zeros(1, 1, device="cuda")
    q_next     = torch.full((1, 1, 1), 5.0, device="cuda")
    actions    = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
    rewards    = torch.ones(1, 1, device="cuda")
    dones      = torch.ones(1, 1, device="cuda")   # boundary
    truncateds = torch.ones(1, 1, device="cuda")   # truncated, not terminated

    out = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, dones,
                                  gamma=1.0, truncateds=truncateds)
    torch.testing.assert_close(out, torch.tensor([[6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_truncated_c_boundary_zero_bootstrap_kept():
    # Truncated boundary with c[T]=0: trace continuation is stopped but the
    # one-step bootstrap inside delta is unaffected.
    # seq_len=2: t=0 interior, t=1 truncated boundary.
    # pi=mu=1 (c=1), Q=0, q_next=5 everywhere, rewards=1, gamma=1.
    #
    # t=1 (truncated boundary): done=1, truncated=1 -> terminated=0
    #   delta[1] = 1 + 1*5*(1-0) - 0 = 6
    #   decay[1] = gamma*c[2]*(1-done[1]) = 0  (c[2] out-of-bounds AND done=1)
    #   Delta[1] = 6
    #
    # t=0 (interior): done=0
    #   delta[0] = 1 + 1*5*(1-0) - 0 = 6
    #   decay[0] = gamma*c[1]*(1-done[0]) = 1*1*1 = 1
    #   Delta[0] = 6 + 1*6 = 12
    #
    # Q_ret = [12, 6]
    probs_t    = torch.ones(1, 2, 1, device="cuda")
    probs_b    = torch.ones(1, 2, device="cuda")
    q          = torch.zeros(1, 2, device="cuda")
    q_next     = torch.full((1, 2, 1), 5.0, device="cuda")
    actions    = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards    = torch.ones(1, 2, device="cuda")
    dones      = torch.tensor([[0.0, 1.0]], device="cuda")
    truncateds = torch.tensor([[0.0, 1.0]], device="cuda")

    out = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, dones,
                                  gamma=1.0, truncateds=truncateds)
    torch.testing.assert_close(out, torch.tensor([[12.0, 6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Correctness vs reference
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len,num_actions", [
    (1,   1,  2),
    (1,   7,  4),
    (4,   128, 8),
    (32,  333, 4),
    (128, 1024, 6),
])
def test_retrace_correctness_shapes(num_envs, seq_len, num_actions):
    args = _make_inputs(num_envs, seq_len, num_actions=num_actions, seed=42)
    expected = reference_retrace(*args, gamma=0.99)
    actual   = compute_retrace(*args, gamma=0.99)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_lambda():
    """lambda_ < 1 reduces the effective trace length."""
    args     = _make_inputs(16, 256, seed=4)
    expected = reference_retrace(*args, gamma=0.99, lambda_=0.5)
    actual   = compute_retrace(*args, gamma=0.99, lambda_=0.5)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_clipping():
    """c_bar < 1 clips IS ratios; result must match the reference with the same clip."""
    args     = _make_inputs(16, 256, seed=5)
    expected = reference_retrace(*args, gamma=0.99, c_bar=0.5)
    actual   = compute_retrace(*args, gamma=0.99, c_bar=0.5)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_on_policy_matches_td():
    """On-policy (pi=mu, lambda=1, c_bar=inf): reduces to n-step TD with gamma-discounted sum."""
    torch.manual_seed(6)
    num_envs, seq_len, num_actions = 8, 64, 4
    # pi = mu: uniform over actions, behavior = 1/num_actions.
    probs_t   = torch.full((num_envs, seq_len, num_actions), 1.0 / num_actions, device="cuda")
    probs_b   = torch.full((num_envs, seq_len), 1.0 / num_actions, device="cuda")
    q         = torch.randn(num_envs, seq_len, device="cuda")
    q_next    = torch.randn(num_envs, seq_len, num_actions, device="cuda")
    actions   = torch.randint(0, num_actions, (num_envs, seq_len), device="cuda")
    rewards   = torch.randn(num_envs, seq_len, device="cuda")
    dones     = torch.zeros(num_envs, seq_len, device="cuda")

    args = probs_t, probs_b, q, q_next, actions, rewards, dones
    expected = reference_retrace(*args, gamma=0.99, lambda_=1.0, c_bar=1e6)
    actual   = compute_retrace(*args, gamma=0.99, lambda_=1.0, c_bar=1e6)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_non_contiguous_input():
    """Wrapper must handle non-contiguous inputs via .contiguous()."""
    torch.manual_seed(7)
    base = torch.randn(16, 256, 4, 2, device="cuda")
    probs_t   = torch.softmax(base[..., 0], dim=-1)   # non-contiguous
    probs_b   = torch.rand(16, 256, 2, device="cuda")[..., 0]
    q         = torch.randn(16, 256, 2, device="cuda")[..., 0]
    q_next    = torch.randn(16, 256, 4, 2, device="cuda")[..., 0]
    actions   = torch.randint(0, 4, (16, 256), device="cuda")
    rewards   = torch.randn(16, 256, 2, device="cuda")[..., 0]
    dones     = torch.zeros(16, 256, 2, device="cuda")[..., 0]

    args = probs_t, probs_b, q, q_next, actions, rewards, dones

    def contiguous(t):
        return t.contiguous()

    expected = reference_retrace(*[contiguous(a) for a in args], gamma=0.99)
    actual   = compute_retrace(*args, gamma=0.99)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def _np_to_triton_retrace(*args_gpu, gamma, lambda_=1.0, c_bar=1.0):
    """NumPy → GPU Triton → NumPy round-trip: measures full adoption-path latency."""
    args_np = tuple(
        a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in args_gpu
    )
    apt_np, apb_np, q_np, nqa_np, act_np, r_np, d_np = args_np
    to_f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.float32)
    to_i = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.int64)
    out = compute_retrace(
        to_f(apt_np), to_f(apb_np), to_f(q_np), to_f(nqa_np),
        to_i(act_np), to_f(r_np), to_f(d_np),
        gamma=gamma, lambda_=lambda_, c_bar=c_bar,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


BENCH_CONFIGS = [
    (64,  512),
    (128, 1024),
    (256, 1024),
    (512, 2048),
]


@cuda_only
@pytest.mark.slow
def test_retrace_performance():
    """
    Sweep over (num_envs, seq_len) configs comparing:

      triton            — Triton scan kernel  (CUDA events)
      pt.compile(vec)   — torch.compile on vectorized_retrace  (CUDA events)
      pt.compile(loop)  — torch.compile on reference_retrace  (wall-clock)
      np→triton→np      — NumPy → GPU → NumPy adoption path  (wall-clock)
      numpy(cpu)        — reference_retrace on CPU tensors  (wall-clock)

    triton and pt.compile(vec) are fully vectorized — no Python loop — so CUDA
    events are appropriate for both.  pt.compile(loop) and np→triton→np include
    CPU stalls; wall-clock captures those correctly.

    Assertions:
      - Triton must be >=1.5x faster than pt.compile(vec) — the strongest
        fully-vectorized PyTorch baseline.
    """
    compiled_vec  = torch.compile(vectorized_retrace)
    compiled_loop = torch.compile(reference_retrace)

    _args = _make_inputs(64, 512)
    compiled_vec(*_args, gamma=0.99)
    compiled_loop(*_args, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>8} {'compile(vec)':>14} {'compile(loop)':>15} "
        f"{'np→tri→np':>12} {'numpy(cpu)':>12} "
        f"{'vs vec':>8} {'vs loop':>9} {'e2e vs np':>11} {'vs numpy':>10}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_vec = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        args_cpu = _make_inputs(num_envs, seq_len, device="cpu")
        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)

        triton_ms   = _bench_gpu(compute_retrace,  *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
        vec_ms      = _bench_gpu(compiled_vec,             *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
        loop_ms     = _bench_cpu(compiled_loop,            *args_gpu, gamma=0.99)
        e2e_ms      = _bench_cpu(_np_to_triton_retrace,   *args_gpu, gamma=0.99)
        numpy_ms    = _bench_cpu(reference_retrace,        *args_cpu, gamma=0.99)

        su_vec   = vec_ms   / triton_ms
        su_loop  = loop_ms  / triton_ms
        su_e2e   = numpy_ms / e2e_ms
        su_numpy = numpy_ms / triton_ms
        all_speedups_vec.append(su_vec)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>7.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms "
            f"{e2e_ms:>11.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_e2e:>9.1f}x {su_numpy:>9.1f}x"
        )

    print(
        "\ntriton          : CUDA events — pure kernel time, no CPU overhead."
        "\ncompile(vec)    : CUDA events — vectorized log-space cumsum, no Python loop."
        "\ncompile(loop)   : wall-clock  — one CUDA op per timestep from Python;"
        "\n                  CUDA events would miss the CPU stall."
        "\nnp→triton→np    : wall-clock  — NumPy → GPU → NumPy adoption path."
        "\nnumpy(cpu)      : wall-clock  — reference loop on CPU tensors."
        "\nspeedups vs triton kernel; e2e vs np = numpy_cpu / np→triton→np."
    )

    assert min(all_speedups_vec) >= 1.5, (
        f"Expected >=1.5x speedup over pt.compile(vec) across all configs, "
        f"worst was {min(all_speedups_vec):.2f}x"
    )
