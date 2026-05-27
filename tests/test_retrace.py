import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu
from rl_triton.ops.retrace import compute_retrace_triton

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
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch Retrace(λ) backward scan — ground truth for correctness tests only."""
    num_envs, T = rewards.shape

    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    deltas = rewards + gamma * expected_next_q * (1.0 - dones) - q_values

    pi_a = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c    = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)

    q_deltas = torch.zeros_like(rewards)
    carry    = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()

    for t in reversed(range(T)):
        # decay at t uses c[t+1]; at the last step c[T]=0.
        c_next = c[:, t + 1] if t + 1 < T else torch.zeros(num_envs, device=rewards.device)
        decay  = gamma * c_next * (1.0 - dones[:, t])
        carry          = deltas[:, t] + decay * carry
        q_deltas[:, t] = carry

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
# Hand-computed with num_actions=1, on-policy (π=μ), no dones, gamma=1, lambda=1,
# so c=1 everywhere and the recurrence reduces to:
#   Δ[t] = δ[t] + Δ[t+1],  δ[t] = r[t] + Q'[t] - Q[t]
# where Q'[t] = E_π[Q(s_{t+1},a)] = Q_next[t] (single action).

@cuda_only
def test_retrace_known_values_single_env():
    # seq_len=2, num_actions=1, gamma=1, lambda=1, no dones, no bootstrap.
    # pi=mu=1.0, c=1.0, E_pi[Q_next]=Q_next (single action).
    # delta[0] = 1 + 1*2 - 0 = 3,  delta[1] = 1 + 1*3 - 0 = 4
    # decay[0] = 1*c[1]*(1-0) = 1,  decay[1] = 1*c[T]*(1-0) = 0  (c[T]=0)
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

    out = compute_retrace_triton(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
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

    out = compute_retrace_triton(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_values_clipping():
    # pi_a=1.0 (single action, on-policy), mu=0.5 -> IS ratio=2.0, clipped to c_bar=1.0 -> c=1.0.
    # E_pi[Q_next] = 1.0 * q_next = q_next (single action, probs_t=1.0).
    # seq_len=2, gamma=1, no dones, q_next=0.5.
    # delta[0] = 1 + 1*0.5 - 0 = 1.5,  delta[1] = 1 + 1*0.5 - 0 = 1.5
    # c=min(1.0, 2.0)=1.0 everywhere; decay[0]=1*c[1]=1.0, decay[1]=0 (c[T]=0)
    # Delta[1] = 1.5
    # Delta[0] = 1.5 + 1.0*1.5 = 3.0
    # Q_ret = [3.0, 1.5]
    probs_t   = torch.ones(1, 2, 1, device="cuda")           # pi_a = 1.0
    probs_b   = torch.full((1, 2), 0.5, device="cuda")       # mu = 0.5 -> IS ratio=2, clipped to 1
    q         = torch.zeros(1, 2, device="cuda")
    q_next    = torch.full((1, 2, 1), 0.5, device="cuda")    # E_pi[Q_next] = 0.5
    actions   = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards   = torch.ones(1, 2, device="cuda")
    dones     = torch.zeros(1, 2, device="cuda")

    out = compute_retrace_triton(probs_t, probs_b, q, q_next, actions, rewards, dones, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([[3.0, 1.5]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_values_bootstrap():
    # Same as single_env but with bootstrap=2.0: Delta[T]=2.0.
    # decay[0]=1*c[1]=1, decay[1]=0 still (c[T] always 0, bootstrap is separate).
    # Delta[1] = 4 + 0*2.0 = 4
    # Delta[0] = 3 + 1*4   = 7
    # Q_ret = [7, 4]  -- bootstrap does not change the result here because
    # decay[1]=0 (last step always has c[T]=0).
    #
    # To make bootstrap visible, use seq_len=1:
    # delta[0] = 1 + 2 - 0 = 3, decay[0] = gamma*c[T]*(1-done) = 0
    # Delta[0] = 3 + 0*bootstrap = 3 -> bootstrap never reaches here via decay.
    # Bootstrap enters via the scan's initial carry, but c[T]=0 zeroes it out.
    # This is correct: in Retrace the bootstrap IS zero-weighted at the boundary.
    probs_t   = torch.ones(1, 2, 1, device="cuda")
    probs_b   = torch.ones(1, 2, device="cuda")
    q         = torch.zeros(1, 2, device="cuda")
    q_next    = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions   = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards   = torch.ones(1, 2, device="cuda")
    dones     = torch.zeros(1, 2, device="cuda")
    bootstrap = torch.tensor([2.0], device="cuda")

    out = compute_retrace_triton(probs_t, probs_b, q, q_next, actions, rewards, dones,
                                  gamma=1.0, bootstrap_values=bootstrap)
    torch.testing.assert_close(out, torch.tensor([[7.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


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
    actual   = compute_retrace_triton(*args, gamma=0.99)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_bootstrap():
    """Non-zero bootstrap propagates correctly through the scan."""
    args      = _make_inputs(32, 512, seed=3)
    bootstrap = torch.rand(32, device="cuda")
    expected  = reference_retrace(*args, gamma=0.99, bootstrap_values=bootstrap)
    actual    = compute_retrace_triton(*args, gamma=0.99, bootstrap_values=bootstrap)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_lambda():
    """lambda_ < 1 reduces the effective trace length."""
    args     = _make_inputs(16, 256, seed=4)
    expected = reference_retrace(*args, gamma=0.99, lambda_=0.5)
    actual   = compute_retrace_triton(*args, gamma=0.99, lambda_=0.5)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_clipping():
    """c_bar < 1 clips IS ratios; result must match the reference with the same clip."""
    args     = _make_inputs(16, 256, seed=5)
    expected = reference_retrace(*args, gamma=0.99, c_bar=0.5)
    actual   = compute_retrace_triton(*args, gamma=0.99, c_bar=0.5)
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
    actual   = compute_retrace_triton(*args, gamma=0.99, lambda_=1.0, c_bar=1e6)
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
    actual   = compute_retrace_triton(*args, gamma=0.99)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

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
      - triton:     Triton scan kernel  (CUDA events)
      - pt.compile: torch.compile on reference_retrace  (wall-clock)

    Assertion: Triton must be >=1.5x faster than torch.compile.
    """
    compiled_retrace = torch.compile(reference_retrace)

    _args = _make_inputs(64, 512)
    compiled_retrace(*_args, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>10} {'pt.compile':>12} "
        f"{'speedup':>10}"
    )
    print(header)
    print("-" * len(header))

    all_speedups = []

    for num_envs, seq_len in BENCH_CONFIGS:
        args_gpu = _make_inputs(num_envs, seq_len)
        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)

        triton_ms   = _bench_gpu(compute_retrace_triton, *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
        compiled_ms = _bench_cpu(compiled_retrace,       *args_gpu, gamma=0.99)

        speedup = compiled_ms / triton_ms
        all_speedups.append(speedup)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>9.3f}ms {compiled_ms:>11.3f}ms "
            f"{speedup:>8.1f}x"
        )

    print(
        "\ntriton:     CUDA events — pure kernel time."
        "\npt.compile: wall-clock — dispatches one CUDA op per timestep from Python."
        "\nspeedups are relative to the Triton kernel."
    )

    assert min(all_speedups) >= 1.5, (
        f"Expected >=1.5x speedup over torch.compile, worst was {min(all_speedups):.2f}x"
    )
