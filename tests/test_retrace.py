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
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    rho_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch Retrace(λ) backward scan — ground truth for correctness tests only."""
    num_envs, T = rewards.shape

    dones = (terminateds + truncateds).clamp(max=1.0)

    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    deltas = rewards + gamma * expected_next_q * (1.0 - terminateds) - q_values

    pi_a = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c    = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    rho  = torch.clamp(pi_a / action_probs_behavior, max=rho_bar)

    q_deltas = torch.zeros_like(rewards)
    carry    = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        c_next = c[:, t + 1] if t + 1 < T else torch.zeros(num_envs, device=rewards.device)
        decay          = gamma * c_next * (1.0 - dones[:, t])
        carry          = deltas[:, t] + decay * carry
        q_deltas[:, t] = carry

    retrace_targets = q_deltas + q_values

    next_q_ret = torch.zeros_like(retrace_targets)
    next_q_ret[:, :-1] = retrace_targets[:, 1:]
    advantages = rho * (rewards + gamma * next_q_ret * (1.0 - terminateds) - q_values)

    return retrace_targets, advantages


def vectorized_retrace(
    action_probs_target: torch.Tensor,
    action_probs_behavior: torch.Tensor,
    q_values: torch.Tensor,
    next_q_values_all: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    rho_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    dones = (terminateds + truncateds).clamp(max=1.0)

    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    u = rewards + gamma * expected_next_q * (1.0 - terminateds) - q_values

    pi_a = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c    = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    rho  = torch.clamp(pi_a / action_probs_behavior, max=rho_bar)

    c_next          = torch.empty_like(c)
    c_next[:, :-1]  = c[:, 1:]
    c_next[:, -1]   = 0.0
    v = gamma * c_next * (1.0 - dones)

    log_suffix   = torch.flip(
        torch.cumsum(torch.flip(torch.log(v.clamp(min=1e-38)), [1]), dim=1), [1]
    )
    weights      = torch.exp(log_suffix)
    q_deltas     = torch.flip(
        torch.cumsum(torch.flip(u * weights, [1]), dim=1), [1]
    ) / weights

    retrace_targets = q_deltas + q_values

    next_q_ret = torch.zeros_like(retrace_targets)
    next_q_ret[:, :-1] = retrace_targets[:, 1:]
    advantages = rho * (rewards + gamma * next_q_ret * (1.0 - terminateds) - q_values)

    return retrace_targets, advantages


def _make_inputs(num_envs, seq_len, num_actions=4, device="cuda", seed=0):
    torch.manual_seed(seed)
    action_probs_target   = torch.softmax(torch.randn(num_envs, seq_len, num_actions, device=device), dim=-1)
    action_probs_behavior = torch.rand(num_envs, seq_len, device=device) * 0.8 + 0.1
    q_values              = torch.randn(num_envs, seq_len, device=device)
    next_q_values_all     = torch.randn(num_envs, seq_len, num_actions, device=device)
    actions               = torch.randint(0, num_actions, (num_envs, seq_len), device=device)
    rewards               = torch.randn(num_envs, seq_len, device=device)
    terminateds           = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    truncateds            = torch.zeros_like(terminateds)
    return action_probs_target, action_probs_behavior, q_values, next_q_values_all, actions, rewards, terminateds, truncateds


# ---------------------------------------------------------------------------
# Ground-truth value tests (targets)
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
    # decay[1] = 1*c[2]*(1-0) = 0  — c[T] is out-of-bounds, forced to 0.
    # Delta[1] = 4,  Delta[0] = 3 + 1*4 = 7
    # Q_ret = [7, 4]
    probs_t     = torch.ones(1, 2, 1, device="cuda")
    probs_b     = torch.ones(1, 2, device="cuda")
    q           = torch.zeros(1, 2, device="cuda")
    q_next      = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions     = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 2, device="cuda")
    terminateds = torch.zeros(1, 2, device="cuda")
    truncateds  = torch.zeros(1, 2, device="cuda")

    targets, _ = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(targets, torch.tensor([[7.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_values_done():
    # terminated[0]=1 cuts the trace: decay[0]=0, delta[0]=1+2*(1-1)=1.
    # Q_ret = [1, 4]
    probs_t     = torch.ones(1, 2, 1, device="cuda")
    probs_b     = torch.ones(1, 2, device="cuda")
    q           = torch.zeros(1, 2, device="cuda")
    q_next      = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions     = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 2, device="cuda")
    terminateds = torch.tensor([[1.0, 0.0]], device="cuda")
    truncateds  = torch.zeros(1, 2, device="cuda")

    targets, _ = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(targets, torch.tensor([[1.0, 4.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_values_clipping():
    # pi_a=1.0, mu=0.5 -> IS ratio=2.0, clipped to c_bar=1.0.
    # delta[t] = 1 + 0.5 = 1.5; c=1 everywhere; decay[0]=1, decay[1]=0.
    # Delta[1]=1.5, Delta[0]=3.0.  Q_ret = [3.0, 1.5]
    probs_t     = torch.ones(1, 2, 1, device="cuda")
    probs_b     = torch.full((1, 2), 0.5, device="cuda")
    q           = torch.zeros(1, 2, device="cuda")
    q_next      = torch.full((1, 2, 1), 0.5, device="cuda")
    actions     = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 2, device="cuda")
    terminateds = torch.zeros(1, 2, device="cuda")
    truncateds  = torch.zeros(1, 2, device="cuda")

    targets, _ = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(targets, torch.tensor([[3.0, 1.5]], device="cuda"), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Terminal vs. truncated boundary tests
# ---------------------------------------------------------------------------

@cuda_only
def test_retrace_terminal_no_bootstrap():
    # terminated=1 -> bootstrap zeroed; delta[0] = 1 + 5*(1-1) = 1
    probs_t     = torch.ones(1, 1, 1, device="cuda")
    probs_b     = torch.ones(1, 1, device="cuda")
    q           = torch.zeros(1, 1, device="cuda")
    q_next      = torch.full((1, 1, 1), 5.0, device="cuda")
    actions     = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 1, device="cuda")
    terminateds = torch.ones(1, 1, device="cuda")
    truncateds  = torch.zeros(1, 1, device="cuda")

    targets, _ = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(targets, torch.tensor([[1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_truncated_keeps_bootstrap():
    # terminated=0, truncated=1 -> bootstrap kept; delta[0] = 1 + 5 = 6
    probs_t     = torch.ones(1, 1, 1, device="cuda")
    probs_b     = torch.ones(1, 1, device="cuda")
    q           = torch.zeros(1, 1, device="cuda")
    q_next      = torch.full((1, 1, 1), 5.0, device="cuda")
    actions     = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 1, device="cuda")
    terminateds = torch.zeros(1, 1, device="cuda")
    truncateds  = torch.ones(1, 1, device="cuda")

    targets, _ = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(targets, torch.tensor([[6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_truncated_c_boundary_zero_bootstrap_kept():
    # seq_len=2: t=1 truncated (done severs trace, bootstrap kept), t=0 interior.
    # delta[1]=6, Delta[1]=6; delta[0]=6, decay[0]=1, Delta[0]=12.
    # Q_ret = [12, 6]
    probs_t     = torch.ones(1, 2, 1, device="cuda")
    probs_b     = torch.ones(1, 2, device="cuda")
    q           = torch.zeros(1, 2, device="cuda")
    q_next      = torch.full((1, 2, 1), 5.0, device="cuda")
    actions     = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 2, device="cuda")
    terminateds = torch.tensor([[0.0, 0.0]], device="cuda")
    truncateds  = torch.tensor([[0.0, 1.0]], device="cuda")

    targets, _ = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(targets, torch.tensor([[12.0, 6.0]], device="cuda"), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Advantage ground-truth tests
# ---------------------------------------------------------------------------
#
# A[t] = ρ_t · (r_t + γ · Q_ret[t+1] · (1-terminated[t]) - Q(s_t,a_t))
# On-policy (pi=mu=1, rho=1), no termination, gamma=1.

@cuda_only
def test_retrace_known_advantages_single_env():
    # seq_len=2, pi=mu=1, Q=0, no dones, gamma=1, rho=1.
    # From test_retrace_known_values_single_env: Q_ret = [7, 4].
    # A[0] = 1*(1 + 1*Q_ret[1]*(1-0) - 0) = 1 + 4 = 5
    # A[1] = 1*(1 + 1*Q_ret[2]*(1-0) - 0) = 1 + 0 = 1  (Q_ret[T] = 0)
    probs_t     = torch.ones(1, 2, 1, device="cuda")
    probs_b     = torch.ones(1, 2, device="cuda")
    q           = torch.zeros(1, 2, device="cuda")
    q_next      = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions     = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 2, device="cuda")
    terminateds = torch.zeros(1, 2, device="cuda")
    truncateds  = torch.zeros(1, 2, device="cuda")

    _, advantages = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(advantages, torch.tensor([[5.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


@cuda_only
def test_retrace_known_advantages_terminal():
    # At terminated[0]=1: Q_ret[1] is zeroed in the advantage formula.
    # From test_retrace_known_values_done: Q_ret = [1, 4].
    # A[0] = 1*(1 + 1*Q_ret[1]*(1-1) - 0) = 1 + 0 = 1
    # A[1] = 1*(1 + 1*Q_ret[2]*(1-0) - 0) = 1 + 0 = 1  (Q_ret[T] = 0)
    probs_t     = torch.ones(1, 2, 1, device="cuda")
    probs_b     = torch.ones(1, 2, device="cuda")
    q           = torch.zeros(1, 2, device="cuda")
    q_next      = torch.tensor([[[2.0], [3.0]]], device="cuda")
    actions     = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    rewards     = torch.ones(1, 2, device="cuda")
    terminateds = torch.tensor([[1.0, 0.0]], device="cuda")
    truncateds  = torch.zeros(1, 2, device="cuda")

    _, advantages = compute_retrace(probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds, gamma=1.0)
    torch.testing.assert_close(advantages, torch.tensor([[1.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)


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
    exp_targets, exp_adv = reference_retrace(*args, gamma=0.99)
    act_targets, act_adv = compute_retrace(*args, gamma=0.99)
    torch.testing.assert_close(act_targets, exp_targets, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_adv,     exp_adv,     atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_lambda():
    """lambda_ < 1 reduces the effective trace length."""
    args = _make_inputs(16, 256, seed=4)
    exp_targets, exp_adv = reference_retrace(*args, gamma=0.99, lambda_=0.5)
    act_targets, act_adv = compute_retrace(*args, gamma=0.99, lambda_=0.5)
    torch.testing.assert_close(act_targets, exp_targets, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_adv,     exp_adv,     atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_correctness_clipping():
    """c_bar < 1 clips IS ratios; rho_bar clips advantage scaling."""
    args = _make_inputs(16, 256, seed=5)
    exp_targets, exp_adv = reference_retrace(*args, gamma=0.99, c_bar=0.5, rho_bar=0.8)
    act_targets, act_adv = compute_retrace(*args, gamma=0.99, c_bar=0.5, rho_bar=0.8)
    torch.testing.assert_close(act_targets, exp_targets, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_adv,     exp_adv,     atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_on_policy_matches_td():
    """On-policy (pi=mu): c and rho both equal 1 (with large clip), reducing to n-step TD."""
    torch.manual_seed(6)
    num_envs, seq_len, num_actions = 8, 64, 4
    probs_t     = torch.full((num_envs, seq_len, num_actions), 1.0 / num_actions, device="cuda")
    probs_b     = torch.full((num_envs, seq_len), 1.0 / num_actions, device="cuda")
    q           = torch.randn(num_envs, seq_len, device="cuda")
    q_next      = torch.randn(num_envs, seq_len, num_actions, device="cuda")
    actions     = torch.randint(0, num_actions, (num_envs, seq_len), device="cuda")
    rewards     = torch.randn(num_envs, seq_len, device="cuda")
    terminateds = torch.zeros(num_envs, seq_len, device="cuda")
    truncateds  = torch.zeros(num_envs, seq_len, device="cuda")

    args = probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds
    exp_targets, exp_adv = reference_retrace(*args, gamma=0.99, lambda_=1.0, c_bar=1e6, rho_bar=1e6)
    act_targets, act_adv = compute_retrace(*args, gamma=0.99, lambda_=1.0, c_bar=1e6, rho_bar=1e6)
    torch.testing.assert_close(act_targets, exp_targets, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_adv,     exp_adv,     atol=1e-4, rtol=1e-4)


@cuda_only
def test_retrace_non_contiguous_input():
    """Wrapper must handle non-contiguous inputs via .contiguous()."""
    torch.manual_seed(7)
    base        = torch.randn(16, 256, 4, 2, device="cuda")
    probs_t     = torch.softmax(base[..., 0], dim=-1)
    probs_b     = torch.rand(16, 256, 2, device="cuda")[..., 0]
    q           = torch.randn(16, 256, 2, device="cuda")[..., 0]
    q_next      = torch.randn(16, 256, 4, 2, device="cuda")[..., 0]
    actions     = torch.randint(0, 4, (16, 256), device="cuda")
    rewards     = torch.randn(16, 256, 2, device="cuda")[..., 0]
    terminateds = torch.zeros(16, 256, 2, device="cuda")[..., 0]
    truncateds  = torch.zeros(16, 256, 2, device="cuda")[..., 0]

    args = probs_t, probs_b, q, q_next, actions, rewards, terminateds, truncateds

    def c(t):
        return t.contiguous()

    exp_targets, exp_adv = reference_retrace(*[c(a) for a in args], gamma=0.99)
    act_targets, act_adv = compute_retrace(*args, gamma=0.99)
    torch.testing.assert_close(act_targets, exp_targets, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_adv,     exp_adv,     atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Truncation classification note
# ---------------------------------------------------------------------------
#
# Retrace(λ) takes `truncateds` as a required positional argument — it is always
# present.  There is NO separate bootstrap_values parameter because the Q-bootstrap
# γ·E_π[Q(s_{t+1},·)] is folded into `next_q_values_all` (the caller supplies the
# full Q-table for the next state).  The truncated flag stops trace decay exactly
# as terminated does, but the one-step δ[t] keeps the next-Q term (gated only by
# terminated, not truncated) because the Q-function already provides the correct
# continuation value.
#
# Consequence: there is no distinct HAS_TRUNCATIONS fast/slow dispatch in Retrace.
# The existing test_retrace_performance already exercises `truncateds=zeros`, which
# is the realistic production case (most steps are not truncated).  A separate
# truncation-path benchmark would compare the same kernel to itself with a ~5%
# density difference — not a meaningful coverage gap.  No additional truncation
# benchmark or correctness test is added here.
#
# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def _np_to_triton_retrace(*args_gpu, gamma, lambda_=1.0, c_bar=1.0, rho_bar=1.0):
    """NumPy → GPU Triton → NumPy round-trip: measures full adoption-path latency."""
    args_np = tuple(
        a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in args_gpu
    )
    apt_np, apb_np, q_np, nqa_np, act_np, r_np, term_np, trunc_np = args_np
    to_f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.float32)
    to_i = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda", torch.int64)
    targets, adv = compute_retrace(
        to_f(apt_np), to_f(apb_np), to_f(q_np), to_f(nqa_np),
        to_i(act_np), to_f(r_np), to_f(term_np), to_f(trunc_np),
        gamma=gamma, lambda_=lambda_, c_bar=c_bar, rho_bar=rho_bar,
    )
    torch.cuda.synchronize()
    return targets.cpu().numpy(), adv.cpu().numpy()


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

    Assertions:
      - Triton must be >=1.5x faster than pt.compile(vec).
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

        triton_ms   = _bench_gpu(compute_retrace,          *args_gpu, gamma=0.99, n_warmup=gpu_warmup, n_iter=gpu_iter)
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
        "\ncompile(loop)   : wall-clock  — one CUDA op per timestep from Python."
        "\nnp→triton→np    : wall-clock  — NumPy → GPU → NumPy adoption path."
        "\nnumpy(cpu)      : wall-clock  — reference loop on CPU tensors."
        "\nspeedups vs triton kernel; e2e vs np = numpy_cpu / np→triton→np."
    )

    assert min(all_speedups_vec) >= 1.5, (
        f"Expected >=1.5x speedup over pt.compile(vec) across all configs, "
        f"worst was {min(all_speedups_vec):.2f}x"
    )
