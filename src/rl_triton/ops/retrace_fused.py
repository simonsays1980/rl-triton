import torch
import triton

from rl_triton.kernels.retrace_fused import retrace_fused_kernel

_FLAT_MAX_SEQ_LEN = 131072
_WARPS = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def compute_retrace_fused(
    action_probs_target: torch.Tensor,
    action_probs_behavior: torch.Tensor,
    q_values: torch.Tensor,
    next_q_values_all: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    rho_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully-fused Retrace(λ) via a single Triton kernel.

    Computes E_π[Q(s_{t+1},a)], IS ratios, u[t], v[t], the backward
    associative scan, Q-value targets, and advantages all in one kernel —
    no intermediate tensor allocations.

    Only valid for seq_len <= 131072.  Use compute_retrace for longer
    sequences (it auto-dispatches here for short sequences and falls back to
    the chunked path otherwise).

    Args:
        action_probs_target:   [num_envs, seq_len, num_actions], float32, CUDA.
        action_probs_behavior: [num_envs, seq_len], float32, CUDA.
        q_values:              [num_envs, seq_len], float32, CUDA.
        next_q_values_all:     [num_envs, seq_len, num_actions], float32, CUDA.
        actions:               [num_envs, seq_len], int64, CUDA.
        rewards:               [num_envs, seq_len], float32, CUDA.
        dones:                 [num_envs, seq_len], float32, CUDA.
        terminated:            [num_envs, seq_len], float32, CUDA.
                               Same as dones when no truncateds; otherwise
                               zeros at truncated steps so bootstrap is kept.
        gamma:                 Discount factor.
        lambda_:               Trace decay parameter.
        c_bar:                 IS ratio clip for trace weights.
        rho_bar:               IS ratio clip for advantage scaling.

    Returns:
        retrace_targets: Q_ret[t], shape [num_envs, seq_len], float32.
        advantages:      A[t],     shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = rewards.shape
    num_actions = action_probs_target.shape[2]

    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds flat kernel limit {_FLAT_MAX_SEQ_LEN}."
    )

    out        = torch.empty_like(rewards)
    advantages = torch.empty_like(rewards)

    BLOCK_SIZE   = triton.next_power_of_2(seq_len)
    ACTION_BLOCK = triton.next_power_of_2(num_actions)
    num_warps    = _WARPS.get(BLOCK_SIZE, 16)
    num_stages   = 2 if BLOCK_SIZE >= 2048 else 1

    retrace_fused_kernel[(num_envs,)](
        action_probs_target,
        action_probs_behavior,
        q_values,
        next_q_values_all,
        actions,
        rewards,
        dones,
        terminated,
        out,
        advantages,
        seq_len,
        num_actions,
        rewards.stride(0),
        action_probs_target.stride(0),
        gamma=gamma,
        lambda_=lambda_,
        c_bar=c_bar,
        rho_bar=rho_bar,
        BLOCK_SIZE=BLOCK_SIZE,
        ACTION_BLOCK=ACTION_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, advantages
