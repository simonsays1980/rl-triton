import torch
import triton

from rl_triton.kernels.gae import gae_fused_kernel
from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN

_WARPS = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence:
      A[t] = delta[t] + gamma * lambda * (1 - done[t]) * A[t+1],  A[T] = bootstrap

    where delta[t] = r[t] + gamma * (1 - done[t]) * V(s_{t+1}) - V(s_t).

    V(s_{t+1}) is read directly from values[:, t+1]; the caller does NOT need to
    pass a separate next_values tensor.  For the last timestep, V(s_T) is taken
    from bootstrap_values (which is also A[T] = 0 for terminated episodes, or
    V(s_T) for truncated ones).

    Maps to the shared linear recurrence A[t] = u[t] + v[t] * A[t+1] with:
      u[t] = delta[t]
      v[t] = gamma * lambda * (1 - done[t])

    Dispatches to the fully-fused single-block kernel for seq_len <= 131072
    (delta and decay computed inside the kernel, no intermediate tensors),
    and falls back to the chunked scan + PyTorch elementwise ops for longer
    sequences.

    Terminated vs. truncated episodes
    ----------------------------------
    Pass only true terminations in `dones` (not truncations).  For a terminated
    step, `(1 - done[t])` zeroes the bootstrap, reflecting that s_{t+1} has no
    value.  For a truncated step, `done[t]` must be 0 so the bootstrap
    V(s_{t+1}) = values[:, t+1] is kept.  If the truncation falls on the last
    rollout step, pass V(s_T) via `bootstrap_values` instead.

    Autoreset mode and loss masking (next-step mode only)
    -----------------------------------------------------
    With Gymnasium's next-step autoreset, `step()` returns the terminal
    observation s_terminal as `next_obs` when `terminated=True`, and the actual
    reset is deferred to the following `step()` call.  This means the rollout
    buffer at position t+1 after termination contains s_terminal as the current
    observation, while the action taken on it is silently ignored by the
    environment.

    Two consequences for the caller:

    1. Terminated boundary — `obs[t+1] = s_terminal` is stale: the policy acted
       on it but the env discarded that action.  The advantage at position t+1 is
       meaningless and must be masked from the actor and critic losses:

         episode_over = terminated | truncated
         mask = ~episode_over  # shape [num_envs, seq_len], shift by rollout below
         loss = (advantages * mask).mean()

    2. Truncated boundary — `obs[t+1]` is the genuine continuation state, so
       V(s_{t+1}) = values[:, t+1] is a valid bootstrap and no observation is
       stale.  However, if the truncation falls on the last step of a rollout
       window, the GAE accumulator from the old window would bleed into the new
       one.  Apply the same mask at rollout boundaries regardless:

         mask = np.concatenate([initial_episode_over, ~episode_over[:-1]])

    Same-step autoreset does not produce stale observations and needs no masking.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), same shape, float32, CUDA.
                          V(s_{t+1}) is read as values[:, t+1] inside the kernel.
        dones:            True termination flags only (1.0=terminated), same shape,
                          float32.  Do NOT pass terminated | truncated here; pass
                          truncated episodes with done=0 and supply V(s_T) via
                          bootstrap_values instead.
        gamma:            Discount factor.
        lambda_:          GAE trace parameter in [0, 1].
        bootstrap_values: Per-environment V(s_T) / boundary value A[T], shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Also used as V(s_{t+1}) at t=T-1 inside the kernel.
                          Defaults to zeros.

    Returns:
        advantages: A[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [("rewards", rewards), ("values", values), ("dones", dones)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape

    rewards = rewards.contiguous()
    values  = values.contiguous()
    dones   = dones.contiguous()

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=rewards.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    out = torch.empty_like(rewards)

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1

        gae_fused_kernel[(num_envs,)](
            rewards, values, dones,
            out,
            bootstrap_values,
            seq_len,
            rewards.stride(0),
            gamma=gamma,
            lambda_=lambda_,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # Chunked path for seq_len > 131072: PyTorch pre-computation + chunked scan.
    not_done    = 1.0 - dones
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = bootstrap_values
    deltas = rewards + gamma * not_done * next_values - values
    decays = gamma * lambda_ * not_done
    return _run_scan(deltas, decays, bootstrap_values)
