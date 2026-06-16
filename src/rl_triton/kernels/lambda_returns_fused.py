import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def lambda_returns_fused_kernel(
    rewards_ptr, next_values_ptr, dones_ptr,
    out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    gamma,
    lambda_,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused TD(λ) returns kernel: one program per environment.

    Recurrence:
      G[t] = r[t] + γ(1-d[t]) * [(1-λ)*V(s_{t+1}) + λ*G[t+1]],  G[T] = bootstrap

    Maps to A[t] = a[t] + b[t] * A[t+1] with:
      a[t] = r[t] + γ*(1-λ)*(1-d[t])*V(s_{t+1})
      b[t] = γ*λ*(1-d[t])

    next_values_ptr holds V(s_{t+1}) pre-aligned: next_values_ptr[env, t] = V(s_{t+1}).
    The caller is responsible for setting next_values[:, T-1] to the appropriate
    terminal value (V(s_T) for truncated, 0 for terminated).

    bootstrap_ptr is the scan carry boundary G[T] and is independent of next_values.
    For terminated episodes, both should be 0. For truncated episodes, both should
    be V(s_T) — callers can pass bootstrap_values=next_values[:, -1].

    Fuses the three separate PyTorch ops (not_done, u, v) and the scan into one
    kernel, eliminating intermediate tensor writes and re-reads.

    Indexing (reversed):
      offs = 0, 1, …, BLOCK_SIZE-1
      rev  = seq_len - 1 - offs      (offs=0 → t=T-1)

    Args:
        rewards_ptr:     [num_envs, seq_len], float32.
        next_values_ptr: V(s_{t+1}), same shape. next_values_ptr[env, t] = V(s_{t+1}).
        dones_ptr:       Episode termination flags (1.0=done), same shape.
        out_ptr:         Output G[t], same shape.
        bootstrap_ptr:   Scan boundary G[T] per env, [num_envs], float32.
        seq_len:         Number of timesteps.
        stride_env:      Row stride in elements.
        gamma:           Discount factor.
        lambda_:         Trace parameter.
        BLOCK_SIZE:      Power-of-2 >= seq_len (constexpr).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs
    mask = offs < seq_len

    r    = tl.load(rewards_ptr     + base + rev, mask=mask, other=0.0)
    nv   = tl.load(next_values_ptr + base + rev, mask=mask, other=0.0)
    done = tl.load(dones_ptr       + base + rev, mask=mask, other=1.0)

    not_done = 1.0 - done
    u     = r + gamma * (1.0 - lambda_) * not_done * nv
    decay = gamma * lambda_ * not_done

    out_local, decay_prod = tl.associative_scan((u, decay), axis=0, combine_fn=_combine)

    bootstrap = tl.load(bootstrap_ptr + env_idx)
    tl.store(out_ptr + base + rev, out_local + decay_prod * bootstrap, mask=mask)
