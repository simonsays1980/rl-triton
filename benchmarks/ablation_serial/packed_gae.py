"""Packed-environment tree GAE kernel (Task 3).

Identical math to the shipped gae_fused_kernel plain path (HAS_TRUNCATIONS=False,
HAS_BOOTSTRAP=True; src/rl_triton/kernels/gae.py) and the same combine function
(rl_triton.kernels.scan._combine). The only change is the program geometry: each
program owns a [BLOCK_E, BLOCK_T] tile -- BLOCK_E env rows instead of one -- so the
grid is cdiv(num_envs, BLOCK_E) programs instead of num_envs. The scan runs along
axis=1 (time, reversed), so dependency depth is still log2(BLOCK_T) and each input
element is still read once from HBM into registers, each output written once.
"""
import torch
import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def gae_packed_kernel(
    rewards_ptr, values_ptr, terminateds_ptr,
    out_ptr,
    bootstrap_ptr,
    num_envs,
    seq_len,
    stride_env,
    gamma,
    lambda_,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid    = tl.program_id(0)
    env    = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    offs   = tl.arange(0, BLOCK_T)
    rev    = seq_len - 1 - offs
    env_ok = env < num_envs

    base = env.to(tl.int64)[:, None] * stride_env
    mask = env_ok[:, None] & (offs < seq_len)[None, :]
    ptr  = base + rev[None, :]

    r          = tl.load(rewards_ptr     + ptr, mask=mask, other=0.0)
    v          = tl.load(values_ptr      + ptr, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr + ptr, mask=mask, other=1.0)

    not_terminated = 1.0 - terminated
    bootstrap  = tl.load(bootstrap_ptr + env, mask=env_ok, other=0.0)
    # values[t+1] for interior steps; the boundary lane (offs==0, t=T-1) gets the
    # bootstrap as next-state value. Carry A[T] = 0, exactly as the shipped kernel.
    v_next_raw = tl.load(values_ptr + ptr + 1, mask=mask & (offs > 0)[None, :], other=0.0)
    v_next = tl.where((offs == 0)[None, :], bootstrap[:, None], v_next_raw)

    delta = r + gamma * v_next * not_terminated - v
    decay = gamma * lambda_ * not_terminated

    out_local, _ = tl.associative_scan((delta, decay), axis=1, combine_fn=_combine)
    tl.store(out_ptr + ptr, out_local, mask=mask)


def gae_packed(rewards, values, terminateds, last_value, gamma, lambda_, BLOCK_E=4, num_warps=2):
    """Thin wrapper, same style as serial_gae.py: allocate, launch once, return."""
    num_envs, seq_len = rewards.shape
    out = torch.empty_like(rewards)
    gae_packed_kernel[(triton.cdiv(num_envs, BLOCK_E),)](
        rewards, values, terminateds, out, last_value,
        num_envs, seq_len, rewards.stride(0),
        gamma, lambda_,
        BLOCK_E=BLOCK_E, BLOCK_T=triton.next_power_of_2(seq_len), num_warps=num_warps,
    )
    return out
