"""Fused SERIAL GAE kernels for the tree-vs-serial ablation.

Same semantics as gae_fused_kernel's plain path (HAS_TRUNCATIONS=False,
HAS_BOOTSTRAP=True), src/rl_triton/kernels/gae.py:

  delta[t] = r[t] + gamma*(1-term[t])*V(s_{t+1}) - V(s_t),  V(s_T) = bootstrap
  beta[t]  = gamma*lambda*(1-term[t])
  A[t]     = delta[t] + beta[t]*A[t+1],  A[T] = 0

Both kernels: one launch, every input element read once, output written once,
delta/beta built in-register, carry in registers. They differ from the tree
kernel ONLY in dependency depth (T serial steps vs. log2(T) scan levels) and
in access pattern (one program owns BLOCK_E envs and walks time, instead of
one program per env holding the whole time axis).

V(s_{t+1}) is not re-loaded: the V(s_t) loaded at step t+1 of the backward
walk is kept in a register and reused as next-state value at step t. That
makes every element of `values` an exactly-once HBM read (the tree kernel
issues two loads of `values`, the second mostly L1/L2-hit).

  S_env:  inputs [num_envs, T] (library layout). Each step's BLOCK_E loads are
          strided by T -- uncoalesced, but successive steps hit the same 32B
          sectors, so L1/L2 absorb most of it.
  S_time: inputs [T, num_envs] (CleanRL-style buffer). Each step's BLOCK_E
          loads are contiguous -- fully coalesced.
"""
import torch
import triton
import triton.language as tl

from rl_triton.kernels.gae import gae_fused_kernel
from rl_triton.ops.gae import _WARPS


@triton.jit
def _serial_step(out_ptr, off, mask, r, v, terminated, v_next, carry, gamma, gl):
    """One backward step of the recurrence; returns (carry, v_next for step t-1)."""
    not_terminated = 1.0 - terminated
    delta = r + gamma * v_next * not_terminated - v
    carry = delta + gl * not_terminated * carry
    tl.store(out_ptr + off, carry, mask=mask)
    return carry, v


@triton.jit
def gae_serial_kernel(
    rewards_ptr, values_ptr, terminateds_ptr,
    out_ptr,
    bootstrap_ptr,
    num_envs,
    seq_len,
    stride_env,
    stride_t,
    gamma,
    lambda_,
    BLOCK_E: tl.constexpr,
    UNROLL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Serial backward walk over t for BLOCK_E envs. Layout is set entirely by
    (stride_env, stride_t): (T, 1) for env-major, (1, num_envs) for time-major.

    UNROLL (1/2/4/8) issues the loads of UNROLL consecutive steps before
    running those UNROLL dependent steps -- latency hiding via ILP only; the
    dependency chain is still exactly T steps long and the access pattern is
    unchanged. UNROLL=1, NUM_STAGES=1 is the literal one-step-per-iteration
    loop. NUM_STAGES is tl.range's software-pipelining depth (1 = off).
    Requires seq_len % UNROLL == 0 (checked by the wrappers).
    """
    pid    = tl.program_id(0)
    offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask   = offs_e < num_envs
    row    = offs_e.to(tl.int64) * stride_env

    v_next = tl.load(bootstrap_ptr + offs_e, mask=mask, other=0.0)
    carry  = tl.zeros((BLOCK_E,), dtype=tl.float32)
    gl     = gamma * lambda_
    for i in tl.range(0, seq_len, UNROLL, num_stages=NUM_STAGES):
        o0 = row + (seq_len - 1 - i).to(tl.int64) * stride_t
        r0 = tl.load(rewards_ptr + o0, mask=mask, other=0.0)
        v0 = tl.load(values_ptr + o0, mask=mask, other=0.0)
        d0 = tl.load(terminateds_ptr + o0, mask=mask, other=1.0)
        if UNROLL >= 2:
            o1 = o0 - stride_t
            r1 = tl.load(rewards_ptr + o1, mask=mask, other=0.0)
            v1 = tl.load(values_ptr + o1, mask=mask, other=0.0)
            d1 = tl.load(terminateds_ptr + o1, mask=mask, other=1.0)
        if UNROLL >= 4:
            o2 = o1 - stride_t
            o3 = o2 - stride_t
            r2 = tl.load(rewards_ptr + o2, mask=mask, other=0.0)
            v2 = tl.load(values_ptr + o2, mask=mask, other=0.0)
            d2 = tl.load(terminateds_ptr + o2, mask=mask, other=1.0)
            r3 = tl.load(rewards_ptr + o3, mask=mask, other=0.0)
            v3 = tl.load(values_ptr + o3, mask=mask, other=0.0)
            d3 = tl.load(terminateds_ptr + o3, mask=mask, other=1.0)
        if UNROLL >= 8:
            o4 = o3 - stride_t
            o5 = o4 - stride_t
            o6 = o5 - stride_t
            o7 = o6 - stride_t
            r4 = tl.load(rewards_ptr + o4, mask=mask, other=0.0)
            v4 = tl.load(values_ptr + o4, mask=mask, other=0.0)
            d4 = tl.load(terminateds_ptr + o4, mask=mask, other=1.0)
            r5 = tl.load(rewards_ptr + o5, mask=mask, other=0.0)
            v5 = tl.load(values_ptr + o5, mask=mask, other=0.0)
            d5 = tl.load(terminateds_ptr + o5, mask=mask, other=1.0)
            r6 = tl.load(rewards_ptr + o6, mask=mask, other=0.0)
            v6 = tl.load(values_ptr + o6, mask=mask, other=0.0)
            d6 = tl.load(terminateds_ptr + o6, mask=mask, other=1.0)
            r7 = tl.load(rewards_ptr + o7, mask=mask, other=0.0)
            v7 = tl.load(values_ptr + o7, mask=mask, other=0.0)
            d7 = tl.load(terminateds_ptr + o7, mask=mask, other=1.0)

        carry, v_next = _serial_step(out_ptr, o0, mask, r0, v0, d0, v_next, carry, gamma, gl)
        if UNROLL >= 2:
            carry, v_next = _serial_step(out_ptr, o1, mask, r1, v1, d1, v_next, carry, gamma, gl)
        if UNROLL >= 4:
            carry, v_next = _serial_step(out_ptr, o2, mask, r2, v2, d2, v_next, carry, gamma, gl)
            carry, v_next = _serial_step(out_ptr, o3, mask, r3, v3, d3, v_next, carry, gamma, gl)
        if UNROLL >= 8:
            carry, v_next = _serial_step(out_ptr, o4, mask, r4, v4, d4, v_next, carry, gamma, gl)
            carry, v_next = _serial_step(out_ptr, o5, mask, r5, v5, d5, v_next, carry, gamma, gl)
            carry, v_next = _serial_step(out_ptr, o6, mask, r6, v6, d6, v_next, carry, gamma, gl)
            carry, v_next = _serial_step(out_ptr, o7, mask, r7, v7, d7, v_next, carry, gamma, gl)


# ---------------------------------------------------------------------------
# Thin wrappers -- identical style for all three kernels: allocate the output,
# launch once, return. No asserts, no .contiguous(), no dtype checks.
# ---------------------------------------------------------------------------

def gae_tree(rewards, values, terminateds, last_value, gamma, lambda_):
    """Library tree kernel, plain path, with the library's own dispatch config
    (BLOCK_SIZE=next_pow2(T), _WARPS table, num_stages) -- i.e. exactly what
    compute_gae launches for T <= _FLAT_MAX_SEQ_LEN, minus its asserts."""
    num_envs, seq_len = rewards.shape
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    out = torch.empty_like(rewards)
    gae_fused_kernel[(num_envs,)](
        rewards, values, terminateds, None,
        out, last_value,
        seq_len, rewards.stride(0),
        gamma=gamma, lambda_=lambda_,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=_WARPS.get(BLOCK_SIZE, 16),
        num_stages=2 if BLOCK_SIZE >= 2048 else 1,
        HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=True,
    )
    return out


def gae_serial_env(rewards, values, terminateds, last_value, gamma, lambda_,
                   BLOCK_E=128, num_warps=4, UNROLL=1, NUM_STAGES=1):
    """S_env: inputs/outputs [num_envs, T]."""
    num_envs, seq_len = rewards.shape
    assert seq_len % UNROLL == 0
    out = torch.empty_like(rewards)
    gae_serial_kernel[(triton.cdiv(num_envs, BLOCK_E),)](
        rewards, values, terminateds, out, last_value,
        num_envs, seq_len, seq_len, 1,
        gamma, lambda_,
        BLOCK_E=BLOCK_E, UNROLL=UNROLL, NUM_STAGES=NUM_STAGES, num_warps=num_warps,
    )
    return out


def gae_serial_time(rewards_t, values_t, terminateds_t, last_value, gamma, lambda_,
                    BLOCK_E=128, num_warps=4, UNROLL=1, NUM_STAGES=1):
    """S_time: inputs/outputs [T, num_envs]."""
    seq_len, num_envs = rewards_t.shape
    assert seq_len % UNROLL == 0
    out = torch.empty_like(rewards_t)
    gae_serial_kernel[(triton.cdiv(num_envs, BLOCK_E),)](
        rewards_t, values_t, terminateds_t, out, last_value,
        num_envs, seq_len, 1, num_envs,
        gamma, lambda_,
        BLOCK_E=BLOCK_E, UNROLL=UNROLL, NUM_STAGES=NUM_STAGES, num_warps=num_warps,
    )
    return out
