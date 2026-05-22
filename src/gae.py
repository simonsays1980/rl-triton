import torch
import triton
import triton.language as tl

@triton.jit
def combine_fn(u_a, v_a, u_b, v_b):
    """
    Associative scan combine function for linear recurrence.
    f_B(f_A(x)) = u_B + v_B * (u_A + v_A * x)
    """
    u_c = u_b + v_b * u_a
    v_c = v_a * v_b
    return u_c, v_c

@triton.jit
def gae_scan_kernel(
    delta_ptr, decay_ptr, adv_ptr,
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr
):
    # Map each Triton block to one environment in the batch
    env_idx = tl.program_id(0)
    base_idx = env_idx * stride_env
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Reverse offsets to process the sequence backward (T-1 down to 0)
    # This avoids the overhead of calling torch.flip() before the kernel
    rev_offsets = seq_len - 1 - offsets
    mask = offsets < seq_len
    
    # Load TD errors (delta) and decay factors (gamma * lambda * (1 - done))
    # Important: If mask is false, other_decay must be 1.0 (multiplicative identity)
    delta = tl.load(delta_ptr + base_idx + rev_offsets, mask=mask, other=0.0)
    decay = tl.load(decay_ptr + base_idx + rev_offsets, mask=mask, other=1.0)
    
    # Perform the parallel associative scan
    adv, _ = tl.associative_scan((delta, decay), axis=0, combine_fn=combine_fn)
    
    # Store the computed advantages, reversing the offsets again to restore time order
    tl.store(adv_ptr + base_idx + rev_offsets, adv, mask=mask)