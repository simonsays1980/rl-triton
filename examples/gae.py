import torch
import triton

from src.gae import gae_scan_kernel

def compute_gae_triton(deltas: torch.Tensor, decays: torch.Tensor):
    # deltas and decays are shape [num_envs, seq_len]
    num_envs, seq_len = deltas.shape
    advantages = torch.empty_like(deltas)
    
    # Find the next power of 2 for the block size (required by Triton)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    
    # Launch 1D grid over the number of environments
    gae_scan_kernel[(num_envs,)](
        deltas, decays, advantages,
        seq_len,
        deltas.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return advantages

