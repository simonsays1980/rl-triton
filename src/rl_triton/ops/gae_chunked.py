import torch
import triton

from rl_triton.kernels.gae_chunked import chunked_gae_kernel

# Triton's associative_scan requires BLOCK_SIZE to be a power of 2 and imposes
# a hardware limit of 2^17 elements per block.  The flat gae_scan_kernel sets
# BLOCK_SIZE = next_power_of_2(seq_len), so it cannot handle seq_len > 131072.
# The chunked kernel lifts this ceiling by iterating over fixed-size chunks.
_CHUNK_SIZE = triton.next_power_of_2(1024)  # chunk size; next_power_of_2 guards against accidental non-power-of-2 edits


def compute_gae_chunked(deltas: torch.Tensor, decays: torch.Tensor) -> torch.Tensor:
    """
    Compute GAE via a chunked backward scan for sequences of any length.

    Splits seq_len into fixed chunks of size _CHUNK_SIZE and sweeps right-to-left,
    carrying A[chunk_end+1] across boundaries.  Prefer compute_gae_triton for
    seq_len <= 131072; use this when sequences exceed that limit.

    Args:
        deltas: TD residuals, shape [num_envs, seq_len], float32, CUDA, contiguous.
        decays: Per-step decay factors (gamma * lambda * (1 - done)), same shape.

    Returns:
        advantages: shape [num_envs, seq_len], float32.
    """
    assert deltas.shape == decays.shape, "deltas and decays must have the same shape"
    assert deltas.is_cuda and decays.is_cuda, "Inputs must be on CUDA"
    assert deltas.dtype == torch.float32, f"Expected float32, got {deltas.dtype}"
    assert decays.dtype == torch.float32, f"Expected float32, got {decays.dtype}"

    deltas = deltas.contiguous()
    decays = decays.contiguous()

    num_envs, seq_len = deltas.shape
    advantages = torch.empty_like(deltas)

    chunked_gae_kernel[(num_envs,)](
        deltas, decays, advantages,
        seq_len,
        deltas.stride(0),
        BLOCK_SIZE=_CHUNK_SIZE,
    )
    return advantages