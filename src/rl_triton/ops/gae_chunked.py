import torch
import triton

from rl_triton.kernels.gae_chunked import chunked_gae_kernel

# Triton's associative_scan requires BLOCK_SIZE to be a power of 2 and imposes
# a hardware limit of 2^17 elements per block.  The flat gae_scan_kernel sets
# BLOCK_SIZE = next_power_of_2(seq_len), so it cannot handle seq_len > 131072.
# The chunked kernel lifts this ceiling by iterating over fixed-size chunks.
_CHUNK_SIZE = triton.next_power_of_2(1024)  # chunk size; next_power_of_2 guards against accidental non-power-of-2 edits


def compute_gae_chunked(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute GAE via a chunked backward scan for sequences of any length.

    Recurrence: A[t] = delta[t] + decay[t] * A[t+1], A[T] = bootstrap_values.

    Splits seq_len into fixed chunks of size _CHUNK_SIZE and sweeps right-to-left,
    carrying A[chunk_end+1] across boundaries.  Prefer compute_gae_triton for
    seq_len <= 131072; use this when sequences exceed that limit.

    Args:
        deltas:           TD residuals, shape [num_envs, seq_len], float32, CUDA.
        decays:           Per-step decay factors (gamma * lambda * (1 - done)), same shape.
        bootstrap_values: Per-environment boundary value A[T], shape [num_envs], float32.
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros (all episodes terminated).

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

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=deltas.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    advantages = torch.empty_like(deltas)

    chunked_gae_kernel[(num_envs,)](
        deltas, decays, advantages,
        bootstrap_values,
        seq_len,
        deltas.stride(0),
        BLOCK_SIZE=_CHUNK_SIZE,
    )
    return advantages
