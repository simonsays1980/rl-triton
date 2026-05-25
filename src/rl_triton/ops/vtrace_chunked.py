import torch
import triton

from rl_triton.kernels.vtrace_chunked import chunked_vtrace_kernel

_CHUNK_SIZE = triton.next_power_of_2(1024)  # chunk size; next_power_of_2 guards against accidental non-power-of-2 edits


def compute_vtrace_chunked(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run the V-Trace backward scan for sequences of any length.

    This is an internal helper called by compute_vtrace_triton when
    seq_len > 131072.  It operates on the pre-computed scan inputs
    (deltas, decays) rather than raw RL quantities.

    Args:
        deltas:           Clipped TD residuals ρ * (r + γV' - V), [num_envs, seq_len], float32, CUDA.
        decays:           Per-step decay factors γ * c * (1-done), same shape.
        bootstrap_values: Boundary value Δ[T] per environment, shape [num_envs], float32.
                          Defaults to zeros (terminated episodes).

    Returns:
        value_deltas: Δ values from the scan, shape [num_envs, seq_len], float32.
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

    value_deltas = torch.empty_like(deltas)

    chunked_vtrace_kernel[(num_envs,)](
        deltas, decays, value_deltas,
        bootstrap_values,
        seq_len,
        deltas.stride(0),
        BLOCK_SIZE=_CHUNK_SIZE,
    )
    return value_deltas
