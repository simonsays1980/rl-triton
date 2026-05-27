import torch

from rl_triton.ops._scan import _run_scan_forward


def compute_episodic_prefix_sum(
    inputs: torch.Tensor,
    dones: torch.Tensor,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cumulative sum that resets to zero at episode boundaries.

    Recurrence:  C[t] = x[t] + (1 - d[t]) * C[t-1],  C[-1] = seed

    Maps to the forward linear recurrence e[t] = u[t] + v[t] * e[t-1] with:
      u[t] = inputs[t]
      v[t] = 1 - dones[t]   (1.0 while episode ongoing, 0.0 at terminal step)

    When d[t] = 1, v[t] = 0 and C[t] = x[t]: the accumulation resets.
    When d[t] = 0, v[t] = 1 and C[t] = x[t] + C[t-1]: standard prefix sum.

    Limited to seq_len <= 131072 (flat kernel only; see NOTES.md).

    Args:
        inputs:      Values to accumulate x[t], [num_envs, seq_len], float32, CUDA.
        dones:       Episode termination flags (1.0 = done), same shape, float32.
        seed_values: Initial carry C[-1] per environment, shape [num_envs].
                     Defaults to zeros (accumulation starts from scratch).

    Returns:
        prefix_sums: C[t], shape [num_envs, seq_len], same dtype as inputs.
    """
    assert inputs.is_cuda and dones.is_cuda, "inputs and dones must be on CUDA"
    assert inputs.dtype == torch.float32, f"inputs: expected float32, got {inputs.dtype}"
    assert dones.dtype == torch.float32,  f"dones: expected float32, got {dones.dtype}"
    assert inputs.shape == dones.shape,   "inputs and dones must have the same shape"

    v = (1.0 - dones).to(inputs.dtype)
    return _run_scan_forward(inputs, v, seed_values)
