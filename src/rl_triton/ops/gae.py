import torch

from rl_triton.ops._scan import _run_scan


def compute_gae_triton(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence: A[t] = delta[t] + decay[t] * A[t+1], A[T] = bootstrap_values.

    Dispatches to the flat single-block kernel for seq_len <= 131072 and falls
    back to the chunked kernel for longer sequences.

    Args:
        deltas:           TD residuals δ[t] = r[t] + γV'[t](1-d[t]) - V[t],
                          shape [num_envs, seq_len], float32, CUDA.
        decays:           Per-step decay factors γλ(1-d[t]), same shape.
        bootstrap_values: Per-environment boundary value A[T], shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros.

    Returns:
        advantages: shape [num_envs, seq_len], float32.
    """
    return _run_scan(deltas, decays, bootstrap_values)
