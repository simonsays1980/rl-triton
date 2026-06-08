__version__ = "0.1.0"

from rl_triton.ops.gae import compute_gae_triton
from rl_triton.ops.retrace import compute_retrace_triton
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum
from rl_triton.ops.returns import compute_discounted_returns, compute_eligibility_traces, compute_lambda_returns
from rl_triton.ops.vtrace import compute_vtrace_triton

__all__ = [
    "compute_discounted_returns",
    "compute_eligibility_traces",
    "compute_episodic_prefix_sum",
    "compute_gae_triton",
    "compute_lambda_returns",
    "compute_retrace_triton",
    "compute_vtrace_triton",
]
