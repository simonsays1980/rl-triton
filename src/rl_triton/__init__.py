__version__ = "0.1.3"

from rl_triton.ops.gae import compute_gae
from rl_triton.ops.retrace import compute_retrace
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum
from rl_triton.ops.returns import compute_discounted_returns, compute_eligibility_traces, compute_lambda_returns
from rl_triton.ops.vtrace import compute_vtrace

__all__ = [
    "compute_discounted_returns",
    "compute_eligibility_traces",
    "compute_episodic_prefix_sum",
    "compute_gae",
    "compute_lambda_returns",
    "compute_retrace",
    "compute_vtrace",
]
