# Backward-compatibility re-exports.  Scan primitives live in kernels/scan.py.
from rl_triton.kernels.scan import _combine, _combine_fwd, backward_scan_kernel, forward_scan_kernel

gae_scan_kernel = backward_scan_kernel

__all__ = ["_combine", "_combine_fwd", "gae_scan_kernel", "forward_scan_kernel"]
