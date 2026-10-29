"""Task B worker: synchronized re-timing of the sequential PyTorch loop baselines.

Same method as task4_loop_sync.py: the loop function and inputs are exactly the ones
tests/bench_release.py passes to _bench_cpu for the "loop gpu" column of benchmarks.md,
but timed with _bench_gpu (CUDA events, torch.cuda.synchronize() before the start event
and after the end event of every call; 20 warmup, 5 trials x 50, min of trial medians).
The loop output is first gated (atol=rtol=1e-4) against the shipped library op on the
same inputs, as bench_release does. The same-session _bench_cpu value is also recorded.

    python loop_sync_worker.py ALGO NUM_ENVS SEQ_LEN   # prints one JSON row
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import torch  # noqa: E402

import bench_release as br  # noqa: E402
from bench_utils import _bench_cpu, _bench_gpu, _warmup_gpu, assert_correctness  # noqa: E402
from rl_triton.ops.gae import compute_gae  # noqa: E402
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum  # noqa: E402
from rl_triton.ops.retrace import compute_retrace  # noqa: E402
from rl_triton.ops.returns import (  # noqa: E402
    compute_discounted_returns, compute_eligibility_traces, compute_lambda_returns)
from rl_triton.ops.vtrace_fused import compute_vtrace_fused  # noqa: E402
from test_prefix_sum import reference_episodic_prefix_sum  # noqa: E402


def job(algo, E, T):
    """(loop_fn, loop_args, loop_kw, kernel_output) -- args as in bench_release's _bench_cpu call."""
    if algo == "gae":
        a = br._make_gae(E, T); kw = dict(gamma=0.99, lambda_=0.95)
        return br._ref_gae, a, kw, compute_gae(*a, **kw)
    if algo == "vtrace":
        a = br._make_vtrace(E, T); kw = dict(gamma=0.99)
        return br._ref_vtrace, a, kw, compute_vtrace_fused(*a, **kw)
    if algo == "retrace":
        a = br._make_retrace(E, T); kw = dict(gamma=0.99)
        return br._ref_retrace, a, kw, compute_retrace(*br._retrace_kernel_args(a), **kw)[0]
    if algo == "lambda_returns":
        r, nv, d = br._make_returns(E, T); kw = dict(gamma=0.99, lambda_=0.95)
        return br._ref_lambda, (r, nv, d), kw, compute_lambda_returns(r, nv, d, **kw)
    if algo == "discounted_returns":
        r, _, d = br._make_returns(E, T); kw = dict(gamma=0.99)
        return br._ref_disc, (r, d), kw, compute_discounted_returns(r, d, **kw)
    if algo == "eligibility_traces":
        r, _, d = br._make_returns(E, T); kw = dict(gamma=0.99, lambda_=0.9)
        return br._ref_traces, (r, d), kw, compute_eligibility_traces(r, d, **kw)
    if algo == "prefix_sum":
        a = br._make_prefix_sum(E, T)
        return reference_episodic_prefix_sum, a, {}, compute_episodic_prefix_sum(*a)
    raise ValueError(algo)


def main():
    algo, E, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fn, args, kw, kern = job(algo, E, T)
    assert_correctness(kern, fn(*args, **kw), f"{algo}[{E}x{T}] loop vs kernel")
    loop_out = fn(*args, **kw)
    pairs = zip(kern, loop_out) if isinstance(kern, tuple) else [(kern, loop_out)]
    err = max((a - b).abs().max().item() for a, b in pairs)
    cpu_ms = _bench_cpu(fn, *args, **kw)
    _warmup_gpu(fn, *args, n_warmup=20, **kw)
    sync_ms = _bench_gpu(fn, *args, n_iter=50, n_trials=5, **kw)
    print(json.dumps({"algo": algo, "num_envs": E, "seq_len": T, "validated": True,
                      "max_abs_err_loop_vs_kernel": err,
                      "loop_sync_ms": sync_ms, "loop_bench_cpu_same_session_ms": cpu_ms,
                      "gpu": torch.cuda.get_device_name(0)}))


if __name__ == "__main__":
    main()
