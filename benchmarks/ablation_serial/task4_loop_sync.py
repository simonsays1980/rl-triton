"""Task 4: re-time the GAE sequential PyTorch loop at (4096,128) with CUDA events
and torch.cuda.synchronize() around every call, and compare with benchmarks.md
v0.1.3 H100 (loop gpu 10.312 ms; triton full-call 0.033 ms; "vs loop" 316.9x).

That table's loop column came from tests/bench_release.py::bench_gae:
`_bench_cpu(_ref_gae, ...)`. It measures wall clock over back-to-back calls, with no
sync between calls and no final device sync. This script times the SAME function
on the SAME inputs (bench_release._make_gae) three ways in one process:
  1. _bench_cpu, exactly as the release sweep does (same-session reference);
  2. _bench_gpu: 20 warmup, 5 trials x 50, min of trial medians, with
     torch.cuda.synchronize() before the start event and after the end event
     of every call;
  3. compute_gae full-call with the same _bench_gpu protocol, for the ratio.

    python benchmarks/ablation_serial/task4_loop_sync.py
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
from test_gae import reference_gae  # noqa: E402

RECORDED = {"loop_ms": 10.312, "triton_ms": 0.033, "speedup": 316.9}


def main():
    args = br._make_gae(4096, 128)  # rewards, values, dones; ~5% terminations, seed 0
    kw = {"gamma": 0.99, "lambda_": 0.95}
    ref = reference_gae(*args, 0.99, 0.95)  # independent reference (no bootstrap, as in the table)
    assert_correctness(br._ref_gae(*args, **kw), ref, "loop (_ref_gae)")
    assert_correctness(compute_gae(*args, **kw), ref, "compute_gae")

    cpu_ms = _bench_cpu(br._ref_gae, *args, **kw)
    _warmup_gpu(br._ref_gae, *args, n_warmup=20, **kw)
    sync_ms = _bench_gpu(br._ref_gae, *args, n_iter=50, n_trials=5, **kw)
    _warmup_gpu(compute_gae, *args, n_warmup=20, **kw)
    tri_ms = _bench_gpu(compute_gae, *args, n_iter=50, n_trials=5, **kw)

    out = {
        "shape": [4096, 128], "gpu": torch.cuda.get_device_name(0),
        "recorded_benchmarks_md_v0.1.3": RECORDED,
        "loop_bench_cpu_same_session_ms": cpu_ms,
        "loop_synchronized_events_ms": sync_ms,
        "compute_gae_fullcall_ms": tri_ms,
        "speedup_synchronized_loop_over_compute_gae": sync_ms / tri_ms,
        "speedup_bench_cpu_loop_over_compute_gae": cpu_ms / tri_ms,
        "synchronized_vs_recorded_loop": sync_ms / RECORDED["loop_ms"],
        "synchronized_vs_same_session_bench_cpu": sync_ms / cpu_ms,
    }
    print(json.dumps(out, indent=2))
    (Path(__file__).resolve().parent / "task4_loop_sync.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
