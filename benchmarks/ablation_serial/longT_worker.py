"""Task 2 worker: long-horizon, few-env shapes. One (kernel, shape) per process.

Reuses bench_worker.py unchanged (make_inputs, check, fullcall, device,
run_serial); only bench_worker.BLOCK_ES is narrowed per shape to BLOCK_E <=
num_envs. Because the original set {32..256} is empty at num_envs=16, BLOCK_E
in {8, 16} is added to the candidate list (then filtered by <= num_envs).

Kernels: tree (library dispatch, untuned; _WARPS has no entry above
BLOCK_SIZE=16384, so T >= 32768 falls back to 16 warps), serial_env,
serial_time (+ naive / transpose rows), scan, loop (tests/test_gae.py::
reference_gae -- the sequential PyTorch loop, timed with _bench_gpu, which
calls torch.cuda.synchronize() immediately before the start event and after
the end event of every timed call).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_worker as bw  # noqa: E402  (sets up sys.path for src/ and tests/)
from bench_utils import _device_profile  # noqa: E402
from serial_gae import gae_tree  # noqa: E402
from test_gae import reference_gae, vectorized_gae_with_truncations  # noqa: E402

import torch  # noqa: E402

LONG_T_BLOCK_ES = (8, 16, 32, 64, 128, 256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, choices=["tree", "serial_env", "serial_time", "scan", "loop"])
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    shape = {"num_envs": a.num_envs, "seq_len": a.seq_len}

    def emit(row):
        with open(a.out, "a") as f:
            f.write(json.dumps({**shape, **row}) + "\n")

    r, v, d, b = bw.make_inputs(a.num_envs, a.seq_len)
    ref = reference_gae(r, v, d, bw.GAMMA, bw.LAMBDA, bootstrap_values=b)
    torch.cuda.synchronize()

    if a.kernel in ("serial_env", "serial_time"):
        bw.BLOCK_ES = tuple(be for be in LONG_T_BLOCK_ES if be <= a.num_envs)
        bw.run_serial(a.kernel, r, v, d, b, ref, emit)
        return

    note = ""
    if a.kernel == "tree":
        call, cfg = (lambda: gae_tree(r, v, d, b, bw.GAMMA, bw.LAMBDA)), "library dispatch"
        if a.seq_len > 16384:
            note = "BLOCK_SIZE > 16384: _WARPS has no entry, falls back to 16 warps (untuned)"
    elif a.kernel == "scan":
        torch._dynamo.reset()
        compiled = torch.compile(vectorized_gae_with_truncations)
        trunc0 = torch.zeros_like(r)
        bsv = torch.zeros_like(r)
        bsv[:, -1] = b
        call = lambda: compiled(r, v, d, trunc0, bsv, bw.GAMMA, bw.LAMBDA)
        cfg = "torch.compile(vectorized_gae_with_truncations)"
    else:
        call = lambda: reference_gae(r, v, d, bw.GAMMA, bw.LAMBDA, bootstrap_values=b)
        cfg = "tests/test_gae.py::reference_gae (sequential PyTorch loop)"
        note = ("synchronized: torch.cuda.synchronize() before start event and after end event "
                "of every timed call (_bench_gpu); device_ms from 1 profiler trial of 2 calls")

    ok, err = bw.check(call(), ref, a.kernel)
    row = {"kind": "result", "kernel": a.kernel, "layout": "env-major [E,T]", "config": cfg,
           "validated": ok, "max_abs_err": err, "note": note}
    if ok:
        row["fullcall_ms"] = bw.fullcall(call)
        if a.kernel == "loop":
            # Profiling 50 calls x T steps x ~6 kernels is millions of events; use 2 calls.
            row["device_ms"] = _device_profile(call, n_iter=2, n_warmup=1)[0]
        else:
            row["device_ms"] = bw.device(call)
    emit(row)


if __name__ == "__main__":
    main()
