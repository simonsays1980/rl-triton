"""Task 3 worker: packed-env tree kernel, one shape per process.

Tuning: BLOCK_E in {1,2,4,8,16,32} x num_warps in {1,2,4,8}. Every config is
correctness-gated (bench_worker.check: atol=rtol=1e-4 vs reference_gae); configs
that fail to compile are logged and skipped. Selection mirrors run_serial: screen
by device time (1 profiler trial), full protocol on the top 4, best full-call
wins, winner re-measured independently. The config with the best screened
device time is also re-measured (`device_best_*`), since Task 3's figure is
device time.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_worker as bw  # noqa: E402
from bench_utils import _device_profile  # noqa: E402
from packed_gae import gae_packed  # noqa: E402
from test_gae import reference_gae  # noqa: E402

import torch  # noqa: E402

BLOCK_ES = (1, 2, 4, 8, 16, 32)
WARPS = (1, 2, 4, 8)
TOP_K = 4
MAX_WARP_ELEMS = 32768
COMPILE_TIMEOUT_S = 120


def precompile(num_envs, seq_len, be, nw):
    """Compile one config in a child process with a hard timeout.
    Returns None on success, else a skip reason."""
    cmd = [sys.executable, __file__, "--num-envs", str(num_envs), "--seq-len", str(seq_len),
           "--out", os.devnull, "--precompile", str(be), str(nw)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        _, err = proc.communicate(timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return f"compile timeout (>{COMPILE_TIMEOUT_S}s)"
    if proc.returncode != 0:
        return f"compile failure (rc={proc.returncode}): {err.decode()[-200:].strip()}"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--precompile", nargs=2, type=int, metavar=("BLOCK_E", "NUM_WARPS"),
                   help="(internal) compile one config and exit")
    a = p.parse_args()
    shape = {"num_envs": a.num_envs, "seq_len": a.seq_len}

    def emit(row):
        with open(a.out, "a") as f:
            f.write(json.dumps({**shape, **row}) + "\n")

    r, v, d, b = bw.make_inputs(a.num_envs, a.seq_len)
    if a.precompile:
        gae_packed(r, v, d, b, bw.GAMMA, bw.LAMBDA, BLOCK_E=a.precompile[0], num_warps=a.precompile[1])
        torch.cuda.synchronize()
        return
    block_t = 1 << (a.seq_len - 1).bit_length()
    ref = reference_gae(r, v, d, bw.GAMMA, bw.LAMBDA, bootstrap_values=b)
    torch.cuda.synchronize()

    def mk(c):
        return lambda: gae_packed(r, v, d, b, bw.GAMMA, bw.LAMBDA, BLOCK_E=c["block_e"], num_warps=c["num_warps"])

    sweep = []
    for be in BLOCK_ES:
        for nw in WARPS:
            c = {"config": f"BLOCK_E={be},num_warps={nw}", "block_e": be, "num_warps": nw}
            per_warp = be * block_t // nw
            skip = (f"size cap: {per_warp} tile elements per warp > {MAX_WARP_ELEMS}"
                    if per_warp > MAX_WARP_ELEMS else precompile(a.num_envs, a.seq_len, be, nw))
            if skip:
                c.update(validated=False, max_abs_err=float("nan"), skip_reason=skip)
                print(f"  skipped {c['config']}: {skip}", flush=True)
                sweep.append(c)
                continue
            try:
                ok, err = bw.check(mk(c)(), ref, f"packed[{c['config']}]")
            except Exception as e:
                ok, err, c["error"] = False, float("nan"), type(e).__name__
                print(f"  compile/launch failure {c['config']}: {type(e).__name__}", flush=True)
            c.update(validated=ok, max_abs_err=err)
            if ok:
                c["screen_device_ms"] = _device_profile(mk(c), n_iter=20, n_warmup=5)[0]
            sweep.append(c)

    valid = [c for c in sweep if c["validated"]]
    if not valid:
        emit({"kind": "result", "kernel": "packed", "validated": False, "config": "",
              "note": "no config compiled and passed the correctness gate"})
    else:
        top = sorted(valid, key=lambda c: c["screen_device_ms"])[:TOP_K]
        for c in top:
            c["fullcall_ms"] = bw.fullcall(mk(c))
        best = min(top, key=lambda c: c["fullcall_ms"])
        dbest = top[0]
        emit({"kind": "result", "kernel": "packed", "layout": "env-major [E,T]", "validated": True,
              "config": best["config"], "max_abs_err": best["max_abs_err"],
              "fullcall_ms": bw.fullcall(mk(best)), "device_ms": bw.device(mk(best)),
              "device_best_config": dbest["config"], "device_best_ms": bw.device(mk(dbest)),
              "n_configs_valid": len(valid), "n_configs": len(sweep)})
    for c in sweep:
        emit({"kind": "sweep", "kernel": "packed", **c})


if __name__ == "__main__":
    main()
