"""Task A worker: shipped gae_fused_kernel with an explicit num_warps override.

Launch is identical to the library dispatch (serial_gae.gae_tree / compute_gae's
flat path: BLOCK_SIZE=next_pow2(T), num_stages=2 if BLOCK_SIZE>=2048, plain path
HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=True) except num_warps, which is swept over
{4, 8, 16, 32}. _WARPS is not edited. Each config is compiled first in a child
process (own process group) with a 120 s timeout, then correctness-gated
(bench_worker.check, atol=rtol=1e-4 vs reference_gae) and timed with the
standard protocol (bench_worker.fullcall / bench_worker.device).

    python benchmarks/ablation_serial/warp_worker.py --num-envs 16 --seq-len 32768 --out warp_raw.jsonl
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
from test_gae import reference_gae  # noqa: E402
from rl_triton.kernels.gae import gae_fused_kernel  # noqa: E402
from rl_triton.ops.gae import _WARPS  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402

WARPS = (4, 8, 16, 32)
COMPILE_TIMEOUT_S = 120


def gae_tree_warps(rewards, values, terminateds, last_value, gamma, lambda_, num_warps):
    num_envs, seq_len = rewards.shape
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    out = torch.empty_like(rewards)
    gae_fused_kernel[(num_envs,)](
        rewards, values, terminateds, None,
        out, last_value,
        seq_len, rewards.stride(0),
        gamma=gamma, lambda_=lambda_,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        num_stages=2 if BLOCK_SIZE >= 2048 else 1,
        HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=True,
    )
    return out


def precompile(num_envs, seq_len, nw):
    cmd = [sys.executable, __file__, "--num-envs", str(num_envs), "--seq-len", str(seq_len),
           "--out", os.devnull, "--precompile", str(nw)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
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
    p.add_argument("--precompile", type=int, help="(internal) compile one num_warps config and exit")
    a = p.parse_args()

    r, v, d, b = bw.make_inputs(a.num_envs, a.seq_len)
    if a.precompile:
        gae_tree_warps(r, v, d, b, bw.GAMMA, bw.LAMBDA, a.precompile)
        torch.cuda.synchronize()
        return
    ref = reference_gae(r, v, d, bw.GAMMA, bw.LAMBDA, bootstrap_values=b)
    torch.cuda.synchronize()
    shipped_nw = _WARPS.get(triton.next_power_of_2(a.seq_len), 16)

    for nw in WARPS:
        row = {"num_envs": a.num_envs, "seq_len": a.seq_len, "num_warps": nw,
               "shipped_num_warps": shipped_nw, "is_shipped_config": nw == shipped_nw}
        skip = precompile(a.num_envs, a.seq_len, nw)
        if skip:
            row.update(validated=False, skip_reason=skip)
        else:
            call = lambda nw=nw: gae_tree_warps(r, v, d, b, bw.GAMMA, bw.LAMBDA, nw)
            ok, err = bw.check(call(), ref, f"tree[num_warps={nw}]")
            row.update(validated=ok, max_abs_err=err)
            if ok:
                row.update(fullcall_ms=bw.fullcall(call), device_ms=bw.device(call))
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
