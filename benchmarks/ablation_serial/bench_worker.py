"""One (kernel, shape) measurement, run in its own subprocess by run_ablation.py.

Timing reuses tests/bench_utils.py unchanged (the paper's harness):
  fullcall_ms -- _warmup_gpu(20) + _bench_gpu(n_iter=50, n_trials=5): CUDA
                 events around each call, min of the 5 trial medians.
  device_ms   -- _device_profile (torch.profiler CUDA kernel time, ms/call,
                 20 warmup + 50 profiled calls), min over 5 such trials.

Every config is correctness-gated (assert_correctness, atol=rtol=1e-4) against
the independent sequential reference (tests/test_gae.py::reference_gae) BEFORE
it is timed; a config that fails is recorded with validated=False and no times.

Appends one JSON object per result row to --out.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_LOGS", "-dynamic")

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(Path(__file__).resolve().parent)]

from bench_utils import _bench_gpu, _device_profile, _warmup_gpu, assert_correctness  # noqa: E402
from test_gae import reference_gae, vectorized_gae_with_truncations  # noqa: E402
from serial_gae import gae_serial_env, gae_serial_time, gae_tree  # noqa: E402
from rl_triton.ops.gae import compute_gae  # noqa: E402

GAMMA, LAMBDA = 0.99, 0.95
N_WARMUP, N_ITER, N_TRIALS = 20, 50, 5
BLOCK_ES = (32, 64, 128, 256)
WARPS = (1, 2, 4, 8)
UNROLLS = (1, 2, 4, 8)
STAGES = (1, 3)
TOP_K = 4


def make_inputs(num_envs, seq_len, seed=0):
    """tests/test_gae.py::_make_inputs (randn r/v, ~5%/step terminations) plus a
    nonzero bootstrap V(s_T) ~ N(0,1) per env."""
    torch.manual_seed(seed)
    r = torch.randn(num_envs, seq_len, device="cuda")
    v = torch.randn(num_envs, seq_len, device="cuda")
    d = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
    b = torch.randn(num_envs, device="cuda")
    return r, v, d, b


def fullcall(fn):
    _warmup_gpu(fn, n_warmup=N_WARMUP)
    return _bench_gpu(fn, n_iter=N_ITER, n_trials=N_TRIALS)


def device(fn, n_trials=N_TRIALS):
    return min(_device_profile(fn, n_iter=N_ITER, n_warmup=N_WARMUP)[0] for _ in range(n_trials))


def check(out, ref, label):
    try:
        assert_correctness(out, ref, label)
        return True, (out - ref).abs().max().item()
    except AssertionError as e:
        print(f"  CORRECTNESS FAIL {label}: {str(e)[:300]}", flush=True)
        return False, float("nan")


def run_serial(kernel, r, v, d, b, ref, emit):
    """Tuning: validate every config in BLOCK_E x num_warps x UNROLL x NUM_STAGES
    (128 configs), screen each by device time (1 profiler trial), then run the
    full protocol on the TOP_K screened configs and report the one with the best
    full-call time. The spec-literal loop (UNROLL=1, NUM_STAGES=1; best over
    BLOCK_E x num_warps) is reported separately as `<kernel>_naive` with the same
    selection rule."""
    if kernel == "serial_env":
        args, fn_k, unT = (r, v, d), gae_serial_env, (lambda o: o)
        layout = "env-major [E,T]"
    else:
        args = (r.T.contiguous(), v.T.contiguous(), d.T.contiguous())
        fn_k, unT = gae_serial_time, (lambda o: o.T)
        layout = "time-major [T,E]"

    def mk(c):
        return lambda: fn_k(*args, b, GAMMA, LAMBDA, BLOCK_E=c["block_e"], num_warps=c["num_warps"],
                            UNROLL=c["unroll"], NUM_STAGES=c["num_stages"])

    sweep = []
    for be in BLOCK_ES:
        for nw in WARPS:
            for un in UNROLLS:
                for ns in STAGES:
                    c = {"config": f"BLOCK_E={be},num_warps={nw},UNROLL={un},NUM_STAGES={ns}",
                         "block_e": be, "num_warps": nw, "unroll": un, "num_stages": ns}
                    try:
                        ok, err = check(unT(mk(c)()), ref, f"{kernel}[{c['config']}]")
                    except Exception as e:  # Triton compile failure -> excluded config
                        ok, err = False, float("nan")
                        c["error"] = f"{type(e).__name__}"
                    c.update(validated=ok, max_abs_err=err)
                    if ok:
                        c["screen_device_ms"] = _device_profile(mk(c), n_iter=20, n_warmup=5)[0]
                    sweep.append(c)

    def finalize(cands, name):
        valid = [c for c in cands if c["validated"]]
        if not valid:
            emit({"kind": "result", "kernel": name, "layout": layout, "validated": False,
                  "config": "", "note": "no config passed correctness gate"})
            return None
        top = sorted(valid, key=lambda c: c["screen_device_ms"])[:TOP_K]
        for c in top:
            c.setdefault("fullcall_ms", fullcall(mk(c)))
        best = min(top, key=lambda c: c["fullcall_ms"])
        # Re-measure the winner with an independent full-call run, so the
        # reported number is not a min over TOP_K noisy draws (winner's curse).
        emit({"kind": "result", "kernel": name, "layout": layout, "validated": True,
              "config": best["config"], "max_abs_err": best["max_abs_err"],
              "fullcall_ms": fullcall(mk(best)), "fullcall_ms_selection": best["fullcall_ms"],
              "device_ms": device(mk(best)),
              "n_configs_valid": len(valid), "n_configs": len(cands)})
        return best

    best = finalize(sweep, kernel)
    finalize([c for c in sweep if c["unroll"] == 1 and c["num_stages"] == 1], kernel + "_naive")
    for c in sweep:
        emit({"kind": "sweep", "kernel": kernel, "layout": layout, **c})

    if kernel == "serial_time" and best is not None:
        # Cost of getting a [E,T] buffer into S_time's layout: three input
        # transposes (rewards, values, terminateds) via .contiguous(). Output is
        # left time-major (a CleanRL-style consumer reads it as [T,E]).
        def transpose_only():
            return r.T.contiguous(), v.T.contiguous(), d.T.contiguous()

        def incl_transpose():
            rt, vt, dt = transpose_only()
            return gae_serial_time(rt, vt, dt, b, GAMMA, LAMBDA, BLOCK_E=best["block_e"],
                                   num_warps=best["num_warps"], UNROLL=best["unroll"],
                                   NUM_STAGES=best["num_stages"])

        ok, err = check(incl_transpose().T, ref, "serial_time+transpose")
        emit({"kind": "result", "kernel": "transpose_inputs", "layout": "[E,T]->[T,E] x3",
              "validated": True, "config": "torch .T.contiguous() x3",
              "fullcall_ms": fullcall(transpose_only), "device_ms": device(transpose_only)})
        row = {"kind": "result", "kernel": "serial_time+transpose",
               "layout": "env-major in, time-major out",
               "validated": ok, "config": best["config"], "max_abs_err": err}
        if ok:
            row.update(fullcall_ms=fullcall(incl_transpose), device_ms=device(incl_transpose))
        emit(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True,
                   choices=["tree", "tree_lib", "serial_env", "serial_time", "scan"])
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    shape = {"num_envs": a.num_envs, "seq_len": a.seq_len}

    def emit(row):
        with open(a.out, "a") as f:
            f.write(json.dumps({**shape, **row}) + "\n")

    r, v, d, b = make_inputs(a.num_envs, a.seq_len)
    ref = reference_gae(r, v, d, GAMMA, LAMBDA, bootstrap_values=b)
    torch.cuda.synchronize()

    if a.kernel in ("serial_env", "serial_time"):
        run_serial(a.kernel, r, v, d, b, ref, emit)
        return

    if a.kernel == "tree":
        call, layout, cfg = (lambda: gae_tree(r, v, d, b, GAMMA, LAMBDA)), "env-major [E,T]", "library dispatch"
    elif a.kernel == "tree_lib":
        # Public compute_gae (asserts + .contiguous()) -- connects to the paper's numbers.
        call = lambda: compute_gae(r, v, d, gamma=GAMMA, lambda_=LAMBDA, last_value=b)
        layout, cfg = "env-major [E,T]", "compute_gae() public API"
    else:
        # Same compiled baseline as tests/bench_release.py::bench_gae:
        # vectorized_gae_with_truncations compiled directly, zero truncateds and
        # the bootstrap column built in EAGER code outside the timed region.
        torch._dynamo.reset()
        compiled = torch.compile(vectorized_gae_with_truncations)
        trunc0 = torch.zeros_like(r)
        bsv = torch.zeros_like(r)
        bsv[:, -1] = b
        call = lambda: compiled(r, v, d, trunc0, bsv, GAMMA, LAMBDA)
        layout, cfg = "env-major [E,T]", "torch.compile(vectorized_gae_with_truncations)"

    ok, err = check(call(), ref, a.kernel)
    row = {"kind": "result", "kernel": a.kernel, "layout": layout, "config": cfg,
           "validated": ok, "max_abs_err": err}
    if ok:
        row.update(fullcall_ms=fullcall(call), device_ms=device(call))
    emit(row)


if __name__ == "__main__":
    main()
