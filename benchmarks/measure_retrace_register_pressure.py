"""ONE-OFF standalone measurement for Retrace: register/spill footprint of `retrace_fused_kernel`,
plus a path-timing comparison of fused vs. generic vs. torch.compile.

Part 1 -- register pressure. `src/rl_triton/ops/retrace.py`'s module docstring,
`src/rl_triton/kernels/retrace_fused.py`'s docstring, and NOTES.md all cite "128 regs/thread, 132
spills/thread, 25% occupancy at seq_len=4096" for the pre-fix kernel (the fix landed in commit
b2fc9fb, which cut spills to 100 without changing regs/thread or occupancy -- see those files' own
comments for the full account). A draft paper sentence citing "132 spills per thread" was flagged as
unverifiable: Triton's own `CompiledKernel.n_spills` is NOT a count of spilled registers and is not
raw ptxas bytes either -- see the cross-check below for what it actually is. Nothing tied any of
these numbers to a specific GPU/driver/Triton version, so none of them can be assumed current
without remeasuring. Part 1 records compiled-kernel metadata for `retrace_fused_kernel`, plus a
direct `ptxas -v` cross-check of one shape's PTX.

Part 2 -- path timing. Table 4 (benchmarks.md, tests/bench_release.py) reports compute_retrace at
(512, 4096) -- the dispatched *generic* path, since 4096 > _TRITON_SEQ_LEN_CEILING -- while the
_TRITON_SEQ_LEN_CEILING comment in src/rl_triton/ops/retrace.py cites "fused 0.73x vs generic 0.47x
at 4096" and "generic overtakes fused at 8192" with no benchmark behind them. Part 2 times, on the
same inputs and with the release protocol (bench_utils._warmup_gpu 20 / _bench_gpu / _device_profile,
n_iter=_n_iter_gpu, torch._dynamo.reset() per shape, one fresh process per shape and repeat):
  A dispatched      compute_retrace (public API; what Table 4 measures)
  B fused-forced    compute_retrace_fused
  C generic-forced  compute_retrace with retrace._TRITON_SEQ_LEN_CEILING patched to 0
  D baseline        torch.compile(vectorized_retrace), called as bench_release.bench_retrace does
Every arm is gated against tests/test_retrace.py::reference_retrace (atol=rtol=1e-4) before timing.

What `CompiledKernel.n_regs` / `.n_spills` actually are (confirmed by reading
`triton/backends/nvidia/driver.c`'s `loadBinary`, not assumed):
  - `n_regs`   = `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)`            -- exact register
    count per thread, matches ptxas's "Used N registers" line directly.
  - `n_spills` = `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) / 4` -- i.e. the
    per-thread **cumulative stack frame size in 4-byte words**, NOT a register-spill count and
    NOT raw bytes (it's already divided by 4). ptxas separately reports "N bytes spill stores",
    "N bytes spill loads", and "N bytes cumulative stack size" as three distinct numbers;
    `n_spills` corresponds to the *stack size* one specifically (confirmed below: matches
    `stack_size_bytes / 4` exactly, not `spill_store_bytes / 4` or `spill_load_bytes / 4`,
    which differ from it here). Reporting `n_spills` as "spills per thread" conflates a
    stack-frame word count with an instruction/byte count ptxas never reports under that name.

Occupancy is derived from `n_regs` alone (128 regs/thread x 32 threads/warp x
`num_warps` = 4096 regs/warp-group at num_warps=16 -> exactly H100's 65536-register SM file
at num_warps=16, giving 16 of 64 max warps/SM = 25%). It does NOT depend on `n_spills` --
spills are a *consequence* of the allocator capping registers per thread to fit the file, not
an independent input to the occupancy formula. Do not present the two numbers as jointly
producing the occupancy figure.

Shapes (Part 1): num_envs=512, num_actions=4 -- the exact config `tests/h100_profiling_report.md`
(commit dc246a4) and `tests/h100_short_horizon_l2_retrace_ppo_report.md` (commit b2fc9fb) used
for this kernel's original register-pressure study, and `num_actions=4` is
`tests/bench_release.py::_make_retrace`'s default (never overridden across CONFIGS). seq_len
swept over {1024, 2048, 4096, 8192} -- brackets the fused kernel's confirmed-win/regression
boundary at `_TRITON_SEQ_LEN_CEILING=2048` (src/rl_triton/ops/retrace.py). Part 2 uses the same
num_envs=512 seq_len sweep plus (128, 4096) and (4096, 4096).

Not part of bench_release.py, CI, or the test suite; run on demand. GPU/driver/CUDA/Triton
version specific by design (register allocation is compiler-version dependent) -- re-run
whenever a cited register/spill or timing number needs to be checked instead of trusted.

Usage:
    python benchmarks/measure_retrace_register_pressure.py [--gpu LABEL] [--no-timing] [--repeats N]

Writes benchmarks/retrace_register_pressure.md (both parts; Part 2 omitted with --no-timing).
"""
import argparse
import contextlib
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bench_utils import (
    _bench_gpu, _device_profile, _n_iter_gpu, _warmup_gpu, assert_correctness, print_environment_header,
)
from test_retrace import reference_retrace, vectorized_retrace
import rl_triton.ops.retrace as retrace_mod
from rl_triton.kernels.retrace_fused import retrace_fused_kernel
from rl_triton.ops.retrace import compute_retrace
from rl_triton.ops.retrace_fused import compute_retrace_fused

NUM_ENVS = 512
NUM_ACTIONS = 4
SEQ_LENS = [1024, 2048, 4096, 8192]
SEED = 0
GAMMA = 0.99
TIMING_SHAPES = [(NUM_ENVS, t) for t in SEQ_LENS] + [(128, 4096), (4096, 4096)]
ARMS = ("A", "B", "C", "D")
ARM_NAMES = {"A": "dispatched compute_retrace", "B": "fused-forced compute_retrace_fused",
             "C": "generic-forced compute_retrace (ceiling=0)", "D": "torch.compile(vectorized_retrace)"}
EXPECTED_ENV = {"gpu": "NVIDIA H100 80GB HBM3", "torch": "2.4.1+cu124", "triton": "3.0.0",
                "ptxas": "12.4.99"}

# Same launch-config table compute_retrace_fused uses (src/rl_triton/ops/retrace_fused.py) --
# duplicated here (not imported) because this script calls retrace_fused_kernel directly to
# get the CompiledKernel handle back, which compute_retrace_fused's wrapper doesn't expose.
_WARPS = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}


def _make_inputs(num_envs, seq_len, num_actions, device="cuda"):
    torch.manual_seed(SEED)
    return dict(
        action_probs_target=torch.softmax(
            torch.randn(num_envs, seq_len, num_actions, device=device), dim=-1),
        action_probs_behavior=torch.rand(num_envs, seq_len, device=device) * 0.8 + 0.1,
        q_values=torch.randn(num_envs, seq_len, device=device),
        next_q_values_all=torch.randn(num_envs, seq_len, num_actions, device=device),
        actions=torch.randint(0, num_actions, (num_envs, seq_len), device=device),
        rewards=torch.randn(num_envs, seq_len, device=device),
        truncateds=(torch.rand(num_envs, seq_len, device=device) < 0.05).float(),
        terminated=(torch.rand(num_envs, seq_len, device=device) < 0.05).float(),
    )


def _launch(num_envs, seq_len, num_actions, inputs):
    """Direct retrace_fused_kernel launch (bypassing compute_retrace_fused's wrapper) so the
    CompiledKernel handle -- and its n_regs/n_spills metadata -- comes straight back."""
    out = torch.empty(num_envs, seq_len, device=inputs["rewards"].device)
    advantages = torch.empty_like(out)
    pi_at_scratch = torch.empty_like(out)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    ACTION_BLOCK = triton.next_power_of_2(num_actions)
    num_warps = _WARPS.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 2048 else 1

    ck = retrace_fused_kernel[(num_envs,)](
        inputs["action_probs_target"], inputs["action_probs_behavior"], inputs["q_values"],
        inputs["next_q_values_all"], inputs["actions"], inputs["rewards"],
        inputs["truncateds"], inputs["terminated"], out, advantages, pi_at_scratch,
        seq_len, num_actions, inputs["rewards"].stride(0), inputs["action_probs_target"].stride(0),
        gamma=0.99, lambda_=1.0, c_bar=1.0, rho_bar=1.0,
        BLOCK_SIZE=BLOCK_SIZE, ACTION_BLOCK=ACTION_BLOCK,
        num_warps=num_warps, num_stages=num_stages,
    )
    return ck, BLOCK_SIZE, num_warps, num_stages


def _occupancy_pct(n_regs, num_warps, regs_per_sm=65536, max_warps_per_sm=64):
    """Theoretical occupancy from registers alone -- does NOT use n_spills. Spills are a
    consequence of the allocator capping registers per thread to fit the file, not an
    independent input to this formula."""
    regs_per_warp_group = n_regs * 32 * num_warps
    blocks_per_sm = regs_per_sm // regs_per_warp_group
    warps_resident = blocks_per_sm * num_warps
    return 100.0 * warps_resident / max_warps_per_sm


def _ptxas_cross_check(ck, capability: int) -> str:
    """Re-run ptxas -v directly on this kernel's PTX (extracted from the CompiledKernel's own
    asm dict) to capture the verbatim 'ptxas info:' lines Triton's compile step normally
    discards after parsing n_regs/n_spills out of them."""
    from triton.backends.nvidia.compiler import _path_to_binary
    ptxas, _ = _path_to_binary("ptxas")
    suffix = "a" if capability == 90 else ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ptx", delete=False) as f:
        f.write(ck.asm["ptx"])
        ptx_path = f.name
    cubin_path = ptx_path + ".o"
    cmd = [ptxas, "-lineinfo", "-v", f"--gpu-name=sm_{capability}{suffix}", ptx_path, "-o", cubin_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stderr


# ---------------------------------------------------------------------------
# Part 2: path timing (fused vs. generic vs. torch.compile)
# ---------------------------------------------------------------------------

def _api_args(inputs):
    """compute_retrace / reference_retrace / vectorized_retrace positional order:
    ..., rewards, terminateds, truncateds."""
    return (inputs["action_probs_target"], inputs["action_probs_behavior"], inputs["q_values"],
            inputs["next_q_values_all"], inputs["actions"], inputs["rewards"],
            inputs["terminated"], inputs["truncateds"])


def _fused_args(inputs):
    """compute_retrace_fused positional order: ..., rewards, truncateds, terminated
    (truncateds BEFORE terminated, unlike compute_retrace)."""
    return (inputs["action_probs_target"], inputs["action_probs_behavior"], inputs["q_values"],
            inputs["next_q_values_all"], inputs["actions"], inputs["rewards"],
            inputs["truncateds"], inputs["terminated"])


@contextlib.contextmanager
def _generic_forced():
    """Arm C: _TRITON_SEQ_LEN_CEILING=0, so compute_retrace (which reads the module constant at
    call time) takes the materialize-u/v + _run_scan branch at every seq_len. Always restored."""
    original = retrace_mod._TRITON_SEQ_LEN_CEILING
    retrace_mod._TRITON_SEQ_LEN_CEILING = 0
    try:
        yield
    finally:
        retrace_mod._TRITON_SEQ_LEN_CEILING = original


def _dispatched_path(args, ctx):
    """Which branch compute_retrace takes under `ctx`: spies on the module-level
    compute_retrace_fused / _run_scan names compute_retrace calls through (spies are only
    installed for this one check call, never during timing)."""
    calls = {"fused": 0, "scan": 0}
    orig_fused, orig_scan = retrace_mod.compute_retrace_fused, retrace_mod._run_scan

    def fused_spy(*a, **k):
        calls["fused"] += 1
        return orig_fused(*a, **k)

    def scan_spy(*a, **k):
        calls["scan"] += 1
        return orig_scan(*a, **k)

    retrace_mod.compute_retrace_fused, retrace_mod._run_scan = fused_spy, scan_spy
    try:
        with ctx():
            compute_retrace(*args, gamma=GAMMA)
    finally:
        retrace_mod.compute_retrace_fused, retrace_mod._run_scan = orig_fused, orig_scan
    if calls == {"fused": 1, "scan": 0}:
        return "fused"
    if calls == {"fused": 0, "scan": 1}:
        return "generic"
    raise AssertionError(f"unexpected compute_retrace call pattern: {calls}")


def _timing_worker(num_envs, seq_len):
    """One shape, all four arms, in this (fresh) process. Prints one JSON line."""
    torch._dynamo.reset()
    inputs = _make_inputs(num_envs, seq_len, NUM_ACTIONS)
    api, fused = _api_args(inputs), _fused_args(inputs)
    ref = reference_retrace(*api, gamma=GAMMA)
    compiled_vec = torch.compile(vectorized_retrace)
    arms = {
        "A": (compute_retrace, api, contextlib.nullcontext),
        "B": (compute_retrace_fused, fused, contextlib.nullcontext),
        "C": (compute_retrace, api, _generic_forced),
        "D": (compiled_vec, api, contextlib.nullcontext),
    }
    ceiling = retrace_mod._TRITON_SEQ_LEN_CEILING
    row = dict(num_envs=num_envs, seq_len=seq_len, n_iter=_n_iter_gpu(seq_len, num_envs),
               ceiling=ceiling, A_path=_dispatched_path(api, contextlib.nullcontext))
    assert row["A_path"] == ("fused" if seq_len <= ceiling else "generic"), row
    assert _dispatched_path(api, _generic_forced) == "generic", "ceiling patch did not take effect"
    assert retrace_mod._TRITON_SEQ_LEN_CEILING == ceiling, "ceiling not restored"

    # Correctness gate: every arm vs. reference_retrace, both outputs (targets, advantages).
    for k, (fn, args, ctx) in arms.items():
        with ctx():
            out = fn(*args, gamma=GAMMA)
        assert_correctness(out, ref, f"{k}[{num_envs}x{seq_len}]")
        row[f"{k}_max_abs_err"] = max((o - r).abs().max().item() for o, r in zip(out, ref))

    for k, (fn, args, ctx) in arms.items():
        with ctx():
            _warmup_gpu(fn, *args, gamma=GAMMA)
            row[f"{k}_ms"] = _bench_gpu(fn, *args, gamma=GAMMA, n_iter=row["n_iter"])
            row[f"{k}_dev_ms"], _ = _device_profile(fn, *args, gamma=GAMMA)
    assert retrace_mod._TRITON_SEQ_LEN_CEILING == ceiling, "ceiling not restored"
    print(json.dumps(row), flush=True)


def _run_timing(repeats):
    """Every (repeat, shape) in its own subprocess; returns (rows, skipped)."""
    rows, skipped = [], []
    for rep in range(1, repeats + 1):
        for num_envs, seq_len in TIMING_SHAPES:
            p = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--timing-worker", str(num_envs), str(seq_len)],
                capture_output=True, text=True)
            if p.returncode != 0:
                if "out of memory" in p.stderr.lower():
                    skipped.append(dict(repeat=rep, num_envs=num_envs, seq_len=seq_len, reason="CUDA OOM"))
                    print(f"  repeat {rep}  [{num_envs}x{seq_len}]  SKIPPED (CUDA OOM)", flush=True)
                    continue
                raise RuntimeError(f"timing worker {num_envs}x{seq_len} failed:\n{p.stderr[-3000:]}")
            row = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
            row["repeat"] = rep
            rows.append(row)
            print(f"  repeat {rep}  [{num_envs:>4}x{seq_len:>4}]  A({row['A_path']}) {row['A_ms']:.4f}  "
                  f"B {row['B_ms']:.4f}  C {row['C_ms']:.4f}  D {row['D_ms']:.4f} ms  "
                  f"D/A {row['D_ms'] / row['A_ms']:.2f}  D/B {row['D_ms'] / row['B_ms']:.2f}  "
                  f"D/C {row['D_ms'] / row['C_ms']:.2f}", flush=True)
    return rows, skipped


def _ratio(row, arm, dev=False):
    sfx = "_dev_ms" if dev else "_ms"
    return row["D" + sfx] / row[arm + sfx]


def _stats(rows, num_envs, seq_len, arm, dev=False):
    vals = sorted(_ratio(r, arm, dev) for r in rows if (r["num_envs"], r["seq_len"]) == (num_envs, seq_len))
    if not vals:
        return None
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return dict(median=med, lo=vals[0], hi=vals[-1], vals=vals)


def _fmt_stat(st):
    return "n/a" if st is None else f"{st['median']:.2f} ({st['lo']:.2f}-{st['hi']:.2f})"


def _summary_lines(rows):
    lines = [
        "| num_envs | seq_len | A path | D/A full-call | D/B full-call | D/C full-call | "
        "D/A device | D/B device | D/C device |",
        "|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for num_envs, seq_len in TIMING_SHAPES:
        shape_rows = [r for r in rows if (r["num_envs"], r["seq_len"]) == (num_envs, seq_len)]
        if not shape_rows:
            lines.append(f"| {num_envs} | {seq_len} | skipped | | | | | | |")
            continue
        cells = [_fmt_stat(_stats(rows, num_envs, seq_len, a, dev)) for dev in (False, True) for a in "ABC"]
        lines.append(f"| {num_envs} | {seq_len} | {shape_rows[0]['A_path']} | " + " | ".join(cells) + " |")
    return lines


def _claim_line(label, claim, st, half_unit):
    """A claim holds if it lies inside the measured [min, max] widened by half a unit of the
    claim's last printed digit (its own rounding precision)."""
    if st is None:
        return f"- {label}: claimed {claim}x; shape skipped, not measured."
    holds = st["lo"] - half_unit <= claim <= st["hi"] + half_unit
    diff, spread = st["median"] - claim, st["hi"] - st["lo"]
    beyond_rounding = max(0.0, abs(diff) - half_unit)
    return (f"- {label}: claimed {claim}x; measured median {st['median']:.3f}x "
            f"(range {st['lo']:.3f}-{st['hi']:.3f}, spread {spread:.3f}). **{'Holds' if holds else 'Does not hold'}.** "
            f"Median - claim = {diff:+.3f}; beyond the claim's rounding (+/-{half_unit}) that is "
            f"{beyond_rounding:.3f}, which {'exceeds' if beyond_rounding > spread else 'does not exceed'} "
            f"the run-to-run spread.")


def _order_line(label, rows, num_envs, seq_len, generic_faster_claimed):
    shape_rows = [r for r in rows if (r["num_envs"], r["seq_len"]) == (num_envs, seq_len)]
    if not shape_rows:
        return f"- {label}: shape skipped, not measured."
    n_generic = sum(r["C_ms"] < r["B_ms"] for r in shape_rows)
    n = len(shape_rows)
    gaps = [r["B_ms"] / r["C_ms"] for r in shape_rows]
    holds = (n_generic == n) if generic_faster_claimed else (n_generic == 0)
    return (f"- {label}: generic (C) faster than fused (B) in {n_generic}/{n} repeats at "
            f"{num_envs}x{seq_len} (B/C full-call time ratio {min(gaps):.3f}-{max(gaps):.3f}). "
            f"**{'Holds' if holds else 'Does not hold'}**.")


def _timing_markdown(rows, skipped, repeats, gpu_busy_note):
    ref_note = (
        "Correctness reference: `tests/test_retrace.py::reference_retrace`, gating both outputs "
        "(targets, advantages) at atol=rtol=1e-4 for every arm before it is timed. "
        "`bench_release.bench_retrace` gates against `_ref_retrace` instead, but `_ref_retrace` takes a "
        "single `dones` flag and applies it to the TD target as well, whereas these inputs carry separate "
        "terminations and truncations (5% each, from `_make_inputs` above; Table 4's `_make_retrace` "
        "inputs have zero truncations). `reference_retrace` takes the same (terminateds, truncateds) "
        "arguments as `compute_retrace`, so it is the compatible one. Timing does not depend on the "
        "flag values: every arm executes the same operations either way.")
    lines = [
        "",
        "## Path timing: fused vs. generic vs. torch.compile",
        "",
        f"num_actions={NUM_ACTIONS}, gamma={GAMMA}; all arms see identical inputs at a shape (`_make_inputs`, seed {SEED}).",
        "",
        "| arm | what is called |",
        "|:---:|---|",
        "| A | `compute_retrace` -- public API, dispatched path (what Table 4 measures) |",
        "| B | `compute_retrace_fused` -- fused kernel forced at every seq_len (args mapped to its truncateds-before-terminated order) |",
        "| C | `compute_retrace` with `rl_triton.ops.retrace._TRITON_SEQ_LEN_CEILING = 0` (try/finally, restored) -- generic materialize-u/v + `_run_scan` path at every seq_len; a spy check asserts this branch was taken |",
        "| D | `torch.compile(vectorized_retrace)` -- baseline, called as `bench_release.bench_retrace` does |",
        "",
        f"Protocol (same as `tests/bench_release.py`): `torch._dynamo.reset()` per shape; `_warmup_gpu` (20); "
        f"full-call = `_bench_gpu` (CUDA events, 5 trials x `n_iter=_n_iter_gpu(seq_len, num_envs)`, min of "
        f"trial medians); device = `_device_profile` (torch.profiler kernel time). Each shape runs in its own "
        f"fresh subprocess; the whole sweep is repeated {repeats}x, each repeat as separate subprocesses. "
        f"D/X > 1 means arm X is faster than torch.compile.",
        "",
        ref_note,
        "",
        gpu_busy_note,
        "",
    ]
    if skipped:
        lines += ["Skipped shapes: " + "; ".join(
            f"repeat {s['repeat']} {s['num_envs']}x{s['seq_len']} ({s['reason']})" for s in skipped), ""]
    for rep in range(1, repeats + 1):
        lines += [
            f"### Repeat {rep}",
            "",
            "| num_envs | seq_len | A path | A ms | B ms | C ms | D ms | D/A | D/B | D/C | "
            "A dev ms | B dev ms | C dev ms | D dev ms | D/A dev | D/B dev | D/C dev |",
            "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in (r for r in rows if r["repeat"] == rep):
            lines.append(
                f"| {r['num_envs']} | {r['seq_len']} | {r['A_path']} | "
                + " | ".join(f"{r[a + '_ms']:.4f}" for a in ARMS) + " | "
                + " | ".join(f"{_ratio(r, a):.2f}" for a in "ABC") + " | "
                + " | ".join(f"{r[a + '_dev_ms']:.4f}" for a in ARMS) + " | "
                + " | ".join(f"{_ratio(r, a, dev=True):.2f}" for a in "ABC") + " |")
        lines.append("")
    max_err = max(r[f"{a}_max_abs_err"] for r in rows for a in ARMS) if rows else float("nan")
    lines += [
        f"All {len(rows)} (repeat, shape) runs passed the gate for all four arms; max abs error vs. "
        f"reference over all arms and runs: {max_err:.2e}.",
        "",
        f"### Summary over {repeats} repeats: median (min-max) of D/X",
        "",
        *_summary_lines(rows),
        "",
        "### Comparison with existing claims (full-call, num_envs=512)",
        "",
        "Rule: a numeric claim holds if it lies inside the measured min-max range widened by half a unit "
        "of the claim's last digit (0.05 for 0.6x, 0.005 for 0.73x / 0.47x). Spread = max - min over repeats. "
        "The retrace.py comment does not state num_envs; num_envs=512 is compared here.",
        "",
        _claim_line("Table 4, A (dispatched, = generic) at 512x4096", 0.6,
                    _stats(rows, NUM_ENVS, 4096, "A"), 0.05),
        _claim_line("retrace.py comment, fused (B) at 4096", 0.73, _stats(rows, NUM_ENVS, 4096, "B"), 0.005),
        _claim_line("retrace.py comment, generic (C) at 4096", 0.47, _stats(rows, NUM_ENVS, 4096, "C"), 0.005),
        _order_line("retrace.py comment, fused faster than generic at 4096", rows, NUM_ENVS, 4096, False),
        _order_line("retrace.py comment, generic overtakes fused at 8192", rows, NUM_ENVS, 8192, True),
    ]
    return lines


def _ptxas_version():
    from triton.backends.nvidia.compiler import _path_to_binary
    ptxas, _ = _path_to_binary("ptxas")
    out = subprocess.run([ptxas, "--version"], capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-2].split(" V")[-1] if " V" in out else out.strip()


def _gpu_busy_note():
    """Other compute processes on the GPU right now (this process's own context excluded)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    others = [l.strip() for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != str(os.getpid())]
    if others:
        return "**Other processes on the GPU before timing started:** " + "; ".join(others)
    return "`nvidia-smi` showed no other compute process on the GPU before timing started."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="", metavar="LABEL")
    parser.add_argument("--no-timing", action="store_true", help="skip Part 2 (path timing)")
    parser.add_argument("--repeats", type=int, default=3, metavar="N", help="Part 2 sweep repeats (default 3)")
    parser.add_argument("--timing-worker", nargs=2, type=int, metavar=("NUM_ENVS", "SEQ_LEN"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this script requires a GPU.")
        sys.exit(1)
    if args.timing_worker:
        _timing_worker(*args.timing_worker)
        return

    print_environment_header("measure_retrace_register_pressure.py")
    driver_version = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    gpu_label = args.gpu or torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability()
    capability = cc_major * 10 + cc_minor
    print(f"driver:           {driver_version}")
    print(f"CUDA (torch):     {torch.version.cuda}")
    ptxas_version = _ptxas_version()
    print(f"compute capability: sm_{capability}")
    print(f"ptxas:            {ptxas_version}")
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "triton": triton.__version__, "ptxas": ptxas_version}
    env_mismatch = [f"{k} is {env[k]} (expected {v})" for k, v in EXPECTED_ENV.items() if env[k] != v]
    for m in env_mismatch:
        print(f"WARNING: {m}")
    print(f"num_envs={NUM_ENVS}, num_actions={NUM_ACTIONS}, seq_lens={SEQ_LENS}\n", flush=True)

    rows = []
    ptxas_log_4096 = None
    for seq_len in SEQ_LENS:
        inputs = _make_inputs(NUM_ENVS, seq_len, NUM_ACTIONS)
        ck, block_size, num_warps, num_stages = _launch(NUM_ENVS, seq_len, NUM_ACTIONS, inputs)
        occ = _occupancy_pct(ck.n_regs, num_warps)
        rows.append(dict(
            seq_len=seq_len, block_size=block_size, num_warps=num_warps, num_stages=num_stages,
            n_regs=ck.n_regs, n_spills=ck.n_spills, occupancy_pct=occ,
        ))
        print(f"  seq_len={seq_len:>5}  BLOCK_SIZE={block_size:>5}  num_warps={num_warps:>2}  "
              f"n_regs={ck.n_regs:>4}  n_spills={ck.n_spills:>4}  occupancy={occ:.1f}%", flush=True)

        if seq_len == 4096:
            ptxas_log_4096 = _ptxas_cross_check(ck, capability)

    print("\n--- ptxas -v cross-check at seq_len=4096 (verbatim stderr) ---")
    print(ptxas_log_4096)

    timing_lines = []
    if not args.no_timing:
        del inputs, ck
        torch.cuda.empty_cache()
        gpu_busy_note = _gpu_busy_note()
        print(f"\n--- Part 2: path timing, {args.repeats} repeats, shapes {TIMING_SHAPES} ---")
        print(gpu_busy_note.replace("**", ""), flush=True)
        timing_rows, skipped = _run_timing(args.repeats)
        timing_lines = _timing_markdown(timing_rows, skipped, args.repeats, gpu_busy_note)
        print(f"\nSummary over {args.repeats} repeats: median (min-max) of D/X (>1 = faster than torch.compile)")
        print("\n".join(_summary_lines(timing_rows)))
        print("\n" + "\n".join(l for l in timing_lines[timing_lines.index(
            "### Comparison with existing claims (full-call, num_envs=512)"):] if l.startswith("- ")))

    date = datetime.date.today().isoformat()
    lines = [
        "# Retrace fused-kernel register pressure and path timing",
        "",
        f"## {gpu_label} (driver {driver_version}, CUDA {torch.version.cuda}, "
        f"Triton {triton.__version__}, torch {torch.__version__}, ptxas {ptxas_version}) · {date}",
        "",
        *[f"**Environment differs from the expected H100 setup: {m}.**\n" for m in env_mismatch],
        "Generated by `benchmarks/measure_retrace_register_pressure.py`: (1) the register/spill "
        "footprint of `retrace_fused_kernel`, and "
        + ("(2) a timing comparison of the fused path vs. the generic path vs. torch.compile "
           "(section \"Path timing\" below)." if not args.no_timing else
           "(2) a fused vs. generic vs. torch.compile timing comparison, skipped in this run (`--no-timing`)."),
        "",
        f"`retrace_fused_kernel`, num_envs={NUM_ENVS}, num_actions={NUM_ACTIONS} -- the shape "
        "`tests/h100_profiling_report.md` and `tests/h100_short_horizon_l2_retrace_ppo_report.md` "
        "used for this kernel's original register-pressure study.",
        "",
        "`n_regs` = ptxas \"Used N registers\" (exact). `n_spills` = Triton's "
        "`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4`, i.e. the per-thread cumulative stack frame "
        "size in 4-byte words -- confirmed below to match ptxas's \"N bytes cumulative stack "
        "size\" / 4, NOT \"N bytes spill stores\" or \"N bytes spill loads\" / 4. Occupancy is "
        "derived from `n_regs` alone (registers/thread x 32 x num_warps vs. the SM's "
        "65536-register file); it does not depend on `n_spills`.",
        "",
        "| seq_len | BLOCK_SIZE | num_warps | n_regs | n_spills (words) | occupancy |",
        "|:-------:|:----------:|:---------:|:------:|:-----------------:|:---------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seq_len']:>7} | {r['block_size']:>10} | {r['num_warps']:>9} | "
            f"{r['n_regs']:>6} | {r['n_spills']:>18} | {r['occupancy_pct']:>8.1f}% |"
        )
    lines += [
        "",
        "### `ptxas -v` cross-check at seq_len=4096 (verbatim stderr, re-run directly on the "
        "PTX Triton generated for this kernel)",
        "",
        "```",
        ptxas_log_4096.strip(),
        "```",
        *timing_lines,
    ]

    out_path = Path(__file__).parent / "retrace_register_pressure.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
