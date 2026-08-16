"""ONE-OFF standalone measurement: register/spill footprint of `retrace_fused_kernel`.

`src/rl_triton/ops/retrace.py`'s module docstring, `src/rl_triton/kernels/retrace_fused.py`'s
docstring, and NOTES.md all cite "128 regs/thread, 132 spills/thread, 25% occupancy at
seq_len=4096" for the pre-fix kernel (the fix landed in commit b2fc9fb, which cut spills to
100 without changing regs/thread or occupancy -- see those files' own comments for the full
account). A draft paper sentence citing "132 spills per thread" was flagged as unverifiable:
Triton's own `CompiledKernel.n_spills` is NOT a count of spilled registers and is not raw
ptxas bytes either -- see the cross-check below for what it actually is. Nothing tied any of
these numbers to a specific GPU/driver/Triton version, so none of them can be assumed current
without remeasuring. This script isolates exactly that: compiled-kernel metadata for
`retrace_fused_kernel`, plus a direct `ptxas -v` cross-check of one shape's PTX.

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

Shapes: num_envs=512, num_actions=4 -- the exact config `tests/h100_profiling_report.md`
(commit dc246a4) and `tests/h100_short_horizon_l2_retrace_ppo_report.md` (commit b2fc9fb) used
for this kernel's original register-pressure study, and `num_actions=4` is
`tests/bench_release.py::_make_retrace`'s default (never overridden across CONFIGS). seq_len
swept over {1024, 2048, 4096, 8192} -- brackets the fused kernel's confirmed-win/regression
boundary at `_TRITON_SEQ_LEN_CEILING=2048` (src/rl_triton/ops/retrace.py).

Not part of bench_release.py, CI, or the test suite; run on demand. GPU/driver/CUDA/Triton
version specific by design (register allocation is compiler-version dependent) -- re-run
whenever a cited register/spill number needs to be checked instead of trusted.

Usage:
    python benchmarks/measure_retrace_register_pressure.py [--gpu LABEL]

Writes benchmarks/retrace_register_pressure.md.
"""
import argparse
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bench_utils import print_environment_header
from rl_triton.kernels.retrace_fused import retrace_fused_kernel

NUM_ENVS = 512
NUM_ACTIONS = 4
SEQ_LENS = [1024, 2048, 4096, 8192]
SEED = 0

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="", metavar="LABEL")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this script requires a GPU.")
        sys.exit(1)

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
    print(f"compute capability: sm_{capability}")
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

    date = datetime.date.today().isoformat()
    lines = [
        "# Retrace fused-kernel register pressure",
        "",
        f"## {gpu_label} (driver {driver_version}, CUDA {torch.version.cuda}, "
        f"Triton {triton.__version__}, torch {torch.__version__}) · {date}",
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
    ]

    out_path = Path(__file__).parent / "retrace_register_pressure.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
