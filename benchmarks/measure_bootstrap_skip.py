"""ONE-OFF standalone measurement: cost of the HAS_BOOTSTRAP scalar-allocation skip.

`gae_fused_kernel`'s docstring (src/rl_triton/kernels/gae.py) and NOTES.md both cite
"~33% of this op's total time at 128x1024" for the HAS_BOOTSTRAP=False optimization --
skipping the wrapper's `torch.zeros(num_envs)` allocation-and-launch when the caller has
no `last_value`/`bootstrap_values` to inject. That figure predates this session and
nothing ties it to a specific GPU/driver/Triton version, so it cannot be assumed current
without remeasuring. This script isolates exactly that mechanism -- not a proxy for it.

Why the existing plain-vs-with-truncations tables in benchmarks.md do NOT measure this:
the with-truncations path forces HAS_TRUNCATIONS=True, which unconditionally also sets
HAS_BOOTSTRAP=True (src/rl_triton/ops/gae.py) and additionally pulls in the full 2D
truncateds array and 2D bootstrap tensor -- a categorically larger code path, not an
isolated view of the one scalar allocation this docstring describes. Comparing those two
tables conflates truncation-handling cost with bootstrap-allocation cost.

This script instead calls `gae_fused_kernel` directly (bypassing compute_gae's dispatch)
in both configurations at the exact same shape, holding everything else fixed:
  - HAS_BOOTSTRAP=True,  bootstrap_ptr = torch.zeros(num_envs)  (the pre-optimization
    behavior: the wrapper always allocated + passed this buffer, even though the kernel
    never reads its contents when there is nothing to bootstrap from)
  - HAS_BOOTSTRAP=False, bootstrap_ptr = None                   (current default path)

Correctness: both configurations must produce identical output (HAS_BOOTSTRAP=True with
an all-zeros buffer is mathematically the same computation as HAS_BOOTSTRAP=False) --
asserted before any timing number is trusted, not assumed from the kernel's docstring.

Not part of bench_release.py, CI, or the test suite; run on demand. GPU-and-toolchain
specific by design (this is a launch-overhead measurement, not a portable benchmark) --
re-run whenever the ~33% figure needs to be checked instead of trusted.

Usage:
    python benchmarks/measure_bootstrap_skip.py [--gpu LABEL] [--shapes 128x1024,64x512]

Writes benchmarks/bootstrap_skip.md.
"""
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TORCH_LOGS", "-dynamic")

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bench_utils import _bench_gpu, _warmup_gpu, assert_correctness
from rl_triton.kernels.gae import gae_fused_kernel

GAMMA = 0.99
LAMBDA = 0.95
SEED = 0

# (num_envs, seq_len) -- 128x1024 is the shape the existing "~33%" claim cites;
# the others bracket it to show whether the effect is 128x1024-specific or general.
DEFAULT_SHAPES = [(128, 1024), (64, 512), (512, 512), (4096, 128)]

_WARPS = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}


def _make_inputs(num_envs, seq_len, device="cuda"):
    torch.manual_seed(SEED)
    rewards     = torch.randn(num_envs, seq_len, device=device)
    values      = torch.randn(num_envs, seq_len, device=device)
    terminateds = (torch.rand(num_envs, seq_len, device=device) < 0.05).float()
    return rewards, values, terminateds


def _run_kernel(rewards, values, terminateds, has_bootstrap: bool):
    """Direct gae_fused_kernel invocation, HAS_TRUNCATIONS=False fixed, varying
    only HAS_BOOTSTRAP -- the one axis this script isolates."""
    num_envs, seq_len = rewards.shape
    out = torch.empty_like(rewards)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps  = _WARPS.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 2048 else 1

    if has_bootstrap:
        # Pre-optimization behavior: the wrapper always allocated this, even
        # though the kernel never reads it when there's nothing to bootstrap
        # from. Allocation happens INSIDE the timed closure -- exactly what
        # the old wrapper paid on every call.
        bootstrap_ptr = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    else:
        bootstrap_ptr = None

    gae_fused_kernel[(num_envs,)](
        rewards, values, terminateds, None,
        out, bootstrap_ptr,
        seq_len, rewards.stride(0),
        gamma=GAMMA, lambda_=LAMBDA,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
        HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=has_bootstrap,
    )
    return out


def _detect_gpu() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown GPU"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="", metavar="LABEL")
    parser.add_argument("--shapes", default="", metavar="LIST",
                        help="Comma-separated num_envsxseq_len, e.g. 128x1024,64x512. "
                             "Default: the built-in DEFAULT_SHAPES.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this script requires a GPU.")
        sys.exit(1)

    gpu_label = args.gpu or _detect_gpu()
    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = []
        for tok in args.shapes.split(","):
            ne, sl = tok.strip().split("x")
            shapes.append((int(ne), int(sl)))

    rows = []
    for num_envs, seq_len in shapes:
        rewards, values, terminateds = _make_inputs(num_envs, seq_len)

        out_with = _run_kernel(rewards, values, terminateds, has_bootstrap=True)
        out_without = _run_kernel(rewards, values, terminateds, has_bootstrap=False)
        assert_correctness(
            out_without, out_with,
            f"bootstrap_skip[{num_envs}x{seq_len}] (HAS_BOOTSTRAP=False vs =True, "
            "must be identical -- an all-zeros bootstrap buffer is mathematically "
            "a no-op)",
        )

        _warmup_gpu(_run_kernel, rewards, values, terminateds, has_bootstrap=True)
        with_ms = _bench_gpu(_run_kernel, rewards, values, terminateds, has_bootstrap=True)
        _warmup_gpu(_run_kernel, rewards, values, terminateds, has_bootstrap=False)
        without_ms = _bench_gpu(_run_kernel, rewards, values, terminateds, has_bootstrap=False)

        saved_ms = with_ms - without_ms
        pct_of_with = saved_ms / with_ms * 100 if with_ms > 0 else float("nan")
        rows.append(dict(
            num_envs=num_envs, seq_len=seq_len,
            with_ms=with_ms, without_ms=without_ms,
            saved_ms=saved_ms, pct_of_with=pct_of_with,
        ))
        print(f"  [{num_envs}x{seq_len}] HAS_BOOTSTRAP=True: {with_ms:.4f}ms  "
              f"HAS_BOOTSTRAP=False: {without_ms:.4f}ms  "
              f"saved: {saved_ms:.4f}ms ({pct_of_with:.1f}% of the True-path total)")

    date = datetime.date.today().isoformat()
    header = [
        "# HAS_BOOTSTRAP scalar-allocation skip: isolated measurement",
        "",
        "Isolates `gae_fused_kernel`'s `HAS_BOOTSTRAP` compile-time branch "
        "directly, bypassing `compute_gae`'s dispatch, so no other cost "
        "(truncation handling, wrapper overhead) is mixed in.",
        "",
        "Run via `python benchmarks/measure_bootstrap_skip.py [--gpu LABEL]`. "
        "Not part of the release sweep (`bench_release.py`) -- "
        "GPU/Triton/driver-version specific by design; re-run to check "
        "whether a cited percentage (e.g. the `gae_fused_kernel` "
        "docstring's \"~33% at 128x1024\") is still current rather than "
        "trusting it indefinitely.",
        "",
        "Both configurations are asserted bit-for-bit identical in output "
        "before any timing is trusted (an all-zeros bootstrap buffer is "
        "mathematically a no-op, so `HAS_BOOTSTRAP=True` with zeros and "
        "`HAS_BOOTSTRAP=False` must agree exactly).",
        "",
        "This file is an UPSERT KEYED BY GPU, like "
        "`docs/benchmark-history/unreleased.md` -- re-running on the same "
        "GPU replaces that GPU's own section in place; running on a "
        "different GPU appends a new section, leaving others untouched. "
        "Never a wholesale overwrite.",
    ]

    section_lines = [
        f"## {gpu_label} · {date}",
        "",
        "| num_envs | seq_len | HAS_BOOTSTRAP=True (ms) | HAS_BOOTSTRAP=False (ms) | saved (ms) | saved (% of True-path total) |",
        "|:--------:|:-------:|:------------------------:|:--------------------------:|:----------:|:-----------------------------:|",
    ]
    for r in rows:
        section_lines.append(
            f"| {r['num_envs']:>8} | {r['seq_len']:>7} | "
            f"{r['with_ms']:>24.4f} | {r['without_ms']:>26.4f} | "
            f"{r['saved_ms']:>10.4f} | {r['pct_of_with']:>29.1f} |"
        )
    new_section = "\n".join(section_lines) + "\n"

    out_path = Path(__file__).parent / "bootstrap_skip.md"
    section_re = re.compile(r"(?m)^## (.+?) · \d{4}-\d{2}-\d{2}\n\n(?:\|.*\n)+")
    if out_path.exists():
        existing = out_path.read_text()
        # Split off everything before the first "## <gpu> ·" section (the
        # shared header block) so it can be regenerated fresh each run
        # without accumulating stale duplicates.
        first_section_match = re.search(r"(?m)^## ", existing)
        prior_sections_text = existing[first_section_match.start():] if first_section_match else ""
        kept_full = [
            m.group(0) for m in section_re.finditer(prior_sections_text)
            if m.group(1) != gpu_label
        ]
        body = "\n".join(kept_full) + ("\n" if kept_full else "") + new_section
    else:
        body = new_section

    out_path.write_text("\n".join(header) + "\n\n" + body)
    print(f"\nWrote {out_path} (section for {gpu_label!r})")


if __name__ == "__main__":
    main()
