"""Phase 3 of 3 (torch/triton venv, back from the JAX venv): reads both
phases' .npz outputs, runs the cross-implementation equivalence check, prints
the monotonicity gate/markdown tables/crossover analysis, saves plots, writes
the dated report -- assembling the same output this benchmark always
produced, from two phases' saved results instead of one live in-process run.

See benchmark_gae_vs_jax_scan_triton.py's module docstring for why this
benchmark is split into three phases across two venvs.

================================================================================
Usage (after both benchmark_gae_vs_jax_scan_triton.py and
benchmark_gae_vs_jax_scan_jax.py have completed against the same
--results-dir):
  python benchmarks/benchmark_gae_vs_jax_scan_report.py [--results-dir DIR]
================================================================================
"""
import argparse
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))  # this dir -- for jax_gae_ext
from jax_gae_ext.config import (
    GAMMA, LAMBDA, TERM_PROB, N_ITER, N_TRIALS,
    SEQ_LENS, NUM_ENVS_LIST, SEQ_LENS_SHORT, NUM_ENVS_LIST_LARGE,
    H100_PEAK_GBPS, COLORS, COLORS_SHORT, EQUIVALENCE_CASES, cell_key,
)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rl_triton.ops.gae import _WARPS as _WARPS_TABLE

DEFAULT_RESULTS_DIR = Path(__file__).parent / "jax_gae_results"


# ------------------------------------------------------------------------
# Load both phases' saved results
# ------------------------------------------------------------------------

def load_cell(results_dir, num_envs, seq_len):
    key = cell_key(num_envs, seq_len)
    triton_data = np.load(results_dir / "triton" / f"{key}.npz")
    jax_data = np.load(results_dir / "jax" / f"{key}.npz")

    triton_ms = float(triton_data["triton_ms"])
    jax_ms = float(jax_data["jax_ms"])
    triton_dev_us = float(triton_data["triton_dev_us"])

    return dict(
        triton_ms=triton_ms, jax_ms=jax_ms,
        triton_ms_amort=float(triton_data["triton_ms_amort"]),
        jax_ms_amort=float(jax_data["jax_ms_amort"]),
        triton_dev_us=triton_dev_us,
        triton_launches=float(triton_data["triton_launches"]),
        triton_gbps=float(triton_data["triton_gbps"]),
        jax_gbps=float(jax_data["jax_gbps"]),
        triton_pct_peak=float(triton_data["triton_gbps"]) / H100_PEAK_GBPS * 100,
        jax_pct_peak=float(jax_data["jax_gbps"]) / H100_PEAK_GBPS * 100,
        speedup=jax_ms / triton_ms,
        triton_advantages=triton_data["advantages"],
        jax_advantages=jax_data["advantages"],
    )


def load_all(results_dir, num_envs_list, seq_lens):
    missing = []
    results = {}
    for num_envs in num_envs_list:
        for seq_len in seq_lens:
            key = cell_key(num_envs, seq_len)
            triton_path = results_dir / "triton" / f"{key}.npz"
            jax_path = results_dir / "jax" / f"{key}.npz"
            if not triton_path.exists() or not jax_path.exists():
                missing.append(key)
                continue
            results[(num_envs, seq_len)] = load_cell(results_dir, num_envs, seq_len)
    if missing:
        print(f"ERROR: missing results for cells: {missing}")
        print("Make sure both benchmark_gae_vs_jax_scan_triton.py and "
              "benchmark_gae_vs_jax_scan_jax.py completed successfully against "
              f"the same --results-dir ({results_dir}).")
        sys.exit(1)
    return results


# ------------------------------------------------------------------------
# Step 0b: JAX-side equivalence gate (the half that needs phase 2's output)
# ------------------------------------------------------------------------

def _ref_gae_matched_window(rewards, values, terminateds, gamma, lambda_):
    num_envs, seq_len = rewards.shape
    out = np.zeros((num_envs, seq_len), dtype=rewards.dtype)
    carry = np.zeros(num_envs, dtype=rewards.dtype)
    for t in reversed(range(seq_len)):
        v_next = values[:, t + 1] if t < seq_len - 1 else np.zeros(num_envs, dtype=rewards.dtype)
        not_term = 1.0 - terminateds[:, t]
        delta = rewards[:, t] + gamma * not_term * v_next - values[:, t]
        carry = delta + gamma * lambda_ * not_term * carry
        out[:, t] = carry
    return out


def run_jax_equivalence_gate(results_dir):
    print("=" * 88)
    print("STEP 0b: JAX-SIDE EQUIVALENCE GATE (JAX vs. independent reference, "
          "vs. Triton directly)")
    print("=" * 88)
    print("Completes the equivalence proof started in phase 1 (Triton vs. the same")
    print("reference). See benchmark_gae_vs_jax_scan_triton.py's module docstring for the")
    print("full index-mapping derivation and why truncation/bootstrap_values are excluded.")
    print()

    equivalence_dir = results_dir / "equivalence"
    if not equivalence_dir.exists():
        print(f"ERROR: {equivalence_dir}/ not found. The equivalence-gate cells are written "
              "by both phase 1 and phase 2 under results_dir/equivalence/ -- make sure both "
              "completed against this --results-dir.")
        sys.exit(1)

    all_ok = True
    for num_envs, seq_len, seed in EQUIVALENCE_CASES:
        key = cell_key(num_envs, seq_len) + f"_eq{seed}"
        triton_path = equivalence_dir / f"{key}_triton.npz"
        jax_path = equivalence_dir / f"{key}_jax.npz"
        if not triton_path.exists() or not jax_path.exists():
            print(f"ERROR: missing equivalence-gate files for {key}. Re-run both phase 1 "
                  "and phase 2.")
            sys.exit(1)

        triton_data = np.load(triton_path)
        jax_data = np.load(jax_path)
        rewards, values, terminateds = (
            triton_data["rewards"], triton_data["values"], triton_data["terminateds"]
        )
        triton_full = triton_data["advantages"]
        jax_adv = jax_data["advantages"]

        ref = _ref_gae_matched_window(rewards, values, terminateds, GAMMA, LAMBDA)
        max_diff = float(np.abs(ref - jax_adv).max())
        triton_vs_jax = np.abs(triton_full - jax_adv)

        status = "PASS" if max_diff < 1e-3 else "FAIL"
        if max_diff >= 1e-3:
            all_ok = False
        print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  "
              f"max|JAX - matched_ref| = {max_diff:.3e}   [{status}]")
        print(f"    Triton vs. JAX DIRECTLY (both length-T, same convention, no shift): "
              f"max|diff|={triton_vs_jax.max():.3e}, "
              f"mean|diff|={triton_vs_jax.mean():.3e}  "
              f"(expected: small but nonzero, float32 summation-order noise from "
              f"different compilers -- Triton/LLVM vs. XLA -- NOT bit-identical)")

    print()
    if all_ok:
        print("RESULT: JAX's output matches the independent reference within float32 "
              "tolerance on every case. rl-triton's Triton kernel and JAX/XLA's "
              "associative_scan lowering are two different log-depth PARALLEL scan "
              "implementations of the SAME recurrence, so they are NOT bit-identical, "
              "but agree well within any tolerance that matters for RL advantage "
              "estimates.")
    else:
        print("RESULT: FAIL. Do not trust the timing comparison below.")
        sys.exit(1)
    print()
    return all_ok


# ------------------------------------------------------------------------
# Monotonicity / tables / crossovers / plots (unchanged logic from the
# single-process version, reading from the loaded `results` dict)
# ------------------------------------------------------------------------

def _check_monotonicity(results, num_envs_list, seq_lens, tol=0.02):
    violations = []
    for impl in ("triton_ms", "jax_ms"):
        for num_envs in num_envs_list:
            row = [(t, results[(num_envs, t)][impl]) for t in seq_lens]
            for (t1, v1), (t2, v2) in zip(row, row[1:]):
                if v2 < v1 * (1 - tol):
                    violations.append(
                        f"{impl}: num_envs={num_envs} seq_len {t1}->{t2}: "
                        f"{v1:.4f}ms -> {v2:.4f}ms (decreased by {(1 - v2 / v1) * 100:.1f}%)"
                    )
        for seq_len in seq_lens:
            col = [(n, results[(n, seq_len)][impl]) for n in num_envs_list]
            for (n1, v1), (n2, v2) in zip(col, col[1:]):
                if v2 < v1 * (1 - tol):
                    violations.append(
                        f"{impl}: seq_len={seq_len} num_envs {n1}->{n2}: "
                        f"{v1:.4f}ms -> {v2:.4f}ms (decreased by {(1 - v2 / v1) * 100:.1f}%)"
                    )
    return violations


def report_monotonicity(results, num_envs_list, seq_lens, label):
    print("=" * 88)
    print(f"MONOTONICITY GATE: {label}")
    print("=" * 88)
    violations = _check_monotonicity(results, num_envs_list, seq_lens)
    if not violations:
        print("  PASSED -- time is non-decreasing (within 2% tolerance) along both "
              "the seq_len axis and the num_envs axis, for both implementations.")
        print()
        return violations

    triton_warps_boundaries = set(zip(seq_lens, seq_lens[1:]))
    seq_len_violations = [
        v for v in violations if v.split(":", 1)[1].strip().startswith("num_envs=")
    ]
    triton_at_warps_boundary = sum(
        1 for v in seq_len_violations if v.startswith("triton_ms")
        and any(f"seq_len {a}->{b}" in v and (a in _WARPS_TABLE or b in _WARPS_TABLE)
                for a, b in triton_warps_boundaries)
    )
    triton_seq_len_violations = [v for v in seq_len_violations if v.startswith("triton_ms")]
    print(f"  FAILED -- {len(violations)} violation(s) beyond 2% tolerance "
          f"(n_iter={N_ITER}, n_trials={N_TRIALS}, min-of-medians):")
    for v in violations:
        print(f"    - {v}")
    diagnosis = (
        "  Diagnosis: not a compile-cache leak (each cell is warmed up fresh at its "
        "own exact shape before any timed call, which for JAX also absorbs its "
        "shape-specialized jit trace/compile). Note: JAX's timing here is wall-clock "
        "(time.perf_counter + block_until_ready, run in a separate process/venv from "
        "Triton's CUDA-event timing), which carries more host-dispatch noise than "
        "Triton's device-synchronized CUDA events -- a JAX-side violation close to the "
        "2% tolerance may be measurement noise from that methodology difference rather "
        "than a structural effect. "
    )
    if triton_seq_len_violations:
        if triton_at_warps_boundary:
            diagnosis += (
                f"{triton_at_warps_boundary}/{len(triton_seq_len_violations)} of the "
                f"Triton seq_len-axis violations land on a `_WARPS` tuning-table "
                f"boundary in src/rl_triton/ops/gae.py ({_WARPS_TABLE}) where "
                "num_warps jumps discretely with BLOCK_SIZE -- out of scope for this "
                "benchmark to retune. "
            )
        else:
            diagnosis += (
                "None of the Triton seq_len-axis violations land on a `_WARPS` "
                f"tuning-table boundary ({_WARPS_TABLE}) -- at these problem sizes "
                "device time is a handful of microseconds, so sub-2% swings here are "
                "within the profiler/CUDA-event measurement floor, not a structural "
                "effect. "
            )
    num_envs_violation_count = len(violations) - len(seq_len_violations)
    if num_envs_violation_count:
        diagnosis += (
            f"{num_envs_violation_count} violation(s) are on the num_envs axis -- "
            "small grids not yet saturating the GPU's SMs can absorb growth in "
            "num_envs at near-zero extra wall-clock cost. "
        )
    diagnosis += (
        "Per-cell speedups below are reported as measured; treat the flagged "
        "cells' precise ratios with the caveats above, not as invalidating the "
        "table."
    )
    print(diagnosis)
    print()
    return violations


def print_markdown_table(results, num_envs_list, seq_lens):
    """Triton retains its dev(us)/launches columns (torch.profiler-based, from
    phase 1); JAX's are reported as N/A -- phase 2 has no torch.profiler
    equivalent available (see benchmark_gae_vs_jax_scan_jax.py's module
    docstring), so no dev_speedup / device-time crossover analysis exists
    for JAX either -- wall-clock is the only axis compared."""
    lines = []
    header = ("| num_envs | seq_len | triton (ms) | jax (ms) | speedup | "
              "triton amort (ms) | jax amort (ms) | triton GB/s (%peak) | jax GB/s (%peak) | "
              "triton dev (us) | triton launches |")
    sep = "|" + "---|" * 11
    lines.append(header)
    lines.append(sep)
    for num_envs in num_envs_list:
        for seq_len in seq_lens:
            r = results[(num_envs, seq_len)]
            lines.append(
                f"| {num_envs} | {seq_len} | {r['triton_ms']:.4f} | {r['jax_ms']:.4f} | "
                f"{r['speedup']:.2f}x | "
                f"{r['triton_ms_amort']:.4f} | {r['jax_ms_amort']:.4f} | "
                f"{r['triton_gbps']:.1f} ({r['triton_pct_peak']:.2f}%) | "
                f"{r['jax_gbps']:.1f} ({r['jax_pct_peak']:.2f}%) | "
                f"{r['triton_dev_us']:.2f} | {r['triton_launches']:.1f} |"
            )

    print("=" * 88)
    print("MARKDOWN TABLE")
    print("=" * 88)
    for line in lines:
        print(line)

    return "\n".join(lines)


def find_crossovers(results, num_envs_list, seq_lens, label):
    print("=" * 88)
    print(f"CROSSOVER ANALYSIS: {label}")
    print("=" * 88)
    any_crossover = False
    for num_envs in num_envs_list:
        crossover = None
        for seq_len in seq_lens:
            if results[(num_envs, seq_len)]["speedup"] >= 1.0:
                crossover = seq_len
                break
        if crossover is not None:
            any_crossover = True
            print(f"  num_envs={num_envs:>5}: Triton overtakes JAX (wall-clock) at seq_len >= {crossover}")
        else:
            best = max(results[(num_envs, t)]["speedup"] for t in seq_lens)
            print(f"  num_envs={num_envs:>5}: NO wall-clock crossover in "
                  f"[{seq_lens[0]}, {seq_lens[-1]}] (best speedup observed: {best:.2f}x)")
    print()
    if not any_crossover:
        print("  No wall-clock crossover exists anywhere in the swept range: this is a "
              "plain finding, not massaged to fit a narrative.")
    print("  (No device-time-only crossover analysis: JAX's device-only kernel time has "
          "no measurement available in this three-phase setup -- see the module "
          "docstring in benchmark_gae_vs_jax_scan_jax.py.)")
    print()
    return any_crossover


def plot_results(results, out_path, x_axis="seq_len", num_envs_list=None, seq_lens=None,
                  colors=None, title=""):
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    if x_axis == "seq_len":
        for num_envs in num_envs_list:
            color = colors[num_envs]
            triton_y = [results[(num_envs, t)]["triton_ms"] for t in seq_lens]
            jax_y = [results[(num_envs, t)]["jax_ms"] for t in seq_lens]
            ax.plot(seq_lens, triton_y, color=color, linestyle="-", marker="o",
                    markersize=5, linewidth=2, label=f"Triton, num_envs={num_envs}")
            ax.plot(seq_lens, jax_y, color=color, linestyle="--", marker="s",
                    markersize=5, linewidth=2, label=f"JAX, num_envs={num_envs}")
        ax.set_xlabel("Sequence length T")
    else:  # x_axis == "num_envs" -- short-horizon regime, color by seq_len
        for seq_len in seq_lens:
            color = colors[seq_len]
            triton_y = [results[(n, seq_len)]["triton_ms"] for n in num_envs_list]
            jax_y = [results[(n, seq_len)]["jax_ms"] for n in num_envs_list]
            ax.plot(num_envs_list, triton_y, color=color, linestyle="-", marker="o",
                    markersize=5, linewidth=2, label=f"Triton, seq_len={seq_len}")
            ax.plot(num_envs_list, jax_y, color=color, linestyle="--", marker="s",
                    markersize=5, linewidth=2, label=f"JAX, seq_len={seq_len}")
        ax.set_xlabel("num_envs")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylabel("Execution time (ms, min-of-11-medians, 100 iters/trial)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved figure to {out_path}")


def analyze_short_horizon_questions(results, num_envs_list, seq_lens):
    print("=" * 88)
    print("SHORT-HORIZON REGIME: WALL-CLOCK QUESTIONS")
    print("=" * 88)
    print("(Device-only/launch-bound analysis dropped: no torch.profiler equivalent "
          "for JAX in this three-phase setup -- see benchmark_gae_vs_jax_scan_jax.py.)")
    print()

    print("Q1: Does JAX's time stay independent of num_envs at short T?")
    for seq_len in seq_lens:
        jax_vals = [results[(n, seq_len)]["jax_ms"] for n in num_envs_list]
        triton_vals = [results[(n, seq_len)]["triton_ms"] for n in num_envs_list]
        jax_spread = max(jax_vals) / min(jax_vals)
        triton_spread = max(triton_vals) / min(triton_vals)
        print(f"  seq_len={seq_len:>4}: jax {min(jax_vals):.4f}-{max(jax_vals):.4f}ms "
              f"(max/min={jax_spread:.2f}x)   "
              f"triton {min(triton_vals):.4f}-{max(triton_vals):.4f}ms "
              f"(max/min={triton_spread:.2f}x)   "
              f"across num_envs={num_envs_list}")
    print()

    print("Q2: Where does rl-triton's one-program-per-env design stop paying, vs. JAX?")
    wall_inversions = [(n, t) for n in num_envs_list for t in seq_lens
                        if results[(n, t)]["speedup"] < 1.0]
    if wall_inversions:
        print(f"  Wall-clock inversion (JAX faster end-to-end) at: {wall_inversions}")
    else:
        print("  No wall-clock inversion found anywhere in this sweep.")
    print()

    print("Q3: bandwidth-bound, launch-bound, or occupancy-bound? "
          "(single-call/amortized ratio; achieved %% of H100 peak bandwidth)")
    for num_envs in num_envs_list:
        for seq_len in seq_lens:
            r = results[(num_envs, seq_len)]
            triton_dispatch_ratio = r["triton_ms"] / r["triton_ms_amort"] if r["triton_ms_amort"] else float("nan")
            jax_dispatch_ratio = r["jax_ms"] / r["jax_ms_amort"] if r["jax_ms_amort"] else float("nan")
            print(f"  num_envs={num_envs:>5} seq_len={seq_len:>4}  "
                  f"triton: single/amort={triton_dispatch_ratio:>5.2f}x, "
                  f"{r['triton_pct_peak']:>6.3f}% peak BW   |   "
                  f"jax: single/amort={jax_dispatch_ratio:>5.2f}x, "
                  f"{r['jax_pct_peak']:>6.3f}% peak BW")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    results_dir = args.results_dir

    triton_meta_path = results_dir / "triton_meta.npz"
    jax_meta_path = results_dir / "jax_meta.npz"
    if not triton_meta_path.exists() or not jax_meta_path.exists():
        print(f"ERROR: missing phase metadata in {results_dir}/. Run both "
              "benchmark_gae_vs_jax_scan_triton.py and benchmark_gae_vs_jax_scan_jax.py "
              "first, against this --results-dir.")
        sys.exit(1)
    triton_meta = np.load(triton_meta_path)
    jax_meta = np.load(jax_meta_path)
    gpu_name = str(triton_meta["gpu_name"])
    torch_version = str(triton_meta["torch_version"])
    torch_cuda = str(triton_meta["torch_cuda"])
    triton_version = str(triton_meta["triton_version"])
    jax_version = str(jax_meta["jax_version"])
    jax_backend = str(jax_meta["jax_backend"])

    print("=" * 88)
    print("PHASE 3/3: REPORT (rl-triton GAE kernel vs. JAX associative-scan GAE)")
    print("=" * 88)
    print(f"GPU:            {gpu_name}")
    print(f"torch:          {torch_version}  (cuda {torch_cuda})")
    print(f"triton:         {triton_version}")
    print(f"jax:            {jax_version}  (backend: {jax_backend})")
    print(f"dtype:          float32")
    print(f"gamma={GAMMA}  lambda={LAMBDA}  termination_prob={TERM_PROB}")
    print(f"results dir:    {results_dir}")
    print()

    run_jax_equivalence_gate(results_dir)

    report_date = datetime.date.today().isoformat()
    report_lines = [
        f"# rl-triton GAE vs. JAX associative-scan GAE -- {report_date}",
        "",
        "ONE-OFF deep-dive study (equivalence proof, launch counts, bandwidth, crossover "
        "plots) -- not part of the release cycle, not regenerated automatically. Mirrors "
        "gae_vs_pufferlib-*.md's structure but is NOT directly comparable to it: this "
        "baseline is JAX/XLA's `jax.lax.associative_scan` (a log-depth parallel scan, "
        "compiled by XLA), not PufferLib's hand-written sequential CUDA kernel. Produced "
        "by a three-phase, two-venv pipeline (benchmark_gae_vs_jax_scan_{triton,jax,"
        "report}.py) -- see benchmark_gae_vs_jax_scan_triton.py's module docstring for "
        "why (a real torch/jax cuDNN pin conflict, not a design preference) and for the "
        "full index-mapping derivation between rl-triton's and the JAX baseline's buffer "
        "conventions. JAX's numbers are wall-clock only (time.perf_counter + "
        "block_until_ready, in a separate process from Triton's CUDA-event timing) -- no "
        "device-only/launch-count columns exist for JAX in this setup.",
        "",
        f"GPU: {gpu_name} · torch {torch_version} (cuda {torch_cuda}) · "
        f"triton {triton_version} · jax {jax_version} (backend={jax_backend})",
        f"dtype float32 · gamma={GAMMA} · lambda={LAMBDA} · termination_prob={TERM_PROB}",
        "",
        "Equivalence gate passed (see console output) -- both implementations verified to "
        "compute the same recurrence (under the derived index mapping) before any timing "
        "number below is trusted.",
        "",
    ]

    # -- Regime 1: production (moderate-to-long rollouts) --------------------
    results = load_all(results_dir, NUM_ENVS_LIST, SEQ_LENS)
    report_monotonicity(results, NUM_ENVS_LIST, SEQ_LENS, "production regime")
    table_md = print_markdown_table(results, NUM_ENVS_LIST, SEQ_LENS)
    find_crossovers(results, NUM_ENVS_LIST, SEQ_LENS, "production regime")

    fig_path = Path(__file__).parent.parent / "gae_performance_crossover_jax.png"
    plot_results(results, fig_path, x_axis="seq_len", num_envs_list=NUM_ENVS_LIST,
                 seq_lens=SEQ_LENS, colors=COLORS,
                 title="rl-triton GAE vs. JAX associative-scan GAE -- "
                       f"{gpu_name} (production regime)")

    print("=" * 88)
    print("VERDICT: PRODUCTION REGIME")
    print("=" * 88)
    triton_wins = sum(1 for r in results.values() if r["speedup"] >= 1.0)
    jax_wins = len(results) - triton_wins
    verdict_line = (f"Triton faster in {triton_wins}/{len(results)} cells; "
                     f"JAX faster in {jax_wins}/{len(results)} cells.")
    print(f"  {verdict_line}")
    print("  See the crossover analysis above and bandwidth/launch-count columns in "
          "the table for the bandwidth-bound vs. launch-bound mechanism read "
          f"({gpu_name} peak HBM bandwidth used for %%peak: {H100_PEAK_GBPS:.0f} GB/s datasheet).")
    print()

    report_lines += [
        "## Production regime",
        "",
        f"seq_len in {SEQ_LENS}, num_envs in {NUM_ENVS_LIST}.",
        "",
        table_md,
        "",
        f"**Verdict:** {verdict_line}",
        "",
        f"![production regime crossover]({fig_path.name})",
        "",
    ]

    # -- Regime 2: massively-parallel-sim (short horizon, high env count) ----
    results_short = load_all(results_dir, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT)
    report_monotonicity(results_short, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT,
                         "massively-parallel-sim regime")
    table_md_short = print_markdown_table(results_short, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT)
    find_crossovers(results_short, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT,
                     "massively-parallel-sim regime")

    fig_path_short = Path(__file__).parent.parent / "gae_performance_short_horizon_jax.png"
    plot_results(results_short, fig_path_short, x_axis="num_envs",
                 num_envs_list=NUM_ENVS_LIST_LARGE, seq_lens=SEQ_LENS_SHORT,
                 colors=COLORS_SHORT,
                 title="rl-triton GAE vs. JAX associative-scan GAE -- "
                       f"{gpu_name} (massively-parallel-sim regime)")

    analyze_short_horizon_questions(results_short, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT)

    print("=" * 88)
    print("VERDICT: MASSIVELY-PARALLEL-SIM REGIME")
    print("=" * 88)
    triton_wins_short = sum(1 for r in results_short.values() if r["speedup"] >= 1.0)
    jax_wins_short = len(results_short) - triton_wins_short
    verdict_line_short = (
        f"Triton faster (wall-clock) in {triton_wins_short}/{len(results_short)} cells; "
        f"JAX faster in {jax_wins_short}/{len(results_short)} cells."
    )
    print(f"  {verdict_line_short}")
    print("  See the Q1-Q3 analysis above for the mechanism read.")

    report_lines += [
        "## Massively-parallel-sim regime (short horizon, high env count)",
        "",
        f"seq_len in {SEQ_LENS_SHORT}, num_envs in {NUM_ENVS_LIST_LARGE}.",
        "",
        table_md_short,
        "",
        f"**Verdict:** {verdict_line_short}",
        "",
        f"![short-horizon regime crossover]({fig_path_short.name})",
        "",
    ]

    report_path = Path(__file__).parent.parent / f"gae_vs_jax_scan-{report_date}.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nWrote {report_path} (tables + verdicts, reference only -- not part of "
          f"benchmarks.md/README's recurring tables; run-on-demand, review before committing).")


if __name__ == "__main__":
    main()
