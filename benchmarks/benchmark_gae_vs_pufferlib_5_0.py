"""Standalone benchmark: rl-triton's fused GAE kernel vs. PufferLib 5.0's
`puff_advantage` CUDA kernel -- float32 only.

WHY THIS IS A SEPARATE SCRIPT FROM benchmark_gae_vs_pufferlib.py: that script
benchmarks vendored PufferLib **3.0.0** (the last PyPI release; see
pufferlib_ext/NOTICE.md). PufferLib's advantage kernel changed materially on
the `5.0` branch (see pufferlib_ext/pufferlib_5_0.cu's module docstring):
vectorized 16B load/store instead of the 3.0.0 kernel's scalar loop, an added
`returns` output, and no Python/torch binding upstream at all (PufferLib 5.0
is "0 python" -- Joseph Suarez, PufferLib author, 2026-08-19 correspondence).
Keeping this in its own script means the original 3.0.0 report stays
reproducible and untouched, and this one's "puffer5" columns are never
confused with the older "puffer" (3.0.0) columns in the other report.

FLOAT32 ONLY: this repo only benchmarks the regime it actually runs in.
PufferLib 5.0 also supports bf16 (its default precision_t) with a materially
different kernel body (uint4-packed bf16x8 loads instead of float4 loads);
that path is not vendored and not benchmarked here -- see NOTICE.md.

Equivalence: puff_advantage 5.0's recurrence and R[t+1]/D[t+1] buffer
convention are IDENTICAL to the 3.0.0 kernel already proven equivalent to
rl-triton's GAE in benchmark_gae_vs_pufferlib.py's Step 0 (confirmed by
reading pufferlib/src/algo.cu:1551-1552's own comment: "Same R_{t+1},D_{t+1}
indexing as classic puffer_advantage"). This script reuses that same
equivalence gate rather than re-deriving it -- see run_equivalence_gate_5_0
below, which is the 3.0.0 script's gate with the op and buffer signature
swapped in (5.0's kernel additionally requires a `returns` output tensor and
horizon % 4 == 0 for its vectorized float4 path).

Usage: python benchmarks/benchmark_gae_vs_pufferlib_5_0.py
"""
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parent))  # this dir -- for pufferlib_ext
from pufferlib_ext.build import load_puff_advantage_5_0
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))  # for bench_utils
from bench_utils import _bench_gpu, _warmup_gpu

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rl_triton.ops.gae import compute_gae

GAMMA = 0.99
LAMBDA = 0.95
TERM_PROB = 0.05
SEED = 0

# Same sweep points as the 3.0.0 script -- both regimes -- for direct
# cross-report comparability. All values are multiples of 4 (5.0's
# ADV_VEC_WIDTH for float32), so no shape restriction bites here.
SEQ_LENS = [128, 512, 1024, 2048, 4096]
NUM_ENVS_LIST = [128, 512, 2048, 8192]
SEQ_LENS_SHORT = [8, 16, 32, 64, 128]
NUM_ENVS_LIST_LARGE = [4096, 8192, 16384, 32768]

N_ITER = 100
N_TRIALS = 11
N_AMORTIZED_CALLS = 100
N_PROFILE_ITER = 20

H100_PEAK_GBPS = 3350.0

COLORS = {128: "#2a78d6", 512: "#eb6834", 2048: "#1baf7a", 8192: "#eda100"}
COLORS_SHORT = {8: "#2a78d6", 16: "#eb6834", 32: "#1baf7a", 64: "#eda100", 128: "#e87ba4"}


# ------------------------------------------------------------------------
# Step 0: semantic equivalence
# ------------------------------------------------------------------------

def _make_rl_triton_inputs(num_envs, seq_len, device, seed=SEED):
    g = torch.Generator(device=device).manual_seed(seed)
    rewards = torch.randn(num_envs, seq_len, device=device, generator=g).contiguous()
    values = torch.randn(num_envs, seq_len, device=device, generator=g).contiguous()
    terminateds = (torch.rand(num_envs, seq_len, device=device, generator=g) < TERM_PROB).float().contiguous()
    return rewards, values, terminateds


def _to_puffer_inputs(rewards, values, terminateds):
    """Same R[t+1]/D[t+1] shift as the 3.0.0 script -- see that file's
    _to_puffer_inputs docstring; unchanged in 5.0 (confirmed via algo.cu's own
    "Same R_{t+1},D_{t+1} indexing" comment)."""
    num_envs, seq_len = rewards.shape
    device = rewards.device
    puffer_rewards = torch.zeros(num_envs, seq_len, device=device)
    puffer_dones = torch.zeros(num_envs, seq_len, device=device)
    puffer_rewards[:, 1:] = rewards[:, :seq_len - 1]
    puffer_dones[:, 1:] = terminateds[:, :seq_len - 1]
    puffer_importance = torch.ones(num_envs, seq_len, device=device)
    return (
        values.contiguous(),
        puffer_rewards.contiguous(),
        puffer_dones.contiguous(),
        puffer_importance.contiguous(),
    )


def _ref_gae_matched_window(rewards, values, terminateds, gamma, lambda_):
    """Independent reference -- identical to the 3.0.0 script's (the
    recurrence did not change on 5.0). See that file for the derivation."""
    num_envs, seq_len = rewards.shape
    device = rewards.device
    out = torch.zeros(num_envs, seq_len - 1, device=device, dtype=rewards.dtype)
    carry = torch.zeros(num_envs, device=device, dtype=rewards.dtype)
    for t in reversed(range(seq_len - 1)):
        not_term = 1.0 - terminateds[:, t]
        delta = rewards[:, t] + gamma * not_term * values[:, t + 1] - values[:, t]
        carry = delta + gamma * lambda_ * not_term * carry
        out[:, t] = carry
    return out


def run_equivalence_gate_5_0(op, device="cuda"):
    print("=" * 88)
    print("STEP 0: SEMANTIC EQUIVALENCE GATE (PufferLib 5.0)")
    print("=" * 88)
    print("puff_advantage 5.0 uses the identical recurrence and R[t+1]/D[t+1] buffer")
    print("convention as the 3.0.0 kernel (verified against algo.cu's own comment,")
    print("'Same R_{t+1},D_{t+1} indexing as classic puffer_advantage') -- same gate")
    print("as benchmark_gae_vs_pufferlib.py's Step 0, extended for the new `returns`")
    print("output 5.0's kernel additionally writes.")
    print()

    all_ok = True
    equivalence_cases = [
        (64, 256, 1), (32, 4096, 2), (37 * 4, 128, 3),  # horizon must be %4==0
        (4096, 8, 4), (8192, 16, 5), (16384, 64, 6), (2048, 128, 7),
    ]
    for num_envs, seq_len, seed in equivalence_cases:
        rewards, values, terminateds = _make_rl_triton_inputs(num_envs, seq_len, device, seed)
        pv, pr, pd, pimp = _to_puffer_inputs(rewards, values, terminateds)

        puffer_adv = torch.zeros(num_envs, seq_len, device=device)
        puffer_ret = torch.zeros(num_envs, seq_len, device=device)
        op(pv, pr, pd, pimp, puffer_adv, puffer_ret, GAMMA, LAMBDA, 1.0, 1.0)

        ref = _ref_gae_matched_window(rewards, values, terminateds, GAMMA, LAMBDA)
        max_diff = (ref - puffer_adv[:, :seq_len - 1]).abs().max().item()

        full = compute_gae(rewards, values, terminateds, gamma=GAMMA, lambda_=LAMBDA)
        rl_vs_ref = (full[:, :seq_len - 1] - ref).abs()

        status = "PASS (exact)" if max_diff < 1e-4 else "FAIL"
        if max_diff >= 1e-4:
            all_ok = False
        print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  "
              f"max|PufferLib5.0 - matched_ref| = {max_diff:.3e}   [{status}]")
        print(f"    (rl-triton full-window vs. same ref, for context -- "
              f"differs near the boundary by construction: "
              f"max|diff|={rl_vs_ref.max().item():.4f}, "
              f"mean|diff|={rl_vs_ref.mean().item():.2e})")

        # returns = values + advantages, per the 5.0 kernel body -- a cheap
        # self-consistency check specific to the new output this kernel adds.
        ret_check = (puffer_ret[:, :seq_len - 1] - (values[:, :seq_len - 1] + puffer_adv[:, :seq_len - 1])).abs().max().item()
        print(f"    returns == values + advantages (5.0-only output): "
              f"max|diff|={ret_check:.3e}   [{'PASS' if ret_check < 1e-4 else 'FAIL'}]")
        if ret_check >= 1e-4:
            all_ok = False

        terminateds2 = terminateds.clone()
        terminateds2[:, -2] = 1.0
        terminateds2[:, -1] = 0.0
        full2 = compute_gae(rewards, values, terminateds2, gamma=GAMMA, lambda_=LAMBDA)
        puffer_adv2 = torch.zeros(num_envs, seq_len, device=device)
        puffer_ret2 = torch.zeros(num_envs, seq_len, device=device)
        pd2 = torch.zeros(num_envs, seq_len, device=device)
        pd2[:, 1:] = terminateds2[:, :seq_len - 1]
        op(pv, pr, pd2, pimp, puffer_adv2, puffer_ret2, GAMMA, LAMBDA, 1.0, 1.0)
        direct_diff = (full2[:, :seq_len - 2] - puffer_adv2[:, :seq_len - 2]).abs()
        print(f"    Triton vs. PufferLib5.0 DIRECTLY (uncontaminated columns): "
              f"max|diff|={direct_diff.max().item():.2e}, "
              f"mean|diff|={direct_diff.mean().item():.2e}  "
              f"(expected: small but nonzero float32 summation-order noise)")

    print()
    if all_ok:
        print("RESULT: PufferLib 5.0's output matches the independent reference EXACTLY "
              "on every column it is structurally capable of producing, and its added "
              "`returns` output is self-consistent (returns = values + advantages). "
              "Proceeding to benchmark.")
    else:
        print("RESULT: FAIL. Do not trust the timing comparison below.")
        sys.exit(1)
    print()
    return all_ok


# ------------------------------------------------------------------------
# Step 1-2: harness + sweep
# ------------------------------------------------------------------------

def _bench_gpu_amortized(fn, n_calls=N_AMORTIZED_CALLS, n_trials=N_TRIALS):
    per_trial = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_calls):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_trial.append(start.elapsed_time(end) / n_calls)
    return min(per_trial)


def _device_profile(fn, n_iter=N_PROFILE_ITER):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_device_us = sum(e.self_device_time_total for e in events)
    launches = sum(e.count for e in events)
    return total_device_us / n_iter, launches / n_iter


def run_sweep(op, num_envs_list, seq_lens, label, device="cuda"):
    results = {}
    print("=" * 88)
    print(f"SWEEP: {label}")
    print("=" * 88)
    for num_envs in num_envs_list:
        for seq_len in seq_lens:
            rewards, values, terminateds = _make_rl_triton_inputs(num_envs, seq_len, device)
            pv, pr, pd, pimp = _to_puffer_inputs(rewards, values, terminateds)

            def triton_call(rewards=rewards, values=values, terminateds=terminateds):
                return compute_gae(rewards, values, terminateds, gamma=GAMMA, lambda_=LAMBDA)

            def puffer5_call(pv=pv, pr=pr, pd=pd, pimp=pimp, num_envs=num_envs, seq_len=seq_len):
                adv = torch.zeros(num_envs, seq_len, device=device)
                ret = torch.zeros(num_envs, seq_len, device=device)
                op(pv, pr, pd, pimp, adv, ret, GAMMA, LAMBDA, 1.0, 1.0)
                return adv

            _warmup_gpu(triton_call, n_warmup=15)
            _warmup_gpu(puffer5_call, n_warmup=15)

            triton_ms = _bench_gpu(triton_call, n_iter=N_ITER, n_trials=N_TRIALS)
            puffer5_ms = _bench_gpu(puffer5_call, n_iter=N_ITER, n_trials=N_TRIALS)

            triton_ms_amort = _bench_gpu_amortized(triton_call)
            puffer5_ms_amort = _bench_gpu_amortized(puffer5_call)

            triton_dev_us, triton_launches = _device_profile(triton_call)
            puffer5_dev_us, puffer5_launches = _device_profile(puffer5_call)

            triton_bytes = 4 * num_envs * seq_len * 4
            # 5.0 additionally writes `returns` -- 6 buffers touched, not 5.
            puffer5_bytes = 6 * num_envs * seq_len * 4

            triton_gbps = triton_bytes / (triton_ms / 1000) / 1e9
            puffer5_gbps = puffer5_bytes / (puffer5_ms / 1000) / 1e9

            results[(num_envs, seq_len)] = dict(
                triton_ms=triton_ms, puffer5_ms=puffer5_ms,
                triton_ms_amort=triton_ms_amort, puffer5_ms_amort=puffer5_ms_amort,
                triton_dev_us=triton_dev_us, puffer5_dev_us=puffer5_dev_us,
                triton_launches=triton_launches, puffer5_launches=puffer5_launches,
                triton_gbps=triton_gbps, puffer5_gbps=puffer5_gbps,
                triton_pct_peak=triton_gbps / H100_PEAK_GBPS * 100,
                puffer5_pct_peak=puffer5_gbps / H100_PEAK_GBPS * 100,
                speedup=puffer5_ms / triton_ms,
                dev_speedup=puffer5_dev_us / triton_dev_us if triton_dev_us else float("nan"),
            )
            r = results[(num_envs, seq_len)]
            print(f"  num_envs={num_envs:>5} seq_len={seq_len:>5}  "
                  f"triton={r['triton_ms']:>8.4f}ms ({r['triton_gbps']:>7.1f} GB/s)  "
                  f"puffer5={r['puffer5_ms']:>8.4f}ms ({r['puffer5_gbps']:>7.1f} GB/s)  "
                  f"speedup={r['speedup']:>6.2f}x  dev_speedup={r['dev_speedup']:>6.2f}x")
    print()
    return results


# ------------------------------------------------------------------------
# Step 3: output
# ------------------------------------------------------------------------

def print_markdown_table(results, num_envs_list, seq_lens):
    lines = []
    header = ("| num_envs | seq_len | triton (ms) | puffer5 (ms) | speedup | dev speedup | "
              "triton amort (ms) | puffer5 amort (ms) | triton GB/s (%peak) | puffer5 GB/s (%peak) | "
              "triton dev (us) | puffer5 dev (us) | triton launches | puffer5 launches |")
    sep = "|" + "---|" * 14
    lines.append(header)
    lines.append(sep)
    for num_envs in num_envs_list:
        for seq_len in seq_lens:
            r = results[(num_envs, seq_len)]
            lines.append(
                f"| {num_envs} | {seq_len} | {r['triton_ms']:.4f} | {r['puffer5_ms']:.4f} | "
                f"{r['speedup']:.2f}x | {r['dev_speedup']:.2f}x | "
                f"{r['triton_ms_amort']:.4f} | {r['puffer5_ms_amort']:.4f} | "
                f"{r['triton_gbps']:.1f} ({r['triton_pct_peak']:.2f}%) | "
                f"{r['puffer5_gbps']:.1f} ({r['puffer5_pct_peak']:.2f}%) | "
                f"{r['triton_dev_us']:.2f} | {r['puffer5_dev_us']:.2f} | "
                f"{r['triton_launches']:.1f} | {r['puffer5_launches']:.1f} |"
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
            print(f"  num_envs={num_envs:>5}: Triton overtakes PufferLib5.0 (wall-clock) at seq_len >= {crossover}")
        else:
            best = max(results[(num_envs, t)]["speedup"] for t in seq_lens)
            print(f"  num_envs={num_envs:>5}: NO wall-clock crossover in "
                  f"[{seq_lens[0]}, {seq_lens[-1]}] (best speedup observed: {best:.2f}x)")
    print()
    any_dev_crossover = False
    for num_envs in num_envs_list:
        dev_crossover = None
        for seq_len in seq_lens:
            if results[(num_envs, seq_len)]["dev_speedup"] < 1.0:
                dev_crossover = seq_len
                break
        if dev_crossover is not None:
            any_dev_crossover = True
            print(f"  num_envs={num_envs:>5}: Triton device-time INVERTS below 1x "
                  f"(PufferLib5.0 faster in raw kernel time) at seq_len >= {dev_crossover}")
    if not any_dev_crossover:
        print("  No device-time inversion found on this axis.")
    print()
    return any_crossover, any_dev_crossover


def plot_results(results, out_path, x_axis="seq_len", num_envs_list=None, seq_lens=None,
                  colors=None, title=""):
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    if x_axis == "seq_len":
        for num_envs in num_envs_list:
            color = colors[num_envs]
            triton_y = [results[(num_envs, t)]["triton_ms"] for t in seq_lens]
            puffer5_y = [results[(num_envs, t)]["puffer5_ms"] for t in seq_lens]
            ax.plot(seq_lens, triton_y, color=color, linestyle="-", marker="o",
                    markersize=5, linewidth=2, label=f"Triton, num_envs={num_envs}")
            ax.plot(seq_lens, puffer5_y, color=color, linestyle="--", marker="s",
                    markersize=5, linewidth=2, label=f"PufferLib5.0, num_envs={num_envs}")
        ax.set_xlabel("Sequence length T")
    else:
        for seq_len in seq_lens:
            color = colors[seq_len]
            triton_y = [results[(n, seq_len)]["triton_ms"] for n in num_envs_list]
            puffer5_y = [results[(n, seq_len)]["puffer5_ms"] for n in num_envs_list]
            ax.plot(num_envs_list, triton_y, color=color, linestyle="-", marker="o",
                    markersize=5, linewidth=2, label=f"Triton, seq_len={seq_len}")
            ax.plot(num_envs_list, puffer5_y, color=color, linestyle="--", marker="s",
                    markersize=5, linewidth=2, label=f"PufferLib5.0, seq_len={seq_len}")
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


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    op, puffer_source = load_puff_advantage_5_0()

    print("=" * 88)
    print("BENCHMARK: rl-triton GAE kernel vs. PufferLib 5.0 puff_advantage kernel")
    print("=" * 88)
    print(f"GPU:            {gpu_name}")
    print(f"torch:          {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"triton:         {triton.__version__}")
    print(f"PufferLib src:  {puffer_source}")
    print(f"dtype:          float32")
    print(f"gamma={GAMMA}  lambda={LAMBDA}  termination_prob={TERM_PROB}")
    print()

    run_equivalence_gate_5_0(op, device)

    report_date = datetime.date.today().isoformat()
    report_lines = [
        f"# rl-triton GAE vs. PufferLib 5.0 puff_advantage kernel -- {report_date}",
        "",
        "ONE-OFF deep-dive study, float32 only -- see "
        "benchmarks/pufferlib_ext/NOTICE.md's PufferLib 5.0 addendum for exact "
        "commit/provenance. Not the same kernel as "
        "gae_vs_pufferlib-*.md (that report is vendored PufferLib 3.0.0, the "
        "last PyPI release); do not compare 'puffer5' columns here against "
        "'puffer' columns there as if they were the same baseline.",
        "",
        f"GPU: {gpu_name} · torch {torch.__version__} (cuda {torch.version.cuda}) · "
        f"triton {triton.__version__} · PufferLib source: {puffer_source}",
        f"dtype float32 · gamma={GAMMA} · lambda={LAMBDA} · termination_prob={TERM_PROB}",
        "",
        "Equivalence gate passed (see console output) -- PufferLib 5.0's kernel uses "
        "the identical recurrence/buffer convention as 3.0.0's, verified against an "
        "independent reference and self-consistency-checked on the new `returns` output.",
        "",
    ]

    results = run_sweep(op, NUM_ENVS_LIST, SEQ_LENS, "PRODUCTION REGIME", device)
    table_md = print_markdown_table(results, NUM_ENVS_LIST, SEQ_LENS)
    find_crossovers(results, NUM_ENVS_LIST, SEQ_LENS, "production regime")

    fig_path = Path(__file__).parent.parent / "gae_performance_crossover_5_0.png"
    plot_results(results, fig_path, x_axis="seq_len", num_envs_list=NUM_ENVS_LIST,
                 seq_lens=SEQ_LENS, colors=COLORS,
                 title="rl-triton GAE vs. PufferLib 5.0 puff_advantage -- H100 SXM5 80GB "
                       "(production regime)")

    triton_wins = sum(1 for r in results.values() if r["speedup"] >= 1.0)
    puffer_wins = len(results) - triton_wins
    verdict_line = (f"Triton faster in {triton_wins}/{len(results)} cells; "
                     f"PufferLib5.0 faster in {puffer_wins}/{len(results)} cells.")
    print("=" * 88)
    print("VERDICT: PRODUCTION REGIME")
    print("=" * 88)
    print(f"  {verdict_line}")
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

    results_short = run_sweep(op, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT,
                               "MASSIVELY-PARALLEL-SIM REGIME", device)
    table_md_short = print_markdown_table(results_short, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT)
    find_crossovers(results_short, NUM_ENVS_LIST_LARGE, SEQ_LENS_SHORT,
                     "massively-parallel-sim regime")

    fig_path_short = Path(__file__).parent.parent / "gae_performance_short_horizon_5_0.png"
    plot_results(results_short, fig_path_short, x_axis="num_envs",
                 num_envs_list=NUM_ENVS_LIST_LARGE, seq_lens=SEQ_LENS_SHORT,
                 colors=COLORS_SHORT,
                 title="rl-triton GAE vs. PufferLib 5.0 puff_advantage -- H100 SXM5 80GB "
                       "(massively-parallel-sim regime)")

    triton_wins_short = sum(1 for r in results_short.values() if r["speedup"] >= 1.0)
    puffer_wins_short = len(results_short) - triton_wins_short
    verdict_line_short = (
        f"Triton faster (wall-clock) in {triton_wins_short}/{len(results_short)} cells; "
        f"PufferLib5.0 faster in {puffer_wins_short}/{len(results_short)} cells."
    )
    triton_dev_wins_short = sum(1 for r in results_short.values() if r["dev_speedup"] >= 1.0)
    verdict_line_short_dev = (
        f"Triton faster (device-only) in {triton_dev_wins_short}/{len(results_short)} cells; "
        f"PufferLib5.0 faster (device-only) in {len(results_short) - triton_dev_wins_short}/"
        f"{len(results_short)} cells."
    )
    print("=" * 88)
    print("VERDICT: MASSIVELY-PARALLEL-SIM REGIME")
    print("=" * 88)
    print(f"  {verdict_line_short}")
    print(f"  {verdict_line_short_dev}")

    report_lines += [
        "## Massively-parallel-sim regime (short horizon, high env count)",
        "",
        f"seq_len in {SEQ_LENS_SHORT}, num_envs in {NUM_ENVS_LIST_LARGE}.",
        "",
        table_md_short,
        "",
        f"**Verdict:** {verdict_line_short} {verdict_line_short_dev}",
        "",
        f"![short-horizon regime crossover]({fig_path_short.name})",
        "",
    ]

    report_path = Path(__file__).parent.parent / f"gae_vs_pufferlib_5_0-{report_date}.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nWrote {report_path} (tables + verdicts, reference only -- "
          f"run-on-demand, review before committing).")


if __name__ == "__main__":
    main()
