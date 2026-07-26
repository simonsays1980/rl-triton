# Chunked-kernel dispatch vs. torch.compile at long seq_len — a robustness finding

**Scope note: this study is OUTSIDE the project's target regime.** rl-triton's target regime
is production RL rollouts, seq_len ~80-128 (see the production-regime table in
`benchmarks.md`). Everything below runs at seq_len 65,536-524,288 — 500-4000x longer than
that target — specifically to probe the flat/chunked dispatch boundary. Treat this as a
robustness/correctness demonstration at an extreme scale, not a headline performance result.

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-25 · num_envs=16 fixed (matches tests/test_scan_chunked.py's own long-sequence configs).*

**One-off study, not a recurring benchmark.** Not part of bench_release.py, CI, or the test suite. `_FLAT_MAX_SEQ_LEN = 131072` is the shared flat/chunked dispatch threshold for every backward-scan op (GAE, V-Trace, lambda-returns, discounted-returns, and Retrace via its own reroute). Rows at seq_len <= 131072 use the flat kernel; rows above it use the chunked kernel — this is the first time that boundary has been benchmarked against a compiled baseline in this repository (a pre-existing internal-only test, `test_scan_chunked.py::test_scan_performance`, times the same dispatch but excludes any torch.compile comparison entirely).

Eligibility-traces and episodic-prefix-sum are excluded: their forward-scan variant has no chunked kernel and simply rejects seq_len > 131072.

**Finding: no valid vectorized PyTorch baseline exists at this scale.** At seq_len>=65536 the
standard vectorized log-space formulation (`exp(cumsum(log(decay)))`, the same "vec" baseline
used everywhere else in this project) underflows to NaN in float32; rl-triton's chunked
dispatch remains numerically correct throughout (verified against the exact sequential
reference at every row). This is reported as `N/A (baseline invalid)` below, not papered over
with a misleading speedup ratio against broken output.

Root cause, confirmed directly (not assumed): the log-space formulation already clamps
`decay` to a minimum of 1e-38 before taking `log()`, specifically to keep `log(0)` finite at
episode-terminated steps (decay=0 there). But in log-space, each termination inside a row's
suffix therefore contributes `log(1e-38) ≈ -87.5` to the running suffix sum. At this study's
~5% per-step termination rate, only 2-3 terminations — expected within the first couple dozen
steps of *any* window, independent of overall seq_len — already push the log-suffix below
float32's underflow floor (`log(min float32 subnormal) ≈ -103.3`); `exp()` of that then
rounds to exact `0.0`, and dividing by it downstream produces inf/nan. Measured directly on a
65536-length row: the suffix weight underflows to exact `0.0` starting ~17 steps before the
end of the sequence. No epsilon/clamp adjustment fixes this without abandoning the
whole-sequence log-space approach: raising the clamp floor would corrupt the correctness of
the termination boundary itself (the point of decay=0 there), and lowering it only delays,
rather than removes, the same compounding-underflow failure. A structurally different
formulation — chunked/blocked, periodically renormalized, never accumulating one suffix
product across an unbounded run of resets — would avoid this; that is exactly rl-triton's own
kernel strategy at this scale. Building a chunked log-space *PyTorch* baseline to match is a
real implementation option, not attempted here because seq_len>=65536 is outside this
project's target regime (see scope note above).

| algorithm | seq_len | dispatch | triton full-call (ms) | triton device (ms) | compile(vec) full-call (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAE | 65,536 | flat | 0.3174 | 0.2804 | N/A (baseline invalid) | N/A | N/A | N/A |
| GAE | 131,072 | flat | 0.6787 | 0.6339 | N/A (baseline invalid) | N/A | N/A | N/A |
| GAE | 262,144 | chunked | 0.5621 | 0.5177 | N/A (baseline invalid) | N/A | N/A | N/A |
| GAE | 524,288 | chunked | 1.1170 | 1.0755 | N/A (baseline invalid) | N/A | N/A | N/A |
| V-Trace | 65,536 | flat | 0.4384 | 0.3432 | N/A (baseline invalid) | N/A | N/A | N/A |
| V-Trace | 131,072 | flat | 1.0348 | 0.9765 | N/A (baseline invalid) | N/A | N/A | N/A |
| V-Trace | 262,144 | chunked | 0.8552 | 0.7304 | N/A (baseline invalid) | N/A | N/A | N/A |
| V-Trace | 524,288 | chunked | 1.6332 | 1.5663 | N/A (baseline invalid) | N/A | N/A | N/A |
| lambda-returns | 65,536 | flat | 0.3402 | 0.2873 | N/A (baseline invalid) | N/A | N/A | N/A |
| lambda-returns | 131,072 | flat | 0.6184 | 0.5742 | N/A (baseline invalid) | N/A | N/A | N/A |
| lambda-returns | 262,144 | chunked | 0.5083 | 0.4692 | N/A (baseline invalid) | N/A | N/A | N/A |
| lambda-returns | 524,288 | chunked | 1.0279 | 0.9848 | N/A (baseline invalid) | N/A | N/A | N/A |
| discounted-returns | 65,536 | flat | 0.2216 | 0.1563 | N/A (baseline invalid) | N/A | N/A | N/A |
| discounted-returns | 131,072 | flat | 0.6580 | 0.6173 | N/A (baseline invalid) | N/A | N/A | N/A |
| discounted-returns | 262,144 | chunked | 0.4619 | 0.4319 | N/A (baseline invalid) | N/A | N/A | N/A |
| discounted-returns | 524,288 | chunked | 0.9181 | 0.8899 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 65,536 | flat | 0.6046 | 0.3857 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 131,072 | flat | 0.8448 | 0.7084 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 262,144 | chunked | 0.9267 | 0.8659 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 524,288 | chunked | 1.8618 | 1.8048 | N/A (baseline invalid) | N/A | N/A | N/A |
