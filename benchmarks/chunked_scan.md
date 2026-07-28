# Chunked-kernel dispatch vs. torch.compile at long seq_len

**Scope note: this study is OUTSIDE the project's target regime.** rl-triton's target regime is production RL rollouts, seq_len ~80-128 (see the production-regime table in `benchmarks.md`). Everything below runs at seq_len 65,536-524,288 — 500-4000x longer than that target — specifically to probe the flat/chunked dispatch boundary. Treat this as a robustness/correctness demonstration at an extreme scale, not a headline performance result for the project's actual target regime.

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-28 · num_envs=16 fixed (matches tests/test_scan_chunked.py's own long-sequence configs).*

**One-off study, not a recurring benchmark.** Not part of bench_release.py, CI, or the test suite. `_FLAT_MAX_SEQ_LEN = 131072` is the shared flat/chunked dispatch threshold for every backward-scan op (GAE, V-Trace, lambda-returns, discounted-returns, and Retrace via its own reroute). Rows at seq_len <= 131072 use the flat kernel; rows above it use the chunked kernel — this is the first time that boundary has been benchmarked against a compiled baseline in this repository (a pre-existing internal-only test, `test_scan_chunked.py::test_scan_performance`, times the same dispatch but excludes any torch.compile comparison entirely).

Eligibility-traces and episodic-prefix-sum are excluded: their forward-scan variant has no chunked kernel and simply rejects seq_len > 131072.

**Finding: the vectorized PyTorch baseline is valid — and gives a real comparison — for four of the five algorithms below.** GAE, V-Trace, λ-returns, and discounted-returns all use the same log2(T)-doubling associative-scan baseline (`parallel_suffix_scan`, no log-space) that fixed the log-space underflow bug documented in NOTES.md's log-space-underflow note; it stays numerically finite and correct at every seq_len swept here (verified against the exact sequential reference at every row), so the table below reports a genuine `vs vec` speedup ratio for all four.

**A valid comparison is not a flattering one.** At both flat-dispatch lengths (65,536 and 131,072), Triton loses to the compiled PyTorch baseline for every one of these four algorithms (0.35x-0.89x). At the chunked-dispatch lengths, GAE, λ-returns, and discounted-returns flip to a real win (1.13x-1.27x) — but **V-Trace never crosses 1x at any length in this sweep, including chunked** (0.87x, 0.77x). This is consistent with the scope note above: this regime is 500-4000x longer than rl-triton's actual target (seq_len ~80-128), and the flat kernel in particular was never tuned for it. Read this table as "the dispatch boundary is correct and chunking recovers most of the loss for three of four algorithms," not as a speedup claim for this regime.

**Retrace is the exception.** Its baseline here (`_vec_retrace`, `tests/bench_release.py`) still uses the older log-space suffix-cumsum formulation and underflows to inf/nan at every seq_len in this sweep — reported as `N/A (baseline invalid)` below, not papered over with a misleading ratio against broken output. Root cause, confirmed directly (not assumed): that formulation clamps `decay` to a minimum of 1e-38 before taking `log()`, specifically to keep `log(0)` finite at episode-terminated steps (decay=0 there). In log-space, each termination inside a row's suffix therefore contributes `log(1e-38) ~= -87.5` to the running suffix sum. At this study's ~5% per-step termination rate, only 2-3 terminations — expected within the first couple dozen steps of *any* window, independent of overall seq_len — already push the log-suffix below float32's underflow floor (`log(min float32 subnormal) ~= -103.3`); `exp()` of that then rounds to exact `0.0`, and dividing by it downstream produces inf/nan. This is the exact mechanism that once broke GAE/V-Trace/λ-returns/discounted-returns' baselines too; those were fixed by switching to the doubling-scan formulation above, which is why they no longer appear here as N/A. Retrace's baseline in *this script* has not had the same fix applied — a fixed, log-space-free `vectorized_retrace` already exists (`tests/test_retrace.py`, used by `tests/bench_safeguard.py`) and could plausibly be pointed at from here instead of `_vec_retrace`, but that substitution was not made in this pass. A structurally different formulation — chunked/blocked, periodically renormalized, never accumulating one suffix product across an unbounded run of resets — avoids this underflow entirely; that is exactly rl-triton's own kernel strategy at this scale, and exactly what the other four baselines now do too.

| algorithm | seq_len | dispatch | triton full-call (ms) | triton device (ms) | compile(vec) full-call (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAE | 65,536 | flat | 0.3079 | 0.2805 | 0.1901 | 0.1141 | 0.62x | 0.41x |
| GAE | 131,072 | flat | 0.6632 | 0.6324 | 0.3623 | 0.2988 | 0.55x | 0.47x |
| GAE | 262,144 | chunked | 0.5640 | 0.5198 | 0.6474 | 0.5891 | 1.15x | 1.13x |
| GAE | 524,288 | chunked | 1.1096 | 1.0668 | 1.2978 | 1.2361 | 1.17x | 1.16x |
| V-Trace | 65,536 | flat | 0.3855 | 0.3438 | 0.1883 | 0.1119 | 0.49x | 0.33x |
| V-Trace | 131,072 | flat | 1.0097 | 0.9679 | 0.3491 | 0.2838 | 0.35x | 0.29x |
| V-Trace | 262,144 | chunked | 0.7969 | 0.7320 | 0.6903 | 0.6235 | 0.87x | 0.85x |
| V-Trace | 524,288 | chunked | 1.6360 | 1.5690 | 1.2532 | 1.1888 | 0.77x | 0.76x |
| lambda-returns | 65,536 | flat | 0.3071 | 0.2760 | 0.1566 | 0.1011 | 0.51x | 0.37x |
| lambda-returns | 131,072 | flat | 0.6064 | 0.5749 | 0.3471 | 0.2892 | 0.57x | 0.50x |
| lambda-returns | 262,144 | chunked | 0.5064 | 0.4679 | 0.6353 | 0.5828 | 1.25x | 1.25x |
| lambda-returns | 524,288 | chunked | 1.0120 | 0.9760 | 1.2783 | 1.2221 | 1.26x | 1.25x |
| discounted-returns | 65,536 | flat | 0.1820 | 0.1530 | 0.1615 | 0.0997 | 0.89x | 0.65x |
| discounted-returns | 131,072 | flat | 0.6387 | 0.6100 | 0.3187 | 0.2623 | 0.50x | 0.43x |
| discounted-returns | 262,144 | chunked | 0.4653 | 0.4299 | 0.5784 | 0.5283 | 1.24x | 1.23x |
| discounted-returns | 524,288 | chunked | 0.9130 | 0.8818 | 1.1629 | 1.1106 | 1.27x | 1.26x |
| Retrace | 65,536 | flat | 0.5162 | 0.3863 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 131,072 | flat | 0.7610 | 0.7076 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 262,144 | chunked | 0.9264 | 0.8699 | N/A (baseline invalid) | N/A | N/A | N/A |
| Retrace | 524,288 | chunked | 1.8559 | 1.8040 | N/A (baseline invalid) | N/A | N/A | N/A |
