# Chunked-vs-flat dispatch robustness at long seq_len

**Scope note: this study is OUTSIDE the project's target regime.** rl-triton's target regime
is production RL rollouts, seq_len ~80-128 (see the production-regime table in
`benchmarks.md`). Everything below runs at seq_len 65,536-524,288 — 500-4000x longer than
that target — specifically to probe the flat/chunked dispatch boundary. Treat this as a
robustness/correctness demonstration at an extreme scale, **not a speedup comparison against
torch.compile** — no valid PyTorch baseline exists at this scale (see below), so there is
nothing to compare against.

*Measured on NVIDIA H200 · 2026-07-25 · num_envs=16 fixed (matches tests/test_scan_chunked.py's own long-sequence configs).*

**One-off study, not a recurring benchmark.** Not part of bench_release.py, CI, or the test suite. `_FLAT_MAX_SEQ_LEN = 131072` is the shared flat/chunked dispatch threshold for every backward-scan op (GAE, V-Trace, lambda-returns, discounted-returns, and Retrace via its own reroute). Rows at seq_len <= 131072 use the flat kernel; rows above it use the chunked kernel — this is the first time that boundary has been exercised end-to-end and checked for correctness in this repository (a pre-existing internal-only test, `test_scan_chunked.py::test_scan_performance`, times the same dispatch but makes no correctness claim beyond that).

Eligibility-traces and episodic-prefix-sum are excluded: their forward-scan variant has no chunked kernel and simply rejects seq_len > 131072.

**Finding: no valid vectorized PyTorch baseline exists at this scale.** At seq_len>=65536 the
standard vectorized log-space formulation (`exp(cumsum(log(decay)))`, the same "vec" baseline
used everywhere else in this project) underflows to NaN in float32; rl-triton's chunked
dispatch remains numerically correct throughout (verified against the exact sequential
reference at every row). There is consequently no PyTorch number to report a ratio against —
the table below reports rl-triton's own triton/device timings only, split by which dispatch
path (flat vs. chunked) each row uses.

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
rather than removes, the same compounding-underflow failure.

**The honest scope of this finding, stated explicitly:** a structurally different
formulation — chunked/blocked, periodically renormalized, never accumulating one suffix
product across an unbounded run of resets — would avoid this underflow entirely, and that is
exactly rl-triton's own kernel strategy at this scale. Building a chunked, periodically-
renormalized log-space *PyTorch* baseline to match is a real implementation option; it was
**not attempted here only because seq_len>=65536 is outside this project's target regime**
(see scope note above), not because no correct PyTorch baseline could exist. Read this study
as "torch.compile's simplest correct baseline fails outside the target regime, and a more
sophisticated one was never built," not as "torch.compile cannot do this."

| algorithm | seq_len | dispatch | triton full-call (ms) | triton device (ms) |
|:---|:---:|:---:|:---:|:---:|
| GAE | 65,536 | flat | 0.3174 | 0.2804 |
| GAE | 131,072 | flat | 0.6787 | 0.6339 |
| GAE | 262,144 | chunked | 0.5621 | 0.5177 |
| GAE | 524,288 | chunked | 1.1170 | 1.0755 |
| V-Trace | 65,536 | flat | 0.4384 | 0.3432 |
| V-Trace | 131,072 | flat | 1.0348 | 0.9765 |
| V-Trace | 262,144 | chunked | 0.8552 | 0.7304 |
| V-Trace | 524,288 | chunked | 1.6332 | 1.5663 |
| lambda-returns | 65,536 | flat | 0.3402 | 0.2873 |
| lambda-returns | 131,072 | flat | 0.6184 | 0.5742 |
| lambda-returns | 262,144 | chunked | 0.5083 | 0.4692 |
| lambda-returns | 524,288 | chunked | 1.0279 | 0.9848 |
| discounted-returns | 65,536 | flat | 0.2216 | 0.1563 |
| discounted-returns | 131,072 | flat | 0.6580 | 0.6173 |
| discounted-returns | 262,144 | chunked | 0.4619 | 0.4319 |
| discounted-returns | 524,288 | chunked | 0.9181 | 0.8899 |
| Retrace | 65,536 | flat | 0.6046 | 0.3857 |
| Retrace | 131,072 | flat | 0.8448 | 0.7084 |
| Retrace | 262,144 | chunked | 0.9267 | 0.8659 |
| Retrace | 524,288 | chunked | 1.8618 | 1.8048 |

Correctness holds on both sides of the flat/chunked boundary at every row above — the
property this study set out to check.
