# Chunked-kernel dispatch vs. torch.compile at long seq_len

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-25 · num_envs=16 fixed (matches tests/test_scan_chunked.py's own long-sequence configs).*

**One-off study, not a recurring benchmark.** Not part of bench_release.py, CI, or the test suite. `_FLAT_MAX_SEQ_LEN = 131072` is the shared flat/chunked dispatch threshold for every backward-scan op (GAE, V-Trace, lambda-returns, discounted-returns, and Retrace via its own reroute). Rows at seq_len <= 131072 use the flat kernel; rows above it use the chunked kernel — this is the first time that boundary has been benchmarked against a compiled baseline in this repository (a pre-existing internal-only test, `test_scan_chunked.py::test_scan_performance`, times the same dispatch but excludes any torch.compile comparison entirely).

Eligibility-traces and episodic-prefix-sum are excluded: their forward-scan variant has no chunked kernel and simply rejects seq_len > 131072.

**Baseline validity.** The "vec" compiled baseline (log-space cumsum / suffix-product) is checked against the exact sequential reference at every row, not assumed valid — it produces inf/nan from float32 underflow at these extreme lengths for some algorithms. Where it diverges, this is reported as `N/A (baseline invalid)`, not papered over with a misleading speedup ratio against broken output. Triton's correctness (via its chunked/flat auto-dispatch) is independently verified against the same exact reference at every row.

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
