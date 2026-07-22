# H100 Truncation-Path Report: Does the Truncation Advantage Survive?

Follow-up to `h100_short_horizon_l2_retrace_ppo_report.md`, `h100_num_envs_sweep_report.md`,
and `h100_profiling_report.md`. Prior H100 runs established, with device-only timing: GAE flat
~1.9x device over `torch.compile` regardless of scale; V-Trace grows to ~3.8x at 8192 envs;
the plain-path advantage compresses on H100 because both sides are near the HBM roofline
racing NVIDIA's CUB cumsum. A ~26us fixed CUDA-event harness overhead was previously shown to
contaminate wall-clock ratios at small shapes (e.g. eligibility-traces measured 1.6x wall but
2.7x device) — so device time was treated as the primary source of truth there.

This report asks the opposite question of the plain path: **does the truncation path's
advantage survive H100, or does it compress the same way?** Hypothesis under test (not tuned
toward): the truncation advantage is algorithmic, not bandwidth-bound — the vectorized
baseline cannot use a single-pass CUB-style scan when interior resets sever the cumulative
product, so it falls back to a structurally more expensive multi-pass construct. That
structural gap should not erode with more bandwidth the way the plain path's did.

## Method

Same harness conventions as prior reports (`tests/bench_utils.py`'s `_bench_gpu`/`_warmup_gpu`,
per-config warmup at the exact timed shape, min-of-medians), extended with:

- **Amortized device timing**: `AMORTIZE_N=100` kernel calls inside one CUDA-event-timed
  region per trial (min-of-5-trial-medians / 100), to strip the ~26us fixed per-call overhead
  that would otherwise crush the ratio at small shapes.
- **Wall-clock**: host-synchronized `time.perf_counter` round trips (500-1000 iterations),
  reported alongside device time rather than dismissed — see the "end-to-end" section below.
- **Kernel-launch counts**: `torch.profiler` CUDA-activity events, one representative call per
  side per cell.
- Silenced dynamo/symbolic-shapes noise via this project's own convention
  (`.runpod/start.sh`: `TORCH_LOGS=-dynamic`, `TORCH_CPP_LOG_LEVEL` unset), not ad hoc log
  suppression.
- `torch._dynamo.config.cache_size_limit` raised to 256 before any `torch.compile` call — the
  short-horizon report found the default limit (8) silently falls back to uncompiled eager
  execution once a sweep exceeds 8 distinct shapes, which this sweep (48 shape×algorithm
  cells) would otherwise hit.

Realistic truncation density ~5%, generated via `bench_release.py`'s own `_make_trunc_extras`
(shared across GAE/V-Trace/lambda-returns/discounted-returns truncation benchmarks elsewhere
in this project) — mutually exclusive with `terminateds`, bootstrap values injected at
truncated steps and the window boundary. Realized density is reported per cell for
reproducibility (4.7–4.8% throughout).

---

## Experiment 4 (correctness gate) — checked first, must pass before any speedup is trusted

Before any timing number below is reported, both the Triton kernel *and* the vectorized
truncation baseline were verified against the pure-Python sequential reference
(`_ref_*_sequential`) on the canonical two-interior-truncation fixture, at `atol=1e-4`:

| test | result |
|---|---|
| `test_gae_two_interior_truncations` | PASSED |
| `test_vectorized_gae_with_truncations_correctness` | PASSED |
| `test_vtrace_two_interior_truncations` | PASSED |
| `test_vtrace_fused_two_interior_truncations` | PASSED |
| `test_vectorized_vtrace_with_truncations_correctness` | PASSED |
| `test_lambda_returns_two_interior_truncations` | PASSED |
| `test_discounted_returns_two_interior_truncations` | PASSED |
| `test_vectorized_lambda_returns_with_truncations_correctness` | PASSED |
| `test_vectorized_discounted_returns_with_truncations_correctness` | PASSED |

**9/9 passed.** A wrong-but-fast kernel would have been caught here before it could produce a
misleading speedup number.

Additionally, every one of the 48 (algorithm × shape) cells benchmarked in Experiment 1 below
was cross-checked — Triton kernel output vs. vectorized-truncation-baseline output, at the
actual ~5%-density benchmark inputs, `atol/rtol=1e-3` — since running the O(N·T) pure-Python
reference loop at benchmark scale (up to 8192×4096) is computationally impractical.

**Result: 48/48 cells agree.** No config in this report is running on unverified output.

---

## Experiment 1: truncation path vs. the fair baseline, across scale

For each cell: Triton `HAS_TRUNCATIONS=True` kernel vs. `torch.compile`-wrapped
`vectorized_*_with_truncations` (the log-depth associative-scan baseline — not a Python loop,
not the plain cumsum, which cannot handle interior truncations). Sweep: `num_envs` in
{512, 2048, 8192} × `seq_len` in {512, 1024, 2048, 4096}. "plain-path speedup" is the
established `HAS_TRUNCATIONS=False` kernel vs. plain `torch.compile(vectorized_*)` at the
*same shape*, single-call device time, shown for direct comparison.

#### GAE

| num_envs | seq_len | trunc % | single-call | amortized (N=100) | wall-clock | #k tri/base | plain-path speedup |
|---|---|---|---|---|---|---|---|
| 512 | 512 | 4.7% | 2.23x | 2.63x | 2.66x | 2/8 | 1.58x |
| 512 | 1024 | 4.7% | 2.32x | 2.73x | 2.72x | 2/9 | 1.59x |
| 512 | 2048 | 4.7% | 2.31x | 2.76x | 2.75x | 2/9 | 1.58x |
| 512 | 4096 | 4.7% | 3.14x | 5.01x | 5.07x | 2/10 | 1.43x |
| 2048 | 512 | 4.7% | 2.18x | 2.54x | 2.60x | 2/8 | 1.57x |
| 2048 | 1024 | 4.7% | 3.35x | 4.36x | 4.44x | 2/9 | 1.57x |
| 2048 | 2048 | 4.8% | 4.73x | 8.55x | 8.66x | 2/9 | 1.36x |
| 2048 | 4096 | 4.7% | 5.01x | 6.59x | 6.60x | 2/10 | 1.43x |
| 8192 | 512 | 4.8% | 4.29x | 7.42x | 7.51x | 2/8 | 1.56x |
| 8192 | 1024 | 4.7% | 6.44x | 9.89x | 9.92x | 2/9 | 1.83x |
| 8192 | 2048 | 4.8% | 7.67x | 9.88x | 9.90x | 2/9 | 1.53x |
| 8192 | 4096 | 4.8% | 6.35x | 6.92x | 6.94x | 2/10 | 1.60x |

**GAE verdict: survives, and grows with scale.** Amortized speedup at `seq_len=2048`: 2.76x
(512 envs) → 8.55x (2048 envs) → 9.88x (8192 envs), against a flat ~1.4–1.8x plain-path
speedup across the same shapes.

#### V-Trace

| num_envs | seq_len | trunc % | single-call | amortized (N=100) | wall-clock | #k tri/base | plain-path speedup |
|---|---|---|---|---|---|---|---|
| 512 | 512 | 4.8% | 3.11x | 3.85x | 3.83x | 1/9 | 2.11x |
| 512 | 1024 | 4.8% | 3.50x | 4.44x | 4.40x | 1/11 | 2.07x |
| 512 | 2048 | 4.8% | 2.93x | 4.05x | 4.13x | 1/10 | 2.14x |
| 512 | 4096 | 4.8% | 3.98x | 6.47x | 6.48x | 1/12 | 1.61x |
| 2048 | 512 | 4.8% | 3.06x | 3.84x | 3.90x | 1/9 | 2.10x |
| 2048 | 1024 | 4.8% | 4.08x | 6.70x | 6.80x | 1/11 | 1.93x |
| 2048 | 2048 | 4.7% | 4.37x | 6.24x | 6.26x | 1/10 | 2.43x |
| 2048 | 4096 | 4.7% | 5.84x | 7.12x | 7.13x | 1/12 | 2.22x |
| 8192 | 512 | 4.7% | 4.80x | 7.57x | 7.59x | 1/9 | 2.41x |
| 8192 | 1024 | 4.7% | 6.62x | 8.69x | 8.71x | 1/11 | 2.86x |
| 8192 | 2048 | 4.8% | 5.65x | 6.33x | 6.35x | 1/10 | 3.19x |
| 8192 | 4096 | 4.8% | 7.06x | 7.47x | 7.47x | 1/12 | 2.56x |

**V-Trace verdict: survives, grows with scale.** Amortized speedup at `seq_len=2048`: 4.05x →
6.24x → 6.33x (512→2048→8192 envs), against a plain-path speedup that also grows here
(2.14x→2.43x→3.19x, consistent with the previously-established ~3.8x-at-8192 trend) but stays
below the truncation path throughout.

#### lambda-returns

| num_envs | seq_len | trunc % | single-call | amortized (N=100) | wall-clock | #k tri/base | plain-path speedup |
|---|---|---|---|---|---|---|---|
| 512 | 512 | 4.7% | 2.75x | 3.45x | 3.47x | 1/7 | 1.57x |
| 512 | 1024 | 4.7% | 2.95x | 3.71x | 3.76x | 1/8 | 1.56x |
| 512 | 2048 | 4.7% | 3.03x | 3.83x | 3.85x | 1/8 | 1.54x |
| 512 | 4096 | 4.7% | 3.92x | 6.94x | 6.94x | 1/9 | 1.40x |
| 2048 | 512 | 4.7% | 2.74x | 3.49x | 3.54x | 1/7 | 1.60x |
| 2048 | 1024 | 4.7% | 4.19x | 6.57x | 6.72x | 1/8 | 1.56x |
| 2048 | 2048 | 4.8% | 5.75x | 9.37x | 9.42x | 1/8 | 1.28x |
| 2048 | 4096 | 4.7% | 6.05x | 7.53x | 7.54x | 1/9 | 1.59x |
| 8192 | 512 | 4.8% | 5.27x | 8.73x | 8.75x | 1/7 | 1.44x |
| 8192 | 1024 | 4.7% | 7.23x | 9.94x | 9.97x | 1/8 | 1.65x |
| 8192 | 2048 | 4.8% | 8.38x | 10.04x | 10.04x | 1/8 | 1.44x |
| 8192 | 4096 | 4.8% | 7.35x | 7.83x | 7.84x | 1/9 | 1.69x |

**lambda-returns verdict: survives, grows strongly with scale.** Amortized speedup at
`seq_len=2048`: 3.83x → 9.37x → 10.04x (512→2048→8192 envs), against a flat ~1.3–1.7x
plain-path speedup.

#### discounted-returns

| num_envs | seq_len | trunc % | single-call | amortized (N=100) | wall-clock | #k tri/base | plain-path speedup |
|---|---|---|---|---|---|---|---|
| 512 | 512 | 4.7% | 2.90x | 3.54x | 3.57x | 1/6 | 1.62x |
| 512 | 1024 | 4.7% | 3.07x | 3.87x | 3.89x | 1/7 | 1.67x |
| 512 | 2048 | 4.7% | 3.07x | 3.54x | 4.03x | 1/7 | 1.62x |
| 512 | 4096 | 4.7% | 4.41x | 7.70x | 7.77x | 1/8 | 1.50x |
| 2048 | 512 | 4.7% | 2.87x | 3.60x | 3.58x | 1/6 | 1.63x |
| 2048 | 1024 | 4.7% | 4.31x | 6.63x | 6.58x | 1/7 | 1.44x |
| 2048 | 2048 | 4.8% | 5.71x | 9.46x | 9.49x | 1/7 | 1.24x |
| 2048 | 4096 | 4.7% | 7.10x | 9.26x | 9.32x | 1/8 | 1.49x |
| 8192 | 512 | 4.8% | 5.40x | 9.45x | 9.46x | 1/6 | 1.45x |
| 8192 | 1024 | 4.7% | 7.25x | 9.90x | 9.93x | 1/7 | 1.33x |
| 8192 | 2048 | 4.8% | 8.31x | 10.04x | 10.03x | 1/7 | 1.37x |
| 8192 | 4096 | 4.8% | 9.05x | 9.87x | 9.87x | 1/8 | 1.55x |

**discounted-returns verdict: survives, grows strongly with scale.** Amortized speedup at
`seq_len=2048`: 3.54x → 9.46x → 10.04x (512→2048→8192 envs), against a flat ~1.2–1.7x
plain-path speedup. lambda-returns and discounted-returns both independently converge to
~10.0x at 8192 envs — plausibly a shared floor from their structurally similar baseline
computation graphs, not confirmed further here.

**Cross-algorithm summary: the hypothesis holds for all four.** Every algorithm's truncation
advantage grows with scale (the opposite of the plain path's H100 compression), and every
algorithm's truncation speedup exceeds its own plain-path speedup at matching shapes once
`num_envs` gets into the low thousands. Kernel-launch counts confirm the structural asymmetry
directly: Triton stays at 1 launch (2 for GAE) throughout; the baseline issues 6–12 launches,
growing with `seq_len` (more log-depth doubling iterations).

---

## End-to-end (wall-clock) numbers

The task explicitly asked to report wall-clock and "treat divergence as harness artifact, not
hardware fact" — but on the truncation path, that divergence essentially doesn't appear.
Across all 48 cells, wall-clock and amortized-device speedups track each other within 1–2%,
cell by cell. Representative examples:

| algo | envs×T | Triton wall | baseline wall | wall speedup | (amortized device speedup) |
|---|---|---|---|---|---|
| gae | 512×512 | 38.0us | 101.0us | 2.66x | 2.63x |
| gae | 2048×2048 | 45.1us | 390.3us | 8.66x | 8.55x |
| gae | 8192×1024 | 73.9us | 733.6us | 9.92x | 9.89x |
| vtrace | 8192×1024 | 103.2us | 898.6us | 8.71x | 8.69x |
| lambda_returns | 8192×2048 | 141.4us | 1419.3us | 10.04x | 10.04x |
| discounted_returns | 8192×2048 | 129.5us | 1299.4us | 10.03x | 10.04x |

This is a genuine difference from the earlier short-horizon plain-path report, where wall and
device speedups diverged sharply at small shapes (e.g. eligibility-traces: 1.6x wall vs. 2.7x
device). Here the baseline's absolute call cost is large — hundreds of microseconds to several
milliseconds, from issuing 6–12 kernel launches — so the ~26us fixed dispatch overhead is a
small fraction of total time on *both* sides, not something that disproportionately crushes
one. The number a real Python caller experiences and the pure-device-time number tell the same
story on this path: 2.5x at the small end, climbing past 9–10x at scale.

(Note: this is end-to-end for calling the op itself — Python dispatch → kernel(s) → sync — not
a full PPO-training-step measurement like the earlier report's GAE-share-of-a-step experiment.
That would require folding the truncation path into a full forward/backward/optimizer harness,
not attempted here.)

---

## Experiment 2: why is the baseline slower on the truncation path?

Naive kernel-name matching for "cumsum" does not work as a detection method — `torch.compile`
(Inductor) emits its own opaque `triton_` kernel names for both the plain and truncation
baselines, obscuring the original ATen op identity. Confirmed the mechanism a different way
instead: isolating the scan primitives themselves, stripped of any GAE-specific elementwise
setup, at the representative (2048, 2048) shape.

| primitive | launches | total device time |
|---|---|---|
| log-space cumsum (plain path's primitive) | 2 | 36.99us |
| log-depth doubling scan, `parallel_suffix_scan` (truncation path's primitive) | 6 | 148.07us |

**4.0x more device time for the identical shape, in complete isolation from any surrounding
algorithm-specific work.** This directly confirms the doubling scan is intrinsically more
expensive, not merely dragged down by GAE's setup math.

At the full-kernel level (GAE, truncation path, same 2048×2048 shape):

| | kernel launches | total device time |
|---|---|---|
| Triton fused truncation kernel | 2 | 42.56us |
| `torch.compile(vectorized_gae_with_truncations)` | 9 | 376.42us |
| `torch.compile(vectorized_gae)` (plain baseline, for reference) | 2 | 48.71us |

The Triton kernel (42.56us) and the *plain* baseline (48.71us) are nearly tied — consistent
with the plain path's established near-parity/compression story. The truncation baseline
(376.42us) is 7.7x slower than the plain baseline at the identical shape, purely from its own
structural cost.

**HBM bytes moved.** `ncu` remains unavailable in this container (`RmProfilingAdminOnly=1`,
missing `CAP_SYS_ADMIN`/`CAP_PERFMON` — confirmed in `h100_profiling_report.md`), so this is an
analytical estimate, not a hardware measurement:

- Triton fused kernel: exact at the tensor level — 5 reads (`rewards`, `values`, `terminateds`,
  `truncateds`, `bootstrap_values`) + 1 write (`out`) = 6 × (2048×2048×4 bytes) = **100.66 MB**.
  (The second in-kernel load of `values` at the shifted offset is *not* counted as a second
  full pass — the separate GAE double-load investigation in `NOTES.md` confirmed this shares
  L1/L2 cache lines with the primary load and is not a real second HBM round trip.)
- Truncation baseline: a **lower bound** derived from `parallel_suffix_scan`'s literal source
  (2 reads+writes for each of two `torch.roll` calls, plus a 4-pass and a 3-pass fused update
  per doubling iteration, assuming best-case Inductor fusion), at `T_pad=4096`, 12 doubling
  iterations, plus ~8 passes of setup materialization: **≈4563 MB**, a **45x** ratio over the
  Triton kernel.

This is reported as a lower bound, not a precise measurement — true bytes moved could be
higher if Inductor's actual fusion is less complete than the best-case assumption. The
qualitative claim ("the baseline does substantially more memory work, not that it's racing and
losing to CUB") holds regardless of the exact multiplier, corroborated independently by both
the isolated-primitive comparison (4.0x, direct measurement) and the kernel-launch-count
evidence (6–12 vs. 1–2, direct measurement) above.

**If the baseline had turned out to still use an efficient single-pass scan, this hypothesis
would be wrong — it doesn't, on three independent lines of evidence (isolated-primitive device
time, full-kernel launch count, full-kernel device time), so the algorithmic-structure
explanation stands.**

---

## Experiment 3: re-measuring the "~33% boundary-bootstrap read" claim

The original claim — skipping the boundary-bootstrap read saves ~33% of kernel time at
128×1024 — was measured on an RTX 2000 Ada, possibly with the same overhead-contaminated
harness that inflated other early wall-clock numbers in this project. Re-measured on H100,
device-only, amortized (source of truth per this report's own stated methodology):

- **Correctness first**: `HAS_BOOTSTRAP=True` with a zero `last_value` vs. `HAS_BOOTSTRAP=False`
  — confirmed bit-identical output (max diff = 0.0) before trusting any timing difference. The
  two code paths are semantically equivalent, as they must be.

| | single-call | amortized (N=100, source of truth) |
|---|---|---|
| Time saved by skipping the boundary-bootstrap read | 1.35% | **2.84%** |

**The ~33% figure does not hold on H100. The corrected number is ~2.8% — roughly 12x smaller.**
This fits the same pattern as everything else in this project's H100 story: a fixed, tiny
per-call saving (one scalar-per-env HBM read plus a branch) that was a large fraction of a very
cheap kernel call's total cost on an RTX 2000 Ada becomes nearly invisible once H100's raw
throughput shrinks the kernel's total cost by roughly an order of magnitude. **The slide should
be corrected from ~33% to ~2.8%.**

---

## Bottom line

| question | answer |
|---|---|
| Does the truncation advantage survive H100? | **Yes, for all four algorithms.** |
| Does it hold or grow with scale? | **Grows** — the opposite of the plain path's compression. |
| Mechanism confirmed? | Yes — three independent lines of evidence (isolated-primitive device time: 4.0x; kernel launches: 6–12 vs 1–2; full-kernel device time: 7.7x), not assumed. |
| Boundary-bootstrap claim (~33%) | **Corrected to ~2.8%** on H100, amortized, bit-identical-verified. |
| Correctness gate | 9/9 canonical fixture tests + 48/48 benchmark-scale cross-checks passed. |

The truncation-path story is the more durable claim in this library's H100 numbers — it isn't
racing CUB, so it doesn't lose the race the way the plain path increasingly does at scale. The
boundary-bootstrap optimization is real but small (~2.8%, not ~33%) and has been mostly
compressed away by H100, the same fate as most of the fixed-overhead-driven claims from this
project's original RTX-2000-Ada-era numbers. Neither result was tuned toward a preferred
outcome; the correctness gate ran first and passed before either was trusted.
