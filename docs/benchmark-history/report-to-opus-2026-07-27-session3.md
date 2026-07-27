# Session report — to Claude Opus (session 3)

**From:** Claude Sonnet 5, branch `bench/h100-release`, NVIDIA H200 (this session's actual hardware —
see the GPU-mismatch note below; the branch name reflects the *target* release, not this pod's card).
**Date:** 2026-07-27.
**Continues directly from `docs/benchmark-history/report-to-opus-2026-07-26-session2.md` (session 2),
which ended blocked on a new, distinct Inductor autotuning crash (`illegal memory access`) specific to
`parallel_prefix_scan`'s flip/roll pattern, with a menu of options presented and no decision made yet.**

This session resolved that blocker, ran the full sweep to a clean exit, regenerated
`benchmarks.md`/the README draft, and produced (but did not commit) a floor-table proposal for this
session's actual GPU.

---

## 1. Prefix-scan blocker — resolved via a native forward-scan rewrite (not eager fallback)

`parallel_prefix_scan` (`tests/bench_utils.py`) was built as `flip -> parallel_suffix_scan -> flip`.
Rewrote it as a **native forward (left-to-right) Hillis-Steele doubling scan** — the direct mirror of
`parallel_suffix_scan`'s own doubling loop, using `torch.roll(+stride)` and zeroing the *front* `stride`
elements instead of `torch.roll(-stride)` zeroing the tail. No `torch.flip` anywhere.

**Verified, in order:**
- **(a) Correctness.** Matches the sequential reference to ~1e-6 (float32) / ~3.5e-15 (float64) for both
  eligibility-traces and prefix-sum, at every benchmarked shape including the non-power-of-2 production
  shape `(4096, 80)`. `vectorized_episodic_prefix_sum`'s correctness (found 95% wrong in session 2) was
  re-proven against the reference from scratch, not assumed.
- **(b) Compiles clean.** `torch.compile`'d across the full shape sweep (`CONFIGS` + `PRODUCTION_CONFIGS`
  + `BOUNDARY_CONFIG`, including `4096x80`) with zero illegal-memory-access / autotuning crashes.
- **(c) Baseline finalized.** This compiled native forward scan is now the baseline for both
  eligibility-traces and prefix-sum — no eager fallback was needed.

**Answer to "was the double-flip the provocation, or a fundamental Inductor bug?": the double-flip was
the provocation.** The native rewrite compiles cleanly on the exact shapes that crashed the flip-based
version, in the same environment, same torch/triton versions.

---

## 2. Process-state poisoning — real, reproduced independent of `reset()`, broader than known, mechanism still unproven

The subprocess-per-algorithm architecture (session 2) rested on: "even a brand-new `torch.compile`
object in the same process is corrupted once any prior compile happened." That repro still used
`torch._dynamo.reset()` calls elsewhere in the process, leaving open whether it was a reset-accumulation
artifact. This was checked directly this session, with **zero `torch._dynamo.reset()` calls anywhere**:

- Compiled two different functions (`vectorized_gae_with_truncations`,
  `vectorized_vtrace_with_truncations`) across the full non-padding `CONFIGS` grid, then reused one
  compiled object into `PRODUCTION_CONFIGS` (crossing into the padding-needing `seq_len=80` shape) —
  **reproduced the same wrong-output failure with zero resets.** This rules out "reset-accumulation
  artifact" as the explanation — the bug exists independent of resets.
- **New, broader finding:** a second, distinct instance was found in `vectorized_discounted_returns_with_truncations`,
  reusing one compiled object from `(64, 512)` to `(128, 1024)` — **neither shape needs padding.** ~22.5%
  of elements wrong (not NaN/inf), reproducing even in a subprocess that only ever compiles this single
  object (no other algorithm involved) — i.e. not about *total* accumulated compiles either; two
  compiles of one object is already enough. A fresh object's first-ever compile at either shape alone,
  or the same object used at only one shape, is always correct.
- **Mechanism: probable, not proven.** "Dynamo/Inductor process-global caching/guard-state corruption"
  is the working description (matches: reuse-across-shapes triggers it, a single fresh compile never
  does) but neither this nor the original padding-transition bug was root-caused via IR/generated-kernel
  inspection. Documented precisely in `NOTES.md` ("Bug 2 addendum") distinguishing the observed fact from
  the unproven mechanism, specifically so a future upstream PyTorch report doesn't overstate what's known.
- **Distinct from the illegal-memory-access CUDA crash** that originally justified subprocess isolation
  (session 2, bug #4): that crash was separately root-caused to the old flip/roll `parallel_prefix_scan`
  kernel (§1 above) and is believed fully resolved by the rewrite — a different failure mode (hard CUDA
  exception vs. silent wrong-but-finite numeric result caught by `assert_correctness`).

**Practical fixes applied (`tests/bench_release.py`):**
- Subprocess isolation kept (correctness-neutral, cheap) but made **finer-grained**: the `returns` group
  (lambda_returns + discounted_returns + eligibility_traces bundled into one subprocess, on the theory
  that splitting would only add redundant compute for no isolation benefit) was split into three solo
  subprocesses. This alone did *not* fix the discounted-returns bug — `bench_returns()` unconditionally
  computed and interleaved all three objects regardless of `--algos` — so `bench_returns()` was refactored
  to actually gate which objects get built/compiled/exercised by the `selected` algorithms, not just which
  tables get kept afterward.
- `bench_returns()`'s `CONFIGS` loop now calls `torch._dynamo.reset()` before *every* shape (not only at
  padding transitions, since this instance isn't about padding). Safe here because each subprocess
  accumulates on the order of a dozen resets, far under the count that separately corrupts CUDA state
  when resets accumulate unbounded in one long-lived process.
- A latent bug in the subprocess-isolation harness itself was also found and fixed: the per-group child
  process called `sys.exit(1)` merely because the *monotonicity gate* (a soft, noise-tolerant check —
  2% band, ~2-4% GPU timing jitter is expected) found violations, causing the parent to abort the entire
  sweep after the first group. Monotonicity violations are now collected and reported at the end (matching
  the single-process path's behavior) without halting subsequent groups; only genuine failures
  (`assert_correctness` raising, an uncaught exception) are treated as fatal.

---

## 3. Full sweep — clean exit, rigorously verified

`python tests/bench_release.py --parent-sweep --gpu "NVIDIA H200" --skip-readme`: **exit code 0**, all 7
subprocess groups (gae, vtrace, retrace, lambda_returns, discounted_returns, eligibility_traces,
prefix_sum) completed, zero tracebacks, zero mismatches.

**Per-shape correctness accounting** (not just "at exit"): `assert_correctness` raises immediately on
failure, before any row is printed or appended — so a printed row is proof that shape passed. Parsed the
full log and cross-referenced against the expected shape sets:
- All 7 base-CONFIGS loops (12/12 shapes each) and all 4 truncation-path loops (12/12 shapes each): no
  missing, no unexpected extras.
- All 7 production-regime loops (11/11 shapes each: 10 `PRODUCTION_CONFIGS` + 1 `BOUNDARY_CONFIG`): no
  missing.

**No silent gaps in the merged `benchmarks.md`:** every one of the 11 per-algorithm tables has exactly
14 rows (header + separator + 12 data rows, matching `CONFIGS`); the production-regime table has exactly
79 rows (header + separator + 77 data rows = 7 algorithms × 11 shapes exactly) — confirmed by counting
each algorithm label's occurrences (11 each, no duplication from the `returns`-trio subprocess split).

Regenerated: `benchmarks.md` (all 7 algorithms, both timing granularities, plain + truncation, production
table), `benchmarks/readme_table_draft.md` (includes V-Trace + truncation headline), `NOTES.md` (Bug 2
addendum, §2 above).

---

## 4. Floor table — proposed for H200, NOT committed

**GPU mismatch, flagged to the user before proceeding:** `bench_safeguard.py`'s `_FLOOR_TABLE` keys off
device-name substrings (`"RTX 2000 Ada"`, `"H100"`); this session's actual hardware (confirmed via
`nvidia-smi`) is an **NVIDIA H200**, not H100. The existing per-GPU skip logic correctly refused to reuse
another GPU's floor rather than silently mismeasuring. Asked the user how to proceed; they chose:
measure on H200, propose a new `h200_sxm` entry, leave the (void, pre-this-session) `h100_sxm` entries
untouched pending real H100 hardware access.

Measured via a standalone script mirroring `bench_safeguard.py`'s exact methodology (single 128×1024
shape, `torch.compile`'d vec baseline, `_bench_gpu_spread` 5-trial spread) without modifying that file —
3 independent process runs, min-of-3-runs × 0.9 margin (matching the existing table's convention):

| algo | run1 | run2 | run3 | gate value | proposed `h200_sxm` floor |
|---|---:|---:|---:|---:|---:|
| gae | 3.474 | 3.396 | 3.389 | 3.389 (min) | 3.05 |
| vtrace | 3.237 | 3.159 | 3.165 | 3.159 (min) | 2.84 |
| retrace | 2.127 | 2.145 | 2.129 | 2.127 (min) | 1.91 |
| lambda_returns | 2.960 | 3.021 | 2.892 | 2.892 (min) | 2.60 |
| discounted_returns | 3.131 | 3.146 | 3.071 | 3.071 (min) | 2.76 |
| eligibility_traces | 2.877 | 2.936 | 2.836 | 2.836 (min) | 2.55 |
| prefix_sum | 3.061 | 3.126 | 2.924 | 2.924 (median — prefix_sum gates on median, see its test's own comment) | 2.63 |

**Not committed to `bench_safeguard.py`** — this table is a proposal only, per the standing
STOP-and-report constraint on floor values.

---

## Constraints honored

- Every baseline (all 7 algorithms) passes `assert_correctness` vs. the sequential reference at every
  shape — verified per-shape, not just at exit; checks kept permanently.
- No concurrent GPU work while any sweep was live.
- No baseline was weakened or defaulted to eager to avoid the rewrite — the native forward scan is a
  genuine, correct, compiled replacement.
- `src/`, `README.md`, and paper prose untouched. No floor values committed without approval.

## Files touched this session

`tests/bench_utils.py` (native forward-scan rewrite), `tests/bench_release.py` (subprocess-granularity
fix, per-shape reset fix, monotonicity-abort fix), `benchmarks.md` / `benchmarks/readme_table_draft.md`
/ `docs/benchmark-history/unreleased.md` (regenerated from the fixed baselines), `NOTES.md` (Bug 2
addendum). `tests/bench_safeguard.py` intentionally untouched (floor proposal not committed).
