# Tree-vs-serial GAE ablation: results brief

This brief is self-contained, for a model or co-author drafting the paper's ablation section.
Every number here comes from `results.csv`, `e2e_gapfill.csv` and `resip_gapfill.csv` in this
directory. Branch `ablation-serial`, commit `2a9af83`. Full write-up: `SUMMARY.md`. Deviations
from the original spec: `NOTES.md`.

## Question

The fused Triton GAE kernel computes `A[t] = δ[t] + β[t]·A[t+1]` with a tree scan
(`tl.associative_scan`, depth O(log T)) in a single fused launch. Where does its speedup come
from: the **log-depth scan**, or **fusion and a single launch**?

To separate the two, the tree kernel is compared against **fused serial kernels**. These are
identical in everything except dependency depth (T) and access pattern. They read each input
once, write the output once, use one launch, and build δ and β in registers.

## Setup

- **Hardware and software**: NVIDIA H100 80GB HBM3, driver 595.91.07, torch 2.4.1+cu124,
  Triton 3.0.0. These are the paper's versions, not upgraded.
- **Semantics**: the library's plain path (`HAS_TRUNCATIONS=False`, `HAS_BOOTSTRAP=True`).
  Terminations zero both the successor value and β. The window-boundary bootstrap V(s_T)
  enters δ[T-1]. The carry starts at A[T]=0.
- **Kernels:**
  - **tree**: the library's `gae_fused_kernel` with the library's own dispatch: one program
    per env, `BLOCK_SIZE = next_pow2(T)`, and the `_WARPS` table. It is not re-tuned.
  - **S_env**: serial walk from t=T-1 down to 0. Each program owns BLOCK_E envs. Inputs are
    [E,T] (the library layout), so the loads at each step are strided.
  - **S_time**: the same logic on [T,E] inputs (CleanRL-style buffers), so the loads at each
    step are coalesced. Timed on pre-transposed inputs. A separate row,
    `S_time+transpose`, includes three `.T.contiguous()` input transposes.
  - **scan**: `torch.compile` of the existing doubling-scan baseline
    (`vectorized_gae_with_truncations`).
- **Serial tuning**: 128 configs per shape and layout: BLOCK_E ∈ {32,64,128,256} ×
  num_warps ∈ {1,2,4,8} × UNROLL ∈ {1,2,4,8} × NUM_STAGES ∈ {1,3}.
  - UNROLL issues the loads for UNROLL steps before running those dependent steps. The chain
    stays T steps long.
  - Without UNROLL the serial loop exposes one memory latency per step, which is up to 7.7×
    slower and would unfairly favour the tree.
  - The untuned loop is reported as `serial_*_naive`. It is slower, but leads to the same
    qualitative conclusion.
- **Correctness**: every reported row passed against the independent sequential reference at
  atol=rtol=1e-4. Inputs use ~5% terminations per step and a nonzero bootstrap. Max abs error
  ≤ 6e-6.
  - 720 of 7680 sweep configs (all `NUM_STAGES=3`) were excluded as Triton 3.0
    software-pipeliner compile failures.
  - No config produced wrong numbers, and no shape was excluded.

**Cross-kernel equivalence:**

- **Check** (`check_equivalence.py`, output in `equivalence_check.txt`).
  Serial kernels are compared directly against the tree kernel, the compiled scan and
  `compute_gae`, plus fp32 and fp64 sequential references. The 10 cases are 16384×128,
  512×4096, 32768×8, 100×40 (T not a power of 2, E not a multiple of BLOCK_E), 1×8, 3×1,
  0% and 100% terminations, γ=λ=1, and λ=0.
  - Serial vs. tree: max difference 1.4e-6 to 5.7e-6 (1.1e-5 at γ=λ=1).
  - Serial vs. scan: max difference 1.9e-6 to 5.7e-6 (1.3e-5 at γ=λ=1).
  - All kernels are within 1e-4 of the reference in all 10 cases.
  - **Not bit-identical**, which is expected because the summation order differs. Outputs are
    bit-identical only where nothing accumulates (T=1, all terminated, λ=0).
  - All serial variants (both layouts, 4 configs each) are bit-identical to each other. The tree
    wrapper is bit-identical to `compute_gae`.
  - Against fp64, the serial kernels' error grows faster with T than the tree's (5.7e-6 vs.
    2.8e-6 at T=4096): T sequential roundings vs. about log T. Both are far inside the 1e-4
    tolerance.
  - T must be a multiple of UNROLL (asserted, not handled). All benchmarked T are powers of two.
- **Timing protocol** (the paper's):
  - Full-call: CUDA events, 20 warmup iterations, then 5 trials × 50 iterations. The reported
    value is the min of the 5 trial medians.
  - Device time: torch.profiler kernel time, min over 5 trials.
  - Each (shape, kernel) pair ran in its own subprocess.
- **Noise** (from fresh-process reruns of 6 shapes):
  - Device time reproduces to ≤ 0.1% (tree) and ≤ 14% (serial; the spread comes from
    re-picking among near-tied configs).
  - Full-call reproduces to about ±5%, with occasional outliers.
  - **Full-call ratios in [0.9, 1.1] are treated as ties.** Every Triton launch through the
    Python wrapper has a floor of about 25–30 µs.

## Headline findings

1. **The tree's win at the paper's long-T and moderate-E shapes is not explained by fusion.**
   - The fused serial kernels lose at every Grid A shape with num_envs ≤ 4096 or T ≥ 512:
     1.11×–11.8× on full-call, 1.13×–52× on device time.
   - The gap grows with T and shrinks with num_envs. Serial parallelism is only E/BLOCK_E
     programs, which is 2–16 programs at E ≤ 512. The tree exposes parallelism along T.
2. **In the massively parallel, short-T regime, serial beats the tree.**
   - **Device time**: S_time wins at 16/30 shapes, by up to **11×** (32768×8: 1.8 µs vs.
     20.9 µs).
   - **Full-call time**:
     - S_time wins by **24–33%** at 32768 envs with T ≤ 64.
     - S_env wins by 14–30% at (16384,8), (32768,8) and (32768,16).
     - The remaining wins are within the ±10% tie band.
   - The tree's device time is flat at ~21 µs for every T from 8 to 128 at 32768 envs. That is
     about 200 GB/s effective, versus about 2.3 TB/s for S_time. This is consistent with the
     one-program-per-env, BLOCK_SIZE=T design being bound by program count at short T. It is
     inferred from timings only; I did not profile it with ncu.
3. **At the paper's (16384,128) shape, S_time ties the tree on full-call (1.01×) and is 1.24×
   faster on device time** (9.3 vs. 11.5 µs).
4. **The tree's practical edge in the short-T regime is layout, not depth.** Including the
   [E,T]→[T,E] transposes S_time needs, the tree wins full-call at all 30 shapes (1.33×–8.1×).
   The transposes alone cost 6× the S_time kernel at 16384×128 (55.7 vs. 9.3 µs device).
5. **The compiled scan beats the tree on device time at 6 shapes** (8192–32768 envs × T ∈ {8,16},
   by up to 3.3×). It loses on full-call everywhere (1.37×–5.66×) because of its multi-kernel
   launch overhead.

### Claims the data supports / does not support

| Claim | Status |
|---|---|
| "Speedup over a fused serial kernel grows with T; log-depth scan matters at long horizons / few envs" | **Supported** |
| "The speedup is mainly due to fusion / single launch" | **Not supported**: all arms are fused |
| "Tree kernel is fastest for massively-parallel short-horizon RL (E ≥ 16k, T ≤ 64)" | **Contradicted**: a coalesced serial kernel is faster |
| "Tree kernel is fastest given the library's [E,T] layout, including layout conversion" | **Supported** at all 30 shapes (full-call) |
| "Tree beats compiled scan" | **Supported on full-call** everywhere; **not on device time** at E ≥ 8192, T ≤ 16 |

## Crossover seq_len (tree becomes faster above this T)

Values are log₂-interpolated between grid points. "Tie" means the full-call ratio is within
[0.9, 1.1].

| num_envs | S_env full-call | S_time full-call | S_env device | S_time device |
|---|---|---|---|---|
| ≤ 512 | tree wins at all T | tree wins at all T | tree wins at all T | tree wins at all T |
| 4096 | tie at T ≤ 32; tree from T=64 | tie at T ≤ 64; tree from T=128 | T≈25 | T≈57 |
| 8192 | tie at T=8; tree from T=16 | tie at all T ≤ 128 | T≈29 | T≈106 |
| 16384 | T≈24 (serial wins at T=8) | T≈117 (serial ahead 8–10% at T=16–64) | T≈31 | T≈313 |
| 32768 | T≈29 (serial wins at T ≤ 16) | T≈114 (serial wins at T ≤ 64) | T≈31 | T≈126 |

## Full per-shape results

Each timing cell is device time / full-call time, in µs. The speedup columns give
serial ÷ tree: above 1 the tree is faster, below 1 serial is faster. The value in parentheses
is the device-time speedup.

| grid | num_envs | T | tree | S_env | S_time | S_time+transpose | scan | tree speedup vs S_env | tree speedup vs S_time |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 512 | 1.5 / 24.0 | 48.8 / 69.0 | 24.3 / 45.9 | 31.6 / 77.5 | 7.8 / 78.0 | 2.87× (33.04×) | 1.91× (16.43×) |
| A | 128 | 1024 | 1.9 / 26.6 | 96.2 / 116.7 | 47.2 / 68.5 | 59.5 / 103.4 | 11.4 / 85.1 | 4.39× (51.59×) | 2.58× (25.32×) |
| A | 256 | 1024 | 2.4 / 24.7 | 96.4 / 118.5 | 47.4 / 67.6 | 64.0 / 103.0 | 14.5 / 85.1 | 4.79× (40.07×) | 2.73× (19.70×) |
| A | 512 | 128 | 1.6 / 23.9 | 13.1 / 35.1 | 6.8 / 27.4 | 13.3 / 55.6 | 7.0 / 71.6 | 1.47× (8.20×) | 1.14× (4.26×) |
| A | 512 | 512 | 2.2 / 25.2 | 48.7 / 70.4 | 24.0 / 45.4 | 37.5 / 78.8 | 13.3 / 78.0 | 2.79× (21.69×) | 1.80× (10.70×) |
| A | 512 | 2048 | 7.0 / 27.4 | 190.9 / 212.1 | 93.8 / 114.1 | 153.0 / 174.2 | 41.0 / 84.6 | 7.73× (27.42×) | 4.16× (13.48×) |
| A | 512 | 4096 | 15.8 / 35.2 | 396.2 / 416.6 | 216.2 / 237.4 | 272.6 / 286.6 | 93.8 / 128.4 | 11.82× (25.08×) | 6.74× (13.69×) |
| B | 4096 | 8 | 3.7 / 26.0 | 1.6 / 27.6 | 1.4 / 28.8 | 6.3 / 56.7 | 3.9 / 52.4 | 1.06× (0.43×) | 1.10× (0.40×) |
| B | 4096 | 16 | 3.7 / 27.5 | 2.2 / 26.6 | 2.0 / 27.5 | 7.5 / 58.2 | 4.7 / 57.3 | 0.97× (0.60×) | 1.00× (0.55×) |
| B | 4096 | 32 | 3.7 / 27.4 | 4.8 / 27.5 | 3.0 / 29.2 | 10.4 / 55.9 | 6.5 / 61.8 | 1.00× (1.31×) | 1.06× (0.80×) |
| B | 4096 | 64 | 3.8 / 26.5 | 7.6 / 29.8 | 4.0 / 26.2 | 15.3 / 54.7 | 10.7 / 70.5 | 1.12× (2.00×) | 0.99× (1.05×) |
| A+B | 4096 | 128 | 4.0 / 24.9 | 13.6 / 34.8 | 8.0 / 28.5 | 25.1 / 57.9 | 17.0 / 71.4 | 1.40× (3.39×) | 1.15× (2.00×) |
| A | 4096 | 512 | 8.4 / 28.2 | 49.6 / 71.6 | 28.4 / 53.5 | 86.2 / 100.1 | 77.3 / 112.9 | 2.54× (5.91×) | 1.90× (3.39×) |
| A | 4096 | 2048 | 54.2 / 71.7 | 211.0 / 231.8 | 120.9 / 140.6 | 334.3 / 348.2 | 372.6 / 405.9 | 3.23× (3.89×) | 1.96× (2.23×) |
| B | 8192 | 8 | 6.1 / 27.0 | 1.6 / 28.7 | 1.6 / 27.5 | 6.7 / 59.1 | 4.2 / 52.6 | 1.06× (0.27×) | 1.02× (0.25×) |
| B | 8192 | 16 | 6.1 / 25.1 | 2.9 / 29.1 | 2.0 / 27.6 | 8.2 / 57.7 | 5.5 / 51.6 | 1.16× (0.47×) | 1.10× (0.33×) |
| B | 8192 | 32 | 6.1 / 26.2 | 7.1 / 30.8 | 3.0 / 27.0 | 14.1 / 54.4 | 8.9 / 61.9 | 1.17× (1.15×) | 1.03× (0.49×) |
| B | 8192 | 64 | 6.3 / 26.7 | 12.9 / 35.4 | 4.8 / 26.9 | 23.1 / 55.1 | 15.1 / 74.0 | 1.33× (2.05×) | 1.01× (0.77×) |
| B | 8192 | 128 | 6.5 / 26.7 | 24.6 / 45.6 | 7.1 / 29.0 | 39.0 / 59.6 | 32.5 / 69.3 | 1.71× (3.80×) | 1.09× (1.10×) |
| B | 16384 | 8 | 11.0 / 30.9 | 2.0 / 26.6 | 1.7 / 29.8 | 7.3 / 59.8 | 4.8 / 53.9 | 0.86× (0.19×) | 0.97× (0.15×) |
| B | 16384 | 16 | 11.1 / 29.6 | 4.0 / 26.8 | 2.2 / 27.1 | 10.0 / 54.1 | 6.4 / 54.8 | 0.91× (0.36×) | 0.92× (0.20×) |
| B | 16384 | 32 | 11.1 / 31.0 | 11.9 / 33.0 | 3.2 / 28.0 | 20.5 / 56.0 | 12.8 / 61.1 | 1.07× (1.08×) | 0.90× (0.29×) |
| B | 16384 | 64 | 11.2 / 31.4 | 21.9 / 43.7 | 5.1 / 28.8 | 36.7 / 56.8 | 29.0 / 71.6 | 1.39× (1.95×) | 0.92× (0.45×) |
| A+B | 16384 | 128 | 11.5 / 30.1 | 42.2 / 63.0 | 9.3 / 30.5 | 67.3 / 81.4 | 68.9 / 100.4 | 2.09× (3.66×) | 1.01× (0.80×) |
| A | 16384 | 512 | 45.3 / 64.6 | 159.6 / 181.7 | 51.0 / 71.4 | 256.1 / 270.8 | 327.3 / 360.1 | 2.81× (3.53×) | 1.11× (1.13×) |
| B | 32768 | 8 | 20.9 / 40.1 | 2.6 / 28.0 | 1.8 / 27.6 | 8.8 / 58.5 | 6.3 / 55.8 | 0.70× (0.13×) | 0.69× (0.09×) |
| B | 32768 | 16 | 20.9 / 40.7 | 7.0 / 29.4 | 2.6 / 27.2 | 13.1 / 54.7 | 9.5 / 55.9 | 0.72× (0.34×) | 0.67× (0.13×) |
| B | 32768 | 32 | 20.9 / 41.3 | 22.0 / 43.2 | 4.0 / 31.4 | 32.9 / 55.2 | 24.2 / 62.1 | 1.04× (1.05×) | 0.76× (0.19×) |
| B | 32768 | 64 | 21.2 / 40.0 | 42.8 / 65.3 | 6.8 / 29.1 | 66.0 / 80.4 | 61.9 / 94.0 | 1.63× (2.02×) | 0.73× (0.32×) |
| B | 32768 | 128 | 23.2 / 41.6 | 87.5 / 108.2 | 23.7 / 44.3 | 127.5 / 141.4 | 143.5 / 175.7 | 2.60× (3.78×) | 1.07× (1.02×) |

The chosen serial configs are in the `best_config` column of `results.csv`. UNROLL=8 wins at
most shapes.

## End-to-end (Step 4)

The kernel-level differences above do not show up end-to-end. GAE is ≤ 0.3% of a PPO update
in both settings measured here.

| Benchmark | Config | Arms | GAE share | E2E (scan ÷ triton) | Contamination |
|---|---|---|---|---|---|
| PPO E2E (`ppo_e2e_measurement.py`) | hidden (128,128), 16384 envs, T=128, eager nets; 40 warmup / 30 timed, 9 reruns, interleaved arms | triton vs. compiled scan | triton 0.041–0.042%, scan 0.140–0.141% | **0.9974×–1.0056×** (median 1.0008×; GAE alone predicts 1.0010×) | 0/9 flagged (max 0.46% non-GAE drift; threshold 1%) |
| ResiP-inspired learner (`resip_ppo_e2e_measurement.py`, from branch `origin/resip-ppo-benchmark`, run unmodified; script's own protocol of 20 warmup, 5×100) | 1024 envs, T=700, 50 epochs, GAE every epoch | triton vs. compiled scan | triton 0.099%, scan 0.295% | **1.0025×** (IQR 0.0097, spans 1.0; GAE alone predicts 1.0020×) | clean (non-GAE ratio 1.0005) |

Step totals: PPO about 70.8 ms per update; ResiP about 1006 ms per update.
**Neither benchmark shows a resolvable end-to-end speedup.**

## Caveats to state in the paper

- **Serial tuning was extended** with UNROLL and NUM_STAGES beyond the planned
  BLOCK_E × num_warps sweep. Without them the serial baseline is a strawman (2–8× slower).
  The naive numbers are in `results.csv` (`serial_*_naive`).
- **The tree kernel used the library's shipped dispatch** and was not re-tuned for short T.
  Its short-T weakness may be fixable, for example by packing several envs per program.
- **The full-call gap at short T is dominated by Triton's ~25–30 µs launch floor**, so device
  time is the cleaner signal for the depth question.
- **Device time** is torch.profiler kernel time (the repo's existing harness), not ncu. The
  occupancy explanation for the tree's flat ~21 µs is an inference from timings.
- **The ResiP script is not on `main`.**

## Files in this directory

- Code: `serial_gae.py`, `bench_worker.py`, `run_ablation.py`, `make_report.py`, `e2e_gapfill.py`.
- Kernel results: `results.csv`, `sweep_serial.csv` (all 7680 configs), `raw_results.jsonl`,
  `repeat_check.jsonl`, `report_stats.txt`.
- Figures: `fig_tree_vs_serial.pdf` (full-call), `fig_tree_vs_serial_device.pdf`.
- End-to-end: `e2e_gapfill.csv`, `resip_gapfill.csv`, plus `*_log.txt` run logs.
- Equivalence check: `check_equivalence.py`, `equivalence_check.txt`.
- Write-ups: `SUMMARY.md`, `NOTES.md`.
