# Ablation: why is the fused Triton GAE kernel fast? Tree scan vs. fused serial

**Bottom line.** All three kernels are fused, single-launch, read-once/write-once kernels.
Against that fused serial baseline, the tree kernel's speedup comes from exposing the time
axis as parallel work. It does **not** come from fusion. It appears wherever T is long or
num_envs is small. Across the paper's Grid A the tree is 1.40×–11.8× faster than S_env on
full-call (3.4×–52× device) and 1.01×–6.7× faster than S_time (0.80×–25× device), with the
gap growing with T. **The exception is Grid A's (16384,128), the Gigaflow/PufferLib-style
shape.** There S_time ties the tree on full-call (1.01×) and is 1.24× *faster* on device time
(9.3 vs. 11.5 µs). **Serial wins in the short-horizon, massively-parallel regime.** A tuned
time-major (coalesced) serial kernel beats the tree:

- on device time at 16/30 shapes, by up to **11×** (32768×8: 1.8 µs vs. 20.9 µs);
- on full-call time by **24–33% at num_envs=32768, T≤64**, by **9–30% for S_env at
  num_envs≥16384, T≤16**, and by 8–10% for S_time at 16384×{16..64}, which is at the edge
  of the noise band.

Elsewhere in Grid B the full-call ratios sit inside the ±10% noise band (the ~25–30 µs launch
floor hides the kernel). Once S_time has to pay `.contiguous()` transposes from the library's
[E,T] layout, the tree wins full-call at every shape (1.33×–8.1×).

## Environment

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (132 SMs) |
| Driver | 595.91.07 |
| torch | 2.4.1+cu124 |
| Triton | 3.0.0 |
| Python | 3.11 |
| Branch | `ablation-serial` off `main` @ ff0b694; nothing outside `benchmarks/ablation_serial/` modified |

## What was compared

- **tree**: library `gae_fused_kernel`, plain path (`HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=True`),
  launched with the library's own dispatch (`BLOCK_SIZE=next_pow2(T)`, `_WARPS`, `num_stages`).
  It is not re-tuned. `tree_lib` (public `compute_gae`, which adds asserts) is also in `results.csv`.
- **S_env / S_time** (`serial_gae.py`): one program per `BLOCK_E` envs, backward loop over t,
  delta/beta built in-register, carry in registers, V(s_{t+1}) register-carried (every input
  element read exactly once), one store per element, one launch. S_env reads [E,T] (strided per
  step). S_time reads [T,E] (coalesced per step).
- **scan**: `torch.compile(vectorized_gae_with_truncations)` (log-depth doubling scan), with
  zero truncations and the bootstrap column prebuilt in eager code, exactly as in `tests/bench_release.py`.
- All kernels use the same thin wrapper style (allocate the output, launch, return).

**Serial tuning budget (deviation from spec, see NOTES.md).** I swept BLOCK_E ∈ {32,64,128,256} ×
num_warps ∈ {1,2,4,8} as specified, plus **UNROLL ∈ {1,2,4,8}** and **NUM_STAGES ∈ {1,3}**.
That makes 128 configs per shape and layout. UNROLL issues the loads for UNROLL steps before
running those UNROLL dependent steps, so the chain is still T steps long. This was necessary:
the spec-literal loop exposes one memory latency per step, and the tuned kernel is up to
7.7× faster than it on device time (S_time 4096×2048). The spec-literal loop is still reported,
as `serial_*_naive`. Selection works in three stages:

1. Correctness-gate every config.
2. Screen by device time.
3. Run the full protocol on the top 4, pick the best full-call, and re-measure the winner
   independently. That re-measurement is the reported number.

The chosen configs are in `results.csv` under `best_config`. UNROLL=8 wins at most shapes.

## Correctness gate (Step 2)

- Inputs: randn rewards and values, terminations at ~5% per step, bootstrap V(s_T) ~ N(0,1).
- Reference: `tests/test_gae.py::reference_gae`, sequential, run on every shape.
- Tolerance: `assert_correctness` with atol=rtol=1e-4.
- **Every reported row of tree, tree_lib, scan, S_env, S_time and S_time+transpose passed, at
  all 32 shapes.** Max abs error ≤ 6e-6.
- **Excluded configs:** 720 of 7680 serial sweep configs, all `NUM_STAGES=3`, all of them
  Triton 3.0 software-pipeliner *compile* failures (`arith.select` layout mismatch). No config
  compiled and then produced wrong numbers. No shape was excluded.
- **Direct cross-kernel check** (`check_equivalence.py`, output in `equivalence_check.txt`).
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

## Timing protocol (Step 3)

- **Full-call**: `tests/bench_utils.py` `_warmup_gpu`(20) + `_bench_gpu`. That is CUDA events,
  5 trials × 50 iterations, min of the trial medians.
- **Device**: `_device_profile` (torch.profiler CUDA kernel time, 20 warmup + 50 calls), min
  over 5 trials.
- **Isolation**: one subprocess per (shape, kernel). 160 subprocesses, 0 failures.
- **Noise.** I reran 6 short-T shapes in fresh processes (`repeat_check.jsonl`). Device time
  reproduces to 0.1% for tree and 0–14% for serial; the serial spread comes from re-picking
  among near-tied configs. Full-call reproduces to about ±5%, with one +43% outlier
  (tree 4096×64). **I treat full-call ratios in [0.9, 1.1] as ties**; the figure shades this band.

## Crossover seq_len per num_envs and layout

Ratio = serial time / tree time; >1 means the tree is faster. Crossover points are
log₂-interpolated between the bracketing grid points (raw ratios are in `report_stats.txt`).

**Full-call** (headline, `fig_tree_vs_serial.pdf`):

| num_envs | T measured | S_env | S_time |
|---|---|---|---|
| 64, 128, 256 | 512 / 1024 / 1024 | tree wins (2.9×, 4.4×, 4.8×) | tree wins (1.9×, 2.6×, 2.7×) |
| 512 | 128–4096 | tree wins everywhere (1.47× → 11.8×) | tree wins everywhere (1.14× → 6.7×) |
| 4096 | 8–2048 | tie at T ≤ 32 (0.97–1.06); tree from **T=64** (1.12×) | tie at T ≤ 64 (0.99–1.10); tree from **T=128** (1.15×) |
| 8192 | 8–128 | tie at T=8 (1.06); tree from **T=16** (1.16×) | **tie at every T ≤ 128** (1.01–1.10); no crossover in range |
| 16384 | 8–512 | **serial wins at T=8** (0.86×), T=16 borderline (0.91×); crossover **T≈24**; tree clearly from T=64 (1.39×) | serial ahead at T=16–64 (0.90–0.92×, band edge); tie at 128 (1.01); tree at T=512 (1.11×); crossover **T≈117** |
| 32768 | 8–128 | **serial wins at T ≤ 16** (0.70×, 0.72×); crossover **T≈29**; tree from T=64 (1.63×) | **serial wins at T ≤ 64** (0.67–0.76×); crossover **T≈114**; T=128 tie (1.07) |

**Device time** (`fig_tree_vs_serial_device.pdf`):

| num_envs | S_env crossover | S_time crossover |
|---|---|---|
| ≤ 512 | none, tree wins everywhere (8×–52×) | none, tree wins everywhere (4×–25×) |
| 4096 | **T≈25** (16: 0.60× → 32: 1.31×) | **T≈57** (32: 0.80× → 64: 1.05×) |
| 8192 | **T≈29** (16: 0.47× → 32: 1.15×) | **T≈106** (64: 0.77× → 128: 1.10×) |
| 16384 | **T≈31** (16: 0.36× → 32: 1.08×) | **T≈313** (128: 0.80× → 512: 1.13×) |
| 32768 | **T≈31** (16: 0.34× → 32: 1.05×) | **T≈126** (64: 0.32× → 128: 1.02×, a tie) |

## Max / min tree-vs-serial ratios (all 30 shapes)

| comparison | full-call max | full-call min | device max | device min |
|---|---|---|---|---|
| tree vs S_env (tuned) | 11.82× @ 512×4096 | **0.70× @ 32768×8** | 51.6× @ 128×1024 | **0.13× @ 32768×8** |
| tree vs S_time (tuned) | 6.74× @ 512×4096 | **0.67× @ 32768×16** | 25.3× @ 128×1024 | **0.09× @ 32768×8** |
| tree vs S_env_naive (spec-literal) | 21.6× @ 512×4096 | 0.67× @ 32768×8 | 88.9× @ 128×1024 | 0.19× @ 32768×8 |
| tree vs S_time_naive (spec-literal) | 35.4× @ 512×4096 | 0.67× @ 32768×16 | 93.9× @ 128×1024 | 0.13× @ 32768×8 |
| tree vs S_time + 3 input transposes | 8.13× @ 512×4096 | 1.33× @ 32768×32 | 31.9× @ 128×1024 | 0.42× @ 32768×8 |
| tree vs compiled scan | 5.66× @ 4096×2048 | 1.37× @ 32768×16 | 9.2× @ 4096×512 | **0.30× @ 32768×8** |

Shapes where serial (or scan) beats the tree:

- **S_env**: full-call at 5/30 shapes (4096×16 0.97, 16384×8 0.86, 16384×16 0.91, 32768×8 0.70,
  32768×16 0.72); device at 8/30.
- **S_time**: full-call at 9/30 shapes (4096×64 0.99, 16384×{8,16,32,64} 0.90–0.97,
  32768×{8,16,32,64} 0.67–0.76); device at 16/30. Ratios ≥ 0.9 are ties; the rest are outside
  the noise band.
- **Compiled scan** also beats the tree on *device* time at 6 shapes (8192–32768 × T∈{8,16},
  by up to 3.3×). It loses on full-call everywhere because of its multi-kernel launch overhead.

**Absolute numbers:**

- At 32768 envs the tree's device time is flat at ~21 µs for every T from 8 to 128. At
  32768×8 that is ≈200 GB/s effective, while S_time moves the same bytes in 1.8 µs (≈2.3 TB/s).
  This is consistent with the one-program-per-env, `BLOCK_SIZE=T` design being bound by program
  count and occupancy at short T, not by bandwidth or depth. This is inferred from timings only;
  I did not profile it with ncu.
- At 32768×128 both kernels are near bandwidth (23.2 vs. 23.7 µs) and tie.

**Transpose cost for S_time.** Three `.T.contiguous()` input transposes from [E,T] take 33–35 µs
full-call at small shapes (three launches) and 55.7 µs device / 68.1 µs full-call at 16384×128.
That is **6× the S_time kernel itself** (9.3 µs), and up to 213 µs device at 4096×2048. With the
transpose included, serial never wins full-call. On device time it still wins at 16384×{8,16}
and 32768×{8,16}.

## Interpretation (for the paper)

1. **Fusion and single launch alone do not explain the speedup at the paper's shapes.** All
   three kernels are fused. Against a fused, single-launch, latency-hidden serial kernel, the
   tree still wins at every Grid A shape with num_envs ≤ 4096 or T ≥ 512: 1.11×–11.8×
   full-call and 1.13×–52× device. The gap grows with T and shrinks with num_envs. That is the signature of the tree exposing T-way parallelism that
   the serial kernel can only get from E. The serial kernel has E/BLOCK_E programs, which is
   only 2–16 programs at E ≤ 512.
2. **The log-depth advantage disappears in the high-E / short-T regime, and the tree loses
   there.** With E ≥ 16384 and T ≤ 64, a coalesced serial walk has plenty of parallelism and
   beats the tree by up to 11× on device time and up to 33% full-call. Any claim that the tree
   kernel is the fastest option for massively parallel short-horizon rollouts (Isaac Gym /
   GPUDrive-style T=8–128) is **not supported** against this baseline, unless it is
   conditioned on the library's [E,T] layout. The tree's edge in that regime is avoiding a
   layout transpose, not scan depth.
3. In much of Grid B the launch floor (~25–30 µs for any Triton 3.0 launch through the Python
   wrapper) makes tree and serial indistinguishable on full-call.

## Step 4: end-to-end gap fill

**PPO E2E**, hidden=(128,128), num_envs=16384, seq_len=128, Triton arm vs. compiled-scan arm.
Run through `ppo_e2e_measurement.py`'s own `Config`/`correctness_gate`/`measure`/`_row`,
`e2e_gapfill.py`, `e2e_gapfill.csv`. Protocol: eager nets, event timing, 40 warmup / 30 timed,
interleaved rotating arms, 9 reruns each in a fresh process.

- Correctness gate: scan vs. triton passes all 9 reruns.
- **Contamination: 0/9 reruns flagged** (|non-GAE vs ref − 1| ≤ 0.46%; threshold 1%).
- Step total: 70.70–70.92 ms (triton), 70.78–71.07 ms (scan).
- **GAE share: triton 0.041–0.042%** (GAE 0.0293–0.0297 ms), **scan 0.140–0.141%**
  (0.0988–0.0998 ms).
- **E2E speedup (scan total / triton total, per-rerun median): 0.9974×–1.0056×**, median of
  reruns 1.0008×. The GAE difference alone predicts 1.0010×. The run-to-run spread (±0.5%,
  tracking the non-GAE drift) is 5× larger than that, so **there is no resolvable end-to-end
  speedup at this configuration.**

**ResiP-inspired learner benchmark.** This script is not on `main`. I ran
`origin/resip-ppo-benchmark:benchmarks/resip_ppo_e2e_measurement.py` unmodified, from a
scratch copy: `--seq-lens 700 --epochs 50 --arms triton,scan --recompute every_epoch`. It uses
its own protocol (20 warmup, 5 trials × 100 iterations, min trial median), not the 40/30/9
protocol above. Outputs are in `resip_gapfill.csv` and `resip_gapfill_log.txt`.

- num_envs=1024, T=700, 50 epochs, GAE recomputed every epoch (50 calls per update).
- Correctness gate: PASS (max abs 3.6e-7). Non-GAE vs ref: 1.0005, **not contaminated**.
- Triton: total 1006.2 ms, GAE 0.999 ms (device 0.699 ms), **share 0.099%**.
- Scan: total 1009.9 ms, GAE 2.976 ms (device 2.676 ms), **share 0.295%**.
- **E2E: scan/triton = 1.0025×** (winning-trial IQR 0.0097, trial spread 0.0025). The GAE
  difference alone predicts 1.0020×. The IQR spans 1.0, so this is **within noise**.

## Files

| file | contents |
|---|---|
| `results.csv` | shape, grid, kernel, layout, best_config, device_ms, fullcall_ms, validated, max_abs_err, tree speedups |
| `sweep_serial.csv` | all 7680 serial configs: validated, error, screening device time, full-call (top-4) |
| `raw_results.jsonl`, `repeat_check.jsonl`, `report_stats.txt`, `run_ablation_log.txt` | raw worker output, repeatability rerun, full statistics dump, sweep log |
| `fig_tree_vs_serial.pdf` (+ `_device.pdf`, PNGs) | tree speedup over serial vs. T, one line per num_envs, S_env and S_time panels |
| `e2e_gapfill.csv`, `e2e_gapfill_log.txt` | Step 4 PPO E2E, 9 reruns × 2 arms |
| `resip_gapfill.csv`, `resip_gapfill_log.txt` | Step 4 ResiP learner benchmark |
| `serial_gae.py`, `bench_worker.py`, `run_ablation.py`, `make_report.py`, `e2e_gapfill.py` | code; rerun with `python benchmarks/ablation_serial/run_ablation.py && python benchmarks/ablation_serial/make_report.py` |
| `check_equivalence.py`, `equivalence_check.txt` | direct serial vs. tree / scan / fp64 equivalence check, incl. edge cases |
| `RESULTS_FOR_OPUS.md` | self-contained results brief |
| `NOTES.md` | Step 0 paths and every deviation from the task spec |
