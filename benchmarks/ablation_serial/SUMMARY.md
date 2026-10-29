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

## Memory behaviour of each implementation

- **Tree (shipped `gae_fused_kernel`)**
  - One program per env row.
  - Loads the whole row from HBM into registers once.
  - Runs a log-depth scan (`tl.associative_scan`) on-chip.
  - Writes the output once, in a single launch.
  - Precision note: it issues a second, shifted load of `values` for V(s_{t+1}). That load
    hits mostly in L1/L2, so HBM traffic is still about one read per input element.
- **Fused serial (S_env / S_time)**
  - One program per block of BLOCK_E envs, in ONE launch.
  - A loop over t runs INSIDE the kernel and loads one timestep per iteration from HBM.
    With UNROLL > 1, it loads UNROLL timesteps per iteration.
  - The carry and delta/beta stay in registers.
  - Each input element is read once and each output written once, so total HBM traffic
    equals the tree's.
  - The difference is load timing (per step vs. whole row) and dependency depth (T vs. log T).
- **Compiled PyTorch scan (`torch.compile(vectorized_gae_with_truncations)`)**
  - log2(T) doubling stages. Every kernel writes and re-reads full [E,T] tensors in HBM.
  - Measured with torch.profiler, Inductor fuses some stages, so the launch count is below
    one per stage: **3 launches at T=8 (3 stages), 5 at T=128 (7 stages), 8 at T=4096
    (12 stages)**.
- **PyTorch loop (`reference_gae`)**
  - T steps, each several kernel launches: **about 9 launches per step measured** (76 per
    call at T=8, 1156 at T=128).
  - The carry and all intermediates go to HBM and back every step.

## Follow-up results (second session, 2026-10-29)

These tasks were run in order. Every timed row passed the correctness check against the
independent sequential reference (atol=rtol=1e-4, ~5% terminations, nonzero bootstrap)
before being timed. Each (shape, kernel) pair ran in its own subprocess. Protocol as in
Step 3. All four tasks are committed and pushed to `origin/ablation-serial`.

### Task 1: headline provenance

**Verdict: after fix.**

- **Fix** (`compiled_extra_fn`, zero-tensor allocation moved out of the timed compiled
  closure): `c5695c5c14fede4325008295ccf8b25298fb7989`, 2026-08-04 13:02:39 +0000.
- **Last change to the production-regime tables in benchmarks.md** (v0.1.3 promote):
  `3e7088776db4cfaed7ce642f87913631571547ce`, 2026-08-08 14:13:30 +0100.
- The H100 sweep behind the tables (`90d3c26`, 2026-08-08) has the fix as an ancestor.
  `tests/bench_release.py` at that commit contains `compiled_extra_fn`.
- Details: `task1_provenance.md`.

### Task 2: long-horizon, few-env shapes

Shapes: num_envs ∈ {16, 64, 256} × T ∈ {8192, 16384, 32768, 65536}. Outputs:
`longT_results.csv`, `fig_longT.pdf`.

**Notes on the setup:**
- **Tree warps:** tree warp counts are tuned only up to BLOCK_SIZE=16384. Above that, `_WARPS`
  has no entry and dispatch falls back to 16 warps, so T ≥ 32768 runs untuned.
- **Serial tuning:** BLOCK_E ≤ num_envs, with BLOCK_E ∈ {8, 16} added (see NOTES.md).
- **Loop:** timed only for T ≤ 16384, with `torch.cuda.synchronize()` before the start event
  and after the end event of every call.
- **Excluded configs:** 560 of 3072 serial configs were excluded, all `NUM_STAGES=3` compile
  failures. None produced wrong numbers.
- **Scan rows:** the 12 compiled-scan rows were rerun in isolation, because the original ones
  overlapped a few seconds of a small unrelated GPU probe. Device time matched within 1%.
  11/12 full-call values were within ±10% and kept. 16×16384 (+10.6%) was replaced by the
  rerun value. Conclusions are unchanged.

**Absolute device time (ms):**

| num_envs | T | tree | S_env | S_time | scan | loop (full-call, synced) |
|---|---|---|---|---|---|---|
| 16 | 8192 | 0.0089 | 0.392 | 0.354 | 0.0133 | 460.9 |
| 16 | 16384 | 0.0171 | 0.784 | 0.707 | 0.0194 | 964.6 |
| 16 | 32768 | **0.155** | 1.568 | 1.414 | 0.0269 | — |
| 16 | 65536 | **0.387** | 3.135 | 2.834 | 0.0544 | — |
| 64 | 8192 | 0.0090 | 0.392 | 0.362 | 0.0247 | 474.5 |
| 64 | 16384 | 0.0178 | 0.784 | 0.723 | 0.0507 | 978.6 |
| 64 | 32768 | **0.227** | 1.696 | 1.705 | 0.1033 | — |
| 64 | 65536 | **0.660** | 3.542 | 3.473 | 0.2647 | — |
| 256 | 8192 | 0.0194 | 0.433 | 0.431 | 0.0993 | 495.8 |
| 256 | 16384 | 0.0522 | 0.890 | 0.873 | 0.2413 | 970.0 |
| 256 | 32768 | **1.057** | 1.782 | 1.733 | 0.4704 | — |
| 256 | 65536 | **2.767** | 3.566 | 3.464 | 1.0067 | — |

**Tree speedup on full-call** (other ÷ tree; above 1 means the tree is faster):

| num_envs | vs S_env (T=8192 / 16384 / 32768 / 65536) | vs S_time | vs scan | vs loop (T=8192 / 16384) |
|---|---|---|---|---|
| 16 | 14.9 / 21.5 / 9.1 / 7.8 | 13.6 / 19.4 / 8.2 / 7.0 | 3.24 / 2.81 / **0.58 / 0.26** | 16 669 / 25 699 |
| 64 | 14.4 / 21.9 / 7.0 / 5.2 | 13.3 / 20.2 / 7.0 / 5.1 | 3.32 / 2.70 / **0.57 / 0.44** | 16 495 / 26 546 |
| 256 | 10.8 / 12.7 / 1.67 / 1.29 | 10.7 / 12.4 / 1.63 / 1.25 | 3.18 / 3.86 / **0.47 / 0.38** | 11 755 / 13 466 |

**Crossover T per num_envs:**
- **Tree vs. S_env and vs. S_time:** none. The tree is faster at every long-T shape, on both
  full-call (1.25×–21.9×) and device time.
- **Tree vs. compiled scan: the scan overtakes the tree** between T=16384 and T=32768 at every
  num_envs.
  - Full-call crossover: T≈25 800 (E=16), ≈25 500 (E=64), ≈25 600 (E=256).
  - Device-time crossover: T≈17 200, ≈24 300, ≈25 800.

**Interpretation.**
- The tree's device time jumps 9× (E=16), 13× (E=64) and 20× (E=256) when T doubles from 16384
  to 32768. That is exactly where dispatch leaves the tuned `_WARPS` table. A whole 32k–64k row
  per program at 16 warps means 64–128 elements per thread per tensor, consistent with
  register spilling.
- This is inferred from timings only; ncu was not used. Re-tuning warps for BLOCK_SIZE ≥ 32768
  is a library-side follow-up and was not done here, since library kernels were not to be modified.
- The serial kernels never get close. Their dependency chain is 32k–64k steps long, and with
  E ≤ 256 there are at most 32 programs.

### Task 3: packed-environment tree kernel

`packed_gae.py` uses the shipped plain-path math and the shipped `_combine`. Each program owns
a [BLOCK_E, BLOCK_T] tile, and the scan runs along axis=1. **Triton 3.0 compiles the axis=1 scan
on a 2D tile without problems.**

- **Tuning:** BLOCK_E ∈ {1,2,4,8,16,32} × num_warps ∈ {1,2,4,8}.
- **Shapes:** Grid B plus the long-T controls (512,512) and (512,4096).
- **Configs:** 523/528 validated, 0 mismatches. 5 were skipped at 512×4096: 3 by the
  32k-elements-per-warp size cap and 2 by the 120 s compile timeout. All are listed in
  `packed_results.csv` and NOTES.md.
- **512×4096 history:** the first attempt at that shape was stopped after ptxas spent more than
  25 min on one oversized config. The shape was rerun with both rules and finished in 8.3 min.
- **Outputs:** `packed_results.csv`, joined with the earlier tree / S_env / S_time / scan rows,
  and `fig_packed.pdf`.

**Device time (µs).** Packed uses its best-device config, re-measured over 5 trials. The other
columns are from `results.csv`.

| num_envs | T=8 tree / packed / S_time | T=16 | T=32 | T=64 | T=128 |
|---|---|---|---|---|---|
| 4096 | 3.7 / **1.3** / 1.4 | 3.7 / **1.4** / 2.0 | 3.7 / **1.6** / 3.0 | 3.8 / **2.1** / 4.0 | 4.0 / **3.1** / 8.0 |
| 8192 | 6.1 / **1.4** / 1.6 | 6.1 / **1.6** / 2.0 | 6.1 / **1.9** / 3.0 | 6.3 / **3.0** / 4.8 | 6.5 / **4.7** / 7.1 |
| 16384 | 11.0 / **1.6** / 1.7 | 11.1 / **1.9** / 2.2 | 11.1 / **2.7** / 3.2 | 11.2 / **4.6** / 5.1 | 11.5 / **7.6** / 9.3 |
| 32768 | 20.9 / 1.9 / **1.8** | 20.9 / **2.5** / 2.6 | 20.9 / **4.0** / 4.0 | 21.2 / 8.4 / **6.8** | 23.2 / **23.1** / 23.7 |

- **Shipped tree ÷ packed (device time):** 1.28×–10.8× on Grid B, largest at 32768×8. At
  32768×128 it is 1.00×: both are at bandwidth, and BLOCK_E=1 wins, which is the shipped geometry.
- **Long-T controls:**
  - 512×512: packed 2.3 µs vs. tree 2.2 µs, a tie with BLOCK_E=1.
  - 512×4096: packed 11.6 µs vs. tree 15.8 µs (1.36×), but with BLOCK_E=1 and 4 warps. That is
    the shipped geometry with fewer warps than `_WARPS`' 16, so it is a warp-count effect, not
    packing.
- **Full-call:** packed vs. tree at 32768 envs is 27.9–30.0 µs vs. 40.0–41.3 µs for T ≤ 64,
  which removes the tree's full-call loss there. Against S_time, packed full-call sits at
  0.93×–1.13× everywhere on Grid B, inside the tie band.

**Does packing close the gap to S_time at T ≤ 64? Yes, except at one shape.**
- At 13 of the 16 Grid B shapes with T ≤ 64, the packed tree is at least as fast as S_time on
  device time. S_time ÷ packed is 1.05×–1.86× there.
- At 32768×8 (0.95×) and 32768×32 (0.99×) it is within 5%.
- At **32768×64, S_time is still 19% faster** (6.8 vs. 8.4 µs).
- So the shipped tree's flat ~21 µs at 32768 envs was a **program-count artifact** of the
  one-env-per-program geometry, not a cost of the tree scan itself.

### Task 4: synchronized loop re-timing at (4096,128)

Same function and inputs as the benchmarks.md row (`bench_release._ref_gae`, `_make_gae`).
CUDA events, `torch.cuda.synchronize()` around every call, 20 warmup, 5 × 50, min of medians.
Files: `task4_loop_sync.py`, `task4_loop_sync.json`, `task4_diag.py`, `task4_diag.json`.

| | loop (ms) | compute_gae full-call (ms) | loop ÷ compute_gae |
|---|---|---|---|
| benchmarks.md v0.1.3 H100 (recorded) | 10.312 | 0.033 | 316.9× |
| release method `_bench_cpu`, this session | 9.123 | 0.0296 | 308.5× |
| **synchronized, CUDA events (this task)** | **7.052** | **0.0296** | **238.5×** |

- **The synchronized loop is faster than the recorded number**: 0.68× of it, and 0.77× of the
  same-session `_bench_cpu` figure.
- The diagnostic (`task4_diag.py`, fresh process, per-call `perf_counter`) shows that
  synchronizing itself costs about +6% (7.27 → 7.73 ms mean). Mean vs. min-of-medians
  differs by only about 2%.
- Neither explains the gap. The likely cause is `_bench_cpu`'s 3-call warmup and 0.5 s window
  (about 55 calls, with single calls up to 11.5 ms). This is inferred, not proven.
- Consequence: the recorded 316.9× loop speedup is **overstated relative to a synchronized
  measurement**. The synchronized figure is 238.5×.

## Follow-up results (third session, 2026-10-29)

### Task A: warp re-tuning of the shipped tree kernel at long T

**Method.**
- The shipped `gae_fused_kernel` is launched directly, exactly as the library dispatch does:
  `BLOCK_SIZE=next_pow2(T)`, `num_stages=2`, plain path with bootstrap. The only change is an
  explicit `num_warps ∈ {4, 8, 16, 32}`; `_WARPS` is not edited (`warp_worker.py`).
- The shipped value is `_WARPS.get(next_pow2(T), 16)`: 32 warps at T=16384, and 16 warps at
  T ≥ 32768, which falls back to the default because the table has no entry above 16384.
- One subprocess per shape; each config is compiled in a child process with a 120 s timeout.
- Every config is gated against the reference (atol=rtol=1e-4). **All 36 configs passed and none
  were skipped** (max abs err ≤ 1e-5).
- Best config = lowest full-call. Ratios use the compiled-scan, S_env and S_time rows of
  `longT_results.csv` (Task 2). A ratio < 1 means the tree is faster.
- The whole sweep was repeated in a second process (`warp_raw_repeat.jsonl`).
  - Device times agree within 1% for every config.
  - Full-call times agree within 6% at T ≥ 32768.
  - At T=16384, full-call jitters between 0.036 and 0.070 ms on device times of only
    0.013–0.017 ms. The full-call winner there is noise: it flips between 8 and 32 warps at
    E=16. The device-time winner at T=16384 is 16 warps at every E.
- The shipped rows reproduce the Task 2 tree device times within 2% (e.g., 0.158 vs 0.155 ms at
  16×32768).

Times are in ms, full-call / device; ratio columns show full-call (device).

| E × T | shipped warps | shipped | best warps | best re-tuned | gain (full) | tree / scan | tree / S_env | tree / S_time |
|---|---|---|---|---|---|---|---|---|
| 16 × 16384 | 32 | 0.0684 / 0.0173 | 8 | 0.0362 / 0.0146 | 1.89× | 0.34 (0.75) | 0.045 (0.019) | 0.050 (0.021) |
| 64 × 16384 | 32 | 0.0755 / 0.0178 | 8 | 0.0483 / 0.0154 | 1.56× | 0.49 (0.30) | 0.060 (0.020) | 0.065 (0.021) |
| 256 × 16384 | 32 | 0.0788 / 0.0524 | 16 | 0.0756 / 0.0428 | 1.04× | 0.27 (0.18) | 0.083 (0.048) | 0.085 (0.049) |
| 16 × 32768 | 16 | 0.1871 / 0.1578 | 32 | 0.0765 / 0.0461 | 2.45× | 0.75 (1.71) | 0.048 (0.029) | 0.053 (0.033) |
| 64 × 32768 | 16 | 0.2543 / 0.2288 | 32 | 0.1068 / 0.0648 | 2.38× | 0.76 (0.63) | 0.062 (0.038) | 0.062 (0.038) |
| 256 × 32768 | 16 | 1.0787 / 1.0551 | 32 | 0.2221 / 0.1899 | 4.86× | 0.44 (0.40) | 0.123 (0.107) | 0.127 (0.110) |
| 16 × 65536 | 16 | 0.4140 / 0.3921 | 32 | 0.2410 / 0.2192 | 1.72× | 2.27 (4.03) | 0.076 (0.070) | 0.084 (0.077) |
| 64 × 65536 | 16 | 0.6876 / 0.6655 | 32 | 0.4001 / 0.3711 | 1.72× | 1.32 (1.40) | 0.112 (0.105) | 0.114 (0.107) |
| 256 × 65536 | 16 | 2.7940 / 2.7725 | 32 | 1.6179 / 1.5975 | 1.73× | 1.55 (1.59) | 0.450 (0.448) | 0.464 (0.461) |

**Does re-tuning remove the T=32768 slowdown? Largely, but not completely.**
- 32 warps wins every shape at T ≥ 32768. At T=32768 it is 2.4–4.9× faster than the shipped 16 warps (1.7× at T=65536).
- Going from T=16384 to T=32768 (device time, best config at each T), the cost now grows
  3.5× / 4.7× / 4.4× at E = 16 / 64 / 256. With the shipped table it grew 9.1× / 12.9× / 20×.
- That is still more than the 2× that doubling T alone would explain. A smaller cliff remains,
  which a `num_warps` override cannot remove: 32 warps is the hardware maximum of 1024 threads.
- The likely cause is per-thread register pressure (32 elements per thread per tensor at
  T=32768). That is inferred: no ptxas or spill statistics were collected.

**Does the tree beat the compiled scan again at T ≥ 32768? At T=32768 yes; at T=65536 no.**
- **T=32768:**
  - Full-call: the re-tuned tree beats the scan at all three E (0.44–0.76× the scan's time).
  - Device time: it beats the scan at E=64 and E=256 (0.63×, 0.40×), but not at E=16 (1.71×).
    At E=16 the scan's full-call (0.10 ms) is dominated by launch overhead.
- **T=65536:** the scan stays faster at every E, by 1.3–2.3× full-call and 1.4–4.0× device.
- **Against S_env and S_time:** the re-tuned tree is faster at every shape, by 2.2–22× full-call.

Files: `warp_worker.py`, `warp_run.py`, `warp_report.py`, `warp_raw.jsonl`,
`warp_raw_repeat.jsonl`, `warp_run_log.txt`, `warp_run_repeat_log.txt`, `warp_retune.csv`.

### Task B: synchronized loop re-measurement

**Method.**
- Same as Task 4: CUDA events with `torch.cuda.synchronize()` before the start event and after
  the end event of every call; 20 warmup, 5 × 50, min of trial medians.
- One subprocess per job (`loop_sync_worker.py`). Each loop is the exact function, inputs and
  arguments `tests/bench_release.py` passes to `_bench_cpu` for the "loop gpu" column:
  - `_ref_gae`, `_ref_vtrace`, `_ref_retrace`, `_ref_lambda`, `_ref_disc`, `_ref_traces`;
  - `reference_episodic_prefix_sum` for the prefix sum;
  - inputs from the `_make_*` helpers (seed 0).
- Each loop output was first gated (atol=rtol=1e-4) against the shipped library op on the same
  inputs. **All 9 passed** (max abs err ≤ 1e-5).
- The same-session `_bench_cpu` (release method) value was also recorded.
- **Tree time source:** the "triton full-call (ms)" column of that algorithm's own table in
  benchmarks.md, section "v0.1.3 – 2026-08-08 – NVIDIA H100 80GB HBM3" (exact line in the last
  column). Every algorithm's table has a 4096×128 row, so the Production-regime table was not
  needed.
- **Speedups:**
  - Recorded speedup = that row's "vs loop" column.
  - Synced speedup = synchronized loop ÷ recorded tree full-call. The tree times have only
    3-decimal precision (±1.5% at 0.033 ms).
- The first run hit a bug in the worker's error-printing line: V-Trace returns a tuple. The
  correctness gate itself had passed. The line was fixed and all 9 jobs were rerun together;
  that rerun is what is reported.

| algorithm | E × T | loop, synced (ms) | loop, `_bench_cpu` same session (ms) | loop, recorded (ms) | tree full-call, recorded (ms) | speedup synced | speedup recorded | synced / recorded | tree source |
|---|---|---|---|---|---|---|---|---|---|
| GAE | 4096 × 128 | 7.455 | 8.510 | 10.312 | 0.033 | 225.9× | 316.9× | 0.71 | benchmarks.md:328 (GAE) |
| V-Trace | 4096 × 128 | 2.872 | 2.992 | 4.094 | 0.043 | 66.8× | 96.2× | 0.69 | benchmarks.md:362 (V-Trace) |
| Retrace(λ) | 4096 × 128 | 2.653 | 2.880 | 3.908 | 0.054 | 49.1× | 72.2× | 0.68 | benchmarks.md:396 (Retrace) |
| λ-returns | 4096 × 128 | 6.121 | 6.400 | 9.261 | 0.035 | 174.9× | 267.0× | 0.66 | benchmarks.md:430 (λ-returns) |
| Discounted returns | 4096 × 128 | 4.197 | 4.510 | 5.662 | 0.034 | 123.4× | 167.4× | 0.74 | benchmarks.md:464 (Discounted) |
| Eligibility traces | 4096 × 128 | 4.317 | 4.433 | 5.904 | 0.031 | 139.3× | 191.0× | 0.73 | benchmarks.md:498 (Eligibility traces) |
| Episodic prefix sum | 4096 × 128 | 3.343 | 3.516 | 4.933 | 0.030 | 111.4× | 167.0× | 0.67 | benchmarks.md:515 (Episodic prefix sum) |
| GAE | 64 × 512 | 30.059 | 31.797 | 45.968 | 0.031 | 969.7× | 1464.3× | 0.66 | benchmarks.md:321 (GAE) |
| GAE | 512 × 4096 | 254.413 | 243.894 | 330.190 | 0.044 | 5782.1× | 7570.4× | 0.76 | benchmarks.md:325 (GAE) |

**Findings.**
- **Every recorded loop speedup is overstated relative to a synchronized measurement**, by a
  consistent factor: the synced speedup is 0.66–0.76× the recorded one (all 9 rows). The
  ordering of algorithms is unchanged.
- **Most of the gap is not caused by synchronization.**
  - In this session, the release method (`_bench_cpu`) gives 0.96–1.14× the synchronized time
    (synced / `_bench_cpu` = 0.88–1.04). Both are well below the recorded values.
  - So the loop, which is bound by Python and kernel-launch dispatch, simply ran faster in this
    session than in the 2026-08-08 sweep.
  - Task 4 saw a larger same-session gap at GAE 4096×128 (7.05 vs 9.12 ms). That number
    differs between sessions too.
- The cause of the session-to-session drift (host CPU state, driver, clocks) was not
  investigated.
- For the paper, the synchronized figures are the defensible ones. For example, GAE at
  4096×128 is about 226× rather than 317×, and the 512×4096 headline is about 5.8k× rather
  than 7.6k×.

Files: `loop_sync_worker.py`, `loop_sync_run.py`, `loop_sync_report.py`, `loop_sync_raw.jsonl`,
`loop_sync_run_log.txt`, `loop_sync.csv`.

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
| `task1_provenance.md` | Task 1 provenance check |
| `longT_*.py`, `longT_results.csv`, `longT_sweep_serial.csv`, `longT_raw.jsonl`, `longT_scan_rerun.jsonl`, `longT_report_stats.txt`, `fig_longT.pdf` | Task 2 long-T sweep |
| `packed_gae.py`, `packed_*.py`, `packed_results.csv`, `packed_sweep.csv`, `packed_raw.jsonl`, `packed_report_stats.txt`, `fig_packed.pdf` | Task 3 packed tree kernel |
| `task4_loop_sync.py`/`.json`, `task4_diag.py`/`.json` | Task 4 synchronized loop re-timing |
| `warp_*.py`, `warp_raw.jsonl`, `warp_raw_repeat.jsonl`, `warp_run*_log.txt`, `warp_retune.csv` | Task A warp re-tuning |
| `loop_sync_*.py`, `loop_sync_raw.jsonl`, `loop_sync_run_log.txt`, `loop_sync.csv` | Task B synchronized loop re-measurement |
| `NOTES.md` | Step 0 paths and every deviation from the task spec, incl. follow-up skips |
