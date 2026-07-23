# H100 Report: rl-triton GAE vs. PufferLib Advantage Kernel

Standalone hardware micro-benchmark comparing rl-triton's fused GAE kernel against
PufferLib's real advantage-calculation CUDA kernel, isolated from env stepping and network
forward passes. Produced by `tests/benchmark_gae_vs_pufferlib.py`. Two regimes, two figures:

- **Production regime** — `gae_performance_crossover.png` — moderate-to-long on-policy
  rollouts (`seq_len` 128-4096, `num_envs` 128-8192).
- **Massively-parallel-sim regime** — `gae_performance_short_horizon.png` — Isaac Gym/Isaac
  Lab-style GPU simulation, the opposite aspect ratio (`seq_len` 8-128, `num_envs` 4096-32768).

**Naming correction.** The kernel does not live at `pufferlib/torch_pufferl.py` /
`puff_advantage_row` — those names do not exist in the published `pufferlib==3.0.0` package
(verified against the PyPI sdist). The real entry points are `compute_puff_advantage()` in
`pufferlib/pufferl.py` (Python wrapper) and `puff_advantage_row_cuda()` in
`pufferlib/extensions/cuda/pufferlib.cu` — a genuine hand-written CUDA kernel (one thread per
environment row, sequential O(T) scan within the thread), not a Python loop. Compiled from
vendored, sha256-pinned source (`tests/pufferlib_ext/`) rather than `pip install pufferlib`,
which also builds raylib/Box2D/dozens of unrelated env bindings and mutates process-wide state
(`SIGINT` handler, warning filters) on import.

## Environment

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM5) |
| torch | 2.4.1+cu124 (CUDA 12.4) |
| triton | 3.0.0 |
| PufferLib | 3.0.0, vendored `pufferlib.cpp`/`pufferlib.cu`, JIT-compiled |
| dtype | float32 |
| gamma / lambda | 0.99 / 0.95 |
| termination probability | 5% per step |

## Step 0: semantic equivalence

Both are backward scans of the standard GAE recurrence `A[t] = delta[t] + decay[t]*A[t+1]`,
but they are not drop-in equivalent:

- **Buffer convention.** PufferLib's `rewards`/`dones` are indexed one slot ahead of `values`
  (pairs `PR[t+1]`/`PD[t+1]` with `V[t]`/`V[t+1]`) — PufferLib's own source carries a TODO
  ("t_next works and t doesn't. Check original formula"). The exact index mapping was derived
  (`PR[1:] = R[:-1]`, `PD[1:] = D[:-1]`) and verified to reproduce rl-triton's per-step
  delta/decay exactly. Costs nothing at benchmark time — a real PufferLib caller's rollout
  buffer is already laid out this way.
- **Row-length capability gap.** A length-T buffer gives PufferLib only **T-1** usable
  advantage estimates — row T-1 is structurally uncomputable (its delta would need
  `PR[T]`/`PD[T]`, one slot past the buffer). rl-triton produces a genuine T-th estimate via an
  explicit (default zero) boundary bootstrap. Not a bug on either side, but it means a literal
  `max|full_T_output_A - full_T_output_B|` is the wrong equivalence check.
- **terminated vs. truncated.** PufferLib has exactly one `dones` flag per step — no
  distinction between "true episode end" and "time-limit cutoff with true continuation value",
  and no `bootstrap_values` equivalent. rl-triton's interior-truncation path
  (`HAS_TRUNCATIONS=True`) has **no PufferLib counterpart to benchmark against at all** — this
  is why the sweeps below cover only the overlapping regime (terminations only, on-policy).
- **No V-trace equivalent on rl-triton's GAE side.** PufferLib's `importance`/`rho_clip`/
  `c_clip` are pinned to `1.0` (no-op) so it degenerates to plain GAE.
- Neither side normalizes advantages or returns them fused with targets/returns; both return a
  raw `[num_envs, seq_len]` tensor. Layout is natively `[num_envs, seq_len]` on both sides — no
  transpose needed.
- **Equivalence does not depend on seq_len.** The math above holds for any T>=3; re-checked
  explicitly at the massively-parallel-sim regime's shapes below rather than assumed to carry
  over.

**Numerical check** (independent sequential reference, not either kernel), covering both
regimes' shapes:

| num_envs | seq_len | max\|PufferLib − ref\| | Triton vs. PufferLib directly (uncontaminated cols) |
|---|---|---|---|
| 64 | 256 | 0.000e+00 (exact) | max 3.34e-06, mean 2.65e-07 |
| 32 | 4096 | 0.000e+00 (exact) | max 3.81e-06, mean 2.82e-07 |
| 37 | 129 | 0.000e+00 (exact) | max 2.09e-06, mean 2.58e-07 |
| 4096 | 8 | 0.000e+00 (exact) | max 1.43e-06, mean 9.07e-08 |
| 8192 | 16 | 0.000e+00 (exact) | max 2.38e-06, mean 1.50e-07 |
| 16384 | 64 | 0.000e+00 (exact) | max 4.05e-06, mean 2.44e-07 |
| 2048 | 128 | 0.000e+00 (exact) | max 5.25e-06, mean 2.64e-07 |

PufferLib matches the reference **bit-exactly** (0.0 diff) at every tested shape, including
`seq_len=8` (the smallest swept value) and `num_envs=16384` — both do a plain sequential,
left-to-right accumulation in the same order. rl-triton's Triton kernel computes the identical
recurrence via `tl.associative_scan`, a **log-depth parallel tree reduction** — a different
summation order, so it is **not bit-identical** to PufferLib or the reference (float32 +/* is
not associative). Measured max abs diff ~1-5e-6 across all shapes tested, confirmed
deterministic run-to-run (0.0 diff between two Triton runs on identical inputs, so this is
rounding, not kernel nondeterminism). Both are numerically correct implementations of the same
quantity, agreeing far inside any tolerance that matters for RL advantage estimates, in both
regimes.

**Conclusion:** benchmark only the overlapping regime — on-policy, no interior truncations,
columns `0..T-2` — at both aspect ratios.

## Harness

`tests/bench_utils.py` conventions (`_warmup_gpu`/`_bench_gpu`, min-of-11-trial-medians,
100 iterations/trial, per-config warmup at the exact timed shape), extended with an amortized
variant (100 calls in one timed CUDA-event region, strips per-call dispatch/sync overhead),
`torch.profiler` device-only time, and kernel-launch counts. Bandwidth formula: Triton = 4 x N
x T x 4 bytes (rewards, values, terminateds reads + out write — the shifted `values` re-load
shares L1/L2 lines with the primary load per this repo's existing double-load investigation,
not counted twice); PufferLib = 5 x N x T x 4 bytes (values, rewards, dones, importance reads +
advantages write). "dev speedup" = PufferLib device time / Triton device time (torch.profiler,
CUDA-activity only, no dispatch overhead) — reported separately from wall-clock speedup because
the two can and do disagree (see the massively-parallel-sim regime below).

---

## Regime 1: production (`num_envs` 128-8192 x `seq_len` 128-4096)

| num_envs | seq_len | triton (ms) | puffer (ms) | speedup | dev speedup | triton amort (ms) | puffer amort (ms) | triton GB/s (%peak) | puffer GB/s (%peak) | triton dev (µs) | puffer dev (µs) | launches (tri/puf) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 0.0372 | 0.0777 | 2.09x | 44.5x | 0.0230 | 0.0671 | 7.0 (0.21%) | 4.2 (0.13%) | 1.44 | 63.90 | 1/2 |
| 128 | 512 | 0.0415 | 0.2619 | 6.31x | 156.7x | 0.0251 | 0.2518 | 25.3 (0.75%) | 5.0 (0.15%) | 1.57 | 246.78 | 1/2 |
| 128 | 1024 | 0.0416 | 0.5027 | 12.08x | 258.1x | 0.0255 | 0.4984 | 50.4 (1.50%) | 5.2 (0.16%) | 1.88 | 484.87 | 1/2 |
| 128 | 2048 | 0.0410 | 1.0328 | 25.18x | 327.3x | 0.0249 | 1.0166 | 102.2 (3.05%) | 5.1 (0.15%) | 3.05 | 999.84 | 1/2 |
| 128 | 4096 | 0.0422 | 1.9457 | 46.13x | 398.9x | 0.0255 | 1.9524 | 198.9 (5.94%) | 5.4 (0.16%) | 4.86 | 1938.98 | 1/2 |
| 512 | 128 | 0.0414 | 0.1324 | 3.20x | 51.3x | 0.0253 | 0.1182 | 25.3 (0.76%) | 9.9 (0.30%) | 2.24 | 115.23 | 1/2 |
| 512 | 512 | 0.0417 | 0.4811 | 11.53x | 202.6x | 0.0248 | 0.4665 | 100.5 (3.00%) | 10.9 (0.33%) | 2.27 | 460.25 | 1/2 |
| 512 | 1024 | 0.0416 | 0.9353 | 22.47x | 268.4x | 0.0253 | 0.9181 | 201.5 (6.01%) | 11.2 (0.33%) | 3.42 | 917.72 | 1/2 |
| 512 | 2048 | 0.0420 | 1.8393 | 43.81x | 227.2x | 0.0249 | 1.8222 | 399.6 (11.93%) | 11.4 (0.34%) | 8.05 | 1829.72 | 1/2 |
| 512 | 4096 | 0.0476 | 3.8776 | 81.49x | 249.8x | 0.0258 | 3.8649 | 705.2 (21.05%) | 10.8 (0.32%) | 15.45 | 3859.70 | 1/2 |
| 2048 | 128 | 0.0417 | 0.1473 | 3.53x | 22.1x | 0.0253 | 0.1331 | 100.5 (3.00%) | 35.6 (1.06%) | 5.93 | 130.77 | 1/2 |
| 2048 | 512 | 0.0420 | 0.5060 | 12.03x | 102.1x | 0.0256 | 0.4917 | 399.0 (11.91%) | 41.4 (1.24%) | 4.75 | 484.90 | 1/2 |
| 2048 | 1024 | 0.0432 | 0.9529 | 22.07x | 94.6x | 0.0257 | 0.9423 | 777.3 (23.20%) | 44.0 (1.31%) | 9.93 | 939.09 | 1/2 |
| 2048 | 2048 | 0.0642 | 1.9381 | 30.19x | 57.9x | 0.0346 | 1.9319 | 1045.4 (31.21%) | 43.3 (1.29%) | 33.34 | 1929.66 | 1/2 |
| 2048 | 4096 | 0.1003 | 3.8347 | 38.24x | 53.7x | 0.0706 | 3.8326 | 1338.3 (39.95%) | 43.8 (1.31%) | 71.33 | 3830.50 | 1/2 |
| 8192 | 128 | 0.0508 | 0.1333 | 2.63x | 5.9x | 0.0260 | 0.1201 | 330.6 (9.87%) | 157.3 (4.70%) | 19.98 | 117.66 | 1/2 |
| 8192 | 512 | 0.0567 | 0.4930 | 8.69x | 20.1x | 0.0257 | 0.4847 | 1183.5 (35.33%) | 170.2 (5.08%) | 23.94 | 481.60 | 1/2 |
| 8192 | 1024 | 0.0770 | 0.9548 | 12.40x | 20.5x | 0.0478 | 0.9471 | 1742.5 (52.02%) | 175.7 (5.25%) | 45.97 | 943.55 | 1/2 |
| 8192 | 2048 | 0.1504 | 1.9045 | 12.66x | 15.8x | 0.1206 | 1.8961 | 1784.8 (53.28%) | 176.2 (5.26%) | 120.18 | 1892.82 | 1/2 |
| 8192 | 4096 | 0.2686 | 3.8829 | 14.45x | 16.0x | 0.2371 | 3.8702 | 1998.5 (59.66%) | 172.8 (5.16%) | 242.50 | 3872.61 | 1/2 |

**Launch counts.** PufferLib is always 2 (its advantage kernel + a `torch.zeros` memset for the
output buffer, matching real PufferLib usage in `pufferl.py`); Triton is always 1
(`torch.empty_like`, no memset needed — the kernel writes every position).

**Monotonicity gate**: FAILED with 2 small violations (puffer_ms, seq_len=128 and seq_len=512,
both at the num_envs 2048->8192 transition, 2.6-9.5%) — consistent across repeated runs with
this repo's `tests/h100_num_envs_sweep_report.md` finding that small grids don't yet saturate
the H100's 132 SMs, so growing num_envs there can be absorbed by idle SMs at near-zero extra
wall-clock cost. Not a compile leak: each cell warms up fresh at its own exact shape, and
re-running at higher iteration/trial counts across multiple runs shuffled *which* specific
cells tripped the 2% threshold (11 -> 7 -> 4 -> 3 -> 2 violations across five runs) rather than
reproducing the same ones — the signature of measurement noise, not a deterministic bug. GPU
was confirmed idle/unthrottled via `nvidia-smi` (P0, 0% other-process utilization, no
thermal/power throttling) during these runs.

**Crossover**: none. Triton is faster (wall-clock) at every one of the 20 cells, from (128,128)
at 2.09x up to (512,4096) at 81.49x. No device-time inversion on this axis either.

**Mechanism.** PufferLib is launch/latency-bound at small T, then flatly bandwidth-starved at
large T — achieved bandwidth tops out around 175-180 GB/s regardless of num_envs or seq_len, a
small fraction of the H100's ~3350 GB/s HBM3 datasheet peak, consistent with one CUDA thread
per row doing a fully sequential O(T) scan (a row's completion latency scales linearly with T;
threads within a warp/block cannot cooperate to hide memory latency the way a block-wide
reduction can). Triton is dispatch-floor-flat at small problem sizes (device-only time 1.4-243µs
is far below the ~25-50µs of fixed CUDA-event/sync dispatch overhead, which is why wall-clock
barely moves across the smallest configs), then genuinely bandwidth-bound at the largest
configs (num_envs>=2048, seq_len>=2048), where achieved bandwidth climbs to 1000-2000+ GB/s
(up to 59.7% of peak at (8192, 4096)).

See `gae_performance_crossover.png` — one line pair per num_envs; PufferLib's four dashed lines
nearly overlap regardless of num_envs, visually confirming the O(T)-dominated,
num_envs-insensitive read above.

---

## Regime 2: massively-parallel-sim (`num_envs` 4096-32768 x `seq_len` 8-128)

**Motivation.** Isaac Gym/Isaac Lab-style GPU simulation runs the opposite aspect ratio from
regime 1: thousands of envs, horizon in the tens of steps. Prior work in this repo
(`tests/h100_short_horizon_l2_retrace_ppo_report.md`, Experiment 1) found rl-triton's
device-only time inverts below 1x against `torch.compile`'s vectorized GAE baseline at
`num_envs=16384`, `seq_len>=64` — one program per env means more grid waves per SM as num_envs
grows, while the competing kernel sits on a flatter per-launch floor. That finding was against
`torch.compile`, not PufferLib, whose per-row-sequential-CUDA-thread design is structurally
different (and, at these tiny T, structurally very cheap). This regime re-tests the same
question against the real PufferLib kernel rather than assuming the prior finding transfers.

| num_envs | seq_len | triton (ms) | puffer (ms) | speedup | dev speedup | triton amort (ms) | puffer amort (ms) | triton GB/s (%peak) | puffer GB/s (%peak) | triton dev (µs) | puffer dev (µs) | launches (tri/puf) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 8 | 0.0430 | 0.0253 | **0.59x** | **0.50x** | 0.0257 | 0.0157 | 12.2 (0.36%) | 25.9 (0.77%) | 7.68 | 3.82 | 1/2 |
| 4096 | 16 | 0.0426 | 0.0265 | **0.62x** | 1.17x | 0.0258 | 0.0159 | 24.6 (0.73%) | 49.5 (1.48%) | 7.97 | 9.32 | 1/2 |
| 4096 | 32 | 0.0420 | 0.0461 | 1.10x | 3.66x | 0.0254 | 0.0319 | 49.9 (1.49%) | 56.9 (1.70%) | 8.12 | 29.68 | 1/2 |
| 4096 | 64 | 0.0413 | 0.0750 | 1.82x | 5.88x | 0.0251 | 0.0610 | 101.6 (3.03%) | 69.9 (2.09%) | 9.98 | 58.72 | 1/2 |
| 4096 | 128 | 0.0423 | 0.1420 | 3.36x | 11.81x | 0.0253 | 0.1282 | 198.3 (5.92%) | 73.9 (2.21%) | 10.58 | 124.94 | 1/2 |
| 8192 | 8 | 0.0446 | 0.0252 | **0.56x** | **0.27x** | 0.0257 | 0.0156 | 23.5 (0.70%) | 52.0 (1.55%) | 14.29 | 3.88 | 1/2 |
| 8192 | 16 | 0.0458 | 0.0265 | **0.58x** | **0.64x** | 0.0256 | 0.0159 | 45.8 (1.37%) | 98.8 (2.95%) | 14.84 | 9.46 | 1/2 |
| 8192 | 32 | 0.0452 | 0.0461 | 1.02x | 1.98x | 0.0254 | 0.0322 | 92.7 (2.77%) | 113.6 (3.39%) | 15.07 | 29.87 | 1/2 |
| 8192 | 64 | 0.0494 | 0.0752 | 1.52x | 3.16x | 0.0258 | 0.0614 | 169.8 (5.07%) | 139.4 (4.16%) | 18.72 | 59.08 | 1/2 |
| 8192 | 128 | 0.0502 | 0.1337 | 2.67x | 5.93x | 0.0258 | 0.1205 | 334.4 (9.98%) | 156.8 (4.68%) | 19.90 | 117.97 | 1/2 |
| 16384 | 8 | 0.0557 | 0.0251 | **0.45x** | **0.15x** | 0.0258 | 0.0156 | 37.6 (1.12%) | 104.5 (3.12%) | 27.52 | 4.04 | 1/2 |
| 16384 | 16 | 0.0563 | 0.0260 | **0.46x** | **0.34x** | 0.0259 | 0.0157 | 74.6 (2.23%) | 201.5 (6.02%) | 28.59 | 9.64 | 1/2 |
| 16384 | 32 | 0.0576 | 0.0463 | **0.80x** | 1.04x | 0.0265 | 0.0325 | 145.6 (4.35%) | 226.6 (6.76%) | 29.01 | 30.29 | 1/2 |
| 16384 | 64 | 0.0644 | 0.0753 | 1.17x | 1.65x | 0.0335 | 0.0621 | 260.3 (7.77%) | 278.6 (8.32%) | 36.27 | 59.88 | 1/2 |
| 16384 | 128 | 0.0710 | 0.1368 | 1.93x | 3.00x | 0.0399 | 0.1258 | 472.8 (14.11%) | 306.7 (9.15%) | 40.86 | 122.38 | 1/2 |
| 32768 | 8 | 0.0784 | 0.0252 | **0.32x** | **0.09x** | 0.0469 | 0.0158 | 53.5 (1.60%) | 208.2 (6.21%) | 54.02 | 5.10 | 1/2 |
| 32768 | 16 | 0.0802 | 0.0266 | **0.33x** | **0.18x** | 0.0491 | 0.0159 | 104.6 (3.12%) | 394.8 (11.78%) | 56.11 | 10.22 | 1/2 |
| 32768 | 32 | 0.0815 | 0.0466 | **0.57x** | **0.55x** | 0.0505 | 0.0334 | 205.9 (6.15%) | 450.1 (13.44%) | 56.97 | 31.25 | 1/2 |
| 32768 | 64 | 0.1026 | 0.0781 | **0.76x** | **0.84x** | 0.0718 | 0.0668 | 327.2 (9.77%) | 537.0 (16.03%) | 75.90 | 63.62 | 1/2 |
| 32768 | 128 | 0.1130 | 0.1453 | 1.29x | 1.56x | 0.0835 | 0.1371 | 593.9 (17.73%) | 577.4 (17.24%) | 85.10 | 133.06 | 1/2 |

(bold = PufferLib faster, either wall-clock or device-only)

**Monotonicity gate**: FAILED with 1 violation (puffer_ms, seq_len=128, num_envs 4096->8192,
-5.8%) — same benign num_envs-saturation mechanism as regime 1. None of this regime's seq_lens
(8-128) hit a `_WARPS` tuning-table boundary (`{512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32,
16384: 32}` — all below the table's smallest key), so the seq_len-axis mechanism identified in
regime 1 does not apply here; the script's diagnostic correctly detected zero matches and
attributed the violation to num_envs saturation instead.

**Crossover** (wall-clock): Triton overtakes PufferLib only once `seq_len>=32` at N=4096/8192,
`seq_len>=64` at N=16384, and not until `seq_len=128` at N=32768. **Device-time inversion is
broader still**: PufferLib is faster in raw kernel time at every N once `seq_len<=16`, and at
N=32768 the inversion persists through `seq_len=64` (device speedup 0.84x).

### The three questions

**Q1 — Does PufferLib's time stay independent of num_envs at short T?** Yes, confirmed at every
tested seq_len:

| seq_len | puffer ms range (max/min) | triton ms range (max/min) |
|---|---|---|
| 8 | 0.0251-0.0253 (1.01x) | 0.0430-0.0784 (1.82x) |
| 16 | 0.0260-0.0266 (1.02x) | 0.0426-0.0802 (1.88x) |
| 32 | 0.0461-0.0466 (1.01x) | 0.0420-0.0815 (1.94x) |
| 64 | 0.0750-0.0781 (1.04x) | 0.0413-0.1026 (2.48x) |
| 128 | 0.1337-0.1453 (1.09x) | 0.0423-0.1130 (2.67x) |

(range across num_envs = [4096, 8192, 16384, 32768])

PufferLib's cost is essentially T-bound and num_envs-insensitive (max/min <=1.09x across a 8x
range of num_envs, every seq_len). Triton's grows with num_envs (up to 2.67x max/min at the
same T) — this is exactly the one-program-per-env grid-scaling cost, and it is why PufferLib
overtakes Triton at high N (see `gae_performance_short_horizon.png`: PufferLib's dashed lines
are visually flat; Triton's solid lines climb and cross them).

**Q2 — Where does rl-triton's one-program-per-env design stop paying, vs. PufferLib?** The
prior torch.compile comparison inverted (device-only) at `num_envs=16384, seq_len>=64`. Against
PufferLib the inversion is worse on both axes:

- Device-time inversion: `num_envs>=4096` once `seq_len<=16`; at `num_envs=32768` it persists
  through `seq_len=64`.
- Wall-clock inversion (broader than device, since PufferLib's low dispatch overhead — 2
  launches, both cheap at this scale — doesn't cost it here the way it might elsewhere):
  11/20 cells, up to `(16384, 32)` and `(32768, 64)`.

Triton only reliably wins both metrics once `seq_len>=64` at `num_envs<=8192`, or `seq_len=128`
regardless of num_envs up to 16384. At `num_envs=32768` Triton doesn't win wall-clock until
`seq_len=128`.

**Q3 — bandwidth-bound, launch-bound, or occupancy-bound?** Both sides carry a real
single-call/amortized dispatch premium in this regime (1.1x-2.2x) and both stay under ~18% of
H100 peak bandwidth everywhere swept — this regime is dispatch/occupancy-sensitive on both
sides, not bandwidth-bound. The distinguishing signal is *how* Triton's dispatch ratio moves:
it grows with num_envs at fixed T (1.67x -> 2.16x from N=4096->16384 at T=8, before easing back
down at N=32768 as the amortized cost itself starts dominated by real grid-wave work) — pointing
at grid-launch/wave overhead scaling with program count, i.e. occupancy/launch-bound from the
one-program-per-env design, not classic HBM saturation. PufferLib's dispatch ratio is flatter
and its %-of-peak bandwidth climbs faster with num_envs (up to 17.2% at (32768,128) vs.
Triton's 17.7% at the same cell — parity only at the largest, longest-horizon cell in this
regime) — consistent with many cheap, short-lived per-row threads packing the GPU efficiently
at this problem size, unlike Triton's one-program-per-env grid.

### Verdict: massively-parallel-sim regime

**PufferLib wins 11/20 wall-clock cells (55%) in this regime — the expected result given
O(log T) vs O(T) collapses at small T and this repo's known one-program-per-env grid
degradation at high env counts, confirmed here against a second, independent baseline (not just
`torch.compile`).** Triton's advantage in this regime is confined to longer horizons
(`seq_len>=64`) combined with moderate env counts (`num_envs<=8192`); at Isaac-Gym-scale env
counts (16384-32768) with short horizons (`seq_len<=32`), PufferLib's per-row sequential design
— cheap enough at these tiny T that its O(T) cost never matters — beats rl-triton's grid design
outright, on both device time and wall clock.

See `gae_performance_short_horizon.png` (x-axis num_envs, one line pair per seq_len, log-log):
PufferLib's dashed lines are flat; Triton's solid lines rise and cross them at different points
depending on T, visually reproducing the crossover thresholds above.

---

## Overall verdict

Two regimes, two different answers — reported honestly rather than picking the one that flatters
either kernel:

- **Production regime** (`seq_len>=128`, `num_envs<=8192`): Triton wins all 20/20 cells,
  2.09x-81.49x, no crossover.
- **Massively-parallel-sim regime** (`seq_len<=128`, `num_envs>=4096`): PufferLib wins 11/20
  wall-clock cells, up to ~3x, concentrated at short horizons and high env counts.

Neither result is the product of handicapping either side — PufferLib's kernel is real,
hand-written CUDA, JIT-compiled unmodified from the official source, and both kernels were
verified numerically equivalent (Step 0) before any timing was taken, at the shapes used in
*both* regimes. The boundary between the two outcomes is real and load-bearing for anyone
choosing between these kernels for a specific workload: rl-triton's one-program-per-env grid
design is the right choice for long-horizon on-policy rollouts, and the wrong one for
short-horizon massively-parallel simulation at current env counts — a genuine, actionable
limitation of the current grid design, not a benchmarking artifact.
