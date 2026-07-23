# H100 Report: rl-triton GAE vs. PufferLib Advantage Kernel

Standalone hardware micro-benchmark comparing rl-triton's fused GAE kernel against
PufferLib's real advantage-calculation CUDA kernel, isolated from env stepping and network
forward passes. Produced by `tests/benchmark_gae_vs_pufferlib.py`; figure at
`gae_performance_crossover.png`.

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
  is why the sweep below covers only the overlapping regime (terminations only, on-policy).
- **No V-trace equivalent on rl-triton's GAE side.** PufferLib's `importance`/`rho_clip`/
  `c_clip` are pinned to `1.0` (no-op) so it degenerates to plain GAE.
- Neither side normalizes advantages or returns them fused with targets/returns; both return a
  raw `[num_envs, seq_len]` tensor. Layout is natively `[num_envs, seq_len]` on both sides — no
  transpose needed.

**Numerical check** (independent sequential reference, not either kernel):

| num_envs | seq_len | max\|PufferLib − ref\| | Triton vs. PufferLib directly (uncontaminated cols) |
|---|---|---|---|
| 64 | 256 | 0.000e+00 (exact) | max 3.34e-06, mean 2.65e-07 |
| 32 | 4096 | 0.000e+00 (exact) | max 3.81e-06, mean 2.82e-07 |
| 37 | 129 | 0.000e+00 (exact) | max 2.09e-06, mean 2.58e-07 |

PufferLib matches the reference **bit-exactly** (0.0 diff) — both do a plain sequential,
left-to-right accumulation in the same order. rl-triton's Triton kernel computes the identical
recurrence via `tl.associative_scan`, a **log-depth parallel tree reduction** — a different
summation order, so it is **not bit-identical** to PufferLib or the reference (float32 +/* is
not associative). Measured max abs diff ~3-4e-6, confirmed deterministic run-to-run (0.0 diff
between two Triton runs on identical inputs, so this is rounding, not kernel nondeterminism).
Both are numerically correct implementations of the same quantity, agreeing far inside any
tolerance that matters for RL advantage estimates.

**Conclusion:** benchmark only the overlapping regime — on-policy, no interior truncations,
columns `0..T-2`.

## Step 1-2: harness and sweep

`tests/bench_utils.py` conventions (`_warmup_gpu`/`_bench_gpu`, min-of-5-trial-medians,
per-config warmup at the exact timed shape), extended with an amortized variant (100 calls in
one timed CUDA-event region, strips per-call dispatch/sync overhead), `torch.profiler`
device-only time, and kernel-launch counts. `n_iter=100`, `n_trials=11`.

Sweep: `num_envs` in [128, 512, 2048, 8192] x `seq_len` in [128, 512, 1024, 2048, 4096], 20
cells. Bandwidth formula: Triton = 4 x N x T x 4 bytes (rewards, values, terminateds reads +
out write — the shifted `values` re-load shares L1/L2 lines with the primary load per this
repo's existing double-load investigation, not counted twice); PufferLib = 5 x N x T x 4 bytes
(values, rewards, dones, importance reads + advantages write).

| num_envs | seq_len | triton (ms) | puffer (ms) | speedup | triton amort (ms) | puffer amort (ms) | triton GB/s | puffer GB/s | triton dev (µs) | puffer dev (µs) | triton launches | puffer launches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 0.0354 | 0.0760 | 2.15x | 0.0230 | 0.0662 | 7.4 | 4.3 | 1.43 | 63.36 | 1.0 | 2.0 |
| 128 | 512 | 0.0421 | 0.2605 | 6.18x | 0.0257 | 0.2500 | 24.9 | 5.0 | 1.57 | 243.10 | 1.0 | 2.0 |
| 128 | 1024 | 0.0412 | 0.5113 | 12.42x | 0.0254 | 0.4971 | 51.0 | 5.1 | 1.89 | 482.55 | 1.0 | 2.0 |
| 128 | 2048 | 0.0415 | 0.9773 | 23.53x | 0.0253 | 0.9884 | 101.0 | 5.4 | 3.05 | 958.56 | 1.0 | 2.0 |
| 128 | 4096 | 0.0421 | 1.9404 | 46.08x | 0.0255 | 1.9696 | 199.2 | 5.4 | 4.85 | 1954.48 | 1.0 | 2.0 |
| 512 | 128 | 0.0413 | 0.1317 | 3.19x | 0.0254 | 0.1171 | 25.4 | 10.0 | 2.23 | 114.49 | 1.0 | 2.0 |
| 512 | 512 | 0.0417 | 0.4837 | 11.61x | 0.0253 | 0.4695 | 100.7 | 10.8 | 2.27 | 461.91 | 1.0 | 2.0 |
| 512 | 1024 | 0.0417 | 0.9276 | 22.23x | 0.0255 | 0.9117 | 201.0 | 11.3 | 3.42 | 909.37 | 1.0 | 2.0 |
| 512 | 2048 | 0.0418 | 1.8149 | 43.46x | 0.0255 | 1.8011 | 401.8 | 11.6 | 8.04 | 1804.99 | 1.0 | 2.0 |
| 512 | 4096 | 0.0472 | 3.7783 | 80.10x | 0.0253 | 3.7636 | 711.4 | 11.1 | 15.33 | 3760.27 | 1.0 | 2.0 |
| 2048 | 128 | 0.0412 | 0.1470 | 3.56x | 0.0252 | 0.1327 | 101.7 | 35.7 | 5.95 | 130.65 | 1.0 | 2.0 |
| 2048 | 512 | 0.0420 | 0.4769 | 11.34x | 0.0259 | 0.4632 | 399.0 | 44.0 | 4.76 | 460.22 | 1.0 | 2.0 |
| 2048 | 1024 | 0.0430 | 0.9487 | 22.08x | 0.0279 | 0.9364 | 780.8 | 44.2 | 10.02 | 933.37 | 1.0 | 2.0 |
| 2048 | 2048 | 0.0644 | 1.9358 | 30.08x | 0.0345 | 1.9215 | 1042.8 | 43.3 | 33.46 | 1916.42 | 1.0 | 2.0 |
| 2048 | 4096 | 0.1012 | 3.7444 | 37.01x | 0.0704 | 3.7378 | 1326.5 | 44.8 | 71.05 | 3738.86 | 1.0 | 2.0 |
| 8192 | 128 | 0.0495 | 0.1324 | 2.67x | 0.0255 | 0.1189 | 338.9 | 158.4 | 19.95 | 116.73 | 1.0 | 2.0 |
| 8192 | 512 | 0.0548 | 0.4836 | 8.82x | 0.0255 | 0.4747 | 1224.3 | 173.5 | 23.99 | 471.31 | 1.0 | 2.0 |
| 8192 | 1024 | 0.0768 | 0.9558 | 12.45x | 0.0477 | 0.9528 | 1748.4 | 175.5 | 46.04 | 951.87 | 1.0 | 2.0 |
| 8192 | 2048 | 0.1505 | 1.8931 | 12.58x | 0.1201 | 1.8829 | 1783.3 | 177.3 | 119.97 | 1879.17 | 1.0 | 2.0 |
| 8192 | 4096 | 0.2673 | 3.7743 | 14.12x | 0.2372 | 3.7659 | 2008.8 | 177.8 | 242.57 | 3763.70 | 1.0 | 2.0 |

**Launch counts.** PufferLib is always 2 (its advantage kernel + a `torch.zeros` memset for the
output buffer, matching real PufferLib usage in `pufferl.py`); Triton is always 1
(`torch.empty_like`, no memset needed — the kernel writes every position).

## Monotonicity gate

FAILED with 3 small (<=6%, one run measured up to 9.9% on a single cell) violations at
`n_iter=100, n_trials=11`. Diagnosis, not swept under the rug:

- **seq_len-axis, Triton only**: lands exactly on `_WARPS` tuning-table boundaries in
  `src/rl_triton/ops/gae.py` (`{512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}`)
  where `num_warps` jumps discretely with `BLOCK_SIZE` — e.g. seq_len=128 falls outside the
  table and gets the default 16 warps for a single-block 128-element reduction (likely
  over-provisioned relative to the work), while seq_len=512 gets 4. A discrete occupancy change
  can make the nominally-bigger config faster in absolute terms — a property of the existing
  tuning table, out of scope for this benchmark to retune.
- **num_envs-axis, both implementations**: violations sit at transitions into higher num_envs
  (e.g. 2048->8192), matching `tests/h100_num_envs_sweep_report.md`'s finding that small grids
  don't yet saturate the H100's 132 SMs — added parallel programs get absorbed by idle SMs at
  near-zero extra wall-clock cost.

Not a Triton-compile leak: each cell is warmed up fresh at its own exact shape before any timed
call, and re-running at 2.2x the iterations/trials changed *which* cells tripped the 2%
threshold from run to run (11 -> 7 -> 4 -> 3 violations across four independent runs) — that is
what measurement noise does; a deterministic compile-leak bug would not shuffle. GPU was
confirmed idle/unthrottled (P0, 0% other-process utilization, no thermal/power throttling) via
`nvidia-smi` at the time of these runs, ruling out a contended/shared-environment explanation.

## Crossover analysis

No crossover exists anywhere in the swept range on either axis: Triton is faster than PufferLib
at every tested `(num_envs, seq_len)` cell, from the smallest (128, 128) to the largest
(8192, 4096).

## Mechanism

- **PufferLib is launch/latency-bound at small T, then flatly bandwidth-starved at large T.**
  Its achieved bandwidth tops out around 175-180 GB/s regardless of num_envs or seq_len — a
  small fraction of the H100's ~3350 GB/s HBM3 datasheet peak. This matches its design: one
  CUDA thread per row does a fully sequential O(T) scan, so a row's completion latency scales
  linearly with T and threads within a warp/block cannot cooperate to hide memory latency the
  way a block-wide reduction can.
- **Triton is dispatch-floor-flat at small problem sizes, then bandwidth-bound at large ones.**
  Device-only time (1.4-243µs) is far below the ~25-50µs of fixed CUDA-event/sync dispatch
  overhead at small (num_envs, seq_len), which is why wall-clock time barely moves across the
  smallest configs (visible as the near-flat blue/orange lines at the bottom of the figure).
  At the largest configs (num_envs>=2048, seq_len>=2048) achieved bandwidth climbs to
  1000-2000+ GB/s, approaching a meaningful fraction of HBM3 peak — genuinely bandwidth-bound,
  consistent with its single-pass, O(log T)-depth in-SRAM design.

## Verdict

Triton faster in **20/20** cells (2.15x-80.1x); PufferLib faster in 0/20. Not the result of
handicapping PufferLib — its kernel is real, hand-written CUDA, JIT-compiled unmodified from
the official source, and was verified numerically equivalent before any timing was taken.
PufferLib's ceiling is its per-row sequential design; it does not have an interior-truncation
capability to lose ground on in this comparison (rl-triton's truncation path has no PufferLib
equivalent to benchmark against at all — see Step 0).

See `gae_performance_crossover.png` for the log-log seq_len-vs-time plot (one line pair per
num_envs; PufferLib's four dashed lines nearly overlap regardless of num_envs, confirming the
O(T)-dominated, num_envs-insensitive read above).
