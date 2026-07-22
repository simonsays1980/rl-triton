# H100 Follow-up: Short-Horizon Regime, L2 Hypothesis, Retrace Spill Fix, End-to-End PPO Share

Follow-up to `h100_profiling_report.md` (register-spill/occupancy analysis of Retrace and
eligibility-traces at `(512, 4096)`) and `h100_num_envs_sweep_report.md` (`num_envs` scaling
at fixed `seq_len=2048`). Four independent experiments, run in order with a hard sanity gate
between each — do not proceed past a failed check. All results below are measured on this
H100 SXM; none are projected, adjusted, or tuned toward a preferred outcome.

## Method

Same harness as the rest of the suite: `tests/bench_utils.py`'s `_bench_gpu`/`_warmup_gpu`
(CUDA-event, min-of-5-medians), per-config warmup at the exact timed shape, and
`torch.profiler` for device-only kernel timing and kernel-launch counts. Wall-clock timing
additionally uses host-synchronized `time.perf_counter` round trips to expose dispatch
overhead the CUDA-event method doesn't capture. Scripts used are throwaway diagnostics run
manually (same convention as `bench_num_envs_sweep.py`) and are not part of the release suite.

---

## Experiment 1: the untested short-horizon regime (num_envs 4096–16384 × seq_len 8–128)

**Motivation.** Every benchmark up to this point starts at `seq_len=512`. Production
massively-parallel GPU simulation (Isaac Gym / Isaac Lab) uses the opposite aspect ratio:
`num_envs` in the thousands, horizon 8–64. Prior `num_envs` sweeps fixed `seq_len=2048` and
never tested this regime. Hypothesis under test: at short `seq_len`, both sides become
launch-bound rather than bandwidth-bound, and the fused Triton kernel's 1-launch design should
win by roughly the launch-count ratio against a multi-launch vectorized baseline.

**A sanity-gate failure, caught before any numbers were trusted.** The first sweep pass hit
`torch._dynamo.config.cache_size_limit` (default 8) after the 8th distinct shape. Past that
point `torch.compile` silently fell back to *uncompiled eager* execution instead of raising —
this inflated the compiled baseline from 2 kernels to 14–18 (confirmed via
`torch._dynamo.explain`-style recompile warnings) and made Triton look artificially better at
the larger configs. Fixed by raising `cache_size_limit=64` before any `torch.compile` call and
re-running; recompile warnings disappeared and kernel counts became stable (1 vs 2) across all
30 (algorithm × config) cells. All numbers below are from the corrected run.

### Results

| algo | num_envs | seq_len | dev speedup | wall speedup | #kernels (tri/cmp) |
|---|---|---|---|---|---|
| gae | 4096 | 8 | 1.53x | 1.70x | 1 / 2 |
| gae | 4096 | 16 | 1.56x | 1.75x | 1 / 2 |
| gae | 4096 | 32 | 1.58x | 1.78x | 1 / 2 |
| gae | 4096 | 64 | 1.57x | 1.79x | 1 / 2 |
| gae | 4096 | 128 | 1.55x | 1.79x | 1 / 2 |
| gae | 8192 | 8 | 1.47x | 1.75x | 1 / 2 |
| gae | 8192 | 16 | 1.49x | 1.99x | 1 / 2 |
| gae | 8192 | 32 | 1.44x | 1.77x | 1 / 2 |
| gae | 8192 | 64 | 1.36x | 1.77x | 1 / 2 |
| gae | 8192 | 128 | 1.33x | 1.98x | 1 / 2 |
| gae | 16384 | 8 | 1.20x | 1.77x | 1 / 2 |
| gae | 16384 | 16 | 1.18x | 2.02x | 1 / 2 |
| gae | 16384 | 32 | 1.18x | 1.77x | 1 / 2 |
| gae | 16384 | 64 | **1.02x** | 1.45x | 1 / 2 |
| gae | 16384 | 128 | **0.98x (slower)** | 1.23x | 1 / 2 |
| disc_ret | 4096 | 8 | 1.60x | 1.81x | 1 / 2 |
| disc_ret | 4096 | 16 | 1.59x | 1.78x | 1 / 2 |
| disc_ret | 4096 | 32 | 1.60x | 1.83x | 1 / 2 |
| disc_ret | 4096 | 64 | 1.59x | 2.05x | 1 / 2 |
| disc_ret | 4096 | 128 | 1.57x | 1.80x | 1 / 2 |
| disc_ret | 8192 | 8 | 1.59x | 1.80x | 1 / 2 |
| disc_ret | 8192 | 16 | 1.57x | 1.82x | 1 / 2 |
| disc_ret | 8192 | 32 | 1.54x | 1.84x | 1 / 2 |
| disc_ret | 8192 | 64 | 1.41x | 1.79x | 1 / 2 |
| disc_ret | 8192 | 128 | 1.38x | 1.82x | 1 / 2 |
| disc_ret | 16384 | 8 | 1.34x | 1.82x | 1 / 2 |
| disc_ret | 16384 | 16 | 1.28x | 1.81x | 1 / 2 |
| disc_ret | 16384 | 32 | 1.27x | 2.04x | 1 / 2 |
| disc_ret | 16384 | 64 | 1.11x | 1.55x | 1 / 2 |
| disc_ret | 16384 | 128 | 1.06x | 1.40x | 1 / 2 |

(dev speedup = CUDA-event min-of-5-medians device time, compiled/triton; wall speedup =
host-synchronized round trip over 2000 iterations, compiled/triton; kernel counts from
`torch.profiler` CUDA-activity events, one representative call per cell.)

### Verdict: hypothesis not supported — and a genuinely new limitation surfaces

Kernel-launch counts are **flat at 1 (Triton) vs 2 (compiled) across the entire sweep**. The
`log2(T)` launch-count hypothesis never materializes because the "vec" `torch.compile`
baseline used throughout this project (`vectorized_gae` / `vectorized_discounted_returns`) is
a log-space-cumsum formulation, not the doubling associative scan (`parallel_suffix_scan` is
only used by the separate truncation-supporting "assoc" baseline, not exercised here). So
launch-count-ratio is not the mechanism, and this is not the largest relative win in the
library — it's the **smallest** (1.0–1.8x vs GAE's established flat 1.9x, V-Trace's 3.8x).

What the data shows instead, and it's a real, useful finding: `torch.compile`'s device time is
nearly flat (~57–68us for GAE) across the entire regime — it's sitting on a fixed per-launch
floor. Triton's device time starts below that floor at 4096 envs but grows with `num_envs`
(one program per env → more grid waves per SM), crossing the floor and **inverting below 1x at
16384 envs × seq_len≥64**. This is a genuine, reproducible device-time loss, not previously
tested (prior sweeps fixed `seq_len=2048`, well outside this regime). Triton still wins on
wall-clock everywhere (lower dispatch overhead from 1 launch vs 2), but the device-time
inversion is a real limitation of the current one-program-per-env grid design at
Isaac-Gym-scale env counts with short horizons — worth documenting, not hiding.

---

## Experiment 2: is the eligibility-traces "ceiling" just L2?

**Hypothesis under test.** Observed per-env cost steps up between 1024 and 4096 envs then
appears to pin flat through 8192 (6.6ns → 5.4ns → 8.5ns → 8.5ns at `seq_len=2048`). H100 has
50MB of L2; the working set for eligibility-traces (3 fp32 tensors) crosses 50MB between 1024
and 4096 envs. Hypothesis: below the crossing the kernel runs L2-resident (inflated,
better-than-HBM-roofline numbers); above it, true HBM bandwidth applies, and the "flat
plateau" from 4096→8192 is the roofline, not a kernel flaw.

### Part (a): seq_len=2048, sweep num_envs (achieved bandwidth = bytes moved / device time)

| num_envs | bytes | device time | achieved BW | % of 3.35 TB/s peak |
|---|---|---|---|---|
| 512 | 12.6 MB | 29.66us | 0.42 TB/s | 12.7% |
| 1024 | 25.2 MB | 30.30us | 0.83 TB/s | 24.8% |
| 4096 | 100.7 MB | 56.64us | 1.78 TB/s | 53.1% |
| 8192 | 201.3 MB | 89.25us | 2.26 TB/s | 67.3% |

Extended beyond the originally-requested range to test whether the apparent 4096→8192
plateau is real or still climbing:

| num_envs | bytes | device time | achieved BW | % peak |
|---|---|---|---|---|
| 8192 | 201.3 MB | 90.02us | 2.24 TB/s | 66.8% |
| 16384 | 402.7 MB | 153.34us | 2.63 TB/s | 78.4% |
| 32768 | 805.3 MB | 285.06us | 2.83 TB/s | 84.3% |

### Part (b): num_envs=8192, sweep seq_len (crosses the 50MB boundary between seq_len=256 and 512)

| seq_len | bytes | device time | achieved BW | % peak |
|---|---|---|---|---|
| 64 | 6.3 MB | 38.21us | 0.16 TB/s | 4.9% |
| 128 | 12.6 MB | 38.21us | 0.33 TB/s | 9.8% |
| 256 | 25.2 MB | 41.76us | 0.60 TB/s | 18.0% |
| 512 | 50.3 MB | 39.74us | 1.27 TB/s | 37.8% |
| 1024 | 100.7 MB | 59.01us | 1.71 TB/s | 50.9% |
| 2048 | 201.3 MB | 89.44us | 2.25 TB/s | 67.2% |

### Verdict: L2 hypothesis refuted — this is occupancy ramp-up, not a cache-tier effect

Two independent pieces of evidence kill it:

1. **No config ever exceeds HBM peak.** The 512/1024-env points sit at 13%/25% of the 3.35
   TB/s peak — nowhere near the ">3.35 TB/s" signature that would confirm L2 residency was
   inflating the small-env numbers.
2. **No discontinuity at the 50MB boundary.** Part (b)'s bandwidth curve rises smoothly through
   the crossing (between `seq_len=256` at 25.2MB and `seq_len=512` at 50.3MB) with no kink or
   plateau — if L2 residency were driving the small-size numbers, a sharp drop should appear
   right at that boundary.

The "plateau" between 4096 and 8192 envs in the original per-env-cost framing isn't real: it
reproduces almost exactly when interpreted as the **marginal** slope between adjacent points
(Δtime/Δenvs ≈ 8.6ns 1024→4096, ≈ 8.0ns 4096→8192 — matching the reported 8.5ns/8.5ns), but
extending to 16384 and 32768 envs shows achieved bandwidth still climbing (67%→78%→84% of
peak) — two points on a smoothly-rising occupancy curve were mistaken for a ceiling.

**Actual mechanism:** this kernel launches one program per env (`grid=(num_envs,)`). At small
`num_envs` the grid doesn't generate enough concurrent thread blocks to saturate H100's 132 SMs
and hide memory latency, so achieved bandwidth is occupancy-limited, not L2- or HBM-limited —
a standard small-problem roofline ramp. Even 32768 envs hasn't reached the ceiling. This
reframes the finding from "unexplained ceiling" to a known, well-characterized, and in
principle addressable occupancy limitation (e.g. persistent-kernel or multi-env-per-program
designs at small grid sizes).

---

## Experiment 3: fixing Retrace's spill regression

**Root cause (confirmed, matches `h100_profiling_report.md` exactly):** at `(512, 4096)`,
`retrace_fused_kernel` compiles to 128 regs/thread, 132 spills/thread, 25% occupancy
(128×512=65536=exactly the SM's full register file at `num_warps=16`). The prior profiling
report's own recommended fix — independently re-derived here — is to stop reloading the full
`[num_envs, seq_len, num_actions]` `action_probs_target` tensor a second time (once for
`pi_all`/`nq_all` at `t`, again for `pi_next_all` at `t+1`) and instead shift the
already-computed `pi_at_t` scalar by one timestep.

### Fix applied

In `src/rl_triton/kernels/retrace_fused.py`, `c[t+1]`'s IS ratio previously required a second
full `[BLOCK_SIZE, ACTION_BLOCK]` load (`pi_next_all`) plus an `action_next` load and a
re-summation. The fix stores the already-computed `pi_at_t` (a `[BLOCK_SIZE]` vector) to a new
small `[num_envs, seq_len]` scratch buffer and reloads it at the shifted real timestep index
— the same store→`tl.debug_barrier()`→reload idiom the kernel already uses for `next_q_ret`.
This eliminates the second 3D tensor entirely; only one `[BLOCK_SIZE, ACTION_BLOCK]` tensor
(`pi_all`) needs to stay register-resident instead of two.

```python
# before: second full 3D reload
pi_next_all  = tl.load(action_probs_target_ptr + row_base_next, mask=next_mask_3d, other=0.0)
action_next  = tl.load(actions_ptr + base_2d + rev_next, mask=next_mask_2d, other=0)
pi_at_next   = tl.sum(pi_next_all * (a_offs[None, :] == action_next[:, None]).to(tl.float32), axis=1)

# after: shift the already-computed scalar through a small scratch buffer
tl.store(pi_at_scratch_ptr + base_2d + rev, pi_at_t, mask=mask)
tl.debug_barrier()
pi_at_next = tl.load(pi_at_scratch_ptr + base_2d + rev_next, mask=next_mask_2d, other=0.0)
```

### 1. Correctness

Bit-identical to pre-fix (max|before−after| = 0.0) across three shapes
(512×4096×4, 64×1024×8, 128×8192×4) using `git stash` to diff the pre- and post-fix kernel on
identical seeded inputs. Full `tests/test_retrace.py` suite: **17 passed** (1 perf test
deselected by default marker config), run with `RL_TRITON_CORRECTNESS_WARNINGS=1` (the actual
correctness-assertion flag in this codebase — `RL_TRITON_DEBUG` does not exist here, noted
rather than silently substituted). No dedicated "two-interior-truncation" test exists for
Retrace, unlike GAE/V-Trace/returns — Retrace's truncation model gates trace decay only and
does not inject a `bootstrap_values` tensor, so that fixture pattern doesn't apply; the two
truncation tests that do exist (`test_retrace_truncated_keeps_bootstrap`,
`test_retrace_truncated_c_boundary_zero_bootstrap_kept`) both pass.

### 2. ptxas metadata, before vs after

| | before | after |
|---|---|---|
| regs/thread | 128 | 128 |
| spills/thread | 132 | **100** |
| blocks/SM (reg-bound) | 1 | 1 |
| occupancy | 25% | 25% |

Spills drop 24% (132→100) — a real reduction in register-spill traffic — but registers/thread
does not move and occupancy stays pinned at 25%, because `num_warps=16` (512 threads) alone
already saturates the SM's 65536-register file at 128 regs/thread regardless of spill count.
The prior report speculated the fix "would cut roughly half the retained 3D-array registers" —
the measured outcome is a smaller, honest fraction of that.

### 3. Benchmark, seq_len ∈ {1024, 2048, 4096, 8192}, num_envs ∈ {512, 4096}

| num_envs | seq_len | before (speedup) | fused kernel + fix (speedup) |
|---|---|---|---|
| 512 | 1024 | 1.39x | 1.42x |
| 512 | 2048 | 1.14x | 1.08x |
| 512 | 4096 | 0.71x | 0.73x |
| 512 | 8192 | 0.19x | 0.24x |
| 4096 | 1024 | 1.21x | 1.38x |
| 4096 | 2048 | 1.51x | 1.32x |
| 4096 | 4096 | 0.63x | 0.67x |
| 4096 | 8192 | 0.16x | 0.20x |

**The fix alone does not clear the spill regression.** Improvement is marginal-to-noise at
`seq_len=4096` (0.71x→0.73x, 0.63x→0.67x) and the inversion is dramatically worse at
`seq_len=8192` (0.16–0.24x) than anything previously measured. Per this experiment's own
stated criterion, this must not ship as a speedup claim.

### Ceiling dispatch (per the task's fallback instruction)

Added `_TRITON_SEQ_LEN_CEILING = 2048` in `src/rl_triton/ops/retrace.py` — the confirmed-win
boundary from the table above. Above the ceiling, `compute_retrace` now routes through the
codebase's **existing, already-tested** generic-scan path (materialize `u`/`v` with plain
PyTorch ops, then call `_run_scan`, which was already used for `seq_len > 131072` and picks the
flat or chunked scan kernel internally) rather than the specialized fused kernel.

An initial attempt used a fresh log-space-cumsum PyTorch reimplementation (mirroring
`vectorized_retrace`, itself documented as "not production-hardened") as the fallback instead.
That produced **inf/nan** at `seq_len ∈ {4096, 8192}` from float32 underflow in the suffix
log-sum cumsum over long horizons. This was discarded — it would have shipped a correctness
bug instead of a performance regression, which is strictly worse. Reusing the codebase's
already-verified generic scan infrastructure avoided inventing a new failure mode.

**Correctness of the reroute**, verified against a ground-truth Python loop at shapes above the
ceiling (max|diff| in the 1e-6 range, no nan/inf):

| shape | max\|out−ref\| | max\|adv−ref\| |
|---|---|---|
| 8×4096×4 | 9.5e-7 | 9.5e-7 |
| 4×8192×3 | 2.9e-6 | 1.9e-6 |
| 2×2049×5 | 9.5e-7 | 9.5e-7 |

**Performance of the reroute is not uniformly better than the fixed fused kernel** — an honest,
disclosed non-monotonicity, not tuned away:

| num_envs | seq_len | fused kernel + fix | ceiling → generic-scan reroute |
|---|---|---|---|
| 512 | 4096 | 0.73x | 0.47x |
| 512 | 8192 | 0.24x | 0.42x |
| 4096 | 4096 | 0.67x | 0.39x |
| 4096 | 8192 | 0.20x | 0.38x |

At `seq_len=4096` the fixed fused kernel is actually *faster* than the reroute (extra kernel
launches for materializing `u`/`v` outweigh the spill penalty at this size); the reroute only
wins clearly at `seq_len=8192`, where the fused kernel's spill/occupancy penalty dominates.
**Both remain losses vs `torch.compile` above the 2048 ceiling either way** — the ceiling
prevents the worst-case blowup (0.16x) but does not restore a speedup. Existing safeguard test
`test_perf_retrace` (128×1024, within the ceiling, floor 1.35x) is unaffected: 1.57–1.61x.

Regression-tested: full suite (`pytest tests/`) passes 134/134 (21 perf tests deselected by
default marker config) after this change.

---

## Experiment 4: GAE's share of a real PPO step — the number that decides the paper's framing

**Setup.** MLP policy+value, Isaac-Gym-Ant-like (obs_dim=60, action_dim=8), `num_envs=4096`,
`seq_len=32`. One full update: forward, GAE, PPO clipped policy loss + value loss, backward,
Adam step. Wall-clock breakdown via host-synchronized timers around each stage (50 timed
iterations after 15 warmup iterations, per stage summed to `total`).

### hidden=(256, 256)

| stage | Triton GAE | torch.compile GAE |
|---|---|---|
| forward | 0.997 ms (23.4%) | 1.047 ms (20.8%) |
| gae | 0.084 ms (**2.0%**) | 0.183 ms (**3.6%**) |
| loss | 0.151 ms (3.5%) | 0.225 ms (4.5%) |
| backward | 2.738 ms (64.2%) | 3.068 ms (60.9%) |
| optimizer | 0.294 ms (6.9%) | 0.518 ms (10.3%) |
| **total** | **4.264 ms** | **5.042 ms** |

End-to-end speedup: **1.18x**.

### hidden=(1024, 1024)

| stage | Triton GAE | torch.compile GAE |
|---|---|---|
| forward | 7.395 ms (31.9%) | 7.399 ms (31.9%) |
| gae | 0.086 ms (**0.37%**) | 0.134 ms (**0.58%**) |
| loss | 0.154 ms (0.67%) | 0.158 ms (0.68%) |
| backward | 15.233 ms (65.7%) | 15.152 ms (65.4%) |
| optimizer | 0.305 ms (1.3%) | 0.326 ms (1.4%) |
| **total** | **23.173 ms** | **23.169 ms** |

End-to-end speedup: **1.000x — no measurable difference.**

### Verdict: report it plainly

GAE is under 2% of step time even with a tiny (256,256) network, and under 1% with a
(1024,1024) network — where the end-to-end speedup from the Triton kernel is exactly 1.000x.
Backward pass dominates the step (60–66%) in both cases. The kernel-level speedup on GAE in
isolation (the established flat ~1.9x device number) is real and reproducible, but it is
essentially invisible in full training throughput once a real policy network's forward/backward
pass is in the loop. **The paper cannot honestly claim meaningful end-to-end training-speed
gains from this kernel** — only the isolated-microbenchmark claim holds, and that should be
stated as the actual scope of the contribution.

---

## Bottom line across all four experiments

- **Short-horizon regime (Exp 1):** the motivating launch-bound hypothesis is not supported —
  launch counts don't scale with `seq_len` for the baseline actually used here — but a real,
  previously-untested device-time inversion appears at `num_envs=16384` with short horizons,
  extending (not contradicting) the earlier `num_envs`-sweep finding that GAE shows no
  small-grid recovery effect at `seq_len=2048`.
- **Eligibility-traces "ceiling" (Exp 2):** not an L2 effect. It's a small-grid occupancy ramp
  that keeps improving well past the previously-tested 8192-env point (confirmed to 32768
  envs) — a known, addressable limitation rather than an architectural mystery.
- **Retrace spill fix (Exp 3):** the targeted register-pressure fix is real (24% fewer spills)
  but insufficient to close the regression (occupancy unchanged, benchmark improvement is
  noise-level at 4096 and the inversion is worse at 8192 than previously known). Shipped
  honestly as a documented seq_len ceiling (2048) with a correctness-first (not
  performance-first) fallback, rather than as a fixed speedup claim.
- **PPO step share (Exp 4):** GAE is <1–2% of a realistic training step, and the measured
  end-to-end speedup from using the Triton kernel ranges from a modest 1.18x (tiny network) to
  exactly 1.000x (realistic network size). This should directly shape how the paper frames its
  contribution — as a kernel-level result, not a training-throughput result.

None of these findings were tuned toward a preferred outcome; where a fix underperformed or a
hypothesis failed, that is reported as the result.
