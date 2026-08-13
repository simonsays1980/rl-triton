# PPO end-to-end measurement -- single config -- 2026-08-13

ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table. Single custom config (not part of the axis-a/b/c grid), added to test the Amdahl-paragraph claim that GAE's share rises as the policy net shrinks, at PufferLib's published operating point (~150k-param net, short rollouts, massive single-GPU parallelism).

GPU: NVIDIA H100 80GB HBM3 · torch 2.4.1+cu124
Config `envs=16384 seq=128 hidden=(128, 128) eager [puffer-op-point]`, timing mode `event`, arms `triton,loop`, warmup 40 (default is 10), interleaved with rotating arm order, median (IQR) across iterations.

## Harness fixes made investigating this config (see METHODOLOGY NOTE 5/6/7 in `ppo_e2e_measurement.py`)

The first run at this config (`resid` -0.96/-0.21, non-GAE drift 3.46%) prompted a diagnosis pass, timeboxed at one hour:

1. **`resid` and `non-GAE vs ref` were computed wrong under high variance (fixed).** Both were differences/ratios of *already-aggregated* per-stage medians (`stage_sum = sum of per-stage medians`, then compared against `median(total)`). That is a different quantity from "do stages account for the total, per iteration" once iteration-to-iteration variance is large -- medians of marginals don't sum the way medians of a joint distribution do. Per-iteration totals at this config range 69-116ms. Recomputing both as median-of-per-iteration-differences (matching how `speedup` was already computed) shows the *true* per-iteration resid is small and stable: -0.27 to -0.40ms, under 0.5% of total, consistent across every rerun. The old aggregate-first resid (-0.96 to -2.15ms across reruns of the identical config) was substantially a computation artifact, not missing device time. **Stage accounting itself was never broken.**
2. **CUDA-event objects were allocated fresh on every `start()`/`stop()` call** (~84/iteration at this config, ~2500 across a 30-iter run). Pooled and reused instead. Tested in isolation: no measurable effect on resid or contamination. Kept anyway as a minor efficiency fix, not because it was the cause.
3. **Allocator state carried across arms within an iteration, and 10 warmup iterations wasn't enough** for this config's clock/allocator ramp-up (rollout tensors are ~0.6GB here vs ~150MB at the grid's largest existing point). Fixed with a `sync()+empty_cache()` after every arm's step (not just at iteration boundaries) plus a config-level `n_warmup=40` override (`Config.n_warmup`, new field, default unchanged at 10 so other grids are unaffected). **Neither alone was sufficient** -- allocator fix alone at 10-iter warmup was 0/2 clean in testing; the two together were needed.
4. **Clock throttling: ruled out directly.** Monitored `nvidia-smi` (SM clock, throttle-reason bits, power, temp) through a full run. SM clock was pinned at the 1980MHz boost ceiling for the entire measured region; `clocks_event_reasons.active` was `0x0` throughout compute (only showed the idle bit before the first kernel); power drew 300-400W against a 700W cap; temperature ran 29-39degC. No throttling of any kind occurred. This is not the source of the variance.
5. **One-line bug, unrelated to the above:** the `--probe-launch-bound` CUDA-graph capture failed on `Adam` (`capturable=False`) and the resulting `nan` silently printed as "(compute-bound)" (`nan > 1.3` is `False`). Fixed the comparison to report "inconclusive" on `nan`, and separately gave the probe's own optimizer `capturable=True` so it now actually captures instead of failing.

## Repeatability check -- 9 reruns of the identical config after the fix

The three fixes above (aggregation, event pooling, allocator+warmup) reduced but did **not eliminate** the non-GAE contamination gate tripping. 9 back-to-back reruns of the exact same config, all through the fixed harness:

| # | loop share | loop vs-triton | non-GAE vs ref | gate |
|---|---|---|---|---|
| 1 | 12.934% | 1.1406x | 1.0062 | clean |
| 2 | 14.426% | 1.1551x | 0.9989 | clean |
| 3 | 14.760% | 1.1510x | 1.0029 | clean |
| 4 | 14.960% | 1.1241x | 0.9592 | **CONTAMINATED** |
| 5 | 12.354% | 1.1525x | 1.0057 | clean |
| 6 | 9.839%  | 1.1136x | 1.0121 | **CONTAMINATED** |
| 7 | 13.770% | 1.1468x | 1.0000 | clean |
| 8 | 13.038% | 1.1245x | 0.9692 | **CONTAMINATED** |
| 9 | 12.863% | 1.1276x | 1.0073 | clean |

6/9 clean (67%), 3/9 contaminated (33%). Mean `non-GAE vs ref` across all 9 = 0.9957 -- i.e. **no systematic directional bias** (loop's non-GAE stages are not consistently faster or slower than triton's; the mean sits within 0.5% of 1.0). The failures are individual draws landing 1.2-4.1% away from 1.0 by chance, against a threshold (1%) that is tight relative to this config's per-iteration noise floor. This reads as **unbiased sampling noise interacting with a strict gate at N=30 iterations**, not a remaining accounting bug -- the per-iteration `resid` (item 1 above) is small and stable in every one of these 9 runs, which is the actual test for "are stages being measured correctly."

Per the task's own rule ("contamination persists after warmup, allocator, and clock fixes is a useful answer and ends the task"): **this is that outcome, partially.** The fix measurably helped (contamination rate dropped from what looked like ~4/5 before the fix to 3/9 after), but does not reliably certify a single run clean. Raising `n_iters` beyond 30 would very likely finish the job (unbiased noise around a threshold shrinks with more samples) but is out of scope for this pass -- it is not one of the three suspects the task named, and touches a knob (`n_iters`) not covered by "no new arms, grids, or axes." Left as a note, not attempted.

## What is defensible from this data

- **Correctness gate: PASS** in every trial (loop vs triton, atol=1e-4 rtol=1e-4, max abs diff ~4.8e-6). The loop arm's advantages are numerically correct; only the *timing comparison's* validity is what's at issue above.
- **GAE share for the loop arm at this config: ~10-15%, centered around 13%** across 9 trials (clean and contaminated draws alike -- the share number itself doesn't move much between clean and flagged runs, which is further evidence the flag is a threshold/noise artifact rather than a real confound corrupting the measurement). This is roughly **6x** the 2.17% share measured at `hidden=(1024,1024)`.
- **Speedup (loop total / triton total): ~1.11-1.16x**, i.e. the loop baseline is 11-16% slower than the Triton-kernel arm at this config, vs the 1.0224x measured at `(1024,1024)`.
- **Launch-bound probe (now working, was previously `nan`/mislabeled): ratio consistently ~1.02-1.08x across all post-fix runs → compute-bound, not launch-bound**, at this net size. Per-stage breakdown corroborates this: stage times span almost two orders of magnitude within a single arm (forward ~20ms, backward ~42-64ms, optimizer ~0.7ms) rather than collapsing toward a common floor, which is the signature of a compute-bound rather than overhead-dominated step even at this small a net -- the large batch (16384x128) keeps backward/forward real work dominant.

### Below: full detail for one representative clean run (#9 above)

## Correctness gate

atol=0.0001 rtol=0.0001 (matches tests/test_gae.py's cross-implementation convention).

| config | arm | result | max abs diff | max rel diff |
|---|---|---|---|---|
| envs=16384 seq=128 hidden=(128, 128) eager [puffer-op-point] | loop | PASS | 4.768e-06 | 1.023e-01 |

## Timing

| config | timing | arm | total | GAE | GAE dev | share | vs-triton | resid |
|---|---|---|---|---|---|---|---|---|
| envs=16384 seq=128 hidden=(128, 128) eager [puffer-op-point] | event | triton | 78.592 (15.590) | 0.0292 | 0.0232 | 0.037% | 1.0000 (0.0000) | -0.272 |
| envs=16384 seq=128 hidden=(128, 128) eager [puffer-op-point] | event | loop | 91.237 (19.837) | 11.7361 | 11.7159 | 12.863% | 1.1276 (0.1123) | -0.284 |

## Per-stage breakdown (contamination check)

| config | timing | arm | forward | gather | perm | gae | loss | backward | optimizer | non-GAE | non-GAE vs ref (IQR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| envs=16384 seq=128 hidden=(128, 128) eager [puffer-op-point] | event | triton | 20.429 | 3.932 | 1.513 | 0.029 | 2.589 | 49.584 | 0.671 | 78.299 | 1.0000 (0.0000) |
| envs=16384 seq=128 hidden=(128, 128) eager [puffer-op-point] | event | loop | 20.469 | 3.965 | 1.595 | 11.736 | 2.583 | 51.371 | 0.666 | 80.640 | 1.0073 (0.0696) |

## Citable rows (this run only)

Contamination threshold: |non-GAE vs ref - 1.0| < 0.01 AND correctness gate PASS. See "Repeatability check" above -- this run happened to land clean, but 3 of 9 identical reruns did not, so treat this run's exact numbers as one draw from the ~10-15% GAE-share / ~1.11-1.16x speedup range, not a precise point estimate.

**Excluded (0):** (none this run)
**Citable (2):** triton, loop (this run)

## Launch-bound probe

Optimizer-step-only microbenchmark: eager vs CUDA-graph replay. Ratio > 1.3x would indicate launch-bound.

eager 2.8612ms vs graph-replay 2.7917ms -> 1.02x (compute-bound)
