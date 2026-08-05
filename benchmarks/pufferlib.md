# rl-triton vs. PufferLib

*Measured on NVIDIA H100 80GB HBM3 · 2026-08-05 · pufferlib 3.0.*

Capability + design comparison, not a scoreboard. PufferLib's kernels are real, hand-written CUDA: one thread per environment row, sequential O(T) scan within the thread. rl-triton's kernels are Triton: one program per environment row, O(log T) in-SRAM tree reduction via `tl.associative_scan`. Different mechanisms, different tradeoffs -- PufferLib's flat per-thread cost tends to win at very short horizons; rl-triton's parallel scan tends to win as horizon grows.

Both full-call wall time (headline -- includes launch/wrapper overhead, what a caller pays every invocation) and device-only kernel time (diagnostic) are reported throughout, since per-call overhead is central to the short-horizon comparison.

## GAE comparison

**Capability gap (not benchmarked, because there is nothing on the other side to benchmark against):** PufferLib takes a single `dones` flag per step, with no distinction between true episode termination and a time-limit truncation, and no `bootstrap_values` mechanism. rl-triton's interior-truncation path (`terminateds`/`truncateds`/`bootstrap_values`) has no PufferLib equivalent at all.

#### Production regime (seq_len [80,128] x num_envs [4096..38400])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 80 | 0.0355 | 0.0039 | 0.0684 | 0.0548 | 1.93x | 13.89x |
| 4096 | 128 | 0.0404 | 0.0040 | 0.1418 | 0.1234 | 3.51x | 31.08x |
| 8192 | 80 | 0.0404 | 0.0064 | 0.0715 | 0.0538 | 1.77x | 8.39x |
| 8192 | 128 | 0.0397 | 0.0065 | 0.1371 | 0.1196 | 3.45x | 18.54x |
| 16384 | 80 | 0.0413 | 0.0113 | 0.0724 | 0.0560 | 1.75x | 4.94x |
| 16384 | 128 | 0.0422 | 0.0114 | 0.1364 | 0.1205 | 3.23x | 10.56x |
| 32768 | 80 | 0.0524 | 0.0215 | 0.0835 | 0.0696 | 1.59x | 3.23x |
| 32768 | 128 | 0.0529 | 0.0230 | 0.1410 | 0.1282 | 2.67x | 5.57x |
| 38400 | 80 | 0.0549 | 0.0249 | 0.1591 | 0.1458 | 2.90x | 5.85x |
| 38400 | 128 | 0.0570 | 0.0268 | 0.2811 | 0.2693 | 4.93x | 10.04x |

#### Boundary marker (seq_len [8,16,32]) -- PufferLib's best-case regime

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 8 | 0.0376 | 0.0036 | 0.0250 | 0.0038 | 0.66x | 1.05x |
| 4096 | 16 | 0.0393 | 0.0036 | 0.0265 | 0.0093 | 0.67x | 2.55x |
| 4096 | 32 | 0.0388 | 0.0037 | 0.0459 | 0.0295 | 1.18x | 8.07x |
| 8192 | 8 | 0.0388 | 0.0061 | 0.0252 | 0.0040 | 0.65x | 0.65x |
| 8192 | 16 | 0.0400 | 0.0061 | 0.0263 | 0.0094 | 0.66x | 1.54x |
| 8192 | 32 | 0.0390 | 0.0061 | 0.0520 | 0.0350 | 1.33x | 5.72x |
| 16384 | 8 | 0.0412 | 0.0110 | 0.0255 | 0.0041 | 0.62x | 0.37x |
| 16384 | 16 | 0.0403 | 0.0110 | 0.0263 | 0.0096 | 0.65x | 0.87x |
| 16384 | 32 | 0.0412 | 0.0111 | 0.0463 | 0.0301 | 1.12x | 2.73x |
| 32768 | 8 | 0.0502 | 0.0209 | 0.0251 | 0.0051 | 0.50x | 0.24x |
| 32768 | 16 | 0.0501 | 0.0209 | 0.0266 | 0.0102 | 0.53x | 0.49x |
| 32768 | 32 | 0.0505 | 0.0209 | 0.0475 | 0.0321 | 0.94x | 1.53x |
| 38400 | 8 | 0.0545 | 0.0243 | 0.0253 | 0.0067 | 0.46x | 0.28x |
| 38400 | 16 | 0.0536 | 0.0243 | 0.0330 | 0.0171 | 0.62x | 0.70x |
| 38400 | 32 | 0.0534 | 0.0243 | 0.0830 | 0.0678 | 1.56x | 2.79x |

## V-Trace comparison

PufferLib's `puff_advantage_row_cuda` is genuinely V-trace-capable (real `rho_clip`/`c_clip` and an importance-ratio input), not just a GAE kernel -- unlike the GAE section above, importance ratios here are real and vary (independently-sampled `log_pi_target`/`log_pi_behavior`), not pinned to 1.0.

**Not the same output quantity -- verified, not assumed.** Re-deriving PufferLib's exact recurrence shows its `advantages` output is algebraically identical to rl-triton's `targets - values` (the raw V-trace correction sum), confirmed exactly against an independent reference. It is NOT the same as rl-triton's own `advantages` return value, which does one further step (uses the recursively-corrected next target, per the full IMPALA formula) -- confirmed to diverge from the same reference, as expected. PufferLib has no equivalent of that final step. The timing comparison below is rl-triton's full `compute_vtrace_fused` call (both `targets` and `advantages` computed) against PufferLib's single-output kernel -- rl-triton does strictly more work per call here, which the reader should weigh alongside the speedup number, not as a hidden asterisk.

#### Production regime (seq_len [80,128] x num_envs [4096..38400])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 80 | 0.0464 | 0.0042 | 0.0690 | 0.0528 | 1.49x | 12.49x |
| 4096 | 128 | 0.0480 | 0.0043 | 0.1332 | 0.1169 | 2.78x | 27.41x |
| 8192 | 80 | 0.0471 | 0.0067 | 0.0705 | 0.0546 | 1.50x | 8.13x |
| 8192 | 128 | 0.0476 | 0.0069 | 0.1343 | 0.1178 | 2.82x | 17.09x |
| 16384 | 80 | 0.0506 | 0.0119 | 0.0834 | 0.0580 | 1.65x | 4.86x |
| 16384 | 128 | 0.0829 | 0.0205 | 0.1425 | 0.1231 | 1.72x | 6.01x |
| 32768 | 80 | 0.0624 | 0.0256 | 0.0819 | 0.0678 | 1.31x | 2.65x |
| 32768 | 128 | 0.0770 | 0.0407 | 0.1405 | 0.1285 | 1.83x | 3.16x |
| 38400 | 80 | 0.0674 | 0.0300 | 0.1597 | 0.1461 | 2.37x | 4.88x |
| 38400 | 128 | 0.0850 | 0.0471 | 0.2698 | 0.2586 | 3.18x | 5.49x |

#### Boundary marker (seq_len [8,16,32])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 8 | 0.0470 | 0.0038 | 0.0250 | 0.0038 | 0.53x | 1.00x |
| 4096 | 16 | 0.0476 | 0.0038 | 0.0264 | 0.0093 | 0.55x | 2.42x |
| 4096 | 32 | 0.0470 | 0.0039 | 0.0458 | 0.0295 | 0.97x | 7.65x |
| 8192 | 8 | 0.0484 | 0.0063 | 0.0251 | 0.0039 | 0.52x | 0.63x |
| 8192 | 16 | 0.0470 | 0.0063 | 0.0264 | 0.0094 | 0.56x | 1.49x |
| 8192 | 32 | 0.0462 | 0.0063 | 0.0464 | 0.0297 | 1.00x | 4.70x |
| 16384 | 8 | 0.0525 | 0.0112 | 0.0254 | 0.0041 | 0.48x | 0.36x |
| 16384 | 16 | 0.0489 | 0.0112 | 0.0266 | 0.0095 | 0.54x | 0.85x |
| 16384 | 32 | 0.0485 | 0.0113 | 0.0461 | 0.0302 | 0.95x | 2.68x |
| 32768 | 8 | 0.0591 | 0.0211 | 0.0250 | 0.0051 | 0.42x | 0.24x |
| 32768 | 16 | 0.0590 | 0.0211 | 0.0265 | 0.0101 | 0.45x | 0.48x |
| 32768 | 32 | 0.0588 | 0.0211 | 0.0475 | 0.0319 | 0.81x | 1.51x |
| 38400 | 8 | 0.0622 | 0.0244 | 0.0250 | 0.0065 | 0.40x | 0.27x |
| 38400 | 16 | 0.0613 | 0.0245 | 0.0319 | 0.0162 | 0.52x | 0.66x |
| 38400 | 32 | 0.0625 | 0.0247 | 0.0811 | 0.0667 | 1.30x | 2.70x |


Reference: [PufferLib](https://github.com/PufferAI/PufferLib).
