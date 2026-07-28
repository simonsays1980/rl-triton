# rl-triton vs. PufferLib

*Measured on NVIDIA H200 · 2026-07-25 · pufferlib vendored pufferlib==3.0.0 pufferlib.cpp/pufferlib.cu, JIT-compiled.*

**Reproducibility note: the numbers below are not reproducible from the committed script
as-is.** `benchmarks/compare_pufferlib.py` only imports the real pip `pufferlib` package
(no vendored fallback), which was not installed in the environment these numbers came from;
they were instead produced by a separate, uncommitted scratch runner against a vendored,
sha256-pinned JIT build. Re-running `compare_pufferlib.py` as committed will not reproduce
this file unless pip `pufferlib` is installed with a working prebuilt extension. See the
method note at the bottom of this file for the full detail.

Capability + design comparison, not a scoreboard. PufferLib's kernels are real, hand-written CUDA: one thread per environment row, sequential O(T) scan within the thread. rl-triton's kernels are Triton: one program per environment row, O(log T) in-SRAM tree reduction via `tl.associative_scan`. Different mechanisms, different tradeoffs — PufferLib's flat per-thread cost tends to win at very short horizons; rl-triton's parallel scan tends to win as horizon grows.

Both full-call wall time (headline — includes launch/wrapper overhead, what a caller pays every invocation) and device-only kernel time (diagnostic) are reported throughout, since per-call overhead is central to the short-horizon comparison.

## GAE comparison

**Capability gap (not benchmarked, because there is nothing on the other side to benchmark against):** PufferLib takes a single `dones` flag per step, with no distinction between true episode termination and a time-limit truncation, and no `bootstrap_values` mechanism. rl-triton's interior-truncation path (`terminateds`/`truncateds`/`bootstrap_values`) has no PufferLib equivalent at all.

#### Production regime (seq_len [80,128] x num_envs [4096..38400])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 80 | 0.0499 | 0.0039 | 0.0723 | 0.0546 | 1.45x | 13.83x |
| 4096 | 128 | 0.0576 | 0.0040 | 0.1478 | 0.1233 | 2.57x | 31.12x |
| 8192 | 80 | 0.0590 | 0.0064 | 0.0775 | 0.0541 | 1.31x | 8.45x |
| 8192 | 128 | 0.0584 | 0.0064 | 0.1418 | 0.1182 | 2.43x | 18.34x |
| 16384 | 80 | 0.0587 | 0.0113 | 0.0788 | 0.0565 | 1.34x | 4.99x |
| 16384 | 128 | 0.0588 | 0.0114 | 0.1446 | 0.1228 | 2.46x | 10.80x |
| 32768 | 80 | 0.0656 | 0.0215 | 0.0882 | 0.0690 | 1.34x | 3.21x |
| 32768 | 128 | 0.0645 | 0.0229 | 0.1477 | 0.1289 | 2.29x | 5.64x |
| 38400 | 80 | 0.0678 | 0.0248 | 0.1653 | 0.1458 | 2.44x | 5.87x |
| 38400 | 128 | 0.0681 | 0.0267 | 0.2867 | 0.2690 | 4.21x | 10.08x |

#### Boundary marker (seq_len [8,16,32]) — PufferLib's best-case regime

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 8 | 0.0559 | 0.0036 | 0.0349 | 0.0038 | 0.62x | 1.05x |
| 4096 | 16 | 0.0566 | 0.0037 | 0.0349 | 0.0093 | 0.62x | 2.54x |
| 4096 | 32 | 0.0559 | 0.0037 | 0.0516 | 0.0295 | 0.92x | 8.03x |
| 8192 | 8 | 0.0568 | 0.0061 | 0.0350 | 0.0039 | 0.62x | 0.65x |
| 8192 | 16 | 0.0579 | 0.0061 | 0.0349 | 0.0094 | 0.60x | 1.55x |
| 8192 | 32 | 0.0553 | 0.0061 | 0.0524 | 0.0304 | 0.95x | 4.95x |
| 16384 | 8 | 0.0567 | 0.0110 | 0.0351 | 0.0041 | 0.62x | 0.37x |
| 16384 | 16 | 0.0567 | 0.0110 | 0.0348 | 0.0096 | 0.61x | 0.87x |
| 16384 | 32 | 0.0576 | 0.0111 | 0.0520 | 0.0302 | 0.90x | 2.73x |
| 32768 | 8 | 0.0628 | 0.0209 | 0.0350 | 0.0050 | 0.56x | 0.24x |
| 32768 | 16 | 0.0640 | 0.0209 | 0.0350 | 0.0102 | 0.55x | 0.49x |
| 32768 | 32 | 0.0632 | 0.0209 | 0.0530 | 0.0320 | 0.84x | 1.53x |
| 38400 | 8 | 0.0676 | 0.0243 | 0.0351 | 0.0067 | 0.52x | 0.28x |
| 38400 | 16 | 0.0665 | 0.0243 | 0.0387 | 0.0170 | 0.58x | 0.70x |
| 38400 | 32 | 0.0664 | 0.0243 | 0.0887 | 0.0669 | 1.34x | 2.75x |

## V-Trace comparison

PufferLib's `puff_advantage_row_cuda` is genuinely V-trace-capable (real `rho_clip`/`c_clip` and an importance-ratio input), not just a GAE kernel — unlike the GAE section above, importance ratios here are real and vary (independently-sampled `log_pi_target`/`log_pi_behavior`), not pinned to 1.0.

**Not the same output quantity — verified, not assumed.** Re-deriving PufferLib's exact recurrence shows its `advantages` output is algebraically identical to rl-triton's `targets - values` (the raw V-trace correction sum), confirmed exactly against an independent reference. It is NOT the same as rl-triton's own `advantages` return value, which does one further step (uses the recursively-corrected next target, per the full IMPALA formula) — confirmed to diverge from the same reference, as expected. PufferLib has no equivalent of that final step. The timing comparison below is rl-triton's full `compute_vtrace_fused` call (both `targets` and `advantages` computed) against PufferLib's single-output kernel — rl-triton does strictly more work per call here, which the reader should weigh alongside the speedup number, not as a hidden asterisk.

#### Production regime (seq_len [80,128] x num_envs [4096..38400])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 80 | 0.0691 | 0.0042 | 0.0748 | 0.0527 | 1.08x | 12.41x |
| 4096 | 128 | 0.0668 | 0.0043 | 0.1397 | 0.1168 | 2.09x | 27.18x |
| 8192 | 80 | 0.0672 | 0.0067 | 0.0762 | 0.0544 | 1.13x | 8.09x |
| 8192 | 128 | 0.0680 | 0.0069 | 0.1389 | 0.1168 | 2.04x | 17.02x |
| 16384 | 80 | 0.0682 | 0.0119 | 0.0763 | 0.0550 | 1.12x | 4.63x |
| 16384 | 128 | 0.0729 | 0.0202 | 0.1412 | 0.1214 | 1.94x | 6.02x |
| 32768 | 80 | 0.0783 | 0.0253 | 0.0879 | 0.0683 | 1.12x | 2.70x |
| 32768 | 128 | 0.0928 | 0.0400 | 0.1466 | 0.1281 | 1.58x | 3.21x |
| 38400 | 80 | 0.0822 | 0.0292 | 0.1641 | 0.1446 | 2.00x | 4.95x |
| 38400 | 128 | 0.0993 | 0.0465 | 0.2762 | 0.2587 | 2.78x | 5.56x |

#### Boundary marker (seq_len [8,16,32])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 8 | 0.0680 | 0.0038 | 0.0345 | 0.0038 | 0.51x | 1.00x |
| 4096 | 16 | 0.0676 | 0.0038 | 0.0346 | 0.0093 | 0.51x | 2.43x |
| 4096 | 32 | 0.0690 | 0.0039 | 0.0519 | 0.0294 | 0.75x | 7.61x |
| 8192 | 8 | 0.0668 | 0.0063 | 0.0346 | 0.0039 | 0.52x | 0.63x |
| 8192 | 16 | 0.0668 | 0.0063 | 0.0347 | 0.0094 | 0.52x | 1.50x |
| 8192 | 32 | 0.0673 | 0.0063 | 0.0570 | 0.0338 | 0.85x | 5.34x |
| 16384 | 8 | 0.0671 | 0.0112 | 0.0352 | 0.0040 | 0.53x | 0.36x |
| 16384 | 16 | 0.0677 | 0.0112 | 0.0348 | 0.0096 | 0.51x | 0.85x |
| 16384 | 32 | 0.0686 | 0.0113 | 0.0522 | 0.0301 | 0.76x | 2.68x |
| 32768 | 8 | 0.0738 | 0.0211 | 0.0349 | 0.0050 | 0.47x | 0.24x |
| 32768 | 16 | 0.0753 | 0.0211 | 0.0349 | 0.0100 | 0.46x | 0.48x |
| 32768 | 32 | 0.1094 | 0.0211 | 0.0687 | 0.0311 | 0.63x | 1.47x |
| 38400 | 8 | 0.0780 | 0.0244 | 0.0349 | 0.0065 | 0.45x | 0.27x |
| 38400 | 16 | 0.0792 | 0.0245 | 0.0379 | 0.0161 | 0.48x | 0.66x |
| 38400 | 32 | 0.0785 | 0.0247 | 0.0857 | 0.0648 | 1.09x | 2.63x |


Reference: [PufferLib](https://github.com/PufferAI/PufferLib).


---

**Method note (one-time study):** produced on 2026-07-25 via the vendored, sha256-pinned JIT build at benchmarks/pufferlib_ext/ (vendored pufferlib==3.0.0 pufferlib.cpp/pufferlib.cu, JIT-compiled) — pip `pufferlib` was not installed in this environment. `benchmarks/compare_pufferlib.py` itself only imports the real pip package and has no vendored fallback (skips cleanly without it); this file's numbers were produced by a separate, uncommitted scratch runner that reuses compare_pufferlib.py's equivalence-gate/sweep functions with the vendored kernel substituted in directly, matching how the GAE section above was originally produced. Re-running `compare_pufferlib.py` as committed will not reproduce this path unless pip `pufferlib` is actually installed with a working prebuilt extension.
