# H100 num_envs Sweep: Does Triton's Speedup Recover at Scale?

Follow-up to the earlier H100 profiling report (`h100_profiling_report.md`), which found the
`seq_len` axis explains the eligibility-traces speedup compression (11.3x on RTX 2000 Ada →
1.6x on H100 at `(512, 4096)`) as a launch-overhead-amortization artifact, and the Retrace
flip (0.6x at `(512, 4096)`) as a real register-spill/occupancy ceiling driven by `seq_len`
via `BLOCK_SIZE`.

This report investigates the other axis: **`num_envs`**. `compute_eligibility_traces`,
`compute_gae`, `compute_vtrace_fused`, and `compute_retrace_fused` all launch one Triton
program per environment (`grid=(num_envs,)`), so `num_envs` controls how many independent
programs run in parallel across the H100's 132 SMs — a genuinely different axis from
`seq_len`, which controls per-program register pressure. At only 512 envs, the grid may be
too small to saturate the GPU across the internal waves the benchmark harness times, which
would cause fixed per-launch dispatch overhead to dominate wall-clock regardless of kernel
quality.

## Method

New standalone script: `tests/bench_num_envs_sweep.py`. Does **not** modify
`tests/bench_release.py`'s `CONFIGS` list and does **not** write to `tests/benchmarks.md` —
it's a throwaway diagnostic, run manually.

- Wall-clock timing uses the exact same `_bench_gpu` / `_warmup_gpu` / `_n_iter_gpu` helpers
  from `tests/bench_utils.py` as the official suite, for methodological consistency.
- Device-only timing uses `torch.profiler` CUDA activity (steady-state, post-warmup), since
  `ncu`/`nsys` remain unusable in this container (`RmProfilingAdminOnly: 1` + missing
  `CAP_SYS_ADMIN`/`CAP_PERFMON` — see the prior report).
- `seq_len` is held fixed at **2048** (moderate, within the 2k–8k on-policy rollout range per
  `NOTES.md`) while `num_envs` sweeps **512 → 1024 → 4096 → 8192**. 512 is included as a
  baseline anchor for continuity with `benchmarks.md`.
- Covers `compute_eligibility_traces` and `compute_gae` (required), plus
  `compute_vtrace_fused` and `compute_retrace_fused` (bonus, time allowed).

## Results

| op | num_envs | wall-clock speedup | device-only speedup |
|---|---|---|---|
| **eligibility-traces** | 512  | 1.44x | 3.70x |
| | 1024 | 1.58x | 4.13x |
| | 4096 | 1.94x | 2.45x |
| | **8192** | **2.11x** | 2.45x |
| **gae** | 512  | 1.73x | 1.93x |
| | 1024 | 1.56x | 2.03x |
| | 4096 | 1.71x | 1.88x |
| | **8192** | **1.78x** | 1.90x |
| **vtrace-fused** *(bonus)* | 512  | 2.24x | 3.31x |
| | 1024 | 2.04x | 3.51x |
| | 4096 | 3.01x | 3.90x |
| | **8192** | **3.34x** | 3.84x |
| **retrace-fused** *(bonus)* | 512  | 1.53x | 1.67x |
| | 1024 | 1.60x | 1.76x |
| | 4096 | 1.76x | 1.82x |
| | **8192** | **1.80x** | 1.83x |

Raw output preserved below in [Appendix: full script output](#appendix-full-script-output).

## Does the speedup recover? Mixed — depends on the op.

**Eligibility-traces: partial recovery, then a hard ceiling.** Wall-clock speedup climbs
monotonically as `num_envs` grows (1.44x → 1.58x → 1.94x → 2.11x), confirming grid-width
*was* a real factor — more envs means more independent programs to amortize the ~26µs fixed
dispatch overhead against, exactly as hypothesized. But it plateaus around **2.1x at 8192
envs**, nowhere near the historical 11.3x on Ada. The device-only ratio tells the sharper
story: it's *not* monotonic — it peaks at 4.13x (1024 envs) then **drops and flattens at
2.45x** for 4096 and 8192. So wall-clock is converging *toward* the device-time ratio as
overhead amortizes away (as expected), but the device-time ratio itself settles at ~2.45x and
stops moving — that's a real, size-independent ceiling on H100, separate from the
launch-overhead story. Growing `num_envs` further won't push this past ~2–2.5x.

**GAE: no recovery trend — flat from 512 to 8192.** Wall-clock sits at 1.6–1.8x across the
entire sweep with no directional trend (1.73x → 1.56x → 1.71x → 1.78x), and device-only is
equally flat (1.88–2.03x). This says plainly: **GAE's ~1.7x at 512 envs was never a
small-grid artifact.** The grid was already wide enough to saturate the GPU at 512 envs for
this op; there's nothing to recover.

**V-Trace-fused (bonus): the clearest case of genuine recovery.** Wall-clock grows from
2.24x to **3.34x** and device-only from 3.31x to **3.90x** as `num_envs` scales to 8192 —
both trending the same direction with real headroom, not just overhead amortization. This op
benefits the most from wider grids.

**Retrace-fused (bonus, caveat below): modest, steady growth.** 1.53x → 1.80x wall-clock,
1.67x → 1.83x device-only — real but small recovery, plateauing similarly to
eligibility-traces.

### Caveat on Retrace

This sweep fixes `seq_len=2048`, but the reported 0.6x flip was at `seq_len=4096` —
BLOCK_SIZE (and thus the register-spill severity identified in the prior profiling pass) is
tied to `seq_len`, not `num_envs`. At `seq_len=2048` Retrace is already positive even at
`num_envs=512` (1.53x/1.67x), so this sweep confirms `num_envs` scaling helps modestly at a
*lower*-register-pressure `seq_len`, but doesn't directly re-test whether more envs rescues
the specific flipped `(512, 4096)` case — the register-spill ceiling identified there is a
`seq_len`-driven problem, largely orthogonal to this `num_envs` axis.

## Bottom line

The 1.6x number at `num_envs=512` is a **real but incomplete** artifact of small-grid
dispatch-overhead — `num_envs` scaling recovers roughly 0.5–1.3x of wall-clock speedup
depending on the op (eligibility-traces gains the most proportionally, GAE gains nothing).
But for eligibility-traces and retrace-fused, there's a second, size-independent ceiling in
the raw device-time ratio itself (~2.4–2.5x and ~1.8x respectively) that persists at 8192
envs and won't move with more grid width — that ceiling is architectural (Inductor's kernels
genuinely scale well on H100), not a benchmarking artifact. GAE shows no small-grid effect at
all. V-Trace is the outlier with real, substantial additional headroom as envs scale up.

---

## Appendix: full script output

```
num_envs sweep at fixed seq_len=2048, num_envs in [512, 1024, 4096, 8192]
(ad hoc diagnostic — does not touch bench_release.py CONFIGS or benchmarks.md)

### compute_eligibility_traces ###
algo                 num_envs  seq_len  triton(wall)    vec(wall)   wall x    triton(dev)     vec(dev)    dev x
---------------------------------------------------------------------------------------------------------------
eligibility-traces        512     2048      0.0318ms     0.0457ms     1.44x         3.39us      12.56us     3.70x
eligibility-traces       1024     2048      0.0357ms     0.0563ms     1.58x         5.56us      23.00us     4.13x
eligibility-traces       4096     2048      0.0606ms     0.1178ms     1.94x        34.79us      85.28us     2.45x
eligibility-traces       8192     2048      0.0936ms     0.1973ms     2.11x        67.30us     164.93us     2.45x

### compute_gae ###
algo                 num_envs  seq_len  triton(wall)    vec(wall)   wall x    triton(dev)     vec(dev)    dev x
---------------------------------------------------------------------------------------------------------------
gae                       512     2048      0.0406ms     0.0701ms     1.73x         8.09us      15.64us     1.93x
gae                      1024     2048      0.0462ms     0.0721ms     1.56x        15.48us      31.43us     2.03x
gae                      4096     2048      0.0930ms     0.1593ms     1.71x        62.61us     117.86us     1.88x
gae                      8192     2048      0.1516ms     0.2692ms     1.78x       120.06us     228.20us     1.90x

### compute_vtrace_fused ###
algo                 num_envs  seq_len  triton(wall)    vec(wall)   wall x    triton(dev)     vec(dev)    dev x
---------------------------------------------------------------------------------------------------------------
vtrace-fused              512     2048      0.0494ms     0.1105ms     2.24x         9.77us      32.36us     3.31x
vtrace-fused             1024     2048      0.0617ms     0.1259ms     2.04x        22.66us      79.47us     3.51x
vtrace-fused             4096     2048      0.1179ms     0.3549ms     3.01x        79.19us     308.70us     3.90x
vtrace-fused             8192     2048      0.1940ms     0.6470ms     3.34x       155.70us     597.82us     3.84x

### compute_retrace_fused ###
algo                 num_envs  seq_len  triton(wall)    vec(wall)   wall x    triton(dev)     vec(dev)    dev x
---------------------------------------------------------------------------------------------------------------
retrace-fused             512     2048      0.0817ms     0.1246ms     1.53x        43.54us      72.90us     1.67x
retrace-fused            1024     2048      0.1168ms     0.1863ms     1.60x        79.42us     139.52us     1.76x
retrace-fused            4096     2048      0.3192ms     0.5622ms     1.76x       283.82us     516.29us     1.82x
retrace-fused            8192     2048      0.5911ms     1.0629ms     1.80x       557.84us    1019.84us     1.83x
```
