# H100 Profiling Report: `compute_eligibility_traces` and `compute_retrace_fused`

Target: NVIDIA H100 80GB HBM3 SXM (132 SMs, 65536 registers/SM, 2048 threads/SM), at config `(num_envs=512, seq_len=4096)`.

## Method note

`ncu` is installed but unusable in this environment — `/proc/driver/nvidia/params` shows
`RmProfilingAdminOnly: 1`, and this container's capability set explicitly excludes
`CAP_SYS_ADMIN`/`CAP_PERFMON` (confirmed via `capsh`). Root inside the container isn't
enough; the driver blocks hardware performance-counter access, and instead of failing fast,
`ncu` hangs indefinitely waiting on it. `nsys` isn't installed at all.

The numbers below come from three things that don't require the perf-counter path:

1. **Triton's own compiled-kernel metadata** — `n_regs` / `n_spills` / `shared` come straight
   from `ptxas` via Triton's `driver.utils.load_binary` (exact, not estimated). The same
   introspection was applied to Inductor's generated kernels via a
   `CompiledKernel._init_handles` hook.
2. **Hand-computed theoretical occupancy** from those exact register/shared-mem numbers
   against H100's real SM limits (confirmed via `torch.cuda.get_device_properties`: 132 SMs,
   65536 regs/SM, 2048 threads/SM).
3. **Measured device-kernel time** via `torch.profiler` CUDA activity (steady-state,
   post-autotune / post-warmup).

If the container can be granted `CAP_SYS_ADMIN` + `CAP_PERFMON` (or this is run outside the
container), `ncu` could be re-run for real warp-state / memory-throughput counters to sanity
-check the estimates below — flagged where an estimate is used instead of a hard measurement.

---

## Eligibility traces (512, 4096)

| | ours (`eligibility_traces_fused_kernel`) | Inductor (`triton_red_fused_..._0`) |
|---|---|---|
| regs/thread | 51 | 60 |
| spills | **0** | **0** |
| shared mem | 256 B | 64 B |
| block config | 16 warps (512 threads) | 8 warps (256 threads) |
| blocks/SM (reg-bound) | 2 | 4 |
| **theoretical occupancy** | **50%** (32/64 warps) | **50%** (32/64 warps) |
| device kernel time | **6.61 µs** | 17.93 µs |

Occupancy is a dead heat — 50/50. Our kernel is still **2.7× faster in raw device time** (a
plain FMA-style associative scan vs. Inductor's log→cumsum→exp segment-correction trick,
which burns extra SFU cycles on transcendentals). Neither kernel is register- or
occupancy-limited.

So why does the reported speedup compress from 11.3x (RTX 2000 Ada) → 1.6x (H100)? Because at
6.6–18µs, **both** kernels are dwarfed by the ~25–30µs of fixed CUDA-event/dispatch overhead
that `_bench_gpu` in `tests/bench_utils.py` incurs identically on both sides
(`benchmarks.md`'s own H100 table shows our kernel's *wall-clock* time is nearly flat,
0.033→0.035ms, across all 5 configs — it isn't scaling with the workload at all on H100,
confirming it's dispatch-bound at this size). This is launch-overhead amortization, not a
kernel regression. **Nothing to fix in the kernel.**

---

## Retrace (512, 4096) — the real finding

| | ours (`retrace_fused_kernel`) | Inductor (4 kernels) |
|---|---|---|
| regs/thread | **128** | 18–65 (elementwise stages), 63–106 (reduction stage) |
| spills | **132** | **0** (every stage, every autotuned candidate) |
| shared mem | 4096 B | 0–16384 B |
| block config | 16 warps (512 threads) | 4–8 warps per stage |
| blocks/SM (reg-bound) | **1** (128×512 = 65536 = *exactly* the full register file) | 2–4 |
| **theoretical occupancy** | **25%** (16/64 warps) | **25–50%** depending on stage/config |
| device kernel time | **180.8 µs** (1 launch) | **118.8 µs** total (4 launches: 44.0 + 35.9 + 29.9 + 9.0) |

`118.8 / 180.8 = 0.66×` — this matches `benchmarks.md`'s reported H100 number (0.6x) almost
exactly. **The flip is real, not a measurement artifact.** At BLOCK_SIZE=4096, our kernel's
register footprint (128 regs/thread, filling the SM's entire 65536-register file with a
single 512-thread block) forces 132 register spills/thread and caps occupancy at 25% — while
Inductor's decomposition into 4 lighter kernels never spills at all.

### Empirical `num_warps` / `num_stages` sweep (measured on this H100, not modeled)

| num_warps | regs/thread | spills | device time |
|---|---|---|---|
| 4 | 32 | 1332 | 755.0 µs |
| 8 | 255 (hw cap) | 242 | 236.8 µs |
| **16 (current default)** | **128** | **132** | **181.2 µs** ← best |
| 32 | 64 | 90 (fewest) | 208.8 µs |

`num_stages` (1 vs 2) made no measurable difference at any width. **The current `_WARPS`
table entry (16 warps @ BLOCK=4096) is already the local optimum** — going to 32 warps does
cut spills (90 vs 132) but is *slower*, because halving elements/thread requires more
cross-warp communication in the associative-scan reduction tree (shared mem doubles to
8192B), and that cost outweighs the spill savings. Going to fewer warps is much worse in both
dimensions. **This is not a missed H100 tuning-table entry — the table is already at the
local minimum for this design.**

---

## Answering the four questions

**1. Is the Triton kernel actually low-occupancy/high-register-pressure at this size?**
Yes for Retrace — 25% theoretical occupancy, 132 confirmed register spills/thread, driven
exactly by the mechanism suspected: BLOCK_SIZE=4096 loading the whole row (including two
separate full reads of the `[num_envs, seq_len, num_actions]` tensors — once for
`pi_all`/`nq_all` at `t`, again for `pi_next_all` at `t+1`) forces the entire per-timestep
live state to be resident in registers simultaneously. Eligibility traces is *not*
register-pressured (0 spills, 50% occupancy) — it's a much thinner per-step footprint (2
scalar tensors vs. 4 tensors including two 3D ones).

**2. Is Inductor's kernel achieving meaningfully higher occupancy on H100?**
For eligibility traces, no — occupancy is tied at 50/50; the gap is arithmetic-intensity
(transcendentals), not occupancy. For Retrace, yes — Inductor's kernels hit 25–50%
occupancy *with zero spills*, vs. our 25% *with* 132 spills/thread, because it never has to
hold the full fused live-state in one register file at once.

**3. Launch-overhead amortization or actively suboptimal?**
Both, but for different ops. Eligibility traces: pure launch-overhead-amortization artifact
— real kernel is still 2.7x faster on H100, wall-clock compression is a
benchmarking-methodology effect of H100 being fast enough that ~30µs fixed dispatch cost
dominates both sides equally. Retrace: genuinely suboptimal for this problem size — confirmed
via the register/spill numbers and the warps sweep, not just inferred.

**4. Retune vs. inherent ceiling?**
For Retrace, retuning `_WARPS`/`BLOCK_SIZE` **will not close this gap** — the tuning space
was tested directly and 16 warps is already best-of-breed for the current single-CTA-per-env,
whole-row-resident design. It's a structural ceiling of that design at this size, not a
missed tuning value. The actual fix mirrors what Inductor does naturally: reduce
simultaneous live state. The highest-leverage, most surgical change is in
`src/rl_triton/kernels/retrace_fused.py:154-174` — `pi_next_all` / `action_next` / `mu_next`
are a *second* full reload of the same `[BLOCK, ACTION_BLOCK]` probability/action tensors
already read one lane over for `pi_at_t`/`is_ratio_t`; shifting that already-computed
IS-ratio by one lane (e.g. via a small intra-block shuffle or a second lightweight scan pass)
instead of redundantly reloading and re-summing the 3D tensors would cut roughly half the
retained 3D-array registers driving the 132-register spill, without touching `num_warps` at
all. If that's insufficient, the fallback is splitting into a wide-grid elementwise pre-pass
(computing `expected_next_q`/`rho`/`c_next` into small 2D intermediates, like the *un-fused*
path this kernel's own docstring says was eliminated to save ~23% of time at *small* seq_len)
+ a lighter scan-only kernel — i.e., re-introducing that extra launch specifically for large
seq_len, since the launch-vs-register-pressure tradeoff flips direction at 4096 vs. the small
sizes the fusion was originally tuned for.

For eligibility traces: no action recommended — it's healthy and not the source of the
reported compression.
