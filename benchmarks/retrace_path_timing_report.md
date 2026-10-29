# Retrace at seq_len=4096: fused vs. generic vs. torch.compile (H100)

**Data commit:** `001b8ba` on `ablation-serial`
**Script:** `benchmarks/measure_retrace_register_pressure.py`; run it with
`python benchmarks/measure_retrace_register_pressure.py`
**Generated output:** `benchmarks/retrace_register_pressure.md`. It has two parts: the
register/spill table (unchanged), then a new "Path timing" section with per-repeat tables.
This file summarizes that output. Every number below is copied from it.

## Environment
- NVIDIA H100 80GB HBM3; driver 595.91.07; CUDA 12.4; torch 2.4.1+cu124; Triton 3.0.0; ptxas 12.4.99.
  This matches the expected setup.
- `nvidia-smi` showed no other compute process on the GPU before timing started.
- The register/spill results are identical to the previous run, which used driver 580.126.09:
  at seq_len=4096, 128 regs/thread, n_spills=100 words (400-byte stack frame), 25% occupancy.

## Arms
All arms use num_actions=4, gamma=0.99, and identical inputs at each shape.

| arm | what is called |
|:---:|---|
| A | `compute_retrace`: public API, dispatched path (what Table 4 measures) |
| B | `compute_retrace_fused`: fused kernel forced at every seq_len |
| C | `compute_retrace` with `_TRITON_SEQ_LEN_CEILING = 0` (patched in a try/finally that restores it); a check asserts that the generic branch was taken |
| D | `torch.compile(vectorized_retrace)`: baseline, called as `bench_release.bench_retrace` calls it |

## Method
- Same protocol as `tests/bench_release.py`:
  - `torch._dynamo.reset()` before every shape, and 20 warmup calls (`_warmup_gpu`).
  - Full-call time: `_bench_gpu` (CUDA events, 5 trials × `_n_iter_gpu`, min of trial medians).
  - Device time: `_device_profile`.
- Each shape runs in its own fresh process, and the whole sweep is repeated 3 times.
- Correctness check before timing:
  - Every arm is compared against `tests/test_retrace.py::reference_retrace` on both outputs
    (targets and advantages), with atol=rtol=1e-4.
  - All 18 runs passed for all four arms; max abs error 3.8e-06. No shape ran out of memory.
- Why `reference_retrace` and not `_ref_retrace`:
  - The script's `_make_inputs` has 5% terminations and 5% truncations.
  - `_ref_retrace` merges both into one `dones` flag, so it does not match the library on these
    inputs. `reference_retrace` takes terminations and truncations separately, like the library.
  - Timing does not depend on the flag values.

## Results
Each cell is the median over 3 repeats of D/X, with min–max in brackets. D/X > 1 means arm X is
faster than torch.compile.

| num_envs | seq_len | A path | D/A full-call | D/B full-call | D/C full-call | D/A device | D/B device | D/C device |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 512 | 1024 | fused | 2.09 (2.07-2.14) | 2.17 (2.16-2.22) | 0.37 (0.37-0.39) | 2.00 (2.00-2.02) | 2.01 (1.99-2.01) | 0.36 (0.36-0.36) |
| 512 | 2048 | fused | 1.48 (1.45-1.49) | 1.48 (1.45-1.51) | 0.43 (0.39-0.44) | 1.65 (1.65-1.66) | 1.65 (1.65-1.66) | 0.50 (0.50-0.51) |
| 512 | 4096 | generic | 0.61 (0.61-0.62) | 0.98 (0.97-0.98) | 0.61 (0.60-0.61) | 0.56 (0.56-0.56) | 0.93 (0.93-0.93) | 0.56 (0.56-0.56) |
| 512 | 8192 | generic | 0.60 (0.60-0.60) | 0.34 (0.34-0.34) | 0.60 (0.60-0.60) | 0.57 (0.57-0.57) | 0.31 (0.31-0.31) | 0.57 (0.57-0.57) |
| 128 | 4096 | generic | 0.53 (0.53-0.53) | 1.53 (1.47-1.53) | 0.41 (0.40-0.42) | 0.39 (0.39-0.40) | 0.80 (0.80-0.81) | 0.39 (0.39-0.40) |
| 4096 | 4096 | generic | 0.56 (0.56-0.56) | 0.97 (0.96-0.97) | 0.56 (0.56-0.57) | 0.55 (0.55-0.55) | 0.95 (0.95-0.95) | 0.55 (0.55-0.55) |

## Comparison with existing claims (full-call, num_envs=512)
Rule: a claim holds if it lies within the measured min–max range, widened by the claim's own
rounding (±0.05 for 0.6x, ±0.005 for 0.73x and 0.47x). The retrace.py comment does not state
num_envs, so num_envs=512 is used.

- **Table 4: 0.6x for A at 512×4096. Holds.** Measured 0.608x (range 0.605–0.617).
- **retrace.py: fused 0.73x at 4096. Does not hold.** Measured 0.978x (range 0.971–0.979).
  The gap beyond rounding (0.243) far exceeds the run-to-run spread (0.008).
- **retrace.py: generic 0.47x at 4096. Does not hold.** Measured 0.607x (range 0.605–0.610).
  The gap beyond rounding (0.132) far exceeds the run-to-run spread (0.005).
- **retrace.py: fused is faster than generic at 4096. Holds.** Fused was faster in 3/3 repeats;
  fused time ÷ generic time = 0.619–0.628.
- **retrace.py: generic overtakes fused at 8192. Holds.** Generic was faster in 3/3 repeats;
  fused time ÷ generic time = 1.765–1.773.
- At 4096, the dispatched path (generic) is slower than the fused path at every num_envs
  measured: 128, 512 and 4096.
