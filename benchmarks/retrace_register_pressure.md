# Retrace fused-kernel register pressure

## NVIDIA H100 80GB HBM3 (driver 580.126.09, CUDA 12.4, Triton 3.0.0, torch 2.4.1+cu124) · 2026-08-15

`retrace_fused_kernel`, num_envs=512, num_actions=4 -- the shape `tests/h100_profiling_report.md` and `tests/h100_short_horizon_l2_retrace_ppo_report.md` used for this kernel's original register-pressure study.

`n_regs` = ptxas "Used N registers" (exact). `n_spills` = Triton's `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4`, i.e. the per-thread cumulative stack frame size in 4-byte words -- confirmed below to match ptxas's "N bytes cumulative stack size" / 4, NOT "N bytes spill stores" or "N bytes spill loads" / 4. Occupancy is derived from `n_regs` alone (registers/thread x 32 x num_warps vs. the SM's 65536-register file); it does not depend on `n_spills`.

| seq_len | BLOCK_SIZE | num_warps | n_regs | n_spills (words) | occupancy |
|:-------:|:----------:|:---------:|:------:|:-----------------:|:---------:|
|    1024 |       1024 |         8 |    125 |                  0 |     25.0% |
|    2048 |       2048 |        16 |    127 |                  0 |     25.0% |
|    4096 |       4096 |        16 |    128 |                100 |     25.0% |
|    8192 |       8192 |        32 |     32 |                310 |    100.0% |

### `ptxas -v` cross-check at seq_len=4096 (verbatim stderr, re-run directly on the PTX Triton generated for this kernel)

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'retrace_fused_kernel' for 'sm_90a'
ptxas info    : Function properties for retrace_fused_kernel
    400 bytes stack frame, 456 bytes spill stores, 456 bytes spill loads
ptxas info    : Used 128 registers, 400 bytes cumulative stack size
```
