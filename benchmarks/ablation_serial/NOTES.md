# Ablation: tree scan vs. fused serial GAE -- orientation notes

Paths found in Step 0 (all reused unchanged; nothing outside this directory was modified):

| What | Path |
|---|---|
| Fused tree-scan GAE kernel (plain path `HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=True`) | `src/rl_triton/kernels/gae.py::gae_fused_kernel` (one program per env, `tl.associative_scan` over a `BLOCK_SIZE=next_pow2(T)` register tile, `_combine` from `kernels/scan.py`) |
| Python wrapper + dispatch table | `src/rl_triton/ops/gae.py::compute_gae` (`_WARPS` table, `num_stages=2 if BLOCK_SIZE>=2048`, `last_value=` selects the plain path with bootstrap) |
| Independent sequential reference | `tests/test_gae.py::reference_gae` (PyTorch backward loop over t, `bootstrap_values=[E]` gives V(s_T)) |
| Timing harness | `tests/bench_utils.py`: `_warmup_gpu` (20), `_bench_gpu` (CUDA events, 5 trials x n_iter, min of trial medians), `_device_profile` (torch.profiler kernel time), `assert_correctness` (atol=rtol=1e-4, finiteness check) |
| torch.compile doubling-scan baseline | `tests/test_gae.py::vectorized_gae_with_truncations` -> `tests/bench_utils.py::parallel_suffix_scan`; compiled directly with zero truncateds / bootstrap column built in eager code, as in `tests/bench_release.py::bench_gae` |
| Paper sweep grid (Grid A) | `tests/bench_release.py::CONFIGS` (identical 12 shapes) |
| Per-shape subprocess isolation pattern | `tests/bench_release.py::_run_per_shape_sweep` / `_ONLY_SHAPE` |
| Short-horizon grid (Grid B) + figure style | `benchmarks/benchmark_gae_vs_pufferlib.py` (`SEQ_LENS_SHORT`, `NUM_ENVS_LIST_LARGE`, dataviz palette, log2 x-axis) |
| PPO end-to-end script | `benchmarks/ppo_e2e_measurement.py` (`Config`, `measure`, `_row`, `correctness_gate`, arms `triton`/`scan`/`loop`/`loop_compiled`, `_CONTAMINATION_THRESHOLD=0.01`) -- the (128,128)/16384 config is not in its `build_grid`, so `e2e_gapfill.py` here builds that `Config` and calls the script's own functions |
| ResiP-inspired learner benchmark | **not on `main`**. Exists only on `origin/resip-ppo-benchmark` (`benchmarks/resip_ppo_e2e_measurement.py`, commits 7f87266 / 452b2ff). See SUMMARY.md for whether it was run. |

Deviations from the task spec, and why:

- **Serial tuning space is larger than `BLOCK_E x num_warps`.** A first probe showed the
  literal one-step-per-iteration loop exposes a full memory latency on every step
  (the loads do not depend on the carry, but in-order issue serializes them behind
  the previous step). Unrolling the load issue by 4-8 steps made S_time 2-3x faster
  at several shapes. Leaving that out would attribute un-hidden load latency to
  "dependency depth". So the tuned serial kernels also sweep `UNROLL in {1,2,4,8}`
  (issue UNROLL steps' loads, then run UNROLL dependent steps; chain length still T)
  and `tl.range(num_stages in {1,3})`. The spec-literal loop (`UNROLL=1, NUM_STAGES=1`,
  best over `BLOCK_E x num_warps`) is reported separately as `serial_env_naive` /
  `serial_time_naive`.
- **V(s_{t+1}) is register-carried, not re-loaded.** The value loaded as V(s_t) at
  step t+1 is reused as the next-state value at step t (bootstrap at t=T-1), so every
  input element is read from memory exactly once.
- **Config selection.** All 128 configs are correctness-gated, then screened by device
  time (1 profiler trial); the top 4 get the full protocol and the best full-call wins.
  The winner is then re-measured in an independent full-protocol run, and that run is the
  reported number (avoids a winner's-curse bias toward serial). Triton 3.0's software
  pipeliner fails to compile some `NUM_STAGES=3` configs (`arith.select` layout mismatch);
  those configs are excluded. `n_configs_valid` in the raw rows records how many survived.
- **Tree config is the library's own dispatch** (`_WARPS`, `num_stages`), not re-tuned,
  so it matches the paper. `tree_lib` (public `compute_gae` with its asserts) is also
  measured, to tie these numbers to the paper's.
