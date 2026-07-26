# Session report — to Claude Opus (session 2, same day)

**From:** Claude Sonnet 5, branch `bench/h100-release`, NVIDIA H100 80GB HBM3.
**Date:** 2026-07-26.
**Continues directly from `docs/benchmark-history/report-to-opus-2026-07-26.md` (session 1), which ended blocked on §4's torch.compile/Dynamo instability with three options proposed (A/B/C). This session tried all three, in order, with hard empirical verification at each step. Nothing merged to `main`, nothing in `src/` touched, `README.md` untouched, no floors committed. `benchmarks.md` is still NOT regenerated — still blocked, but on a NEW, different bug than session 1 left off on.**

Current uncommitted changes: `tests/bench_release.py`, `tests/test_prefix_sum.py` (`git status --short` / `git diff --stat` confirm only these two files touched this session).

---

## 1. Option A (raise Dynamo cache limits) — tested, DISPROVEN

Session 1 hypothesized bug #2 (cross-shape wrong output, e.g. GAE production shape `(4096,80)`) was `torch._dynamo.config.cache_size_limit` (default 8) being exceeded once >8 distinct shapes compiled against one `torch.compile(...)` object — same mechanism as the original short-horizon sweep's silent eager-fallback bug.

**Implemented:** set `torch._dynamo.config.cache_size_limit = 128` and `accumulated_cache_size_limit = 512` at module load in `bench_release.py` (top of file, right after `import torch._dynamo`). Removed all 9 per-shape `torch._dynamo.reset()` calls that session 1 had added (defeats the compile cache, was the direct cause of bug #3's CUDA corruption).

**Result: did NOT fix it.** Ran a GAE-only smoke test (`python tests/bench_release.py --algos gae --no-update`) — crashed at production shape `(4096,80)` with 44% mismatched elements vs. reference, identical failure mode to before. Cache-size theory falsified.

**Kept anyway** (harmless, avoids a *different* real failure mode — silent eager fallback past the cache limit): the raised limits are still in the file.

---

## 2. Option B (`torch._dynamo.mark_dynamic`) — tested, DISPROVEN

Hypothesis: mark `num_envs`/`seq_len` dims dynamic so one compiled artifact serves all shapes, avoiding recompilation (and whatever bug recompilation triggers).

**Tested via standalone repro** (not committed to the file — this was a throwaway diagnostic): compiled `vectorized_gae_with_truncations`, warmed through all 12 `CONFIGS` shapes with `mark_dynamic` applied to every input tensor's dims 0/1, then hit `(4096,80)`. **Still 40% mismatched.** `mark_dynamic` does not help — falsified.

---

## 3. Actual root cause of bug #2, found and FIXED

Isolation testing (standalone repro scripts, not in the committed file) established:
- A **fresh process**, first-and-only compile at `(4096,80)`: correct (0 mismatches).
- Same process, compile a no-padding shape first (e.g. `(512,1024)`), discard that compiled object, create a **brand-new** `torch.compile(...)` object, compile `(4096,80)`: **still 40% wrong.** This proves it's process-level Dynamo/Inductor state corruption, not object reuse.
- The bug is specific to crossing from a **power-of-2 `seq_len`** (no padding needed in `parallel_suffix_scan`/`parallel_prefix_scan`) to a **non-power-of-2 `seq_len`** (needs padding to next power of 2). Every `CONFIGS`/`HEADLINE_CONFIGS`/`BOUNDARY_CONFIG` seq_len is a power of 2; only `PRODUCTION_SEQ_LENS`'s `seq_len=80` triggers the padding branch. This is a real Inductor buffer-reuse/aliasing bug around the padding's internally-allocated zero tensor (`torch.cat([x, torch.zeros(...)])` inside the scan function) — same *class* of bug as session 1's bug #1 (the caller-supplied zero-tensor aliasing bug), just triggered internally.

**Fix implemented** (`tests/bench_release.py`):
- New `_needs_pad(seq_len)` helper (module level, near `CONFIGS`/`BOUNDARY_CONFIG` definitions) — returns whether `parallel_suffix_scan` will pad this `seq_len`.
- `_bench_production_regime()` (the shared production-regime helper) now tracks `prev_needs_pad` and calls `torch._dynamo.reset()` **only when `_needs_pad(seq_len)` differs from the previous shape's** — not on every shape. Since padding depends only on `seq_len` (not `num_envs`), and `PRODUCTION_CONFIGS` only has ONE non-power-of-2 seq_len (80), this means only ~2 resets per compiled object (entering the 80-block, then leaving it back to 128) instead of ~11.
- Retrace's own inline production loop (can't use the shared helper — different arg ordering) got the identical `prev_needs_pad`/reset logic applied separately, same file.
- Total resets across a would-be full sweep: ~14-18 (one ~2 transitions × ~7-9 compiled objects with production regimes), far under the ~38-70 threshold session 1 found corrupts CUDA state.

**Verified reliable, repeatedly:** `python tests/bench_release.py --algos gae --no-update` passed correctness at every shape including `(4096,80)` across multiple repeated runs (deterministic, not flaky). Confirmed also for V-Trace and Retrace in the full-sweep run (§5 below) before it hit the *next* bug.

---

## 4. Bonus finding: a real, independent algorithm bug in `vectorized_episodic_prefix_sum` — FOUND AND FIXED

While validating prefix-sum in isolation (`python tests/bench_release.py --algos prefix_sum --no-update`), hit an assertion failure: **44-77% mismatched elements vs. reference, at `(4096,80)` AND at every other CONFIGS shape too, including `(64,512)`** — i.e., this baseline was wrong basically always, at every shape, not just the new production shape.

**Isolated to a pure algorithm bug, unrelated to torch.compile**: reproduces identically in eager mode with no `torch.compile` involved at all. Root cause: the old formula (`running = cumsum(inputs); boundary = running*dones; offset = cumsum(boundary)-boundary; result = running-offset`) resets the accumulator **one step too late** relative to this codebase's convention (`reference_episodic_prefix_sum`'s ground truth, and the Triton kernel's own documented tests, define `carry[t] = inputs[t] + (1-dones[t])*carry[t-1]` — i.e. `dones[t]=1` means the reset **already applies to step t itself**, not "t is the last step before reset"). Verified against `test_prefix_sum_known_values_reset_at_start`'s own docstring, which explicitly confirms this convention.

**Why never caught before:** `bench_prefix_sum()`'s CONFIGS loop only ever asserted the Triton kernel against the reference — never the vec/compiled baseline's own correctness. Confirmed prefix-sum's "never broken, ~1.2x, no log-space" claim (from session 1's report) was **wrong** — it was simply never checked.

**Fix implemented:**
- `tests/test_prefix_sum.py`: `vectorized_episodic_prefix_sum` now returns `parallel_prefix_scan(inputs, 1.0 - dones)` (reuses the already-validated helper, added `from bench_utils import ... parallel_prefix_scan` import). Old buggy cumsum-difference formula deleted.
- `tests/bench_release.py`: `bench_prefix_sum()`'s CONFIGS loop now also asserts `vectorized_episodic_prefix_sum`'s own correctness against the reference (previously missing) — `vec_out = vectorized_episodic_prefix_sum(...); assert_correctness(vec_out, ref_out, ...)`.
- Verified: `parallel_prefix_scan(inputs, 1-dones)` matches `reference_episodic_prefix_sum` to ~1e-6 at both `(4096,80)` and `(64,512)`. Old formula was wrong at 95.8% of elements even at `(64,512)` — confirms this was **never** correct, at any shape, ever.

**Not yet regenerated:** benchmarks.md's prefix-sum numbers everywhere in this repo (including v0.1.0.md, unreleased.md, any prior benchmarks.md) reflect the OLD BROKEN baseline. Prefix-sum's real speedup vs. a correct compiled baseline is currently unknown — likely different from the previously reported ~1.2x, same pattern as the other 6 algorithms in session 1's §2.

---

## 5. Option C (subprocess-per-algorithm-group isolation) — IMPLEMENTED, not yet validated end-to-end

A full multi-algorithm sweep (`python tests/bench_release.py --gpu "NVIDIA H100 80GB HBM3" --skip-readme`, run in background, PID 20263) completed GAE, GAE-truncation, V-Trace, V-Trace-truncation, Retrace, Retrace-production cleanly (correctness passed throughout, only minor monotonicity-band noise ~2-3%, not a correctness issue) — then **crashed** with `RuntimeError: CUDA error: an illegal memory access was encountered` during eligibility-traces' `_warmup_gpu(c_traces_vec, ...)` at shape `(256,1024)`, in `bench_returns()`'s CONFIGS loop. This happened with only ~6 total `torch._dynamo.reset()` calls so far in the whole process (all from GAE/V-Trace/Retrace's production-regime transitions) — far below session 1's ~38-70 corruption threshold, and not near any reset call at all (this crash is in the CONFIGS loop, which has zero resets). Checked `dmesg`/ECC error counters — clean, ruling out a hardware fault; GPU memory fully freed after the crash. This confirmed session 1's suspicion that a single process accumulates enough Dynamo/Inductor/CUDA state across many algorithms to eventually corrupt, **independent of `reset()` usage**.

**Implemented in `tests/bench_release.py`:**
- New CLI flags: `--output-json PATH` (dump this invocation's `full_tables`/`headline_tables`/`production_rows_all`/`all_violations` as JSON instead of writing benchmarks.md/README, used by subprocess children) and `--parent-sweep` (spawn one subprocess per algorithm group, merge results, then do the normal finalize).
- `_PARENT_SWEEP_GROUPS`: `[("gae",["gae"]), ("vtrace",["vtrace"]), ("retrace",["retrace"]), ("returns",["lambda_returns","discounted_returns","eligibility_traces"]), ("prefix_sum",["prefix_sum"])]` — the returns trio stays together since they already share one `bench_returns()` call (one shared CONFIGS loop over 3 compiled objects); splitting them would triple redundant input construction for no isolation benefit.
- `_run_parent_sweep(selected_algos, args)`: for each group, `subprocess.run([sys.executable, script, "--algos", ..., "--gpu", ..., "--no-update", "--output-json", tmp_path])`; aborts loudly (`sys.exit`) if any subprocess fails — no silent partial results. Merges JSON payloads across groups.
- `_finalize(full_tables, headline_tables, production_rows_all, all_violations, args)`: extracted from the tail of `main()` (builds the one combined production table, prints gate summary, writes benchmarks.md/README unless `--no-update`) — now shared by both the normal single-process path and `_run_parent_sweep`, so there's exactly one finalize code path regardless of how the tables were assembled.
- `main()`'s CHILD path: right before building the (per-invocation) production table, if `--output-json` is set, dumps the payload and returns early — does NOT build a production table itself (that only happens once, in `_finalize`, after the parent merges every group).

**Not yet run end-to-end** (`--parent-sweep` has not been exercised against the full 7-algorithm sweep) because bug #4 (next section) was discovered first, in the *returns* subprocess-equivalent path, and blocks it regardless of process isolation.

---

## 6. NEW, unresolved bug #4 — where this session is actually stuck

While re-validating prefix-sum in isolation after fixing its algorithm bug (§4), hit a **third, distinct** crash:

```
RuntimeError: CUDA error: an illegal memory access was encountered
```
...raised inside `torch/_inductor/runtime/triton_heuristics.py`'s `benchmark_all_configs`/`autotune_to_one_config` (Inductor's own kernel-autotuning harness, timing candidate Triton kernel configs via `triton.testing.do_bench`'s `torch.cuda.synchronize()`), for a fused kernel named `triton_poi_fused_add_fill_flip_lift_fresh_mul_roll_5` — the `flip`/`roll` in that name points at `parallel_prefix_scan`'s `torch.flip(parallel_suffix_scan(torch.flip(a,[1]), torch.flip(b,[1])), [1])` implementation.

**Key evidence:**
- This is a **different crash mechanism** than bug #2/#3 — not near any `reset()` call, and reproduces with very few total resets in the process.
- It affects the **only two callers of `parallel_prefix_scan`**: eligibility-traces (crashed here in the §5 full-sweep run) and prefix-sum (crashed here after the §4 fix moved it onto the same helper).
- **Nondeterministic in exactly where it triggers**: `python tests/bench_release.py --algos prefix_sum --no-update` crashed at the 6th CONFIGS shape `(512,128)` in one run, and at the 3rd shape `(256,1024)` in another run of the *same* code (calling `bench_prefix_sum()` directly, skipping `_precompile_triton_kernels()`). This points to a nondeterministic memory-safety bug (e.g. an out-of-bounds/uninitialized-memory read in an Inductor-generated candidate kernel, whose corruption-visibility depends on incidental allocator state) rather than a deterministic logic error.
- **Does NOT reproduce in a minimal standalone script**: compiling `parallel_prefix_scan` directly, or `vectorized_episodic_prefix_sum` directly, across the same 6-shape sequence with no other context, both ran clean with zero crashes. The crash needs the fuller `bench_release.py` execution context (interleaved raw Triton kernel launches + `torch.compile`'d calls + correctness checks in the same loop iteration) to manifest — not yet narrowed down further than that.
- Ruled out `_precompile_triton_kernels()` as the sole trigger: removing it from the repro (calling `bench_prefix_sum()` directly without it) still crashes, just at a different shape.

**This is NOT one of session 1's three original bugs and NOT fixed by anything in Options A/B/C.** It looks like a genuine Triton/Inductor codegen or autotuning-harness bug in this environment (torch 2.4.1+cu124) specific to the flip/roll-heavy fused kernel `parallel_prefix_scan` generates, with a nondeterministic manifestation point.

---

## 7. Where this was paused — awaiting user direction

I presented four options to the user and was interrupted before getting an answer (session paused here, mid-decision, per user request to persist state instead):

1. **Rewrite `parallel_prefix_scan` to avoid its double-flip pattern** (e.g. a direct suffix-scan-style forward formula) — may dodge whatever Inductor codegen path is crashing. Bounded, concrete, keeps the compiled baseline for both algorithms it's used by.
2. **Fall back to eager (uncompiled) baseline for just eligibility-traces and prefix-sum**, documented as a known limitation — unblocks the sweep immediately but understates torch.compile's real speedup for these two algorithms specifically.
3. **Keep investigating the Inductor bug directly** (disable specific fusions, try a different torch/triton version, etc.) — open-ended, no guaranteed fix.
4. **Pause entirely** for separate user review before deciding.

**No decision has been made yet.** Resume by asking the user which option they want, or by re-presenting this exact menu.

---

## 8. Concrete next steps, in order

1. Get a decision on §7's options (or infer from further user instructions if given before this file is read).
2. Once bug #4 is resolved (however it's resolved): re-run `python tests/bench_release.py --algos eligibility_traces,prefix_sum --no-update` to confirm both pass cleanly (correctness + no crash) in isolation first.
3. Then run the full `--parent-sweep` (not yet exercised end-to-end): `python tests/bench_release.py --parent-sweep --gpu "NVIDIA H100 80GB HBM3" --skip-readme` — verify no subprocess fails, all correctness gates pass, benchmarks.md gets written.
4. Recalibrate the H100 floor table (`bench_safeguard.py`'s `_FLOOR_TABLE`) against the newly-correct numbers — the existing proposal from session 1 is void (built on broken baselines); prefix-sum's floor in particular needs fresh calibration since its baseline was never correct before. **STOP-and-report, do not commit floors** — propose only, same as every prior session.
5. Regenerate the README draft (`benchmarks/readme_table_draft.md`) from the corrected benchmarks.md.
6. Report back: full correct sweep output, confirmation baseline correctness passed at every shape (all 7 algorithms + both truncation paths + production regime), and the new floor table for approval.

## Constraints carried forward (same as every prior session)

- Keep baseline correctness checks (`assert_correctness` on the vec/compiled side) permanently in every sweep — they are what caught bug #2, bug #4, AND the prefix-sum algorithm bug. Do not remove or weaken them.
- Do NOT run any concurrent GPU process while a sweep is live.
- Do NOT cap scope / ship partial. `benchmarks.md` must be fully regenerated against correct baselines (now including the prefix-sum fix) before anything is published.
- Nothing merged to `main`, `src/` untouched, `README.md` untouched, no floors committed without explicit approval — same STOP-and-report boundary as sessions 1 and 2.
