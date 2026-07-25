# Session report — to Claude Opus

**From:** Claude Sonnet 5, on branch `bench/h100-release` (mirrored to `h100-benchmark-sweep`), NVIDIA H100 80GB HBM3.
**Date:** 2026-07-25.
**All work is committed and pushed to both branches at `e91ca2c`. Nothing merged to `main`. Nothing in `src/` was touched (permission-locked to me this session; also a STOP-and-report boundary in the original task brief).**

This is a handoff/continuity note in case you pick this branch up next. It covers what was actually done, what I got wrong before getting it right (worth knowing so you don't re-trust an early wrong number if you see it in scrollback), and what's still open.

---

## 1. What this session covered

Original task: extend `tests/bench_release.py` into a full H100 release benchmark — production-regime sweep (high-N/short-T, matching GPUDrive/Gigaflow/PufferLib-style massively-parallel sim RL), two timing granularities (full-call headline + device-only diagnostic + amortized variant), tolerance-based correctness gates, monotonicity gates, PufferLib comparison, PPO end-to-end measurement, `bench_safeguard.py` CI gates. Full details/rationale for that phase are in `docs/benchmark-history/smoke-test-gae-only-2026-07-25-NOTE.md` and the commit messages from `bac9325` through `868188b`.

After that landed, the user asked follow-up questions that turned into real additional work:

1. **PufferLib benchmarks were initially skipped** (pip `pufferlib` not installed) — user asked to actually run them. I wired a vendored-JIT-build fallback into `compare_pufferlib.py`, then the user correctly pushed back: that contradicted the "no vendoring" design for that script. Resolved by: `compare_pufferlib.py` now imports the real pip package ONLY (skips cleanly without it); the vendored, sha256-pinned kernel lives at `benchmarks/pufferlib_ext/` (moved there from `tests/pufferlib_ext/` per explicit request — nothing in CI/bench_release/test suite depends on it, verified via `pytest --collect-only`); real PufferLib numbers were produced via a throwaway, **uncommitted** scratch runner that reuses `compare_pufferlib.py`'s functions with the vendored kernel substituted in directly, and written into `benchmarks/pufferlib.md` as a one-time study with the method explicitly noted (so a future plain re-run of the committed script won't reproduce that path unless pip `pufferlib` is actually installed).

2. **Retrace long-seq_len discussion** → led to reading the actual kernel/ops source (register spill at seq_len>2048, the `_TRITON_SEQ_LEN_CEILING=2048` reroute, why truncation doesn't help — none of this was new work, just explaining what's already documented in `tests/h100_short_horizon_l2_retrace_ppo_report.md` and `src/rl_triton/ops/retrace.py`'s own comments).

3. **Two new one-off studies, built and run this session:**
   - `benchmarks/compare_chunked.py` → `benchmarks/chunked_scan.md` — never-before-tested flat/chunked kernel dispatch (`_FLAT_MAX_SEQ_LEN=131072`) at seq_len 65536–524288, for all 5 backward-scan algorithms.
   - `benchmarks/compare_pufferlib.py` V-Trace section → `benchmarks/pufferlib.md` — genuine V-trace-vs-V-trace comparison (real importance ratios), not the GAE-degenerate mode the original comparison used.

---

## 2. Findings (the actual numbers)

**Chunked dispatch vs. torch.compile** (`benchmarks/chunked_scan.md`): Triton's flat→chunked auto-dispatch stays correct (verified against the exact sequential reference) across all 5 algorithms at every length up to 524288. The headline finding: **the "vec" compiled baselines used everywhere else in this project (log-space cumsum / suffix-product) are numerically invalid — inf/nan from float32 underflow — at every tested length ≥65536, for every single algorithm.** This is universal, not algorithm-specific. There is currently no valid fast PyTorch comparison at this scale with the existing baseline formulation (a correct-but-slow O(T) Python loop exists and was used as the correctness oracle, ~15-90s/call, not a serious perf contender). See my answer in-thread to "not possible or not implemented" — it's an implementation gap (a chunked/blocked log-space scan, mirroring the Triton kernel's own strategy, would fix it), not a fundamental limitation, and it wasn't built because seq_len>131072 is explicitly out of scope for the project's real target regime.

**PufferLib V-Trace** (`benchmarks/pufferlib.md`): Production regime: rl-triton wins 1.08×–2.78× full-call (smaller margin than GAE's 1.62×–5.17×, because rl-triton's `compute_vtrace_fused` computes two outputs — targets and advantages — per call vs. PufferLib's one; noted explicitly in the file). Boundary marker (seq_len 8-32): PufferLib wins at the shortest horizons, same capability/design tradeoff pattern as GAE. Equivalence gate: PufferLib's `advantages` output is algebraically identical to rl-triton's `targets - values` (verified exactly, not assumed) once λ=1.0 is used on PufferLib's side to match rl-triton's implicit λ=1 — see §3 for how I initially got this wrong.

**Pre-existing findings this session did NOT change, just surfaced:** three `bench_safeguard.py` perf-floor tests (`test_perf_gae`, `test_perf_lambda_returns`, `test_perf_eligibility_traces`) currently measure below their calibrated floors (~1.5×/1.5×/1.2× vs floors of 1.9×/1.6×/1.85×) — reproducible across two runs, floors left untouched (STOP-and-report item), likely this H100 instance's clock/power state differing from calibration conditions but not confirmed. Retrace has a genuine, already-documented, disclosed register-spill regression above seq_len=2048 (down to 0.16-0.24x pre-mitigation; the shipped `_TRITON_SEQ_LEN_CEILING` reroute caps it at 0.38-0.47x, still a loss, not restored to a win).

---

## 3. Mistakes I made and caught before trusting the numbers — read this before trusting anything upstream of the final commit

Worth knowing so you don't have to rediscover these if you look at earlier scrollback or draft code:

1. **First `compare_chunked.py` attempt crashed**: I assumed the "vec" baselines would scale safely to seq_len=524288 since they have no Python loop. Wrong — same float32 log-space underflow issue as finding #1 above. Fixed by switching the correctness ground truth to the exact `_ref_*` sequential references (verified fast enough: 15-90s per shape, one call, not per iteration) and treating the vec baseline's own validity as something to *check*, not assume, per row.

2. **First PufferLib V-Trace equivalence gate failed both sides.** Root cause: I copied `LAMBDA=0.95` from the GAE test reflexively without checking it applies to V-trace — and separately, just plain omitted a `lambda_` term from my derived reference's decay computation entirely. PufferLib's kernel has an extra λ trace-decay parameter that rl-triton's `compute_vtrace_fused` doesn't have (its decay is `gamma*c*(1-dones)`, implicitly λ=1) — the fix was `lambda=1.0` on PufferLib's side, not `0.95`.

3. **Even after that fix, the "direct" rl-triton-vs-PufferLib check still failed** — because I was comparing rl-triton's raw full-window output against the independent reference directly, without accounting for a genuine boundary-convention difference (rl-triton's zero-bootstrap-at-T-1 vs. PufferLib's "column T-1 doesn't structurally exist") that's expected to differ and isn't a bug. The GAE equivalence gate (`benchmarks/benchmark_gae_vs_pufferlib.py`) already has the fix for exactly this: force a termination 2 steps before the window edge to sever the boundary contamination before doing a direct comparison. I'd missed applying that same technique to the new V-trace gate; added it, re-verified at small shapes before re-running the full (expensive) sweep, then it passed with ~1e-6 diffs (matching the GAE case's expected float32 summation-order noise between Triton's parallel scan and PufferLib's sequential one).

4. **A real process gap, not a math bug**: the `tests/`→`benchmarks/` move's sys.path fix for `benchmark_gae_vs_pufferlib.py` was verified working in an earlier turn but never actually got committed — my `git add` at the time listed files explicitly and missed it. Caught and fixed only because a later `git status` surfaced it as still-modified. Worth double-checking `git status` is clean after any commit where you used an explicit file list rather than `git add -u`/`-A`.

**Takeaway if you're continuing this work: don't trust a first correctness-gate pass at extreme scales or with new parametrizations as validation — in three separate cases this session, the first attempt was wrong in a way that only surfaced by actually running it and reading the diff numbers, not by reasoning about the math up front.** Small offline smoke tests (small shapes, fast) before committing to an expensive full sweep saved real time here — recommend keeping that pattern.

---

## 4. Open items / not done

- **Three `bench_safeguard.py` floor failures** (§2) — needs investigation (is this H100 instance's clock state genuinely different from calibration, or something else?), not something I resolved or was authorized to resolve (floor changes are a STOP-and-report item).
- **λ trace-decay parameter for rl-triton's V-trace** — discussed at length in-thread (theoretically valid, matches DeepMind's own reference IMPALA implementation, PufferLib isn't inventing it). Scoped but not implemented (kernel change, `src/`, outside what I could touch). See in-thread discussion for exact file/line-level scope (`kernels/vtrace_fused.py` lines ~125,172; `ops/vtrace_fused.py`; `ops/vtrace.py`). Conclusion reached: adding it would NOT require new performance benchmarking (memory-bandwidth-bound kernel, one extra scalar multiply, `rho_bar`/`c_bar` precedent shows runtime scalars don't trigger recompilation) — only new correctness tests, plus one cheap sanity spot-check to empirically confirm the "performance is λ-invariant" claim rather than just assert it.
- **README draft** (`benchmarks/readme_table_draft.md`) — prepared, not applied. README.md itself untouched all session (STOP-and-report boundary).
- **Two stale comment-only path references** in `src/rl_triton/ops/gae.py` and `retrace_fused.py` (still say `tests/benchmark_gae_vs_pufferlib.py`, should say `benchmarks/...`) — harmless, but I can't fix them (permission-locked). Trivial one-line fix if you have `src/` access.
- **PPO end-to-end measurement** already done earlier in the session (`docs/benchmark-history/ppo-e2e-measurement-2026-07-25.md`): GAE is <0.5% of step time in every config measured (net size × compile mode × TF32), end-to-end speedup ~0.99-1.01x (noise-level). This is a bound for the paper's evaluation section, explicitly not wired into any recurring table, no win/lose conclusion drawn (STOP-and-report item, per the original task brief).

---

## 5. Where everything lives

- `tests/bench_release.py`, `tests/bench_utils.py`, `tests/bench_safeguard.py` — the main release harness + CI gates.
- `benchmarks.md` — latest release only (replace-on-update scheme); prior releases archived to `docs/benchmark-history/<version>.md`.
- `benchmarks/compare_pufferlib.py` → `benchmarks/pufferlib.md` — GAE + V-Trace vs PufferLib, one-time study, pip-package-only (no vendoring).
- `benchmarks/pufferlib_ext/` — vendored PufferLib CUDA source (sha256-pinned, `NOTICE.md` has attribution), used only by the separate historical `benchmarks/benchmark_gae_vs_pufferlib.py`.
- `benchmarks/compare_chunked.py` → `benchmarks/chunked_scan.md` — flat/chunked dispatch vs. compiled baselines at extreme seq_len.
- `benchmarks/ppo_e2e_measurement.py` → `docs/benchmark-history/ppo-e2e-measurement-2026-07-25.md`.
- `docs/benchmark-history/smoke-test-gae-only-2026-07-25-NOTE.md` — the GAE-only smoke test that gated the full sweep.

Good luck.
