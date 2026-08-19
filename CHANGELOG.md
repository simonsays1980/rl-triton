# Changelog

All notable changes to rl-triton are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [Unreleased]

## [0.1.3] - 2026-08-17

### Added
- Root `LICENSE` (MIT) and `benchmarks/pufferlib_ext/LICENSE-PufferLib`, so the
  vendored, verbatim-copied PufferLib CUDA source (`pufferlib.cpp`/`pufferlib.cu`)
  carries its own copyright notice as MIT requires. `pyproject.toml` already
  declared the MIT classifier without a corresponding file.
- GitHub Pages publication for the mkdocs site: a `docs.yml` workflow
  (`mkdocs gh-deploy` on push to `main`, plus manual `workflow_dispatch`) and a
  `docs` extra in `pyproject.toml` for the mkdocs toolchain.
- A "Directory layout" section in `benchmarks/README.md` clarifying the four
  locations benchmark results can live in and who writes each, plus a stated
  rule for one-off study write-ups: hand-run scripts' output lives in
  `benchmarks/`, `bench_release.py`-written output lives in
  `docs/benchmark-history/`.

### Changed
- Moved the two PPO end-to-end study files (`ppo-e2e-measurement-2026-07-25.md`,
  `ppo-e2e-single-puffer-op-point-2026-08-13.md`) from `docs/benchmark-history/`
  to `benchmarks/` to match the new one-off-study placement rule -- both were
  hand-run and explicitly disclaimed membership in the release cycle in their
  own headers. Neither matches `ppo_e2e_measurement.py`'s current report
  format closely enough to keep tracked as-is, so both are now gitignored
  (kept on disk for reference); the script itself remains tracked.
- `docs/index.md`'s Kernels, Installation, Usage, and Testing sections are now
  pulled live from `README.md` via `include-markdown`, instead of being
  hand-duplicated copies that had already drifted (the docs copy of the
  kernels table had gained a column the README didn't have).
- `mkdocs.yml`: added `Home: index.md` as the first nav entry so the homepage
  is reachable from the site navigation; added `exclude_docs` so
  `mkdocs build`/`gh-deploy` never pick up non-doc files that happen to be
  sitting in `docs/` on disk, regardless of git tracking state.

### Removed
- `benchmarks/compare_pufferlib.py` and its output `benchmarks/pufferlib.md`
  are no longer tracked (kept on disk, gitignored). Superseded by
  `benchmark_gae_vs_pufferlib.py` / `gae_vs_pufferlib-2026-08-05.md`, which is
  what the paper's PufferLib figures actually cite; `pufferlib.md`'s own
  header already documented its numbers as not reproducible from the
  committed script.
- `docs/paper.md`, a markdown mirror of the paper that was tracked but unlinked
  from `mkdocs.yml`'s nav and contained a personal email address in plaintext.
  No longer tracked (kept on disk, gitignored).

## [0.1.2] - 2026-08-05

### Fixed
- **GAE and V-Trace double-counted the window-boundary bootstrap value.** The
  backward scan's additive boundary carry (`A[T]` / `Δ[T]`) was seeded with
  the window bootstrap `V(s_T)` in addition to using it inside
  `delta[T-1]`/`α[T-1]` as the next-state value. An advantage/value-delta
  carry represents trace mass past the buffer, of which there is none, so
  seeding it with a value double-counted `V(s_T)` by
  `(gamma*lambda)^(T-t) * V(s_T)` at every position. The error is silent
  (finite, plausible-looking output, not NaN/inf) and vanishes when
  `V(s_T)=0` (episode terminates at the window edge), which is why most
  existing tests passed despite the bug. Verified against the
  implementation-independent identity that at `lambda=1`, GAE must telescope
  to the Monte-Carlo advantage `A_t = G_t - V(s_t)`. V-Trace has the same bug
  class (on-policy, `rho=c=1`, its recurrence reduces to GAE at `lambda=1`).
  Retrace already seeded its carry at 0; lambda-returns and discounted-returns
  are structurally correct and unaffected (their analogous carry legitimately
  carries a nonzero weight that sums to 1 with the in-`delta`/`alpha` term,
  rather than overlapping with it). Adds regression tests that check the
  recurrence against an independently hand-rolled Monte-Carlo return rather
  than the sequential oracles, so a future regression can't share the
  oracles' own bug. `docs/kernels/gae.md` and `docs/kernels/vtrace.md`
  corrected to match -- both previously documented the double-counting
  behavior as intentional, dual-purpose design.
- **Production-regime benchmark table reported an inflated speedup for GAE,
  V-Trace, lambda-returns, and discounted-returns**, disagreeing with the
  main config-grid table by up to ~24% at shared shapes (e.g. GAE at
  `num_envs=4096, seq_len=128`: 2.97x production-regime vs. 2.4x main-grid
  on RTX 2000 Ada). Cause: the production-regime measurement's eager
  wrapper allocated the zero truncateds/bootstrap-values tensors it feeds
  the `torch.compile` baseline *inside* the timed closure, so that
  allocation ran on every timed iteration, while the main-grid table
  allocates its equivalent tensors once per shape, outside the timed
  region -- costing the baseline ~14% of its measured full-call time.
  Fixed by building those tensors once per shape in both tables alike. The
  two tables now agree at shared shapes on both RTX 2000 Ada and H100. See
  `NOTES.md` for the full investigation, including the ruled-out
  Dynamo/Inductor cross-shape-state hypothesis. v0.1.2 has been
  re-promoted with corrected numbers; the pre-fix v0.1.2 content is
  preserved at `docs/benchmark-history/v0.1.2.md`.

### Added
- `np->triton->np` (NumPy adoption-path) baseline timing for episodic prefix
  sum, lambda-returns, discounted-returns, and eligibility-traces -- GAE,
  V-Trace, and Retrace already had this baseline; the other four were
  missing it with no documented reason. Episodic prefix sum additionally
  gained the `loop(gpu)`/`numpy(cpu)` baselines every other algorithm already
  had. All seven algorithms now report the identical set of baseline
  comparisons.

### Changed
- Unified the column ordering of every benchmark table and console printout:
  all raw timing values first, then all speedup ratios. Previously GAE,
  V-Trace, and Retrace interleaved value/ratio pairs per baseline while the
  remaining algorithms used a block layout -- and even the block layout mixed
  both conventions within a single row (block for the first baseline,
  interleaved for `loop`/`numpy`). One convention everywhere now.
- Removed the `torch.zeros(num_envs)` bootstrap/seed-default allocation from
  the no-bootstrap/no-seed kernel path across all kernels (GAE, V-Trace,
  Lambda Returns, Discounted Returns, Eligibility Traces, Prefix Sum, and the
  shared scan fallback), via a `HAS_BOOTSTRAP`/`HAS_SEED` compile-time flag
  that substitutes a literal `0.0` instead. Eliminates an extra CUDA kernel
  launch that previously cost 28-40% of total op time at small sizes.
  Bit-identical output verified for every kernel. `bench_safeguard.py` floors
  recalibrated accordingly (e.g. GAE 1.4x → 1.9x, Prefix Sum flips from a
  0.75x non-regression guard to a genuine win with a 1.1x floor). Prefix Sum's
  safeguard gates on median rather than min speedup, since its short duration
  makes the min exposed to single-trial GPU clock-ramp transients that don't
  reflect its real (~1.24x median) performance.

## [0.1.0] - 2026-06-08

### Added
- GAE kernel: fused backward associative scan, 1.6x over torch.compile at 128×1024
- V-Trace kernel: fused IS-weighted scan, 1.8x over torch.compile at 128×1024
- Retrace kernel: 2.2x over torch.compile at 128×1024
- Lambda Returns kernel: 1.6x over torch.compile at 128×1024
- Discounted Returns kernel: 1.3x over torch.compile at 128×1024
- Eligibility Traces kernel: 1.6x over torch.compile at 128×1024
- Episodic Prefix Sum kernel: cumulative sum with done-mask episode resets
- Safeguard benchmark suite enforcing minimum speedup thresholds
- PyTorch wrappers for all kernels with full docstrings
