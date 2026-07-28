# Benchmark results index

This directory holds benchmark *scripts*; the results they produce live in a few different
places depending on scope. This file is a hand-maintained index and is not touched by any
generator — safe to read as a stable map even right after a release sweep regenerates the
files it points to.

| What | Where | Produced by |
|---|---|---|
| Headline release sweep (all 7 algorithms, all timing granularities, production regime) | [`benchmarks.md`](../benchmarks.md) (repo root) | `tests/bench_release.py` |
| README summary table draft (human applies manually) | [`readme_table_draft.md`](readme_table_draft.md) | `tests/bench_release.py` |
| PufferLib comparison (GAE + V-Trace, both regimes) | [`pufferlib.md`](pufferlib.md) | `compare_pufferlib.py` / `benchmark_gae_vs_pufferlib.py` (see that file's reproducibility note) |
| Chunked-vs-flat dispatch robustness study (long seq_len, out of target regime) | [`chunked_scan.md`](chunked_scan.md) | `compare_chunked.py` |
| PPO end-to-end measurement (GAE's share of a real training step) | [`../docs/benchmark-history/ppo-e2e-measurement-2026-07-25.md`](../docs/benchmark-history/ppo-e2e-measurement-2026-07-25.md) | `ppo_e2e_measurement.py` |
| Prior release history (archived on each new release) | [`../docs/benchmark-history/`](../docs/benchmark-history/) | `tests/bench_release.py` (archives the outgoing section automatically) |
| Why the code is the way it is (root causes, correctness bugs, design tradeoffs found while benchmarking) | [`../NOTES.md`](../NOTES.md) | hand-maintained |

## Benchmark placement policy

Four rules govern where a benchmark result is allowed to land, and when:

1. **Per-PR: `tests/bench_safeguard.py` only, publishes nothing to tracked files.** The
   safeguard suite runs on every PR (see `.github/workflows/gpu-tests.yml`'s `safeguard` job)
   and posts its result as a PR comment. It never writes `benchmarks.md` or anything else in
   this repo.
2. **Release candidates are staged, never pushed directly — this is `tests/bench_release.py`'s
   own default, not a CI-only convention.** Running `python tests/bench_release.py` with no
   extra flags writes ONLY to
   [`../docs/benchmark-history/unreleased.md`](../docs/benchmark-history/unreleased.md) (via
   `stage_unreleased_md()`) plus a regenerated `readme_table_draft.md`; it never reads or
   writes `benchmarks.md` or `README.md`. This holds whether you run it by hand or via the
   `release-bench` workflow (`workflow_dispatch` only — no automatic trigger, never pushes to
   `main`; it opens a PR with the staged candidate for human review instead).
3. **Promotion to a real release is a separate, explicit opt-in — `--release --version <tag>`.**
   Only this flag combination writes `benchmarks.md` (as the new "latest release") and archives
   the outgoing section to a version-tagged file, `docs/benchmark-history/<prior-version>.md`.
   `--release` without `--version` is refused outright (nothing should ever land in
   `benchmarks.md` still headed "unreleased"). After promotion, re-running the default (staging)
   command resets `unreleased.md` to the next candidate — it holds at most one not-yet-promoted
   candidate at a time, overwritten wholesale by each new run, never appended to as a running
   log. `README.md` is untouched even by `--release` unless you also omit `--skip-readme`
   (README prose is always a STOP-and-report item for a human, never auto-applied).
4. **`main` is never written to between releases.** Nothing in this policy pushes directly to
   `main` outside of a human merging a PR (the safeguard job comments only; the release-bench
   job opens a PR; promotion is a manual commit a human makes deliberately).

One-off studies in this directory (`pufferlib.md`, `chunked_scan.md`,
`ppo-e2e-measurement-*.md`) are outside this cycle entirely: they stay dated to when they were
run and are **not** regenerated as part of a release — rerun the relevant script by hand if you
need fresh numbers for one of them.
