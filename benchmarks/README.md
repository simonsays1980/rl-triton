# Benchmark results index

This directory holds benchmark *scripts*; the results they produce live in a few different
places depending on scope. This file is a hand-maintained index and is not touched by any
generator -- safe to read as a stable map even right after a release sweep regenerates the
files it points to.

## Directory layout

Four locations, each with one writer:

| Location | Holds | Written by |
|---|---|---|
| `benchmarks/` | benchmark scripts + one-off study write-ups | run by hand |
| `benchmarks.md` (repo root) | latest release sweep only | `tests/bench_release.py --promote` |
| `docs/benchmarks.md` | mkdocs include shim -> root `benchmarks.md` | static, 71 bytes, never hand-edited |
| `docs/benchmark-history/` | archived releases + `unreleased.md` staging | `tests/bench_release.py` |

The rule for one-off study write-ups (not covered by the placement policy below, since
none of them are part of the release cycle): **if `bench_release.py` wrote it, it lives in
`docs/benchmark-history/`; if a human ran a script by hand, it lives in `benchmarks/`.**
`docs/benchmark-history/` therefore holds only version-tagged archives and
`unreleased.md` -- contents matching its name, one writer. The PPO end-to-end studies
used to live there despite being hand-run and explicitly disclaiming membership in the
release cycle in their own headers; they have moved to `benchmarks/` to match the rule.
Their dated result files are paper-specific output, not reproducible from
`ppo_e2e_measurement.py` as currently structured (see that script's docstring), so unlike
this directory's other one-off studies they are gitignored rather than tracked -- the
script is the tracked artifact, kept on disk for reference but not published.

| What | Where | Produced by |
|---|---|---|
| Headline release sweep (all 7 algorithms, all timing granularities, production regime) | [`benchmarks.md`](../benchmarks.md) (repo root) | `tests/bench_release.py` |
| README summary table draft (human applies manually) | [`readme_table_draft.md`](readme_table_draft.md) | `tests/bench_release.py` |
| PufferLib comparison (GAE + V-Trace, both regimes) | [`gae_vs_pufferlib-2026-08-05.md`](gae_vs_pufferlib-2026-08-05.md) | `benchmark_gae_vs_pufferlib.py` -- this script's output is what the paper's PufferLib figures are built from |
| Chunked-vs-flat dispatch robustness study (long seq_len, out of target regime) | [`chunked_scan.md`](chunked_scan.md) | `compare_chunked.py` |
| `HAS_BOOTSTRAP=False` scalar-allocation skip, isolated | [`bootstrap_skip.md`](bootstrap_skip.md) | `measure_bootstrap_skip.py` |
| Retrace register-pressure study (the `seq_len<=2048` dispatch ceiling) | [`retrace_register_pressure.md`](retrace_register_pressure.md) | `measure_retrace_register_pressure.py` |
| PPO end-to-end measurement (GAE's share of a real training step) | not tracked -- paper-specific dated output, see `ppo_e2e_measurement.py`'s docstring and `.gitignore` | `ppo_e2e_measurement.py` |
| Prior release history (archived on each new release) | [`../docs/benchmark-history/`](../docs/benchmark-history/) | `tests/bench_release.py` (archives the outgoing section automatically) |
| Why the code is the way it is (root causes, correctness bugs, design tradeoffs found while benchmarking) | [`../NOTES.md`](../NOTES.md) | hand-maintained |

## Benchmark placement policy

Four rules govern where a benchmark result is allowed to land, and when:

1. **Per-PR: `tests/bench_safeguard.py` only, publishes nothing to tracked files.** The
   safeguard suite runs on every PR (see `.github/workflows/gpu-tests.yml`'s `safeguard` job)
   and posts its result as a PR comment. It never writes `benchmarks.md` or anything else in
   this repo.
2. **Release candidates are staged, never pushed directly -- this is `tests/bench_release.py`'s
   own default, not a CI-only convention.** Running `python tests/bench_release.py` with no
   extra flags writes ONLY to
   [`../docs/benchmark-history/unreleased.md`](../docs/benchmark-history/unreleased.md) (via
   `stage_unreleased_md()`) plus a regenerated `readme_table_draft.md`; it never reads or
   writes `benchmarks.md` or `README.md`. This holds whether you run it by hand or via the
   `release-bench` workflow (`workflow_dispatch` only -- no automatic trigger, never pushes to
   `main`; it opens a PR with the staged candidate for human review instead).
3. **Promotion to a real release is a separate, explicit, mechanical step -- `--promote --version
   <tag>` -- and the ONLY way anything reaches `benchmarks.md`.** There is no direct-from-sweep
   path to `benchmarks.md` in `tests/bench_release.py`; a fresh sweep only ever stages (rule 2).
   `--promote` relocates the currently staged candidate
   ([`unreleased.md`](../docs/benchmark-history/unreleased.md)) into `benchmarks.md` as the new
   latest release: it archives the outgoing release to a version-tagged file,
   `docs/benchmark-history/<its-own-version>.md` (version parsed from that outgoing section's
   own heading, never from `--version`, never guessed), rewrites the staged section's heading
   from "unreleased" to `<tag>`, and resets `unreleased.md` to its stub. It runs no benchmarks
   and needs no GPU -- it only relocates already-measured, already-reviewed numbers. `--promote`
   without `--version` is refused outright, as is promoting when there is no staged candidate,
   when the outgoing release's own header doesn't parse as a real version tag, or when the
   archive destination already exists (release history is immutable -- never silently
   overwritten). Running it twice in a row is safe: the second run finds `unreleased.md` already
   back at its stub and refuses with nothing left to promote. `unreleased.md` holds at most one
   not-yet-promoted candidate at a time (one section per GPU staged, upserted by GPU label --
   never appended to as an unbounded log). `README.md` is untouched by `--promote` -- pasting from
   `readme_table_draft.md` is always a manual, human step (see rule 2's link to that file).
4. **`main` is never written to between releases.** Nothing in this policy pushes directly to
   `main` outside of a human merging a PR (the safeguard job comments only; the release-bench
   job opens a PR; promotion is a manual commit a human makes deliberately).

Tracked one-off studies in this directory (`gae_vs_pufferlib-*.md`, `chunked_scan.md`,
`bootstrap_skip.md`, `retrace_register_pressure.md`) are outside this cycle entirely: they
stay dated to when they were run and are **not** regenerated as part of a release -- rerun
the relevant script by hand if you need fresh numbers for one of them. The PPO end-to-end
studies (`ppo_e2e_measurement.py`) follow the same rerun-by-hand model but their dated
`.md` output is gitignored, not tracked (see above).
