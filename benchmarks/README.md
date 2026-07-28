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
