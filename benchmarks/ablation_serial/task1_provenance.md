# Task 1: Provenance of benchmarks.md v0.1.3 "Production regime" tables

**Question.** Were the tables generated *after* the fix that moved zero-tensor allocation out
of the timed compiled closure (`compiled_extra_fn` in `tests/bench_release.py`)?

**Method.** Git history only (`git log -S`, `git log -L`, `git merge-base`). Nothing was rerun.

| Event | Commit | Date |
|---|---|---|
| Fix introduced (`compiled_extra_fn`; "production-regime wrapper allocated zero truncateds/bootstrap_values via torch.zeros_like inside the timed closure … Fixed by moving allocation outside the timed region for GAE, V-Trace, lambda-returns, and discounted-returns") | `c5695c5c14fede4325008295ccf8b25298fb7989` | 2026-08-04 13:02:39 +0000 |
| First tables produced with the fix (v0.1.2 re-promote) | `21aedc2` "bench: re-promote v0.1.2 with the production-regime allocation fix" | 2026-08-04 20:25:06 +0100 |
| H100 v0.1.3 sweep committed to `unreleased.md` (section header: "Measured on NVIDIA H100 80GB HBM3 · 2026-08-08") | `90d3c26504e64a66913d22240c3833649f6f00f3` | 2026-08-08 13:02:24 +0000 |
| RTX 2000 Ada v0.1.3 sweep committed | `12b714b` | 2026-08-08 12:34:38 +0000 |
| **Last change to the production-regime tables in benchmarks.md** (v0.1.3 promote; `git log -L` on both production-regime table ranges) | `3e7088776db4cfaed7ce642f87913631571547ce` "Promoted benchmarks for v0.1.3" | 2026-08-08 14:13:30 +0100 |

**Checks.**

- `git merge-base --is-ancestor c5695c5 90d3c26` succeeds: the fix is an ancestor of the
  sweep commit.
- `tests/bench_release.py` at `90d3c26` contains `compiled_extra_fn` (11 occurrences).
- The only commits that touched `tests/bench_release.py` between the fix and the promote are
  `7d626b7` and `8267168` (dones-convention fixes, 2026-08-07). Neither reverts the allocation fix.
- The v0.1.3 H100 GAE production numbers differ from the v0.1.2 numbers they replaced. For
  example, 4096×80 went from 0.0634 ms to 0.0331 ms. So they are a fresh sweep, not carried
  over from before the fix.

**Verdict: after fix.**

**Limit.** Git records when the results were *committed*. It cannot prove which working tree
the sweep process actually ran from. But the staged candidate's own "Measured … 2026-08-08"
header is dated four days after the fix landed.
