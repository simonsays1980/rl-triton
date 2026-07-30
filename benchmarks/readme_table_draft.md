<!-- README_BENCH_DRAFT: NOT auto-applied. Regenerated 2026-07-30 from the promoted benchmarks.md v0.1.1 (both staged GPUs), no new sweep run -- see render_readme_table_draft() for the generator and NOTES.md for the shape-dependence caveat. -->

### NVIDIA H100 80GB HBM3

All rows at num_envs=4096, seq_len=128 (PufferLib/Gigaflow-default rollout size) for a genuine apples-to-apples comparison across algorithms.

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 3.5× |
| V-Trace | 3.4× |
| Retrace | 1.8× |
| lambda-returns | 3.0× |
| discounted-returns | 3.2× |
| eligibility-traces | 2.2× |
| prefix-sum | 2.4× |

With truncations (terminations + time-limit truncations + bootstrap values), same config. Two algorithms have no row here, for two DIFFERENT reasons, stated explicitly rather than silently dropped: Retrace's kernel supports truncations (terminated/truncated are both mandatory, distinct arguments) and has a valid vectorized baseline (`vectorized_retrace`), but that baseline isn't wired into this headline table yet — a real gap to close, not a validity problem. Eligibility-traces and episodic-prefix-sum have no truncation-path concept at all: both kernels take only a single `dones` flag with no terminated/truncated distinction and no bootstrap values, so there is no baseline to compare against for either — a structural non-applicability, not an unwired gap:

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 2.0× |
| V-Trace | 2.8× |
| lambda-returns | 2.5× |
| discounted-returns | 2.5× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.

### NVIDIA RTX 2000 Ada Generation

All rows at num_envs=4096, seq_len=128 (PufferLib/Gigaflow-default rollout size) for a genuine apples-to-apples comparison across algorithms.

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 5.2× |
| V-Trace | 5.7× |
| Retrace | 1.6× |
| lambda-returns | 4.6× |
| discounted-returns | 4.4× |
| eligibility-traces | 2.1× |
| prefix-sum | 2.2× |

With truncations (terminations + time-limit truncations + bootstrap values), same config. Two algorithms have no row here, for two DIFFERENT reasons, stated explicitly rather than silently dropped: Retrace's kernel supports truncations (terminated/truncated are both mandatory, distinct arguments) and has a valid vectorized baseline (`vectorized_retrace`), but that baseline isn't wired into this headline table yet — a real gap to close, not a validity problem. Eligibility-traces and episodic-prefix-sum have no truncation-path concept at all: both kernels take only a single `dones` flag with no terminated/truncated distinction and no bootstrap values, so there is no baseline to compare against for either — a structural non-applicability, not an unwired gap:

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 2.5× |
| V-Trace | 4.5× |
| lambda-returns | 3.7× |
| discounted-returns | 3.7× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.
