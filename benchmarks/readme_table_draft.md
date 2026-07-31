<!-- README_BENCH_DRAFT: NOT auto-applied. Prepared 2026-07-31 on NVIDIA RTX 2000 Ada Generation. -->

All rows at num_envs=4096, seq_len=128 (PufferLib/Gigaflow-default rollout size) for a genuine apples-to-apples comparison across algorithms.

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 6.3× |
| V-Trace | 6.8× |
| Retrace | 1.6× |
| lambda-returns | 5.7× |
| discounted-returns | 6.0× |
| eligibility-traces | 2.4× |
| prefix-sum | 2.4× |

With truncations (terminations + time-limit truncations + bootstrap values), same config. Eligibility-traces and episodic-prefix-sum have no row here, for a structural reason, not an unwired gap: both kernels take only a single `dones` flag with no terminated/truncated distinction and no bootstrap values, so there is no truncation-path baseline to compare against for either. Every other algorithm, including Retrace (terminated/truncated are both mandatory, distinct arguments, and no separate bootstrap_values parameter -- the continuation value is folded into next_q_values_all every step, see docs/kernels/retrace.md §4), has a row below.

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 3.4× |
| V-Trace | 5.7× |
| Retrace | 1.7× |
| lambda-returns | 4.8× |
| discounted-returns | 4.8× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.