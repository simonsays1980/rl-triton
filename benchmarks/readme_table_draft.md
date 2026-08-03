<!-- README_BENCH_DRAFT: NOT auto-applied. Prepared 2026-08-03 on NVIDIA RTX 2000 Ada Generation. -->

All rows at num_envs=4096, seq_len=128 (PufferLib/Gigaflow-default rollout size) for a genuine apples-to-apples comparison across algorithms.

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 3.0× |
| V-Trace | 3.3× |
| Retrace | 1.5× |
| lambda-returns | 3.9× |
| discounted-returns | 4.0× |
| eligibility-traces | 2.2× |
| prefix-sum | 2.2× |

With truncations (terminations + time-limit truncations + bootstrap values), same config. Eligibility-traces and episodic-prefix-sum have no row here, for a structural reason, not an unwired gap: both kernels take only a single `dones` flag with no terminated/truncated distinction and no bootstrap values, so there is no truncation-path baseline to compare against for either. Every other algorithm, including Retrace (terminated/truncated are both mandatory, distinct arguments, and no separate bootstrap_values parameter -- the continuation value is folded into next_q_values_all every step, see docs/kernels/retrace.md §4), has a row below.

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 1.4× |
| V-Trace | 2.6× |
| Retrace | 1.6× |
| lambda-returns | 3.4× |
| discounted-returns | 3.5× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.