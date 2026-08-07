# Episodic Prefix Sums (Segmented Scans)

## Introduction

While algorithms like GAE and V-Trace accumulate values *backward* in time, many core infrastructure tasks in reinforcement learning and sequence modeling require accumulating values *forward* in time.

The **Episodic Prefix Sum** (known in computer science literature as a **Segmented Scan**) computes a running total across an array, but resets the sum whenever a boundary condition is met -- such as an episode ending or a padding token appearing.

By mapping this segmented logic into a Triton associative scan kernel, we avoid Python sequential loops and standard PyTorch masking overhead, enabling parallel computation for data loaders, memory buffers, and token packing pipelines.

---

## 1. The Mathematical Recurrence

A standard prefix sum computes a running total:

$$C_t = x_t + C_{t-1}$$

The sum naturally resets at a segment boundary because $x_t = 0$ and $C_t = 0$ beyond it. In a buffer containing multiple segments, the kernel introduces a binary boundary flag $d_t$ to make this explicit -- but *where* the reset lands relative to $d_t$ depends on what the flag means, and this kernel supports both interpretations via the `boundary` parameter, because its two primary use cases disagree about that.

### `boundary="ends_at"` (default) -- RL rollout buffers

$d_t = 1$ means the segment **ends at** $t$; the reset lands at $t+1$. This is the convention every other kernel in this package uses (`compute_gae`, `compute_eligibility_traces`, etc.), and the convention a raw Gymnasium `terminated`/`truncated` flag already carries without shifting:

$$C_t = x_t + (1 - d_{t-1})\, C_{t-1}, \qquad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}, \qquad d_{-1} := 0$$

When $d_{t-1} = 1$ (the *preceding* step ended a segment), the factor $(1 - d_{t-1})$ evaluates to $0$ at $t$, multiplying the incoming accumulated sum by zero and restarting the count with just the current $x_t$. Use this mode for RL-side accumulation -- episode-return logging, timestep-since-reset counters, prioritized-replay CDFs -- computed from the same `terminated`/`truncated` buffer used elsewhere in a rollout.

### `boundary="starts_at"` -- sequence packing

$d_t = 1$ means the segment **starts at** $t$; the reset lands at $t$ itself, immediately:

$$C_t = x_t + (1 - d_t)\, C_{t-1}$$

Use this mode when the boundary flag marks a new segment's *first* element directly -- the sequence-packing case below, where a document-boundary flag marks a new document's first token and a RoPE position counter must reset exactly there, not one token later. This is not the default; pass `boundary="starts_at"` explicitly.

Unlike the backward algorithms (GAE, V-Trace, Retrace), the distinction between terminated and truncated steps does not matter in either mode here -- neither carries a bootstrap value that must enter the recurrence. Both cases reset $C_t$ identically.

---

## 2. Mapping to the Associative Scan

This is a first-order linear recurrence $f(x) = \alpha + \beta x$, using the same universal associative operator ($\oplus$) as the advantage estimators:

$$(\alpha_B, \beta_B) \oplus (\alpha_A, \beta_A) = (\alpha_B + \beta_B \alpha_A, \,\, \beta_A \beta_B)$$

The inputs map to hardware tuples $(\alpha_t, \beta_t)$ as follows:

* **Value to accumulate ($\alpha_t$):** $\alpha_t = x_t$
* **Boundary reset mask ($\beta_t$):** $\beta_t = 1 - d_{t-1}$ (`ends_at`, default) or $\beta_t = 1 - d_t$ (`starts_at`)

Unlike the backward recurrence in GAE or Retrace, this is a **forward** accumulation. Threads pull data from lower memory indices (the chronological past), executing the scan strictly left-to-right. The reduction tree mechanics are identical to those described in the [GAE tutorial](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree); only the scan direction differs.

---

## 3. Applications

The episodic prefix sum appears wherever a running accumulation must reset at logical boundaries within a packed array.

**RL rollout accumulation (`boundary="ends_at"`, default).** In vectorized RL logging, environments stack rewards from multiple episodes into one buffer; passing this through the scan with `terminateds | truncateds` as $d_t$ recovers each episode's total return -- the scan value at the step before each new episode begins -- without any Python-side episode tracking. For time-aware RL (see Pardo et al. (2018)), a tensor of $1.0$s passed through the scan produces correct per-step timestep indices that reset to $1$ at the first step of each new episode, replacing per-environment integer counters in Python. For prioritized experience replay (Schaul et al., 2016), a prefix sum over a flat priority array produces the cumulative distribution function needed for GPU-side parallel sampling via binary search, replacing the CPU-bound sum-tree entirely.

**Sequence packing (`boundary="starts_at"`, explicit).** LLM pre-training pipelines pack multiple documents into a single flat sequence and reset `position_ids` at each document's first token for RoPE embeddings -- the reset must land exactly at the flagged token, not one step later, so this application needs `boundary="starts_at"` passed explicitly:

```python
position_ids = compute_episodic_prefix_sum(
    inputs=torch.ones_like(document_boundaries),
    dones=document_boundaries,  # 1.0 at each new document's first token
    boundary="starts_at",
).to(torch.int64)
```

The same structure underlies Decision Transformer trajectory packing (Chen et al., 2021) and Mamba's document-boundary reset gates (Gu & Dao, 2023) -- both also reset *at* the flagged boundary token, so both need `boundary="starts_at"`.

---

## References

* Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). *Time Limits in Reinforcement Learning.* ICML 2018. arXiv:1712.00378.
* Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. arXiv:1511.05952.
