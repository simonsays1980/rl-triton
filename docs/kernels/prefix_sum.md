# Episodic Prefix Sums (Segmented Scans)

## Introduction

While algorithms like GAE and V-Trace accumulate values *backward* in time, many core infrastructure tasks in reinforcement learning and sequence modeling require accumulating values *forward* in time.

The **Episodic Prefix Sum** (known in computer science literature as a **Segmented Scan**) computes a running total across an array, but resets the sum whenever a boundary condition is met -- such as an episode ending or a padding token appearing.

By mapping this segmented logic into a Triton associative scan kernel, we avoid Python sequential loops and standard PyTorch masking overhead, enabling parallel computation over rollout buffers and other flat arrays holding many variable-length segments.

---

## 1. The Mathematical Recurrence

A standard prefix sum computes a running total:

$$C_t = x_t + C_{t-1}$$

The sum naturally resets at a segment boundary because $x_t = 0$ and $C_t = 0$ beyond it. In a buffer containing multiple segments, the kernel introduces a binary boundary flag $d_t$ to make this explicit -- but *where* the reset lands relative to $d_t$ depends on what the flag means, and this kernel supports both interpretations via the `boundary` parameter, because its two primary use cases disagree about that.

### `boundary="ends_at"` (default) -- RL rollout buffers

$d_t = 1$ means the segment **ends at** $t$; the reset lands at $t+1$. This is the convention every other kernel in this package uses (`compute_gae`, `compute_eligibility_traces`, etc.), and the convention a raw Gymnasium `terminated`/`truncated` flag already carries without shifting:

$$C_t = x_t + (1 - d_{t-1})\, C_{t-1}, \qquad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}, \qquad d_{-1} := 0$$

When $d_{t-1} = 1$ (the *preceding* step ended a segment), the factor $(1 - d_{t-1})$ evaluates to $0$ at $t$, multiplying the incoming accumulated sum by zero and restarting the count with just the current $x_t$. Use this mode for RL-side accumulation -- episode-return logging, timestep-since-reset counters -- computed from the same `terminated`/`truncated` buffer used elsewhere in a rollout.

### `boundary="starts_at"` -- segment-start flags

$d_t = 1$ means the segment **starts at** $t$; the reset lands at $t$ itself, immediately:

$$C_t = x_t + (1 - d_t)\, C_{t-1}$$

Use this mode when the boundary flag marks a new segment's *first* element directly, so the counter must reset exactly there rather than one step later. This is not the default; pass `boundary="starts_at"` explicitly.

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

**RL rollout accumulation (`boundary="ends_at"`, default).** In vectorized RL, a rollout buffer may contain rewards from multiple episodes. Scanning rewards with $d_t=\texttt{terminateds}\lor\texttt{truncateds}$ recovers each episode's total return at its final step, without Python-side episode bookkeeping. The same mechanism can generate per-step episode-time indices: scanning a tensor of ones resets the count to $1$ at the start of each new episode, providing the time signal used in time-aware RL (Pardo et al., 2018) without per-environment Python counters.

**Segment-start boundaries (`boundary="starts_at"`, explicit).** When a boundary flag marks a new segment's *first* element rather than the previous segment's last, the reset must land at the flagged element itself. Passing `boundary="starts_at"` gives a within-segment counter:

```python
step_in_segment = compute_episodic_prefix_sum(
    inputs=torch.ones_like(segment_starts),
    dones=segment_starts,  # 1.0 at each new segment's first element
    boundary="starts_at",
).to(torch.int64)
```

Mamba (Gu & Dao, 2023, §3.5.2) treats document packing and RL episode boundaries as the same recurrence-reset problem: state must not propagate across marked boundaries in a flat sequence. The equivalence is semantic, not implementation-level -- Mamba resets via its input-dependent selection mechanism rather than an explicit boundary mask.

---

## References

* Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). *Time Limits in Reinforcement Learning.* ICML 2018. arXiv:1712.00378.
* Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
