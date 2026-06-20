# Episodic Prefix Sums (Segmented Scans)

## Introduction

While algorithms like GAE and V-Trace accumulate values *backward* in time, many core infrastructure tasks in reinforcement learning and sequence modeling require accumulating values *forward* in time.

The **Episodic Prefix Sum** (known in computer science literature as a **Segmented Scan**) computes a running total across an array, but resets the sum whenever a boundary condition is met — such as an episode ending or a padding token appearing.

By mapping this segmented logic into a Triton associative scan kernel, we avoid Python sequential loops and standard PyTorch masking overhead, enabling parallel computation for data loaders, memory buffers, and token packing pipelines.

---

## 1. The Mathematical Recurrence

A standard prefix sum computes a running total:

$$C_t = x_t + C_{t-1}$$

The sum naturally resets at an episode boundary because $x_t = 0$ and $C_t = 0$ beyond it. In a rollout buffer containing multiple episodes, the kernel introduces a binary boundary flag $d_t$ to make this explicit:

$$C_t = x_t + (1 - d_t)\, C_{t-1}, \qquad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

When $d_t = 1$, the factor $(1 - d_t)$ evaluates to $0$, multiplying the entire accumulated sum by zero and restarting the count with just the current $x_t$.

Unlike the backward algorithms (GAE, V-Trace, Retrace), the distinction between terminated and truncated steps does not matter here — neither carries a bootstrap value that must enter the recurrence. Both cases reset $C_t$ identically.

---

## 2. Mapping to the Associative Scan

This is a first-order linear recurrence $f(x) = \alpha + \beta x$, using the same universal associative operator ($\oplus$) as the advantage estimators:

$$(\alpha_B, \beta_B) \oplus (\alpha_A, \beta_A) = (\alpha_B + \beta_B \alpha_A, \,\, \beta_A \beta_B)$$

The inputs map to hardware tuples $(\alpha_t, \beta_t)$ as follows:

* **Value to accumulate ($\alpha_t$):** $\alpha_t = x_t$
* **Boundary reset mask ($\beta_t$):** $\beta_t = 1 - d_t$

Unlike the backward recurrence in GAE or Retrace, this is a **forward** accumulation. Threads pull data from lower memory indices (the chronological past), executing the scan strictly left-to-right. The reduction tree mechanics are identical to those described in the [GAE tutorial](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree); only the scan direction differs.

---

## 3. Applications

The episodic prefix sum appears wherever a running accumulation must reset at logical boundaries within a packed array. In vectorized RL logging, environments stack rewards from multiple episodes into one buffer; passing this through the scan with `terminateds | truncateds` as $d_t$ recovers each episode's total return — the scan value at every boundary step — without any Python-side episode tracking. For time-aware RL (see Pardo et al. (2018)), a tensor of $1.0$s passed through the scan produces correct per-step timestep indices that reset to $1$ at each episode boundary, replacing per-environment integer counters in Python. For prioritized experience replay (Schaul et al., 2016), a prefix sum over a flat priority array produces the cumulative distribution function needed for GPU-side parallel sampling via binary search, replacing the CPU-bound sum-tree entirely. The same reset structure also underlies sequence packing in LLM pre-training (resetting `position_ids` at `<EOS>` tokens for RoPE embeddings), Decision Transformer trajectory packing (Chen et al., 2021), and Mamba's document-boundary reset gates (Gu & Dao, 2023) — all structurally identical to the $d_t$ mask used here.

---

## References

* Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). *Time Limits in Reinforcement Learning.* ICML 2018. arXiv:1712.00378.
* Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. arXiv:1511.05952.
