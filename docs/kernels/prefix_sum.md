# Tutorial: Episodic Prefix Sums (Segmented Scans) in AI Infrastructure

## Introduction

While algorithms like GAE and V-Trace accumulate values *backward* in time, many core infrastructure tasks in reinforcement learning and sequence modeling require accumulating values *forward* in time.

The **Episodic Prefix Sum** (known in computer science literature as a **Segmented Scan**) computes a running total across an array, but resets the sum whenever a boundary condition is met — such as an episode ending or a padding token appearing.

By mapping this segmented logic into a Triton associative scan kernel, we avoid Python sequential loops and standard PyTorch masking overhead, enabling parallel computation for data loaders, memory buffers, and token packing pipelines.

---

## 1. The Mathematical Recurrence

A standard prefix sum computes a running total: $C_t = x_t + C_{t-1}$.

To make it *episodic* (segmented), we introduce a binary terminal flag $d_t$, where $1$ indicates a boundary (e.g., the end of an episode or a document).

The recurrence becomes:

$$C_t = x_t + (1 - d_t) C_{t-1}$$

If the previous step was a terminal state ($d_{t-1} = 1$), the $(1-d_t)$ mask evaluates to $0$, multiplying the entire historical sum by zero and restarting the count with just the current $x_t$.

---

## 2. Mapping to the Associative Scan

This is a first-order linear recurrence $f(x) = u + vx$, using the same universal associative operator ($\oplus$) as the advantage estimators:

$$(u_B, v_B) \oplus (u_A, v_A) = (u_B + v_B u_A, \,\, v_A v_B)$$

We map the inputs for our hardware tuples $(u, v)$ as follows:

* **Value to accumulate ($u_t$):** $u_t = x_t$
* **Boundary reset mask ($v_t$):** $v_t = 1 - d_t$

Unlike the backward recurrence in GAE or Retrace, this is a **forward** accumulation. Threads pull data from lower memory indices (the chronological past), executing the scan strictly left-to-right. The reduction tree mechanics are identical to those described in the [GAE tutorial](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree); only the scan direction differs.

---

## 3. Production Use Cases

Segmented scans appear wherever large contiguous memory blocks must be processed while respecting logical boundaries within that memory.

### A. Vectorized Logging (Total Episodic Return)

When environments stack data from multiple episodes together, computing the total episodic return naively requires tracking which rewards belong to which episode. Because environments reset at unpredictable times, a simple `torch.sum(rewards)` is insufficient.

A forward episodic prefix sum over the rewards tensor creates a running tally of the score. The value of the prefix sum at each index where `done == 1` is the final episodic return for that episode. This computes individual episode scores across all parallel environments in a single kernel launch.

### B. Time-Aware RL (State Augmentation)

In environments with strict time limits, the Markov property is violated unless the agent knows how much time it has left. Standard practice is to append the current timestep to the observation.

Instead of tracking separate integer counters per environment in Python, a tensor of $1.0$s passed through an episodic prefix sum produces the correct per-step timestep index for every environment simultaneously, resetting to $1$ at each episode boundary.

### C. LLM Training: Sequence Packing and `position_ids`

To maximize GPU utilization during LLM pre-training, engineers use **sequence packing**: concatenating multiple independent documents into one long sequence separated by `<EOS>` tokens.

The transformer's positional embeddings (e.g., RoPE) require per-document positions, not global sequence positions — the `position_ids` must reset at every `<EOS>`. Passing an array of $1.0$s through a segmented scan with `<EOS>` locations as the $d_t$ mask produces the correct resetting `position_ids` directly on the GPU.

### D. Offline RL: Decision Transformers and Trajectory Packing

The Decision Transformer (Chen et al., 2021) treats RL as a sequence modeling problem, feeding the model a sequence of `(Return, State, Action)` tokens. Like LLM sequence packing, multiple short trajectories are packed into a single context window and the model requires a per-step timestep embedding that resets at trajectory boundaries. A segmented prefix sum produces these embeddings without CPU-side bookkeeping.

### E. State Space Models (Mamba)

Mamba (Gu & Dao, 2023) replaces attention with an associative scan to achieve linear scaling. When training on packed sequences, the internal hidden state of one document must not bleed into the next. Mamba handles this by treating document boundaries as reset gates — structurally identical to the $d_t$ mask used here.

### F. Prioritized Experience Replay (PER)

To sample from a prioritized replay buffer entirely on the GPU, priorities can be represented as a flat array. A parallel prefix sum over this array produces a cumulative distribution function (CDF), which then supports parallel sampling via binary search — replacing CPU-bound sum trees (Schaul et al., 2016).

---

## References

* Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS 2021. arXiv:2106.01345.
* Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
* Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. arXiv:1511.05952.
