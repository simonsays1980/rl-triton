# Retrace(λ) with Triton Associative Scans

## Introduction

Having successfully accelerated both Generalized Advantage Estimation (GAE) and V-Trace, we can now adapt our $O(\log N)$ parallel associative scan to evaluate action-values using **Retrace($\lambda$)**.

Retrace($\lambda$) (Munos et al., 2016) is a foundational off-policy algorithm that shares the same recursive DNA as V-Trace, but is designed for Q-learning rather than Actor-Critic state values. Because the underlying first-order recurrence remains identical, the Triton kernel requires no changes -- only the PyTorch inputs need adjustment to account for an essential **index shift** in the importance sampling weights.

---

## 1. The Retrace Target: Accumulating TD Errors Over Q-Values

Before mapping Retrace to our hardware, it is worth asking the same question we asked for V-Trace: *Why does Retrace accumulate TD errors rather than raw discounted rewards?*

The answer is again a **Telescoping Sum**, but this time applied to the action-value $Q(s_t, a_t)$ instead of the state value $V(s_t)$.

Recall the Retrace TD error for a single step:

$$\delta_t = r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$$

Adding this error to the baseline $Q(s_t, a_t)$ gives:

$$Q(s_t, a_t) + \delta_t = Q(s_t, a_t) + \left[ r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t) \right]$$

The $Q(s_t, a_t)$ terms cancel instantly, leaving the 1-step return:

$$= r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)]$$

Extending this to two steps:

$$Q(s_t, a_t) + \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 \mathbb{E}_{a \sim \pi}[Q(s_{t+2}, a)]$$

The intermediate $Q$ estimates telescope and collapse into a clean $n$-step return. This telescoping property is what makes the TD-error format algebraically natural for Retrace: we can attach the per-step importance weights $c_{t+1}$ to each individual error term rather than trying to untangle them from a block of summed rewards.

---

## 2. Deriving the Backward Recurrence

The full Retrace target for state-action pair $(s_t, a_t)$ is defined in the paper (Munos et al., 2016) as an infinite-horizon sum over all future steps in the trajectory:

$$Q^{ret}_t = Q(s_t, a_t) + \sum_{k \geq t} \gamma^{k-t} \left( \prod_{i=t+1}^{k} c_i \right) \delta_k$$

where the empty product for $k=t$ is defined as $1$. The **target delta** is defined as:

$$\Delta_t = Q^{ret}_t - Q(s_t, a_t) = \sum_{k \geq t} \gamma^{k-t} \left( \prod_{i=t+1}^{k} c_i \right) \delta_k$$

The windowed implementation truncates this sum at $k = T-1$, the last step available in the rollout buffer. This introduces a truncation bias equal to the dropped tail $\sum_{k \geq T} \gamma^{k-t}\left(\prod_{i=t+1}^{k} c_i\right)\delta_k$, which is geometric in window depth and suppressed by trace clipping ($c_i \leq \lambda \leq 1$), vanishing when the critic is exact at the boundary.

Unrolling the first two terms makes the structure explicit:

$$\Delta_t = \delta_t + \gamma c_{t+1} \delta_{t+1} + \gamma^2 c_{t+1} c_{t+2} \delta_{t+2} + \dots$$

Factoring $\gamma c_{t+1}$ from every term after the first:

$$\Delta_t = \delta_t + \gamma c_{t+1} \left( \delta_{t+1} + \gamma c_{t+2} \delta_{t+2} + \dots \right)$$

The expression in parentheses is exactly $\Delta_{t+1}$, which yields the **first-order backward recurrence**:

$$\Delta_t = \delta_t + \gamma c_{t+1} \cdot \Delta_{t+1}$$

This is the mathematically exact recurrence within a single episode. No done flags appear -- the sum in the Retrace definition naturally terminates at the episode boundary.

In a rollout buffer a single sequence may contain multiple complete episodes or end mid-episode at a truncation boundary. To prevent accumulation from crossing episode boundaries, the kernel introduces masked versions of $\alpha_t$ and $\beta_t$ that depend on two mutually exclusive flags, $d_t^{\text{term}}$ and $d_t^{\text{trunc}}$:

$$\alpha_t = \delta_t = r_t + \gamma\,(1 - d_t^{\text{term}})\,\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$$

$$\beta_t = \gamma c_{t+1}(1 - d_t), \quad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

Section 4 details what this means for the caller.

Notice the critical difference from V-Trace: the decay coefficient multiplying $\Delta_{t+1}$ is $\gamma c_{t+1}$ (next-step weight), not $\gamma c_t$ (current-step weight). This is examined in detail in the next section.

Throughout this document $T$ denotes the **sequence length** (the number of steps in the rollout buffer), not necessarily an episode termination.

---

## 3. The Core Difference: The Index Shift ($c_{t+1}$)

If you look closely at the backward recurrence derived above, the trace weight applied to $\Delta_{t+1}$ is **$c_{t+1}$**, whereas in V-Trace the corresponding weight was **$c_t$**.

This index shift follows directly from what each algorithm estimates:

* **V-Trace** evaluates a state $s_t$. It must correct for the very first action $a_t$ taken from that state, so it applies importance sampling immediately ($c_t$).
* **Retrace** evaluates a state-action pair $(s_t, a_t)$. Because the action $a_t$ is already fixed as the subject of our evaluation, we do not reject or correct it. We only begin correcting for off-policy divergence on the *next* action taken in the trajectory ($a_{t+1}$).

The importance weight $c_t$ itself uses a $\lambda$-scaled clip at $1$ rather than V-Trace's $\bar{c}$ clip:

$$c_t = \lambda \min\left(1, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$$

Setting $\lambda = 1$ recovers the full importance sampling weight; setting $\lambda < 1$ increases the bias–variance trade-off toward lower variance.

---

## 4. Handling Episode Boundaries and the Window Bootstrap

The kernel consumes `q_values`, `next_q_values_all`, `terminateds`, and `truncateds`. It never sees observations. The equations for $\alpha_t$ and $\beta_t$ were established in Section 2; this section describes the caller's responsibility for preparing the input arrays correctly.

Retrace differs structurally from GAE and V-Trace in one important respect: it requires no `bootstrap_values` argument. The continuation value $\mathbb{E}_\pi[Q(s_{t+1},\cdot)]$ is embedded directly in $\alpha_t$ via `next_q_values_all`, which the caller must supply for every step including the final column. This means the scan carry $\Delta_T$ is always zero -- all boundary information is already encoded in $\alpha$ before the scan runs.

### Terminated steps

At a terminated step $t$, $(1 - d_t^{\text{term}})$ zeros $\mathbb{E}_\pi[Q(s_{t+1},\cdot)]$ inside $\alpha_t$ -- correct, a terminal state has no continuation value. $(1 - d_t)$ zeros $\beta_t$, stopping trace propagation into the previous episode.

### Truncated steps

At a truncated step $t$, the episode continues, so the bootstrap $\mathbb{E}_\pi[Q(s_{t+1},\cdot)]$ is kept in $\alpha_t$ -- `next_q_values_all[env, t, :]` holds the correct continuation values and is used directly. $(1 - d_t)$ zeros $\beta_t$ so no accumulation propagates across the boundary.

### Window boundary at $t = T-1$

At the last step of the window, `next_q_values_all[env, T-1, :]` must hold $Q(s_T, \cdot)$ -- the action-values at the state one step past the buffer. The caller supplies this from a fresh critic forward pass when the episode continues past the window, or passes zeros if the episode terminated at $T-1$ (in which case $d_{T-1}^{\text{term}} = 1$ zeros it anyway).

The scan carry $\Delta_T = 0$ is correct in both cases: all future information at the boundary is already encoded in $\alpha_{T-1}$ via `next_q_values_all`. Additionally, the decay coefficient that would propagate a nonzero carry into $\Delta_{T-1}$ is $\beta_{T-1} = \gamma c_T (1 - d_{T-1})$, where $c_T$ requires the action $a_T$ drawn by the behavior policy at $s_T$ -- an action that was never sampled and is not in the rollout buffer. The kernel hardcodes $c_T = 0$, making $\beta_{T-1} = 0$ regardless of $\Delta_T$.

### The unifying rule

The caller supplies `terminateds[env, t] = 1` at true episode endings and `truncateds[env, t] = 1` at time-limit boundaries. The kernel derives $d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$ internally to sever the trace at any boundary, while using `terminateds` alone to gate the bootstrap in $\alpha_t$. The two flags are mutually exclusive: a step cannot be both.

### Boundary summary

| Situation | $d_t$ | $d_t^{\text{term}}$ | Bootstrap in $\alpha_t$ | Trace decay $\beta_t$ |
|---|---|---|---|---|
| **Terminated** | 1 | 1 | Zeroed: $r_t - Q_t$ | $0$ -- scan stops |
| **Truncated** | 1 | 0 | Kept: $r_t + \gamma\mathbb{E}_\pi[Q(s_{t+1},\cdot)] - Q_t$ | $0$ -- scan stops |
| **Window end** | 0 | 0 | Kept: $r_{T-1} + \gamma\mathbb{E}_\pi[Q(s_T,\cdot)] - Q_{T-1}$ | $0$ -- $c_T$ undefined |

**Performance note.** Unlike GAE/V-Trace, `retrace_fused_kernel` has no `HAS_TRUNCATIONS`-style compile-time specialization -- it unconditionally loads both `terminated_ptr` and `truncated_ptr` on every call. Measured directly (H200 SXM, num_envs=4096, seq_len=128): full-call and device time are identical whether `truncateds` is all-zero or has real truncations mixed in (0.0114ms device either way, full-call within noise across repeated runs). Truncation support costs zero marginal runtime here.

---

## 5. Mapping to the Associative Scan

Despite the index shift, the mathematical structure is still a perfect first-order recurrence conforming to our kernel's expected $f(x) = \alpha + \beta x$ transformation.

The inputs map to hardware tuples $(\alpha, \beta)$ as follows:

* **TD accumulation ($\alpha_t$):** $\alpha_t = \delta_t = r_t + \gamma(1-d_t^{\text{term}})\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$
* **Trace decay coefficient ($\beta_t$):** $\beta_t = \gamma c_{t+1}(1-d_t)$

Because $c_{t+1}$ requires looking one step into the future, the fused kernel reads the action probabilities at $t+1$ directly into registers for each timestep rather than pre-shifting the array in PyTorch before the kernel launches.

### Verification via the reduction tree

The reduction tree mechanics are identical to those described in the [GAE tutorial](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree). The following verifies only what is specific to Retrace: how the index shift manifests in the accumulated polynomial. As with GAE and V-Trace, the array is reversed so that threads look "left" to reach chronologically later data.

After the $O(\log N)$ scan, Thread 4 (at reversed index 4, chronological $t=1$) holds the accumulated result $\alpha_{1..4}$. Applying the $\oplus$ operator:

$$\alpha_{1..4} = \alpha_4 + \beta_4 \alpha_3 + \beta_4 \beta_3 \alpha_2 + \beta_4 \beta_3 \beta_2 \alpha_1$$

Substituting our chronological Retrace definitions (reversed index $k$ maps to time $t = k$):

$$\alpha_{1..4} = \delta_1 + (\gamma c_2)\delta_2 + (\gamma c_2)(\gamma c_3)\delta_3 + (\gamma c_2)(\gamma c_3)(\gamma c_4)\delta_4$$

Notice that $c_1$ does not appear anywhere in this polynomial. This is the index shift in action. The Retrace definition assigns each $\delta_k$ a weight equal to $\prod_{i=t+1}^{k} c_i$: the product starts one step ahead of the base step $t$, so the base term $\delta_1$ (where $k=t=1$) carries the empty product and enters unscaled. The first IS correction, $c_2$, only appears on $\delta_2$ -- reflecting that Retrace treats the action $a_1$ being evaluated as fixed and begins correcting for off-policy divergence only from $a_2$ onward.

This is exactly $\Delta_1$ from the unrolled recurrence above: each TD error $\delta_k$ is multiplied by the product of $c$ weights starting at index $k$ (one step ahead of the originating state), precisely matching the $\prod_{i=t+1}^{k} c_i$ product in the Retrace target.

---

## 6. Reconstructing the Q-Value Target and Advantages

Once the $O(\log N)$ scan finishes, the fused kernel computes the Q-value target and the actor advantage in the same Triton program before returning to the host. This mirrors the V-Trace fused kernel exactly.

**Step 3 -- Q-value targets.** The scan result $\Delta_t$ is added to $Q(s_t, a_t)$ in registers before the final write to HBM, avoiding a separate round-trip:

$$Q^{ret}_t = \Delta_t + Q(s_t, a_t)$$

**Step 4 -- Advantages.** After a synchronisation barrier, each thread reloads $Q^{ret}_{t+1}$ from HBM and computes:

$$A_t = \rho_t \left( r_t + \gamma\,(1 - d_t^{\text{term}})\,Q^{ret}_{t+1} - Q(s_t, a_t) \right)$$

where $\rho_t = \min\!\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$ is the clipped importance weight for the actor update. At $t = T-1$, $Q^{ret}_T$ is out-of-bounds and treated as $0$ -- the one-step bootstrap is already inside $\delta_{T-1}$ via `next_q_values_all`, so the advantage reflects only the immediate TD correction at the window boundary.

For sequences exceeding the flat kernel limit, the chunked path performs both additions in PyTorch after the scan.

---

## 7. Retrace Applications

Retrace($\lambda$) was introduced by DeepMind to solve the variance explosion problem inherent in standard importance sampling, allowing algorithms to safely utilize off-policy data without discarding multi-step returns (Munos et al., 2016). Its ability to process off-policy trajectories from experience replay buffers made it a cornerstone of sample-efficient reinforcement learning. Specifically, Retrace was combined with trust region optimization to create the Actor-Critic with Experience Replay (ACER) algorithm, which demonstrated strong sample efficiency in both discrete and continuous control environments (Wang et al., 2017).

---

## References

* Munos, R., Stepleton, T., Harutyunyan, A., & Bellemare, M. G. (2016). *Safe and Efficient Off-Policy Reinforcement Learning.* Advances in Neural Information Processing Systems.
* Wang, Z., Bapst, V., Heess, N., Mnih, V., Munos, R., Kavukcuoglu, K., & de Freitas, N. (2017). *Sample Efficient Actor-Critic with Experience Replay.* arXiv:1611.01224.

## Further Reading

* [Off-Policy Correction](https://pseudo-rnd-thoughts.github.io/blog/off-policy-correction/) -- A visual explanation of off-policy correction methods, covering the importance sampling rationale behind Retrace and related algorithms.
