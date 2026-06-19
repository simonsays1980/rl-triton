# Retrace(λ) with Triton Associative Scans

### Introduction

Having successfully accelerated both Generalized Advantage Estimation (GAE) and V-Trace, we can now adapt our $O(\log N)$ parallel associative scan to evaluate action-values using **Retrace($\lambda$)**. 

Retrace($\lambda$) (Munos et al., 2016) is a foundational off-policy algorithm that shares the same recursive DNA as V-Trace, but is designed for Q-learning rather than Actor-Critic state values. Because the underlying first-order recurrence remains identical, the Triton kernel requires no changes - only the PyTorch inputs need adjustment to account for an essential **index shift** in the importance sampling weights.

---

### 1. The Retrace Target: Accumulating TD Errors Over Q-Values

Before mapping Retrace to our hardware, it is worth asking the same question we asked for V-Trace: *Why does Retrace accumulate TD errors rather than raw discounted rewards?*

The answer is again a **Telescoping Sum**, but this time applied to the action-value $Q(s_t, a_t)$ instead of the state value $V(s_t)$.

Recall the Retrace TD error for a single step:

$$\delta_t = r_t + \gamma(1-d_t)\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$$

Adding this error to the baseline $Q(s_t, a_t)$ gives:

$$Q(s_t, a_t) + \delta_t = Q(s_t, a_t) + \left[ r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t) \right]$$

The $Q(s_t, a_t)$ terms cancel instantly, leaving the 1-step return:

$$= r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)]$$

Extending this to two steps:

$$Q(s_t, a_t) + \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 \mathbb{E}_{a \sim \pi}[Q(s_{t+2}, a)]$$

The intermediate $Q$ estimates telescope and collapse into a clean $n$-step return. This telescoping property is what makes the TD-error format algebraically natural for Retrace: we can attach the per-step importance weights $c_{t+1}$ to each individual error term rather than trying to untangle them from a block of summed rewards.

---

### 2. Deriving the Backward Recurrence

The full Retrace target for state-action pair $(s_t, a_t)$ is:

$$Q^{ret}_t = Q(s_t, a_t) + \sum_{k=t}^{T-1} \gamma^{k-t} \left( \prod_{i=t+1}^{k} c_i \right) \delta_k$$

where the empty product for $k=t$ is defined as $1$. The **target delta** is defined as:

$$\Delta_t = Q^{ret}_t - Q(s_t, a_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \left( \prod_{i=t+1}^{k} c_i \right) \delta_k$$

Unrolling the first two terms makes the structure explicit:

$$\Delta_t = \delta_t + \gamma c_{t+1} \delta_{t+1} + \gamma^2 c_{t+1} c_{t+2} \delta_{t+2} + \dots$$

Factoring $\gamma c_{t+1}$ from every term after the first:

$$\Delta_t = \delta_t + \gamma c_{t+1} \left( \delta_{t+1} + \gamma c_{t+2} \delta_{t+2} + \dots \right)$$

The expression in parentheses is exactly $\Delta_{t+1}$, which yields the **first-order backward recurrence**:

$$\Delta_t = \delta_t + \gamma c_{t+1}(1-d_t)\Delta_{t+1}$$

*(The binary done flag $d_t$ is appended to prevent the trace from bleeding across episode boundaries.)*

Notice the critical difference from V-Trace: the decay coefficient multiplying $\Delta_{t+1}$ is $\gamma c_{t+1}$ (next-step weight), not $\gamma c_t$ (current-step weight). This is examined in detail in the next section.

---

### 3. The Core Difference: The Index Shift ($c_{t+1}$)

If you look closely at the backward recurrence derived above, the trace weight applied to $\Delta_{t+1}$ is **$c_{t+1}$**, whereas in V-Trace the corresponding weight was **$c_t$**.

This index shift is the defining mathematical difference between the two algorithms, and it exists for a very specific reason:

* **V-Trace** evaluates a state $s_t$. It must correct for the very first action $a_t$ taken from that state, so it applies importance sampling immediately ($c_t$).
* **Retrace** evaluates a state-action pair $(s_t, a_t)$. Because the action $a_t$ is already fixed as the subject of our evaluation, we do not reject or correct it. We only begin correcting for off-policy divergence on the *next* action taken in the trajectory ($a_{t+1}$).

The importance weight $c_t$ itself uses a $\lambda$-scaled clip at $1$ rather than V-Trace's $\bar{c}$ clip:

$$c_t = \lambda \min\left(1, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$$

Setting $\lambda = 1$ recovers the full importance sampling weight; setting $\lambda < 1$ increases the bias–variance trade-off toward lower variance.

---

### 4. Bootstrap and Boundary Flags

Before mapping to hardware it is worth clarifying what "bootstrap" means in Retrace and how the two boundary-flag conventions map onto the implementation.

#### How the bootstrap enters the TD error, and the scan carry

Retrace has no dedicated `bootstrap_values` argument: the scan carry into the kernel is discarded entirely (`Δ_T = 0`), which is safe because the trace decay coefficient at the last step, $b_{T-1} = \gamma c_T (1-d_{T-1})$, is always zero — independent of the done flag. $c_T = c_{t+1}$ at $t=T-1$ is out-of-bounds (one step past the window) and is hardcoded to $0$ in the implementation (`c_next[:, -1] = 0.0`). This structural zero holds whether the boundary is a termination, a truncation, or neither, so a nonzero carry would have multiplied out regardless of $d_{T-1}$.

GAE and V-Trace differ here: their scalar `bootstrap_values` $= V(s_T)$ enters in *two* places, not one — once inside the TD error $a_{T-1}$, and again as the scan carry itself ($\Delta_T = V(s_T)$, added on top of the local scan result at every position). See [GAE §5](gae.md#5-handling-truncated-episodes) and [V-Trace §5](vtrace.md#5-reconstructing-targets-and-advantages) for the full derivation.

In GAE and V-Trace the TD error at $t=T-1$ requires the scalar $V(s_T)$:

$$a_{T-1}^{\text{V-Trace}} = \rho_{T-1}\!\left(r_{T-1} + \gamma V(s_T) - V(s_{T-1})\right)$$

$V(s_T)$ is out of window and not present in any existing input tensor, so it must be passed as a dedicated `bootstrap_values` argument $\in \mathbb{R}^{N_{\text{env}}}$.

In Retrace the TD error at $t=T-1$ requires $\mathbb{E}_{a \sim \pi}[Q(s_T, a)]$ — an expectation over the full action distribution:

$$a_{T-1}^{\text{Retrace}} = r_{T-1} + \gamma \cdot \mathbb{E}_{a \sim \pi}[Q(s_T, a)](1-d_{T-1}^{\text{term}}) - Q(s_{T-1}, a_{T-1})$$

This expectation requires the full action-probability vector $\pi(\cdot|s_T)$ and the Q-values for all actions at $s_T$. The `next_q_values_all` tensor $\in \mathbb{R}^{N_{\text{env}} \times T \times |\mathcal{A}|}$ is therefore unavoidable for every step anyway (each $a_t$ needs it). The boundary value $\mathbb{E}_\pi[Q(s_T, a)]$ is simply read from `next_q_values_all[:, T-1, :]` — already present, no separate argument needed.

The reason Retrace takes no `bootstrap_values` argument is therefore not algorithmic but structural: a state value function $V(s)$ is a scalar that requires a separate boundary argument, whereas a Q-function expectation $\mathbb{E}_\pi[Q(s,a)]$ requires a full action-distribution tensor that is present for all steps, including the boundary.

#### Terminated vs. truncated

Two distinct reasons a trajectory can end mid-sequence require different treatment:

| Condition | Meaning | Correct action |
|---|---|---|
| **terminated** $d_t^{\text{term}} = 1$ | Environment signalled a true episode end. $s_{t+1}$ is a reset state with no meaningful value. | Zero the bootstrap: multiply $\gamma \cdot \mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)]$ by $(1 - d_t^{\text{term}})$. |
| **truncated** $d_t^{\text{trunc}} = 1$ | Window or time-limit cutoff; the episode continues. $s_{t+1}$ is a real state. | Keep the bootstrap: leave $\gamma \cdot \mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)]$ untouched. |

In both cases the **trace decay** $b_t = \gamma c_{t+1}(1 - d_t)$ is zeroed at the boundary (where $d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$), stopping the backward scan from propagating $\Delta_{t+1}$ into $\Delta_t$ across episode or window edges.

This means a **truncated** rollout window introduces a small bias that is structurally different from GAE/V-Trace's. There, the carry $A_T = V(s_T)$ supplies a value-based estimate of everything beyond the window. In Retrace, $c_{t+1}$ at $t=T-1$ would need $c_T$, which does not exist and is fixed to $0$ — so the windowed estimator drops the entire $\gamma c_T \Delta_T$ tail rather than approximating it. The one-step bootstrap $\mathbb{E}_\pi[Q(s_T,\cdot)]$ inside $a_{T-1}$ still corrects the *last* TD error, but nothing beyond it is estimated. This is a known limitation of windowed Retrace, not a kernel defect, and shrinks as the window grows relative to $1/(1-\gamma c)$.

The two quantities the implementation therefore uses are:

$$d_t^{\text{term}} = d_t - d_t^{\text{trunc}} \qquad \text{(gates the bootstrap in } a_t\text{)}$$
$$d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}} \qquad \text{(gates trace decay in } b_t\text{)}$$

#### Mapping from common API conventions

**Gymnasium** returns `terminated` and `truncated` separately - pass them directly:
```python
compute_retrace(..., dones=terminated | truncated, truncateds=truncated)
```

**Single done-flag APIs** (older Gym, most existing RL codebases) merge both into one flag. Pass it as `dones` and omit `truncateds`. The bootstrap is then zeroed on every boundary - correct for purely episodic data, conservative for windowed trajectories:
```python
compute_retrace(..., dones=done)   # truncateds defaults to None
```

---

### 5. Mapping to the Associative Scan

Despite the index shift, the mathematical structure is still a perfect first-order recurrence conforming to our kernel's expected $f(x) = a + bx$ transformation. 

The inputs map to hardware tuples $(a, b)$ as follows:
* **TD accumulation ($a_t$):** $a_t = \delta_t = r_t + \gamma(1-d_t^{\text{term}})\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$
* **Trace Decay coefficient ($b_t$):** $b_t = \gamma c_{t+1}(1-d_t)$

Because $c_{t+1}$ requires looking one step into the future, we simply shift the $c$ array in PyTorch using `torch.roll` before passing it to the Triton kernel.

#### Verification via the Reduction Tree

The reduction tree mechanics are identical to those described in the [GAE tutorial](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree). The following verifies only what is specific to Retrace: how the index shift manifests in the accumulated polynomial. As with GAE and V-Trace, the array is reversed so that threads look "left" to reach chronologically later data.

After the $O(\log N)$ scan, Thread 4 (at reversed index 4, chronological $t=1$) holds the accumulated result $a_{1..4}$. Applying the $\oplus$ operator:

$$a_{1..4} = a_4 + b_4 a_3 + b_4 b_3 a_2 + b_4 b_3 b_2 a_1$$

Substituting our chronological Retrace definitions (reversed index $k$ maps to time $t = k$):

$$a_{1..4} = \delta_1 + (\gamma c_2)\delta_2 + (\gamma c_2)(\gamma c_3)\delta_3 + (\gamma c_2)(\gamma c_3)(\gamma c_4)\delta_4$$

Notice that $c_1$ does not appear anywhere in this polynomial. This is the index shift in action: $c_1$ is the importance weight for the *last* chronological action in the trajectory. Because Retrace only begins correcting from the *next* step onward, the decay *into* the last TD error $\delta_1$ would require $c_0$ - a weight from one step before the trajectory starts, which does not exist. The implementation forces this out-of-bounds weight to zero (`c_next[:, -1] = 0.0`), so the last step always contributes $\delta_1$ unscaled. $c_1$ itself is simply never needed: no TD error to its left exists that it could weight.

This is exactly $\Delta_1$ from the unrolled recurrence above: each TD error $\delta_k$ is multiplied by the product of $c$ weights starting at index $k$ (one step ahead of the originating state), precisely matching the $\prod_{i=t+1}^{k} c_i$ product in the Retrace target.

---

### 6. Reconstructing Targets and Advantages

Once the $O(\log N)$ kernel finishes, every thread holds its correct $\Delta_t$. The final Retrace targets are reconstructed in PyTorch using parallel vector additions.

**1. Target Q-Value (for Critic Loss):**

$$Q^{ret}_t = \Delta_t + Q(s_t, a_t)$$

**2. Target Advantage (for Actor Policy Gradient):**

The policy gradient uses the same one-step TD structure as V-Trace, but bootstraps with the Retrace Q-target rather than the raw critic estimate:

$$A_t = \rho_t \left( r_t + \gamma Q^{ret}_{t+1} - Q(s_t, a_t) \right)$$

where $\rho_t = \min\!\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$ is the V-Trace-style clipped importance weight for the actor update.

### 7. Retrace Applications

Retrace($\lambda$) was introduced by DeepMind to solve the variance explosion problem inherent in standard importance sampling, allowing algorithms to safely utilize off-policy data without discarding multi-step returns (Munos et al., 2016). Its ability to process off-policy trajectories from experience replay buffers made it a cornerstone of sample-efficient reinforcement learning. Specifically, Retrace was combined with trust region optimization to create the Actor-Critic with Experience Replay (ACER) algorithm, which demonstrated strong sample efficiency in both discrete and continuous control environments (Wang et al., 2017).

---

### References

* Munos, R., Stepleton, T., Harutyunyan, A., & Bellemare, M. G. (2016). *Safe and Efficient Off-Policy Reinforcement Learning.* Advances in Neural Information Processing Systems.
* Wang, Z., Bapst, V., Heess, N., Mnih, V., Munos, R., Kavukcuoglu, K., & de Freitas, N. (2017). *Sample Efficient Actor-Critic with Experience Replay.* arXiv:1611.01224.