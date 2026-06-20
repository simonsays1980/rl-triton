# Retrace(λ) with Triton Associative Scans

### Introduction

Having successfully accelerated both Generalized Advantage Estimation (GAE) and V-Trace, we can now adapt our $O(\log N)$ parallel associative scan to evaluate action-values using **Retrace($\lambda$)**. 

Retrace($\lambda$) (Munos et al., 2016) is a foundational off-policy algorithm that shares the same recursive DNA as V-Trace, but is designed for Q-learning rather than Actor-Critic state values. Because the underlying first-order recurrence remains identical, the Triton kernel requires no changes - only the PyTorch inputs need adjustment to account for an essential **index shift** in the importance sampling weights.

---

### 1. The Retrace Target: Accumulating TD Errors Over Q-Values

Before mapping Retrace to our hardware, it is worth asking the same question we asked for V-Trace: *Why does Retrace accumulate TD errors rather than raw discounted rewards?*

The answer is again a **Telescoping Sum**, but this time applied to the action-value $Q(s_t, a_t)$ instead of the state value $V(s_t)$.

Recall the Retrace TD error for a single step:

$$\delta_t = r_t + \gamma(1-d_t^{\text{term}})\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$$

where $d_t^{\text{term}}$ is the true termination flag (not truncation). A truncated next state is a real continuation state whose value should be kept; only a true terminal state has no meaningful next value.

Adding this error to the baseline $Q(s_t, a_t)$ gives:

$$Q(s_t, a_t) + \delta_t = Q(s_t, a_t) + \left[ r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t) \right]$$

The $Q(s_t, a_t)$ terms cancel instantly, leaving the 1-step return:

$$= r_t + \gamma\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)]$$

Extending this to two steps:

$$Q(s_t, a_t) + \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 \mathbb{E}_{a \sim \pi}[Q(s_{t+2}, a)]$$

The intermediate $Q$ estimates telescope and collapse into a clean $n$-step return. This telescoping property is what makes the TD-error format algebraically natural for Retrace: we can attach the per-step importance weights $c_{t+1}$ to each individual error term rather than trying to untangle them from a block of summed rewards.

---

### 2. Deriving the Backward Recurrence

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

This is the mathematically exact recurrence within a single episode. The sum in the Retrace definition naturally terminates at the episode boundary — no done flags are part of the mathematical definition.

**Implementation: episode boundaries in a rollout buffer.**
The kernel multiplies the decay by $(1-d_t)$ to prevent accumulation from crossing episode edges:

$$\beta_t = \gamma c_{t+1}(1 - d_t), \quad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

The bootstrap inside $\delta_t$ uses $d_t^{\text{term}}$ alone, since a truncated $s_{t+1}$ is a real continuation state whose value should be kept:

$$\delta_t = r_t + \gamma(1 - d_t^{\text{term}})\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$$

* **Terminated** ($d_t^{\text{term}} = 1$): bootstrap zeroed in $\delta_t$ — correct. $(1-d_t)$ zeros $\beta_t$ — correct, $\Delta_{t+1} = 0$ by definition.
* **Truncated** ($d_t^{\text{trunc}} = 1$, $d_t^{\text{term}} = 0$): bootstrap kept in $\delta_t$ — correct. $(1-d_t)$ zeros $\beta_t$ — a scan convention. $\Delta_{t+1}$ of the next window is not zero; it is unavailable, and the one-step bootstrap already present in $\delta_{T-1}$ absorbs the missing tail.

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

Retrace has no dedicated `bootstrap_values` argument. The scan uses $\Delta_T = 0$ as the boundary, which introduces a truncation bias equal to the dropped off-policy tail $\sum_{k \geq T}(\ldots)$. This bias is geometric in window depth and suppressed by trace clipping ($c_i \leq \lambda \leq 1$), vanishing when the critic is exact at the boundary.

The last TD error $\delta_{T-1}$ does in fact contain a boundary bootstrap — $\mathbb{E}_\pi[Q(s_T,\cdot)]$ — which is already present in `next_q_values_all[:, T-1, :]` and requires no action at $s_T$. In this respect Retrace is no different from GAE or V-Trace: both algorithms similarly embed $V(s_T)$ (equivalently, $\mathbb{E}_\pi[Q(s_T,\cdot)] = V^\pi(s_T)$) inside their last TD error, and all three algorithms therefore "see" the boundary value without needing a separately passed scalar.

The structural difference is in the *carry*. GAE and V-Trace have a nonzero scan carry at the window boundary ($\Delta_T = V(s_T)$, passed as `bootstrap_values`), because their recurrence coefficient $\gamma\lambda$ (GAE) or $\gamma c_{T-1}$ (V-Trace) is fully defined and the boundary value is state-only. In Retrace, the recurrence coefficient that would propagate a nonzero carry into $\Delta_{T-1}$ is $\gamma c_T$, where:
$$c_T = \lambda\min\!\left(1,\frac{\pi(a_T|s_T)}{\mu(a_T|s_T)}\right)$$
This requires $a_T$ — the specific action the behavior policy $\mu$ drew at $s_T$, which is not in the rollout buffer because it was never sampled. The kernel therefore hardcodes $c_T = 0$ (line `c_next = tl.where(offs > 0, ..., 0.0)`), making the carry $\gamma c_T \cdot \Delta_T = 0$ for any value of $\Delta_T$. Unlike GAE and V-Trace, there is no meaningful seed to pass: the carry is killed by the coefficient, not the seed value.

A single rollout buffer per environment can contain multiple complete episodes. At any interior step $t < T-1$, `next_q_values_all[:, t, :]` already holds $Q(s_{t+1}, \cdot)$, so each TD error is fully computable from the existing inputs — no extra bootstrap scalar is needed for mid-sequence boundaries, whether terminated or truncated.

The window boundary at $t=T-1$ is handled the same way: `next_q_values_all[:, T-1, :]` holds $\mathbb{E}_\pi[Q(s_T, \cdot)]$, already present in the tensor the caller must supply for every step anyway. Whether that boundary is a termination or an ongoing episode ending at the window edge, the correct value is simply read from there:

$$\delta_{T-1} = r_{T-1} + \gamma (1 - d_{T-1}^{\text{term}}) \cdot \mathbb{E}_{a \sim \pi}[Q(s_T, a)] - Q(s_{T-1}, a_{T-1})$$

When $d_{T-1}^{\text{term}} = 0$, $\mathbb{E}_\pi[Q(s_T, \cdot)]$ is fully included in $\delta_{T-1}$ and propagates through the scan. When $d_{T-1}^{\text{term}} = 1$, it is masked out. The scan seed $\Delta_T = 0$ is correct in both cases because all future information is already encoded in $\delta$ before the scan runs.

GAE and V-Trace differ here: their `values` tensor holds only $V(s_t)$ for $t=0,\ldots,T-1$, so $V(s_T)$ at the window boundary is genuinely absent and must be passed as a dedicated `bootstrap_values` scalar per environment. See [GAE §5](gae.md#5-handling-episode-boundaries-and-the-window-bootstrap) and [V-Trace §5](vtrace.md#5-reconstructing-targets-and-advantages) for the full derivation.

#### Boundary cases

A single trajectory buffer per environment can contain multiple episodes separated by done flags. Two types of episode boundary can occur at any step $t \in \{0,\ldots,T-1\}$; a third case applies only at the final step $T-1$:

| Situation | When | $d_t$ | $d_t^{\text{term}}$ | Bootstrap in $\alpha_t$ | Trace decay $\beta_t$ |
|---|---|---|---|---|---|
| **Terminated** | any $t$ | 1 | 1 | Zeroed: $r_t - Q_t$ | $0$ — scan stops |
| **Truncated** | any $t$ | 1 | 0 | Kept: $r_t + \gamma\mathbb{E}_\pi[Q(s_{t+1},\cdot)] - Q_t$ | $0$ — scan stops |
| **Window end** | $t=T-1$ only | 0 | 0 | Kept: $r_{T-1} + \gamma\mathbb{E}_\pi[Q(s_T,\cdot)] - Q_{T-1}$ | $0$ — scan stops ($c_T$ undefined) |

In the **terminated** case, $s_{t+1}$ is a reset state with no meaningful value, so the bootstrap $\gamma\mathbb{E}_\pi[Q(s_{t+1},\cdot)]$ is zeroed by $(1 - d_t^{\text{term}})$. The scan is also stopped ($d_t=1$ zeros $\beta_t$) to prevent the correction for the new episode from bleeding into the previous one.

In the **truncated** case, $s_{t+1}$ is a real continuation state, so the bootstrap is kept in $\alpha_t$. The scan is still stopped ($d_t=1$ zeros $\beta_t$) for the same episode-isolation reason. The one-step bootstrap already captures the estimated value of the continuation; no separate carry is needed.

At the **window boundary** $t=T-1$, if the episode is still ongoing ($d_{T-1}=0$), the bootstrap $\gamma\mathbb{E}_\pi[Q(s_T,\cdot)]$ is kept in $\alpha_{T-1}$. The scan stops here not because of any done flag but because $c_T$ is undefined — no action was sampled at $s_T$, so $c_T$ is hardcoded to $0$, making $\beta_{T-1} = \gamma \cdot 0 \cdot (1 - d_{T-1}) = 0$.

The two derived quantities the implementation uses are:

$$d_t^{\text{term}} = d_t - d_t^{\text{trunc}} \qquad \text{(gates the bootstrap in } \alpha_t\text{)}$$

$$d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}} \qquad \text{(gates trace decay in } \beta_t\text{)}$$

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

Despite the index shift, the mathematical structure is still a perfect first-order recurrence conforming to our kernel's expected $f(x) = \alpha + \beta x$ transformation. 

The inputs map to hardware tuples $(\alpha, \beta)$ as follows:

* **TD accumulation ($\alpha_t$):** $\alpha_t = \delta_t = r_t + \gamma(1-d_t^{\text{term}})\mathbb{E}_{a \sim \pi}[Q(s_{t+1}, a)] - Q(s_t, a_t)$
* **Trace Decay coefficient ($\beta_t$):** $\beta_t = \gamma c_{t+1}(1-d_t)$

Because $c_{t+1}$ requires looking one step into the future, we simply shift the $c$ array in PyTorch using `torch.roll` before passing it to the Triton kernel.

#### Verification via the Reduction Tree

The reduction tree mechanics are identical to those described in the [GAE tutorial](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree). The following verifies only what is specific to Retrace: how the index shift manifests in the accumulated polynomial. As with GAE and V-Trace, the array is reversed so that threads look "left" to reach chronologically later data.

After the $O(\log N)$ scan, Thread 4 (at reversed index 4, chronological $t=1$) holds the accumulated result $\alpha_{1..4}$. Applying the $\oplus$ operator:

$$\alpha_{1..4} = \alpha_4 + \beta_4 \alpha_3 + \beta_4 \beta_3 \alpha_2 + \beta_4 \beta_3 \beta_2 \alpha_1$$

Substituting our chronological Retrace definitions (reversed index $k$ maps to time $t = k$):

$$\alpha_{1..4} = \delta_1 + (\gamma c_2)\delta_2 + (\gamma c_2)(\gamma c_3)\delta_3 + (\gamma c_2)(\gamma c_3)(\gamma c_4)\delta_4$$

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