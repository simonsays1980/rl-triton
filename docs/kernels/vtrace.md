# V-Trace with Triton Associative Scans

## Introduction

The [GAE tutorial](gae.md) established how a Parallel Associative Scan restructures a first-order linear recurrence into an $O(\log N)$ parallel tree reduction. This tutorial applies the same technique to **V-Trace** (the off-policy target correction algorithm introduced in IMPALA; Espeholt et al., 2018).

Because V-Trace relies on the exact same recurrence structure as GAE, no new hardware algorithm is needed. Instead, we algebraically redefine the mathematical inputs, reuse the same associative operator, and execute the identical Triton reduction tree.

---

## 1. The V-Trace Architecture and the Off-Policy Bottleneck

Unlike GAE, which assumes data is strictly on-policy, V-Trace allows agents to learn from off-policy trajectories. This was originally designed for the IMPALA distributed architecture (Espeholt et al., 2018), where dozens of "Actor" CPUs collect data for a single "Learner" GPU. Because the Learner continuously updates the network, the policy that generated the data (the **behavior policy, $\mu$**) is often older than the policy being trained (the **target policy, $\pi$**).

If we apply standard on-policy math to off-policy data, the value estimates diverge. V-Trace corrects this "policy lag" by introducing two clipped importance sampling weights at each timestep $t$:

* **$\rho_t$ (rho):** Used to scale the immediate Temporal Difference (TD) error. It answers the question, *"How likely was the current policy to take this action compared to the old policy?"*

    $$\rho_t = \min\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$$

* **$c_t$:** Used to cut or scale the trace decay parameter for future steps. It is the importance sampling ratio clipped at $\bar{c}$: large when $\pi$ and $\mu$ agree, small when they diverge, and $0$ only when $\pi(a_t|s_t) = 0$.

    $$c_t = \min\left(\bar{c}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$$

The standard V-Trace formula calculates the corrected $n$-step value target ($v_s$) for a state $s$:

$$v_s = V(s_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t^V$$

Where the TD error for the value function is scaled by $\rho$:

$$\delta_t^V = \rho_t(r_t + \gamma(1 - d_t^{\text{term}}) V(s_{t+1}) - V(s_t))$$

---

## 2. Why Accumulate TD Errors? The Telescoping Sum

Looking at the V-Trace summation formula above, a common question arises: *Why does V-Trace accumulate future TD errors instead of simply accumulating discounted rewards?* Standard $n$-step returns sum up rewards and bootstrap at the end.

The answer lies in a mathematical property called a **Telescoping Sum**. Adding a sum of discounted TD errors to a baseline value estimate perfectly reconstructs the $n$-step reward return because the intermediate value estimates cancel each other out.

Observe what happens if we add just the 1-step uncorrected TD error to our baseline prediction:

$$V(s_t) + \delta_t = V(s_t) + \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right]$$

The positive $V(s_t)$ and negative $V(s_t)$ instantly cancel, leaving:

$$= r_t + \gamma V(s_{t+1})$$

If we expand this to two steps:

$$V(s_t) + \delta_t + \gamma \delta_{t+1} = V(s_t) + \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right] + \gamma \left[ r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}) \right]$$

The intermediate terms telescope and collapse, leaving exactly the 2-step reward return:

$$= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$$

V-Trace explicitly uses this TD-error format for two reasons:
1. **Importance Sampling:** Correcting a raw $n$-step reward sum for off-policy data requires multiplying by the full trajectory ratio $\prod_{i=t}^{t+n-1} \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)}$, which grows or shrinks exponentially with $n$ and causes severe variance. The telescoping structure of TD errors allows a separate clipped weight $c_t$ to be applied at each step independently, keeping variance bounded regardless of trace length.
2. **Recursion:** It creates a perfect recursive loop that maps flawlessly to hardware acceleration.

---

## 3. The Mathematical Bridge: Reshaping to the Associative Scan

The foundational Triton kernel solves any sequence conforming to the linear transformation $f(x)=a+bx$. To align the V-Trace sum-of-products formula with this structure, isolate the summation by defining a new variable **$\Delta_t$** (the value delta), representing the sum of all future trace-decayed TD errors:

$$\Delta_t = v_t - V(s_t)$$

If we unroll the summation for $\Delta_t$, we get:

$$\Delta_t = \delta_t^V + \gamma c_t \delta_{t+1}^V + \gamma^2 c_t c_{t+1} \delta_{t+2}^V + \dots$$

Factoring out $\gamma c_t$ from the second term onward reveals the first-order backward recurrence:

$$\Delta_t = \delta_t^V + \gamma c_t \cdot \Delta_{t+1}$$

This is the mathematically exact recurrence within a single episode. No done flags appear -- the sum in the original V-Trace definition naturally terminates at the episode boundary.

In a rollout buffer a single sequence may contain multiple complete episodes or end mid-episode at a truncation boundary. To prevent accumulation from crossing episode boundaries, the kernel introduces masked versions of $\alpha_t$ and $\beta_t$ that depend on two mutually exclusive flags, $d_t^{\text{term}}$ and $d_t^{\text{trunc}}$:

$$\alpha_t = \delta_t^V = \rho_t\!\left(r_t + \gamma\,(1 - d_t^{\text{term}})\,V(s_{t+1}) - V(s_t)\right)$$

$$\beta_t = \gamma c_t(1 - d_t), \quad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

Because `terminated` and `truncated` are mutually exclusive, $(1 - d_t^{\text{term}})$ zeros $V(s_{t+1})$ in $\alpha_t$ at termination, and $(1 - d_t)$ zeros $\beta_t$ at any boundary. Section 5 details what this means for the caller.

Throughout this document $T$ denotes the **sequence length** (the number of steps in the rollout buffer), not necessarily an episode termination.

---

## 4. Thread-Level Execution: Computing V-Trace in Hardware

Because the mathematical structure is mapped to the same recurrence, the GPU threads execute the same associative combination using our operator $\oplus$:

$$(\alpha_B,\beta_B)\oplus(\alpha_A,\beta_A)=(\alpha_B+\beta_B \alpha_A,\beta_A \beta_B)$$

The following trace mirrors the [GAE reduction tree](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree) exactly -- only the definition of $\alpha_t$ and $\beta_t$ differs. The array is reversed in memory so threads look to their "left" (lower index) to pull chronologically later data.

### Setup (Step 0)

Each thread loads its initial tuple $(\alpha_i, \beta_i)$ into local registers:

* **T1** (index 1, $t=3$): holds $T_1=(\alpha_1, \beta_1)$
* **T2** (index 2, $t=2$): holds $T_2=(\alpha_2, \beta_2)$
* **T3** (index 3, $t=1$): holds $T_3=(\alpha_3, \beta_3)$
* **T4** (index 4, $t=0$): holds $T_4=(\alpha_4, \beta_4)$

### Hardware Loop 1: Parallel Pairs (Distance = 1)

Every thread simultaneously looks 1 step to their "left" and combines using $\oplus$:

* **T1**: no left neighbor, keeps $T_1$.
* **T2**: grabs $T_1$, computes $T_{1..2}=T_2\oplus T_1$.
* **T3**: grabs $T_2$, computes $T_{2..3}=T_3\oplus T_2$.
* **T4**: grabs $T_3$, computes $T_{3..4}=T_4\oplus T_3$.

### Hardware Loop 2: Tree Merge (Distance = 2)

Every thread simultaneously looks 2 steps to their "left":

* **T1**: no neighbor at distance 2, keeps $T_1$.
* **T2**: no neighbor at distance 2, keeps $T_{1..2}$.
* **T3**: grabs $T_1$ from T1, computes $T_{1..3}=T_{2..3}\oplus T_1$.
* **T4**: grabs $T_{1..2}$ from T2, computes $T_{1..4}=T_{3..4}\oplus T_{1..2}$.

### Verification

The accumulated $\alpha$ value in T4's register after the scan:

$$\alpha_{1..4}=(\alpha_4+\beta_4 \alpha_3)+(\beta_3 \beta_4)(\alpha_2+\beta_2 \alpha_1)$$

$$\alpha_{1..4}=\alpha_4+\beta_4 \alpha_3+\beta_4 \beta_3 \alpha_2+\beta_4 \beta_3 \beta_2 \alpha_1$$

Substituting back the V-Trace definitions ($\alpha_i = \delta_i^V$, $\beta_i = \gamma c_i$) and remembering the reversed index mapping ($\alpha_4$ is chronological $t=1$, etc.):

$$\alpha_{1..4} = \delta_1^V + (\gamma c_1)\delta_2^V + (\gamma c_1)(\gamma c_2)\delta_3^V + (\gamma c_1)(\gamma c_2)(\gamma c_3)\delta_4^V$$

This matches the V-Trace sum exactly. Thread 4 built its chunk $T_{3..4}$ in parallel with Thread 2 building $T_{1..2}$, with no sequential wait between them.

---

## 5. Handling Episode Boundaries and the Window Bootstrap

The kernel consumes `values`, `terminateds`, `truncateds`, and `bootstrap_values`, all of shape `[num_envs, seq_len]`. It never sees observations. The equations for $\alpha_t$ and $\beta_t$ were established in Section 3; this section describes the caller's responsibility for preparing the input arrays correctly.

### Terminated steps

At a terminated step $t$, `terminateds[t] = 1` zeros $\gamma V(s_{t+1})$ inside $\alpha_t$ and sets $\beta_t = 0$, stopping trace propagation into the previous episode. `values[t+1]` belongs to the next episode but is harmless: the carry is already severed. No entry in `bootstrap_values` is needed at terminated steps; the kernel ignores it there.

### Truncated steps

At a truncated step $t$, the episode continues, so $\alpha_t$ still needs the true continuation value $V(s_{t+1})$. The stored `values[t+1]` belongs to a different episode and must not be used. The caller supplies the correct value as `bootstrap_values[env, t]`, which the Section 3 formula selects automatically. The carry is still severed ($\beta_t = 0$) so no accumulation propagates across the boundary.

### Window boundary at $t = T-1$

At the last step of the window, $V(s_T)$ lies one step past the end of the `values` tensor. The caller supplies it as `bootstrap_values[env, T-1]` when the episode continues past the window, or leaves it zero if the episode terminated at $T-1$.

This value enters $\alpha_{T-1}$ as the ordinary next-state value that completes the one-step TD residual. The scan's boundary carry is always $\Delta_T = 0$:

$$\Delta_t = (\text{local scan result})_t + (\text{decay product})_t \cdot 0 = (\text{local scan result})_t$$

A value-delta carry would represent trace mass from steps *past* the buffer, and there is none to represent: `bootstrap_values[env, T-1]` already prices $V(s_T)$ into $\alpha_{T-1}$ at weight 1, exactly as the infinite-horizon recurrence in Section 3 requires.

### The unifying rule

`bootstrap_values` holds the true continuation value $V(s_{t+1})$ at exactly those positions where `values[t+1]` is invalid -- truncated steps and the final column -- and zero everywhere else. Its shape is `[num_envs, seq_len]`.

For the common case where no interior truncations occur, the `last_value` argument (shape `[num_envs]`) provides a convenience: the caller passes the window-edge continuation value directly and the kernel populates `bootstrap_values[:, -1]` automatically. `last_value` and `bootstrap_values` are mutually exclusive.

### Boundary summary

| Situation | $d_t$ | $d_t^{\text{term}}$ | Bootstrap $V(s_{t+1})$ in $\alpha_t$ | Decay $\beta_t$ | Scan carry $\Delta_T$ |
|---|---|---|---|---|---|
| **Terminated** | 1 | 1 | Zeroed by $(1-d_t^{\text{term}})$; `bootstrap_values` ignored | $0$ -- scan stops | $0$ |
| **Truncated** | 1 | 0 | Kept; caller supplies `bootstrap_values[env, t]` | $0$ -- scan stops | $0$ |
| **Window end** | 0 | 0 | Kept; caller supplies `bootstrap_values[env, T-1]` | $0$ -- scan stops | $0$ (always -- never seeded from the bootstrap) |

---

## 6. Reconstructing Targets and Advantages

Once the $O(\log N)$ scan finishes, the fused kernel computes targets and advantages in the same Triton program before returning to the host. Targets are formed as $v_t = \Delta_t + V(s_t)$ directly in registers and written to HBM, then a second pass over the stored targets computes the advantages:

$$v_t = \Delta_t + V(s_t)$$

$$A_t = \rho_t(r_t + \gamma v_{t+1} - V(s_t))$$

For sequences exceeding the flat kernel limit, the chunked path performs the equivalent additions in PyTorch after the scan.

---

## 7. Hardware Execution

V-Trace runs on the identical kernel architecture as GAE and inherits the same memory lifecycle: one coalesced HBM load of all $\alpha_t$ and $\beta_t$ arrays into SRAM, the $O(\log N)$ reduction entirely within registers and SRAM, and a single synchronized HBM store of the $\Delta_t$ results. See the [GAE hardware section](gae.md#6-the-hardware-reality-what-triton-actually-does) for the full breakdown.

## 8. V-Trace Applications

V-Trace was originally introduced as the core mathematical component of the IMPALA architecture to correct the "policy lag" that occurs when massively distributed CPU actors generate trajectories asynchronously for a centralized GPU learner (Espeholt et al., 2018). Because of this stability, algorithms like Asynchronous Proximal Policy Optimization (APPO; Berner et al., 2019) rely directly on V-Trace targets to safely optimize policies using stale trajectories collected by out-of-sync workers.

---

## References

* Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., ... & Kavukcuoglu, K. (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* ICML 2018. arXiv:1802.01561.
* Berner, C., Brockman, G., Chan, B., Cheung, V., Dębiak, P., Dennison, C., ... & Zoph, B. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* arXiv:1912.06680.

## Further Reading

* [Off-Policy Correction](https://pseudo-rnd-thoughts.github.io/blog/off-policy-correction/) -- A visual explanation of off-policy correction methods, covering the importance sampling rationale behind V-Trace and related algorithms.
