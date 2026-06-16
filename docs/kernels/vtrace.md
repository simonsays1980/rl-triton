# V-Trace with Triton Associative Scans

### Introduction

The [GAE tutorial](gae.md) established how a Parallel Associative Scan restructures a first-order linear recurrence into an $O(\log N)$ parallel tree reduction. This tutorial applies the same technique to **V-Trace** (the off-policy target correction algorithm introduced in IMPALA; Espeholt et al., 2018).

Because V-Trace relies on the exact same recurrence structure as GAE, no new hardware algorithm is needed. Instead, we algebraically redefine the mathematical inputs, reuse the same associative operator, and execute the identical Triton reduction tree.

---

### 1. The V-Trace Architecture and the Off-Policy Bottleneck

Unlike GAE, which assumes data is strictly on-policy, V-Trace allows agents to learn from off-policy trajectories. This was originally designed for the IMPALA distributed architecture (Espeholt et al., 2018), where dozens of "Actor" CPUs collect data for a single "Learner" GPU. Because the Learner continuously updates the network, the policy that generated the data (the **behavior policy, $\mu$**) is often older than the policy being trained (the **target policy, $\pi$**).

If we apply standard on-policy math to off-policy data, the value estimates diverge. V-Trace corrects this "policy lag" by introducing two clipped importance sampling weights at each timestep $t$:

* **$\rho_t$ (rho):** Used to scale the immediate Temporal Difference (TD) error. It answers the question, *"How likely was the current policy to take this action compared to the old policy?"*

    $$\rho_t = \min\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$$

* **$c_t$:** Used to cut or scale the trace decay parameter for future steps. If the old policy took an action that the new policy completely disagrees with, $c_t$ drops to $0$, stopping any further off-policy future rewards from corrupting the current state's value.

    $$c_t = \min\left(\bar{c}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$$

The standard V-Trace formula calculates the corrected $n$-step value target ($v_s$) for a state $s$:

$$v_s = V(s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t^V$$

Where the TD error for the value function is scaled by $\rho$:

$$\delta_t^V = \rho_t(r_t + \gamma V(s_{t+1}) - V(s_t))$$

---

### 2. Why Accumulate TD Errors? The Telescoping Sum

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
1. **Importance Sampling:** It is algebraically much cleaner to multiply a specific step's error ($\delta_t$) by a trace weight ($c_t$) than to try and isolate off-policy penalties inside a raw block of summed rewards.
2. **Recursion:** It creates a perfect recursive loop that maps flawlessly to hardware acceleration.

---

### 3. The Mathematical Bridge: Reshaping to the Associative Scan

The foundational Triton kernel solves any sequence conforming to the linear transformation $f(x)=a+bx$. To align the V-Trace sum-of-products formula with this structure, isolate the summation by defining a new variable **$\Delta_t$** (the value delta), representing the sum of all future trace-decayed TD errors:

$$\Delta_t = v_t - V(s_t)$$

If we unroll the summation for $\Delta_t$, we get:

$$\Delta_t = \delta_t^V + \gamma c_t \delta_{t+1}^V + \gamma^2 c_t c_{t+1} \delta_{t+2}^V + \dots$$

Factoring out $\gamma c_t$ from the second term onward reveals the exact same first-order backward recurrence used for GAE:

$$\Delta_t = \delta_t^V + \gamma c_t (1-d_t) \Delta_{t+1}$$

*(The binary "done" flag $d_t$ prevents the trace from bleeding across episode boundaries.)*

This perfectly matches the kernel's expected format. The inputs map to hardware tuples $(a, b)$ as follows:

* **Value Delta accumulation ($a_t$):** $a_t = \delta_t^V = \rho_t(r_t + \gamma V(s_{t+1}) - V(s_t))$
* **Trace Decay product ($b_t$):** $b_t = \gamma c_t(1-d_t)$ 

---

### 4. Thread-Level Execution: Computing V-Trace in Hardware

Because the mathematical structure is mapped to the same recurrence, the GPU threads execute the same associative combination using our operator $\oplus$:

$$(a_B,b_B)\oplus(a_A,b_A)=(a_B+b_B a_A,b_A b_B)$$

The following trace mirrors the [GAE reduction tree](gae.md#3-the-mechanism-detailed-trace-of-a-4-step-reduction-tree) exactly - only the definition of $a_t$ and $b_t$ differs. The array is reversed in memory so threads look to their "left" (lower index) to pull chronologically later data.

#### Setup (Step 0)

Each thread loads its initial tuple $(a_i, b_i)$ into local registers:

* **T1** (index 1, $t=3$): holds $T_1=(a_1, b_1)$
* **T2** (index 2, $t=2$): holds $T_2=(a_2, b_2)$
* **T3** (index 3, $t=1$): holds $T_3=(a_3, b_3)$
* **T4** (index 4, $t=0$): holds $T_4=(a_4, b_4)$

#### Hardware Loop 1: Parallel Pairs (Distance = 1)

Every thread simultaneously looks 1 step to their "left" and combines using $\oplus$:

* **T1**: no left neighbor, keeps $T_1$.
* **T2**: grabs $T_1$, computes $T_{1..2}=T_2\oplus T_1$.
* **T3**: grabs $T_2$, computes $T_{2..3}=T_3\oplus T_2$.
* **T4**: grabs $T_3$, computes $T_{3..4}=T_4\oplus T_3$.

#### Hardware Loop 2: Tree Merge (Distance = 2)

Every thread simultaneously looks 2 steps to their "left":

* **T1**: no neighbor at distance 2, keeps $T_1$.
* **T2**: no neighbor at distance 2, keeps $T_{1..2}$.
* **T3**: grabs $T_1$ from T1, computes $T_{1..3}=T_{2..3}\oplus T_1$.
* **T4**: grabs $T_{1..2}$ from T2, computes $T_{1..4}=T_{3..4}\oplus T_{1..2}$.

#### Verification

The accumulated $a$ value in T4's register after the scan:

$$a_{1..4}=(a_4+b_4 a_3)+(b_3 b_4)(a_2+b_2 a_1)$$

$$a_{1..4}=a_4+b_4 a_3+b_4 b_3 a_2+b_4 b_3 b_2 a_1$$

Substituting back the V-Trace definitions ($a_i = \delta_i^V$, $b_i = \gamma c_i$) and remembering the reversed index mapping ($a_4$ is chronological $t=1$, etc.):

$$a_{1..4} = \delta_1^V + (\gamma c_1)\delta_2^V + (\gamma c_1)(\gamma c_2)\delta_3^V + (\gamma c_1)(\gamma c_2)(\gamma c_3)\delta_4^V$$

This matches the V-Trace sum exactly. Thread 4 built its chunk $T_{3..4}$ in parallel with Thread 2 building $T_{1..2}$, with no sequential wait between them.

---

### 5. Reconstructing Targets and Advantages

Once the $O(\log N)$ kernel finishes, every thread holds its correct $\Delta_t$. The final V-Trace targets and advantages are reconstructed in PyTorch using parallel vector additions.

**1. Target Value (for Critic Loss):**

$$v_t=\Delta_t+V(s_t)$$

**2. Target Advantage (for Actor Policy Gradient):**

Once we have the target values $v_t$, we shift the tensor to obtain $v_{t+1}$ and calculate the final advantage:

$$A_t=\rho_t(r_t+\gamma v_{t+1}-V(s_t))$$

For truncated episodes, `bootstrap_values` $= V(s_T)$ is passed as the out-of-window next-state value and is substituted into the TD error at the last step $t=T-1$:

$$a_{T-1} = \rho_{T-1}\!\left(r_{T-1} + \gamma V(s_T) - V(s_{T-1})\right)$$

The scan carry is always $\Delta_T = 0$. This is safe because the trace decay at the boundary is $b_{T-1} = \gamma c_{T-1}(1-d_{T-1}) = 0$ — the done flag zeros it — so $\Delta_T$ multiplies out regardless of its value. The bootstrap enters only through $a_{T-1}$, not through the scan carry.

---

### 6. Hardware Execution

V-Trace runs on the identical kernel architecture as GAE and inherits the same memory lifecycle: one coalesced HBM load of all $a_t$ and $b_t$ arrays into SRAM, the $O(\log N)$ reduction entirely within registers and SRAM, and a single synchronized HBM store of the $\Delta_t$ results. See the [GAE hardware section](gae.md#6-the-hardware-reality-what-triton-actually-does) for the full breakdown.

### 7. V-Trace Applications

V-Trace was originally introduced as the core mathematical component of the IMPALA architecture to correct the "policy lag" that occurs when massively distributed CPU actors generate trajectories asynchronously for a centralized GPU learner (Espeholt et al., 2018). Because of this stability, algorithms like Asynchronous Proximal Policy Optimization (APPO; Berner et al., 2019) rely directly on V-Trace targets to safely optimize policies using stale trajectories collected by out-of-sync workers.

---

### 8. Autoreset Mode and Loss Masking

With Gymnasium's **next-step autoreset**, position `t+1` after a termination holds a stale observation - the policy acted on it but the environment discarded that action. The V-Trace advantage and target at that position are meaningless and must be masked:

```python
episode_over = terminated | truncated          # [num_envs, seq_len]
mask = ~episode_over
actor_loss  = (vtrace_advantages * mask).mean()
critic_loss = ((vtrace_targets - values) ** 2 * mask).mean()
```

For truncated boundaries `obs[t+1]` is the genuine continuation state, but a rollout-window truncation on the last step would cause the scan accumulator from the old window to bleed into the new one. Apply the same mask at rollout boundaries regardless:

```python
mask = torch.cat([initial_episode_over, ~episode_over[:, :-1]], dim=1)
```

**Same-step autoreset** does not produce stale observations and needs no masking.

---

### References

* Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., ... & Kavukcuoglu, K. (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* ICML 2018. arXiv:1802.01561.
* Berner, C., Brockman, G., Chan, B., Cheung, V., Dębiak, P., Dennison, C., ... & Zoph, B. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* arXiv:1912.06680.
