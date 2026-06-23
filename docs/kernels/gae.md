# Technical Deep Dive into the GAE Triton Kernel

## Introduction

Generalized Advantage Estimation (GAE; Schulman et al., 2016) is a first-order linear recurrence: each timestep depends on the next. This sequential dependency prevents straightforward parallelization on GPUs, which are designed to execute the same operation across many independent elements simultaneously.

For typical PPO (Schulman et al., 2017) rollout sizes (a few hundred to a few thousand steps), a well-written PyTorch loop is already fast and the overhead of a custom kernel can offset much of the gain. The Triton kernel described here targets workloads where sequence length or batch size is large enough that the parallel reduction pays off - for example, very long trajectories, continuous control tasks with fine time resolution, or research into scalable RL infrastructure.

This tutorial provides a precise, technical walkthrough of how this kernel uses a **Parallel Associative Scan** to restructure the $O(N)$ sequential recurrence into an $O(\log N)$ parallel tree reduction, and explains exactly when and why that matters. Readers unfamiliar with how Triton kernels map onto GPU hardware (memory layout, strides, BLOCK_SIZE, masking) may find the [GPU Concepts](gpu-concepts.md) reference page useful alongside this tutorial.

---

## 1. The Sequential Dependency in GAE

The definition of GAE (Schulman et al., 2016) computes the advantage at timestep $t$ from the TD error and all future advantages within the same episode:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$A_t = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k} = \delta_t + \gamma\lambda \cdot A_{t+1}$$

This is a pure mathematical recurrence with no done flags. The sum naturally terminates at the episode end because $\delta_t = 0$ and $A_t = 0$ for all $t$ beyond termination.

In a rollout buffer a single sequence may contain multiple complete episodes or end mid-episode at a truncation boundary. To prevent advantages from summing across episode boundaries, the kernel replaces the pure mathematical $\delta_t$ and $\beta_t$ with masked versions that depend on two mutually exclusive flags, $d_t^{\text{term}}$ and $d_t^{\text{trunc}}$:

$$\delta_t = r_t + \gamma\,(1 - d_t^{\text{term}})\,V(s_{t+1}) - V(s_t)$$

$$\beta_t = \gamma\lambda(1 - d_t), \quad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

Here $V(s_{t+1})$ is taken from `values[t+1]` at continuing steps and from `bootstrap_values[env, t]` at truncated steps; $d_t^{\text{term}}$ zeros it entirely at termination. Section 5 details what this means for the caller.

Throughout this document $T$ denotes the **sequence length** (the number of steps in the rollout buffer), not necessarily an episode termination.

---

## 2. Associative Scan via Function Composition

The recurrence $A_t = \delta_t + \beta_t A_{t+1}$ creates a strict sequential dependency chain: timestep 100 cannot be computed until timestep 101 finishes, which waited for 102, all the way to 1024. GPU cores that could be working on independent data are instead stalled waiting on this chain. Additionally, a naive loop issues a separate HBM read/write per step rather than loading the whole sequence once.

We can solve this linear recurrence in parallel by rethinking the math not as calculating *numbers* sequentially, but as composing linear *functions* simultaneously.

Any timestep $t$ in GAE is a linear transformation: $f_t(x) = \delta_t + \beta_t x$ (where $\beta_t = \gamma\lambda$ mathematically, and $\beta_t = \gamma\lambda(1-d_t)$ is the kernel implementation to stop propagation at episode boundaries).

If we want to find the accumulated advantage over two adjacent steps, $A$ and $B$, we don't need the final starting number. We can just compose the two linear functions algebraically:


$$f_B(f_A(x)) = \delta_B + \beta_B(\delta_A + \beta_A x) = (\delta_B + \beta_B \delta_A) + (\beta_B \beta_A)x$$

This algebraic composition defines a custom **associative operator ($\oplus$)** that operates on tuples of $(\alpha, \beta)$, where $\alpha_t = \delta_t$ and $\beta_t = \gamma\lambda(1-d_t)$ (the implementation value, as defined in Section 1):


$$(\alpha_B, \beta_B) \oplus (\alpha_A, \beta_A) = (\alpha_B + \beta_B \alpha_A, \,\, \beta_A \beta_B)$$

* **TD accumulation:** New error is the later error $\alpha_B$ plus the weighted earlier error $\beta_B \alpha_A$.
* **Decay product:** The decay terms just multiply.

**Why this works:** Function composition is associative: $(C \circ B) \circ A = C \circ (B \circ A)$. Since the grouping doesn't matter, we don't have to compute them strictly left-to-right (sequentially). We can group them as chunks and snap the chunks together.

**This is why we can operate *only* on the TD errors and decays and do not need the numerical advantage of the higher timestep ($A_{t+1}$) inside the reduction tree.** The tree is accumulating *functions*, not numbers. The tuples represent the state of function composition, not the final advantage value yet.

---

## 3. The Mechanism: Detailed Trace of a 4-Step Reduction Tree

The following trace covers exactly what happens inside the GPU SRAM using a minimal sequence of 4 timesteps. Because GAE is a backward recurrence, the array is reversed before processing.

Let $T_1$ sit at array index $1$, representing the end of the episode ($t=3$).
Let $T_4$ sit at array index $4$, representing the start of the episode ($t=0$).

This trace uses the standard scan algorithm where threads pull data from their 'left'. In this context, 'left' simply means a lower array index. Because our array is reversed, looking to a lower index correctly means pulling data from a chronologically later timestep.

### Setup (Step 0)

The entire trajectory is loaded once into the Streaming Multiprocessor's (SM) Shared Memory (SRAM). We assign one GPU Thread ($Ti$) to each timestep index $i$. Each thread prepares its initial associative tuple $(\alpha_i, \beta_i)$ sitting in local registers:

* $T1$ holds: $T_1 = (\alpha_1, \beta_1)$
* $T2$ holds: $T_2 = (\alpha_2, \beta_2)$
* $T3$ holds: $T_3 = (\alpha_3, \beta_3)$
* $T4$ holds: $T_4 = (\alpha_4, \beta_4)$

### Hardware Loop 1: Parallel Pairs (Distance = 1)

In the first cycle, the hardware tells *every* thread simultaneously: "Look at the neighbor exactly **1** step to your 'left'. Combine their tuple with yours using $\oplus$."

* Thread 1: No left neighbor. Keeps $T_1$.
* Thread 2: Grabs $T_1$ from Thread 1. Computes $T_{1..2} = T_2 \oplus T_1$. (Using the formula from Section 2).
* Thread 3: Grabs $T_2$ from Thread 2. Computes $T_{2..3} = T_3 \oplus T_2$.
* Thread 4: Grabs $T_3$ from Thread 3. Computes $T_{3..4} = T_4 \oplus T_3$.

### Hardware Loop 2: Tree Merge (Distance = 2)

In the second cycle, the hardware tells *every* thread simultaneously: "Look at the neighbor exactly **2** steps to your 'left'. Take the chunk they are currently holding and combine it with yours."

* Thread 1: No neighbor at distance 2. Keeps $T_1$.
* Thread 2: No neighbor at distance 2. Keeps $T_{1..2}$.
* Thread 3: Grabs $T_1$ from Thread 1. Combines with its current chunk: $T_{2..3} \oplus T_1 = T_{1..3}$.
* **Thread 4:** Grabs the current tuple from Thread 2. *Thread 2 is holding $T_{1..2}$!* Thread 4 performs the combination: $T_{3..4} \oplus T_{1..2} = T_{1..4}$.

### Why it is O(log N) Order

The tree structure is defined by the distance threads look to their "left" (lower index) to acquire data.

1. Loop 1: Threads look distance 1.
2. Loop 2: Threads look distance 2.
3. Loop 3: Threads look distance 4...

The distance doubles every single hardware cycle. If you had 1,024 timesteps, standard GAE takes 1,024 sequential steps. The doubling behavior means Triton covers that entire 1024-step sequence in exactly $\log_2(1024) = 10$ parallel reduction cycles. This is the $O(\log N)$ parallel advantage.

---

## 4. How Threads End Up With the Advantage

The associative scan produces intermediate cumulative results for *every single index simultaneously* in SRAM, not just one number at the end. After the $\log_2 N$ loops complete, we look at the final state of the SRAM registers:

### T=1 (End of Episode) Base Case

Thread 1 holds tuple $T_1$. For GAE, we assume the base case: the advantage after the end of an episode (step $t+1=0$) is $0.0$.
If we apply our linear function $f(x) = \delta + \beta x$ to $x=0$, we get $\alpha + \beta(0) = \alpha$.

* Thread 1 advantage: Just the $\alpha$ term of $T_1$. This is $\delta_1$. Correct.

### T=4 Final State and Verification

Thread 4 is holding the final composed tuple $T_{1..4}$. Its advantage is just the $\alpha$ term of that tuple: $\alpha_{1..4}$.

Let's verify $\alpha_{1..4}$ against sequential GAE, substituting back $\alpha_i = \delta_i$ and $\beta_i = \gamma\lambda(1-d_i)$. Sequential PPO calculates:

* $A_1 = \alpha_1$
* $A_2 = \alpha_2 + \beta_2 A_1 = \alpha_2 + \beta_2 \alpha_1$
* $A_3 = \alpha_3 + \beta_3 A_2 = \alpha_3 + \beta_3 \alpha_2 + \beta_3 \beta_2 \alpha_1$
* $A_4 = \alpha_4 + \beta_4 A_3 = \mathbf{\alpha_4 + \beta_4 \alpha_3 + \beta_4 \beta_3 \alpha_2 + \beta_4 \beta_3 \beta_2 \alpha_1}$

The algebra from the composition of chunks in Loop 2 (T4 grabbing $T_{1..2}$ and combining with $T_{3..4}$):


$$\alpha_{1..4} = \alpha_{3..4} + \beta_{3..4}\,\alpha_{1..2}$$

$$\alpha_{1..4} = (\alpha_4 + \beta_4 \alpha_3) + (\beta_3 \beta_4)(\alpha_2 + \beta_2 \alpha_1)$$

$$\alpha_{1..4} = \mathbf{\alpha_4 + \beta_4 \alpha_3 + \beta_4 \beta_3 \alpha_2 + \beta_4 \beta_3 \beta_2 \alpha_1}$$

**The polynomial sitting in Thread 4's SRAM register is identical to the sequential calculation.** Thread 4 did not wait linearly for Thread 3. It built its own mini-chunk $T_{3..4}$ simultaneously while Thread 2 built $T_{1..2}$.

Every thread knows its full history by snapping pre-computed chunks together logarithmically, touching HBM only twice: once to load all $(\alpha, \beta)$ pairs and once to write all advantages.

---

## 5. Handling Episode Boundaries and the Window Bootstrap

During the scan, the GPU only calculates the $\alpha$ and $\beta$ terms of the function $f_i(x)=\alpha_i+\beta_i x$. It does not need to know the starting base case ($x$) to build the tree. The equations for $\delta_t$ and $\beta_t$ were established in Section 1; this section describes the caller's responsibility for preparing the input arrays correctly.

The kernel consumes `values`, `terminateds`, `truncateds`, and `bootstrap_values`, all of shape `[num_envs, seq_len]`. It never sees observations. The three step types from Section 1 translate directly into array requirements.

#### Terminated steps

At a terminated step $t$, `terminateds[t] = 1` zeros $\gamma V(s_{t+1})$ inside $\delta_t$ and sets $\beta_t = 0$, stopping advantage propagation into the previous episode. `values[t+1]` belongs to the next episode but is harmless: the carry is already severed. No entry in `bootstrap_values` is needed at terminated steps; the kernel ignores it there.

#### Truncated steps

At a truncated step $t$, the episode continues, so $\delta_t$ still needs the true continuation value $V(s_{t+1})$. The stored `values[t+1]` belongs to a different episode and must not be used. The caller supplies the correct value as `bootstrap_values[env, t]`, which the Section 1 bracket selects automatically. The carry is still severed ($\beta_t = 0$) so no advantage propagates across the boundary.

#### Window boundary at $t = T-1$

At the last step of the window, $V(s_T)$ lies one step past the end of the `values` tensor. The caller supplies it as `bootstrap_values[env, T-1]` when the episode continues past the window, or leaves it zero if the episode terminated at $T-1$.

This value serves two roles at once. It enters $\delta_{T-1}$ as the next-state value, and it is the scan carry $A_T$ added to every position's local scan result:

$$A_t = (\text{local scan result})_t + (\text{decay product})_t \cdot A_T$$

Both uses read the same number, so a single entry in `bootstrap_values[:, -1]` satisfies both. The double use is necessary: $\delta_{T-1}$ needs $V(s_T)$ to complete the one-step TD residual, while $A_T$ stands in for the true advantage beyond the window, since $A_{T-1} = \delta_{T-1} + \beta_{T-1} A_T$ and the windowed scan alone can only produce $\delta_{T-1}$.

#### The unifying rule

`bootstrap_values` holds the true continuation value $V(s_{t+1})$ at exactly those positions where `values[t+1]` is invalid — truncated steps and the final column — and zero everywhere else. Its shape is `[num_envs, seq_len]`. Setting `RL_TRITON_CORRECTNESS_WARNINGS=1` activates a debug assertion that catches stray nonzero entries.

For the common case where no interior truncations occur, the `last_value` argument (shape `[num_envs]`) provides a convenience: the caller passes the window-edge continuation value directly and the kernel populates `bootstrap_values[:, -1]` automatically. `last_value` and `bootstrap_values` are mutually exclusive.

**Infinite-horizon chunking:** For sequences longer than the flat kernel limit, the scan is chunked. The final advantage of a chronologically later chunk becomes the carry $A_T$ passed into the earlier chunk.

#### Boundary summary

| Situation | $d_t$ | $d_t^{\text{term}}$ | Bootstrap $V(s_{t+1})$ in $\delta_t$ | Decay $\beta_t$ |
|---|---|---|---|---|
| **Terminated** | 1 | 1 | Zeroed by $(1-d_t^{\text{term}})$; `bootstrap_values` ignored | $0$ — scan stops |
| **Truncated** | 1 | 0 | Kept; caller supplies `bootstrap_values[env, t]` | $0$ — scan stops |
| **Window end** | 0 | 0 | Kept; caller supplies `bootstrap_values[env, T-1]`; also seeds carry $A_T$ | $0$ — carry absorbed |

---

## 6. The Hardware Reality: What Triton Actually Does

The ultimate performance gain comes from eliminating trips to the slow High Bandwidth Memory (HBM). Here is the precise lifecycle of the kernel:

**The Single Load (HBM $\rightarrow$ SRAM):** The GPU issues one coalesced read instruction. All raw TD errors and decays are pulled from main HBM into the Streaming Multiprocessor's (SM) ultra-fast Shared Memory (SRAM).

**Thread Allocation (SRAM $\rightarrow$ Registers):** Each thread reads its specific timestep data from SRAM into its local Arithmetic Logic Unit (ALU) registers.

**The Scan Execution (Registers $\rightarrow$ SRAM):** The $O(\log N)$ reduction occurs. Threads communicate intermediate tuples via direct warp shuffles or extremely fast SRAM exchanges.

**The Carry-Over Application (Registers):** Every thread incorporates the carry-over advantage entirely in its local registers in a single clock cycle.

**The Single Store (Registers $\rightarrow$ HBM):** In one massive, synchronized write instruction (`tl.store`), all threads flush their final advantages directly back to HBM.

By trapping the entire reduction loop inside Registers and SRAM, Triton completely bypasses the memory thrashing that plagues standard PyTorch implementations.

---

## 7. Applications

GAE was introduced as the advantage estimator for Proximal Policy Optimization (Schulman et al., 2017), where low-variance advantage estimates are critical for stable policy gradient updates under the clipped surrogate objective. It remains the default advantage estimator in virtually all modern on-policy deep RL algorithms, including implementations of PPO for large-scale training such as those used in reinforcement learning from human feedback (RLHF) pipelines for language model alignment. The $\lambda$ parameter gives practitioners direct control over the bias–variance tradeoff: $\lambda = 0$ reduces GAE to a one-step TD advantage with low variance but high bias, while $\lambda = 1$ recovers the full Monte Carlo return with low bias but high variance. This flexibility, combined with the $O(\log N)$ parallel scan that makes it efficient at large sequence lengths, makes GAE the natural choice wherever long rollouts or many parallel environments are required.

---

## References

* Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. arXiv:1506.02438.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

## Further Reading

* [Visualising GAE](https://pseudo-rnd-thoughts.github.io/blog/visualising-gae/) — A visual walkthrough of GAE's recurrence structure and how advantages accumulate across timesteps.
