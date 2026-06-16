# Returns and Eligibility Traces with Triton Associative Scans

### Introduction

Having covered state-value and Q-value targets (GAE, V-Trace, Retrace), we now look at three simpler but equally important estimators that complete the toolkit: **discounted returns**, **TD(λ) returns**, and **eligibility traces**.

Despite their apparent simplicity, all three share the same first-order linear recurrence structure as the more complex algorithms. They map directly onto the same Triton scan kernel without modification - the only difference is how the inputs $u$ and $v$ are computed in PyTorch before the kernel launches.

The shared kernel solves the backward recurrence

$$A_t = a_t + b_t \cdot A_{t+1}, \qquad A_T = \text{bootstrap}$$

for any choice of $a$ and $b$. Each estimator below specifies its own $a_t$ and $b_t$; the kernel is unaware of the difference.

---

### 1. Discounted Returns

#### Definition

The discounted return (reward-to-go) accumulates future rewards with exponential discounting:

$$G_t = r_t + \gamma(1-d_t) G_{t+1}, \qquad G_T = 0$$

where $d_t \in \{0, 1\}$ is an episode boundary flag that zeros the carry when the episode ends.

This is the purest form of the backward recurrence. There is no value function, no importance sampling correction, no eligibility weight - just rewards compounded by $\gamma$.

#### Telescoping interpretation

The closed-form sum makes the structure explicit:

$$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} \left(\prod_{i=t}^{k-1}(1-d_i)\right) r_k$$

Each reward $r_k$ is discounted by $\gamma^{k-t}$ and masked by the product of all done-flags between $t$ and $k$: once any boundary is hit the product drops to zero and no further rewards contribute.

#### Mapping to the associative scan

$$a_t = r_t, \qquad b_t = \gamma(1-d_t)$$

This is the simplest possible instantiation of $A_t = a_t + b_t \cdot A_{t+1}$. No derived quantities, no additional tensors.

#### Bootstrap at truncated boundaries

The default boundary value is $G_T = 0$, which is correct when $d_{T-1} = 1$ (true episode end). For truncated windows where the episode continues beyond the sequence, pass `bootstrap_values = V(s_T)` to avoid discarding the tail of the trajectory.

---

### 2. TD(λ) Returns (λ-Returns)

#### Definition

The TD(λ) return interpolates between one-step TD (λ=0) and full Monte Carlo (λ=1):

$$G^\lambda_t = r_t + \gamma(1-d_t)\left[(1-\lambda)V(s_{t+1}) + \lambda G^\lambda_{t+1}\right], \qquad G^\lambda_T = \text{bootstrap}$$

At every step the value function $V(s_{t+1})$ provides an intermediate bootstrap weighted by $(1-\lambda)$, and the remaining weight $\lambda$ propagates the multi-step return forward.

#### Special cases

| λ | Reduces to |
|---|---|
| 0 | One-step TD: $G_t = r_t + \gamma(1-d_t)V(s_{t+1})$ |
| 1 | Pure discounted return: $G_t = r_t + \gamma(1-d_t)G_{t+1}$ |

At λ=1 the value function terms drop out entirely and `compute_lambda_returns` reproduces `compute_discounted_returns` numerically. The two functions exist as separate API entries because their caller profiles differ: discounted returns never receives a value function tensor at all.

#### Mapping to the associative scan

Expanding and grouping the recurrence into $a + b \cdot G^\lambda_{t+1}$ form:

$$a_t = r_t + \gamma(1-\lambda)(1-d_t)V(s_{t+1}), \qquad b_t = \gamma\lambda(1-d_t)$$

The value function is absorbed directly into the additive term $a_t$ at every step. This means `next_values` acts as a per-step bootstrap that is always available - no separate boundary tensor is needed for on-policy truncated episodes as long as `next_values[:, -1]` contains $V(s_T)$.

#### Bootstrap at truncated boundaries

Because $V(s_{t+1})$ is mixed into $a_t$ at every step, the recurrence already carries value information up to (but not including) the boundary. Pass `bootstrap_values = next_values[:, -1]` for truncated windows; omit it (defaults to zero) for terminated episodes where $d_{T-1}=1$.

---

### 3. Eligibility Traces

#### Definition

The eligibility trace is the package's only **forward** recurrence (it scans from $t = 0$ upward, accumulating the past):

$$\mathbf{z}_t = \gamma\lambda(1 - d_t)\,\mathbf{z}_{t-1} + \nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t), \qquad \mathbf{z}_{-1} = \mathbf{0}$$

The trace $\mathbf{z}_t$ is a decayed accumulation of the value-function gradients at all previously visited states. The done flag $d_t$ resets the trace at episode boundaries. This is the general semi-gradient TD(λ) trace and applies to any differentiable function approximator - including the neural-network value functions this library targets.

> The trace accumulates the value gradient $\nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t)$, **not** the feature vector $x_t$. Throughout this documentation $x_t$ denotes features; the eligibility-trace input is the gradient.

#### Where the trace is used, and the connection to TD(λ)

In TD(λ) there are two equivalent views of the same algorithm:

- **Forward view**: compute the full λ-weighted mixture of $n$-step returns $G^\lambda_t$ looking forward in time, then update $\mathbf{w}$ toward that target. This requires the complete trajectory before any update can be issued - it is what `compute_lambda_returns` computes (via a backward scan over the trajectory).

- **Backward view**: at each step $t$, apply a weight update using only the current one-step TD error

$$\delta_t = r_t + \gamma\hat{V}(s_{t+1}, \mathbf{w}_t) - \hat{V}(s_t, \mathbf{w}_t)$$

distributed across all previously visited states according to their current eligibility:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{z}_t$$

The decay rate $\gamma\lambda$ ensures that a state visited $k$ steps ago contributes with weight $(\gamma\lambda)^k$ - more recent states receive more credit. This is the backward-view answer to the credit-assignment problem: rather than waiting for a full trajectory and computing the λ-return forward, the trace propagates each TD error backward to all past states online at each step.

#### Linear special case

For linear function approximation $\hat{V}(s) = \mathbf{w}^\top x(s)$, the gradient is the feature vector itself, $\nabla_\mathbf{w}\hat{V}(s_t) = x_t$, and the general trace reduces to a decayed accumulation of feature vectors:

$$\mathbf{z}_t = \gamma\lambda(1 - d_t)\,\mathbf{z}_{t-1} + x_t$$

This is the **only** place $x_t$ enters the trace. The feature-vector form is the special case of the gradient form, not the other way around.

#### Nonlinear function approximation

The gradient trace is a well-defined, usable update for nonlinear approximators such as neural networks - it is not mathematically unsound. The practical complication is **gradient staleness**: under online updates, $\mathbf{w}$ moves at every step, so consecutive terms in the trace are gradients evaluated at slightly different parameter vectors. This makes the online backward-view update a **biased approximation** whose error scales with the step size $\alpha$ - negligible for shallow networks or small step sizes, and larger for deep networks with large updates. The forward–backward equivalence is exact offline (when $\mathbf{w}$ is held fixed across the episode), and remains exact online only for linear FA via the True Online TD(λ) dutch-trace derivation; for nonlinear online there is no guaranteed exact equivalence, only the biased approximation above.

#### Forward view in deep RL

The field responded to staleness by preferring the forward view: compute the λ-weighted return over a (possibly truncated) rollout, then update. GAE, V-trace, and Retrace are all forward-view variants of this idea - eligibility traces in substance, computed over a trajectory rather than maintained as an online backward accumulation. The backward online trace is rare in deep RL for practical reasons (per-step online updates conflict with minibatch/replay training; the trace vector has as many entries as the full parameter vector; staleness in deep nets), not because the construction is invalid. Recent work has shown the backward trace can be made to work with deep nets by correcting for parameter drift - see Kobayashi (2020/2022), Daley & Amato, and Harb & Precup (2017).

#### Mapping to the associative scan

$$a_t = \mathbf{g}_t, \qquad b_t = \gamma\lambda(1-d_t)$$

The form is structurally identical to discounted returns, but processed left-to-right via `_run_scan_forward` instead of right-to-left via `_run_scan`. The forward kernel reads `u` and `v` in natural time order; no reversal is needed.

The $\lambda=0$ case collapses the trace to $\mathbf{z}_t = \mathbf{g}_t$ (current input only, no history); $\lambda=1$ retains the full undiscounted accumulation of all past inputs.

#### How the parallel scan computes a backward-looking trace

The recurrence $\mathbf{z}_t = \mathbf{g}_t + \gamma\lambda(1-d_t)\,\mathbf{z}_{t-1}$ is sequential in appearance - each step depends on the previous - but the same associative structure that enables the backward scan parallelises it equally well in the forward direction.

The combine function is identical to the backward case:

$$(a_B, b_B) \oplus (a_A, b_A) = (a_B + b_B\,a_A,\; b_A b_B)$$

One thread block per environment loads the entire sequence $(\mathbf{g}_0, \gamma\lambda(1-d_0)), \ldots, (\mathbf{g}_{T-1}, \gamma\lambda(1-d_{T-1}))$ in natural order. `tl.associative_scan` then computes the prefix reduction in $O(\log T)$ parallel steps across SIMD lanes within the block.

After the scan, position $t$ holds the pair $(a_{0..t},\, b_{0..t})$ where:

$$a_{0..t} = \mathbf{g}_t + \gamma\lambda\,\mathbf{g}_{t-1} + (\gamma\lambda)^2\mathbf{g}_{t-2} + \cdots + (\gamma\lambda)^t\,\mathbf{g}_0$$
$$b_{0..t} = \prod_{k=0}^{t} \gamma\lambda(1-d_k)$$

The seed $\mathbf{z}_{-1}$ is then folded in with a single fused addition:

$$\mathbf{z}_t = a_{0..t} + b_{0..t} \cdot \mathbf{z}_{-1}$$

Done flags collapse any $v$ factor to zero, which zeroes all further contributions of past inputs from before the episode boundary - exactly the correct trace reset.

The key observation is that the scan runs the same reduction tree as the backward kernel; the only implementation difference is that the input array is loaded in forward order rather than reversed. Every position in the output corresponds to a partial scan from $t=0$ up to that position, which is precisely the decayed sum of past gradients the trace requires.

#### Implementation note

`compute_eligibility_traces` computes the forward scan over a generic per-step input `gradients` $\mathbf{g}_t$; mathematically that input is the value gradient $\nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t)$ (or the feature vector $x_t$ in the linear special case). Because the trace at step $t$ depends on every state visited from $t = 0$ up to the present, it is inherently a function of the past - there is no choice but to scan forward.

#### Limitation

The forward scan uses the flat single-block kernel only. Sequences longer than 131072 steps are not supported (a chunked forward kernel has not been implemented). See `NOTES.md` for context.

---

### 4. Verification via the Reduction Tree

All three estimators use the same associative operator $\oplus$:

$$(a_B, b_B) \oplus (a_A, b_A) = (a_B + b_B a_A,\; b_A b_B)$$

Each mapping is verified over a 4-step sequence with no done flags and scalar inputs ($\gamma < 1$, $\lambda \in (0,1)$). The array is reversed for the backward scan so that index 4 corresponds to chronological $t=1$.

After the $O(\log N)$ scan, the accumulated result at position 4 is:

$$a_{1..4} = a_4 + b_4 a_3 + b_4 b_3 a_2 + b_4 b_3 b_2 a_1$$

#### Discounted returns

Substituting $a_k = r_k$ and $b_k = \gamma$:

$$a_{1..4} = r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4$$

This is exactly $G_1$, the discounted sum of all rewards from $t=1$ onward.

#### TD(λ) returns

Substituting $a_k = r_k + \gamma(1-\lambda)V(s_{k+1})$ and $b_k = \gamma\lambda$:

$$a_{1..4} = \bigl[r_1 + \gamma(1{-}\lambda)V_2\bigr]
           + \gamma\lambda\bigl[r_2 + \gamma(1{-}\lambda)V_3\bigr]
           + (\gamma\lambda)^2\bigl[r_3 + \gamma(1{-}\lambda)V_4\bigr]
           + (\gamma\lambda)^3\bigl[r_4 + \gamma(1{-}\lambda)V_5\bigr]$$

where $V_k = V(s_k)$. Collecting reward and value terms:

$$= \sum_{k=1}^{4} (\gamma\lambda)^{k-1} r_k + \gamma(1-\lambda)\sum_{k=1}^{4}(\gamma\lambda)^{k-1}V_{k+1}$$

This is the λ-weighted mixture of 1- through 4-step returns truncated at $t=4$, exactly matching $G^\lambda_1$ from the forward-view definition.

#### Eligibility traces

For the forward scan, the array is processed left-to-right. At position $t=4$ (the last step), substituting $a_k = \mathbf{g}_k$ (the per-step input - a value gradient, or a feature vector in the linear special case) and $b_k = \gamma\lambda$:

$$\mathbf{z}_4 = \mathbf{g}_4 + \gamma\lambda\, \mathbf{g}_3 + (\gamma\lambda)^2 \mathbf{g}_2 + (\gamma\lambda)^3 \mathbf{g}_1$$

This is the decayed accumulation of all past inputs up to $t=4$, which is exactly $\mathbf{z}_4$ from the forward recurrence definition.

---

### 5. Outputs

Once the $O(\log N)$ scan finishes, each position holds its result directly with no further post-processing:

$$G_t = A_t \quad \text{(discounted returns)}$$
$$G^\lambda_t = A_t \quad \text{(λ-returns)}$$
$$\mathbf{z}_t = A_t \quad \text{(eligibility traces)}$$

This contrasts with GAE ($A_t = \Delta_t + \delta_t$) and Retrace ($Q^{ret}_t = \Delta_t + Q_t$), where a separate baseline must be added back after the scan. Here, $a_t$ already encodes the full additive term, so $A_t$ is the target directly.

---

### 6. Applications in AI

**Discounted Returns in Policy Gradient Methods**
Discounted returns form the direct reward signal in REINFORCE and its variants. In group-relative policy optimization (GRPO), reward-to-go is used as the advantage signal across a group of sampled responses, providing variance reduction without a learned critic (Shao et al., 2024).

**TD(λ) Returns in On-Policy Actor-Critic**
λ-returns provide a unified interpolation between the high-bias/low-variance one-step TD target and the low-bias/high-variance Monte Carlo return. PPO implementations often use λ-returns (which are numerically equivalent to GAE targets when the value function is used as the baseline) to balance this tradeoff during policy updates (Schulman et al., 2017).

**Eligibility Traces in Online TD(λ)**
The backward-view TD(λ) update $\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta_t\,\mathbf{e}_t$ is the classical algorithm for online credit assignment in tabular and linear-function-approximation settings, allowing single-step updates to propagate learning signals to all previously visited states (Sutton & Barto, 2018).

---

### References

* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
* Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., ... & Luo, F. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
* Kobayashi, T. (2022). *Towards Eligibility Traces for Deep Neural Networks.* Adaptive Behavior.
* Daley, B., & Amato, C. *Reconciling λ-Returns with Experience Replay.*
* Harb, J., & Precup, D. (2017). *Investigating Recurrence and Eligibility Traces in Deep Q-Networks.* arXiv:1704.05495.
