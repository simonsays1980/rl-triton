# Returns and Eligibility Traces with Triton Associative Scans

## Introduction

Having covered state-value and Q-value targets (GAE, V-Trace, Retrace), we now look at three simpler but equally important estimators that complete the toolkit: **discounted returns**, **TD(λ) returns**, and **eligibility traces**.

Despite their apparent simplicity, all three share the same first-order linear recurrence structure as the more complex algorithms. They map directly onto the same Triton scan kernel without modification -- the only difference is how the inputs $\alpha$ and $\beta$ are computed in PyTorch before the kernel launches.

The shared kernel solves the backward recurrence

$$A_t = \alpha_t + \beta_t \cdot A_{t+1}, \qquad A_T = 0$$

for any choice of $\alpha$ and $\beta$. Each estimator below specifies its own $\alpha_t$ and $\beta_t$; the kernel is unaware of the difference.

---

## 1. Discounted Returns

### Definition

The discounted return (reward-to-go) accumulates future rewards with exponential discounting:

$$G_t = r_t + \gamma\, G_{t+1}$$

This is the purest form of the backward recurrence -- no value function, no importance sampling correction, no eligibility weight, just rewards compounded by $\gamma$. The sum naturally terminates at the episode boundary because $r_t = 0$ and $G_t = 0$ beyond termination.

In a rollout buffer a single sequence may contain multiple episodes or end mid-episode at a truncation boundary. To prevent returns from summing across episode boundaries, the kernel introduces masked versions that depend on two mutually exclusive flags, $d_t^{\text{term}}$ and $d_t^{\text{trunc}}$:

$$G_t = r_t + \gamma\,(1 - d_t)\, G_{t+1}, \qquad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

[Handling episode boundaries](#handling-episode-boundaries) details what this means for the caller.

### Telescoping interpretation

The closed-form sum makes the structure explicit:

$$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} \left(\prod_{i=t}^{k-1}(1-d_i)\right) r_k$$

Each reward $r_k$ is discounted by $\gamma^{k-t}$ and masked by the product of all boundary flags between $t$ and $k$.

### Mapping to the associative scan

$$\alpha_t = r_t, \qquad \beta_t = \gamma(1 - d_t)$$

At truncated steps, $\beta_t = 0$ stops the carry, but the true continuation value $V(s_{t+1}^{\text{true}})$ must still enter $G_t$ as a one-step bootstrap. The caller absorbs it into $\alpha_t$ at that step:

$$\alpha_t = r_t + \gamma\,d_t^{\text{trunc}}\,V(s_{t+1}^{\text{true}})$$

where $V(s_{t+1}^{\text{true}})$ is taken from `bootstrap_values[env, t]`. The recurrence structure is unchanged.

### Handling episode boundaries

The kernel consumes `rewards`, `terminateds`, `truncateds`, and `bootstrap_values`, all of shape `[num_envs, seq_len]`. It never sees observations.

#### Terminated steps

At a terminated step $t$, $d_t = 1$ zeros $\beta_t$ -- the carry stops and no continuation value is needed. `bootstrap_values` is ignored at terminated steps.

#### Truncated steps

At a truncated step $t$, the episode continues, so $G_t$ still needs the true continuation value $V(s_{t+1}^{\text{true}})$. The carry is severed ($\beta_t = 0$) but the caller supplies the correct value as `bootstrap_values[env, t]`, which is absorbed into $\alpha_t$.

#### Window boundary at $t = T-1$

At the last step of the window, the caller supplies `bootstrap_values[env, T-1] = V(s_T)` when the episode continues past the rollout, or leaves it zero if the episode terminated at that step.

#### The unifying rule

`bootstrap_values` holds the true continuation value $V(s_{t+1}^{\text{true}})$ at exactly those positions where the episode continues beyond the current step -- truncated steps and the final column -- and zero everywhere else. Its shape is `[num_envs, seq_len]`.

### Boundary summary

| Situation | $d_t$ | $d_t^{\text{term}}$ | $\alpha_t$ | $\beta_t$ |
|---|---|---|---|---|
| **Terminated** | 1 | 1 | $r_t$ | $0$ -- carry stops |
| **Truncated** | 1 | 0 | $r_t + \gamma\,V(s_{t+1}^{\text{true}})$; caller supplies via `bootstrap_values[env, t]` | $0$ -- carry stops |
| **Window end** | 0 | 0 | $r_{T-1} + \gamma\,V(s_T)$; caller supplies via `bootstrap_values[env, T-1]` | $\gamma$ -- continues |

---

## 2. TD(λ) Returns (λ-Returns)

### Definition

The TD(λ) return interpolates between one-step TD ($\lambda=0$) and full Monte Carlo ($\lambda=1$):

$$G^\lambda_t = r_t + \gamma\left[(1-\lambda)V(s_{t+1}) + \lambda\, G^\lambda_{t+1}\right]$$

At every step the value function $V(s_{t+1})$ provides an intermediate bootstrap weighted by $(1-\lambda)$, and the remaining weight $\lambda$ propagates the multi-step return. The sum naturally terminates at the episode boundary.

In a rollout buffer containing multiple episodes or truncation boundaries, the kernel introduces masked versions:

$$G^\lambda_t = r_t + \gamma(1 - d_t)\left[(1-\lambda)V(s_{t+1}) + \lambda\, G^\lambda_{t+1}\right], \qquad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}$$

[Handling episode boundaries](#handling-episode-boundaries_1) details what this means for the caller.

### Special cases

| $\lambda$ | Reduces to |
|---|---|
| 0 | One-step TD: $G_t = r_t + \gamma(1 - d_t)V(s_{t+1})$ |
| 1 | Pure discounted return (value terms drop out entirely) |

At $\lambda=1$ the value function terms drop out and `compute_lambda_returns` reproduces `compute_discounted_returns` numerically. The two functions exist as separate API entries because their caller profiles differ: discounted returns never receives a value function tensor at all.

### Mapping to the associative scan

Expanding and grouping the recurrence into $\alpha_t + \beta_t \cdot G^\lambda_{t+1}$ form:

$$\alpha_t = r_t + \gamma(1-\lambda)(1 - d_t^{\text{term}})\,V(s_{t+1}), \qquad \beta_t = \gamma\lambda(1 - d_t)$$

Here $V(s_{t+1})$ is taken from `values[t+1]` at continuing steps and from `bootstrap_values[env, t]` at truncated steps; $d_t^{\text{term}}$ zeros it entirely at termination. The value function is absorbed into the additive term $\alpha_t$ at every step.

### Handling episode boundaries

The kernel consumes `rewards`, `values`, `terminateds`, `truncateds`, and `bootstrap_values`, all of shape `[num_envs, seq_len]`. It never sees observations.

#### Terminated steps

At a terminated step $t$, `terminateds[t] = 1` zeros $\gamma(1-\lambda)V(s_{t+1})$ inside $\alpha_t$ and sets $\beta_t = 0$, stopping return propagation into the previous episode. `values[t+1]` belongs to the next episode but is harmless: it is zeroed by $d_t^{\text{term}}$ and the carry is already severed. No entry in `bootstrap_values` is needed at terminated steps; the kernel ignores it there.

#### Truncated steps

At a truncated step $t$, the episode continues, so $\alpha_t$ still needs the true continuation value $V(s_{t+1})$. The stored `values[t+1]` belongs to a different episode and must not be used. The caller supplies the correct value as `bootstrap_values[env, t]`, which the formula above selects automatically. The carry is still severed ($\beta_t = 0$) so no return propagates across the boundary.

#### Window boundary at $t = T-1$

At the last step of the window, $V(s_T)$ lies one step past the end of the `values` tensor. The caller supplies it as `bootstrap_values[env, T-1]` when the episode continues past the window, or leaves it zero if the episode terminated at $T-1$.

#### The unifying rule

`bootstrap_values` holds the true continuation value $V(s_{t+1})$ at exactly those positions where `values[t+1]` is invalid -- truncated steps and the final column -- and zero everywhere else. Its shape is `[num_envs, seq_len]`.

### Boundary summary

| Situation | $d_t$ | $d_t^{\text{term}}$ | $V(s_{t+1})$ in $\alpha_t$ | $\beta_t$ |
|---|---|---|---|---|
| **Terminated** | 1 | 1 | Zeroed by $d_t^{\text{term}}$; `bootstrap_values` ignored | $0$ -- carry stops |
| **Truncated** | 1 | 0 | Kept; caller supplies `bootstrap_values[env, t]` | $0$ -- carry stops |
| **Window end** | 0 | 0 | Kept; caller supplies `bootstrap_values[env, T-1]` | $\gamma\lambda$ |

---

## 3. Eligibility Traces

### Definition

The eligibility trace is the package's only **forward** recurrence (it scans from $t = 0$ upward, accumulating the past):

$$\mathbf{z}_t = \gamma\lambda\,\mathbf{z}_{t-1} + \nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t), \qquad \mathbf{z}_{-1} = \mathbf{0}$$

The trace $\mathbf{z}_t$ is a decayed accumulation of the value-function gradients at all previously visited states within a single episode. In a rollout buffer containing multiple episodes, the kernel introduces the masked version:

$$\mathbf{z}_t = \gamma\lambda(1 - d_{t-1})\,\mathbf{z}_{t-1} + \nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t), \qquad d_t = d_t^{\text{term}} \vee d_t^{\text{trunc}}, \qquad d_{-1} := 0$$

$d_t = 1$ means the episode **ends at** $t$ -- the same convention as `compute_gae` and every other kernel in this package, and the convention a raw Gymnasium `terminated`/`truncated` flag already uses without any shifting. The trace carried *into* $t$ is severed whenever the *preceding* step ended an episode ($d_{t-1}=1$), i.e. whenever $t$ is the first step of a new episode, which resets the trace to zero there. This is the general semi-gradient TD(λ) trace and applies to any differentiable function approximator -- including the neural-network value functions this library targets.

Unlike the backward algorithms, the distinction between terminated and truncated steps does not matter here. In GAE, V-Trace, and Retrace, a truncated step requires injecting $V(s_{t+1}^{\text{true}})$ as a bootstrap into the TD error. The eligibility trace carries no TD error and no continuation value: $\alpha_t = \mathbf{g}_t$ is simply the gradient at the current step. At any boundary, terminated or truncated, the correct action is the same -- zero $\beta_t$ to prevent gradients from the ending episode from bleeding into the new one. The step at $t+1$ starts a fresh trace from $\mathbf{g}_{t+1}$ with no memory of what came before -- gating on $d_{t-1}$ rather than $d_t$ is what makes this land at the right index: $d_t$ describes the boundary $t$ itself ends at, not the boundary $t$ begins after. The caller passes `terminateds` and `truncateds` and the kernel combines them as $d_t$, unshifted; no `bootstrap_values` argument exists for eligibility traces.

> The trace accumulates the value gradient $\nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t)$, **not** the feature vector $x_t$. Throughout this documentation $x_t$ denotes features; the eligibility-trace input is the gradient.

### Where the trace is used, and the connection to TD(λ)

In TD(λ) there are two equivalent views of the same algorithm:

- **Forward view**: compute the full λ-weighted mixture of $n$-step returns $G^\lambda_t$ looking forward in time, then update $\mathbf{w}$ toward that target. This requires the complete trajectory before any update can be issued -- it is what `compute_lambda_returns` computes (via a backward scan over the trajectory).

- **Backward view**: at each step $t$, apply a weight update using only the current one-step TD error

$$\delta_t = r_t + \gamma\hat{V}(s_{t+1}, \mathbf{w}_t) - \hat{V}(s_t, \mathbf{w}_t)$$

distributed across all previously visited states according to their current eligibility:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{z}_t$$

The decay rate $\gamma\lambda$ ensures that a state visited $k$ steps ago contributes with weight $(\gamma\lambda)^k$ -- more recent states receive more credit. This is the backward-view answer to the credit-assignment problem: rather than waiting for a full trajectory and computing the λ-return forward, the trace propagates each TD error backward to all past states online at each step.

### Linear special case

For linear function approximation $\hat{V}(s) = \mathbf{w}^\top x(s)$, the gradient is the feature vector itself, $\nabla_\mathbf{w}\hat{V}(s_t) = x_t$, and the general trace reduces to a decayed accumulation of feature vectors:

$$\mathbf{z}_t = \gamma\lambda(1 - d_{t-1})\,\mathbf{z}_{t-1} + x_t$$

This is the **only** place $x_t$ enters the trace. The feature-vector form is the special case of the gradient form, not the other way around.

### Nonlinear function approximation

The gradient trace is a well-defined, usable update for nonlinear approximators such as neural networks -- it is not mathematically unsound. The practical complication is **gradient staleness**: under online updates, $\mathbf{w}$ moves at every step, so consecutive terms in the trace are gradients evaluated at slightly different parameter vectors. This makes the online backward-view update a **biased approximation** whose error scales with the step size $\alpha$ -- negligible for shallow networks or small step sizes, and larger for deep networks with large updates. The forward–backward equivalence is exact offline (when $\mathbf{w}$ is held fixed across the episode), and remains exact online only for linear FA via the True Online TD(λ) dutch-trace derivation; for nonlinear online there is no guaranteed exact equivalence, only the biased approximation above.

### Forward view in deep RL

The field responded to staleness by preferring the forward view: compute the λ-weighted return over a (possibly truncated) rollout, then update. GAE, V-Trace, and Retrace are all forward-view variants of this idea -- eligibility traces in substance, computed over a trajectory rather than maintained as an online backward accumulation. The backward online trace is rare in deep RL for practical reasons (per-step online updates conflict with minibatch/replay training; the trace vector has as many entries as the full parameter vector; staleness in deep nets), not because the construction is invalid. Recent work has shown the backward trace can be made to work with deep nets by correcting for parameter drift -- see Kobayashi (2020/2022), Daley & Amato (2019), and Harb & Precup (2017).

### Mapping to the associative scan

Writing $\mathbf{g}_t := \nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t)$ for the per-step gradient input:

$$\alpha_t = \mathbf{g}_t, \qquad \beta_t = \gamma\lambda(1-d_{t-1}), \qquad d_{-1} := 0$$

The form is structurally identical to discounted returns, but processed left-to-right via `_run_scan_forward` instead of right-to-left via `_run_scan`. The forward kernel reads $\alpha$ and $\beta$ in natural time order; no reversal is needed. $\beta_t$ reads the *preceding* step's done flag, not $t$'s own -- see [Definition](#definition) above for why the index shifts between the backward and forward kernels.

The $\lambda=0$ case collapses the trace to $\mathbf{z}_t = \mathbf{g}_t$ (current input only, no history); $\lambda=1$ retains the full undiscounted accumulation of all past inputs.

### How the parallel scan computes a backward-looking trace

The recurrence $\mathbf{z}_t = \mathbf{g}_t + \gamma\lambda(1-d_{t-1})\,\mathbf{z}_{t-1}$ is sequential in appearance -- each step depends on the previous -- but the same associative structure that enables the backward scan parallelises it equally well in the forward direction.

The combine function is identical to the backward case:

$$(\alpha_B, \beta_B) \oplus (\alpha_A, \beta_A) = (\alpha_B + \beta_B\,\alpha_A,\; \beta_A \beta_B)$$

One thread block per environment loads the entire sequence $(\mathbf{g}_0, \gamma\lambda(1-d_{-1})), \ldots, (\mathbf{g}_{T-1}, \gamma\lambda(1-d_{T-2}))$ in natural order ($d_{-1}=0$ by convention). `tl.associative_scan` then computes the prefix reduction in $O(\log T)$ parallel steps across SIMD lanes within the block.

After the scan, position $t$ holds the pair $(\alpha_{0..t},\, \beta_{0..t})$ where:

$$\alpha_{0..t} = \mathbf{g}_t + \gamma\lambda\,\mathbf{g}_{t-1} + (\gamma\lambda)^2\mathbf{g}_{t-2} + \cdots + (\gamma\lambda)^t\,\mathbf{g}_0$$

$$\beta_{0..t} = \prod_{k=0}^{t} \gamma\lambda(1-d_{k-1})$$

The seed $\mathbf{z}_{-1}$ is then folded in with a single fused addition:

$$\mathbf{z}_t = \alpha_{0..t} + \beta_{0..t} \cdot \mathbf{z}_{-1}$$

Done flags collapse any $\beta$ factor to zero, which zeroes all further contributions of past inputs from before the episode boundary -- exactly the correct trace reset. Because $\beta_t$ reads $d_{t-1}$, the reset takes effect starting at the position *after* the flagged step, matching the "episode ends at $t$, new episode begins at $t+1$" convention used throughout this package.

The key observation is that the scan runs the same reduction tree as the backward kernel; the only implementation difference is that the input array is loaded in forward order rather than reversed. Every position in the output corresponds to a partial scan from $t=0$ up to that position, which is precisely the decayed sum of past gradients the trace requires.

### Implementation note

`compute_eligibility_traces` computes the forward scan over a generic per-step input `gradients` $\mathbf{g}_t$; mathematically that input is the value gradient $\nabla_\mathbf{w}\hat{V}(s_t, \mathbf{w}_t)$ (or the feature vector $x_t$ in the linear special case). Because the trace at step $t$ depends on every state visited from $t = 0$ up to the present, it is inherently a function of the past -- there is no choice but to scan forward.

### Limitation

The forward scan uses the flat single-block kernel only. Sequences longer than 131072 steps are not supported (a chunked forward kernel has not been implemented). See `NOTES.md` for context.

---

## 4. Verification via the Reduction Tree

All three estimators use the same associative operator $\oplus$:

$$(\alpha_B, \beta_B) \oplus (\alpha_A, \beta_A) = (\alpha_B + \beta_B \alpha_A,\; \beta_A \beta_B)$$

Each mapping is verified over a 4-step sequence with no done flags and scalar inputs ($\gamma < 1$, $\lambda \in (0,1)$). The array is reversed for the backward scan so that index 4 corresponds to chronological $t=1$.

After the $O(\log N)$ scan, the accumulated result at position 4 is:

$$\alpha_{1..4} = \alpha_4 + \beta_4 \alpha_3 + \beta_4 \beta_3 \alpha_2 + \beta_4 \beta_3 \beta_2 \alpha_1$$

### Discounted returns

Substituting $\alpha_k = r_k$ and $\beta_k = \gamma$:

$$\alpha_{1..4} = r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4$$

This is exactly $G_1$, the discounted sum of all rewards from $t=1$ onward.

### TD(λ) returns

Assuming no done flags and no truncations in this 4-step sequence, substituting $\alpha_k = r_k + \gamma(1-\lambda)V(s_{k+1})$ and $\beta_k = \gamma\lambda$:

$$\alpha_{1..4} = \bigl[r_1 + \gamma(1{-}\lambda)V_2\bigr]
           + \gamma\lambda\bigl[r_2 + \gamma(1{-}\lambda)V_3\bigr]
           + (\gamma\lambda)^2\bigl[r_3 + \gamma(1{-}\lambda)V_4\bigr]
           + (\gamma\lambda)^3\bigl[r_4 + \gamma(1{-}\lambda)V_5\bigr]$$

where $V_k = V(s_k)$. Collecting reward and value terms:

$$= \sum_{k=1}^{4} (\gamma\lambda)^{k-1} r_k + \gamma(1-\lambda)\sum_{k=1}^{4}(\gamma\lambda)^{k-1}V_{k+1}$$

This is the λ-weighted mixture of 1- through 4-step returns truncated at $t=4$, exactly matching $G^\lambda_1$ from the forward-view definition. The verification assumes no done flags; how truncated steps are handled is described in [Handling episode boundaries](#handling-episode-boundaries_1).

### Eligibility traces

For the forward scan, the array is processed left-to-right. At position $t=4$ (the last step), substituting $\alpha_k = \mathbf{g}_k$ and $\beta_k = \gamma\lambda$:

$$\mathbf{z}_4 = \mathbf{g}_4 + \gamma\lambda\, \mathbf{g}_3 + (\gamma\lambda)^2 \mathbf{g}_2 + (\gamma\lambda)^3 \mathbf{g}_1$$

This is the decayed accumulation of all past inputs up to $t=4$, exactly $\mathbf{z}_4$ from the forward recurrence definition.

---

## 5. Outputs

Once the $O(\log N)$ scan finishes, each position holds its result directly with no further post-processing:

$$G_t = A_t \quad \text{(discounted returns)}$$

$$G^\lambda_t = A_t \quad \text{(λ-returns)}$$

$$\mathbf{z}_t = A_t \quad \text{(eligibility traces)}$$

This contrasts with GAE ($A_t = \Delta_t + V(s_t)$) and Retrace ($Q^{ret}_t = \Delta_t + Q(s_t, a_t)$), where a separate baseline must be added back after the scan. Here, $\alpha_t$ already encodes the full additive term, so $A_t$ is the target directly.

---

## 6. Applications

**Discounted Returns in Policy Gradient Methods**
Discounted returns form the direct reward signal in REINFORCE and its variants, where the return-to-go at each step is the credit assigned to the action taken there. Note that critic-free methods for LLM post-training do not generally use this recurrence: GRPO (Shao et al., 2024) computes a group-normalized scalar per sampled response and broadcasts it across every token, so there is no per-timestep discounted accumulation to scan.

**TD(λ) Returns in On-Policy Actor-Critic**
λ-returns provide a unified interpolation between the high-bias/low-variance one-step TD target and the low-bias/high-variance Monte Carlo return. PPO implementations often use λ-returns (which are numerically equivalent to GAE targets when the value function is used as the baseline) to balance this tradeoff during policy updates (Schulman et al., 2017).

**Eligibility Traces in Online TD(λ)**
The backward-view TD(λ) update $\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta_t\,\mathbf{e}_t$ is the classical algorithm for online credit assignment in tabular and linear-function-approximation settings, allowing single-step updates to propagate learning signals to all previously visited states (Sutton & Barto, 2018).

---

## References

* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
* Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., ... & Luo, F. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
* Kobayashi, T. (2022). *Towards Eligibility Traces for Deep Neural Networks.* Adaptive Behavior.
* Daley, B., & Amato, C. (2019). *Reconciling λ-Returns with Experience Replay.* NeurIPS 2019.
* Harb, J., & Precup, D. (2017). *Investigating Eligibility Traces in Deep Q-Networks.* arXiv:1704.05495.
