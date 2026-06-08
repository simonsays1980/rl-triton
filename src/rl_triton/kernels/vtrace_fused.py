import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def vtrace_fused_kernel(
    log_pi_target_ptr, log_pi_behavior_ptr,
    values_ptr,
    rewards_ptr, dones_ptr,
    targets_ptr, advantages_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    gamma,
    rho_bar,
    c_bar,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused V-Trace kernel: computes targets and advantages in a single program.

    V(s_{t+1}) is read directly from values[t+1] — no separate next_values tensor.
    At the boundary (t=T-1, offs=0), bootstrap_values serves as V(s_T).

    Indexing convention
    -------------------
    offs = 0, 1, …, seq_len-1   (block position, loaded in this order)
    rev  = seq_len - 1 - offs   (real-time array index: offs=0 → t=T-1)

    v_next[t] = V(s_{t+1}):
        offs > 0  →  values[base + rev + 1]   (t < T-1; rev+1 = t+1)
        offs = 0  →  bootstrap                 (t = T-1; V(s_T) from bootstrap_ptr)

    Step 4 — next vtrace target for advantage:
        offs > 0  →  targets[base + rev + 1]   (target[t+1], loaded after barrier)
        offs = 0  →  bootstrap                 (boundary; bootstrap already in register)

    Steps
    -----
    1. Load 5 input arrays in reverse time order; derive IS ratios, delta, decay
       entirely in registers.
    2. Backward associative scan  →  value_delta[t] = Σ_{s≥t} prod_{t..s-1}(c_s) δ_s
    3. targets[t] = value_delta[t] + V(s_t)   (one store pass)
    4. Re-load targets[t+1] (debug_barrier ensures Step 3 stores are visible),
       compute advantages, store.

    Args:
        log_pi_target_ptr:   Log probs under target policy, [num_envs, seq_len].
        log_pi_behavior_ptr: Log probs under behavior policy, same shape.
        values_ptr:          V(s_t), same shape.  V(s_{t+1}) loaded as values[t+1].
        rewards_ptr:         Per-step rewards, same shape.
        dones_ptr:           Episode termination flags (1.0=done), same shape.
        targets_ptr:         Output V-Trace targets, same shape.
        advantages_ptr:      Output V-Trace advantages, same shape.
        bootstrap_ptr:       Boundary value Delta[T] per env, [num_envs].
                             Also used as V(s_T) for the last-step v_next.
        seq_len:             Number of timesteps (runtime value).
        stride_env:          Row stride in elements.
        gamma:               Discount factor (runtime value).
        rho_bar:             IS ratio clip for delta (runtime value).
        c_bar:               IS ratio clip for decay (runtime value).
        BLOCK_SIZE:          Must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs          # offs=0 → last real timestep T-1
    mask = offs < seq_len

    # --- Step 1: load inputs in reverse time order ---
    log_pi_t = tl.load(log_pi_target_ptr   + base + rev, mask=mask, other=0.0)
    log_pi_b = tl.load(log_pi_behavior_ptr + base + rev, mask=mask, other=0.0)
    v        = tl.load(values_ptr          + base + rev, mask=mask, other=0.0)
    r        = tl.load(rewards_ptr         + base + rev, mask=mask, other=0.0)
    done     = tl.load(dones_ptr           + base + rev, mask=mask, other=1.0)

    # v_next[t] = values[t+1]; boundary (offs==0, t=T-1) uses bootstrap as V(s_T).
    bootstrap  = tl.load(bootstrap_ptr + env_idx)
    v_next_raw = tl.load(values_ptr + base + rev + 1,
                         mask=mask & (offs > 0), other=0.0)
    v_next = tl.where(offs == 0, bootstrap, v_next_raw)

    is_ratio = tl.exp(log_pi_t - log_pi_b)
    rho      = tl.minimum(is_ratio, rho_bar)
    c        = tl.minimum(is_ratio, c_bar)
    not_done = 1.0 - done

    delta = rho * (r + gamma * v_next * not_done - v)
    decay = gamma * c * not_done

    # --- Step 2: backward scan ---
    delta_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)
    value_delta = delta_local + decay_prod * bootstrap

    # --- Step 3: targets ---
    target = value_delta + v
    tl.store(targets_ptr + base + rev, target, mask=mask)

    # --- Step 4: advantages ---
    # adv[t] = rho[t] * (r[t] + gamma*(1-done[t])*next_target[t] - V(s_t))
    # next_target[t] = targets[t+1]  for t < T-1  (offs > 0: load targets[rev+1])
    #                = bootstrap      for t = T-1  (offs = 0: already in register)
    tl.debug_barrier()
    next_target_raw = tl.load(targets_ptr + base + rev + 1,
                              mask=mask & (offs > 0), other=0.0)
    next_target = tl.where(offs == 0, bootstrap, next_target_raw)

    advantage = rho * (r + gamma * next_target * not_done - v)
    tl.store(advantages_ptr + base + rev, advantage, mask=mask)
