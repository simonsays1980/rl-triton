import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def vtrace_fused_kernel(
    log_pi_target_ptr, log_pi_behavior_ptr,
    values_ptr,
    rewards_ptr, terminateds_ptr, truncateds_ptr,
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

    bootstrap_ptr is [num_envs, seq_len] — nonzero only at truncated steps and at
    the window boundary (t=T-1).  At each step bootstrap[t] is added to v_next_raw
    (which is zero-masked at those positions), so no tl.where is needed.

    Indexing convention
    -------------------
    offs = 0, 1, …, seq_len-1   (block position, loaded in this order)
    rev  = seq_len - 1 - offs   (real-time array index: offs=0 → t=T-1)

    terminated[t]: 1 for true episode ends — zeros the bootstrap inside delta.
    done[t]:       1 for any boundary (terminated OR truncated) — zeros trace decay.

    Steps
    -----
    1. Load inputs in reverse time order; derive IS ratios, delta, decay in registers.
    2. Backward associative scan  →  value_delta[t]
    3. targets[t] = value_delta[t] + V(s_t)
    4. Re-load targets[t+1] (debug_barrier), compute advantages, store.

    Args:
        log_pi_target_ptr:   Log probs under target policy, [num_envs, seq_len].
        log_pi_behavior_ptr: Log probs under behavior policy, same shape.
        values_ptr:          V(s_t), same shape.  V(s_{t+1}) loaded as values[t+1].
        rewards_ptr:         Per-step rewards, same shape.
        terminateds_ptr:     True termination flags (1.0=terminated), same shape.
        truncateds_ptr:      Time-limit truncation flags (1.0=truncated), same shape.
        targets_ptr:         Output V-Trace targets, same shape.
        advantages_ptr:      Output V-Trace advantages, same shape.
        bootstrap_ptr:       True continuation values [num_envs, seq_len], float32.
                             Nonzero only at truncated steps and the window boundary.
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
    log_pi_t   = tl.load(log_pi_target_ptr   + base + rev, mask=mask, other=0.0)
    log_pi_b   = tl.load(log_pi_behavior_ptr + base + rev, mask=mask, other=0.0)
    v          = tl.load(values_ptr          + base + rev, mask=mask, other=0.0)
    r          = tl.load(rewards_ptr         + base + rev, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr     + base + rev, mask=mask, other=1.0)
    truncated  = tl.load(truncateds_ptr      + base + rev, mask=mask, other=0.0)
    bootstrap  = tl.load(bootstrap_ptr       + base + rev, mask=mask, other=0.0)
    done       = tl.minimum(terminated + truncated, 1.0)

    # v_next[t] = values[t+1] for interior non-truncated steps; 0 otherwise.
    # bootstrap[t] carries the true continuation value at truncated/boundary steps.
    v_next_raw = tl.load(values_ptr + base + rev + 1,
                         mask=mask & (offs > 0) & (truncated == 0.0), other=0.0)
    v_next = v_next_raw + bootstrap

    is_ratio       = tl.exp(log_pi_t - log_pi_b)
    rho            = tl.minimum(is_ratio, rho_bar)
    c              = tl.minimum(is_ratio, c_bar)
    not_terminated = 1.0 - terminated
    not_done       = 1.0 - done

    delta = rho * (r + gamma * v_next * not_terminated - v)
    decay = gamma * c * not_done

    # --- Step 2: backward scan ---
    delta_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

    # Scan carry: bootstrap[T-1] is the window-boundary continuation value.
    carry       = tl.sum(tl.where(offs == 0, bootstrap, 0.0))
    value_delta = delta_local + decay_prod * carry

    # --- Step 3: targets ---
    target = value_delta + v
    tl.store(targets_ptr + base + rev, target, mask=mask)

    # --- Step 4: advantages ---
    # next_target[t] = targets[t+1] for t < T-1; bootstrap[T-1] for t = T-1.
    tl.debug_barrier()
    next_target_raw = tl.load(targets_ptr + base + rev + 1,
                              mask=mask & (offs > 0) & (truncated == 0.0), other=0.0)
    next_target = next_target_raw + bootstrap

    advantage = rho * (r + gamma * next_target * not_terminated - v)
    tl.store(advantages_ptr + base + rev, advantage, mask=mask)
