import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def vtrace_fused_kernel(
    log_pi_target_ptr, log_pi_behavior_ptr,
    values_ptr, next_values_ptr,
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

    Replaces eight separate PyTorch elementwise kernels plus the scan kernel,
    eliminating all intermediate tensor allocations and kernel launch overhead.

    Compared to the unfused path in compute_vtrace_triton, this kernel does all
    work for one environment in a single GPU thread block:
      1. Load 6 input arrays, compute IS ratios, deltas, decays  (in-register)
      2. Run the backward associative scan                        (in-register)
      3. Compute and store vtrace_targets                         (one store pass)
      4. Load targets shifted by +1 in time, compute advantages   (one load + store)

    The shift for next_vtrace_targets is done by re-loading targets at
    rev_offsets - 1 (one position earlier in reversed time = one step later in
    real time).  The last timestep (rev_offset == 0) uses next_values[:, -1].

    Args:
        log_pi_target_ptr:   Log probs under target policy, [num_envs, seq_len].
        log_pi_behavior_ptr: Log probs under behavior policy, same shape.
        values_ptr:          V(s_t), same shape.
        next_values_ptr:     V(s_{t+1}), same shape.
        rewards_ptr:         Per-step rewards, same shape.
        dones_ptr:           Episode termination flags (1.0=done), same shape.
        targets_ptr:         Output V-Trace targets, same shape.
        advantages_ptr:      Output V-Trace advantages, same shape.
        bootstrap_ptr:       Boundary value Delta[T] per env, [num_envs].
        seq_len:             Number of timesteps (runtime value).
        stride_env:          Row stride in elements.
        gamma:               Discount factor (compile-time constant).
        rho_bar:             IS ratio clip for delta (compile-time constant).
        c_bar:               IS ratio clip for decay (compile-time constant).
        BLOCK_SIZE:          Must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base = env_idx * stride_env

    offsets     = tl.arange(0, BLOCK_SIZE)
    rev_offsets = seq_len - 1 - offsets      # maps block position -> real time index
    mask        = offsets < seq_len

    # --- Step 1: load inputs in reverse time order ---
    log_pi_t = tl.load(log_pi_target_ptr   + base + rev_offsets, mask=mask, other=0.0)
    log_pi_b = tl.load(log_pi_behavior_ptr + base + rev_offsets, mask=mask, other=0.0)
    v        = tl.load(values_ptr          + base + rev_offsets, mask=mask, other=0.0)
    v_next   = tl.load(next_values_ptr     + base + rev_offsets, mask=mask, other=0.0)
    r        = tl.load(rewards_ptr         + base + rev_offsets, mask=mask, other=0.0)
    done     = tl.load(dones_ptr           + base + rev_offsets, mask=mask, other=1.0)

    is_ratio = tl.exp(log_pi_t - log_pi_b)
    rho      = tl.minimum(is_ratio, rho_bar)
    c        = tl.minimum(is_ratio, c_bar)
    not_done = 1.0 - done

    delta = rho * (r + gamma * v_next * not_done - v)
    decay = gamma * c * not_done

    # --- Step 2: backward scan ---
    delta_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)
    bootstrap   = tl.load(bootstrap_ptr + env_idx)
    value_delta = delta_local + decay_prod * bootstrap

    # --- Step 3: targets ---
    target = value_delta + v
    tl.store(targets_ptr + base + rev_offsets, target, mask=mask)

    # --- Step 4: advantages ---
    # next_vtrace_target[t] = target[t+1].
    # target[t] is stored at targets_ptr + base + t = targets_ptr + base + rev_offsets.
    # So target[t+1] is at targets_ptr + base + rev_offsets + 1.
    # Boundary: offsets==0 (time t=seq_len-1) uses next_values[:, seq_len-1].
    # debug_barrier() ensures the step-3 stores are visible before we load them back.
    tl.debug_barrier()
    next_v_last = tl.load(next_values_ptr + base + seq_len - 1)
    next_target = tl.load(
        targets_ptr + base + rev_offsets + 1,
        mask=(rev_offsets < seq_len - 1) & mask,
        other=next_v_last,
    )

    advantage = rho * (r + gamma * next_target * not_done - v)
    tl.store(advantages_ptr + base + rev_offsets, advantage, mask=mask)
