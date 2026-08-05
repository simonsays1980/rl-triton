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
    HAS_TRUNCATIONS: tl.constexpr,
    HAS_BOOTSTRAP: tl.constexpr,
):
    """
    Fully-fused V-Trace kernel: computes targets and advantages in a single program.

    HAS_TRUNCATIONS=False (common case -- benchmark path):
      truncateds_ptr is constexpr None: the argument is completely compiled out,
      no register allocated, no HBM read.  bootstrap_ptr is [num_envs] -- one
      scalar per env for the window boundary.
      Full-width HBM reads: log_pi_target, log_pi_behavior, values,
                            rewards, terminateds  (5)

    HAS_TRUNCATIONS=True:
      truncateds_ptr is a real tensor.  bootstrap_ptr is [num_envs, seq_len],
      nonzero only at truncated steps and the window boundary.
      Full-width HBM reads: + truncateds, 2D bootstrap  (7)

    Indexing convention
    -------------------
    offs = 0, 1, …, BLOCK_SIZE-1   (block position, loaded in reversed order)
    rev  = seq_len - 1 - offs       (real-time array index: offs=0 → t=T-1)

    terminated[t]: 1 for true episode ends -- zeros the bootstrap inside delta.
    done[t]:       1 for any boundary -- zeros trace decay.
                   False path: done = terminated.
                   True  path: done = terminated | truncated.

    Steps
    -----
    1. Load inputs in reverse time order; derive IS ratios, delta, decay.
    2. Backward associative scan  →  value_delta[t].
    3. targets[t] = value_delta[t] + V(s_t); store to global memory.
    4. debug_barrier + re-load targets[t+1]; compute and store advantages.

    Why steps 3-4 round-trip through global memory:
      Advantages need targets[t+1], which belongs to the neighboring thread.
      Triton has no direct SRAM API -- the standard pattern is store → debug_barrier
      → load.  The targets data was just written by the same SM, so the read
      typically hits L1/L2 cache, but this is structural latency that is not
      measured here.
      TODO(perf): profile whether this store→reload is the primary reason V-Trace
      trails GAE on the same hardware.  If so, a two-kernel approach (targets,
      then advantages) would trade one kernel launch for eliminating the round-trip
      -- worth benchmarking when widening the gap matters.

    Args:
        log_pi_target_ptr:   Log probs under target policy, [num_envs, seq_len].
        log_pi_behavior_ptr: Log probs under behavior policy, same shape.
        values_ptr:          V(s_t), same shape.  V(s_{t+1}) loaded as values[t+1].
        rewards_ptr:         Per-step rewards, same shape.
        terminateds_ptr:     True termination flags (1.0=terminated), same shape.
        truncateds_ptr:      Time-limit truncation flags, same shape.
                             Constexpr None when HAS_TRUNCATIONS=False -- compiled out.
        targets_ptr:         Output V-Trace targets, same shape.
        advantages_ptr:      Output V-Trace advantages, same shape.
        bootstrap_ptr:       [num_envs, seq_len] when HAS_TRUNCATIONS=True --
                             nonzero only at truncated steps and window boundary.
                             [num_envs] scalar per env when HAS_TRUNCATIONS=False.
        seq_len:             Number of timesteps (runtime value).
        stride_env:          Row stride in elements.
        gamma:               Discount factor (runtime value).
        rho_bar:             IS ratio clip for delta (runtime value).
        c_bar:               IS ratio clip for decay (runtime value).
        BLOCK_SIZE:          Must be >= seq_len and a power of 2 (constexpr).
        HAS_TRUNCATIONS:     Compile-time flag -- False eliminates truncateds_ptr and
                             2D bootstrap_ptr reads entirely.
        HAS_BOOTSTRAP:       Compile-time flag, only meaningful when HAS_TRUNCATIONS=False --
                             False skips the scalar bootstrap_ptr read and uses literal 0.0.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs          # offs=0 → last real timestep T-1
    mask = offs < seq_len

    # --- Step 1: load common inputs ---
    log_pi_t   = tl.load(log_pi_target_ptr   + base + rev, mask=mask, other=0.0)
    log_pi_b   = tl.load(log_pi_behavior_ptr + base + rev, mask=mask, other=0.0)
    v          = tl.load(values_ptr          + base + rev, mask=mask, other=0.0)
    r          = tl.load(rewards_ptr         + base + rev, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr     + base + rev, mask=mask, other=1.0)

    is_ratio       = tl.exp(log_pi_t - log_pi_b)
    rho            = tl.minimum(is_ratio, rho_bar)
    c              = tl.minimum(is_ratio, c_bar)
    not_terminated = 1.0 - terminated

    if HAS_TRUNCATIONS:
        # 7 full-width reads.  truncateds_ptr and 2D bootstrap_ptr are live.
        truncated  = tl.load(truncateds_ptr + base + rev, mask=mask, other=0.0)
        bootstrap  = tl.load(bootstrap_ptr  + base + rev, mask=mask, other=0.0)
        done       = tl.minimum(terminated + truncated, 1.0)
        not_done   = 1.0 - done

        # v_next[t] = values[t+1] at non-truncated interior steps; 0 otherwise.
        # bootstrap[t] = V(s_{t+1}^true) at truncated steps and boundary (offs==0).
        # Exactly one of v_next_raw and bootstrap[t] is nonzero at each position,
        # so their sum gives the correct continuation value.
        v_next_raw = tl.load(values_ptr + base + rev + 1,
                             mask=mask & (offs > 0) & (truncated == 0.0), other=0.0)
        v_next = v_next_raw + bootstrap

        delta = rho * (r + gamma * v_next * not_terminated - v)
        decay = gamma * c * not_done   # beta[t], the scan decay coefficient

        # --- Step 2: backward scan ---
        # Additive boundary carry Delta[T] = 0: bootstrap[T-1] already entered
        # delta[T-1] above (via v_next, weight 1). Seeding the scan's boundary
        # carry with it again here would double-count it -- same class of bug
        # as GAE's window-boundary carry (see kernels/gae.py's module
        # docstring). beta (decay) itself is unaffected by this fix.
        value_delta, _ = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

        # --- Step 3: targets ---
        target = value_delta + v
        tl.store(targets_ptr + base + rev, target, mask=mask)

        # --- Step 4: advantages ---
        # next_target[t] is one of three mutually-exclusive cases:
        #   - Normal interior step (offs>0, truncated==0): targets[t+1]
        #   - Truncated interior step (offs>0, truncated==1): bootstrap[t]
        #   - Boundary (offs==0, t=T-1): bootstrap[T-1]
        # The additive trick: next_target_raw is zero-masked at truncated/boundary
        # positions; bootstrap[t] is zero at normal interior steps.  Their sum
        # selects the correct value at each position.
        # Correctness confirmed by test_vtrace_fused_two_interior_truncations against
        # _ref_vtrace_sequential (pure step-by-step Python loop).
        tl.debug_barrier()
        next_target_raw = tl.load(targets_ptr + base + rev + 1,
                                  mask=mask & (offs > 0) & (truncated == 0.0), other=0.0)
        next_target = next_target_raw + bootstrap

        advantage = rho * (r + gamma * next_target * not_terminated - v)
        tl.store(advantages_ptr + base + rev, advantage, mask=mask)
    else:
        # 5 full-width reads.  truncateds_ptr is constexpr None -- compiled out.
        # bootstrap_ptr is [num_envs]: one scalar per env for the window boundary.
        # HAS_BOOTSTRAP=False (no last_value/bootstrap_values given): skip the
        # scalar load entirely and use literal 0.0 -- saves the wrapper a
        # torch.zeros(num_envs) allocation + launch (same pattern as GAE).
        not_done  = not_terminated          # done[t] = terminated[t]
        if HAS_BOOTSTRAP:
            bootstrap = tl.load(bootstrap_ptr + env_idx)
        else:
            bootstrap = 0.0

        # v_next[t] = values[t+1] for interior steps; scalar bootstrap at boundary.
        v_next_raw = tl.load(values_ptr + base + rev + 1,
                             mask=mask & (offs > 0), other=0.0)
        v_next = tl.where(offs == 0, bootstrap, v_next_raw)

        delta = rho * (r + gamma * v_next * not_terminated - v)
        decay = gamma * c * not_done   # beta[t], the scan decay coefficient

        # --- Step 2: backward scan ---
        # Additive boundary carry Delta[T] = 0: bootstrap already entered
        # delta above (via v_next, weight 1). Seeding the scan's boundary
        # carry with it again here would double-count it -- same class of bug
        # as GAE's window-boundary carry (see kernels/gae.py's module
        # docstring). beta (decay) itself is unaffected by this fix.
        value_delta, _ = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

        # --- Step 3: targets ---
        target = value_delta + v
        tl.store(targets_ptr + base + rev, target, mask=mask)

        # --- Step 4: advantages ---
        # next_target[t] = targets[t+1] for t < T-1; scalar bootstrap at boundary.
        tl.debug_barrier()
        next_target_raw = tl.load(targets_ptr + base + rev + 1,
                                  mask=mask & (offs > 0), other=0.0)
        next_target = tl.where(offs == 0, bootstrap, next_target_raw)

        advantage = rho * (r + gamma * next_target * not_terminated - v)
        tl.store(advantages_ptr + base + rev, advantage, mask=mask)
