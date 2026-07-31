import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def retrace_fused_kernel(
    action_probs_target_ptr,
    action_probs_behavior_ptr,
    q_values_ptr,
    next_q_values_all_ptr,
    actions_ptr,
    rewards_ptr,
    truncated_ptr,
    terminated_ptr,
    out_ptr,
    advantages_ptr,
    pi_at_scratch_ptr,
    seq_len,
    num_actions,
    stride_env,
    stride_env_3d,
    gamma,
    lambda_,
    c_bar,
    rho_bar,
    BLOCK_SIZE:   tl.constexpr,
    ACTION_BLOCK: tl.constexpr,
):
    """
    Fully-fused Retrace(λ) kernel: computes Q-value targets and advantages in
    a single program per environment.

    Eliminates all intermediate tensors (expected_next_q, pi_a, a, b, c_next)
    by computing E_π[Q(s_{t+1},a)] and the IS ratios in registers per timestep,
    then running the backward associative scan before writing the final
    Q-value targets and advantages to HBM.

    Indexing convention
    -------------------
    offs = 0, 1, …, BLOCK_SIZE-1   (block position in reversed order)
    rev  = seq_len - 1 - offs       (real array index; offs=0 → t=T-1)

    For 2D tensors [num_envs, seq_len] (contiguous):
        base_2d  = env_idx * stride_env
        ptr      = base_2d + rev

    For 3D tensors [num_envs, seq_len, num_actions] (contiguous):
        base_3d  = env_idx * stride_env_3d    where stride_env_3d = seq_len * num_actions
        row_base = base_3d + rev * num_actions
        element  = row_base + a               for action a in [0, num_actions)

    Decay convention
    ----------------
    β[t] = γ · c[t+1] · (1 - done[t])

    c[t+1] at reversed position offs is the IS ratio clip at t+1, which lives at
    real array index rev+1 (since rev = seq_len-1-offs, rev+1 = t+1).
    For offs=0 (t=T-1), c[T] is out-of-bounds → c_next=0.

    c[t+1] reuses pi_at_t = π(a_t|s_t), the per-timestep action-probability
    already computed in registers for rho at every block position, instead of
    re-reading action_probs_target[t+1, :] (the full 3D tensor) a second time.
    pi_at_t is stored to a small [num_envs, seq_len] scratch buffer and
    reloaded at the shifted real index t+1 (same store->debug_barrier->reload
    idiom used below for next_q_ret) -- this is exactly pi(a_{t+1}|s_{t+1})
    since the scratch buffer is indexed by real timestep, not block position.
    Previously this loaded the full [BLOCK_SIZE, ACTION_BLOCK] 3D tensor a
    second time (overlapping the window already loaded for pi_all), which
    forced both copies to be simultaneously register-resident and was the
    direct cause of spilling to local memory at long seq_len (measured 128
    regs/thread, 132 spills/thread, 25% occupancy at seq_len=4096). Only one
    [BLOCK_SIZE, ACTION_BLOCK] tensor (pi_all) needs to be live now.

    Advantage computation
    ---------------------
    Steps:
    1. Load inputs; derive IS ratios (rho, c_next), delta (u), decay (v) in registers.
    2. Backward associative scan → Q-value deltas Δ[t].
    3. targets[t] = Δ[t] + Q(s_t, a_t); store to HBM.
    4. Re-load targets[t+1] (debug_barrier); compute and store advantages.

    A[t] = ρ_t · (r_t + γ · Q_ret[t+1] · (1 - terminated[t]) - Q(s_t, a_t))

    For t=T-1, Q_ret[T] is out-of-bounds; we use 0.0 (no continuation target
    beyond the window -- the one-step bootstrap is already inside δ[t] via
    next_q_values_all, so the advantage correctly reflects only the immediate
    TD correction).

    Args:
        action_probs_target_ptr:   [num_envs, seq_len, num_actions], float32.
        action_probs_behavior_ptr: [num_envs, seq_len], float32.
        q_values_ptr:              [num_envs, seq_len], float32.
        next_q_values_all_ptr:     [num_envs, seq_len, num_actions], float32.
        actions_ptr:               [num_envs, seq_len], int64.
        rewards_ptr:               [num_envs, seq_len], float32.
        dones_ptr:                 [num_envs, seq_len], float32.
        terminated_ptr:            [num_envs, seq_len], float32.
        out_ptr:                   [num_envs, seq_len], float32.  Q-value targets output.
        advantages_ptr:            [num_envs, seq_len], float32.  Advantages output.
        seq_len:                   Number of timesteps.
        num_actions:               Number of discrete actions (runtime, for masking).
        stride_env:                Row stride for 2D tensors (seq_len when contiguous).
        stride_env_3d:             Row stride for 3D tensors (seq_len * num_actions).
        gamma:                     Discount factor.
        lambda_:                   Trace decay parameter.
        c_bar:                     IS ratio clip for trace weights.
        rho_bar:                   IS ratio clip for advantage scaling.
        BLOCK_SIZE:                Power-of-2 >= seq_len (constexpr).
        ACTION_BLOCK:              Power-of-2 >= num_actions (constexpr).
    """
    env_idx = tl.program_id(0)

    base_2d = env_idx * stride_env
    base_3d = env_idx * stride_env_3d

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs    # real array index: offs=0 → t=T-1
    mask = offs < seq_len

    # --- Load 2D inputs in reverse time order ---
    q_t       = tl.load(q_values_ptr              + base_2d + rev, mask=mask, other=0.0)
    r         = tl.load(rewards_ptr               + base_2d + rev, mask=mask, other=0.0)
    truncated = tl.load(truncated_ptr             + base_2d + rev, mask=mask, other=0.0)
    term      = tl.load(terminated_ptr            + base_2d + rev, mask=mask, other=1.0)
    # done[t] = terminated[t] | truncated[t]. Computed in-register from the two
    # raw flags instead of taking a precomputed `dones` tensor -- the caller no
    # longer needs a separate (terminateds+truncateds).clamp(max=1.0) kernel
    # launch before this one (measured at ~23% of this op's total time).
    done = tl.minimum(term + truncated, 1.0)

    # --- Current-step IS ratio for rho (advantage scaling) ---
    a_offs = tl.arange(0, ACTION_BLOCK)          # [ACTION_BLOCK]
    seq_mask    = mask[:, None]                   # [BLOCK_SIZE, 1]
    action_mask = a_offs[None, :] < num_actions   # [1, ACTION_BLOCK]
    full_mask   = seq_mask & action_mask          # [BLOCK_SIZE, ACTION_BLOCK]

    # Row base in 3D tensor for each env/timestep: base_3d + rev[i] * num_actions
    row_base = base_3d + rev[:, None] * num_actions + a_offs[None, :]  # [BLOCK_SIZE, ACTION_BLOCK]

    # action_probs_target[env, t, :] and next_q_values_all[env, t, :]
    pi_all = tl.load(action_probs_target_ptr + row_base, mask=full_mask, other=0.0)
    nq_all = tl.load(next_q_values_all_ptr   + row_base, mask=full_mask, other=0.0)

    # E_π[Q_next] = Σ_a π(a) · Q(s_{t+1}, a)
    expected_next_q = tl.sum(pi_all * nq_all, axis=1)   # [BLOCK_SIZE]

    # π(a_t | s_t) for rho: load current-step action and behavior prob
    action_t = tl.load(actions_ptr               + base_2d + rev, mask=mask, other=0)
    mu_t     = tl.load(action_probs_behavior_ptr + base_2d + rev, mask=mask, other=1.0)
    pi_at_t  = tl.sum(
        pi_all * (a_offs[None, :] == action_t[:, None]).to(tl.float32),
        axis=1,
    )
    is_ratio_t = pi_at_t / mu_t
    rho = tl.minimum(is_ratio_t, rho_bar)

    # --- TD error u[t] ---
    not_term = 1.0 - term
    not_done = 1.0 - done
    u = r + gamma * expected_next_q * not_term - q_t

    # --- c[t+1]: shift pi_at_t by one real timestep instead of re-reading the 3D tensor ---
    # Indexing: rev = seq_len-1-offs = real array index of t.
    # t+1 lives at real array index t+1 = (seq_len-1-offs)+1 = rev+1.
    # For offs=0 (t=T-1): c[T] is undefined -> c_next=0 (no trace beyond trajectory end).
    # For offs>0 (t<T-1): t+1 is at real array index rev+1, which is valid.
    tl.store(pi_at_scratch_ptr + base_2d + rev, pi_at_t, mask=mask)
    tl.debug_barrier()

    next_mask_2d = mask & (offs > 0)
    rev_next     = rev + 1                              # real array index of t+1 (valid for offs>0)

    pi_at_next   = tl.load(pi_at_scratch_ptr         + base_2d + rev_next, mask=next_mask_2d, other=0.0)
    mu_next      = tl.load(action_probs_behavior_ptr + base_2d + rev_next, mask=next_mask_2d, other=1.0)

    is_ratio_next = pi_at_next / mu_next
    c_next = tl.where(offs > 0, lambda_ * tl.minimum(is_ratio_next, c_bar), 0.0)

    # v[t] = γ · c[t+1] · (1 - done[t])
    v = gamma * c_next * not_done

    # --- Step 2: Backward associative scan: Δ[T] = 0 ---
    out_local, _ = tl.associative_scan((u, v), axis=0, combine_fn=_combine)

    # --- Step 3: Q_ret[t] = Q(s_t, a_t) + Δ[t]; store to HBM ---
    q_ret = out_local + q_t
    tl.store(out_ptr + base_2d + rev, q_ret, mask=mask)

    # --- Step 4: advantages ---
    # A[t] = ρ_t · (r_t + γ · Q_ret[t+1] · (1 - terminated[t]) - Q(s_t, a_t))
    # Q_ret[t+1] is at real array index rev+1 (valid for offs>0); 0.0 at t=T-1.
    tl.debug_barrier()
    next_q_ret = tl.load(out_ptr + base_2d + rev_next, mask=next_mask_2d, other=0.0)

    advantage = rho * (r + gamma * next_q_ret * not_term - q_t)
    tl.store(advantages_ptr + base_2d + rev, advantage, mask=mask)
