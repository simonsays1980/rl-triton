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
    dones_ptr,
    terminated_ptr,
    out_ptr,
    seq_len,
    num_actions,
    stride_env,
    stride_env_3d,
    gamma,
    lambda_,
    c_bar,
    BLOCK_SIZE:   tl.constexpr,
    ACTION_BLOCK: tl.constexpr,
):
    """
    Fully-fused Retrace(λ) kernel: one program per environment.

    Eliminates all intermediate tensors (expected_next_q, pi_a, u, v, c_next)
    by computing E_π[Q(s_{t+1},a)] and the IS ratios in registers per timestep,
    then running the backward associative scan before writing only the final
    Q-value targets to HBM.

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
    v[t] = γ · c[t+1] · (1 - done[t])

    c[t+1] at reversed position offs is the IS ratio clip at t+1, which lives at
    real array index rev+1 (since rev = seq_len-1-offs, rev+1 = t+1).
    For offs=0 (t=T-1), c[T] is out-of-bounds → c_next=0.

    To compute c[t+1] we load action_probs_target[rev+1, :] and action[rev+1] from
    the next real timestep (rev = seq_len-1-offs, so rev+1 is real index t+1).
    This costs an extra read of the 3D probs array per timestep.  The tradeoff:
    we save writing and re-reading the full intermediate u, v, c_next tensors.

    Args:
        action_probs_target_ptr:   [num_envs, seq_len, num_actions], float32.
        action_probs_behavior_ptr: [num_envs, seq_len], float32.
        q_values_ptr:              [num_envs, seq_len], float32.
        next_q_values_all_ptr:     [num_envs, seq_len, num_actions], float32.
        actions_ptr:               [num_envs, seq_len], int64.
        rewards_ptr:               [num_envs, seq_len], float32.
        dones_ptr:                 [num_envs, seq_len], float32.
        terminated_ptr:            [num_envs, seq_len], float32.
        out_ptr:                   [num_envs, seq_len], float32.
        seq_len:                   Number of timesteps.
        num_actions:               Number of discrete actions (runtime, for masking).
        stride_env:                Row stride for 2D tensors (seq_len when contiguous).
        stride_env_3d:             Row stride for 3D tensors (seq_len * num_actions).
        gamma:                     Discount factor.
        lambda_:                   Trace decay parameter.
        c_bar:                     IS ratio clip for trace weights.
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
    q_t    = tl.load(q_values_ptr              + base_2d + rev, mask=mask, other=0.0)
    r      = tl.load(rewards_ptr               + base_2d + rev, mask=mask, other=0.0)
    done   = tl.load(dones_ptr                 + base_2d + rev, mask=mask, other=1.0)
    term   = tl.load(terminated_ptr            + base_2d + rev, mask=mask, other=1.0)

    # --- 3D load setup ---
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

    # --- TD error u[t] ---
    not_term = 1.0 - term
    not_done = 1.0 - done
    u = r + gamma * expected_next_q * not_term - q_t

    # --- c[t+1]: load action_probs_target and action at next real timestep (rev+1) ---
    # Indexing: rev = seq_len-1-offs = real array index of t.
    # t+1 lives at real array index t+1 = (seq_len-1-offs)+1 = rev+1.
    # For offs=0 (t=T-1): c[T] is undefined → c_next=0 (no trace beyond trajectory end).
    # For offs>0 (t<T-1): t+1 is at real array index rev+1, which is valid.
    next_mask_2d = mask & (offs > 0)
    next_mask_3d = full_mask & (offs[:, None] > 0)

    rev_next      = rev + 1                            # real array index of t+1 (valid for offs>0)
    row_base_next = base_3d + rev_next[:, None] * num_actions + a_offs[None, :]

    pi_next_all  = tl.load(action_probs_target_ptr + row_base_next, mask=next_mask_3d, other=0.0)
    action_next  = tl.load(actions_ptr               + base_2d + rev_next, mask=next_mask_2d, other=0)
    mu_next      = tl.load(action_probs_behavior_ptr + base_2d + rev_next, mask=next_mask_2d, other=1.0)

    pi_at_next   = tl.sum(
        pi_next_all * (a_offs[None, :] == action_next[:, None]).to(tl.float32),
        axis=1,
    )
    is_ratio_next = pi_at_next / mu_next
    c_next = tl.where(offs > 0, lambda_ * tl.minimum(is_ratio_next, c_bar), 0.0)

    # v[t] = γ · c[t+1] · (1 - done[t])
    v = gamma * c_next * not_done

    # --- Backward associative scan: Δ[T] = 0 (no separate bootstrap for retrace) ---
    out_local, _ = tl.associative_scan((u, v), axis=0, combine_fn=_combine)

    # Q_ret[t] = Q(s_t, a_t) + Δ[t]
    tl.store(out_ptr + base_2d + rev, out_local + q_t, mask=mask)
