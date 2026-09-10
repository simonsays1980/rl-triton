"""Top-level (picklable-by-reference, required for multiprocessing spawn)
collector subprocess function for the Part B async pipeline benchmark.

Mirrors uni_rl.algos.appo.worker.appo_collector_fn's structure (weight-sync
pull, per-step MLP inference + env.step, ring-buffer write, EMA collector
timing) but uses BenchActor (this study's lightweight duck-typed stand-in,
not the real rsl_rl.models.MLPModel which needs full distribution_cfg/
resolve_callable config plumbing this benchmark deliberately avoids -- see
step4_appo_e2e.py's run notes) and the SyntheticEnv from synthetic_env.py.

unilab_rl's own worker.py, runner.py, staging.py are NOT imported/modified
here except for the real IPC primitives (RolloutRingBuffer, SharedWeightSync)
-- those are used directly, unmodified, so the actual shared-memory
contention/synchronization mechanics under test are the real ones.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def collector_fn(
    stop_event,
    env_factory,
    num_envs: int,
    steps_per_env: int,
    obs_dim: int,
    action_dim: int,
    ring_buffer_shm_names: dict,
    write_ptr,
    read_ptr,
    actor_weight_sync_name: str,
    actor_weight_param_shapes: dict,
    device: str,
    metrics_queue,
    max_rollouts: int,
):
    import numpy as np
    import torch

    from step4_appo_e2e import BenchActor
    from uni_rl.ipc.rollout_ring_buffer import RolloutRingBuffer
    from uni_rl.ipc.weight_sync import SharedWeightSync

    torch.manual_seed(123)
    ring_buffer = RolloutRingBuffer(
        num_envs=num_envs, num_steps=steps_per_env, obs_dim=obs_dim, action_dim=action_dim,
        critic_dim=0, create=False, shm_name_prefix=ring_buffer_shm_names,
    )
    ring_buffer.attach_sync_primitives(write_ptr, read_ptr)
    actor_weight_sync = SharedWeightSync(actor_weight_param_shapes, create=False, shm_name=actor_weight_sync_name)

    actor = BenchActor(obs_dim, action_dim).to(device)
    actor.eval()
    actor_sd = dict(actor.state_dict())
    actor_weight_sync.read_weights_into(actor_sd)
    actor.load_state_dict(actor_sd)
    local_actor_weight_version = actor_weight_sync.version

    env = env_factory(num_envs, {"obs_dim": obs_dim, "action_dim": action_dim, "episode_len": steps_per_env})
    env_indices = np.arange(num_envs, dtype=np.int32)
    obs_out, _ = env.reset(env_indices)
    obs_np = obs_out["obs"].astype(np.float32, copy=False)

    obs_torch = torch.zeros((num_envs, obs_dim), dtype=torch.float32, device=device)
    obs_td = {"policy": obs_torch}

    rollouts_done = 0
    total_steps = 0
    _EMA = 0.1
    ema_mlp_infer_ms = 0.0
    ema_env_step_ms = 0.0
    ema_rollout_ms = 0.0

    try:
        while not stop_event.is_set() and rollouts_done < max_rollouts:
            t_rollout_start = time.perf_counter()
            if actor_weight_sync.version > local_actor_weight_version:
                actor_sd = dict(actor.state_dict())
                local_actor_weight_version = actor_weight_sync.read_weights_into(actor_sd)
                actor.load_state_dict(actor_sd)

            write_buf = ring_buffer.write_buffer
            for step in range(steps_per_env):
                t_mlp = time.perf_counter()
                with torch.no_grad():
                    obs_torch.copy_(torch.from_numpy(obs_np))
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    actions_np = actions_torch.cpu().numpy()
                ema_mlp_infer_ms = (1 - _EMA) * ema_mlp_infer_ms + _EMA * ((time.perf_counter() - t_mlp) * 1000)

                write_buf["obs"][:, step, :] = obs_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["log_probs"][:, step] = log_probs_torch.cpu().numpy().ravel()

                t_env = time.perf_counter()
                state = env.step(actions_np)
                ema_env_step_ms = (1 - _EMA) * ema_env_step_ms + _EMA * ((time.perf_counter() - t_env) * 1000)

                reward_raw = state.reward.astype(np.float32, copy=False).ravel()
                truncated_raw = state.truncated.astype(np.float32, copy=False).ravel()
                combined_done_raw = (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()

                write_buf["rewards"][:, step] = reward_raw
                write_buf["dones"][:, step] = combined_done_raw
                write_buf["truncated"][:, step] = truncated_raw

                total_steps += num_envs
                obs_np = state.obs["obs"].astype(np.float32, copy=False)

            write_buf["last_obs"][:] = obs_np
            ring_buffer.signal_write_done()
            rollouts_done += 1
            ema_rollout_ms = (1 - _EMA) * ema_rollout_ms + _EMA * ((time.perf_counter() - t_rollout_start) * 1000)

            try:
                metrics_queue.put_nowait({
                    "total_steps": total_steps, "rollouts_done": rollouts_done,
                    "wall_time": time.perf_counter(),
                    "collector_timing_ms": {"rollout_ms": ema_rollout_ms, "mlp_infer_ms": ema_mlp_infer_ms,
                                             "env_step_ms": ema_env_step_ms},
                })
            except Exception:
                pass
    finally:
        ring_buffer.close()
        actor_weight_sync.close()
        env.close()
