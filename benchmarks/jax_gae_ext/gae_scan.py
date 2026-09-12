"""JAX associative-scan GAE implementation, vendored verbatim for use as a
comparison baseline in benchmark_gae_vs_jax_scan.py. Thanks to Sasha
Abramowitz for sharing this code. Not modified beyond this header comment --
see that script's module docstring for the equivalence derivation against
rl-triton's own GAE convention.
"""
from functools import partial

import jax
import jax.numpy as jnp


def backward_affine_scan(increment: jax.Array, coefficient: jax.Array, axis: int = 0) -> jax.Array:
    def compose(later: tuple[jax.Array, jax.Array], earlier: tuple[jax.Array, jax.Array]):
        increment_later, coefficient_later = later
        increment_earlier, coefficient_earlier = earlier
        return (
            increment_earlier + coefficient_earlier * increment_later,
            coefficient_earlier * coefficient_later,
        )

    accumulated, _ = jax.lax.associative_scan(compose, (increment, coefficient), reverse=True, axis=axis)
    return accumulated


def generalised_advantages(
    reward: jax.Array,
    value: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    gamma: float,
    gae_lambda: float,
    axis: int = 0,
) -> jax.Array:
    """GAE over T+1 aligned slots, reading both boundary flags apart.

    The implementation is a reverse cumulative sum and as GAE is associative, it can be done in parallel.

    Args:
        reward: ``(T+1, ...)``, what each slot's action earned.
        value: ``(T+1, ...)``, of each slot's own observation.
        terminated: ``(T+1, ...)`` bool, whether each slot's observation is a Final obs.
        truncated: ``(T+1, ...)`` bool, whether each slot's observation was cut by a time limit.
        gamma: Discount.
        gae_lambda: Bias-variance trade between the one-step residual and the full return.
        axis: Which axis the slots run down. Every other axis is mapped over implicitly, so a
            batch-major caller passes ``axis=1`` rather than wrapping this in a ``vmap``.

    Returns:
        Estimates of length T on ``axis``. The trailing slot is consumed as the bootstrap, never returned.
    """
    leaving = partial(jax.lax.slice_in_dim, start_index=0, limit_index=-1, axis=axis)
    landing = partial(jax.lax.slice_in_dim, start_index=1, limit_index=None, axis=axis)

    bootstrap = jnp.logical_not(landing(terminated)).astype(value.dtype)
    continues = bootstrap * jnp.logical_not(landing(truncated)).astype(value.dtype)

    residual = leaving(reward) + gamma * bootstrap * landing(value) - leaving(value)
    return backward_affine_scan(residual, gamma * gae_lambda * continues, axis)
