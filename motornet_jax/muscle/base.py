"""
Base muscle model utilities.

Contains shared functions for muscle activation dynamics and force computation.
"""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def activation_ode(
    excitation: jnp.ndarray,
    activation: jnp.ndarray,
    tau_activation: float = 0.015,
    tau_deactivation: float = 0.05,
    min_activation: float = 0.001,
) -> jnp.ndarray:
    """Compute activation dynamics ODE.

    Implements the activation dynamics from Thelen (2003) / MuJoCo muscle model.
    The time constant depends on whether activation is increasing or decreasing.

    Args:
        excitation: Neural excitation signal [0, 1]. Shape: (batch, n_muscles)
        activation: Current activation level [0, 1]. Shape: (batch, n_muscles)
        tau_activation: Time constant for activation (s).
        tau_deactivation: Time constant for deactivation (s).
        min_activation: Minimum activation level.

    Returns:
        d_activation/dt. Shape: (batch, n_muscles)
    """
    # Clip inputs to valid range
    excitation = jnp.clip(excitation, min_activation, 1.0)
    activation = jnp.clip(activation, min_activation, 1.0)

    # Time constant depends on whether we're activating or deactivating
    # and on current activation level (from Thelen 2003)
    tmp = 0.5 + 1.5 * activation
    tau = jnp.where(
        excitation > activation,
        tau_activation * tmp,
        tau_deactivation / tmp,
    )

    return (excitation - activation) / tau


@jit
def clip_activation(
    activation: jnp.ndarray,
    min_activation: float = 0.001,
) -> jnp.ndarray:
    """Clip activation to valid range [min_activation, 1.0].

    Args:
        activation: Activation values. Shape: (batch, n_muscles)
        min_activation: Minimum activation level.

    Returns:
        Clipped activation values.
    """
    return jnp.clip(activation, min_activation, 1.0)
