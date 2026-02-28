"""
Loss functions for motor control training.
"""

import jax.numpy as jnp
from jax import jit
from typing import Dict, Optional


@jit
def position_loss(
    fingertip: jnp.ndarray,
    target: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute position error loss.

    Args:
        fingertip: Fingertip positions. Shape: (time, batch, n_dim) or (batch, n_dim)
        target: Target positions. Same shape as fingertip.
        weights: Optional time weights. Shape: (time,) or (time, 1, 1)

    Returns:
        Mean squared position error. Scalar.
    """
    error = fingertip - target
    squared_error = jnp.sum(error ** 2, axis=-1)  # Sum over spatial dimensions

    if weights is not None:
        # Reshape weights for broadcasting if needed
        if weights.ndim == 1 and squared_error.ndim > 1:
            weights = weights.reshape(-1, *([1] * (squared_error.ndim - 1)))
        squared_error = squared_error * weights

    return jnp.mean(squared_error)


@jit
def velocity_loss(
    velocity: jnp.ndarray,
    target_velocity: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute velocity error/penalty loss.

    If target_velocity is None, this is a velocity magnitude penalty.
    If target_velocity is provided, this is velocity tracking error.

    Args:
        velocity: Velocities. Shape: (time, batch, n_dim) or (batch, n_dim)
        target_velocity: Optional target velocities. Same shape as velocity.
        weights: Optional time weights.

    Returns:
        Mean squared velocity error/magnitude. Scalar.
    """
    if target_velocity is not None:
        error = velocity - target_velocity
    else:
        error = velocity

    squared_error = jnp.sum(error ** 2, axis=-1)

    if weights is not None:
        if weights.ndim == 1 and squared_error.ndim > 1:
            weights = weights.reshape(-1, *([1] * (squared_error.ndim - 1)))
        squared_error = squared_error * weights

    return jnp.mean(squared_error)


@jit
def effort_loss(
    actions: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute effort (action magnitude) loss.

    Penalizes large muscle activations.

    Args:
        actions: Muscle activations. Shape: (time, batch, n_muscles) or (batch, n_muscles)
        weights: Optional time weights.

    Returns:
        Mean squared action magnitude. Scalar.
    """
    squared_action = jnp.sum(actions ** 2, axis=-1)

    if weights is not None:
        if weights.ndim == 1 and squared_action.ndim > 1:
            weights = weights.reshape(-1, *([1] * (squared_action.ndim - 1)))
        squared_action = squared_action * weights

    return jnp.mean(squared_action)


@jit
def smoothness_loss(
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute action smoothness loss.

    Penalizes rapid changes in muscle activation.

    Args:
        actions: Muscle activations. Shape: (time, batch, n_muscles)

    Returns:
        Mean squared action derivative. Scalar.
    """
    # Compute temporal derivative
    action_diff = actions[1:] - actions[:-1]
    squared_diff = jnp.sum(action_diff ** 2, axis=-1)
    return jnp.mean(squared_diff)


@jit
def jerk_loss(
    positions: jnp.ndarray,
    dt: float = 0.01,
) -> jnp.ndarray:
    """Compute jerk (third derivative of position) loss.

    Encourages smooth, minimum-jerk-like movements.

    Args:
        positions: Positions over time. Shape: (time, batch, n_dim)
        dt: Timestep size.

    Returns:
        Mean squared jerk. Scalar.
    """
    # Compute velocity (first derivative)
    velocity = (positions[1:] - positions[:-1]) / dt

    # Compute acceleration (second derivative)
    acceleration = (velocity[1:] - velocity[:-1]) / dt

    # Compute jerk (third derivative)
    jerk = (acceleration[1:] - acceleration[:-1]) / dt

    squared_jerk = jnp.sum(jerk ** 2, axis=-1)
    return jnp.mean(squared_jerk)


def combined_loss(
    trajectory: Dict[str, jnp.ndarray],
    target: jnp.ndarray,
    config: Dict[str, float],
) -> jnp.ndarray:
    """Compute combined loss from trajectory.

    Args:
        trajectory: Dict with keys 'fingertip', 'action', 'velocity', etc.
        target: Target position. Shape: (batch, n_dim) or (time, batch, n_dim)
        config: Dict with loss weights:
            - position_weight: Weight for position loss.
            - velocity_weight: Weight for velocity loss (penalty).
            - effort_weight: Weight for effort loss.
            - smoothness_weight: Weight for action smoothness.

    Returns:
        Total weighted loss. Scalar.
    """
    total_loss = 0.0

    # Position loss
    pos_weight = config.get('position_weight', 1.0)
    if pos_weight > 0:
        fingertip = trajectory['fingertip']
        # Broadcast target if needed
        if target.ndim < fingertip.ndim:
            target = jnp.broadcast_to(target, fingertip.shape)
        total_loss = total_loss + pos_weight * position_loss(fingertip, target)

    # Velocity penalty (for smooth stopping)
    vel_weight = config.get('velocity_weight', 0.0)
    if vel_weight > 0 and 'velocity' in trajectory:
        total_loss = total_loss + vel_weight * velocity_loss(trajectory['velocity'])

    # Effort loss
    effort_weight = config.get('effort_weight', 0.01)
    if effort_weight > 0:
        total_loss = total_loss + effort_weight * effort_loss(trajectory['action'])

    # Smoothness loss
    smooth_weight = config.get('smoothness_weight', 0.0)
    if smooth_weight > 0:
        total_loss = total_loss + smooth_weight * smoothness_loss(trajectory['action'])

    return total_loss


def reach_loss(
    fingertip: jnp.ndarray,
    target: jnp.ndarray,
    actions: jnp.ndarray,
    final_velocity: Optional[jnp.ndarray] = None,
    position_weight: float = 1.0,
    effort_weight: float = 0.01,
    final_velocity_weight: float = 0.1,
) -> jnp.ndarray:
    """Loss function for reaching tasks.

    Emphasizes:
    - Reaching the target
    - Stopping at the target (low final velocity)
    - Efficient movements (low effort)

    Args:
        fingertip: Fingertip trajectory. Shape: (time, batch, n_dim)
        target: Target position. Shape: (batch, n_dim)
        actions: Actions trajectory. Shape: (time, batch, n_muscles)
        final_velocity: Final fingertip velocity. Shape: (batch, n_dim)
        position_weight: Weight for position error.
        effort_weight: Weight for effort penalty.
        final_velocity_weight: Weight for final velocity penalty.

    Returns:
        Total loss. Scalar.
    """
    # Final position error (most important for reaching)
    final_pos = fingertip[-1]
    pos_loss = position_loss(final_pos, target)

    # Effort penalty
    eff_loss = effort_loss(actions)

    # Final velocity penalty (stop at target)
    vel_loss = 0.0
    if final_velocity is not None and final_velocity_weight > 0:
        vel_loss = velocity_loss(final_velocity)

    return (
        position_weight * pos_loss +
        effort_weight * eff_loss +
        final_velocity_weight * vel_loss
    )
