"""
Simple ReLU (linear) muscle model.

A simplified muscle where force is proportional to activation.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from motornet_jax.muscle.base import activation_ode, clip_activation
from motornet_jax.types import MuscleState, GeometryState


class ReluMuscleParams(NamedTuple):
    """Parameters for ReLU muscle model.

    Attributes:
        max_isometric_force: Maximum force per muscle. Shape: (n_muscles,)
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
        min_activation: Minimum activation level.
    """
    max_isometric_force: jnp.ndarray
    tau_activation: float = 0.015
    tau_deactivation: float = 0.05
    min_activation: float = 0.001


class ReluMuscle:
    """Simple linear muscle model.

    Force output is simply: F = activation * max_isometric_force

    This is useful for testing and as a baseline.
    """

    def __init__(
        self,
        max_isometric_force: jnp.ndarray,
        tau_activation: float = 0.015,
        tau_deactivation: float = 0.05,
        min_activation: float = 0.001,
    ):
        """Initialize ReLU muscle.

        Args:
            max_isometric_force: Maximum force per muscle. Shape: (n_muscles,)
            tau_activation: Activation time constant (s).
            tau_deactivation: Deactivation time constant (s).
            min_activation: Minimum activation level.
        """
        self.n_muscles = max_isometric_force.shape[0]
        self.params = ReluMuscleParams(
            max_isometric_force=jnp.asarray(max_isometric_force),
            tau_activation=tau_activation,
            tau_deactivation=tau_deactivation,
            min_activation=min_activation,
        )

    def get_params(self) -> ReluMuscleParams:
        """Return parameters for JIT-compiled functions."""
        return self.params

    @staticmethod
    def get_initial_state(
        batch_size: int,
        geometry_state: GeometryState,
        params: ReluMuscleParams,
    ) -> MuscleState:
        """Get initial muscle state from geometry.

        Args:
            batch_size: Batch size.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            Initial muscle state.
        """
        n_muscles = params.max_isometric_force.shape[0]
        return MuscleState(
            activation=jnp.ones((batch_size, n_muscles)) * params.min_activation,
            fiber_length=geometry_state.musculotendon_length,
            fiber_velocity=geometry_state.musculotendon_velocity,
        )

    @staticmethod
    def ode(
        excitation: jnp.ndarray,
        muscle_state: MuscleState,
        params: ReluMuscleParams,
    ) -> jnp.ndarray:
        """Compute muscle state derivatives.

        Args:
            excitation: Neural excitation [0, 1]. Shape: (batch, n_muscles)
            muscle_state: Current muscle state.
            params: Muscle parameters.

        Returns:
            d_activation/dt. Shape: (batch, n_muscles)
        """
        return activation_ode(
            excitation,
            muscle_state.activation,
            params.tau_activation,
            params.tau_deactivation,
            params.min_activation,
        )

    @staticmethod
    def integrate(
        dt: float,
        d_activation: jnp.ndarray,
        muscle_state: MuscleState,
        geometry_state: GeometryState,
        params: ReluMuscleParams,
    ) -> MuscleState:
        """Integrate muscle state forward in time.

        Args:
            dt: Time step.
            d_activation: Activation derivative. Shape: (batch, n_muscles)
            muscle_state: Current muscle state.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            New muscle state.
        """
        new_activation = muscle_state.activation + d_activation * dt
        new_activation = clip_activation(new_activation, params.min_activation)

        return MuscleState(
            activation=new_activation,
            fiber_length=geometry_state.musculotendon_length,
            fiber_velocity=geometry_state.musculotendon_velocity,
        )

    @staticmethod
    def compute_force(
        muscle_state: MuscleState,
        params: ReluMuscleParams,
    ) -> jnp.ndarray:
        """Compute muscle force.

        Args:
            muscle_state: Current muscle state.
            params: Muscle parameters.

        Returns:
            Muscle forces. Shape: (batch, n_muscles)
        """
        return muscle_state.activation * params.max_isometric_force
