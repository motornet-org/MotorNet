"""
Rigid tendon Hill-type muscle model.

Based on Kistemaker et al. (2010) with rigid tendon assumption.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from motornet_jax.muscle.base import activation_ode, clip_activation
from motornet_jax.types import MuscleState, GeometryState


class RigidTendonMuscleParams(NamedTuple):
    """Parameters for rigid tendon Hill muscle model.

    Attributes:
        max_isometric_force: Maximum isometric force (N). Shape: (n_muscles,)
        optimal_fiber_length: Optimal muscle fiber length (m). Shape: (n_muscles,)
        tendon_slack_length: Tendon slack length (m). Shape: (n_muscles,)
        normalized_slack_length: Normalized length at which passive forces start.
        max_velocity: Maximum contraction velocity (L0/s). Shape: (n_muscles,)
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
        min_activation: Minimum activation level.

        # Derived parameters (computed at initialization)
        passive_slack_length: Length at which passive forces start. Shape: (n_muscles,)
        k_pe: Passive element stiffness. Shape: (n_muscles,)

        # Force-velocity curve parameters
        q_crit: Critical activation for force-velocity scaling.
        s_as: Asymptotic slope parameter.
        min_flce: Minimum force-length value.
    """
    max_isometric_force: jnp.ndarray
    optimal_fiber_length: jnp.ndarray
    tendon_slack_length: jnp.ndarray
    normalized_slack_length: jnp.ndarray
    max_velocity: jnp.ndarray
    tau_activation: float
    tau_deactivation: float
    min_activation: float

    # Derived parameters
    passive_slack_length: jnp.ndarray
    k_pe: jnp.ndarray

    # Force-velocity parameters
    q_crit: float = 0.3
    s_as: float = 0.001
    min_flce: float = 0.01
    f_iso_n_den: float = 0.66 ** 2


class RigidTendonMuscle:
    """Rigid tendon Hill-type muscle model.

    This implements a Hill-type muscle with:
    - Force-length relationship for contractile element
    - Force-velocity relationship
    - Passive parallel element
    - Rigid (non-compliant) tendon

    Based on Kistemaker et al. (2010).
    """

    def __init__(
        self,
        max_isometric_force: jnp.ndarray,
        optimal_fiber_length: jnp.ndarray,
        tendon_slack_length: jnp.ndarray,
        normalized_slack_length: float = 1.4,
        tau_activation: float = 0.015,
        tau_deactivation: float = 0.05,
        min_activation: float = 0.001,
    ):
        """Initialize rigid tendon muscle.

        Args:
            max_isometric_force: Maximum isometric force per muscle (N).
            optimal_fiber_length: Optimal fiber length per muscle (m).
            tendon_slack_length: Tendon slack length per muscle (m).
            normalized_slack_length: Normalized length for passive force onset.
            tau_activation: Activation time constant (s).
            tau_deactivation: Deactivation time constant (s).
            min_activation: Minimum activation level.
        """
        max_isometric_force = jnp.asarray(max_isometric_force)
        optimal_fiber_length = jnp.asarray(optimal_fiber_length)
        tendon_slack_length = jnp.asarray(tendon_slack_length)

        self.n_muscles = max_isometric_force.shape[0]

        # Ensure all arrays have the same shape
        if optimal_fiber_length.shape[0] != self.n_muscles:
            optimal_fiber_length = jnp.ones(self.n_muscles) * optimal_fiber_length[0]
        if tendon_slack_length.shape[0] != self.n_muscles:
            tendon_slack_length = jnp.ones(self.n_muscles) * tendon_slack_length[0]

        normalized_slack = jnp.ones(self.n_muscles) * normalized_slack_length

        # Derived parameters
        passive_slack_length = normalized_slack * optimal_fiber_length
        k_pe = 1.0 / ((1.66 - normalized_slack) ** 2)
        max_velocity = 10.0 * optimal_fiber_length

        self.params = RigidTendonMuscleParams(
            max_isometric_force=max_isometric_force,
            optimal_fiber_length=optimal_fiber_length,
            tendon_slack_length=tendon_slack_length,
            normalized_slack_length=normalized_slack,
            max_velocity=max_velocity,
            tau_activation=tau_activation,
            tau_deactivation=tau_deactivation,
            min_activation=min_activation,
            passive_slack_length=passive_slack_length,
            k_pe=k_pe,
        )

    def get_params(self) -> RigidTendonMuscleParams:
        """Return parameters for JIT-compiled functions."""
        return self.params

    @staticmethod
    def get_initial_state(
        batch_size: int,
        geometry_state: GeometryState,
        params: RigidTendonMuscleParams,
    ) -> MuscleState:
        """Get initial muscle state from geometry.

        Args:
            batch_size: Batch size (unused, inferred from geometry_state).
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            Initial muscle state.
        """
        # Infer batch_size from geometry state for JIT compatibility
        activation = jnp.ones_like(geometry_state.musculotendon_length) * params.min_activation

        # Compute initial fiber length from musculotendon length
        fiber_length = jnp.clip(
            geometry_state.musculotendon_length - params.tendon_slack_length,
            a_min=0.001,
        )

        return MuscleState(
            activation=activation,
            fiber_length=fiber_length,
            fiber_velocity=geometry_state.musculotendon_velocity,
        )

    @staticmethod
    def ode(
        excitation: jnp.ndarray,
        muscle_state: MuscleState,
        params: RigidTendonMuscleParams,
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
        params: RigidTendonMuscleParams,
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
        # Update activation
        new_activation = muscle_state.activation + d_activation * dt
        new_activation = clip_activation(new_activation, params.min_activation)

        # For rigid tendon, fiber length comes directly from geometry
        fiber_length = jnp.clip(
            geometry_state.musculotendon_length - params.tendon_slack_length,
            a_min=0.001,
        )

        return MuscleState(
            activation=new_activation,
            fiber_length=fiber_length,
            fiber_velocity=geometry_state.musculotendon_velocity,
        )

    @staticmethod
    def compute_force(
        muscle_state: MuscleState,
        geometry_state: GeometryState,
        params: RigidTendonMuscleParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute muscle force using Hill model.

        Args:
            muscle_state: Current muscle state.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            force: Total muscle force. Shape: (batch, n_muscles)
            flpe: Passive force-length. Shape: (batch, n_muscles)
            flce: Active force-length. Shape: (batch, n_muscles)
            fvce: Force-velocity. Shape: (batch, n_muscles)
        """
        activation = muscle_state.activation
        fiber_length = muscle_state.fiber_length
        fiber_velocity = geometry_state.musculotendon_velocity

        # Normalized quantities
        fiber_length_n = fiber_length / params.optimal_fiber_length
        fiber_velocity_n = fiber_velocity / params.max_velocity

        # Muscle strain for passive element
        muscle_strain = jnp.clip(
            (fiber_length - params.passive_slack_length) / params.optimal_fiber_length,
            a_min=0.0,
        )

        # Passive force-length (parallel element)
        flpe = params.k_pe * (muscle_strain ** 2)

        # Active force-length (contractile element)
        # Bell-shaped curve centered at L0
        flce = jnp.clip(
            1.0 + (-fiber_length_n**2 + 2*fiber_length_n - 1) / params.f_iso_n_den,
            a_min=params.min_flce,
        )

        # Force-velocity relationship (Kistemaker model)
        a_rel_st = jnp.where(fiber_length_n > 1.0, 0.41 * flce, 0.41)

        b_rel_st = jnp.where(
            activation < params.q_crit,
            5.2 * (1 - 0.9 * ((activation - params.q_crit) / (5e-3 - params.q_crit))) ** 2,
            5.2,
        )

        # Inverse of slope at isometric point
        f_x_a = flce * activation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        tmp_p_nom = f_x_a * 0.5
        tmp_p_den = params.s_as - dfdvcon0 * 2.0

        p1 = -tmp_p_nom / tmp_p_den
        p2 = (tmp_p_nom ** 2) / tmp_p_den
        p3 = -1.5 * f_x_a

        # Force-velocity curve (different for shortening vs lengthening)
        nom = jnp.where(
            fiber_velocity_n < 0,
            fiber_velocity_n * activation * a_rel_st + f_x_a * b_rel_st,
            -p1 * p3 + p1 * params.s_as * fiber_velocity_n + p2 - p3 * fiber_velocity_n + params.s_as * fiber_velocity_n ** 2,
        )
        den = jnp.where(
            fiber_velocity_n < 0,
            b_rel_st - fiber_velocity_n,
            p1 + fiber_velocity_n,
        )

        active_force = jnp.clip(nom / den, a_min=0.0)

        # Total force
        force = (active_force + flpe) * params.max_isometric_force

        return force, flpe, flce, active_force
