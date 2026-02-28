"""
Compliant tendon Hill-type muscle model.

Based on Kistemaker et al. (2010) with compliant tendon.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import jit

from motornet_jax.muscle.base import activation_ode, clip_activation
from motornet_jax.types import MuscleState, GeometryState


class CompliantTendonMuscleParams(NamedTuple):
    """Parameters for compliant tendon Hill muscle model.

    Attributes:
        max_isometric_force: Maximum isometric force (N). Shape: (n_muscles,)
        optimal_fiber_length: Optimal muscle fiber length (m). Shape: (n_muscles,)
        tendon_slack_length: Tendon slack length (m). Shape: (n_muscles,)
        normalized_slack_length: Normalized length at which passive forces start.
        max_velocity: Maximum contraction velocity (L0/s). Shape: (n_muscles,)
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
        min_activation: Minimum activation level.

        # Derived parameters
        passive_slack_length: Length at which passive forces start. Shape: (n_muscles,)
        k_pe: Passive element stiffness. Shape: (n_muscles,)
        k_se: Series elastic (tendon) stiffness.

        # Force-velocity parameters
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
    k_se: float = 625.0  # 1 / (0.04 ** 2)

    # Force-velocity parameters
    q_crit: float = 0.3
    s_as: float = 0.001
    min_flce: float = 0.01
    f_iso_n_den: float = 0.66 ** 2


class CompliantTendonMuscle:
    """Compliant tendon Hill-type muscle model.

    This implements a Hill-type muscle with:
    - Force-length relationship for contractile element
    - Force-velocity relationship
    - Passive parallel element
    - Compliant (elastic) tendon

    The muscle fiber length is a state variable that is integrated over time.

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
        min_activation: float = 0.01,
    ):
        """Initialize compliant tendon muscle.

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

        self.params = CompliantTendonMuscleParams(
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

    def get_params(self) -> CompliantTendonMuscleParams:
        """Return parameters for JIT-compiled functions."""
        return self.params

    @staticmethod
    @jit
    def get_initial_state(
        batch_size: int,
        geometry_state: GeometryState,
        params: CompliantTendonMuscleParams,
    ) -> MuscleState:
        """Get initial muscle state from geometry.

        For compliant tendon, we need to find the initial fiber length that
        satisfies force equilibrium between tendon and muscle passive forces.

        Args:
            batch_size: Batch size.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            Initial muscle state.
        """
        n_muscles = params.max_isometric_force.shape[0]
        musculotendon_len = geometry_state.musculotendon_length

        # Compute initial fiber length based on musculotendon length
        # Case 1: MT length < tendon slack length -> minimal muscle length
        # Case 2: MT length < MT slack length -> muscle takes the difference
        # Case 3: MT length > MT slack length -> find equilibrium

        mt_slack = params.passive_slack_length + params.tendon_slack_length

        # Equilibrium solution from Kistemaker model
        # When both tendon and muscle are stretched, find equilibrium
        k_pe = params.k_pe
        k_se = params.k_se
        l0_pe = params.passive_slack_length
        l0_se = params.tendon_slack_length
        l0_ce = params.optimal_fiber_length

        # Analytical solution for equilibrium
        equilibrium_length = (
            k_pe * l0_pe * l0_se**2
            - k_se * l0_ce**2 * musculotendon_len
            + k_se * l0_ce**2 * l0_se
            - l0_ce * l0_se * jnp.sqrt(k_pe * k_se) * (-musculotendon_len + l0_pe + l0_se)
        ) / (k_pe * l0_se**2 - k_se * l0_ce**2)

        fiber_length = jnp.where(
            musculotendon_len < 0,
            0.001 * l0_ce,  # Invalid case
            jnp.where(
                musculotendon_len < l0_se,
                0.001 * l0_ce,  # Tendon not taut
                jnp.where(
                    musculotendon_len < mt_slack,
                    musculotendon_len - l0_se,  # No passive forces yet
                    equilibrium_length,  # Equilibrium solution
                ),
            ),
        )

        fiber_length = jnp.clip(fiber_length, a_min=0.001 * params.optimal_fiber_length)

        # Compute initial velocity from force equilibrium
        activation = jnp.ones((batch_size, n_muscles)) * params.min_activation

        # Compute forces at initial state
        tendon_len = musculotendon_len - fiber_length
        tendon_strain = jnp.clip((tendon_len - l0_se) / l0_se, a_min=0.0)
        muscle_strain = jnp.clip((fiber_length - l0_pe) / l0_ce, a_min=0.0)

        flse = jnp.clip(k_se * (tendon_strain**2), a_max=1.0)
        flpe = k_pe * (muscle_strain**2)
        active_force = jnp.clip(flse - flpe, a_min=0.0)

        # Compute initial velocity
        fiber_velocity_n = CompliantTendonMuscle._normalized_muscle_vel(
            fiber_length / l0_ce, activation, active_force, params
        )
        fiber_velocity = fiber_velocity_n * params.max_velocity

        return MuscleState(
            activation=activation,
            fiber_length=fiber_length,
            fiber_velocity=fiber_velocity,
        )

    @staticmethod
    @jit
    def _normalized_muscle_vel(
        fiber_length_n: jnp.ndarray,
        activation: jnp.ndarray,
        active_force: jnp.ndarray,
        params: CompliantTendonMuscleParams,
    ) -> jnp.ndarray:
        """Compute normalized muscle velocity from force equilibrium.

        This inverts the force-velocity relationship to find the velocity
        that produces the required active force.

        Args:
            fiber_length_n: Normalized fiber length (L/L0).
            activation: Muscle activation.
            active_force: Required active force (normalized).
            params: Muscle parameters.

        Returns:
            Normalized fiber velocity (V/Vmax).
        """
        # Force-length curve
        flce = jnp.clip(
            1.0 + (-fiber_length_n**2 + 2*fiber_length_n - 1) / params.f_iso_n_den,
            a_min=params.min_flce,
        )

        a_rel_st = jnp.where(fiber_length_n < 1.0, 0.41 * flce, 0.41)

        b_rel_st = jnp.where(
            activation < params.q_crit,
            5.2 * (1 - 0.9 * ((activation - params.q_crit) / (5e-3 - params.q_crit))) ** 2,
            5.2,
        )

        f_x_a = flce * activation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        p1 = -f_x_a * 0.5 / (params.s_as - dfdvcon0 * 2.0)
        p3 = -1.5 * f_x_a
        p2_containing_term = (4 * ((f_x_a * 0.5) ** 2) * (-params.s_as)) / (params.s_as - dfdvcon0 * 2.0)

        # Compute velocity from force (inverting the F-V curve)
        sqrt_term = (
            active_force**2
            + 2 * active_force * p1 * params.s_as
            + 2 * active_force * p3
            + p1**2 * params.s_as**2
            + 2 * p1 * p3 * params.s_as
            + p2_containing_term
            + p3**2
        )
        sqrt_term = jnp.clip(sqrt_term, a_min=0.0)

        # Different equations for shortening vs lengthening
        nom = jnp.where(
            active_force < f_x_a,
            b_rel_st * (active_force - f_x_a),
            -active_force + p1 * params.s_as - p3 - jnp.sqrt(sqrt_term),
        )
        den = jnp.where(
            active_force < f_x_a,
            active_force + activation * a_rel_st,
            -2 * params.s_as,
        )

        return nom / den

    @staticmethod
    @jit
    def ode(
        excitation: jnp.ndarray,
        muscle_state: MuscleState,
        geometry_state: GeometryState,
        params: CompliantTendonMuscleParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute muscle state derivatives.

        For compliant tendon, we need to compute both activation and fiber
        velocity derivatives.

        Args:
            excitation: Neural excitation [0, 1]. Shape: (batch, n_muscles)
            muscle_state: Current muscle state.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            d_activation/dt: Shape: (batch, n_muscles)
            d_fiber_length_n/dt: Normalized velocity. Shape: (batch, n_muscles)
        """
        d_activation = activation_ode(
            excitation,
            muscle_state.activation,
            params.tau_activation,
            params.tau_deactivation,
            params.min_activation,
        )

        # Compute active force from current state
        fiber_length = muscle_state.fiber_length
        fiber_length_n = fiber_length / params.optimal_fiber_length
        musculotendon_len = geometry_state.musculotendon_length

        tendon_len = musculotendon_len - fiber_length
        tendon_strain = jnp.clip(
            (tendon_len - params.tendon_slack_length) / params.tendon_slack_length,
            a_min=0.0,
        )
        muscle_strain = jnp.clip(
            (fiber_length - params.passive_slack_length) / params.optimal_fiber_length,
            a_min=0.0,
        )

        flse = jnp.clip(params.k_se * (tendon_strain**2), a_max=1.0)
        flpe = params.k_pe * (muscle_strain**2)
        active_force = jnp.clip(flse - flpe, a_min=0.0)

        # Compute velocity from force equilibrium
        fiber_velocity_n = CompliantTendonMuscle._normalized_muscle_vel(
            fiber_length_n, muscle_state.activation, active_force, params
        )

        return d_activation, fiber_velocity_n

    @staticmethod
    @jit
    def integrate(
        dt: float,
        d_activation: jnp.ndarray,
        fiber_velocity_n: jnp.ndarray,
        muscle_state: MuscleState,
        geometry_state: GeometryState,
        params: CompliantTendonMuscleParams,
    ) -> MuscleState:
        """Integrate muscle state forward in time.

        Args:
            dt: Time step.
            d_activation: Activation derivative. Shape: (batch, n_muscles)
            fiber_velocity_n: Normalized fiber velocity. Shape: (batch, n_muscles)
            muscle_state: Current muscle state.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            New muscle state.
        """
        # Update activation
        new_activation = muscle_state.activation + d_activation * dt
        new_activation = clip_activation(new_activation, params.min_activation)

        # Update fiber length
        fiber_length_n = muscle_state.fiber_length / params.optimal_fiber_length
        new_fiber_length = (fiber_length_n + dt * fiber_velocity_n) * params.optimal_fiber_length

        # Clip to valid range
        new_fiber_length = jnp.clip(
            new_fiber_length,
            a_min=0.001 * params.optimal_fiber_length,
        )

        # Compute velocity in absolute units
        fiber_velocity = fiber_velocity_n * params.max_velocity

        return MuscleState(
            activation=new_activation,
            fiber_length=new_fiber_length,
            fiber_velocity=fiber_velocity,
        )

    @staticmethod
    @jit
    def compute_force(
        muscle_state: MuscleState,
        geometry_state: GeometryState,
        params: CompliantTendonMuscleParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute muscle force.

        For compliant tendon, force is determined by tendon stretch.

        Args:
            muscle_state: Current muscle state.
            geometry_state: Current geometry state.
            params: Muscle parameters.

        Returns:
            force: Total muscle force. Shape: (batch, n_muscles)
            flpe: Passive force-length. Shape: (batch, n_muscles)
            flse: Tendon force. Shape: (batch, n_muscles)
            active_force: Active force component. Shape: (batch, n_muscles)
        """
        fiber_length = muscle_state.fiber_length
        musculotendon_len = geometry_state.musculotendon_length

        tendon_len = musculotendon_len - fiber_length
        tendon_strain = jnp.clip(
            (tendon_len - params.tendon_slack_length) / params.tendon_slack_length,
            a_min=0.0,
        )
        muscle_strain = jnp.clip(
            (fiber_length - params.passive_slack_length) / params.optimal_fiber_length,
            a_min=0.0,
        )

        flse = jnp.clip(params.k_se * (tendon_strain**2), a_max=1.0)
        flpe = params.k_pe * (muscle_strain**2)
        active_force = jnp.clip(flse - flpe, a_min=0.0)

        # Force is determined by tendon
        force = flse * params.max_isometric_force

        return force, flpe, flse, active_force
