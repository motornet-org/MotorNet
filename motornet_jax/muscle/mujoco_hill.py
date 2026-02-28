"""
MuJoCo Hill-type muscle model.

Implementation based on the MuJoCo documentation:
https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators

This is a rigid tendon Hill-type model with configurable
force-length and force-velocity relationships.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


class MujocoHillMuscleParams(NamedTuple):
    """Parameters for MuJoCo Hill-type muscle.

    Attributes:
        n_muscles: Number of muscles.
        max_iso_force: Maximum isometric force per muscle. Shape: (n_muscles,)
        l0_ce: Optimal contractile element length. Shape: (n_muscles,)
        l0_se: Tendon slack length. Shape: (n_muscles,)
        l0_pe: Passive element slack length (normalized). Shape: (n_muscles,)
        lmin: Lower bound on muscle length range. Shape: (n_muscles,)
        lmax: Upper bound on muscle length range. Shape: (n_muscles,)
        vmax: Maximum shortening velocity. Shape: (n_muscles,)
        fvmax: Force at max lengthening velocity. Shape: (n_muscles,)
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
        min_activation: Minimum activation level.
        passive_forces: Scaling factor for passive forces.
    """
    n_muscles: int
    max_iso_force: jnp.ndarray
    l0_ce: jnp.ndarray
    l0_se: jnp.ndarray
    l0_pe: jnp.ndarray
    lmin: jnp.ndarray
    lmax: jnp.ndarray
    vmax: jnp.ndarray
    fvmax: jnp.ndarray
    tau_activation: float = 0.01
    tau_deactivation: float = 0.04
    min_activation: float = 0.0
    passive_forces: float = 1.0


class MujocoHillMuscleState(NamedTuple):
    """State for MuJoCo Hill-type muscle.

    Attributes:
        activation: Current activation level. Shape: (batch, n_muscles)
        fiber_length: Muscle fiber length. Shape: (batch, n_muscles)
        fiber_velocity: Muscle fiber velocity. Shape: (batch, n_muscles)
        force_length_pe: Passive element force-length. Shape: (batch, n_muscles)
        force_length_ce: Contractile element force-length. Shape: (batch, n_muscles)
        force_velocity_ce: Contractile element force-velocity. Shape: (batch, n_muscles)
        force: Total muscle force. Shape: (batch, n_muscles)
    """
    activation: jnp.ndarray
    fiber_length: jnp.ndarray
    fiber_velocity: jnp.ndarray
    force_length_pe: jnp.ndarray
    force_length_ce: jnp.ndarray
    force_velocity_ce: jnp.ndarray
    force: jnp.ndarray


class MujocoHillMuscle:
    """MuJoCo Hill-type muscle model.

    A rigid tendon Hill-type muscle with:
    - Configurable force-length relationship
    - Force-velocity relationship
    - Passive element forces
    - First-order activation dynamics

    Based on MuJoCo documentation.
    """

    def __init__(
        self,
        n_muscles: int = 6,
        max_iso_force: float = 1000.0,
        optimal_muscle_length: float = 0.1,
        tendon_length: float = 0.1,
        normalized_slack_muscle_length: float = 1.3,
        lmin: float = 0.5,
        lmax: float = 1.6,
        vmax: float = 1.5,
        fvmax: float = 1.2,
        tau_activation: float = 0.01,
        tau_deactivation: float = 0.04,
        min_activation: float = 0.0,
        passive_forces: float = 1.0,
    ):
        """Initialize MuJoCo Hill muscle.

        Args:
            n_muscles: Number of muscles.
            max_iso_force: Maximum isometric force (N).
            optimal_muscle_length: Optimal fiber length (m).
            tendon_length: Tendon slack length (m).
            normalized_slack_muscle_length: Normalized passive slack length.
            lmin: Lower bound on operating range (normalized).
            lmax: Upper bound on operating range (normalized).
            vmax: Max shortening velocity (lengths/sec).
            fvmax: Force at max lengthening velocity (relative).
            tau_activation: Activation time constant (s).
            tau_deactivation: Deactivation time constant (s).
            min_activation: Minimum activation level.
            passive_forces: Scaling for passive forces.
        """
        # Convert scalars to arrays
        def to_array(x):
            arr = jnp.atleast_1d(jnp.array(x, dtype=jnp.float32))
            if arr.size == 1:
                arr = jnp.broadcast_to(arr, (n_muscles,))
            return arr

        self.params = MujocoHillMuscleParams(
            n_muscles=n_muscles,
            max_iso_force=to_array(max_iso_force),
            l0_ce=to_array(optimal_muscle_length),
            l0_se=to_array(tendon_length),
            l0_pe=to_array(normalized_slack_muscle_length),
            lmin=to_array(lmin),
            lmax=to_array(lmax),
            vmax=to_array(vmax) * to_array(optimal_muscle_length),  # Convert to m/s
            fvmax=to_array(fvmax),
            tau_activation=tau_activation,
            tau_deactivation=tau_deactivation,
            min_activation=min_activation,
            passive_forces=passive_forces,
        )

        self.n_muscles = n_muscles
        self.state_dim = 7  # activation, length, velocity, flPE, flCE, fvCE, force

    def get_params(self) -> MujocoHillMuscleParams:
        """Get muscle parameters."""
        return self.params

    @staticmethod
    def _bump(L: jnp.ndarray, lmin: jnp.ndarray, mid: jnp.ndarray,
              lmax: jnp.ndarray) -> jnp.ndarray:
        """Skewed bump function (quadratic spline) for force-length.

        Args:
            L: Normalized muscle length.
            lmin: Minimum length.
            mid: Peak length.
            lmax: Maximum length.

        Returns:
            Bump function value in [0, 1].
        """
        left = 0.5 * (lmin + mid)
        right = 0.5 * (mid + lmax)

        out_of_range = (L <= lmin) | (L >= lmax)
        less_than_left = L < left
        less_than_mid = L < mid
        less_than_right = L < right

        x = jnp.where(
            out_of_range, 0.0,
            jnp.where(
                less_than_left, (L - lmin) / (left - lmin),
                jnp.where(
                    less_than_mid, (mid - L) / (mid - left),
                    jnp.where(
                        less_than_right, (L - mid) / (right - mid),
                        (lmax - L) / (lmax - right)
                    )
                )
            )
        )

        half_x_sq = 0.5 * x * x

        y = jnp.where(
            out_of_range, 0.0,
            jnp.where(
                less_than_left, half_x_sq,
                jnp.where(
                    less_than_mid, 1 - half_x_sq,
                    jnp.where(
                        less_than_right, 1 - half_x_sq,
                        half_x_sq
                    )
                )
            )
        )

        return y

    @staticmethod
    def activation_ode(
        action: jnp.ndarray,
        activation: jnp.ndarray,
        params: MujocoHillMuscleParams,
    ) -> jnp.ndarray:
        """Compute activation derivative.

        First-order dynamics with separate activation/deactivation time constants.

        Args:
            action: Input drive (0-1). Shape: (batch, n_muscles)
            activation: Current activation. Shape: (batch, n_muscles)
            params: Muscle parameters.

        Returns:
            Activation derivative. Shape: (batch, n_muscles)
        """
        # Determine if activating or deactivating
        tau = jnp.where(
            action > activation,
            params.tau_activation,
            params.tau_deactivation
        )

        return (action - activation) / tau

    @staticmethod
    def integrate(
        dt: float,
        activation_deriv: jnp.ndarray,
        muscle_state: MujocoHillMuscleState,
        geometry_state: jnp.ndarray,
        params: MujocoHillMuscleParams,
    ) -> MujocoHillMuscleState:
        """Integrate muscle state forward one timestep.

        Args:
            dt: Timestep size.
            activation_deriv: Activation derivative. Shape: (batch, n_muscles)
            muscle_state: Current muscle state.
            geometry_state: Musculotendon geometry [length, velocity, ...]. Shape: (batch, 2+, n_muscles)
            params: Muscle parameters.

        Returns:
            New muscle state.
        """
        # Update activation
        activation = muscle_state.activation + activation_deriv * dt
        activation = jnp.clip(activation, params.min_activation, 1.0)

        # Get musculotendon geometry
        musculotendon_len = geometry_state[:, 0, :]
        musculotendon_vel = geometry_state[:, 1, :]

        # Compute muscle (fiber) length and velocity
        # For rigid tendon: muscle_len = musculotendon_len - tendon_len
        muscle_len = jnp.clip(musculotendon_len - params.l0_se, 0.001, None)
        muscle_len_n = muscle_len / params.l0_ce  # Normalized
        muscle_vel_n = musculotendon_vel / params.vmax  # Normalized

        # Derived quantities for force-length
        b = 0.5 * (1 + params.lmax)
        p1 = b - 1
        p2 = 0.25 * params.l0_pe
        mid = 0.5 * (params.lmin + 0.95)
        c = params.fvmax - 1

        # Passive force-length (PE)
        x = jnp.where(
            muscle_len_n <= 1, 0.0,
            jnp.where(
                muscle_len_n <= b,
                (muscle_len_n - 1) / p1,
                (muscle_len_n - b) / p1
            )
        )

        flpe = jnp.where(
            muscle_len_n <= 1, 0.0,
            jnp.where(
                muscle_len_n <= b,
                p2 * x**3,
                p2 * (1 + 3*x)
            )
        )

        # Active force-length (CE) - bump functions
        flce_main = MujocoHillMuscle._bump(
            muscle_len_n, params.lmin, jnp.ones_like(params.lmin), params.lmax
        )
        flce_secondary = MujocoHillMuscle._bump(
            muscle_len_n, params.lmin, mid, jnp.full_like(params.lmin, 0.95)
        )
        flce = flce_main + 0.15 * flce_secondary

        # Force-velocity (CE)
        fvce = jnp.where(
            muscle_vel_n <= -1, 0.0,
            jnp.where(
                muscle_vel_n <= 0,
                (muscle_vel_n + 1)**2,
                jnp.where(
                    muscle_vel_n <= c,
                    params.fvmax - (c - muscle_vel_n)**2 / c,
                    params.fvmax
                )
            )
        )

        # Total force
        force = (activation * flce * fvce + params.passive_forces * flpe) * params.max_iso_force

        return MujocoHillMuscleState(
            activation=activation,
            fiber_length=muscle_len,
            fiber_velocity=musculotendon_vel,
            force_length_pe=flpe,
            force_length_ce=flce,
            force_velocity_ce=fvce,
            force=force,
        )

    @staticmethod
    def get_initial_state(
        batch_size: int,
        geometry_state: jnp.ndarray,
        params: MujocoHillMuscleParams,
    ) -> MujocoHillMuscleState:
        """Get initial muscle state.

        Args:
            batch_size: Batch size.
            geometry_state: Initial geometry. Shape: (batch, 2+, n_muscles)
            params: Muscle parameters.

        Returns:
            Initial muscle state.
        """
        activation = jnp.full((batch_size, params.n_muscles), params.min_activation)

        # Create dummy state for integration
        dummy_state = MujocoHillMuscleState(
            activation=activation,
            fiber_length=jnp.zeros((batch_size, params.n_muscles)),
            fiber_velocity=jnp.zeros((batch_size, params.n_muscles)),
            force_length_pe=jnp.zeros((batch_size, params.n_muscles)),
            force_length_ce=jnp.zeros((batch_size, params.n_muscles)),
            force_velocity_ce=jnp.zeros((batch_size, params.n_muscles)),
            force=jnp.zeros((batch_size, params.n_muscles)),
        )

        # Integrate with zero derivative to get proper initial forces
        return MujocoHillMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy_state, geometry_state, params
        )
