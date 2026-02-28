"""
Thelen Hill-type muscle model.

Implementation based on:
Thelen DG. Adjustment of muscle mechanics model parameters to simulate
dynamic contractions in older adults. J Biomech Eng. 2003 Feb;125(1):70-7.

This is a rigid tendon Hill-type model with Thelen's force-length
and force-velocity curves.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


class ThelenMuscleParams(NamedTuple):
    """Parameters for Thelen Hill-type muscle.

    Attributes:
        n_muscles: Number of muscles.
        max_iso_force: Maximum isometric force per muscle. Shape: (n_muscles,)
        l0_ce: Optimal contractile element length. Shape: (n_muscles,)
        l0_se: Tendon slack length. Shape: (n_muscles,)
        l0_pe: Passive element slack length. Shape: (n_muscles,)
        vmax: Maximum shortening velocity. Shape: (n_muscles,)
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
        min_activation: Minimum activation level.
        pe_k: Passive element shape factor.
        ce_gamma: Contractile element width parameter.
        ce_Af: Force-velocity shape factor.
        ce_fmlen: Maximum eccentric force.
    """
    n_muscles: int
    max_iso_force: jnp.ndarray
    l0_ce: jnp.ndarray
    l0_se: jnp.ndarray
    l0_pe: jnp.ndarray
    vmax: jnp.ndarray
    tau_activation: float = 0.01
    tau_deactivation: float = 0.04
    min_activation: float = 0.001
    pe_k: float = 5.0
    ce_gamma: float = 0.45
    ce_Af: float = 0.25
    ce_fmlen: float = 1.4


class ThelenMuscleState(NamedTuple):
    """State for Thelen Hill-type muscle.

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


class ThelenMuscle:
    """Thelen Hill-type muscle model.

    A rigid tendon Hill-type muscle with Thelen's formulation:
    - Gaussian force-length relationship for CE
    - Exponential force-length for PE
    - Hyperbolic force-velocity relationship
    - First-order activation dynamics

    Reference:
    Thelen DG. Adjustment of muscle mechanics model parameters to simulate
    dynamic contractions in older adults. J Biomech Eng. 2003.
    """

    def __init__(
        self,
        n_muscles: int = 6,
        max_iso_force: float = 1000.0,
        optimal_muscle_length: float = 0.1,
        tendon_length: float = 0.1,
        normalized_slack_muscle_length: float = 1.0,
        tau_activation: float = 0.01,
        tau_deactivation: float = 0.04,
        min_activation: float = 0.001,
    ):
        """Initialize Thelen muscle.

        Args:
            n_muscles: Number of muscles.
            max_iso_force: Maximum isometric force (N).
            optimal_muscle_length: Optimal fiber length (m).
            tendon_length: Tendon slack length (m).
            normalized_slack_muscle_length: Normalized passive slack length.
            tau_activation: Activation time constant (s).
            tau_deactivation: Deactivation time constant (s).
            min_activation: Minimum activation level.
        """
        # Convert scalars to arrays
        def to_array(x):
            arr = jnp.atleast_1d(jnp.array(x, dtype=jnp.float32))
            if arr.size == 1:
                arr = jnp.broadcast_to(arr, (n_muscles,))
            return arr

        l0_ce = to_array(optimal_muscle_length)

        self.params = ThelenMuscleParams(
            n_muscles=n_muscles,
            max_iso_force=to_array(max_iso_force),
            l0_ce=l0_ce,
            l0_se=to_array(tendon_length),
            l0_pe=l0_ce * normalized_slack_muscle_length,
            vmax=10.0 * l0_ce,  # Standard: 10 optimal lengths per second
            tau_activation=tau_activation,
            tau_deactivation=tau_deactivation,
            min_activation=min_activation,
        )

        # Pre-computed constants
        self._pe_1 = self.params.pe_k / 0.6  # epsilon_0^M from Thelen eq. 3
        self._pe_den = jnp.exp(self.params.pe_k) - 1

        self.n_muscles = n_muscles
        self.state_dim = 7

    def get_params(self) -> ThelenMuscleParams:
        """Get muscle parameters."""
        return self.params

    @staticmethod
    def activation_ode(
        action: jnp.ndarray,
        activation: jnp.ndarray,
        params: ThelenMuscleParams,
    ) -> jnp.ndarray:
        """Compute activation derivative.

        Args:
            action: Input drive (0-1). Shape: (batch, n_muscles)
            activation: Current activation. Shape: (batch, n_muscles)
            params: Muscle parameters.

        Returns:
            Activation derivative. Shape: (batch, n_muscles)
        """
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
        muscle_state: ThelenMuscleState,
        geometry_state: jnp.ndarray,
        params: ThelenMuscleParams,
    ) -> ThelenMuscleState:
        """Integrate muscle state forward one timestep.

        Args:
            dt: Timestep size.
            activation_deriv: Activation derivative. Shape: (batch, n_muscles)
            muscle_state: Current muscle state.
            geometry_state: Musculotendon geometry. Shape: (batch, 2+, n_muscles)
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

        # Compute muscle length (rigid tendon assumption)
        muscle_len = jnp.clip(musculotendon_len - params.l0_se, 0.001, None)
        muscle_vel = musculotendon_vel  # With rigid tendon

        # Pre-compute for force-velocity
        pe_k = params.pe_k
        pe_1 = pe_k / 0.6
        pe_den = jnp.exp(pe_k) - 1
        ce_Af = params.ce_Af
        ce_fmlen = params.ce_fmlen
        vmax = params.vmax

        # Pre-computed terms for force-velocity (from Thelen)
        ce_0 = 3 * vmax
        ce_1 = ce_Af * vmax
        ce_2 = 3 * ce_Af * vmax * ce_fmlen - 3 * ce_Af * vmax
        ce_3 = 8 * ce_Af * ce_fmlen + 8 * ce_fmlen
        ce_4 = ce_Af * ce_fmlen * vmax - ce_1
        ce_5 = 8 * (ce_Af + 1)

        # Force-velocity relationship (Thelen formulation)
        a3 = activation * 3
        concentric = muscle_vel <= 0

        nom = jnp.where(
            concentric,
            ce_Af * (activation * ce_0 + 4 * muscle_vel + vmax),
            ce_2 * activation + ce_3 * muscle_vel + ce_4
        )
        den = jnp.where(
            concentric,
            a3 * ce_1 + ce_1 - 4 * muscle_vel,
            ce_4 * a3 + ce_5 * muscle_vel + ce_4
        )
        fvce = jnp.clip(nom / den, 0.0, None)

        # Passive force-length (exponential)
        flpe = jnp.clip(
            (jnp.exp(pe_1 * (muscle_len - params.l0_pe) / params.l0_ce) - 1) / pe_den,
            0.0, None
        )

        # Active force-length (Gaussian)
        muscle_len_n = muscle_len / params.l0_ce
        flce = jnp.exp(-((muscle_len_n - 1)**2) / params.ce_gamma)

        # Total force
        force = (activation * flce * fvce + flpe) * params.max_iso_force

        return ThelenMuscleState(
            activation=activation,
            fiber_length=muscle_len,
            fiber_velocity=muscle_vel,
            force_length_pe=flpe,
            force_length_ce=flce,
            force_velocity_ce=fvce,
            force=force,
        )

    @staticmethod
    def get_initial_state(
        batch_size: int,
        geometry_state: jnp.ndarray,
        params: ThelenMuscleParams,
    ) -> ThelenMuscleState:
        """Get initial muscle state.

        Args:
            batch_size: Batch size.
            geometry_state: Initial geometry. Shape: (batch, 2+, n_muscles)
            params: Muscle parameters.

        Returns:
            Initial muscle state.
        """
        activation = jnp.full((batch_size, params.n_muscles), params.min_activation)

        dummy_state = ThelenMuscleState(
            activation=activation,
            fiber_length=jnp.zeros((batch_size, params.n_muscles)),
            fiber_velocity=jnp.zeros((batch_size, params.n_muscles)),
            force_length_pe=jnp.zeros((batch_size, params.n_muscles)),
            force_length_ce=jnp.zeros((batch_size, params.n_muscles)),
            force_velocity_ce=jnp.zeros((batch_size, params.n_muscles)),
            force=jnp.zeros((batch_size, params.n_muscles)),
        )

        return ThelenMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy_state, geometry_state, params
        )
