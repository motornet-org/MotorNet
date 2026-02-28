"""
Arm26 Effector - 6-muscle arm with polynomial moment arm approximation.

This implements the lumped-muscle model from Nijhof & Kouwenhoven (2000),
which uses polynomial functions for moment arm approximation instead of
explicit muscle path geometry.

This is the most commonly used arm model in motor control research.
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import lax

from motornet_jax.types import (
    JointState,
    CartesianState,
    MuscleState,
    GeometryState,
    EffectorState,
)
from motornet_jax.skeleton import TwoDofArm
from motornet_jax.skeleton.arm import TwoDofArmParams


class Arm26Params(NamedTuple):
    """Parameters for the Arm26 effector.

    This uses polynomial approximations for moment arms, which is more
    computationally efficient than explicit path geometry.

    Attributes:
        skeleton: TwoDofArm parameters.
        a0: Constant term for musculotendon length. Shape: (6,)
        a1: Linear term for musculotendon length. Shape: (2, 6)
        a2: Quadratic term for musculotendon length. Shape: (2, 6)
        a3: Angle offset. Shape: (2,)
        max_isometric_force: Maximum force per muscle. Shape: (6,)
        optimal_fiber_length: Optimal fiber length. Shape: (6,)
        tendon_slack_length: Tendon slack length. Shape: (6,)
        dt: Timestep.
        n_ministeps: Number of ministeps per step.
        damping: Joint damping coefficient.
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
        min_activation: Minimum activation.
    """
    skeleton: TwoDofArmParams
    a0: jnp.ndarray
    a1: jnp.ndarray
    a2: jnp.ndarray
    a3: jnp.ndarray
    max_isometric_force: jnp.ndarray
    optimal_fiber_length: jnp.ndarray
    tendon_slack_length: jnp.ndarray
    max_velocity: jnp.ndarray
    passive_slack_length: jnp.ndarray
    k_pe: jnp.ndarray
    dt: float
    n_ministeps: int
    damping: float
    tau_activation: float
    tau_deactivation: float
    min_activation: float
    # Kistemaker force-velocity parameters
    q_crit: float = 0.3
    s_as: float = 0.001
    min_flce: float = 0.01
    f_iso_n_den: float = 0.66 ** 2


class Arm26:
    """Arm26: 6-muscle arm with polynomial moment arm approximation.

    Muscles:
    0. Pectoralis (shoulder flexor)
    1. Deltoid (shoulder extensor)
    2. Brachioradialis (elbow flexor)
    3. Triceps lateral (elbow extensor)
    4. Biceps (biarticular flexor)
    5. Triceps long (biarticular extensor)

    Reference:
        Nijhof, E.-J., & Kouwenhoven, E. (2000). Simulation of Multijoint Arm Movements.
        In J. M. Winters & P. E. Crago, Biomechanics and Neural Control of Posture and Movement.
    """

    n_muscles = 6
    n_joints = 2
    n_dim = 2

    muscle_names = [
        'pectoralis',
        'deltoid',
        'brachioradialis',
        'tricepslat',
        'biceps',
        'tricepslong',
    ]

    def __init__(
        self,
        dt: float = 0.01,
        n_ministeps: int = 1,
        damping: float = 0.0,
        skeleton_params: Optional[TwoDofArmParams] = None,
        tau_activation: float = 0.015,
        tau_deactivation: float = 0.05,
        min_activation: float = 0.001,
    ):
        """Initialize Arm26 effector.

        Args:
            dt: Timestep size (s).
            n_ministeps: Number of integration substeps per step.
            damping: Joint damping coefficient.
            skeleton_params: Optional custom skeleton parameters.
            tau_activation: Activation time constant.
            tau_deactivation: Deactivation time constant.
            min_activation: Minimum activation level.
        """
        # Default skeleton (from Nijhof & Kouwenhoven 2000)
        if skeleton_params is None:
            skeleton = TwoDofArm(
                m1=1.82, m2=1.43,
                l1g=0.135, l2g=0.165,
                i1=0.051, i2=0.057,
                l1=0.309, l2=0.333,
                pos_lower_bound=(0.0, 0.0),
                pos_upper_bound=(jnp.deg2rad(135), jnp.deg2rad(155)),
            )
            skeleton_params = skeleton.get_params()

        # Polynomial coefficients for geometry
        # a0: baseline musculotendon length
        a0 = jnp.array([0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989])

        # a1: linear coefficient (moment arm at reference position)
        # Shape: (2, 6) - for shoulder and elbow
        # Muscles: [pectoralis, deltoid, brachioradialis, tricepslat, biceps, tricepslong]
        # From PyTorch flat: [-.03, .03, 0, 0, -.03, .03, 0, 0, -.014, .025, -.016, .03]
        # reshaped to (2, 6) in C order
        a1 = jnp.array([
            [-0.03, 0.03, 0.0, 0.0, -0.03, 0.03],       # shoulder
            [0.0, 0.0, -0.014, 0.025, -0.016, 0.03],     # elbow
        ])

        # a2: quadratic coefficient
        # From PyTorch flat: [0, 0, 0, 0, 0, 0, 0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]
        # reshaped to (2, 6) in C order
        a2 = jnp.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                   # shoulder
            [0.0, 0.0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3],     # elbow
        ])

        # a3: reference angle (offset)
        a3 = jnp.array([jnp.pi / 2, 0.0])

        # Muscle parameters
        max_isometric_force = jnp.array([838.0, 1207.0, 1422.0, 1549.0, 414.0, 603.0])
        tendon_slack_length = jnp.array([0.039, 0.066, 0.172, 0.187, 0.204, 0.217])
        optimal_fiber_length = jnp.array([0.134, 0.140, 0.092, 0.093, 0.137, 0.127])

        # Derived muscle parameters (matching PyTorch RigidTendonHillMuscle)
        normalized_slack = 1.4
        max_velocity = 10.0 * optimal_fiber_length
        passive_slack_length = normalized_slack * optimal_fiber_length
        k_pe = 1.0 / ((1.66 - normalized_slack) ** 2)
        k_pe = jnp.ones(6) * k_pe

        self.params = Arm26Params(
            skeleton=skeleton_params,
            a0=a0,
            a1=a1,
            a2=a2,
            a3=a3,
            max_isometric_force=max_isometric_force,
            optimal_fiber_length=optimal_fiber_length,
            tendon_slack_length=tendon_slack_length,
            max_velocity=max_velocity,
            passive_slack_length=passive_slack_length,
            k_pe=k_pe,
            dt=dt,
            n_ministeps=n_ministeps,
            damping=damping,
            tau_activation=tau_activation,
            tau_deactivation=tau_deactivation,
            min_activation=min_activation,
        )

        self.minidt = dt / n_ministeps

    def get_params(self) -> Arm26Params:
        """Get effector parameters."""
        return self.params

    @staticmethod
    def compute_geometry(
        joint_state: JointState,
        params: Arm26Params,
    ) -> GeometryState:
        """Compute musculotendon geometry using polynomial approximation.

        This is MUCH faster than explicit path geometry computation.

        Args:
            joint_state: Current joint state.
            params: Arm26 parameters.

        Returns:
            Geometry state with musculotendon lengths, velocities, and moment arms.
        """
        pos = joint_state.position  # (batch, 2)
        vel = joint_state.velocity  # (batch, 2)

        # Offset position by reference angle
        pos_offset = pos - params.a3  # (batch, 2)

        # Moment arm: derivative of length w.r.t. angle
        # moment_arm = a1 + 2 * a2 * pos_offset
        # Shape: (batch, 2, 6)
        moment_arm = params.a1 + 2 * params.a2 * pos_offset[:, :, None]

        # Musculotendon length: polynomial in joint angles
        # length = a0 + sum_j((a1[j] + a2[j] * pos[j]) * pos[j])
        # Shape: (batch, 6)
        length_contribution = (params.a1 + params.a2 * pos_offset[:, :, None]) * pos_offset[:, :, None]
        musculotendon_length = params.a0 + jnp.sum(length_contribution, axis=1)

        # Musculotendon velocity: chain rule
        # velocity = sum_j(moment_arm[j] * joint_vel[j])
        # Shape: (batch, 6)
        musculotendon_velocity = jnp.sum(moment_arm * vel[:, :, None], axis=1)

        # Transpose moment arm to (batch, 6, 2) for consistency with effector interface
        moment_arm_transposed = jnp.transpose(moment_arm, (0, 2, 1))

        return GeometryState(
            musculotendon_length=musculotendon_length,
            musculotendon_velocity=musculotendon_velocity,
            moment_arm=moment_arm_transposed,
        )

    @staticmethod
    def activation_ode(
        excitation: jnp.ndarray,
        activation: jnp.ndarray,
        params: Arm26Params,
    ) -> jnp.ndarray:
        """Compute activation dynamics.

        Args:
            excitation: Neural excitation [0, 1]. Shape: (batch, 6)
            activation: Current activation. Shape: (batch, 6)
            params: Arm26 parameters.

        Returns:
            d_activation/dt. Shape: (batch, 6)
        """
        excitation = jnp.clip(excitation, params.min_activation, 1.0)
        activation = jnp.clip(activation, params.min_activation, 1.0)

        tmp = 0.5 + 1.5 * activation
        tau = jnp.where(
            excitation > activation,
            params.tau_activation * tmp,
            params.tau_deactivation / tmp,
        )

        return (excitation - activation) / tau

    @staticmethod
    def compute_muscle_force(
        activation: jnp.ndarray,
        geometry_state: GeometryState,
        params: Arm26Params,
    ) -> jnp.ndarray:
        """Compute muscle forces using rigid tendon Hill model.

        Args:
            activation: Muscle activation. Shape: (batch, 6)
            geometry_state: Current geometry.
            params: Arm26 parameters.

        Returns:
            Muscle forces. Shape: (batch, 6)
        """
        # Clip activation to min_activation (matching PyTorch, prevents NaN in Kistemaker)
        activation = jnp.clip(activation, a_min=params.min_activation)

        # Fiber length (rigid tendon: subtract tendon slack length)
        fiber_length = jnp.clip(
            geometry_state.musculotendon_length - params.tendon_slack_length,
            a_min=0.001,
        )

        # Normalized fiber length and velocity
        fiber_length_n = fiber_length / params.optimal_fiber_length
        fiber_velocity_n = geometry_state.musculotendon_velocity / params.max_velocity

        # Passive force (parallel element)
        muscle_strain = jnp.clip(
            (fiber_length - params.passive_slack_length) / params.optimal_fiber_length,
            a_min=0.0,
        )
        flpe = params.k_pe * (muscle_strain ** 2)

        # Active force-length (contractile element, bell curve)
        flce = jnp.clip(
            1.0 + (-fiber_length_n**2 + 2*fiber_length_n - 1) / params.f_iso_n_den,
            a_min=params.min_flce,
        )

        # Force-velocity (Kistemaker model, matching PyTorch RigidTendonHillMuscle)
        a_rel_st = jnp.where(fiber_length_n > 1.0, 0.41 * flce, 0.41)
        b_rel_st = jnp.where(
            activation < params.q_crit,
            5.2 * (1 - 0.9 * ((activation - params.q_crit) / (5e-3 - params.q_crit))) ** 2,
            5.2,
        )

        f_x_a = flce * activation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        tmp_p_nom = f_x_a * 0.5
        tmp_p_den = params.s_as - dfdvcon0 * 2.0

        p1 = -tmp_p_nom / tmp_p_den
        p2 = (tmp_p_nom ** 2) / tmp_p_den
        p3 = -1.5 * f_x_a

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

        return force

    @staticmethod
    def compute_joint_torques(
        muscle_forces: jnp.ndarray,
        geometry_state: GeometryState,
        joint_velocity: jnp.ndarray,
        joint_load: jnp.ndarray,
        params: Arm26Params,
    ) -> jnp.ndarray:
        """Compute net joint torques.

        Args:
            muscle_forces: Muscle forces. Shape: (batch, 6)
            geometry_state: Current geometry.
            joint_velocity: Joint velocities. Shape: (batch, 2)
            joint_load: External joint loads. Shape: (batch, 2)
            params: Arm26 parameters.

        Returns:
            Net joint torques. Shape: (batch, 2)
        """
        # Torque from muscles: tau = -sum(F * moment_arm)
        # moment_arm: (batch, 6, 2)
        muscle_torques = -jnp.sum(
            muscle_forces[:, :, None] * geometry_state.moment_arm,
            axis=1,
        )  # (batch, 2)

        # Add damping and external loads
        total_torques = muscle_torques + joint_load - params.damping * joint_velocity

        return total_torques

    @staticmethod
    def step(
        state: EffectorState,
        action: jnp.ndarray,
        endpoint_load: jnp.ndarray,
        joint_load: jnp.ndarray,
        params: Arm26Params,
    ) -> EffectorState:
        """Take one simulation step (single ministep, Euler integration).

        Args:
            state: Current effector state.
            action: Muscle excitation. Shape: (batch, 6)
            endpoint_load: External endpoint load. Shape: (batch, 2)
            joint_load: External joint load. Shape: (batch, 2)
            params: Arm26 parameters.

        Returns:
            New effector state.
        """
        dt = params.dt / params.n_ministeps
        activation = state.muscle.activation

        # Compute forces
        forces = Arm26.compute_muscle_force(activation, state.geometry, params)

        # Compute joint torques
        torques = Arm26.compute_joint_torques(
            forces, state.geometry, state.joint.velocity, joint_load, params
        )

        # Skeleton ODE
        _, acceleration = TwoDofArm.ode(
            state.joint, torques, endpoint_load, params.skeleton
        )

        # Activation ODE
        d_activation = Arm26.activation_ode(action, activation, params)

        # Integrate skeleton
        new_joint = TwoDofArm.integrate(state.joint, acceleration, dt, params.skeleton)

        # Integrate activation
        new_activation = jnp.clip(
            activation + dt * d_activation,
            params.min_activation,
            1.0,
        )

        # Recompute geometry
        new_geometry = Arm26.compute_geometry(new_joint, params)

        # Recompute fiber state
        new_fiber_length = jnp.clip(
            new_geometry.musculotendon_length - params.tendon_slack_length,
            a_min=0.001,
        )

        new_muscle = MuscleState(
            activation=new_activation,
            fiber_length=new_fiber_length,
            fiber_velocity=new_geometry.musculotendon_velocity,
        )

        # Compute fingertip position only (skip full FK with velocities)
        new_fingertip = TwoDofArm.fingertip_position(new_joint, params.skeleton)

        return EffectorState(
            joint=new_joint,
            cartesian=CartesianState(position=new_fingertip, velocity=jnp.zeros_like(new_fingertip)),
            muscle=new_muscle,
            geometry=new_geometry,
            fingertip=new_fingertip,
        )

    @staticmethod
    def simulate(
        state: EffectorState,
        action: jnp.ndarray,
        endpoint_load: jnp.ndarray,
        joint_load: jnp.ndarray,
        params: Arm26Params,
        n_ministeps: int,
    ) -> EffectorState:
        """Simulate for n_ministeps with constant action.

        Args:
            state: Initial effector state.
            action: Muscle excitation. Shape: (batch, 6)
            endpoint_load: External endpoint load. Shape: (batch, 2)
            joint_load: External joint load. Shape: (batch, 2)
            params: Arm26 parameters.
            n_ministeps: Number of ministeps.

        Returns:
            Final effector state.
        """
        def body_fn(i, state):
            return Arm26.step(state, action, endpoint_load, joint_load, params)

        return lax.fori_loop(0, n_ministeps, body_fn, state)

    def reset(
        self,
        batch_size: int = 1,
        joint_state: Optional[JointState] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> EffectorState:
        """Reset to initial state.

        Args:
            batch_size: Batch size.
            joint_state: Optional initial joint state.
            key: Random key for initialization.

        Returns:
            Initial effector state.
        """
        if joint_state is None:
            if key is None:
                key = jax.random.PRNGKey(0)
            pos = jax.random.uniform(
                key,
                (batch_size, 2),
                minval=self.params.skeleton.pos_lower_bound,
                maxval=self.params.skeleton.pos_upper_bound,
            )
            vel = jnp.zeros((batch_size, 2))
            joint_state = JointState(position=pos, velocity=vel)

        # Compute geometry
        geometry = self.compute_geometry(joint_state, self.params)

        # Initial muscle state
        activation = jnp.ones((batch_size, 6)) * self.params.min_activation
        fiber_length = jnp.clip(
            geometry.musculotendon_length - self.params.tendon_slack_length,
            a_min=0.001,
        )
        muscle = MuscleState(
            activation=activation,
            fiber_length=fiber_length,
            fiber_velocity=geometry.musculotendon_velocity,
        )

        # Cartesian state
        cartesian = TwoDofArm.joint2cartesian(joint_state, self.params.skeleton)

        return EffectorState(
            joint=joint_state,
            cartesian=cartesian,
            muscle=muscle,
            geometry=geometry,
            fingertip=cartesian.position,
        )


# Also create a compliant tendon version
class Arm26Compliant(Arm26):
    """Arm26 with compliant tendon muscles.

    Uses smaller timestep and RK4 integration for stability.
    """

    def __init__(
        self,
        dt: float = 0.0002,  # Smaller timestep for stability
        n_ministeps: int = 50,  # More ministeps
        damping: float = 0.0,
        **kwargs,
    ):
        # Adjust tendon lengths for compliant version
        super().__init__(dt=dt, n_ministeps=n_ministeps, damping=damping, **kwargs)

        # Update tendon lengths (compliant version has longer tendons)
        compliant_tendon_length = jnp.array([0.070, 0.070, 0.172, 0.187, 0.204, 0.217])
        compliant_a0 = jnp.array([0.182, 0.2362, 0.2859, 0.2355, 0.3329, 0.2989])

        self.params = self.params._replace(
            tendon_slack_length=compliant_tendon_length,
            a0=compliant_a0,
        )
