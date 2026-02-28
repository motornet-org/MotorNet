"""
Core state types for MotorNet-JAX.

All state types are NamedTuples, which are JAX pytree-compatible.
This allows them to be used seamlessly with jit, vmap, grad, and lax operations.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp


class JointState(NamedTuple):
    """State of skeletal joints.

    Attributes:
        position: Joint angles in radians. Shape: (batch, n_joints)
        velocity: Joint angular velocities in rad/s. Shape: (batch, n_joints)
    """
    position: jnp.ndarray
    velocity: jnp.ndarray

    @classmethod
    def zeros(cls, batch_size: int, n_joints: int) -> "JointState":
        """Create zero-initialized joint state."""
        return cls(
            position=jnp.zeros((batch_size, n_joints)),
            velocity=jnp.zeros((batch_size, n_joints)),
        )


class CartesianState(NamedTuple):
    """Cartesian (endpoint) state.

    Attributes:
        position: Cartesian position. Shape: (batch, n_dim) where n_dim is typically 2 or 3
        velocity: Cartesian velocity. Shape: (batch, n_dim)
    """
    position: jnp.ndarray
    velocity: jnp.ndarray

    @classmethod
    def zeros(cls, batch_size: int, n_dim: int = 2) -> "CartesianState":
        """Create zero-initialized cartesian state."""
        return cls(
            position=jnp.zeros((batch_size, n_dim)),
            velocity=jnp.zeros((batch_size, n_dim)),
        )


class MuscleState(NamedTuple):
    """State of muscle fibers.

    Attributes:
        activation: Muscle activation level [0, 1]. Shape: (batch, n_muscles)
        fiber_length: Normalized muscle fiber length. Shape: (batch, n_muscles)
        fiber_velocity: Normalized muscle fiber velocity. Shape: (batch, n_muscles)
    """
    activation: jnp.ndarray
    fiber_length: jnp.ndarray
    fiber_velocity: jnp.ndarray

    @classmethod
    def zeros(cls, batch_size: int, n_muscles: int) -> "MuscleState":
        """Create zero-initialized muscle state (with fiber_length = 1.0)."""
        return cls(
            activation=jnp.zeros((batch_size, n_muscles)),
            fiber_length=jnp.ones((batch_size, n_muscles)),
            fiber_velocity=jnp.zeros((batch_size, n_muscles)),
        )


class GeometryState(NamedTuple):
    """Musculotendon geometry state.

    Attributes:
        musculotendon_length: Total length of muscle-tendon unit. Shape: (batch, n_muscles)
        musculotendon_velocity: Rate of change of musculotendon length. Shape: (batch, n_muscles)
        moment_arm: Moment arms for each muscle at each joint. Shape: (batch, n_muscles, n_joints)
    """
    musculotendon_length: jnp.ndarray
    musculotendon_velocity: jnp.ndarray
    moment_arm: jnp.ndarray

    @classmethod
    def zeros(cls, batch_size: int, n_muscles: int, n_joints: int) -> "GeometryState":
        """Create zero-initialized geometry state."""
        return cls(
            musculotendon_length=jnp.zeros((batch_size, n_muscles)),
            musculotendon_velocity=jnp.zeros((batch_size, n_muscles)),
            moment_arm=jnp.zeros((batch_size, n_muscles, n_joints)),
        )


class EffectorState(NamedTuple):
    """Complete state of the effector (skeleton + muscles + geometry).

    Attributes:
        joint: Joint state (positions and velocities)
        cartesian: Cartesian endpoint state
        muscle: Muscle fiber state
        geometry: Musculotendon geometry
        fingertip: Fingertip/endpoint position (convenience field)
    """
    joint: JointState
    cartesian: CartesianState
    muscle: MuscleState
    geometry: GeometryState
    fingertip: jnp.ndarray  # (batch, n_dim)


class SkeletonParams(NamedTuple):
    """Parameters for skeleton dynamics.

    These are the physical parameters of the skeleton that don't change
    during simulation but are needed for dynamics computation.
    """
    # Segment lengths
    L1: float  # Length of first segment
    L2: float  # Length of second segment

    # Masses
    M1: float  # Mass of first segment
    M2: float  # Mass of second segment

    # Moments of inertia
    I1: float  # Inertia of first segment
    I2: float  # Inertia of second segment

    # Pre-computed inertia matrix components
    inertia_c: jnp.ndarray  # Constant part of inertia matrix (2, 2)
    inertia_m: jnp.ndarray  # Multiplier for cos(q2) term (2, 2)

    # Coriolis coefficients
    coriolis_1: float
    coriolis_2: float

    # Joint limits (optional)
    pos_lower_bound: Optional[jnp.ndarray] = None  # (n_joints,)
    pos_upper_bound: Optional[jnp.ndarray] = None  # (n_joints,)
    vel_lower_bound: Optional[jnp.ndarray] = None  # (n_joints,)
    vel_upper_bound: Optional[jnp.ndarray] = None  # (n_joints,)


class MuscleParams(NamedTuple):
    """Parameters for muscle model.

    Attributes:
        max_isometric_force: Maximum isometric force per muscle. Shape: (n_muscles,)
        optimal_fiber_length: Optimal fiber length. Shape: (n_muscles,)
        tendon_slack_length: Tendon slack length. Shape: (n_muscles,)
        pennation_angle: Pennation angle at optimal length. Shape: (n_muscles,)
        max_contraction_velocity: Maximum contraction velocity (L0/s). Shape: (n_muscles,)

        # Force-length curve parameters
        width: Width of the active force-length curve. Shape: (n_muscles,)

        # Force-velocity curve parameters
        a_f: Shape parameter for force-velocity curve.
        f_max: Maximum eccentric force multiplier.

        # Passive force parameters
        k_pe: Passive element stiffness. Shape: (n_muscles,)

        # Tendon parameters (for compliant tendon)
        k_se: Series elastic element stiffness. Shape: (n_muscles,)

        # Activation dynamics
        tau_activation: Activation time constant.
        tau_deactivation: Deactivation time constant.
    """
    max_isometric_force: jnp.ndarray
    optimal_fiber_length: jnp.ndarray
    tendon_slack_length: jnp.ndarray
    pennation_angle: jnp.ndarray
    max_contraction_velocity: jnp.ndarray

    # Force-length parameters
    width: jnp.ndarray

    # Force-velocity parameters
    a_f: float = 0.25
    f_max: float = 1.4

    # Passive force parameters
    k_pe: jnp.ndarray = None

    # Tendon parameters
    k_se: jnp.ndarray = None

    # Activation dynamics
    tau_activation: float = 0.015
    tau_deactivation: float = 0.050


class EffectorParams(NamedTuple):
    """Combined parameters for the full effector.

    Attributes:
        skeleton: Skeleton parameters
        muscle: Muscle parameters
        dt: Integration timestep

        # Geometry
        fixation_body: Which body each fixation point is attached to. Shape: (n_fixation_points,)
        fixation_local: Local coordinates of fixation points. Shape: (n_fixation_points, 2)
        segment_to_muscle: Mapping from segments to muscles. Shape: (n_segments,)
        n_muscles: Number of muscles
        n_joints: Number of joints

        # Integration
        n_ministeps: Number of integration substeps per environment step
    """
    skeleton: SkeletonParams
    muscle: MuscleParams
    dt: float

    # Geometry mapping
    fixation_body: jnp.ndarray
    fixation_local: jnp.ndarray
    segment_to_muscle: jnp.ndarray
    n_muscles: int
    n_joints: int

    # Integration
    n_ministeps: int = 1
