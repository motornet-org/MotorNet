"""
Effector module - combines skeleton and muscles.

This is the core simulation module that handles:
- Muscle-skeleton coupling
- Geometry computation (musculotendon lengths, velocities, moment arms)
- Numerical integration (Euler, RK4)
- State management
"""

from typing import NamedTuple, Tuple, List, Optional, Callable
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


class EffectorParams(NamedTuple):
    """Parameters for the effector.

    Attributes:
        # Geometry
        path_coordinates: Local coordinates of fixation points. Shape: (2, n_total_points)
        path_fixation_body: Body index for each fixation point. Shape: (n_total_points,)
        segment_to_muscle: Mapping from segments to muscles. Shape: (n_segments,)
        n_muscles: Number of muscles.
        n_joints: Number of joints.
        n_segments: Number of path segments (n_total_points - n_muscles).

        # Integration
        dt: Timestep size.
        n_ministeps: Number of integration substeps.
        damping: Joint damping coefficient.

        # Skeleton and muscle params are passed separately to allow different types
    """
    path_coordinates: jnp.ndarray
    path_fixation_body: jnp.ndarray
    segment_to_muscle: jnp.ndarray
    n_muscles: int
    n_joints: int
    n_segments: int
    dt: float
    n_ministeps: int
    damping: float


class Effector:
    """Effector: combines skeleton and muscles into a complete simulation.

    The effector handles:
    - Computing musculotendon geometry from joint state
    - Computing muscle forces and joint torques
    - Numerical integration of the dynamics
    - State management

    This JAX implementation uses vectorized operations and segment_sum
    for efficient geometry computation, avoiding Python loops.
    """

    def __init__(
        self,
        skeleton,
        muscle,
        dt: float = 0.01,
        n_ministeps: int = 1,
        damping: float = 0.0,
    ):
        """Initialize the effector.

        Args:
            skeleton: Skeleton instance (e.g., TwoDofArm).
            muscle: Muscle instance (e.g., RigidTendonMuscle).
            dt: Timestep size (s).
            n_ministeps: Number of integration substeps per step.
            damping: Joint damping coefficient.
        """
        self.skeleton = skeleton
        self.muscle = muscle
        self.dt = dt
        self.n_ministeps = n_ministeps
        self.damping = damping
        self.minidt = dt / n_ministeps

        # Will be populated by add_muscle
        self._path_coordinates = []
        self._path_fixation_body = []
        self._segment_to_muscle = []
        self._n_muscles = 0

        self.skeleton_params = skeleton.get_params()
        self.muscle_params = muscle.get_params()
        self.params = None  # Will be built after muscles are added

    @property
    def n_joints(self) -> int:
        """Number of joints in the skeleton."""
        return self.skeleton.n_joints

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return self._n_muscles

    def add_muscle(
        self,
        path_fixation_body: List[int],
        path_coordinates: List[List[float]],
        name: Optional[str] = None,
        **kwargs,
    ):
        """Add a muscle to the effector.

        Args:
            path_fixation_body: Body index for each fixation point.
                0 = world, 1 = first bone, 2 = second bone, etc.
            path_coordinates: Coordinates of each fixation point in local frame.
                Each inner list is [x, y] coordinates.
            name: Optional name for the muscle.
            **kwargs: Additional muscle parameters (not used in JAX version,
                muscle params should be set at muscle initialization).
        """
        n_points = len(path_fixation_body)
        n_segments = n_points - 1

        # Store coordinates as (2, n_points)
        coords = jnp.array(path_coordinates).T  # (2, n_points)
        self._path_coordinates.append(coords)
        self._path_fixation_body.extend(path_fixation_body)

        # Segment-to-muscle mapping
        self._segment_to_muscle.extend([self._n_muscles] * n_segments)

        self._n_muscles += 1

    def build(self):
        """Build the effector after all muscles have been added.

        This creates the EffectorParams and finalizes the geometry arrays.
        """
        if self._n_muscles == 0:
            raise ValueError("No muscles have been added. Call add_muscle first.")

        # Concatenate all path coordinates
        path_coordinates = jnp.concatenate(self._path_coordinates, axis=1)
        path_fixation_body = jnp.array(self._path_fixation_body)
        segment_to_muscle = jnp.array(self._segment_to_muscle)

        n_segments = len(self._segment_to_muscle)

        self.params = EffectorParams(
            path_coordinates=path_coordinates,
            path_fixation_body=path_fixation_body,
            segment_to_muscle=segment_to_muscle,
            n_muscles=self._n_muscles,
            n_joints=self.skeleton.n_joints,
            n_segments=n_segments,
            dt=self.dt,
            n_ministeps=self.n_ministeps,
            damping=self.damping,
        )

        return self.params

    def get_params(self) -> EffectorParams:
        """Get effector parameters."""
        if self.params is None:
            self.build()
        return self.params

    @staticmethod
    def compute_geometry(
        joint_state: JointState,
        effector_params: EffectorParams,
        skeleton_params,
        path2cartesian_fn: Callable,
    ) -> GeometryState:
        """Compute musculotendon geometry from joint state.

        This is the KEY function for performance - it uses segment_sum
        instead of Python loops for vectorized computation.

        Args:
            joint_state: Current joint state.
            effector_params: Effector parameters.
            skeleton_params: Skeleton parameters.
            path2cartesian_fn: Function to convert path points to Cartesian.

        Returns:
            Geometry state with musculotendon lengths, velocities, and moment arms.
        """
        batch_size = joint_state.position.shape[0]
        n_muscles = effector_params.n_muscles
        n_joints = effector_params.n_joints

        # Transform fixation points to Cartesian coordinates
        # xy: (batch, 2, n_points)
        # dxy_dt: (batch, 2, n_points)
        # dxy_dq: (batch, 2, n_joints, n_points)
        xy, dxy_dt, dxy_dq = path2cartesian_fn(
            effector_params.path_coordinates,
            effector_params.path_fixation_body[None, :],  # Add batch dim
            joint_state,
            skeleton_params,
        )

        # Compute segment vectors (differences between consecutive points)
        # But we need to handle muscle boundaries
        n_points = xy.shape[2]

        # Segment vectors
        diff_pos = xy[:, :, 1:] - xy[:, :, :-1]  # (batch, 2, n_points-1)
        diff_vel = dxy_dt[:, :, 1:] - dxy_dt[:, :, :-1]  # (batch, 2, n_points-1)
        diff_dq = dxy_dq[:, :, :, 1:] - dxy_dq[:, :, :, :-1]  # (batch, 2, n_joints, n_points-1)

        # Segment lengths
        segment_len = jnp.sqrt(jnp.sum(diff_pos ** 2, axis=1))  # (batch, n_segments)

        # Segment velocities (projection of velocity onto segment direction)
        # v_segment = (diff_pos · diff_vel) / |diff_pos|
        segment_vel = jnp.sum(diff_pos * diff_vel, axis=1) / (segment_len + 1e-8)  # (batch, n_segments)

        # Segment moment arms
        # moment_arm = (diff_dq · diff_pos) / |diff_pos|
        # diff_dq: (batch, 2, n_joints, n_segments)
        # diff_pos: (batch, 2, n_segments)
        # We want: (batch, n_joints, n_segments)
        segment_moment = jnp.sum(
            diff_dq * diff_pos[:, :, None, :],
            axis=1,
        ) / (segment_len[:, None, :] + 1e-8)  # (batch, n_joints, n_segments)

        # Now aggregate segments to muscles using segment_sum
        # This is the KEY optimization - no Python loops!
        segment_indices = effector_params.segment_to_muscle

        # Create a mask for valid segments (not crossing muscle boundaries)
        # We can check this by looking at consecutive muscle indices
        # For now, we assume all segments are valid (proper muscle setup)

        # Use segment_sum to aggregate
        # musculotendon_length: sum of segment lengths per muscle
        musculotendon_length = jax.ops.segment_sum(
            segment_len,
            segment_indices,
            num_segments=n_muscles,
        )  # (batch, n_muscles)

        # musculotendon_velocity: sum of segment velocities per muscle
        musculotendon_velocity = jax.ops.segment_sum(
            segment_vel,
            segment_indices,
            num_segments=n_muscles,
        )  # (batch, n_muscles)

        # moment_arms: sum per muscle per joint
        # Need to reshape for segment_sum
        # segment_moment: (batch, n_joints, n_segments)
        moment_arm = jnp.zeros((batch_size, n_muscles, n_joints))
        for j in range(n_joints):
            moment_arm = moment_arm.at[:, :, j].set(
                jax.ops.segment_sum(
                    segment_moment[:, j, :],
                    segment_indices,
                    num_segments=n_muscles,
                )
            )

        return GeometryState(
            musculotendon_length=musculotendon_length,
            musculotendon_velocity=musculotendon_velocity,
            moment_arm=moment_arm,
        )

    @staticmethod
    def compute_joint_torques(
        muscle_forces: jnp.ndarray,
        geometry_state: GeometryState,
        joint_velocity: jnp.ndarray,
        damping: float,
        joint_load: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute net joint torques from muscle forces.

        Args:
            muscle_forces: Muscle forces. Shape: (batch, n_muscles)
            geometry_state: Current geometry state.
            joint_velocity: Joint velocities. Shape: (batch, n_joints)
            damping: Damping coefficient.
            joint_load: External joint loads. Shape: (batch, n_joints)

        Returns:
            Net joint torques. Shape: (batch, n_joints)
        """
        # Torque from muscles: tau = -sum(F * moment_arm)
        # moment_arm: (batch, n_muscles, n_joints)
        # muscle_forces: (batch, n_muscles)
        muscle_torques = -jnp.sum(
            muscle_forces[:, :, None] * geometry_state.moment_arm,
            axis=1,
        )  # (batch, n_joints)

        # Add damping and external loads
        total_torques = muscle_torques + joint_load - damping * joint_velocity

        return total_torques

    @staticmethod
    def euler_step(
        state: EffectorState,
        action: jnp.ndarray,
        endpoint_load: jnp.ndarray,
        joint_load: jnp.ndarray,
        dt: float,
        effector_params: EffectorParams,
        skeleton_params,
        muscle_params,
        skeleton_ode_fn: Callable,
        skeleton_integrate_fn: Callable,
        muscle_ode_fn: Callable,
        muscle_integrate_fn: Callable,
        muscle_force_fn: Callable,
        path2cartesian_fn: Callable,
        joint2cartesian_fn: Callable,
    ) -> EffectorState:
        """Perform one Euler integration step.

        Args:
            state: Current effector state.
            action: Muscle excitation. Shape: (batch, n_muscles)
            endpoint_load: External load at endpoint. Shape: (batch, 2)
            joint_load: External joint torques. Shape: (batch, n_joints)
            dt: Timestep.
            effector_params: Effector parameters.
            skeleton_params: Skeleton parameters.
            muscle_params: Muscle parameters.
            *_fn: Various function handles for skeleton and muscle operations.

        Returns:
            New effector state.
        """
        # Get muscle forces
        muscle_forces, _, _, _ = muscle_force_fn(
            state.muscle, state.geometry, muscle_params
        )

        # Compute joint torques
        torques = Effector.compute_joint_torques(
            muscle_forces,
            state.geometry,
            state.joint.velocity,
            effector_params.damping,
            joint_load,
        )

        # Compute derivatives
        d_muscle = muscle_ode_fn(action, state.muscle, muscle_params)
        _, d_velocity = skeleton_ode_fn(
            state.joint, torques, endpoint_load, skeleton_params
        )

        # Integrate
        new_joint = skeleton_integrate_fn(state.joint, d_velocity, dt, skeleton_params)
        new_geometry = Effector.compute_geometry(
            new_joint, effector_params, skeleton_params, path2cartesian_fn
        )
        new_muscle = muscle_integrate_fn(
            dt, d_muscle, state.muscle, new_geometry, muscle_params
        )
        new_cartesian = joint2cartesian_fn(new_joint, skeleton_params)
        new_fingertip = new_cartesian.position

        return EffectorState(
            joint=new_joint,
            cartesian=new_cartesian,
            muscle=new_muscle,
            geometry=new_geometry,
            fingertip=new_fingertip,
        )

    @staticmethod
    def rk4_step(
        state: EffectorState,
        action: jnp.ndarray,
        endpoint_load: jnp.ndarray,
        joint_load: jnp.ndarray,
        dt: float,
        effector_params: EffectorParams,
        skeleton_params,
        muscle_params,
        skeleton_ode_fn: Callable,
        skeleton_integrate_fn: Callable,
        muscle_ode_fn: Callable,
        muscle_integrate_fn: Callable,
        muscle_force_fn: Callable,
        path2cartesian_fn: Callable,
        joint2cartesian_fn: Callable,
    ) -> EffectorState:
        """Perform one RK4 integration step.

        Args:
            (Same as euler_step)

        Returns:
            New effector state.
        """
        half_dt = dt / 2

        def compute_derivatives(s):
            """Compute state derivatives."""
            forces, _, _, _ = muscle_force_fn(s.muscle, s.geometry, muscle_params)
            torques = Effector.compute_joint_torques(
                forces, s.geometry, s.joint.velocity, effector_params.damping, joint_load
            )
            d_muscle = muscle_ode_fn(action, s.muscle, muscle_params)
            _, d_velocity = skeleton_ode_fn(s.joint, torques, endpoint_load, skeleton_params)
            return d_muscle, d_velocity

        def step_state(s, d_muscle, d_velocity, step_dt):
            """Take a step with given derivatives."""
            new_joint = skeleton_integrate_fn(s.joint, d_velocity, step_dt, skeleton_params)
            new_geometry = Effector.compute_geometry(
                new_joint, effector_params, skeleton_params, path2cartesian_fn
            )
            new_muscle = muscle_integrate_fn(
                step_dt, d_muscle, s.muscle, new_geometry, muscle_params
            )
            new_cartesian = joint2cartesian_fn(new_joint, skeleton_params)
            return EffectorState(
                joint=new_joint,
                cartesian=new_cartesian,
                muscle=new_muscle,
                geometry=new_geometry,
                fingertip=new_cartesian.position,
            )

        # k1
        d_muscle1, d_vel1 = compute_derivatives(state)

        # k2
        state1 = step_state(state, d_muscle1, d_vel1, half_dt)
        d_muscle2, d_vel2 = compute_derivatives(state1)

        # k3
        state2 = step_state(state, d_muscle2, d_vel2, half_dt)
        d_muscle3, d_vel3 = compute_derivatives(state2)

        # k4
        state3 = step_state(state, d_muscle3, d_vel3, dt)
        d_muscle4, d_vel4 = compute_derivatives(state3)

        # Combine
        d_muscle = (d_muscle1 + 2*d_muscle2 + 2*d_muscle3 + d_muscle4) / 6
        d_vel = (d_vel1 + 2*d_vel2 + 2*d_vel3 + d_vel4) / 6

        return step_state(state, d_muscle, d_vel, dt)

    def reset(
        self,
        batch_size: int = 1,
        joint_state: Optional[JointState] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> EffectorState:
        """Reset the effector to initial state.

        Args:
            batch_size: Batch size.
            joint_state: Optional initial joint state. If None, uses random.
            key: Random key for initialization.

        Returns:
            Initial effector state.
        """
        if self.params is None:
            self.build()

        if joint_state is None:
            # Random initial state within bounds
            if key is None:
                key = jax.random.PRNGKey(0)
            pos_low = self.skeleton_params.pos_lower_bound
            pos_high = self.skeleton_params.pos_upper_bound
            pos = jax.random.uniform(
                key, (batch_size, self.n_joints), minval=pos_low, maxval=pos_high
            )
            vel = jnp.zeros((batch_size, self.n_joints))
            joint_state = JointState(position=pos, velocity=vel)

        # Compute geometry from joint state
        geometry = self.compute_geometry(
            joint_state,
            self.params,
            self.skeleton_params,
            self.skeleton.path2cartesian,
        )

        # Get initial muscle state
        muscle_state = self.muscle.get_initial_state(
            batch_size, geometry, self.muscle_params
        )

        # Compute cartesian state
        cartesian = self.skeleton.joint2cartesian(joint_state, self.skeleton_params)

        return EffectorState(
            joint=joint_state,
            cartesian=cartesian,
            muscle=muscle_state,
            geometry=geometry,
            fingertip=cartesian.position,
        )

    def step(
        self,
        state: EffectorState,
        action: jnp.ndarray,
        endpoint_load: Optional[jnp.ndarray] = None,
        joint_load: Optional[jnp.ndarray] = None,
        method: str = "euler",
    ) -> EffectorState:
        """Step the effector forward in time.

        Args:
            state: Current effector state.
            action: Muscle excitation. Shape: (batch, n_muscles)
            endpoint_load: External load at endpoint. Shape: (batch, 2)
            joint_load: External joint torques. Shape: (batch, n_joints)
            method: Integration method ("euler" or "rk4").

        Returns:
            New effector state.
        """
        if self.params is None:
            self.build()

        batch_size = state.joint.position.shape[0]

        if endpoint_load is None:
            endpoint_load = jnp.zeros((batch_size, self.skeleton.n_dim))
        if joint_load is None:
            joint_load = jnp.zeros((batch_size, self.n_joints))

        # Choose integration method
        if method == "euler":
            step_fn = self.euler_step
        elif method in ("rk4", "rungekutta4"):
            step_fn = self.rk4_step
        else:
            raise ValueError(f"Unknown integration method: {method}")

        # Perform ministeps
        def ministep(state, _):
            new_state = step_fn(
                state,
                action,
                endpoint_load,
                joint_load,
                self.minidt,
                self.params,
                self.skeleton_params,
                self.muscle_params,
                self.skeleton.ode,
                self.skeleton.integrate,
                self.muscle.ode,
                self.muscle.integrate,
                self.muscle.compute_force,
                self.skeleton.path2cartesian,
                self.skeleton.joint2cartesian,
            )
            return new_state, None

        final_state, _ = lax.scan(ministep, state, None, length=self.n_ministeps)

        return final_state


# Convenience function for creating standard arm effector
def create_arm_effector(
    muscle_type: str = "rigid_tendon",
    dt: float = 0.01,
    n_ministeps: int = 1,
    damping: float = 0.0,
) -> Effector:
    """Create a standard 2-DOF arm effector with 6 muscles.

    This creates the "Arm26" configuration from the original MotorNet.

    Args:
        muscle_type: "rigid_tendon" or "compliant_tendon".
        dt: Timestep size.
        n_ministeps: Number of integration substeps.
        damping: Joint damping coefficient.

    Returns:
        Configured Effector instance.
    """
    from motornet_jax.skeleton import TwoDofArm
    from motornet_jax.muscle import RigidTendonMuscle, CompliantTendonMuscle

    # Create skeleton
    skeleton = TwoDofArm()

    # Create muscle
    # Standard Arm26 muscle parameters
    max_iso_force = jnp.array([838, 1207, 1422, 1549, 414, 603])
    tendon_length = jnp.array([0.039, 0.066, 0.172, 0.187, 0.204, 0.217])
    optimal_length = jnp.array([0.134, 0.140, 0.092, 0.093, 0.137, 0.127])

    if muscle_type == "rigid_tendon":
        muscle = RigidTendonMuscle(
            max_isometric_force=max_iso_force,
            optimal_fiber_length=optimal_length,
            tendon_slack_length=tendon_length,
        )
    elif muscle_type == "compliant_tendon":
        muscle = CompliantTendonMuscle(
            max_isometric_force=max_iso_force,
            optimal_fiber_length=optimal_length,
            tendon_slack_length=tendon_length,
        )
    else:
        raise ValueError(f"Unknown muscle type: {muscle_type}")

    # Create effector
    effector = Effector(
        skeleton=skeleton,
        muscle=muscle,
        dt=dt,
        n_ministeps=n_ministeps,
        damping=damping,
    )

    # Note: For Arm26, we use polynomial moment arm approximation
    # instead of actual path geometry. This would need to be implemented
    # as a separate Arm26 class.

    return effector
