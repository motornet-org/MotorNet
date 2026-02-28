"""
Point mass skeleton model.

A simple skeleton consisting of a point mass that can move in 2D space.
This is useful for testing and as a baseline model.
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp

from motornet_jax.types import JointState, CartesianState


class PointMassParams(NamedTuple):
    """Parameters for the point mass skeleton.

    Attributes:
        mass: Mass of the point. Default: 1.0
        viscosity: Viscous damping coefficient. Default: 0.0
        space_dim: Dimensionality of the space (2 or 3). Default: 2
        pos_lower_bound: Lower position bounds. Shape: (space_dim,)
        pos_upper_bound: Upper position bounds. Shape: (space_dim,)
        vel_lower_bound: Lower velocity bounds. Shape: (space_dim,)
        vel_upper_bound: Upper velocity bounds. Shape: (space_dim,)
    """
    mass: float = 1.0
    viscosity: float = 0.0
    space_dim: int = 2
    pos_lower_bound: Optional[jnp.ndarray] = None
    pos_upper_bound: Optional[jnp.ndarray] = None
    vel_lower_bound: Optional[jnp.ndarray] = None
    vel_upper_bound: Optional[jnp.ndarray] = None


class PointMass:
    """Point mass skeleton.

    For a point mass, joint space and Cartesian space are equivalent.
    The "joint" state is actually the Cartesian position and velocity.
    """

    def __init__(
        self,
        mass: float = 1.0,
        viscosity: float = 0.0,
        space_dim: int = 2,
        pos_upper_bound: float = 1.0,
        pos_lower_bound: float = 0.0,
        vel_upper_bound: float = jnp.inf,
        vel_lower_bound: float = -jnp.inf,
    ):
        """Initialize point mass skeleton.

        Args:
            mass: Mass of the point.
            viscosity: Viscous damping coefficient.
            space_dim: Dimensionality of space (2 or 3).
            pos_upper_bound: Upper position bound (scalar or array).
            pos_lower_bound: Lower position bound (scalar or array).
            vel_upper_bound: Upper velocity bound (scalar or array).
            vel_lower_bound: Lower velocity bound (scalar or array).
        """
        self.n_joints = space_dim
        self.n_dim = space_dim

        # Convert bounds to arrays
        pos_upper = jnp.ones(space_dim) * pos_upper_bound
        pos_lower = jnp.ones(space_dim) * pos_lower_bound
        vel_upper = jnp.ones(space_dim) * vel_upper_bound
        vel_lower = jnp.ones(space_dim) * vel_lower_bound

        self.params = PointMassParams(
            mass=mass,
            viscosity=viscosity,
            space_dim=space_dim,
            pos_lower_bound=pos_lower,
            pos_upper_bound=pos_upper,
            vel_lower_bound=vel_lower,
            vel_upper_bound=vel_upper,
        )

    def get_params(self) -> PointMassParams:
        """Return the parameters for JIT-compiled functions."""
        return self.params

    @staticmethod
    def forward_kinematics(
        joint_state: JointState,
        params: PointMassParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward kinematics for point mass.

        For a point mass, this is just the identity mapping.

        Args:
            joint_state: Joint state (position, velocity)
            params: Point mass parameters

        Returns:
            endpoint_position: Same as joint position. Shape: (batch, space_dim)
            endpoint_velocity: Same as joint velocity. Shape: (batch, space_dim)
        """
        return joint_state.position, joint_state.velocity

    @staticmethod
    def inverse_kinematics(
        cartesian_state: CartesianState,
        params: PointMassParams,
    ) -> JointState:
        """Inverse kinematics for point mass.

        For a point mass, this is just the identity mapping.

        Args:
            cartesian_state: Cartesian state (position, velocity)
            params: Point mass parameters

        Returns:
            Joint state (same as cartesian state)
        """
        return JointState(
            position=cartesian_state.position,
            velocity=cartesian_state.velocity,
        )

    @staticmethod
    def compute_jacobian(
        joint_state: JointState,
        params: PointMassParams,
    ) -> jnp.ndarray:
        """Compute the Jacobian matrix.

        For a point mass, the Jacobian is the identity matrix.

        Args:
            joint_state: Joint state
            params: Point mass parameters

        Returns:
            Jacobian matrix. Shape: (batch, space_dim, space_dim)
        """
        batch_size = joint_state.position.shape[0]
        return jnp.tile(jnp.eye(params.space_dim), (batch_size, 1, 1))

    @staticmethod
    def inverse_dynamics(
        joint_state: JointState,
        forces: jnp.ndarray,
        params: PointMassParams,
    ) -> jnp.ndarray:
        """Compute joint accelerations from forces.

        For a point mass: a = (F - viscosity * v) / m

        Args:
            joint_state: Current joint state
            forces: Applied forces. Shape: (batch, space_dim)
            params: Point mass parameters

        Returns:
            Joint accelerations. Shape: (batch, space_dim)
        """
        viscous_force = params.viscosity * joint_state.velocity
        net_force = forces - viscous_force
        acceleration = net_force / params.mass
        return acceleration

    @staticmethod
    def ode(
        joint_state: JointState,
        forces: jnp.ndarray,
        params: PointMassParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute state derivatives for integration.

        Args:
            joint_state: Current joint state
            forces: Applied forces. Shape: (batch, space_dim)
            params: Point mass parameters

        Returns:
            dposition_dt: Velocity. Shape: (batch, space_dim)
            dvelocity_dt: Acceleration. Shape: (batch, space_dim)
        """
        acceleration = PointMass.inverse_dynamics(joint_state, forces, params)
        return joint_state.velocity, acceleration

    @staticmethod
    def integrate(
        joint_state: JointState,
        forces: jnp.ndarray,
        dt: float,
        params: PointMassParams,
    ) -> JointState:
        """Integrate state forward in time using Euler method.

        Args:
            joint_state: Current joint state
            forces: Applied forces. Shape: (batch, space_dim)
            dt: Time step
            params: Point mass parameters

        Returns:
            New joint state after integration
        """
        dpos, dvel = PointMass.ode(joint_state, forces, params)

        new_pos = joint_state.position + dt * dpos
        new_vel = joint_state.velocity + dt * dvel

        # Apply bounds
        if params.pos_lower_bound is not None:
            new_pos = jnp.clip(new_pos, params.pos_lower_bound, params.pos_upper_bound)
        if params.vel_lower_bound is not None:
            new_vel = jnp.clip(new_vel, params.vel_lower_bound, params.vel_upper_bound)

        return JointState(position=new_pos, velocity=new_vel)

    @staticmethod
    def path2cartesian(
        path: jnp.ndarray,
        joint_state: JointState,
        params: PointMassParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Convert path points to Cartesian coordinates.

        For point mass, path points are already in world coordinates relative to origin.

        Args:
            path: Path fixation points. Shape: (n_points, 3) where columns are (body_id, x, y)
            joint_state: Current joint state
            params: Point mass parameters

        Returns:
            xy: Cartesian positions of path points. Shape: (batch, n_points, 2)
            dxy_dt: Velocity of path points. Shape: (batch, n_points, 2)
            dxy_dq: Jacobian of path points w.r.t. joint positions. Shape: (batch, n_points, 2, n_joints)
        """
        batch_size = joint_state.position.shape[0]
        n_points = path.shape[0]

        # For point mass, fixation points are offset from the point mass position
        # path[:, 0] is body_id (0 = world, 1 = point mass)
        # path[:, 1:3] is local (x, y) offset

        body_ids = path[:, 0].astype(jnp.int32)
        local_coords = path[:, 1:3]  # (n_points, 2)

        # World coordinates: if body=0, use local coords; if body=1, add point mass position
        is_attached = (body_ids == 1).astype(jnp.float32)  # (n_points,)

        # Broadcast for batch
        # xy = local_coords + is_attached[:, None] * joint_state.position[:, None, :]
        # Shape: (batch, n_points, 2)
        xy = local_coords[None, :, :] + is_attached[None, :, None] * joint_state.position[:, None, :]

        # Velocity: only attached points move
        dxy_dt = is_attached[None, :, None] * joint_state.velocity[:, None, :]

        # Jacobian: derivative of xy w.r.t. joint position
        # For point mass, dxy_dq[i, j] = is_attached[i] * I
        # Shape: (batch, n_points, 2, n_joints)
        dxy_dq = jnp.zeros((batch_size, n_points, 2, params.space_dim))
        for i in range(params.space_dim):
            dxy_dq = dxy_dq.at[:, :, i, i].set(is_attached)

        return xy, dxy_dt, dxy_dq
