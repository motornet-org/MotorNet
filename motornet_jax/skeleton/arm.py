"""
Two degrees-of-freedom planar arm skeleton.

This implements a 2-DOF arm with shoulder and elbow joints,
matching the dynamics of the PyTorch MotorNet implementation.
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from motornet_jax.types import JointState, CartesianState


class TwoDofArmParams(NamedTuple):
    """Parameters for the two-DOF arm skeleton.

    Attributes:
        L1: Length of the first segment (upper arm) in meters.
        L2: Length of the second segment (forearm) in meters.
        M1: Mass of the first segment in kg.
        M2: Mass of the second segment in kg.
        L1g: Distance to center of gravity of first segment.
        L2g: Distance to center of gravity of second segment.
        I1: Moment of inertia of first segment.
        I2: Moment of inertia of second segment.
        inertia_c: Constant part of inertia matrix. Shape: (2, 2)
        inertia_m: Multiplier for cos(q2) term. Shape: (2, 2)
        coriolis_1: First Coriolis coefficient.
        coriolis_2: Second Coriolis coefficient.
        viscosity: Viscous damping coefficient.
        pos_lower_bound: Lower position bounds. Shape: (2,)
        pos_upper_bound: Upper position bounds. Shape: (2,)
        vel_lower_bound: Lower velocity bounds. Shape: (2,)
        vel_upper_bound: Upper velocity bounds. Shape: (2,)
    """
    L1: float
    L2: float
    M1: float
    M2: float
    L1g: float
    L2g: float
    I1: float
    I2: float
    inertia_c: jnp.ndarray
    inertia_m: jnp.ndarray
    coriolis_1: float
    coriolis_2: float
    viscosity: float
    pos_lower_bound: jnp.ndarray
    pos_upper_bound: jnp.ndarray
    vel_lower_bound: jnp.ndarray
    vel_upper_bound: jnp.ndarray


class TwoDofArm:
    """Two degrees-of-freedom planar arm.

    This implements a planar arm with shoulder and elbow joints.
    The shoulder is at the origin, the elbow is at distance L1,
    and the hand (endpoint) is at distance L1 + L2.

    Joint 0: Shoulder angle (from positive x-axis)
    Joint 1: Elbow angle (relative to upper arm)
    """

    n_joints = 2
    n_dim = 2

    def __init__(
        self,
        m1: float = 1.864572,
        m2: float = 1.534315,
        l1g: float = 0.180496,
        l2g: float = 0.181479,
        i1: float = 0.013193,
        i2: float = 0.020062,
        l1: float = 0.309,
        l2: float = 0.26,
        viscosity: float = 0.0,
        pos_lower_bound: Optional[Tuple[float, float]] = None,
        pos_upper_bound: Optional[Tuple[float, float]] = None,
        vel_lower_bound: float = -1000.0,
        vel_upper_bound: float = 1000.0,
    ):
        """Initialize two-DOF arm.

        Args:
            m1: Mass of upper arm (kg).
            m2: Mass of forearm (kg).
            l1g: Distance to center of gravity of upper arm (m).
            l2g: Distance to center of gravity of forearm (m).
            i1: Moment of inertia of upper arm (kg.m^2).
            i2: Moment of inertia of forearm (kg.m^2).
            l1: Length of upper arm (m).
            l2: Length of forearm (m).
            viscosity: Viscous damping coefficient.
            pos_lower_bound: Lower joint angle bounds (radians). Default: [0, 0]
            pos_upper_bound: Upper joint angle bounds (radians). Default: [140°, 160°]
            vel_lower_bound: Lower velocity bound (rad/s).
            vel_upper_bound: Upper velocity bound (rad/s).
        """
        # Default joint limits (matching PyTorch implementation)
        if pos_lower_bound is None:
            pos_lower_bound = (jnp.deg2rad(0.0), jnp.deg2rad(0.0))
        if pos_upper_bound is None:
            pos_upper_bound = (jnp.deg2rad(140.0), jnp.deg2rad(160.0))

        # Pre-compute inertia matrix components
        inertia_11_c = m1 * l1g**2 + i1 + m2 * (l2g**2 + l1**2) + i2
        inertia_12_c = m2 * l2g**2 + i2
        inertia_22_c = m2 * l2g**2 + i2
        inertia_11_m = 2 * m2 * l1 * l2g
        inertia_12_m = m2 * l1 * l2g

        inertia_c = jnp.array([
            [inertia_11_c, inertia_12_c],
            [inertia_12_c, inertia_22_c],
        ])
        inertia_m = jnp.array([
            [inertia_11_m, inertia_12_m],
            [inertia_12_m, 0.0],
        ])

        # Coriolis coefficients
        coriolis_1 = -m2 * l1 * l2g
        coriolis_2 = m2 * l1 * l2g

        self.params = TwoDofArmParams(
            L1=l1,
            L2=l2,
            M1=m1,
            M2=m2,
            L1g=l1g,
            L2g=l2g,
            I1=i1,
            I2=i2,
            inertia_c=inertia_c,
            inertia_m=inertia_m,
            coriolis_1=coriolis_1,
            coriolis_2=coriolis_2,
            viscosity=viscosity,
            pos_lower_bound=jnp.array(pos_lower_bound),
            pos_upper_bound=jnp.array(pos_upper_bound),
            vel_lower_bound=jnp.array([vel_lower_bound, vel_lower_bound]),
            vel_upper_bound=jnp.array([vel_upper_bound, vel_upper_bound]),
        )

    def get_params(self) -> TwoDofArmParams:
        """Return the parameters for JIT-compiled functions."""
        return self.params

    @staticmethod
    @jit
    def forward_kinematics(
        joint_state: JointState,
        params: TwoDofArmParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward kinematics: joint angles to Cartesian positions.

        Args:
            joint_state: Joint state (position, velocity)
            params: Arm parameters

        Returns:
            elbow_pos: Elbow position. Shape: (batch, 2)
            hand_pos: Hand/endpoint position. Shape: (batch, 2)
            elbow_vel: Elbow velocity. Shape: (batch, 2)
            hand_vel: Hand/endpoint velocity. Shape: (batch, 2)
        """
        pos = joint_state.position
        vel = joint_state.velocity

        sho = pos[:, 0]  # shoulder angle
        elb = pos[:, 1]  # elbow angle (relative to upper arm)
        sho_vel = vel[:, 0]
        elb_vel = vel[:, 1]

        # Absolute angle of forearm
        pos_sum = sho + elb

        c1 = jnp.cos(sho)
        s1 = jnp.sin(sho)
        c12 = jnp.cos(pos_sum)
        s12 = jnp.sin(pos_sum)

        # Elbow position
        elbow_x = params.L1 * c1
        elbow_y = params.L1 * s1
        elbow_pos = jnp.stack([elbow_x, elbow_y], axis=-1)

        # Hand position
        hand_x = elbow_x + params.L2 * c12
        hand_y = elbow_y + params.L2 * s12
        hand_pos = jnp.stack([hand_x, hand_y], axis=-1)

        # Elbow velocity
        elbow_vel_x = -params.L1 * s1 * sho_vel
        elbow_vel_y = params.L1 * c1 * sho_vel
        elbow_vel = jnp.stack([elbow_vel_x, elbow_vel_y], axis=-1)

        # Hand velocity (using chain rule)
        total_vel = sho_vel + elb_vel
        hand_vel_x = elbow_vel_x - params.L2 * s12 * total_vel
        hand_vel_y = elbow_vel_y + params.L2 * c12 * total_vel
        hand_vel = jnp.stack([hand_vel_x, hand_vel_y], axis=-1)

        return elbow_pos, hand_pos, elbow_vel, hand_vel

    @staticmethod
    @jit
    def joint2cartesian(
        joint_state: JointState,
        params: TwoDofArmParams,
    ) -> CartesianState:
        """Convert joint state to Cartesian (endpoint) state.

        Args:
            joint_state: Joint state
            params: Arm parameters

        Returns:
            Cartesian state of the endpoint (hand)
        """
        _, hand_pos, _, hand_vel = TwoDofArm.forward_kinematics(joint_state, params)
        return CartesianState(position=hand_pos, velocity=hand_vel)

    @staticmethod
    @jit
    def compute_jacobian(
        joint_state: JointState,
        params: TwoDofArmParams,
    ) -> jnp.ndarray:
        """Compute the Jacobian matrix (dCartesian/dJoint).

        Args:
            joint_state: Joint state
            params: Arm parameters

        Returns:
            Jacobian matrix. Shape: (batch, 2, 2)
        """
        pos = joint_state.position
        sho = pos[:, 0]
        elb = pos[:, 1]
        pos_sum = sho + elb

        s1 = jnp.sin(sho)
        c1 = jnp.cos(sho)
        s12 = jnp.sin(pos_sum)
        c12 = jnp.cos(pos_sum)

        # Jacobian entries
        j11 = -params.L1 * s1 - params.L2 * s12
        j12 = -params.L2 * s12
        j21 = params.L1 * c1 + params.L2 * c12
        j22 = params.L2 * c12

        jacobian = jnp.stack([
            jnp.stack([j11, j12], axis=-1),
            jnp.stack([j21, j22], axis=-1),
        ], axis=-2)

        return jacobian

    @staticmethod
    @jit
    def inverse_dynamics(
        joint_state: JointState,
        torques: jnp.ndarray,
        endpoint_load: jnp.ndarray,
        params: TwoDofArmParams,
    ) -> jnp.ndarray:
        """Compute joint accelerations from applied torques and endpoint loads.

        Uses the equations of motion: M(q) * qdd + C(q, qd) = tau + J^T * F

        Args:
            joint_state: Current joint state
            torques: Applied joint torques. Shape: (batch, 2)
            endpoint_load: External force at endpoint. Shape: (batch, 2)
            params: Arm parameters

        Returns:
            Joint accelerations. Shape: (batch, 2)
        """
        pos = joint_state.position
        vel = joint_state.velocity

        pos0, pos1 = pos[:, 0], pos[:, 1]
        vel0, vel1 = vel[:, 0], vel[:, 1]
        pos_sum = pos0 + pos1

        c1 = jnp.cos(pos0)
        c2 = jnp.cos(pos1)
        s1 = jnp.sin(pos0)
        s2 = jnp.sin(pos1)
        c12 = jnp.cos(pos_sum)
        s12 = jnp.sin(pos_sum)

        # Inertia matrix: M(q) = M_c + cos(q2) * M_m
        # Shape: (batch, 2, 2)
        inertia = params.inertia_c + c2[:, None, None] * params.inertia_m

        # Coriolis/centrifugal terms + viscous damping
        coriolis_1 = params.coriolis_1 * s2 * (2 * vel0 + vel1) * vel1 + params.viscosity * vel0
        coriolis_2 = params.coriolis_2 * s2 * vel0 * vel0 + params.viscosity * vel1
        coriolis = jnp.stack([coriolis_1, coriolis_2], axis=-1)

        # Jacobian for endpoint load transformation
        j11 = -params.L1 * s1 - params.L2 * s12
        j12 = -params.L2 * s12
        j21 = params.L1 * c1 + params.L2 * c12
        j22 = params.L2 * c12

        # Transform endpoint load to joint torques: tau_ext = J^T @ F
        tau_ext_0 = j11 * endpoint_load[:, 0] + j21 * endpoint_load[:, 1]
        tau_ext_1 = j12 * endpoint_load[:, 0] + j22 * endpoint_load[:, 1]

        # Total torque
        total_torque = torques + jnp.stack([tau_ext_0, tau_ext_1], axis=-1)

        # Right-hand side of dynamics equation
        rhs = total_torque - coriolis

        # Solve M @ acc = rhs using explicit 2x2 inverse
        # For numerical stability with batched small matrices
        det = inertia[:, 0, 0] * inertia[:, 1, 1] - inertia[:, 0, 1] * inertia[:, 1, 0]
        inv_det = 1.0 / det

        # Inverse of 2x2 matrix
        inv_00 = inv_det * inertia[:, 1, 1]
        inv_01 = -inv_det * inertia[:, 0, 1]
        inv_10 = -inv_det * inertia[:, 1, 0]
        inv_11 = inv_det * inertia[:, 0, 0]

        # acc = M^{-1} @ rhs
        acc_0 = inv_00 * rhs[:, 0] + inv_01 * rhs[:, 1]
        acc_1 = inv_10 * rhs[:, 0] + inv_11 * rhs[:, 1]

        return jnp.stack([acc_0, acc_1], axis=-1)

    @staticmethod
    @jit
    def ode(
        joint_state: JointState,
        torques: jnp.ndarray,
        endpoint_load: jnp.ndarray,
        params: TwoDofArmParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute state derivatives for integration.

        Args:
            joint_state: Current joint state
            torques: Applied joint torques. Shape: (batch, 2)
            endpoint_load: External force at endpoint. Shape: (batch, 2)
            params: Arm parameters

        Returns:
            dposition_dt: Velocity. Shape: (batch, 2)
            dvelocity_dt: Acceleration. Shape: (batch, 2)
        """
        acceleration = TwoDofArm.inverse_dynamics(
            joint_state, torques, endpoint_load, params
        )
        return joint_state.velocity, acceleration

    @staticmethod
    @jit
    def clip_velocity(
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        params: TwoDofArmParams,
    ) -> jnp.ndarray:
        """Clip velocities based on bounds and position limits.

        If position is at a boundary, velocity pushing past it is set to zero.

        Args:
            pos: Joint positions. Shape: (batch, 2)
            vel: Joint velocities. Shape: (batch, 2)
            params: Arm parameters

        Returns:
            Clipped velocities. Shape: (batch, 2)
        """
        # First clip to velocity bounds
        vel = jnp.clip(vel, params.vel_lower_bound, params.vel_upper_bound)

        # If at lower bound and velocity is negative, set to zero
        vel = jnp.where(
            jnp.logical_and(vel < 0, pos <= params.pos_lower_bound),
            0.0,
            vel,
        )

        # If at upper bound and velocity is positive, set to zero
        vel = jnp.where(
            jnp.logical_and(vel > 0, pos >= params.pos_upper_bound),
            0.0,
            vel,
        )

        return vel

    @staticmethod
    @jit
    def integrate(
        joint_state: JointState,
        acceleration: jnp.ndarray,
        dt: float,
        params: TwoDofArmParams,
    ) -> JointState:
        """Integrate state forward in time using Euler method.

        Args:
            joint_state: Current joint state
            acceleration: Joint accelerations. Shape: (batch, 2)
            dt: Time step
            params: Arm parameters

        Returns:
            New joint state after integration
        """
        new_vel = joint_state.velocity + dt * acceleration
        new_pos = joint_state.position + dt * joint_state.velocity

        # Clip velocity first (considering new position)
        new_vel = TwoDofArm.clip_velocity(new_pos, new_vel, params)

        # Then clip position
        new_pos = jnp.clip(new_pos, params.pos_lower_bound, params.pos_upper_bound)

        return JointState(position=new_pos, velocity=new_vel)

    @staticmethod
    @jit
    def path2cartesian(
        path_coordinates: jnp.ndarray,
        path_fixation_body: jnp.ndarray,
        joint_state: JointState,
        params: TwoDofArmParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Transform muscle fixation points to Cartesian coordinates.

        This converts local bone-relative coordinates to global Cartesian
        coordinates, also computing velocities and Jacobians for moment arm
        calculation.

        Args:
            path_coordinates: Local coordinates of fixation points. Shape: (2, n_points)
            path_fixation_body: Body index for each point (0=world, 1=upper arm, 2=forearm).
                Shape: (1, n_points)
            joint_state: Current joint state
            params: Arm parameters

        Returns:
            xy: Cartesian positions. Shape: (batch, 2, n_points)
            dxy_dt: Cartesian velocities. Shape: (batch, 2, n_points)
            dxy_dq: Jacobian w.r.t. joint angles. Shape: (batch, 2, 2, n_points)
        """
        n_points = path_fixation_body.shape[1]
        batch_size = joint_state.position.shape[0]

        pos = joint_state.position
        vel = joint_state.velocity

        sho = pos[:, 0:1]  # (batch, 1)
        elb_wrt_sho = pos[:, 1:2]  # (batch, 1)
        elb = elb_wrt_sho + sho  # Absolute elbow angle

        sho_vel = vel[:, 0:1]  # (batch, 1)
        elb_vel = vel[:, 1:2] + sho_vel  # Absolute elbow angular velocity

        # Elbow position in world coordinates
        elb_x = params.L1 * jnp.cos(sho)  # (batch, 1)
        elb_y = params.L1 * jnp.sin(sho)  # (batch, 1)

        # Flatten fixation body for indexing
        flat_body = path_fixation_body.reshape(-1)  # (n_points,)

        # Rotation angle based on which body the point is fixed to
        # body=0: world (no rotation), body=1: upper arm, body=2: forearm
        ang = jnp.where(
            flat_body == 0,
            jnp.zeros((batch_size, n_points)),
            jnp.where(
                flat_body == 1,
                -sho,  # Upper arm uses negative shoulder angle
                -elb,  # Forearm uses negative absolute elbow angle
            ),
        )  # (batch, n_points)

        ca = jnp.cos(ang)  # (batch, n_points)
        sa = jnp.sin(ang)  # (batch, n_points)

        # Rotation matrix components
        # rot1 = [ca, sa], rot2 = [-sa, ca]
        # Path coordinates shape: (2, n_points)
        # We want to rotate each point

        # Rotated coordinates (this is the position relative to bone origin)
        # For a point (x, y), rotation by angle a gives:
        # x' = x*cos(a) - y*sin(a)
        # y' = x*sin(a) + y*cos(a)
        # But we're using negative angles, so:
        # x' = x*cos(-a) - y*sin(-a) = x*ca + y*sa
        # y' = x*sin(-a) + y*cos(-a) = -x*sa + y*ca

        px = path_coordinates[0:1, :]  # (1, n_points)
        py = path_coordinates[1:2, :]  # (1, n_points)

        # Derivative of position w.r.t. the angle of the bone they're fixed on
        # d/da (x*cos(a) - y*sin(a)) = -x*sin(a) - y*cos(a)
        # d/da (x*sin(a) + y*cos(a)) = x*cos(a) - y*sin(a)
        # With negative angle:
        dx_da = -(px * (-sa) + py * (-ca))  # = px*sa + py*ca
        dy_da = px * ca + py * (-sa)  # = px*ca - py*sa

        # But looking at original code more carefully:
        # rot1 = [ca, sa], rot2 = [-sa, ca]
        # dx_da = sum(-path_coordinates * rot2) = -px*(-sa) - py*ca = px*sa - py*ca
        # dy_da = sum(path_coordinates * rot1) = px*ca + py*sa

        # Let me match the PyTorch implementation exactly:
        # rot1 has shape (batch, 2, n_points) with rot1[:, 0, :] = ca, rot1[:, 1, :] = sa
        # rot2 has shape (batch, 2, n_points) with rot2[:, 0, :] = -sa, rot2[:, 1, :] = ca
        # dx_da = sum(-path_coordinates * rot2, dim=1) = -(px * (-sa) + py * ca) = px*sa - py*ca
        # dy_da = sum(path_coordinates * rot1, dim=1) = px*ca + py*sa

        dx_da = px * sa - py * ca  # (batch, n_points)
        dy_da = px * ca + py * sa  # (batch, n_points)

        # Derivative w.r.t. shoulder angle (da1)
        # All attached points have some derivative w.r.t. shoulder
        # For forearm points, add the effect of elbow movement
        dx_da1 = jnp.where(flat_body == 0, 0.0, dx_da) + jnp.where(flat_body == 2, -elb_y, 0.0)
        dy_da1 = jnp.where(flat_body == 0, 0.0, dy_da) + jnp.where(flat_body == 2, elb_x, 0.0)

        # Derivative w.r.t. elbow angle (da2)
        # Only forearm points have derivative w.r.t. elbow
        dx_da2 = jnp.where(flat_body == 2, dx_da, 0.0)
        dy_da2 = jnp.where(flat_body == 2, dy_da, 0.0)

        # Stack into Jacobian: (batch, 2, 2, n_points)
        dxy_dq = jnp.stack([
            jnp.stack([dx_da1, dx_da2], axis=1),  # (batch, 2, n_points) for x
            jnp.stack([dy_da1, dy_da2], axis=1),  # (batch, 2, n_points) for y
        ], axis=1)  # (batch, 2, 2, n_points)

        # Velocity using chain rule: dxy/dt = dxy/da1 * da1/dt + dxy/da2 * da2/dt
        dxy_dt = jnp.stack([
            dx_da1 * sho_vel + dx_da2 * elb_vel,
            dy_da1 * sho_vel + dy_da2 * elb_vel,
        ], axis=1)  # (batch, 2, n_points)

        # Position: xy = [dy_da, -dx_da] + bone_origin
        # This is the position formula from the original code
        # The rotation gives us position relative to bone origin
        bone_origin_x = jnp.where(flat_body == 2, elb_x, 0.0)
        bone_origin_y = jnp.where(flat_body == 2, elb_y, 0.0)

        xy = jnp.stack([
            dy_da + bone_origin_x,
            -dx_da + bone_origin_y,
        ], axis=1)  # (batch, 2, n_points)

        return xy, dxy_dt, dxy_dq
