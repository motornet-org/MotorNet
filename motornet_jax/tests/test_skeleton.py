"""
Tests for skeleton models.
"""

import jax
import jax.numpy as jnp
import pytest

from motornet_jax.skeleton import TwoDofArm, PointMass
from motornet_jax.types import JointState


class TestTwoDofArm:
    """Tests for the TwoDofArm skeleton."""

    @pytest.fixture
    def arm(self):
        """Create a standard arm for testing."""
        return TwoDofArm()

    @pytest.fixture
    def params(self, arm):
        """Get arm parameters."""
        return arm.get_params()

    def test_initialization(self, arm, params):
        """Test that arm initializes correctly."""
        assert arm.n_joints == 2
        assert arm.n_dim == 2
        assert params.L1 > 0
        assert params.L2 > 0

    def test_forward_kinematics_zero_angles(self, arm, params):
        """Test forward kinematics at zero position."""
        joint_state = JointState(
            position=jnp.array([[0.0, 0.0]]),
            velocity=jnp.array([[0.0, 0.0]]),
        )

        elbow_pos, hand_pos, elbow_vel, hand_vel = TwoDofArm.forward_kinematics(
            joint_state, params
        )

        # At zero angles, elbow should be at (L1, 0)
        assert jnp.allclose(elbow_pos[0, 0], params.L1, atol=1e-6)
        assert jnp.allclose(elbow_pos[0, 1], 0.0, atol=1e-6)

        # Hand should be at (L1 + L2, 0)
        assert jnp.allclose(hand_pos[0, 0], params.L1 + params.L2, atol=1e-6)
        assert jnp.allclose(hand_pos[0, 1], 0.0, atol=1e-6)

        # Velocities should be zero
        assert jnp.allclose(elbow_vel, 0.0, atol=1e-6)
        assert jnp.allclose(hand_vel, 0.0, atol=1e-6)

    def test_forward_kinematics_90_degree_shoulder(self, arm, params):
        """Test forward kinematics with shoulder at 90 degrees."""
        joint_state = JointState(
            position=jnp.array([[jnp.pi / 2, 0.0]]),
            velocity=jnp.array([[0.0, 0.0]]),
        )

        elbow_pos, hand_pos, _, _ = TwoDofArm.forward_kinematics(
            joint_state, params
        )

        # At 90 degree shoulder, elbow should be at (0, L1)
        assert jnp.allclose(elbow_pos[0, 0], 0.0, atol=1e-6)
        assert jnp.allclose(elbow_pos[0, 1], params.L1, atol=1e-6)

        # Hand should be at (0, L1 + L2)
        assert jnp.allclose(hand_pos[0, 0], 0.0, atol=1e-6)
        assert jnp.allclose(hand_pos[0, 1], params.L1 + params.L2, atol=1e-6)

    def test_forward_kinematics_batched(self, arm, params):
        """Test that forward kinematics works with batched input."""
        batch_size = 16
        joint_state = JointState(
            position=jnp.zeros((batch_size, 2)),
            velocity=jnp.zeros((batch_size, 2)),
        )

        elbow_pos, hand_pos, elbow_vel, hand_vel = TwoDofArm.forward_kinematics(
            joint_state, params
        )

        assert elbow_pos.shape == (batch_size, 2)
        assert hand_pos.shape == (batch_size, 2)
        assert elbow_vel.shape == (batch_size, 2)
        assert hand_vel.shape == (batch_size, 2)

    def test_inverse_dynamics_static(self, arm, params):
        """Test inverse dynamics at static equilibrium."""
        joint_state = JointState(
            position=jnp.array([[0.5, 0.5]]),
            velocity=jnp.array([[0.0, 0.0]]),
        )
        torques = jnp.array([[0.0, 0.0]])
        endpoint_load = jnp.array([[0.0, 0.0]])

        acceleration = TwoDofArm.inverse_dynamics(
            joint_state, torques, endpoint_load, params
        )

        # With no torques and no velocity, acceleration should be zero
        # (ignoring gravity, which isn't modeled)
        assert jnp.allclose(acceleration, 0.0, atol=1e-6)

    def test_jacobian_shape(self, arm, params):
        """Test Jacobian computation."""
        batch_size = 8
        joint_state = JointState(
            position=jnp.zeros((batch_size, 2)),
            velocity=jnp.zeros((batch_size, 2)),
        )

        jacobian = TwoDofArm.compute_jacobian(joint_state, params)

        assert jacobian.shape == (batch_size, 2, 2)

    def test_integration_preserves_shape(self, arm, params):
        """Test that integration preserves state shape."""
        joint_state = JointState(
            position=jnp.array([[0.5, 0.5]]),
            velocity=jnp.array([[0.1, -0.1]]),
        )
        acceleration = jnp.array([[0.0, 0.0]])

        new_state = TwoDofArm.integrate(joint_state, acceleration, 0.01, params)

        assert new_state.position.shape == joint_state.position.shape
        assert new_state.velocity.shape == joint_state.velocity.shape


class TestPointMass:
    """Tests for the PointMass skeleton."""

    @pytest.fixture
    def pm(self):
        """Create a point mass for testing."""
        return PointMass()

    @pytest.fixture
    def params(self, pm):
        """Get parameters."""
        return pm.get_params()

    def test_initialization(self, pm, params):
        """Test initialization."""
        assert pm.n_joints == 2  # space_dim = 2
        assert pm.n_dim == 2

    def test_forward_kinematics_identity(self, pm, params):
        """For point mass, FK should be identity."""
        joint_state = JointState(
            position=jnp.array([[0.3, 0.4]]),
            velocity=jnp.array([[0.1, 0.2]]),
        )

        pos, vel = PointMass.forward_kinematics(joint_state, params)

        assert jnp.allclose(pos, joint_state.position)
        assert jnp.allclose(vel, joint_state.velocity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
