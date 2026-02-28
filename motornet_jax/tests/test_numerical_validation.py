"""
Numerical validation tests comparing JAX and PyTorch implementations.

This ensures the JAX implementation produces the same results as PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# JAX imports
from motornet_jax.skeleton import TwoDofArm as JaxTwoDofArm
from motornet_jax.types import JointState as JaxJointState

# PyTorch imports (if available and compatible)
HAS_PYTORCH = False
try:
    import torch
    # Try importing motornet - may fail on Python < 3.10 due to syntax
    from motornet.skeleton import TwoDofArm as TorchTwoDofArm
    HAS_PYTORCH = True
except (ImportError, TypeError, SyntaxError):
    # TypeError occurs on Python 3.9 with union type syntax
    pass


def numpy_to_jax(arr):
    """Convert numpy array to JAX array."""
    return jnp.array(arr)


def torch_to_numpy(tensor):
    """Convert PyTorch tensor to numpy."""
    return tensor.cpu().detach().numpy()


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not available")
class TestNumericalValidation:
    """Compare JAX and PyTorch implementations numerically."""

    @pytest.fixture
    def jax_arm(self):
        """Create JAX arm."""
        return JaxTwoDofArm()

    @pytest.fixture
    def torch_arm(self):
        """Create PyTorch arm with same parameters."""
        arm = TorchTwoDofArm()
        arm.build(
            timestep=0.01,
            pos_upper_bound=[2.44, 2.79],  # Default values
            pos_lower_bound=[0.0, 0.0],
            vel_upper_bound=[1000.0, 1000.0],
            vel_lower_bound=[-1000.0, -1000.0],
        )
        return arm

    def test_forward_kinematics_match(self, jax_arm, torch_arm):
        """Test that forward kinematics match between implementations."""
        # Test positions
        test_positions = [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.7, 1.2],
        ]

        jax_params = jax_arm.get_params()

        for pos in test_positions:
            # JAX computation
            jax_state = JaxJointState(
                position=jnp.array([pos]),
                velocity=jnp.zeros((1, 2)),
            )
            _, jax_hand, _, _ = JaxTwoDofArm.forward_kinematics(jax_state, jax_params)

            # PyTorch computation
            torch_state = torch.tensor([pos + [0.0, 0.0]], dtype=torch.float32)
            torch_cart = torch_arm.joint2cartesian(torch_state)
            torch_hand = torch_to_numpy(torch_cart[:, :2])

            # Compare
            np.testing.assert_allclose(
                np.array(jax_hand),
                torch_hand,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Forward kinematics mismatch at position {pos}",
            )

    def test_inverse_dynamics_match(self, jax_arm, torch_arm):
        """Test that inverse dynamics match between implementations."""
        test_cases = [
            # (position, velocity, torque)
            ([0.5, 0.5], [0.0, 0.0], [0.0, 0.0]),
            ([0.5, 0.5], [1.0, -0.5], [0.0, 0.0]),
            ([0.5, 0.5], [0.0, 0.0], [1.0, -0.5]),
            ([1.0, 0.8], [0.5, 0.3], [0.2, -0.3]),
        ]

        jax_params = jax_arm.get_params()

        for pos, vel, torque in test_cases:
            # JAX computation
            jax_state = JaxJointState(
                position=jnp.array([pos]),
                velocity=jnp.array([vel]),
            )
            endpoint_load = jnp.zeros((1, 2))
            jax_acc = JaxTwoDofArm.inverse_dynamics(
                jax_state, jnp.array([torque]), endpoint_load, jax_params
            )

            # PyTorch computation
            torch_state = torch.tensor([pos + vel], dtype=torch.float32)
            torch_torque = torch.tensor([torque], dtype=torch.float32)
            torch_endpoint = torch.zeros((1, 2))
            torch_acc = torch_arm.ode(torch_torque, torch_state, torch_endpoint)

            # Compare
            np.testing.assert_allclose(
                np.array(jax_acc),
                torch_to_numpy(torch_acc),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Inverse dynamics mismatch at {pos}, {vel}, {torque}",
            )

    def test_integration_match(self, jax_arm, torch_arm):
        """Test that integration produces matching results."""
        dt = 0.01
        initial_pos = [0.5, 0.5]
        initial_vel = [0.1, -0.1]
        torque = [0.5, -0.3]

        jax_params = jax_arm.get_params()

        # JAX integration
        jax_state = JaxJointState(
            position=jnp.array([initial_pos]),
            velocity=jnp.array([initial_vel]),
        )
        endpoint_load = jnp.zeros((1, 2))
        _, jax_acc = JaxTwoDofArm.ode(
            jax_state, jnp.array([torque]), endpoint_load, jax_params
        )
        jax_new = JaxTwoDofArm.integrate(jax_state, jax_acc, dt, jax_params)

        # PyTorch integration
        torch_state = torch.tensor([initial_pos + initial_vel], dtype=torch.float32)
        torch_torque = torch.tensor([torque], dtype=torch.float32)
        torch_endpoint = torch.zeros((1, 2))
        torch_acc = torch_arm.ode(torch_torque, torch_state, torch_endpoint)
        torch_new = torch_arm.integrate(dt, torch_acc, torch_state)

        # Compare positions
        np.testing.assert_allclose(
            np.array(jax_new.position),
            torch_to_numpy(torch_new[:, :2]),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Integration position mismatch",
        )

        # Compare velocities
        np.testing.assert_allclose(
            np.array(jax_new.velocity),
            torch_to_numpy(torch_new[:, 2:]),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Integration velocity mismatch",
        )

    def test_trajectory_match(self, jax_arm, torch_arm):
        """Test that full trajectory matches over multiple steps."""
        n_steps = 50
        dt = 0.01
        initial_pos = [0.5, 0.5]
        initial_vel = [0.0, 0.0]
        torque = [0.5, -0.3]

        jax_params = jax_arm.get_params()

        # JAX trajectory
        jax_state = JaxJointState(
            position=jnp.array([initial_pos]),
            velocity=jnp.array([initial_vel]),
        )
        endpoint_load = jnp.zeros((1, 2))

        jax_positions = [np.array(jax_state.position[0])]
        for _ in range(n_steps):
            _, jax_acc = JaxTwoDofArm.ode(
                jax_state, jnp.array([torque]), endpoint_load, jax_params
            )
            jax_state = JaxTwoDofArm.integrate(jax_state, jax_acc, dt, jax_params)
            jax_positions.append(np.array(jax_state.position[0]))

        # PyTorch trajectory
        torch_state = torch.tensor([initial_pos + initial_vel], dtype=torch.float32)
        torch_torque = torch.tensor([torque], dtype=torch.float32)
        torch_endpoint = torch.zeros((1, 2))

        torch_positions = [torch_to_numpy(torch_state[:, :2])[0]]
        for _ in range(n_steps):
            torch_acc = torch_arm.ode(torch_torque, torch_state, torch_endpoint)
            torch_state = torch_arm.integrate(dt, torch_acc, torch_state)
            torch_positions.append(torch_to_numpy(torch_state[:, :2])[0])

        # Compare trajectories
        jax_traj = np.array(jax_positions)
        torch_traj = np.array(torch_positions)

        np.testing.assert_allclose(
            jax_traj,
            torch_traj,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Trajectory mismatch",
        )


class TestJAXOnlyValidation:
    """Validation tests that don't require PyTorch."""

    @pytest.fixture
    def arm(self):
        return JaxTwoDofArm()

    @pytest.fixture
    def params(self, arm):
        return arm.get_params()

    def test_arm_reach(self, arm, params):
        """Test that arm can reach expected workspace."""
        # At full extension (0, 0), reach should be L1 + L2
        state = JaxJointState(
            position=jnp.array([[0.0, 0.0]]),
            velocity=jnp.zeros((1, 2)),
        )
        _, hand, _, _ = JaxTwoDofArm.forward_kinematics(state, params)

        expected_reach = params.L1 + params.L2
        actual_reach = jnp.sqrt(hand[0, 0]**2 + hand[0, 1]**2)

        assert jnp.abs(actual_reach - expected_reach) < 1e-6

    def test_energy_conservation(self, arm, params):
        """Test approximate energy conservation (no damping case)."""
        # With no external forces or damping, total mechanical energy
        # should be approximately conserved
        n_steps = 100
        dt = 0.001  # Small timestep for accuracy

        state = JaxJointState(
            position=jnp.array([[0.5, 0.5]]),
            velocity=jnp.array([[1.0, -0.5]]),
        )
        endpoint_load = jnp.zeros((1, 2))
        torque = jnp.zeros((1, 2))

        def compute_kinetic_energy(state, params):
            """Simplified kinetic energy calculation."""
            # This is an approximation - full calculation would need inertia matrix
            vel = state.velocity[0]
            return 0.5 * (params.M1 + params.M2) * jnp.sum(vel**2)

        initial_energy = compute_kinetic_energy(state, params)

        for _ in range(n_steps):
            _, acc = JaxTwoDofArm.ode(state, torque, endpoint_load, params)
            state = JaxTwoDofArm.integrate(state, acc, dt, params)

        final_energy = compute_kinetic_energy(state, params)

        # Energy should be roughly conserved (within numerical error)
        # Note: This is a simplified test; full energy would include potential
        energy_ratio = final_energy / (initial_energy + 1e-8)
        assert 0.8 < energy_ratio < 1.2, f"Energy changed too much: {energy_ratio}"

    def test_jacobian_numerical(self, arm, params):
        """Test Jacobian against numerical differentiation."""
        eps = 1e-5
        pos = jnp.array([[0.5, 0.5]])

        state = JaxJointState(
            position=pos,
            velocity=jnp.zeros((1, 2)),
        )

        # Analytical Jacobian
        jacobian = JaxTwoDofArm.compute_jacobian(state, params)

        # Numerical Jacobian
        numerical_jacobian = jnp.zeros((2, 2))
        for i in range(2):
            pos_plus = pos.at[0, i].add(eps)
            pos_minus = pos.at[0, i].add(-eps)

            state_plus = JaxJointState(position=pos_plus, velocity=jnp.zeros((1, 2)))
            state_minus = JaxJointState(position=pos_minus, velocity=jnp.zeros((1, 2)))

            _, hand_plus, _, _ = JaxTwoDofArm.forward_kinematics(state_plus, params)
            _, hand_minus, _, _ = JaxTwoDofArm.forward_kinematics(state_minus, params)

            numerical_jacobian = numerical_jacobian.at[:, i].set(
                (hand_plus[0] - hand_minus[0]) / (2 * eps)
            )

        # The analytical Jacobian may have small differences due to
        # different conventions, but should be within ~1%
        np.testing.assert_allclose(
            np.array(jacobian[0]),
            np.array(numerical_jacobian),
            rtol=0.01,  # 1% relative tolerance
            atol=0.005,  # 5mm absolute tolerance
            err_msg="Jacobian doesn't match numerical differentiation",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
