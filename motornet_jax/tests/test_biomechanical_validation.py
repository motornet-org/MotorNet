"""Biomechanical validation: compare JAX Arm26 against PyTorch RigidTendonArm26.

Tests geometry, muscle force computation, and full simulation trajectories
to ensure numerical equivalence between the two implementations.
"""

import numpy as np
import numpy.testing as npt
import pytest
import jax
import jax.numpy as jnp

from motornet_jax.effector import Arm26
from motornet_jax.types import JointState, GeometryState

# Try importing PyTorch MotorNet
# Note: PyTorch MotorNet requires Python 3.10+ (uses `float | list` syntax)
try:
    import torch as th
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from motornet.effector import RigidTendonArm26
    from motornet.muscle import RigidTendonHillMuscle
    HAS_PYTORCH = True
except (ImportError, TypeError):
    HAS_PYTORCH = False

pytestmark = pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch MotorNet not available")


# ---------- Helpers ----------

def create_jax_arm(dt=0.01):
    arm = Arm26(dt=dt, n_ministeps=1)
    return arm, arm.get_params()


def create_pytorch_arm(dt=0.01):
    arm = RigidTendonArm26(muscle=RigidTendonHillMuscle(), timestep=dt)
    arm.reset(options={"batch_size": 1})
    return arm


def pytorch_geometry(arm, joint_pos, joint_vel):
    """Get geometry from PyTorch arm at given joint state."""
    joint_state = th.tensor(
        np.concatenate([joint_pos, joint_vel], axis=-1), dtype=th.float32
    )
    geom = arm._get_geometry(joint_state)
    # geom shape: (batch, 4, 6) = [musculotendon_len, musculotendon_vel, moment_arm_sho, moment_arm_elb]
    return {
        "musculotendon_length": geom[:, 0, :].detach().numpy(),
        "musculotendon_velocity": geom[:, 1, :].detach().numpy(),
        "moment_arm": geom[:, 2:, :].detach().numpy(),  # (batch, 2, 6)
    }


def jax_geometry(params, joint_pos, joint_vel):
    """Get geometry from JAX arm at given joint state."""
    joint_state = JointState(
        position=jnp.array(joint_pos),
        velocity=jnp.array(joint_vel),
    )
    geom = Arm26.compute_geometry(joint_state, params)
    return {
        "musculotendon_length": np.array(geom.musculotendon_length),
        "musculotendon_velocity": np.array(geom.musculotendon_velocity),
        # JAX moment_arm is (batch, 6, 2), transpose to (batch, 2, 6) for comparison
        "moment_arm": np.array(jnp.transpose(geom.moment_arm, (0, 2, 1))),
    }


# ---------- Test Classes ----------

class TestGeometryValidation:
    """Compare geometry computation between PyTorch and JAX."""

    JOINT_CONFIGS = [
        ([0.5, 0.5], "mid-range"),
        ([0.8, 1.2], "center config"),
        ([0.3, 0.3], "flexed"),
        ([1.0, 1.5], "extended"),
        ([1.5, 0.8], "shoulder extended"),
        ([0.1, 2.0], "elbow extended"),
    ]

    @pytest.fixture
    def arms(self):
        jax_arm, jax_params = create_jax_arm()
        pt_arm = create_pytorch_arm()
        return jax_params, pt_arm

    @pytest.mark.parametrize("config", JOINT_CONFIGS, ids=[c[1] for c in JOINT_CONFIGS])
    def test_musculotendon_length_match(self, arms, config):
        jax_params, pt_arm = arms
        joint_angles, _ = config
        pos = np.array([joint_angles], dtype=np.float32)
        vel = np.zeros_like(pos)

        pt_geom = pytorch_geometry(pt_arm, pos, vel)
        jx_geom = jax_geometry(jax_params, pos, vel)

        npt.assert_allclose(
            jx_geom["musculotendon_length"],
            pt_geom["musculotendon_length"],
            rtol=1e-5, atol=1e-6,
            err_msg=f"Musculotendon length mismatch at {joint_angles}",
        )

    @pytest.mark.parametrize("config", JOINT_CONFIGS, ids=[c[1] for c in JOINT_CONFIGS])
    def test_moment_arms_match(self, arms, config):
        jax_params, pt_arm = arms
        joint_angles, _ = config
        pos = np.array([joint_angles], dtype=np.float32)
        vel = np.zeros_like(pos)

        pt_geom = pytorch_geometry(pt_arm, pos, vel)
        jx_geom = jax_geometry(jax_params, pos, vel)

        npt.assert_allclose(
            jx_geom["moment_arm"],
            pt_geom["moment_arm"],
            rtol=1e-5, atol=1e-6,
            err_msg=f"Moment arm mismatch at {joint_angles}",
        )

    def test_musculotendon_velocity_match(self, arms):
        jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]], dtype=np.float32)
        vel = np.array([[0.5, -0.3]], dtype=np.float32)

        pt_geom = pytorch_geometry(pt_arm, pos, vel)
        jx_geom = jax_geometry(jax_params, pos, vel)

        npt.assert_allclose(
            jx_geom["musculotendon_velocity"],
            pt_geom["musculotendon_velocity"],
            rtol=1e-5, atol=1e-6,
        )


class TestMuscleForceValidation:
    """Compare muscle force computation between PyTorch and JAX."""

    @pytest.fixture
    def arms(self):
        jax_arm, jax_params = create_jax_arm()
        pt_arm = create_pytorch_arm()
        return jax_arm, jax_params, pt_arm

    def _compute_pytorch_force(self, pt_arm, joint_pos, joint_vel, activation):
        """Run one PyTorch step to get forces from known state."""
        batch_size = joint_pos.shape[0]
        joint_state = th.tensor(
            np.concatenate([joint_pos, joint_vel], axis=-1), dtype=th.float32
        )
        geom = pt_arm._get_geometry(joint_state)

        # Create muscle state with given activation
        act_tensor = th.tensor(activation, dtype=th.float32).reshape(batch_size, 1, 6)
        # Use the muscle's integrate to get force from activation + geometry
        # The muscle _integrate computes force from activation, geometry
        state_deriv = th.zeros(batch_size, 1, 6)
        muscle_result = pt_arm.muscle._integrate(pt_arm.dt, state_deriv, act_tensor, geom)
        # muscle_result shape: (batch, 7, 6) = [activation, muscle_len, muscle_vel, flpe, flce, active_force, force]
        return {
            "activation": muscle_result[:, 0, :].detach().numpy(),
            "fiber_length": muscle_result[:, 1, :].detach().numpy(),
            "flpe": muscle_result[:, 3, :].detach().numpy(),
            "flce": muscle_result[:, 4, :].detach().numpy(),
            "active_force": muscle_result[:, 5, :].detach().numpy(),
            "force": muscle_result[:, 6, :].detach().numpy(),
        }

    def _compute_jax_force(self, jax_params, joint_pos, joint_vel, activation):
        """Compute JAX muscle force from known state."""
        joint_state = JointState(
            position=jnp.array(joint_pos),
            velocity=jnp.array(joint_vel),
        )
        geom = Arm26.compute_geometry(joint_state, jax_params)
        act = jnp.array(activation)
        force = Arm26.compute_muscle_force(act, geom, jax_params)
        return {"force": np.array(force)}

    def test_force_at_center_config(self, arms):
        """Test force computation at center configuration with moderate activation."""
        _, jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]], dtype=np.float32)
        vel = np.zeros_like(pos)
        act = np.array([[0.5] * 6], dtype=np.float32)

        pt_result = self._compute_pytorch_force(pt_arm, pos, vel, act)
        jx_result = self._compute_jax_force(jax_params, pos, vel, act)

        npt.assert_allclose(
            jx_result["force"], pt_result["force"],
            rtol=1e-4, atol=1e-3,
            err_msg="Force mismatch at center config",
        )

    def test_force_at_varied_activations(self, arms):
        """Test force at different activation levels."""
        _, jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]] * 4, dtype=np.float32)
        vel = np.zeros_like(pos)
        act = np.array([
            [0.1] * 6,
            [0.3] * 6,
            [0.7] * 6,
            [1.0] * 6,
        ], dtype=np.float32)

        pt_result = self._compute_pytorch_force(pt_arm, pos, vel, act)
        jx_result = self._compute_jax_force(jax_params, pos, vel, act)

        npt.assert_allclose(
            jx_result["force"], pt_result["force"],
            rtol=1e-4, atol=1e-3,
            err_msg="Force mismatch at varied activations",
        )

    def test_force_at_varied_positions(self, arms):
        """Test force across different joint configurations."""
        _, jax_params, pt_arm = arms
        pos = np.array([
            [0.5, 0.5], [0.8, 1.2], [1.0, 1.5], [1.5, 0.8],
        ], dtype=np.float32)
        vel = np.zeros_like(pos)
        act = np.array([[0.5] * 6] * 4, dtype=np.float32)

        pt_result = self._compute_pytorch_force(pt_arm, pos, vel, act)
        jx_result = self._compute_jax_force(jax_params, pos, vel, act)

        npt.assert_allclose(
            jx_result["force"], pt_result["force"],
            rtol=1e-4, atol=1e-3,
            err_msg="Force mismatch at varied positions",
        )

    def test_force_with_nonzero_velocity(self, arms):
        """Test force-velocity interaction."""
        _, jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]] * 3, dtype=np.float32)
        vel = np.array([
            [0.0, 0.0],
            [1.0, -0.5],
            [-0.5, 1.0],
        ], dtype=np.float32)
        act = np.array([[0.5] * 6] * 3, dtype=np.float32)

        pt_result = self._compute_pytorch_force(pt_arm, pos, vel, act)
        jx_result = self._compute_jax_force(jax_params, pos, vel, act)

        npt.assert_allclose(
            jx_result["force"], pt_result["force"],
            rtol=1e-4, atol=1e-3,
            err_msg="Force mismatch with non-zero velocity",
        )


class TestTrajectoryValidation:
    """Compare full simulation trajectories between PyTorch and JAX.

    Note: There is a subtle integration order difference. PyTorch stores force in the
    muscle state (computed with updated activation + OLD geometry), while JAX recomputes
    force from scratch at each step (using current activation + current geometry). This
    causes small drift that accumulates over time. The geometry and force tests above
    confirm the biomechanical equations are identical; trajectory differences are purely
    from this numerical integration ordering.
    """

    @pytest.fixture
    def arms(self):
        dt = 0.01
        jax_arm, jax_params = create_jax_arm(dt=dt)
        pt_arm = create_pytorch_arm(dt=dt)
        return jax_arm, jax_params, pt_arm

    def _run_pytorch_trajectory(self, pt_arm, initial_pos, initial_vel, actions, n_steps):
        """Run PyTorch simulation and record trajectory."""
        batch_size = initial_pos.shape[0]
        joint_state = th.tensor(
            np.concatenate([initial_pos, initial_vel], axis=-1), dtype=th.float32
        )
        pt_arm.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        positions = [initial_pos.copy()]
        velocities = [initial_vel.copy()]
        fingertips = [pt_arm.states["fingertip"].detach().numpy().copy()]

        for t in range(n_steps):
            action = th.tensor(actions[t], dtype=th.float32) if actions.ndim == 3 else th.tensor(actions, dtype=th.float32)
            pt_arm.step(action)
            js = pt_arm.states["joint"].detach().numpy()
            pos, vel = np.split(js, 2, axis=-1)
            positions.append(pos.copy())
            velocities.append(vel.copy())
            fingertips.append(pt_arm.states["fingertip"].detach().numpy().copy())

        return {
            "positions": np.array(positions),     # (n_steps+1, batch, 2)
            "velocities": np.array(velocities),   # (n_steps+1, batch, 2)
            "fingertips": np.array(fingertips),    # (n_steps+1, batch, 2)
        }

    def _run_jax_trajectory(self, jax_arm, jax_params, initial_pos, initial_vel, actions, n_steps):
        """Run JAX simulation and record trajectory."""
        batch_size = initial_pos.shape[0]
        joint_state = JointState(
            position=jnp.array(initial_pos),
            velocity=jnp.array(initial_vel),
        )
        state = jax_arm.reset(batch_size=batch_size, joint_state=joint_state)

        positions = [initial_pos.copy()]
        velocities = [initial_vel.copy()]
        fingertips = [np.array(state.fingertip)]

        for t in range(n_steps):
            action = jnp.array(actions[t]) if actions.ndim == 3 else jnp.array(actions)
            endpoint_load = jnp.zeros((batch_size, 2))
            joint_load = jnp.zeros((batch_size, 2))
            state = Arm26.step(state, action, endpoint_load, joint_load, jax_params)
            positions.append(np.array(state.joint.position))
            velocities.append(np.array(state.joint.velocity))
            fingertips.append(np.array(state.fingertip))

        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "fingertips": np.array(fingertips),
        }

    def test_single_step(self, arms):
        """Compare a single simulation step."""
        jax_arm, jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]], dtype=np.float32)
        vel = np.zeros_like(pos)
        action = np.array([[0.5] * 6], dtype=np.float32)

        pt_traj = self._run_pytorch_trajectory(pt_arm, pos, vel, action, n_steps=1)
        jx_traj = self._run_jax_trajectory(jax_arm, jax_params, pos, vel, action, n_steps=1)

        npt.assert_allclose(
            jx_traj["positions"][-1], pt_traj["positions"][-1],
            rtol=1e-4, atol=1e-5,
            err_msg="Position mismatch after single step",
        )
        npt.assert_allclose(
            jx_traj["fingertips"][-1], pt_traj["fingertips"][-1],
            rtol=1e-4, atol=1e-5,
            err_msg="Fingertip mismatch after single step",
        )

    def test_trajectory_50_steps_constant_action(self, arms):
        """Compare 50-step trajectory with constant action."""
        jax_arm, jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]], dtype=np.float32)
        vel = np.zeros_like(pos)
        action = np.array([[0.5] * 6], dtype=np.float32)

        pt_traj = self._run_pytorch_trajectory(pt_arm, pos, vel, action, n_steps=50)
        jx_traj = self._run_jax_trajectory(jax_arm, jax_params, pos, vel, action, n_steps=50)

        # Looser tolerance for multi-step (drift from integration order difference)
        npt.assert_allclose(
            jx_traj["positions"], pt_traj["positions"],
            rtol=5e-2, atol=5e-2,
            err_msg="Position trajectory mismatch over 50 steps",
        )
        npt.assert_allclose(
            jx_traj["fingertips"], pt_traj["fingertips"],
            rtol=5e-2, atol=5e-2,
            err_msg="Fingertip trajectory mismatch over 50 steps",
        )

    def test_trajectory_100_steps_varying_action(self, arms):
        """Compare 100-step trajectory with time-varying sinusoidal actions."""
        jax_arm, jax_params, pt_arm = arms
        pos = np.array([[0.8, 1.2]], dtype=np.float32)
        vel = np.zeros_like(pos)

        # Sinusoidal excitation pattern
        t = np.linspace(0, 2 * np.pi, 100)
        actions = np.zeros((100, 1, 6), dtype=np.float32)
        for i in range(6):
            phase = i * np.pi / 3
            actions[:, 0, i] = 0.3 + 0.2 * np.sin(t + phase)

        pt_traj = self._run_pytorch_trajectory(pt_arm, pos, vel, actions, n_steps=100)
        jx_traj = self._run_jax_trajectory(jax_arm, jax_params, pos, vel, actions, n_steps=100)

        npt.assert_allclose(
            jx_traj["positions"], pt_traj["positions"],
            rtol=1e-1, atol=1e-1,
            err_msg="Position trajectory mismatch over 100 steps with varying actions",
        )
        npt.assert_allclose(
            jx_traj["fingertips"], pt_traj["fingertips"],
            rtol=1e-1, atol=1e-1,
            err_msg="Fingertip trajectory mismatch over 100 steps with varying actions",
        )

    def test_trajectory_from_different_starts(self, arms):
        """Compare trajectories starting from different joint configurations."""
        jax_arm, jax_params, pt_arm = arms
        positions = [
            [0.5, 0.5],
            [1.0, 1.0],
            [0.3, 1.5],
        ]
        action = np.array([[0.4] * 6], dtype=np.float32)

        for start_pos in positions:
            pos = np.array([start_pos], dtype=np.float32)
            vel = np.zeros_like(pos)

            pt_traj = self._run_pytorch_trajectory(pt_arm, pos, vel, action, n_steps=30)
            jx_traj = self._run_jax_trajectory(jax_arm, jax_params, pos, vel, action, n_steps=30)

            npt.assert_allclose(
                jx_traj["positions"], pt_traj["positions"],
                rtol=5e-2, atol=5e-2,
                err_msg=f"Position mismatch from start {start_pos}",
            )
