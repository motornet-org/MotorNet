"""
Tests for the Arm26 effector model.
"""

import jax
import jax.numpy as jnp
import pytest

from motornet_jax.effector import Arm26, Arm26Params
from motornet_jax.types import JointState, EffectorState


class TestArm26:
    """Tests for the Arm26 effector."""

    @pytest.fixture
    def arm(self):
        """Create an Arm26 for testing."""
        return Arm26(dt=0.01, n_ministeps=1)

    @pytest.fixture
    def params(self, arm):
        """Get Arm26 parameters."""
        return arm.get_params()

    def test_initialization(self, arm, params):
        """Test initialization."""
        assert params.a0.shape == (6,)
        assert params.a1.shape == (2, 6)
        assert params.a2.shape == (2, 6)
        assert params.max_isometric_force.shape == (6,)

    def test_compute_geometry(self, arm, params):
        """Test geometry computation."""
        batch_size = 4
        joint_state = JointState(
            position=jnp.ones((batch_size, 2)) * 0.5,
            velocity=jnp.zeros((batch_size, 2)),
        )

        geometry = Arm26.compute_geometry(joint_state, params)

        assert geometry.musculotendon_length.shape == (batch_size, 6)
        assert geometry.musculotendon_velocity.shape == (batch_size, 6)
        assert geometry.moment_arm.shape == (batch_size, 6, 2)

        # Lengths should be positive
        assert jnp.all(geometry.musculotendon_length > 0)

    def test_reset(self, arm):
        """Test reset function."""
        batch_size = 8
        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)

        assert state.joint.position.shape == (batch_size, 2)
        assert state.muscle.activation.shape == (batch_size, 6)
        assert state.geometry.moment_arm.shape == (batch_size, 6, 2)
        assert state.fingertip.shape == (batch_size, 2)

    def test_step(self, arm, params):
        """Test single step."""
        batch_size = 2
        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)

        action = jnp.ones((batch_size, 6)) * 0.5
        endpoint_load = jnp.zeros((batch_size, 2))
        joint_load = jnp.zeros((batch_size, 2))

        new_state = Arm26.step(state, action, endpoint_load, joint_load, params)

        assert new_state.joint.position.shape == (batch_size, 2)
        assert new_state.muscle.activation.shape == (batch_size, 6)

        # Velocity should have changed (position change is small in one step)
        assert not jnp.allclose(new_state.joint.velocity, state.joint.velocity)
        # Activation should have changed
        assert not jnp.allclose(new_state.muscle.activation, state.muscle.activation)

    def test_simulate(self, arm, params):
        """Test multi-step simulation."""
        batch_size = 2
        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)

        action = jnp.ones((batch_size, 6)) * 0.3
        endpoint_load = jnp.zeros((batch_size, 2))
        joint_load = jnp.zeros((batch_size, 2))

        final_state = Arm26.simulate(
            state, action, endpoint_load, joint_load, params, n_ministeps=10
        )

        assert final_state.joint.position.shape == (batch_size, 2)

    def test_muscle_force_computation(self, arm, params):
        """Test muscle force computation."""
        batch_size = 2
        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)

        # Test with different activations
        activations = jnp.array([[0.0] * 6, [1.0] * 6])

        force = Arm26.compute_muscle_force(activations, state.geometry, params)

        assert force.shape == (batch_size, 6)
        # Higher activation should produce more force
        assert jnp.all(force[1] >= force[0])

    def test_jit_compilation(self, arm, params):
        """Test that functions compile with JIT."""
        batch_size = 4
        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)

        action = jnp.ones((batch_size, 6)) * 0.5
        endpoint_load = jnp.zeros((batch_size, 2))
        joint_load = jnp.zeros((batch_size, 2))

        # Compile and run
        step_jit = jax.jit(lambda s: Arm26.step(s, action, endpoint_load, joint_load, params))

        # First call compiles
        new_state = step_jit(state)
        # Second call uses cached
        new_state2 = step_jit(new_state)

        assert new_state2.joint.position.shape == (batch_size, 2)

    def test_vmap_compatibility(self, arm, params):
        """Test batched computation with vmap."""
        key = jax.random.PRNGKey(0)

        # Create single state
        single_state = arm.reset(batch_size=1, key=key)

        # Create batched actions
        batch_size = 8
        actions = jax.random.uniform(key, (batch_size, 6))
        endpoint_loads = jnp.zeros((batch_size, 2))
        joint_loads = jnp.zeros((batch_size, 2))

        # Expand state for batch
        batched_state = EffectorState(
            joint=JointState(
                position=jnp.repeat(single_state.joint.position, batch_size, axis=0),
                velocity=jnp.repeat(single_state.joint.velocity, batch_size, axis=0),
            ),
            cartesian=single_state.cartesian._replace(
                position=jnp.repeat(single_state.cartesian.position, batch_size, axis=0),
                velocity=jnp.repeat(single_state.cartesian.velocity, batch_size, axis=0),
            ),
            muscle=single_state.muscle._replace(
                activation=jnp.repeat(single_state.muscle.activation, batch_size, axis=0),
                fiber_length=jnp.repeat(single_state.muscle.fiber_length, batch_size, axis=0),
                fiber_velocity=jnp.repeat(single_state.muscle.fiber_velocity, batch_size, axis=0),
            ),
            geometry=single_state.geometry._replace(
                musculotendon_length=jnp.repeat(single_state.geometry.musculotendon_length, batch_size, axis=0),
                musculotendon_velocity=jnp.repeat(single_state.geometry.musculotendon_velocity, batch_size, axis=0),
                moment_arm=jnp.repeat(single_state.geometry.moment_arm, batch_size, axis=0),
            ),
            fingertip=jnp.repeat(single_state.fingertip, batch_size, axis=0),
        )

        # Run step
        new_state = Arm26.step(batched_state, actions, endpoint_loads, joint_loads, params)

        assert new_state.joint.position.shape == (batch_size, 2)

    def test_lax_scan_trajectory(self, arm, params):
        """Test trajectory rollout with lax.scan."""
        batch_size = 2
        n_steps = 50
        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)

        action = jnp.ones((batch_size, 6)) * 0.3
        endpoint_load = jnp.zeros((batch_size, 2))
        joint_load = jnp.zeros((batch_size, 2))

        def step_fn(state, _):
            new_state = Arm26.step(state, action, endpoint_load, joint_load, params)
            return new_state, new_state.fingertip

        final_state, trajectory = jax.lax.scan(step_fn, state, None, length=n_steps)

        assert trajectory.shape == (n_steps, batch_size, 2)
        assert final_state.joint.position.shape == (batch_size, 2)


class TestArm26Performance:
    """Performance tests for Arm26."""

    def test_throughput(self):
        """Test simulation throughput."""
        import time

        arm = Arm26(dt=0.01, n_ministeps=1)
        params = arm.get_params()
        batch_size = 64
        n_steps = 100

        key = jax.random.PRNGKey(0)
        state = arm.reset(batch_size=batch_size, key=key)
        action = jnp.ones((batch_size, 6)) * 0.3
        endpoint_load = jnp.zeros((batch_size, 2))
        joint_load = jnp.zeros((batch_size, 2))

        # Compile
        @jax.jit
        def run_episode(state):
            def step_fn(state, _):
                new_state = Arm26.step(state, action, endpoint_load, joint_load, params)
                return new_state, None

            final_state, _ = jax.lax.scan(step_fn, state, None, length=n_steps)
            return final_state

        # Warmup
        state = run_episode(state)
        state.joint.position.block_until_ready()

        # Benchmark
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            state = arm.reset(batch_size=batch_size, key=key)
            state = run_episode(state)
        state.joint.position.block_until_ready()
        elapsed = time.perf_counter() - start

        total_steps = n_runs * n_steps * batch_size
        steps_per_second = total_steps / elapsed

        print(f"\nArm26 Performance:")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per second: {steps_per_second:,.0f}")
        print(f"  Episodes per second: {n_runs / elapsed:,.1f}")

        # Should be fast
        assert steps_per_second > 100_000, f"Too slow: {steps_per_second} steps/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
