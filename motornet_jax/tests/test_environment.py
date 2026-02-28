"""
Tests for environment module.
"""

import jax
import jax.numpy as jnp
import pytest

from motornet_jax.effector import Arm26
from motornet_jax.types import JointState
from motornet_jax.environment import (
    Environment,
    RandomTargetReach,
    CenterOutReach,
)


class TestEnvironment:
    """Tests for the base Environment class."""

    @pytest.fixture
    def effector(self):
        """Create an Arm26 effector for testing."""
        return Arm26()

    @pytest.fixture
    def env(self, effector):
        """Create an Environment wrapping the Arm26 effector."""
        return Environment(effector, max_ep_duration=1.0)

    @pytest.fixture
    def key(self):
        """Create a PRNG key."""
        return jax.random.PRNGKey(42)

    def test_environment_reset(self, env, key):
        """Environment reset returns valid obs and state."""
        batch_size = 4
        env_state, obs, info = env.reset(key, batch_size=batch_size)

        # State should be populated
        assert env_state.effector is not None
        assert env_state.goal.shape == (batch_size, 2)
        assert env_state.step_count == 0
        assert env_state.elapsed == 0.0

        # Obs should be a valid array with correct batch dim
        assert obs.ndim == 2
        assert obs.shape[0] == batch_size

        # Info should contain expected keys
        assert "states" in info
        assert "goal" in info
        assert info["goal"].shape == (batch_size, 2)

    def test_environment_step(self, env, key):
        """Step advances time and updates physics."""
        batch_size = 2
        env_state, obs, info = env.reset(key, batch_size=batch_size)

        # Take a step with zero action
        action = jnp.zeros((batch_size, env.n_muscles))
        new_state, new_obs, reward, terminated, truncated, step_info = env.step(
            env_state, action
        )

        # Step count should advance
        assert new_state.step_count == 1

        # Elapsed time should advance by dt
        assert jnp.isclose(new_state.elapsed, env.dt)

        # Obs should still have correct shape
        assert new_obs.shape == obs.shape

        # Goal should be unchanged
        assert jnp.allclose(new_state.goal, env_state.goal)

    def test_observation_shape(self, env, key):
        """Observation has correct shape matching observation_dim."""
        batch_size = 8
        env_state, obs, info = env.reset(key, batch_size=batch_size)

        expected_obs_dim = env.observation_dim
        assert obs.shape == (batch_size, expected_obs_dim)

        # Verify the components add up
        # goal (2) + vision (2) + proprioception (2 * 6 muscles) = 16
        assert expected_obs_dim == 2 + 2 + 2 * 6


class TestRandomTargetReach:
    """Tests for RandomTargetReach environment."""

    @pytest.fixture
    def effector(self):
        return Arm26()

    @pytest.fixture
    def env(self, effector):
        return RandomTargetReach(effector, max_ep_duration=1.0)

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(123)

    def test_random_target_reach(self, env, key):
        """RandomTargetReach generates valid targets within reachable workspace."""
        batch_size = 16
        env_state, obs, info = env.reset(key, batch_size=batch_size)

        goal = env_state.goal
        assert goal.shape == (batch_size, 2)

        # Goals should be finite
        assert jnp.all(jnp.isfinite(goal))

        # Different random keys should produce different goals
        key2 = jax.random.PRNGKey(456)
        env_state2, _, _ = env.reset(key2, batch_size=batch_size)
        assert not jnp.allclose(env_state.goal, env_state2.goal)


class TestCenterOutReach:
    """Tests for CenterOutReach environment."""

    @pytest.fixture
    def effector(self):
        return Arm26()

    @pytest.fixture
    def env(self, effector):
        return CenterOutReach(
            effector,
            n_targets=8,
            target_radius=0.1,
            max_ep_duration=1.0,
        )

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(789)

    def test_center_out_reach(self, env, key):
        """CenterOutReach targets lie on a circle around center."""
        batch_size = 32
        env_state, obs, info = env.reset(key, batch_size=batch_size)

        goal = env_state.goal
        assert goal.shape == (batch_size, 2)

        # All targets should be at target_radius distance from center
        distances = jnp.sqrt(jnp.sum((goal - env.center_pos) ** 2, axis=-1))
        assert jnp.allclose(distances, env.target_radius, atol=1e-5)

        # Goals should be finite
        assert jnp.all(jnp.isfinite(goal))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
