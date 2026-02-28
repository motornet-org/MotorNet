"""
Tests for policy networks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from motornet_jax.policy import GRUPolicy, MLPPolicy, ModularPolicyGRU, create_modular_policy


class TestGRUPolicy:
    """Tests for the GRU policy network."""

    @pytest.fixture
    def policy(self):
        """Create a GRU policy for testing."""
        return GRUPolicy(
            obs_dim=16,
            action_dim=6,
            hidden_size=64,
            n_gru_layers=1,
            key=jax.random.PRNGKey(0),
        )

    def test_gru_forward_shape(self, policy):
        """Output and hidden shapes are correct."""
        batch_size = 4
        obs = jnp.ones((batch_size, 16))
        hidden = policy.init_hidden(batch_size)

        action, new_hidden = policy(obs, hidden)

        assert action.shape == (batch_size, 6)
        assert new_hidden.shape == (batch_size, 1, 64)

    def test_gru_sigmoid_bounds(self, policy):
        """Actions are bounded in [0, 1] due to sigmoid output."""
        batch_size = 8
        key = jax.random.PRNGKey(99)
        obs = jax.random.normal(key, (batch_size, 16)) * 10.0
        hidden = policy.init_hidden(batch_size)

        action, _ = policy(obs, hidden)

        assert jnp.all(action >= 0.0)
        assert jnp.all(action <= 1.0)

    def test_gru_jit(self, policy):
        """eqx.filter_jit works on the GRU forward pass."""
        batch_size = 2
        obs = jnp.ones((batch_size, 16))
        hidden = policy.init_hidden(batch_size)

        @eqx.filter_jit
        def forward(model, obs, hidden):
            return model(obs, hidden)

        action, new_hidden = forward(policy, obs, hidden)

        assert action.shape == (batch_size, 6)
        assert new_hidden.shape == (batch_size, 1, 64)
        assert jnp.all(jnp.isfinite(action))

    def test_gru_unbatched(self, policy):
        """GRU handles unbatched (single sample) input."""
        obs = jnp.ones((16,))
        action, new_hidden = policy(obs)

        assert action.shape == (6,)
        assert new_hidden.shape == (1, 64)

    def test_gru_multi_layer(self):
        """GRU with multiple layers produces correct hidden shape."""
        policy = GRUPolicy(
            obs_dim=16,
            action_dim=6,
            hidden_size=32,
            n_gru_layers=3,
            key=jax.random.PRNGKey(1),
        )
        batch_size = 4
        obs = jnp.ones((batch_size, 16))
        hidden = policy.init_hidden(batch_size)

        action, new_hidden = policy(obs, hidden)

        assert action.shape == (batch_size, 6)
        assert new_hidden.shape == (batch_size, 3, 32)


class TestMLPPolicy:
    """Tests for the MLP policy network."""

    @pytest.fixture
    def policy(self):
        """Create an MLP policy for testing."""
        return MLPPolicy(
            obs_dim=16,
            action_dim=6,
            hidden_sizes=(64, 64),
            key=jax.random.PRNGKey(0),
        )

    def test_mlp_forward_shape(self, policy):
        """Output shape is correct."""
        batch_size = 4
        obs = jnp.ones((batch_size, 16))
        action = policy(obs)

        assert action.shape == (batch_size, 6)

    def test_mlp_sigmoid_bounds(self, policy):
        """Actions are bounded in [0, 1] due to sigmoid output."""
        key = jax.random.PRNGKey(42)
        obs = jax.random.normal(key, (8, 16)) * 10.0
        action = policy(obs)

        assert jnp.all(action >= 0.0)
        assert jnp.all(action <= 1.0)

    def test_mlp_unbatched(self, policy):
        """MLP handles unbatched (single sample) input."""
        obs = jnp.ones((16,))
        action = policy(obs)

        assert action.shape == (6,)


class TestModularPolicyGRU:
    """Tests for the Modular GRU policy network."""

    @pytest.fixture
    def policy(self):
        """Create a modular GRU policy for testing."""
        input_size = 16
        module_size = [32, 32]
        output_size = 6

        # Split input dims: vision (0,1), proprio (2..13), task (14,15)
        vision_dim = [0, 1]
        proprio_dim = list(range(2, 14))
        task_dim = [14, 15]

        return create_modular_policy(
            input_size=input_size,
            module_size=module_size,
            output_size=output_size,
            vision_dim=vision_dim,
            proprio_dim=proprio_dim,
            task_dim=task_dim,
            key=jax.random.PRNGKey(0),
        )

    def test_modular_gru_forward(self, policy):
        """Smoke test: forward pass produces valid output."""
        batch_size = 4
        obs = jnp.ones((batch_size, 16))
        hidden, h_buffer = policy.init_hidden(batch_size)

        action, new_hidden, new_h_buffer = policy(obs, hidden, h_buffer)

        assert action.shape == (batch_size, 6)
        assert new_hidden.shape == (batch_size, 64)
        assert jnp.all(jnp.isfinite(action))
        assert jnp.all(action >= 0.0)
        assert jnp.all(action <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
