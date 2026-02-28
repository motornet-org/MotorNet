"""
MLP-based policy network using Equinox.

Provides a simple feedforward policy for motor control tasks.
"""

from typing import NamedTuple, Tuple, List, Optional
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx


class MLPPolicyParams(NamedTuple):
    """Parameters structure for MLP policy."""
    hidden_sizes: Tuple[int, ...]


class MLPPolicy(eqx.Module):
    """MLP-based feedforward policy network.

    Architecture:
    - Input layer
    - Hidden layers with ReLU activation
    - Output layer with sigmoid activation for bounded actions [0, 1]
    """

    layers: list
    hidden_sizes: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        key: Optional[jax.random.PRNGKey] = None,
    ):
        """Initialize MLP policy.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            hidden_sizes: Tuple of hidden layer sizes.
            key: Random key for initialization.
        """
        if key is None:
            key = random.PRNGKey(0)

        self.hidden_sizes = hidden_sizes

        # Build layers
        layer_sizes = [obs_dim] + list(hidden_sizes) + [action_dim]
        n_layers = len(layer_sizes) - 1
        keys = random.split(key, n_layers)

        self.layers = []
        for i in range(n_layers):
            layer = eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i])
            self.layers.append(layer)

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: Observation. Shape: (obs_dim,) or (batch, obs_dim)

        Returns:
            action: Action. Shape: same batch dims as input + (action_dim,)
        """
        # Handle batched vs unbatched input
        batched = obs.ndim > 1
        if not batched:
            obs = obs[None, :]

        x = obs

        # Hidden layers with ReLU
        for layer in self.layers[:-1]:
            x = jax.nn.relu(jax.vmap(layer)(x))

        # Output layer with sigmoid
        x = jax.nn.sigmoid(jax.vmap(self.layers[-1])(x))

        if not batched:
            x = x[0]

        return x


class MLPPolicyWithNoise(eqx.Module):
    """MLP policy with exploration noise."""

    policy: MLPPolicy
    noise_std: float = eqx.field(static=True)

    def __init__(self, policy: MLPPolicy, noise_std: float = 0.1):
        self.policy = policy
        self.noise_std = noise_std

    def __call__(
        self,
        obs: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        action = self.policy(obs)

        if not deterministic and key is not None and self.noise_std > 0:
            noise = random.normal(key, action.shape) * self.noise_std
            action = jnp.clip(action + noise, 0.0, 1.0)

        return action


def create_mlp_policy(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Tuple[int, ...] = (128, 128),
    key: Optional[jax.random.PRNGKey] = None,
) -> MLPPolicy:
    """Create an MLP policy.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_sizes: Hidden layer sizes.
        key: Random key.

    Returns:
        Initialized MLP policy.
    """
    if key is None:
        key = random.PRNGKey(0)
    return MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        key=key,
    )
