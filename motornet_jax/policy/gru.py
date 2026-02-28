"""
GRU-based policy network using Equinox.

Provides a recurrent policy for motor control tasks.
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx


class GRUPolicyParams(NamedTuple):
    """Parameters structure for GRU policy.

    This is a convenience wrapper - actual parameters are stored
    in the Equinox module as pytree leaves.
    """
    hidden_size: int
    n_gru_layers: int


class GRUPolicy(eqx.Module):
    """GRU-based recurrent policy network.

    Architecture:
    - Input projection (obs_dim -> hidden_size)
    - GRU layers
    - Output projection (hidden_size -> action_dim)
    - Sigmoid activation for bounded actions [0, 1]

    This uses Equinox for a clean, functional approach to neural networks
    that works seamlessly with JAX transformations.
    """

    input_layer: eqx.nn.Linear
    gru_layers: list
    output_layer: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    n_gru_layers: int = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        n_gru_layers: int = 1,
        key: jax.random.PRNGKey = None,
    ):
        """Initialize GRU policy.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            hidden_size: Hidden layer size.
            n_gru_layers: Number of GRU layers.
            key: Random key for initialization.
        """
        if key is None:
            key = random.PRNGKey(0)

        self.hidden_size = hidden_size
        self.n_gru_layers = n_gru_layers

        keys = random.split(key, n_gru_layers + 2)

        # Input projection
        self.input_layer = eqx.nn.Linear(obs_dim, hidden_size, key=keys[0])

        # GRU layers
        self.gru_layers = []
        for i in range(n_gru_layers):
            gru = eqx.nn.GRUCell(hidden_size, hidden_size, key=keys[i + 1])
            self.gru_layers.append(gru)

        # Output projection
        self.output_layer = eqx.nn.Linear(hidden_size, action_dim, key=keys[-1])

    def _single_forward(
        self,
        obs_single: jnp.ndarray,
        hidden_single: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for a single sample (no batch dim).

        Args:
            obs_single: Observation. Shape: (obs_dim,)
            hidden_single: Hidden state. Shape: (n_layers, hidden_size)

        Returns:
            action: Action. Shape: (action_dim,)
            new_hidden: Updated hidden state. Shape: (n_layers, hidden_size)
        """
        # Input projection with ReLU
        x = jax.nn.relu(self.input_layer(obs_single))  # (hidden_size,)

        # Process through GRU layers
        new_hidden_list = []
        for i, gru in enumerate(self.gru_layers):
            h = hidden_single[i]  # (hidden_size,)
            x = gru(x, h)  # (hidden_size,)
            new_hidden_list.append(x)

        new_hidden = jnp.stack(new_hidden_list, axis=0)  # (n_layers, hidden_size)

        # Output projection with sigmoid for bounded actions
        action = jax.nn.sigmoid(self.output_layer(x))  # (action_dim,)

        return action, new_hidden

    def __call__(
        self,
        obs: jnp.ndarray,
        hidden: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: Observation. Shape: (obs_dim,) or (batch, obs_dim)
            hidden: Hidden state. Shape: (n_layers, hidden_size) or (batch, n_layers, hidden_size)

        Returns:
            action: Action. Shape: same batch dims as input + (action_dim,)
            new_hidden: Updated hidden state. Shape: same as input hidden.
        """
        # Handle unbatched input
        batched = obs.ndim > 1
        if not batched:
            obs = obs[None, :]
            if hidden is not None:
                hidden = hidden[None, :, :]

        batch_size = obs.shape[0]

        # Initialize hidden if not provided
        if hidden is None:
            hidden = jnp.zeros((batch_size, self.n_gru_layers, self.hidden_size))

        # Single vmap over entire forward pass
        action, new_hidden = jax.vmap(self._single_forward)(obs, hidden)

        if not batched:
            action = action[0]
            new_hidden = new_hidden[0]

        return action, new_hidden

    def init_hidden(self, batch_size: int = 1) -> jnp.ndarray:
        """Initialize hidden state.

        Args:
            batch_size: Batch size.

        Returns:
            Initial hidden state. Shape: (batch, n_layers, hidden_size)
        """
        return jnp.zeros((batch_size, self.n_gru_layers, self.hidden_size))


class GRUPolicyWithNoise(eqx.Module):
    """GRU policy with exploration noise.

    Adds Gaussian noise to actions during training for exploration.
    """

    policy: GRUPolicy
    noise_std: float = eqx.field(static=True)

    def __init__(self, policy: GRUPolicy, noise_std: float = 0.1):
        """Initialize.

        Args:
            policy: Base GRU policy.
            noise_std: Standard deviation of action noise.
        """
        self.policy = policy
        self.noise_std = noise_std

    def __call__(
        self,
        obs: jnp.ndarray,
        hidden: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass with optional noise.

        Args:
            obs: Observation.
            hidden: Hidden state.
            key: Random key for noise.
            deterministic: If True, don't add noise.

        Returns:
            action: Action (potentially noisy).
            new_hidden: Updated hidden state.
        """
        action, new_hidden = self.policy(obs, hidden)

        if not deterministic and key is not None and self.noise_std > 0:
            noise = random.normal(key, action.shape) * self.noise_std
            action = jnp.clip(action + noise, 0.0, 1.0)

        return action, new_hidden


# Utility functions

def create_policy(
    obs_dim: int,
    action_dim: int,
    hidden_size: int = 128,
    n_gru_layers: int = 1,
    key: Optional[jax.random.PRNGKey] = None,
) -> GRUPolicy:
    """Create a GRU policy.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_size: Hidden size.
        n_gru_layers: Number of GRU layers.
        key: Random key.

    Returns:
        Initialized GRU policy.
    """
    if key is None:
        key = random.PRNGKey(0)
    return GRUPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        n_gru_layers=n_gru_layers,
        key=key,
    )


def policy_rollout_step(
    policy: GRUPolicy,
    obs: jnp.ndarray,
    hidden: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convenience function for policy rollout.

    Args:
        policy: Policy network.
        obs: Current observation.
        hidden: Current hidden state.

    Returns:
        action: Selected action.
        new_hidden: Updated hidden state.
    """
    return policy(obs, hidden)
