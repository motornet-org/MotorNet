"""
Training utilities for MotorNet-JAX.

Provides training loops and utilities using Optax for optimization.
"""

from typing import NamedTuple, Tuple, Dict, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, lax, random
import optax
import equinox as eqx
from functools import partial

from motornet_jax.training.losses import combined_loss


class TrainingConfig(NamedTuple):
    """Training configuration.

    Attributes:
        learning_rate: Learning rate.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        n_steps: Number of environment steps per episode.
        position_weight: Weight for position loss.
        effort_weight: Weight for effort loss.
        smoothness_weight: Weight for action smoothness loss.
        gradient_clip: Maximum gradient norm (0 = no clipping).
    """
    learning_rate: float = 1e-3
    n_epochs: int = 1000
    batch_size: int = 32
    n_steps: int = 100
    position_weight: float = 1.0
    effort_weight: float = 0.01
    smoothness_weight: float = 0.0
    gradient_clip: float = 1.0


class TrainState(NamedTuple):
    """Training state.

    Attributes:
        policy: Current policy parameters (Equinox module).
        opt_state: Optimizer state.
        step: Current training step.
        key: Random key.
    """
    policy: Any  # Equinox module
    opt_state: Any  # Optax optimizer state
    step: int
    key: jax.random.PRNGKey


def create_optimizer(
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Create an optimizer with optional gradient clipping.

    Args:
        learning_rate: Learning rate.
        gradient_clip: Maximum gradient norm (0 = no clipping).
        weight_decay: Weight decay coefficient.

    Returns:
        Optax optimizer.
    """
    transforms = []

    if gradient_clip > 0:
        transforms.append(optax.clip_by_global_norm(gradient_clip))

    if weight_decay > 0:
        transforms.append(optax.add_decayed_weights(weight_decay))

    transforms.append(optax.adam(learning_rate))

    return optax.chain(*transforms)


def create_train_state(
    policy,
    config: TrainingConfig,
    key: jax.random.PRNGKey,
) -> TrainState:
    """Create initial training state.

    Args:
        policy: Initial policy (Equinox module).
        config: Training configuration.
        key: Random key.

    Returns:
        Initial training state.
    """
    optimizer = create_optimizer(
        learning_rate=config.learning_rate,
        gradient_clip=config.gradient_clip,
    )
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    return TrainState(
        policy=policy,
        opt_state=opt_state,
        step=0,
        key=key,
    )


def compute_loss(
    policy,
    env,
    targets: jnp.ndarray,
    initial_states: Optional[jnp.ndarray],
    config: TrainingConfig,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute loss for a batch.

    This runs full episodes with the policy and computes the loss.

    Args:
        policy: Policy network.
        env: Environment.
        targets: Target positions. Shape: (batch, n_dim)
        initial_states: Initial joint states. Shape: (batch, state_dim) or None.
        config: Training configuration.
        key: Random key.

    Returns:
        loss: Scalar loss.
        metrics: Dict with additional metrics.
    """
    batch_size = targets.shape[0]

    key, reset_key = random.split(key)

    # Reset environment
    env_state, obs = env.reset(reset_key, batch_size)

    # Set targets
    env_state = env_state._replace(target=targets[:, None, :])

    # Initialize hidden state
    hidden = policy.init_hidden(batch_size)

    # Collect trajectory
    def step_fn(carry, step_key):
        state, obs, hidden = carry

        # Get action
        action, new_hidden = policy(obs, hidden)

        # Step environment
        new_state, new_obs, reward, done, info = env.step(state, action, step_key)

        output = {
            'fingertip': state.effector.fingertip,
            'action': action,
            'reward': reward,
        }

        return (new_state, new_obs, new_hidden), output

    # Generate step keys
    step_keys = random.split(key, config.n_steps)

    # Run trajectory
    _, trajectory = lax.scan(
        step_fn,
        (env_state, obs, hidden),
        step_keys,
    )

    # Compute loss
    loss_config = {
        'position_weight': config.position_weight,
        'effort_weight': config.effort_weight,
        'smoothness_weight': config.smoothness_weight,
    }

    loss = combined_loss(trajectory, targets, loss_config)

    # Compute metrics
    final_distance = jnp.mean(jnp.linalg.norm(
        trajectory['fingertip'][-1] - targets, axis=-1
    ))
    mean_reward = jnp.mean(trajectory['reward'])

    metrics = {
        'loss': loss,
        'final_distance': final_distance,
        'mean_reward': mean_reward,
    }

    return loss, metrics


def train_step(
    state: TrainState,
    env,
    targets: jnp.ndarray,
    config: TrainingConfig,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Perform one training step.

    Args:
        state: Current training state.
        env: Environment.
        targets: Target positions. Shape: (batch, n_dim)
        config: Training configuration.

    Returns:
        new_state: Updated training state.
        metrics: Training metrics.
    """
    key, step_key = random.split(state.key)

    # Compute loss and gradients
    def loss_fn(policy):
        loss, metrics = compute_loss(
            policy, env, targets, None, config, step_key
        )
        return loss, metrics

    # Get gradients (only for array parameters)
    policy_arrays, policy_static = eqx.partition(state.policy, eqx.is_array)

    def loss_fn_arrays(arrays):
        policy = eqx.combine(arrays, policy_static)
        return loss_fn(policy)

    (loss, metrics), grads = value_and_grad(loss_fn_arrays, has_aux=True)(policy_arrays)

    # Update parameters
    optimizer = create_optimizer(config.learning_rate, config.gradient_clip)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, policy_arrays)
    new_policy_arrays = optax.apply_updates(policy_arrays, updates)
    new_policy = eqx.combine(new_policy_arrays, policy_static)

    new_state = TrainState(
        policy=new_policy,
        opt_state=new_opt_state,
        step=state.step + 1,
        key=key,
    )

    return new_state, metrics


class Trainer:
    """High-level trainer class.

    Provides a simple interface for training policies on motor control tasks.
    """

    def __init__(
        self,
        env,
        policy,
        config: Optional[TrainingConfig] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        """Initialize trainer.

        Args:
            env: Environment instance.
            policy: Initial policy.
            config: Training configuration.
            key: Random key.
        """
        self.env = env
        self.config = config or TrainingConfig()

        if key is None:
            key = random.PRNGKey(0)

        self.state = create_train_state(policy, self.config, key)
        self.history = []

    def train_epoch(self, n_batches: int = 10) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            n_batches: Number of batches per epoch.

        Returns:
            Dict with mean metrics for the epoch.
        """
        epoch_metrics = []

        for _ in range(n_batches):
            # Generate random targets
            self.state = self.state._replace(
                key=random.split(self.state.key)[0]
            )
            key, target_key = random.split(self.state.key)

            targets = random.uniform(
                target_key,
                (self.config.batch_size, 2),
                minval=jnp.array([0.1, -0.3]),
                maxval=jnp.array([0.5, 0.3]),
            )

            # Training step
            self.state, metrics = train_step(
                self.state, self.env, targets, self.config
            )

            epoch_metrics.append(metrics)

        # Average metrics
        mean_metrics = {
            k: float(jnp.mean(jnp.array([m[k] for m in epoch_metrics])))
            for k in epoch_metrics[0].keys()
        }

        self.history.append(mean_metrics)
        return mean_metrics

    def train(
        self,
        n_epochs: Optional[int] = None,
        n_batches_per_epoch: int = 10,
        print_every: int = 10,
    ) -> None:
        """Train for multiple epochs.

        Args:
            n_epochs: Number of epochs (defaults to config.n_epochs).
            n_batches_per_epoch: Batches per epoch.
            print_every: Print frequency.
        """
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        for epoch in range(n_epochs):
            metrics = self.train_epoch(n_batches_per_epoch)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: "
                      f"loss={metrics['loss']:.4f}, "
                      f"distance={metrics['final_distance']:.4f}")

    def get_policy(self):
        """Get the trained policy."""
        return self.state.policy

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the policy.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Dict with evaluation metrics.
        """
        all_metrics = []

        for _ in range(n_episodes):
            key, eval_key = random.split(self.state.key)
            self.state = self.state._replace(key=key)

            # Random target
            key, target_key = random.split(key)
            targets = random.uniform(
                target_key,
                (1, 2),
                minval=jnp.array([0.1, -0.3]),
                maxval=jnp.array([0.5, 0.3]),
            )

            # Evaluate
            _, metrics = compute_loss(
                self.state.policy,
                self.env,
                targets,
                None,
                self.config,
                eval_key,
            )
            all_metrics.append(metrics)

        return {
            k: float(jnp.mean(jnp.array([m[k] for m in all_metrics])))
            for k in all_metrics[0].keys()
        }
