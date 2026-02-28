"""
Numerical integration methods.

Provides Euler and Runge-Kutta 4 integrators using JAX's lax.scan
for efficient trajectory rollouts.
"""

from typing import Callable, Tuple, Any
import jax
import jax.numpy as jnp
from jax import lax


def euler_step(
    state: Any,
    derivative: Any,
    dt: float,
) -> Any:
    """Perform one Euler integration step.

    Works with any pytree state structure (NamedTuples, dicts, etc.)

    Args:
        state: Current state (pytree).
        derivative: State derivative (same structure as state).
        dt: Timestep.

    Returns:
        New state after integration.
    """
    return jax.tree.map(lambda s, d: s + dt * d, state, derivative)


def rk4_step(
    state: Any,
    ode_fn: Callable,
    dt: float,
    *ode_args,
) -> Any:
    """Perform one RK4 integration step.

    Args:
        state: Current state (pytree).
        ode_fn: Function that computes derivatives: ode_fn(state, *ode_args) -> derivative.
        dt: Timestep.
        *ode_args: Additional arguments to pass to ode_fn.

    Returns:
        New state after integration.
    """
    half_dt = dt / 2

    k1 = ode_fn(state, *ode_args)
    state1 = jax.tree.map(lambda s, k: s + half_dt * k, state, k1)

    k2 = ode_fn(state1, *ode_args)
    state2 = jax.tree.map(lambda s, k: s + half_dt * k, state, k2)

    k3 = ode_fn(state2, *ode_args)
    state3 = jax.tree.map(lambda s, k: s + dt * k, state, k3)

    k4 = ode_fn(state3, *ode_args)

    # Combine: (k1 + 2*k2 + 2*k3 + k4) / 6
    def combine_k(k1, k2, k3, k4):
        return (k1 + 2*k2 + 2*k3 + k4) / 6

    k_combined = jax.tree.map(combine_k, k1, k2, k3, k4)

    return jax.tree.map(lambda s, k: s + dt * k, state, k_combined)


def integrate_trajectory(
    initial_state: Any,
    step_fn: Callable,
    n_steps: int,
    *step_args,
) -> Tuple[Any, Any]:
    """Integrate a trajectory using lax.scan.

    This is the most efficient way to compute trajectories in JAX,
    as it compiles the entire loop into a single XLA computation.

    Args:
        initial_state: Initial state (pytree).
        step_fn: Function that takes (state, step_args) and returns (new_state, output).
        n_steps: Number of steps to integrate.
        *step_args: Additional arguments to pass to step_fn at each step.

    Returns:
        final_state: Final state after n_steps.
        trajectory: Stacked outputs from each step.
    """
    def scan_fn(state, _):
        new_state, output = step_fn(state, *step_args)
        return new_state, output

    final_state, trajectory = lax.scan(scan_fn, initial_state, None, length=n_steps)

    return final_state, trajectory


def integrate_with_inputs(
    initial_state: Any,
    step_fn: Callable,
    inputs: jnp.ndarray,
    *step_args,
) -> Tuple[Any, Any]:
    """Integrate a trajectory with time-varying inputs.

    Args:
        initial_state: Initial state (pytree).
        step_fn: Function that takes (state, input, *step_args) and returns (new_state, output).
        inputs: Array of inputs, one per timestep. Shape: (n_steps, ...).
        *step_args: Additional arguments to pass to step_fn at each step.

    Returns:
        final_state: Final state after all steps.
        trajectory: Stacked outputs from each step.
    """
    def scan_fn(state, inp):
        new_state, output = step_fn(state, inp, *step_args)
        return new_state, output

    final_state, trajectory = lax.scan(scan_fn, initial_state, inputs)

    return final_state, trajectory


def rollout_policy(
    initial_state: Any,
    policy_fn: Callable,
    env_step_fn: Callable,
    n_steps: int,
    policy_params: Any,
    env_params: Any,
    initial_hidden: Any = None,
) -> Tuple[Any, Any, Any]:
    """Roll out a policy in an environment.

    This integrates policy execution and environment stepping
    into a single efficient lax.scan loop.

    Args:
        initial_state: Initial environment state.
        policy_fn: Policy function (obs, hidden, params) -> (action, new_hidden).
        env_step_fn: Environment step function (state, action, params) -> (new_state, obs, reward).
        n_steps: Number of steps to roll out.
        policy_params: Policy parameters.
        env_params: Environment parameters.
        initial_hidden: Initial hidden state for recurrent policy (optional).

    Returns:
        final_state: Final environment state.
        final_hidden: Final policy hidden state.
        trajectory: Dict with 'states', 'actions', 'rewards'.
    """
    def step_fn(carry, _):
        state, hidden = carry

        # Get observation from state
        obs = state  # Simplified; actual implementation would extract obs

        # Get action from policy
        action, new_hidden = policy_fn(obs, hidden, policy_params)

        # Step environment
        new_state, reward = env_step_fn(state, action, env_params)

        output = {
            'state': state,
            'action': action,
            'reward': reward,
        }

        return (new_state, new_hidden), output

    init_carry = (initial_state, initial_hidden)
    (final_state, final_hidden), trajectory = lax.scan(
        step_fn, init_carry, None, length=n_steps
    )

    return final_state, final_hidden, trajectory


# Specialized integrators for common patterns

def integrate_ministeps(
    state: Any,
    action: Any,
    single_step_fn: Callable,
    n_ministeps: int,
    *step_args,
) -> Any:
    """Perform multiple integration ministeps with constant action.

    This is common in motor control where the action is held constant
    for a short period while the physics is integrated at a finer timescale.

    Args:
        state: Initial state.
        action: Action to apply (held constant across ministeps).
        single_step_fn: Function (state, action, *args) -> new_state.
        n_ministeps: Number of ministeps.
        *step_args: Additional arguments to pass to single_step_fn.

    Returns:
        Final state after all ministeps.
    """
    def body_fn(i, state):
        return single_step_fn(state, action, *step_args)

    return lax.fori_loop(0, n_ministeps, body_fn, state)


def batched_integrate(
    initial_states: Any,
    step_fn: Callable,
    n_steps: int,
    *step_args,
) -> Tuple[Any, Any]:
    """Integrate multiple trajectories in parallel using vmap.

    Args:
        initial_states: Batch of initial states.
        step_fn: Single-trajectory step function.
        n_steps: Number of steps.
        *step_args: Arguments to step_fn.

    Returns:
        Batch of final states and trajectories.
    """
    # vmap the trajectory integration
    batched_fn = jax.vmap(
        lambda init: integrate_trajectory(init, step_fn, n_steps, *step_args),
        in_axes=0,
    )
    return batched_fn(initial_states)
