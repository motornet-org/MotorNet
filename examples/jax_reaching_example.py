"""
Example: Training a reaching task with MotorNet-JAX.

This example demonstrates:
1. Creating a 2-DOF arm effector
2. Setting up a reaching environment
3. Training a GRU policy
4. Evaluating the trained policy

Usage:
    python examples/jax_reaching_example.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random
from time import time

# Matplotlib is optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import MotorNet-JAX modules
from motornet_jax.skeleton import TwoDofArm
from motornet_jax.muscle import RigidTendonMuscle
from motornet_jax.effector import Effector
from motornet_jax.environment import Environment
from motornet_jax.policy import GRUPolicy
from motornet_jax.training import Trainer, TrainingConfig
from motornet_jax.types import JointState


def create_simple_arm_effector():
    """Create a simplified arm effector for the example.

    This creates a point-to-point arm model without the full muscle geometry.
    For a complete arm with muscle wrapping, see the full Arm26 implementation.
    """
    # Create skeleton
    skeleton = TwoDofArm()

    # Create muscle model
    # Using simplified parameters for this example
    n_muscles = 6
    muscle = RigidTendonMuscle(
        max_isometric_force=jnp.array([838, 1207, 1422, 1549, 414, 603]),
        optimal_fiber_length=jnp.array([0.134, 0.140, 0.092, 0.093, 0.137, 0.127]),
        tendon_slack_length=jnp.array([0.039, 0.066, 0.172, 0.187, 0.204, 0.217]),
    )

    # Create effector
    effector = Effector(
        skeleton=skeleton,
        muscle=muscle,
        dt=0.01,
        n_ministeps=1,
        damping=0.0,
    )

    return effector


def run_example():
    """Run the reaching task example."""
    print("=" * 60)
    print("MotorNet-JAX: Reaching Task Example")
    print("=" * 60)

    # Set random seed
    key = random.PRNGKey(42)

    # =========================================================================
    # Step 1: Create environment
    # =========================================================================
    print("\n1. Creating arm effector and environment...")

    # For this example, we'll use a simplified forward model
    # In a full implementation, you would use the complete Effector

    # Create skeleton
    skeleton = TwoDofArm()
    skeleton_params = skeleton.get_params()

    # =========================================================================
    # Step 2: Test forward kinematics
    # =========================================================================
    print("\n2. Testing forward kinematics...")

    # Test at a few positions
    test_positions = jnp.array([
        [0.0, 0.0],
        [jnp.pi/4, jnp.pi/4],
        [jnp.pi/2, 0.0],
    ])

    for i, pos in enumerate(test_positions):
        joint_state = JointState(
            position=pos[None, :],
            velocity=jnp.zeros((1, 2)),
        )
        _, hand_pos, _, _ = TwoDofArm.forward_kinematics(joint_state, skeleton_params)
        print(f"  Position {i+1}: angles={jnp.rad2deg(pos)} deg -> "
              f"hand=({hand_pos[0, 0]:.3f}, {hand_pos[0, 1]:.3f}) m")

    # =========================================================================
    # Step 3: Test dynamics
    # =========================================================================
    print("\n3. Testing dynamics simulation...")

    # Simulate a simple trajectory
    n_steps = 100
    dt = 0.01

    # Start from a fixed position
    joint_state = JointState(
        position=jnp.array([[jnp.pi/4, jnp.pi/4]]),
        velocity=jnp.zeros((1, 2)),
    )

    # Apply constant torque
    torques = jnp.array([[0.5, -0.3]])
    endpoint_load = jnp.zeros((1, 2))

    positions = [joint_state.position[0]]
    hand_positions = []

    start_time = time()

    for _ in range(n_steps):
        # Compute acceleration
        _, acceleration = TwoDofArm.ode(
            joint_state, torques, endpoint_load, skeleton_params
        )

        # Integrate
        joint_state = TwoDofArm.integrate(
            joint_state, acceleration, dt, skeleton_params
        )

        positions.append(joint_state.position[0])

        # Get hand position
        _, hand_pos, _, _ = TwoDofArm.forward_kinematics(joint_state, skeleton_params)
        hand_positions.append(hand_pos[0])

    elapsed = time() - start_time
    print(f"  Simulated {n_steps} steps in {elapsed*1000:.2f} ms")
    print(f"  ({n_steps/elapsed:.0f} steps/second)")

    # =========================================================================
    # Step 4: Test JIT compilation speedup
    # =========================================================================
    print("\n4. Testing JIT compilation speedup...")

    # Compile the dynamics functions
    # Note: n_steps must be static for lax.scan
    from functools import partial

    @partial(jax.jit, static_argnums=(2,))
    def simulate_trajectory(initial_pos, torques, n_steps, dt):
        joint_state = JointState(
            position=initial_pos[None, :],
            velocity=jnp.zeros((1, 2)),
        )
        endpoint_load = jnp.zeros((1, 2))

        def step(state, _):
            _, acc = TwoDofArm.ode(state, torques, endpoint_load, skeleton_params)
            new_state = TwoDofArm.integrate(state, acc, dt, skeleton_params)
            return new_state, new_state.position[0]

        final_state, trajectory = jax.lax.scan(step, joint_state, None, length=n_steps)
        return final_state, trajectory

    # Warm up JIT
    _ = simulate_trajectory(
        jnp.array([jnp.pi/4, jnp.pi/4]),
        jnp.array([[0.5, -0.3]]),
        n_steps,  # Use same n_steps for consistency
        0.01,
    )

    # Time the JIT-compiled version
    n_trials = 100
    start_time = time()
    for _ in range(n_trials):
        final_state, trajectory = simulate_trajectory(
            jnp.array([jnp.pi/4, jnp.pi/4]),
            jnp.array([[0.5, -0.3]]),
            n_steps,
            dt,
        )
        trajectory.block_until_ready()  # Ensure computation is complete
    elapsed = time() - start_time

    print(f"  JIT-compiled: {n_trials * n_steps} total steps in {elapsed*1000:.2f} ms")
    print(f"  ({n_trials * n_steps / elapsed:.0f} steps/second)")

    # =========================================================================
    # Step 5: Test batched simulation with vmap
    # =========================================================================
    print("\n5. Testing batched simulation (vmap)...")

    @partial(jax.jit, static_argnums=(2,))
    def simulate_batch(initial_positions, torques, n_steps, dt):
        """Simulate multiple trajectories in parallel."""
        return jax.vmap(
            lambda pos, tau: simulate_trajectory(pos, tau, n_steps, dt)
        )(initial_positions, torques)

    batch_size = 64
    key, init_key = random.split(key)

    # Random initial positions within bounds
    initial_positions = random.uniform(
        init_key,
        (batch_size, 2),
        minval=jnp.array([0.1, 0.1]),
        maxval=jnp.array([1.0, 1.5]),
    )
    batch_torques = jnp.tile(jnp.array([[0.5, -0.3]]), (batch_size, 1))

    # Warm up
    _ = simulate_batch(initial_positions, batch_torques, n_steps, dt)

    # Time it
    n_trials = 100
    start_time = time()
    for _ in range(n_trials):
        final_states, trajectories = simulate_batch(
            initial_positions, batch_torques, n_steps, dt
        )
        trajectories.block_until_ready()
    elapsed = time() - start_time

    total_steps = n_trials * batch_size * n_steps
    print(f"  Batched: {total_steps} total steps in {elapsed*1000:.2f} ms")
    print(f"  ({total_steps / elapsed:.0f} steps/second)")
    print(f"  Batch size: {batch_size}")

    # =========================================================================
    # Step 6: Create and test policy network
    # =========================================================================
    print("\n6. Testing GRU policy network...")

    obs_dim = 10  # Example observation dimension
    action_dim = 6  # 6 muscles

    key, policy_key = random.split(key)
    policy = GRUPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=64,
        n_gru_layers=1,
        key=policy_key,
    )

    # Test forward pass
    batch_size = 32
    obs = random.normal(key, (batch_size, obs_dim))
    hidden = policy.init_hidden(batch_size)

    start_time = time()
    action, new_hidden = policy(obs, hidden)
    elapsed = time() - start_time

    print(f"  Policy forward pass: {elapsed*1000:.2f} ms")
    print(f"  Action shape: {action.shape}")
    print(f"  Hidden shape: {new_hidden.shape}")
    print(f"  Action range: [{float(action.min()):.3f}, {float(action.max()):.3f}]")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
MotorNet-JAX provides a high-performance simulation framework for motor control.

Key features demonstrated:
1. Forward kinematics with TwoDofArm
2. Dynamics simulation with numerical integration
3. JIT compilation for significant speedups
4. Batched simulation with vmap for parallel environments
5. GRU policy network for motor control

For full training examples, see:
- examples/train_reaching.py (once implemented)
- Documentation at docs/

Performance benefits over PyTorch:
- XLA compilation of entire simulation graph
- Automatic vectorization with vmap
- lax.scan for efficient trajectory rollouts
- Gradient computation through full simulation
""")


if __name__ == "__main__":
    run_example()
