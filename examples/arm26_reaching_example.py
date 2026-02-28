"""
Arm26 Reaching Example - demonstrates the 6-muscle arm model.

This example shows how to:
1. Create and simulate the Arm26 effector
2. Visualize trajectories
3. Use the model with different muscle activations
"""

import os
import sys

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from functools import partial

from motornet_jax.effector import Arm26
from motornet_jax.types import JointState

print("Arm26 6-Muscle Arm Example")
print("=" * 50)

# Create Arm26 effector
arm = Arm26(dt=0.01, n_ministeps=1)
params = arm.get_params()

print(f"\nArm26 Configuration:")
print(f"  Number of muscles: {arm.n_muscles}")
print(f"  Muscle names: {arm.muscle_names}")
print(f"  Timestep: {params.dt}s")
print(f"  Max isometric forces: {params.max_isometric_force}")

# Initialize state
batch_size = 1
key = jax.random.PRNGKey(42)

# Start from a specific joint position
initial_joint = JointState(
    position=jnp.array([[0.8, 0.9]]),  # Shoulder and elbow angles in radians
    velocity=jnp.zeros((1, 2)),
)
state = arm.reset(batch_size=1, joint_state=initial_joint)

print(f"\nInitial state:")
print(f"  Joint angles: {state.joint.position[0]}")
print(f"  Hand position: {state.fingertip[0]}")

# Define different muscle activation patterns
activation_patterns = {
    "flexors": jnp.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]),  # Flex shoulder and elbow
    "extensors": jnp.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),  # Extend shoulder and elbow
    "shoulder_flex": jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),  # Just shoulder flexor
    "elbow_flex": jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),  # Just elbow flexor
    "cocontraction": jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]),  # All muscles moderately
}

# Simulate with different patterns
n_steps = 200
endpoint_load = jnp.zeros((1, 2))
joint_load = jnp.zeros((1, 2))

@partial(jax.jit, static_argnums=(4,))
def simulate_trajectory(state, action, endpoint_load, joint_load, n_steps):
    """Simulate and record trajectory."""
    def step_fn(state, _):
        new_state = Arm26.step(state, action, endpoint_load, joint_load, params)
        return new_state, (new_state.fingertip[0], new_state.joint.position[0])

    final_state, (hand_traj, joint_traj) = jax.lax.scan(step_fn, state, None, length=n_steps)
    return final_state, hand_traj, joint_traj

print("\n" + "=" * 50)
print("Simulating different activation patterns...")
print("=" * 50)

trajectories = {}
for name, action in activation_patterns.items():
    state = arm.reset(batch_size=1, joint_state=initial_joint)
    final_state, hand_traj, joint_traj = simulate_trajectory(
        state, action, endpoint_load, joint_load, n_steps
    )
    trajectories[name] = {
        "hand": hand_traj,
        "joint": joint_traj,
        "final_hand": final_state.fingertip[0],
    }
    print(f"\n{name}:")
    print(f"  Final hand position: ({trajectories[name]['final_hand'][0]:.4f}, {trajectories[name]['final_hand'][1]:.4f})")
    print(f"  Final joint angles: ({final_state.joint.position[0, 0]:.4f}, {final_state.joint.position[0, 1]:.4f}) rad")

# Try to plot if matplotlib is available
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot hand trajectories
    ax = axes[0]
    for name, traj in trajectories.items():
        ax.plot(traj["hand"][:, 0], traj["hand"][:, 1], label=name, linewidth=2)
    ax.scatter([initial_joint.position[0, 0]], [initial_joint.position[0, 1]],
               c='green', s=100, marker='o', label='Start', zorder=5)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Hand Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot joint angle trajectories (shoulder)
    ax = axes[1]
    t = jnp.arange(n_steps) * params.dt
    for name, traj in trajectories.items():
        ax.plot(t, jnp.rad2deg(traj["joint"][:, 0]), label=name, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Shoulder angle (deg)")
    ax.set_title("Shoulder Joint Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot joint angle trajectories (elbow)
    ax = axes[2]
    for name, traj in trajectories.items():
        ax.plot(t, jnp.rad2deg(traj["joint"][:, 1]), label=name, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Elbow angle (deg)")
    ax.set_title("Elbow Joint Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("arm26_trajectories.png", dpi=150)
    print("\n\nPlot saved to arm26_trajectories.png")

except ImportError:
    print("\n\nmatplotlib not available - skipping visualization")

# Performance benchmark
print("\n" + "=" * 50)
print("Performance Benchmark")
print("=" * 50)

import time

batch_size = 64
n_steps = 100
n_episodes = 1000

arm = Arm26(dt=0.01, n_ministeps=1)
params = arm.get_params()

key = jax.random.PRNGKey(0)
state = arm.reset(batch_size=batch_size, key=key)
action = jnp.ones((batch_size, 6)) * 0.3
endpoint_load = jnp.zeros((batch_size, 2))
joint_load = jnp.zeros((batch_size, 2))

@jax.jit
def run_episode(state):
    def step_fn(state, _):
        new_state = Arm26.step(state, action, endpoint_load, joint_load, params)
        return new_state, None
    final_state, _ = jax.lax.scan(step_fn, state, None, length=n_steps)
    return final_state

# Warmup
state = arm.reset(batch_size=batch_size, key=key)
_ = run_episode(state)
_.joint.position.block_until_ready()

# Benchmark
start = time.perf_counter()
for _ in range(n_episodes):
    state = arm.reset(batch_size=batch_size, key=key)
    final = run_episode(state)
final.joint.position.block_until_ready()
elapsed = time.perf_counter() - start

total_steps = n_episodes * n_steps * batch_size
steps_per_sec = total_steps / elapsed
episodes_per_sec = n_episodes / elapsed

print(f"\nBatch size: {batch_size}")
print(f"Steps per episode: {n_steps}")
print(f"Total episodes: {n_episodes}")
print(f"Total steps: {total_steps:,}")
print(f"Time elapsed: {elapsed:.2f}s")
print(f"\nPerformance:")
print(f"  Steps per second: {steps_per_sec:,.0f}")
print(f"  Episodes per second: {episodes_per_sec:,.1f}")
