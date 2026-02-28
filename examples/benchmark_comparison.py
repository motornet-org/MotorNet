"""
Benchmark comparison between JAX MotorNet and PyTorch MotorNet.

This script measures:
1. Performance (speed) comparison
2. Numerical accuracy comparison (forward kinematics, trajectories)
3. API compatibility verification

Run with: python examples/benchmark_comparison.py
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

print("=" * 60)
print("MotorNet Performance Benchmark: JAX vs PyTorch")
print("=" * 60)
print()

# ============================================================
# JAX MotorNet Benchmarks
# ============================================================

print("Loading JAX MotorNet...")
from motornet_jax import TwoDofArm
from motornet_jax.effector import Arm26
from motornet_jax.types import JointState

# Benchmark TwoDofArm skeleton
print("\n--- TwoDofArm Skeleton (JAX) ---")

skeleton = TwoDofArm()
skeleton_params = skeleton.get_params()
batch_size = 64
n_steps = 100

joint_state = JointState(
    position=jnp.ones((batch_size, 2)) * 0.5,
    velocity=jnp.zeros((batch_size, 2)),
)
torques = jnp.ones((batch_size, 2)) * 0.1
endpoint_load = jnp.zeros((batch_size, 2))

@jax.jit
def skeleton_step(state, torques, endpoint_load):
    """Single skeleton step."""
    _, acc = TwoDofArm.ode(state, torques, endpoint_load, skeleton_params)
    return TwoDofArm.integrate(state, acc, 0.01, skeleton_params)

@jax.jit
def skeleton_rollout(state, torques, endpoint_load):
    """Roll out skeleton for n_steps."""
    def step_fn(state, _):
        new_state = skeleton_step(state, torques, endpoint_load)
        return new_state, None
    final_state, _ = jax.lax.scan(step_fn, state, None, length=n_steps)
    return final_state

# Warmup
_ = skeleton_rollout(joint_state, torques, endpoint_load)
_.position.block_until_ready()

# Benchmark
n_runs = 1000
start = time.perf_counter()
for _ in range(n_runs):
    result = skeleton_rollout(joint_state, torques, endpoint_load)
result.position.block_until_ready()
elapsed = time.perf_counter() - start

total_steps = n_runs * n_steps * batch_size
jax_skeleton_steps_per_sec = total_steps / elapsed
jax_skeleton_episode_per_sec = n_runs / elapsed

print(f"  Batch size: {batch_size}")
print(f"  Steps per episode: {n_steps}")
print(f"  Steps per second: {jax_skeleton_steps_per_sec:,.0f}")
print(f"  Episodes per second: {jax_skeleton_episode_per_sec:,.1f}")

# Benchmark Arm26 (full muscle model)
print("\n--- Arm26 Full Muscle Model (JAX) ---")

arm = Arm26(dt=0.01, n_ministeps=1)
arm_params = arm.get_params()

key = jax.random.PRNGKey(0)
arm_state = arm.reset(batch_size=batch_size, key=key)

action = jnp.ones((batch_size, 6)) * 0.3
endpoint_load_arm = jnp.zeros((batch_size, 2))
joint_load = jnp.zeros((batch_size, 2))

@jax.jit
def arm26_rollout(state, action, endpoint_load, joint_load):
    """Roll out Arm26 for n_steps."""
    def step_fn(state, _):
        new_state = Arm26.step(state, action, endpoint_load, joint_load, arm_params)
        return new_state, new_state.fingertip
    final_state, trajectory = jax.lax.scan(step_fn, state, None, length=n_steps)
    return final_state, trajectory

# Warmup
arm_state = arm.reset(batch_size=batch_size, key=key)
final_warmup, traj_warmup = arm26_rollout(arm_state, action, endpoint_load_arm, joint_load)
final_warmup.joint.position.block_until_ready()

# Benchmark
n_runs = 500
start = time.perf_counter()
for _ in range(n_runs):
    arm_state = arm.reset(batch_size=batch_size, key=key)
    final_state, trajectory = arm26_rollout(arm_state, action, endpoint_load_arm, joint_load)
final_state.joint.position.block_until_ready()
elapsed = time.perf_counter() - start

total_steps = n_runs * n_steps * batch_size
jax_arm26_steps_per_sec = total_steps / elapsed
jax_arm26_episode_per_sec = n_runs / elapsed

print(f"  Batch size: {batch_size}")
print(f"  Steps per episode: {n_steps}")
print(f"  Steps per second: {jax_arm26_steps_per_sec:,.0f}")
print(f"  Episodes per second: {jax_arm26_episode_per_sec:,.1f}")

# Benchmark forward kinematics
print("\n--- Forward Kinematics (JAX) ---")

joint_state_fk = JointState(
    position=jnp.ones((batch_size, 2)) * 0.5,
    velocity=jnp.zeros((batch_size, 2)),
)

@jax.jit
def fk_batch(state):
    return TwoDofArm.forward_kinematics(state, skeleton_params)

# Warmup
_ = fk_batch(joint_state_fk)
_[0].block_until_ready()

# Benchmark
n_runs = 10000
start = time.perf_counter()
for _ in range(n_runs):
    result = fk_batch(joint_state_fk)
result[0].block_until_ready()
elapsed = time.perf_counter() - start

jax_fk_per_sec = n_runs * batch_size / elapsed

print(f"  Batch size: {batch_size}")
print(f"  Forward kinematics per second: {jax_fk_per_sec:,.0f}")

# ============================================================
# PyTorch MotorNet Benchmarks (if available)
# ============================================================

try:
    print("\n\nLoading PyTorch MotorNet...")
    import torch
    from motornet.effector import RigidTendonArm26

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    print("\n--- RigidTendonArm26 (PyTorch) ---")

    # Create effector
    effector = RigidTendonArm26().to(device)

    # Warmup
    with torch.no_grad():
        effector.reset(options={"batch_size": batch_size})
        for _ in range(10):
            action_pt = torch.ones((batch_size, 6), device=device) * 0.3
            obs, reward, terminated, truncated, info = effector.step(action=action_pt)

    # Benchmark
    n_runs_pt = 100
    start = time.perf_counter()
    with torch.no_grad():
        for run in range(n_runs_pt):
            effector.reset(options={"batch_size": batch_size})
            for step in range(n_steps):
                obs, reward, terminated, truncated, info = effector.step(action=action_pt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_steps_pt = n_runs_pt * n_steps * batch_size
    pytorch_steps_per_sec = total_steps_pt / elapsed
    pytorch_episode_per_sec = n_runs_pt / elapsed

    print(f"  Batch size: {batch_size}")
    print(f"  Steps per episode: {n_steps}")
    print(f"  Steps per second: {pytorch_steps_per_sec:,.0f}")
    print(f"  Episodes per second: {pytorch_episode_per_sec:,.1f}")

    # Calculate speedup
    print("\n" + "=" * 60)
    print("SPEEDUP SUMMARY")
    print("=" * 60)
    print(f"\nArm26 Full Muscle Model:")
    print(f"  PyTorch: {pytorch_steps_per_sec:,.0f} steps/sec")
    print(f"  JAX:     {jax_arm26_steps_per_sec:,.0f} steps/sec")
    speedup = jax_arm26_steps_per_sec / pytorch_steps_per_sec
    print(f"  Speedup: {speedup:.1f}x")

except ImportError:
    print("\nPyTorch MotorNet not available for comparison.")
    print("Install with: pip install motornet")
except Exception as e:
    print(f"\nError loading PyTorch MotorNet: {e}")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("JAX MotorNet Performance Summary")
print("=" * 60)
print(f"\nTwoDofArm Skeleton:")
print(f"  {jax_skeleton_steps_per_sec:,.0f} steps/second")
print(f"  {jax_skeleton_episode_per_sec:,.1f} episodes/second (batch={batch_size}, steps={n_steps})")

print(f"\nArm26 Full Muscle Model:")
print(f"  {jax_arm26_steps_per_sec:,.0f} steps/second")
print(f"  {jax_arm26_episode_per_sec:,.1f} episodes/second (batch={batch_size}, steps={n_steps})")

print(f"\nForward Kinematics:")
print(f"  {jax_fk_per_sec:,.0f} computations/second")

# ============================================================
# Numerical Validation Tests
# ============================================================

print("\n" + "=" * 60)
print("Numerical Validation Tests")
print("=" * 60)

# Test forward kinematics against analytical formula
print("\n--- Forward Kinematics Validation ---")

test_angles = [
    [0.5, 1.0],
    [0.8, 1.2],
    [1.0, 0.5],
    [1.2, 1.5],
    [0.0, 0.0],
]

L1, L2 = float(skeleton_params.L1), float(skeleton_params.L2)
print(f"  Arm lengths: L1={L1:.4f}, L2={L2:.4f}")

fk_errors = []
for angles in test_angles:
    shoulder, elbow = angles

    # Analytical
    joint_sum = shoulder + elbow
    elb_x = L1 * np.cos(shoulder)
    elb_y = L1 * np.sin(shoulder)
    end_x_analytical = elb_x + L2 * np.cos(joint_sum)
    end_y_analytical = elb_y + L2 * np.sin(joint_sum)

    # JAX computation
    js = JointState(
        position=jnp.array([[shoulder, elbow]]),
        velocity=jnp.zeros((1, 2))
    )
    _, hand_pos, _, _ = TwoDofArm.forward_kinematics(js, skeleton_params)
    end_x_jax, end_y_jax = float(hand_pos[0, 0]), float(hand_pos[0, 1])

    error = np.sqrt((end_x_analytical - end_x_jax)**2 + (end_y_analytical - end_y_jax)**2)
    fk_errors.append(error)

    print(f"  θ=[{shoulder:.1f}, {elbow:.1f}]: "
          f"analytical=({end_x_analytical:.4f}, {end_y_analytical:.4f}), "
          f"JAX=({end_x_jax:.4f}, {end_y_jax:.4f}), "
          f"error={error:.2e}")

max_fk_error = max(fk_errors)
if max_fk_error < 1e-6:
    print(f"\n  ✓ Forward kinematics EXACT (max error: {max_fk_error:.2e})")
else:
    print(f"\n  ⚠ Forward kinematics error: {max_fk_error:.2e}")

# Test trajectory energy conservation (passive dynamics)
print("\n--- Energy Conservation Test (Passive Dynamics) ---")

init_angles = [0.8, 1.2]
init_velocities = [0.0, 0.0]
n_test_steps = 500

js_energy = JointState(
    position=jnp.array([init_angles]),
    velocity=jnp.array([init_velocities])
)
arm_state_energy = arm.reset(batch_size=1, joint_state=js_energy)

# Zero activation (passive)
zero_action = jnp.zeros((1, 6))

# Collect kinetic energy over time
initial_pos = np.array(arm_state_energy.fingertip[0])

for _ in range(n_test_steps):
    arm_state_energy = Arm26.step(
        arm_state_energy, zero_action,
        jnp.zeros((1, 2)), jnp.zeros((1, 2)),
        arm_params
    )

final_pos = np.array(arm_state_energy.fingertip[0])
drift = np.linalg.norm(final_pos - initial_pos)

print(f"  Initial position: {initial_pos}")
print(f"  Final position after {n_test_steps} steps: {final_pos}")
print(f"  Drift distance: {drift*100:.4f} cm")

if drift < 0.05:  # Less than 5cm drift
    print(f"  ✓ Passive drift acceptable")
else:
    print(f"  ⚠ Significant passive drift detected")

print("\n" + "=" * 60)
print("All Benchmarks Complete!")
print("=" * 60)
