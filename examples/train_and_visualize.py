"""
Train a reaching network and generate a visualization video.

This script:
1. Trains a GRU policy on random reaching for 5000 epochs
2. Uses effort cost (1e-1) and jerk penalty (1e-4)
3. Generates a video showing the network performing in 2D space
   with both joints visible

Run with: python examples/train_and_visualize.py
"""

import os
import sys
import time
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from motornet_jax.effector import Arm26
from motornet_jax.skeleton import TwoDofArm
from motornet_jax.policy import GRUPolicy
from motornet_jax.types import JointState

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Video generation disabled.")
    print("Install with: pip install matplotlib")

print("=" * 60)
print("MotorNet-JAX: Train and Visualize")
print("=" * 60)

# ============================================================
# Configuration
# ============================================================

# Training parameters
N_EPOCHS = 4000
BATCH_SIZE = 32
N_STEPS = 100
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128

# Loss weights
EFFORT_WEIGHT = 2e-1  # Higher weight for effort
JERK_WEIGHT = 1e1    # Jerk penalty

# Simulation parameters
DT = 0.01

# Video parameters
VIDEO_FPS = 30
VIDEO_DURATION = 1.0  # seconds per trial
N_EVAL_TARGETS = 8

print(f"\nConfiguration:")
print(f"  Epochs: {N_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Steps per episode: {N_STEPS}")
print(f"  Effort weight: {EFFORT_WEIGHT}")
print(f"  Jerk weight: {JERK_WEIGHT}")

# ============================================================
# Setup
# ============================================================

print("\n[1] Setting up model...")

# Create arm and policy
arm = Arm26(dt=DT, n_ministeps=1)
arm_params = arm.get_params()
# IMPORTANT: Use skeleton params from Arm26 (not a separate TwoDofArm instance)
# Arm26 uses different default L2 than standalone TwoDofArm
skeleton_params = arm_params.skeleton

# Policy: obs = fingertip(2) + velocity(2) + target(2) = 6
obs_dim = 6
action_dim = 6

key = jax.random.PRNGKey(42)
policy = GRUPolicy(obs_dim, action_dim, hidden_size=HIDDEN_SIZE, key=key)

n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(policy, eqx.is_array)))
print(f"  Policy parameters: {n_params:,}")

# Optimizer with gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(LEARNING_RATE)
)
opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

# ============================================================
# Loss Function
# ============================================================

def compute_loss(policy, arm_states, targets, n_steps=N_STEPS):
    """Compute loss with L1 position + effort + jerk penalties."""
    batch_size = targets.shape[0]
    hidden = policy.init_hidden(batch_size)

    def step_fn(carry, _):
        state, hidden = carry
        obs = jnp.concatenate([
            state.fingertip,
            state.cartesian.velocity,
            targets,
        ], axis=-1)
        action, new_hidden = policy(obs, hidden)
        new_state = Arm26.step(
            state, action,
            jnp.zeros((batch_size, 2)), jnp.zeros((batch_size, 2)),
            arm_params
        )
        return (new_state, new_hidden), (new_state.fingertip, action)

    (final_state, _), (trajectory, actions) = jax.lax.scan(
        step_fn, (arm_states, hidden), None, length=n_steps
    )

    # L1 Position Loss
    target_trajectory = jnp.broadcast_to(targets[None, :, :], trajectory.shape)
    l1_per_step = jnp.sum(jnp.abs(trajectory - target_trajectory), axis=-1)
    position_loss = jnp.mean(l1_per_step)

    # Effort Cost (squared activations)
    effort = jnp.mean(actions ** 2)

    # Jerk Penalty (3rd derivative of position)
    velocity = jnp.diff(trajectory, axis=0)
    acceleration = jnp.diff(velocity, axis=0)
    jerk = jnp.diff(acceleration, axis=0)
    jerk_penalty = jnp.mean(jerk ** 2)

    # Total Loss
    loss = position_loss + EFFORT_WEIGHT * effort + JERK_WEIGHT * jerk_penalty

    # Metrics
    final_pos = trajectory[-1]
    final_error = jnp.sqrt(jnp.sum((final_pos - targets) ** 2, axis=-1))

    return loss, {
        'position_error': jnp.mean(final_error),
        'position_loss': position_loss,
        'effort': effort,
        'jerk': jerk_penalty,
    }


# ============================================================
# Training
# ============================================================

def get_center_position():
    """Get the center hand position for the starting joint configuration."""
    center_joint = jnp.array([[0.8, 1.2]])
    center_state = JointState(position=center_joint, velocity=jnp.zeros((1, 2)))
    _, center_hand, _, _ = TwoDofArm.forward_kinematics(center_state, skeleton_params)
    return center_hand[0], center_joint


def generate_targets(key, batch_size, n_targets=8, radius=0.1):
    """Generate random reaching targets for training."""
    key, subkey = jax.random.split(key)
    target_indices = jax.random.randint(subkey, (batch_size,), 0, n_targets)
    angles = target_indices * (2 * jnp.pi / n_targets)

    center, center_joint = get_center_position()
    targets = center + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    return targets, center_joint


def generate_eval_targets(n_targets=8, radius=0.1):
    """Generate all targets arranged in a circle for evaluation/visualization.

    Unlike generate_targets which randomly samples, this creates targets
    at indices 0, 1, 2, ..., n_targets-1 to form a proper circle.
    """
    center, center_joint = get_center_position()

    # Sequential indices for a proper circular arrangement
    target_indices = jnp.arange(n_targets)
    angles = target_indices * (2 * jnp.pi / n_targets)

    targets = center + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    return targets, center_joint


def create_batch_state(center_joint, batch_size):
    """Create batched arm state at center."""
    joint_state = JointState(
        position=jnp.tile(center_joint, (batch_size, 1)),
        velocity=jnp.zeros((batch_size, 2)),
    )
    return arm.reset(batch_size=batch_size, joint_state=joint_state)


@eqx.filter_jit
def train_step(policy, opt_state, arm_states, targets):
    """Single training step."""
    def loss_fn(policy):
        return compute_loss(policy, arm_states, targets)

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(policy)
    updates, opt_state_new = optimizer.update(grads, opt_state, eqx.filter(policy, eqx.is_array))
    policy_new = eqx.apply_updates(policy, updates)
    return policy_new, opt_state_new, loss, metrics


print("\n[2] Training...")
print(f"  Loss = L1_position + {EFFORT_WEIGHT}*effort + {JERK_WEIGHT}*jerk")
print()

key = jax.random.PRNGKey(0)
history = {'loss': [], 'position_error': [], 'effort': [], 'jerk': []}

start_time = time.time()
for epoch in range(N_EPOCHS):
    key, target_key = jax.random.split(key)
    targets, center_joint = generate_targets(target_key, BATCH_SIZE)
    arm_states = create_batch_state(center_joint, BATCH_SIZE)

    policy, opt_state, loss, metrics = train_step(policy, opt_state, arm_states, targets)

    history['loss'].append(float(loss))
    history['position_error'].append(float(metrics['position_error']))
    history['effort'].append(float(metrics['effort']))
    history['jerk'].append(float(metrics['jerk']))

    if (epoch + 1) % 500 == 0:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:5d}: loss={loss:.4f}, "
              f"pos_err={metrics['position_error']*100:.2f}cm, "
              f"effort={metrics['effort']:.4f}, "
              f"jerk={metrics['jerk']:.2e} "
              f"({elapsed:.1f}s)")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f}s")
print(f"  Final position error: {history['position_error'][-1]*100:.2f}cm")
print(f"  Final effort: {history['effort'][-1]:.4f}")

# ============================================================
# Evaluation
# ============================================================

print("\n[3] Evaluating trained policy...")

@jax.jit
def evaluate_trajectory(policy, arm_state, target, n_steps=int(VIDEO_DURATION / DT)):
    """Generate trajectory from trained policy."""
    hidden = policy.init_hidden(1)

    def step_fn(carry, _):
        state, hidden = carry
        obs = jnp.concatenate([
            state.fingertip,
            state.cartesian.velocity,
            target[None, :],
        ], axis=-1)
        action, hidden = policy(obs, hidden)
        new_state = Arm26.step(state, action, jnp.zeros((1, 2)), jnp.zeros((1, 2)), arm_params)
        # Return current state BEFORE step (so trajectory starts from initial position)
        return (new_state, hidden), (state.joint.position[0], state.fingertip[0], action[0])

    _, (joint_positions, fingertip_positions, actions) = jax.lax.scan(
        step_fn, (arm_state, hidden), None, length=n_steps
    )

    return joint_positions, fingertip_positions, actions


# Generate evaluation trajectories - use sequential targets for proper circle
eval_targets, center_joint = generate_eval_targets(N_EVAL_TARGETS)

arm_state = arm.reset(
    batch_size=1,
    joint_state=JointState(position=center_joint, velocity=jnp.zeros((1, 2)))
)

all_joint_traj = []
all_fingertip_traj = []
all_actions = []

for i in range(N_EVAL_TARGETS):
    joints, fingertip, actions = evaluate_trajectory(policy, arm_state, eval_targets[i])
    all_joint_traj.append(np.array(joints))
    all_fingertip_traj.append(np.array(fingertip))
    all_actions.append(np.array(actions))

# Compute errors
errors = []
for traj, target in zip(all_fingertip_traj, eval_targets):
    final_pos = traj[-1]
    error = np.sqrt(np.sum((final_pos - np.array(target))**2))
    errors.append(error)

print(f"  Mean final error: {np.mean(errors)*100:.2f}cm")
print(f"  Std final error: {np.std(errors)*100:.2f}cm")

# ============================================================
# Video Generation
# ============================================================

if HAS_MATPLOTLIB:
    print("\n[4] Generating video...")

    # Get arm lengths
    L1 = float(skeleton_params.L1)
    L2 = float(skeleton_params.L2)

    # Muscle names for legend
    muscle_names = arm.muscle_names

    def joint_to_cartesian(joint_angles):
        """Convert joint angles to arm segment positions."""
        shoulder, elbow = joint_angles
        joint_sum = shoulder + elbow
        elb_x = L1 * np.cos(shoulder)
        elb_y = L1 * np.sin(shoulder)
        end_x = elb_x + L2 * np.cos(joint_sum)
        end_y = elb_y + np.sin(joint_sum) * L2
        return (0, 0), (elb_x, elb_y), (end_x, end_y)

    # Create figure with three subplots
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

    # Left: Arm visualization
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(-0.2, 0.8)
    ax1.set_ylim(-0.2, 0.8)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Arm Movement')
    ax1.grid(True, alpha=0.3)

    # Middle: Joint angles over time
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, VIDEO_DURATION)
    ax2.set_ylim(0, 3.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Joint Angle (rad)')
    ax2.set_title('Joint Angles')
    ax2.grid(True, alpha=0.3)

    # Right: Muscle activations over time
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, VIDEO_DURATION)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Activation')
    ax3.set_title('Muscle Activity')
    ax3.grid(True, alpha=0.3)

    # Colors for targets and muscles
    target_colors = plt.cm.hsv(np.linspace(0, 1, N_EVAL_TARGETS))
    muscle_colors = plt.cm.tab10(np.linspace(0, 1, 6))

    # Plot ALL targets on arm view (initially dimmed)
    target_markers = []
    for i, target in enumerate(eval_targets):
        marker = ax1.scatter([float(target[0])], [float(target[1])],
                            color=target_colors[i], s=80, marker='x',
                            alpha=0.3, zorder=5)
        target_markers.append(marker)

    # Current target highlight (larger, brighter)
    current_target_marker = ax1.scatter([], [], color='red', s=200,
                                        marker='o', facecolors='none',
                                        linewidths=3, zorder=6)

    # Initialize arm artists
    upper_arm_line, = ax1.plot([], [], 'b-', linewidth=8, solid_capstyle='round')
    lower_arm_line, = ax1.plot([], [], 'r-', linewidth=6, solid_capstyle='round')
    shoulder_joint = Circle((0, 0), 0.02, color='black', zorder=10)
    elbow_joint = Circle((0, 0), 0.015, color='black', zorder=10)
    hand_marker = Circle((0, 0), 0.01, color='green', zorder=10)
    ax1.add_patch(shoulder_joint)
    ax1.add_patch(elbow_joint)
    ax1.add_patch(hand_marker)

    # Trajectory trace
    trace_line, = ax1.plot([], [], 'g-', linewidth=2, alpha=0.7)

    # Joint angle lines
    t_data = np.arange(len(all_joint_traj[0])) * DT
    shoulder_line, = ax2.plot([], [], 'b-', linewidth=2, label='Shoulder')
    elbow_line, = ax2.plot([], [], 'r-', linewidth=2, label='Elbow')
    time_marker_joints = ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    # Muscle activity lines
    muscle_lines = []
    for i in range(6):
        line, = ax3.plot([], [], color=muscle_colors[i], linewidth=2,
                        label=muscle_names[i])
        muscle_lines.append(line)
    time_marker_muscles = ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=8, ncol=2)

    # Current target indicator text
    target_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                          fontsize=12, verticalalignment='top',
                          fontweight='bold')

    # Frame data storage
    trace_x, trace_y = [], []
    prev_traj_idx = [-1]

    def init():
        upper_arm_line.set_data([], [])
        lower_arm_line.set_data([], [])
        trace_line.set_data([], [])
        shoulder_line.set_data([], [])
        elbow_line.set_data([], [])
        for ml in muscle_lines:
            ml.set_data([], [])
        return [upper_arm_line, lower_arm_line, trace_line,
                shoulder_line, elbow_line] + muscle_lines

    n_frames_per_target = int(VIDEO_DURATION / DT)
    total_frames = N_EVAL_TARGETS * n_frames_per_target

    def animate(frame):
        traj_idx = frame // n_frames_per_target
        frame_in_traj = frame % n_frames_per_target

        # Reset trace and update target highlighting on new trajectory
        if frame_in_traj == 0 or traj_idx != prev_traj_idx[0]:
            trace_x.clear()
            trace_y.clear()
            prev_traj_idx[0] = traj_idx

            # Update target marker visibility - highlight current target
            for i, marker in enumerate(target_markers):
                if i == traj_idx:
                    marker.set_alpha(1.0)
                    marker.set_sizes([150])
                else:
                    marker.set_alpha(0.3)
                    marker.set_sizes([80])

        joint_traj = all_joint_traj[traj_idx]
        fingertip_traj = all_fingertip_traj[traj_idx]
        actions_traj = all_actions[traj_idx]
        target = eval_targets[traj_idx]

        # Update current target circle
        current_target_marker.set_offsets([[float(target[0]), float(target[1])]])

        # Get joint angles for this frame
        joint_angles = joint_traj[frame_in_traj]

        # Compute arm positions from joints (for arm visualization)
        shoulder_pos, elbow_pos, _ = joint_to_cartesian(joint_angles)

        # Use actual fingertip position from simulation (matches target coordinate system)
        hand_pos = (fingertip_traj[frame_in_traj, 0], fingertip_traj[frame_in_traj, 1])

        # Update arm segments
        upper_arm_line.set_data([shoulder_pos[0], elbow_pos[0]],
                               [shoulder_pos[1], elbow_pos[1]])
        lower_arm_line.set_data([elbow_pos[0], hand_pos[0]],
                               [elbow_pos[1], hand_pos[1]])

        # Update joints
        elbow_joint.center = elbow_pos
        hand_marker.center = hand_pos

        # Update trace
        trace_x.append(hand_pos[0])
        trace_y.append(hand_pos[1])
        trace_line.set_data(trace_x, trace_y)

        # Update joint angle plot
        t_current = t_data[:frame_in_traj+1]
        shoulder_angles = joint_traj[:frame_in_traj+1, 0]
        elbow_angles = joint_traj[:frame_in_traj+1, 1]
        shoulder_line.set_data(t_current, shoulder_angles)
        elbow_line.set_data(t_current, elbow_angles)
        time_marker_joints.set_xdata([t_current[-1]])

        # Update muscle activity plot
        for i, ml in enumerate(muscle_lines):
            ml.set_data(t_current, actions_traj[:frame_in_traj+1, i])
        time_marker_muscles.set_xdata([t_current[-1]])

        # Update target text with color
        target_text.set_text(f'Target {traj_idx + 1}/{N_EVAL_TARGETS}')
        target_text.set_color(target_colors[traj_idx])

        return ([upper_arm_line, lower_arm_line, trace_line,
                shoulder_line, elbow_line, time_marker_joints,
                time_marker_muscles, current_target_marker] + muscle_lines)

    # Adjust layout
    plt.tight_layout()

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames,
        interval=int(1000 / VIDEO_FPS),
        blit=False
    )

    # Save video
    print(f"  ({N_EVAL_TARGETS} targets × {VIDEO_DURATION}s = {N_EVAL_TARGETS * VIDEO_DURATION:.0f}s total)")

    # Try FFmpeg (mp4) first, then fall back to pillow (gif)
    video_saved = False
    try:
        video_path = 'reaching_demo.mp4'
        print(f"  Saving to {video_path} (FFmpeg)...")
        writer = animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=2000)
        anim.save(video_path, writer=writer, dpi=100)
        print(f"  ✓ Video saved: {video_path}")
        video_saved = True
    except Exception as e:
        print(f"  FFmpeg not available, trying GIF with pillow...")
        try:
            video_path = 'reaching_demo.gif'
            print(f"  Saving to {video_path} (pillow)...")
            anim.save(video_path, writer='pillow', fps=VIDEO_FPS)
            print(f"  ✓ Animation saved: {video_path}")
            video_saved = True
        except Exception as e2:
            print(f"  Could not save animation: {e2}")
            print("  Install ffmpeg for video export")

    if not video_saved:
        video_path = 'reaching_demo_NOT_SAVED'

    plt.close()

    # Also save training curves
    print("\n[5] Saving training curves...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

    axes2[0, 0].semilogy(history['loss'])
    axes2[0, 0].set_xlabel('Epoch')
    axes2[0, 0].set_ylabel('Total Loss')
    axes2[0, 0].set_title('Training Loss')
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].plot([e * 100 for e in history['position_error']])
    axes2[0, 1].set_xlabel('Epoch')
    axes2[0, 1].set_ylabel('Position Error (cm)')
    axes2[0, 1].set_title('Final Position Error')
    axes2[0, 1].axhline(y=3, color='g', linestyle='--', alpha=0.5, label='3cm target')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].plot(history['effort'])
    axes2[1, 0].set_xlabel('Epoch')
    axes2[1, 0].set_ylabel('Effort')
    axes2[1, 0].set_title(f'Effort Cost (weight={EFFORT_WEIGHT})')
    axes2[1, 0].grid(True, alpha=0.3)

    axes2[1, 1].semilogy(history['jerk'])
    axes2[1, 1].set_xlabel('Epoch')
    axes2[1, 1].set_ylabel('Jerk')
    axes2[1, 1].set_title(f'Jerk Penalty (weight={JERK_WEIGHT})')
    axes2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = 'training_curves.png'
    plt.savefig(curves_path, dpi=150)
    print(f"  ✓ Training curves saved: {curves_path}")
    plt.close()

else:
    print("\n[4] Skipping video generation (matplotlib not available)")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"""
Training:
  Epochs: {N_EPOCHS}
  Final position error: {history['position_error'][-1]*100:.2f}cm
  Final effort: {history['effort'][-1]:.4f}
  Training time: {total_time:.1f}s

Evaluation:
  Mean error: {np.mean(errors)*100:.2f}cm
  Std error: {np.std(errors)*100:.2f}cm

Output files:
  - reaching_demo.mp4 (animation)
  - training_curves.png (loss curves)
""")
print("=" * 60)
