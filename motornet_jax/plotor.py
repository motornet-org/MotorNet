"""Plotting utilities for MotorNet-JAX.

This module contains various functions for plotting data from MotorNet-JAX
training and simulation sessions.

Matches the PyTorch MotorNet plotor module API.
"""

import numpy as np
import jax.numpy as jnp

# Optional imports for plotting (allows import without matplotlib)
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    LineCollection = None
    animation = None

try:
    from IPython.display import HTML, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    HTML = None
    display = None


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def to_numpy(x):
    """Convert JAX array to numpy."""
    if hasattr(x, '__array__'):
        return np.asarray(x)
    return x


def compute_limits(data, margin=0.1):
    """Compute plot limits with margin.

    Args:
        data: Data to compute limits for.
        margin: Fraction of data range to add as margin.

    Returns:
        Tuple of (min_val, max_val).
    """
    data = to_numpy(data)
    data_range = data.ptp()
    m = data_range * margin
    minval = np.min(data) - m
    maxval = np.max(data) + m
    return minval, maxval


def _plot_line_collection(axis, segments, cmap='viridis', linewidth=1, **kwargs):
    """Plot a line collection with color gradient."""
    _check_matplotlib()
    n_gradient = kwargs.get('n_gradient', segments.shape[0])

    norm = plt.Normalize(0, n_gradient)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(0, n_gradient))
    lc.set_linewidth(linewidth)

    axis.add_collection(lc)

    fig = axis.get_figure()
    clb = fig.colorbar(lc, ax=axis)
    clb.set_label('timestep')


def _results_to_line_collection(results):
    """Convert trajectory results to line collection format."""
    results = to_numpy(results)
    space_dim = results.shape[-1]
    points = results[:, :, :, np.newaxis].swapaxes(0, -1).swapaxes(0, 1)
    # (n_samples-1) * 2 * space_dim * batch_size
    segments_by_batch = np.concatenate([points[:-1], points[1:]], axis=1)
    # concatenate batch and time dimensions
    segments_all_batches = np.moveaxis(segments_by_batch, -1, 0).reshape((-1, 2, space_dim))
    return segments_all_batches, points


def plot_pos_over_time(cart_results, axis=None, cmap='viridis'):
    """Plot trajectory position over time with color gradient.

    Plots trajectories with darker colors for earlier timesteps and
    brighter colors for later timesteps.

    Args:
        cart_results: Trajectory data. Shape: (batch, n_timesteps, n_dim)
        axis: Matplotlib axis handle. If None, uses current axis.
        cmap: Colormap name.

    Returns:
        Axis handle.
    """
    _check_matplotlib()
    cart_results = to_numpy(cart_results)

    if axis is None:
        axis = plt.gca()

    n_timesteps = cart_results.shape[1]
    segments, points = _results_to_line_collection(cart_results)
    _plot_line_collection(axis, segments, n_gradient=n_timesteps - 1, cmap=cmap)

    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_aspect('equal', adjustable='box')
    axis.margins(0.05)

    return axis


def plot_2dof_arm_over_time(axis, arm_params, joint_state, cmap='viridis', linewidth=1):
    """Plot a 2-DOF arm configuration over time.

    Shows arm configurations at different timesteps with a color gradient.

    Args:
        axis: Matplotlib axis handle.
        arm_params: Arm parameters with l1 and l2 attributes.
        joint_state: Joint state trajectory. Shape: (1, n_timesteps, 4) or (n_timesteps, 2).
        cmap: Colormap name.
        linewidth: Line width for arm segments.
    """
    _check_matplotlib()
    joint_state = to_numpy(joint_state)

    # Handle different input shapes
    if len(joint_state.shape) == 3:
        if joint_state.shape[0] != 1:
            raise ValueError("Can only plot one trajectory at a time")
        joint_pos = joint_state[0, :, :2]  # (n_timesteps, 2)
    elif len(joint_state.shape) == 2:
        joint_pos = joint_state[:, :2] if joint_state.shape[1] >= 2 else joint_state
    else:
        raise ValueError(f"Invalid joint_state shape: {joint_state.shape}")

    n_timesteps = joint_pos.shape[0]

    # Get arm lengths
    if hasattr(arm_params, 'skeleton'):
        L1 = float(arm_params.skeleton.l1)
        L2 = float(arm_params.skeleton.l2)
    else:
        L1 = float(arm_params.l1)
        L2 = float(arm_params.l2)

    # Compute arm segment positions
    joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]
    elb_pos_x = L1 * np.cos(joint_pos[:, 0])
    elb_pos_y = L1 * np.sin(joint_pos[:, 0])
    end_pos_x = elb_pos_x + L2 * np.cos(joint_angle_sum)
    end_pos_y = elb_pos_y + L2 * np.sin(joint_angle_sum)

    # Create line segments for upper and lower arm
    upper_arm_x = np.stack([np.zeros_like(elb_pos_x), elb_pos_x], axis=1)
    upper_arm_y = np.stack([np.zeros_like(elb_pos_y), elb_pos_y], axis=1)
    upper_arm = np.stack([upper_arm_x, upper_arm_y], axis=2)

    lower_arm_x = np.stack([elb_pos_x, end_pos_x], axis=1)
    lower_arm_y = np.stack([elb_pos_y, end_pos_y], axis=1)
    lower_arm = np.stack([lower_arm_x, lower_arm_y], axis=2)

    segments = np.concatenate([upper_arm, lower_arm], axis=0)
    _plot_line_collection(axis, segments, cmap=cmap, linewidth=linewidth, n_gradient=n_timesteps)

    axis.set_xlim(compute_limits(segments[:, :, 0]))
    axis.set_ylim(compute_limits(segments[:, :, 1]))
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_aspect('equal', adjustable='box')


def plot_trajectories(trajectories, targets=None, start_pos=None, ax=None,
                      colors=None, alpha=0.8, linewidth=2, show_targets=True):
    """Plot multiple trajectories with optional targets.

    Args:
        trajectories: List of trajectories, each shape (n_timesteps, 2).
        targets: Optional target positions, shape (n_trajectories, 2).
        start_pos: Optional start position, shape (2,).
        ax: Matplotlib axis. If None, creates new figure.
        colors: List of colors for each trajectory. If None, uses colormap.
        alpha: Line transparency.
        linewidth: Line width.
        show_targets: Whether to show target markers.

    Returns:
        Axis handle.
    """
    _check_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    n_traj = len(trajectories)
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, n_traj))

    for i, traj in enumerate(trajectories):
        traj = to_numpy(traj)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i],
                alpha=alpha, linewidth=linewidth)

    if targets is not None and show_targets:
        targets = to_numpy(targets)
        for i, target in enumerate(targets):
            ax.scatter([target[0]], [target[1]], color=colors[i],
                      s=100, marker='x', zorder=5)

    if start_pos is not None:
        start_pos = to_numpy(start_pos)
        ax.scatter([start_pos[0]], [start_pos[1]], color='black',
                  s=200, marker='*', zorder=10, label='Start')
        ax.legend()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_muscle_activations(activations, dt, muscle_names=None, ax=None):
    """Plot muscle activations over time.

    Args:
        activations: Activation data, shape (n_timesteps, n_muscles).
        dt: Timestep size.
        muscle_names: Optional list of muscle names.
        ax: Matplotlib axis. If None, creates new figure.

    Returns:
        Axis handle.
    """
    _check_matplotlib()
    activations = to_numpy(activations)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    n_timesteps, n_muscles = activations.shape
    t = np.arange(n_timesteps) * dt

    if muscle_names is None:
        muscle_names = [f'Muscle {i}' for i in range(n_muscles)]

    for m in range(n_muscles):
        ax.plot(t, activations[:, m], label=muscle_names[m], linewidth=2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    return ax


def plot_velocity_profiles(trajectories, dt, ax=None, colors=None):
    """Plot hand speed profiles over time.

    Args:
        trajectories: List of trajectories, each shape (n_timesteps, 2).
        dt: Timestep size.
        ax: Matplotlib axis. If None, creates new figure.
        colors: List of colors for each trajectory.

    Returns:
        Axis handle.
    """
    _check_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    n_traj = len(trajectories)
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, n_traj))

    for i, traj in enumerate(trajectories):
        traj = to_numpy(traj)
        # Compute velocity via finite difference
        velocity = np.diff(traj, axis=0) / dt
        speed = np.sqrt(np.sum(velocity ** 2, axis=1))
        t = np.arange(len(speed)) * dt

        ax.plot(t, speed, color=colors[i], linewidth=2, label=f'Target {i}')
        ax.fill_between(t, 0, speed, color=colors[i], alpha=0.3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.grid(True, alpha=0.3)

    return ax


def plot_training_log(losses, ax=None, xlabel='Epoch', ylabel='Loss'):
    """Plot training loss over time.

    Args:
        losses: List or array of loss values.
        ax: Matplotlib axis. If None, creates new figure.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Axis handle.
    """
    _check_matplotlib()
    losses = to_numpy(losses)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.semilogy(losses)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return ax


def animate_trajectory(joint_positions, arm_params, dt=0.01,
                       save_animation=False, path_name='./arm_animation.mp4'):
    """Create animation of arm movement.

    Args:
        joint_positions: Joint angle trajectory, shape (n_timesteps, 2).
        arm_params: Arm parameters with l1 and l2 attributes.
        dt: Timestep size.
        save_animation: Whether to save the animation.
        path_name: Path to save animation.

    Returns:
        Animation object.
    """
    _check_matplotlib()
    joint_positions = to_numpy(joint_positions)

    # Get arm lengths
    if hasattr(arm_params, 'skeleton'):
        L1 = float(arm_params.skeleton.l1)
        L2 = float(arm_params.skeleton.l2)
    else:
        L1 = float(arm_params.l1)
        L2 = float(arm_params.l2)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(-(L1 + L2) * 1.1, (L1 + L2) * 1.1),
                  ylim=(-0.1, (L1 + L2) * 1.1))

    line, = ax.plot([], [], lw=2, alpha=0.4, color='red')  # Movement path
    L1_line, = ax.plot([], [], lw=4, color='C0')  # Upper arm
    L2_line, = ax.plot([], [], lw=4, color='C1')  # Lower arm

    ax.axis('off')
    ax.set_aspect('equal')

    def joint2cartesian(joint_pos):
        joint_angle_sum = joint_pos[0] + joint_pos[1]
        elb_x = L1 * np.cos(joint_pos[0])
        elb_y = L1 * np.sin(joint_pos[0])
        end_x = elb_x + L2 * np.cos(joint_angle_sum)
        end_y = elb_y + L2 * np.sin(joint_angle_sum)
        return elb_x, elb_y, end_x, end_y

    xdata, ydata = [], []

    def init():
        line.set_data([], [])
        L1_line.set_data([], [])
        L2_line.set_data([], [])
        ax.scatter([0], [0], color='black', s=100, zorder=10)
        return line, L1_line, L2_line

    def animate(i):
        elb_x, elb_y, end_x, end_y = joint2cartesian(joint_positions[i])

        xdata.append(end_x)
        ydata.append(end_y)
        line.set_data(xdata, ydata)

        L1_line.set_data([0, elb_x], [0, elb_y])
        L2_line.set_data([elb_x, end_x], [elb_y, end_y])

        return line, L1_line, L2_line

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(joint_positions),
        interval=int(dt * 1000),
        blit=False
    )

    if save_animation:
        writer = animation.FFMpegWriter(fps=int(1 / dt))
        anim.save(path_name, writer=writer, dpi=400)
    else:
        try:
            display(HTML(anim.to_jshtml()))
        except Exception:
            plt.show()

    return anim
