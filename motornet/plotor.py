"""This module contains various functions for plotting data from `MotorNet` training and simulation sessions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
import torch as th
from IPython.display import HTML, display


def compute_limits(data, margin=0.1):
  """Computes the limits to use for plotting data, given the range of the dataset and a margin size around that range.

  Args:
    data: A `numpy.ndarray` containing the data to plot.
    margin: `Float`, the proportion of the data's range to add as margin for plotting. For instance, if the data
      value range from `0` to `10`, and the margin is set to `0.2`, then the limits would become `[-2, 12]` since
      the range is `10`.

  Returns:
    A `list` of two `float` values, representing the lower and upper limits to use on the plot, in that order.
  """
  data_range = data.ptp()
  m = data_range * margin
  minval = np.min(data) - m
  maxval = np.max(data) + m
  return minval, maxval


def _plot_line_collection(axis, segments, cmap='viridis', linewidth: int = 1, **kwargs):
  n_gradient = kwargs.get('n_gradient', segments.shape[0])

  norm = plt.Normalize(0, n_gradient)  # Create a continuous norm to map from data points to colors
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  lc.set_array(np.arange(0, n_gradient))  # Set the values used for colormapping
  lc.set_linewidth(linewidth)

  axis.add_collection(lc)

  fig = axis.get_figure()
  clb = fig.colorbar(lc, ax=axis)
  clb.set_label('timestep')


def plot_pos_over_time(cart_results, axis, cmap='viridis'):
  """Plots trajectory position over time, giving a darker color to the early part of a trajectory, and a lighter color
  to the later part of the trajectory.

  Args:
    axis: A `matplotlib` axis handle to plot the trajectories.
    cart_results: A `numpy.ndarray` containing the trajectories. Its dimensionality should be
      `(n_batches * n_timesteps * n_dim)`, with `n_dim` being the trajectory's dimensionality. For instance,
      for a planar reach, since the movement is in 2D space, we have`n_dim = 2`, meaning the third dimension of
      the array is `2` (`x` position, `y` position).
    cmap: `String` or `matplotlib` colormap object, the colormap to use to visualize positions over time.
  """
  n_timesteps = cart_results.shape[1]
  segments, points = _results_to_line_collection(cart_results)
  _plot_line_collection(axis, segments, n_gradient=n_timesteps - 1, cmap=cmap)
  axis.set_xlabel('cartesian x')
  axis.set_ylabel('cartesian y')
  axis.set_aspect('equal', adjustable='box')
  axis.margins(0.05)


def _results_to_line_collection(results):
  # each line is a segment of the trajectory (a sample), and it will have its own colour from the gradent
  # each line has two values (per dimension): start point and end point
  # n_samples * 1 * space_dim * batch_size
  results_pos, _ = np.split(results, 2, axis=-1)
  results_pos = results
  space_dim = results_pos.shape[-1]
  points = results_pos[:, :, :, np.newaxis].swapaxes(0, -1).swapaxes(0, 1)
  # (n_samples-1) * 2 * space_dim * batch_size
  segments_by_batch = np.concatenate([points[:-1], points[1:]], axis=1)
  # concatenate batch and time dimensions (b1t1, b2t1, ..., b1t2, b2t2, ....,b1tn, b2tn, ..., bntn)
  # n_lines * 2 * space_dim
  segments_all_batches = np.moveaxis(segments_by_batch, -1, 0).reshape((-1, 2, space_dim))
  return segments_all_batches, points


def plot_2dof_arm_over_time(axis, arm, joint_state, cmap: str = 'viridis', linewidth: int = 1):
  """Plots an arm26 over time, with earlier and later arm configuration in the movement being represented as darker
  and brighter colors, respectively.

  Args:
    axis: A `matplotlib` axis handle.
    arm: :class:`motornet.plants.skeletons.TwoDofArm` object to plot.
    joint_state: A `numpy.ndarray` containing the trajectory. Its dimensionality should be
      `1 * n_timesteps * (2 . n_dim)`, with `n_dim` being the trajectory's dimensionality. For an `arm26`,
      since the arm has 2 degrees of freedom, we have`n_dim = 2`, meaning the third dimension of
      the array is `4` (shoulder position, elbow position, shoulder velocity, elbow velocity).
    cmap: `String`, colormap supported by `matplotlib`.
    linewidth: `Integer`, line width of the arm segments being plotted.
  """

  assert joint_state.shape[0] == 1  # can only take one simulation at a time
  n_timesteps = joint_state.shape[1]
  joint_pos = np.moveaxis(joint_state, 0, -1).squeeze()

  joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]
  elb_pos_x = arm.L1 * np.cos(joint_pos[:, 0])
  elb_pos_y = arm.L1 * np.sin(joint_pos[:, 0])
  end_pos_x = elb_pos_x + arm.L2 * np.cos(joint_angle_sum)
  end_pos_y = elb_pos_y + arm.L2 * np.sin(joint_angle_sum)

  upper_arm_x = np.stack([np.zeros_like(elb_pos_x), elb_pos_x], axis=1)
  upper_arm_y = np.stack([np.zeros_like(elb_pos_y), elb_pos_y], axis=1)
  upper_arm = np.stack([upper_arm_x, upper_arm_y], axis=2)

  lower_arm_x = np.stack([elb_pos_x, end_pos_x], axis=1)
  lower_arm_y = np.stack([elb_pos_y, end_pos_y], axis=1)
  lower_arm = np.stack([lower_arm_x, lower_arm_y], axis=2)

  segments = np.squeeze(np.concatenate([upper_arm, lower_arm], axis=0))
  _plot_line_collection(axis, segments, cmap=cmap, linewidth=linewidth, n_gradient=n_timesteps)

  axis.set_xlim(compute_limits(segments[:, :, 0]))
  axis.set_ylim(compute_limits(segments[:, :, 1]))
  axis.set_xlabel('cartesian x')
  axis.set_ylabel('cartesian y')
  axis.set_aspect('equal', adjustable='box')


def animate_trajectory(joint_position, model, save_animation=False, path_name='./Arm_animation.mp4'):
  # Get Plant and Skeleton info
  state_dim = model.skeleton.state_dim
  dt = model.effector.dt
  skeleton_type = model.skeleton.__name__

  joint_position = th.reshape(joint_position, (-1, state_dim)).detach().numpy()

  if skeleton_type == 'point_mass':
    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
    line, = ax.plot([], [], lw=2, alpha=0.2, color='red')  # Movement path
    endpoint = ax.scatter([], [])

    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')  # Make axis equal
    plt.style.use('dark_background')

    # Initialization of data lists
    def init():
      # creating void frame
      line.set_data([], [])
      endpoint.set_offsets(np.array([0, 0]).T)
      ax.scatter([0], [0], marker='+')
      return line, endpoint

    # Empty List for trajectories and arm position
    xdata, ydata = [], []

    # animation
    def animate(i):
      # Append the endpoint position
      x = joint_position[i:i + 1, 0]
      y = joint_position[i:i + 1, 1]
      xdata.append(x)
      ydata.append(y)
      line.set_data(xdata, ydata)
      endpoint.set_offsets(np.array([x, y]).T)

      return line, endpoint

    # call animation
    anim = animation.FuncAnimation(
      fig,
      animate,
      init_func=init,
      frames=joint_position.shape[0],
      interval=dt * 1000,
      blit=False
    )

  elif skeleton_type == 'two_dof_arm':
    L1_len = model.skeleton.L1
    L2_len = model.skeleton.L2
    fig = plt.figure()
    ax = plt.axes(xlim=(-(L1_len + L2_len), L1_len + L2_len), ylim=(-.1, L1_len + L2_len))
    line, = ax.plot([], [], lw=2, alpha=0.4, color='red')  # Movement path
    L1, = ax.plot([], [], lw=4, color='C0')  # L1
    L2, = ax.plot([], [], lw=4, color='C1')  # L2

    # plt.title('Desired Title')
    # Axis control
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().set_aspect('equal', adjustable='box')  # Make axis equal
    plt.style.use('dark_background')

    def joint2cartesian(joint_pos):
      joint_pos = np.reshape(joint_pos, (-1, state_dim))
      joint_angle_sum = joint_pos[:, 0] + joint_pos[:, 1]

      c1 = np.cos(joint_pos[:, 0])
      s1 = np.sin(joint_pos[:, 0])
      c12 = np.cos(joint_angle_sum)
      s12 = np.sin(joint_angle_sum)

      end_pos_x_l1 = L1_len * c1
      end_pos_y_l1 = L1_len * s1
      end_pos_x_l2 = L2_len * c12
      end_pos_y_l2 = L2_len * s12

      return end_pos_x_l1, end_pos_y_l1, end_pos_x_l2, end_pos_y_l2

    # Initialization of data lists
    def init():
      # creating void frame
      line.set_data([], [])
      L1.set_data([], [])
      L2.set_data([], [])
      ax.scatter([0], [0])
      return line, L1, L2

    # Empty List for trajectories and arm position
    xdata, ydata = [], []

    # animation
    def animate(i):
      # Get the position of end_point, L1, and L2
      end_pos_x_l1, end_pos_y_l1, end_pos_x_l2, end_pos_y_l2 = joint2cartesian(joint_position[i:i + 1, :])

      # Append the endpoint position
      xdata.append(end_pos_x_l1 + end_pos_x_l2)
      ydata.append(end_pos_y_l1 + end_pos_y_l2)
      line.set_data(xdata, ydata)

      # Append the L1 position
      L1_xdata = [0, end_pos_x_l1.item()]
      L1_ydata = [0, end_pos_y_l1.item()]
      L1.set_data(L1_xdata, L1_ydata)

      # Append the L2 position
      L2_xdata = [end_pos_x_l1, end_pos_x_l1 + end_pos_x_l2]
      L2_ydata = [end_pos_y_l1, end_pos_y_l1 + end_pos_y_l2]
      L2.set_data(L2_xdata, L2_ydata)

      return line, L1, L2

    # call animation
    anim = animation.FuncAnimation(
      fig,
      animate,
      init_func=init,
      frames=joint_position.shape[0],
      interval=int(dt * 1000),
      blit=False
    )

  else:
    print("Unknown Skeleton!")
    return None

  # save the animated file
  if save_animation:
    writer = animation.FFMpegWriter(fps=1 / dt)
    anim.save(path_name, writer=writer, dpi=400)
  else:
    #display_html(HTML(anim.to_jshtml()))
    display(HTML(anim.to_jshtml()))
  plt.close()