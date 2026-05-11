import torch
import numpy as np
from typing import Any, Union


class Skeleton(torch.nn.Module):
  """Base class for `Skeleton` objects.

  Args:
    dof: `Integer`, number of degrees of freedom of the skeleton. Typically this is the number of joints.
    space_dim: `Integer`, the dimensionality of the space in which the skeleton evolves. For instance, this would be
      `2` for a cartesian, planar `xy` space.
    name: `String`, the name of the object instance.
    pos_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of joint position. This should be
      a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
      degrees-of-freedom arm, we would have `n=2`.
    pos_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of joint position. This should be
      a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
      degrees-of-freedom arm, we would have `n=2`.
    vel_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of joint velocity. This should be
      a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
      degrees-of-freedom arm, we would have `n=2`.
    vel_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of joint velocity. This should be
      a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
      degrees-of-freedom arm, we would have `n=2`.
    input_dim: `Integer`, dimensionality of the control input (*e.g.* torques). Default: same as `dof`.
    state_dim: `Integer`, typically position and velocity for each degree of freedom. Default: `2 * dof`.
    output_dim: `Integer`, dimensionality of the :meth:`integrate` output, usually the new state.
      Default: same as `state_dim`.
  """

  def __init__(self, dof: int, space_dim: int, name: str = "skeleton",
         pos_lower_bound: Union[float, list, tuple] = -1.,
         pos_upper_bound: Union[float, list, tuple] = +1.,
         vel_lower_bound: Union[float, list, tuple] = -1000.,
         vel_upper_bound: Union[float, list, tuple] = +1000.,
         **kwargs):

    super().__init__()

    self._device = torch.device('cpu')
    self.__name__ = name
    self.dof = dof
    self.space_dim = space_dim
    self.input_dim = kwargs.get('input_dim', self.dof)
    self.state_dim = kwargs.get('state_dim', self.dof * 2)
    self.output_dim = kwargs.get('output_dim', self.state_dim)
    self.geometry_state_dim = 2 + self.dof # two geometry variable per muscle: path_length, path_velocity
    self.default_endpoint_load = torch.zeros((1, self.space_dim)) #tf.zeros((1, self.space_dim), dtype=tf.float32)

    self.pos_lower_bound = pos_lower_bound
    self.pos_upper_bound = pos_upper_bound
    self.vel_lower_bound = vel_lower_bound # cap as defensive code
    self.vel_upper_bound = vel_upper_bound

    self.clip_position = None
    self.init = False
    self.built = False
    #self.detach = lambda x: x.cpu().detach().numpy() if torch.is_tensor(x) else x

    self.clip = self._clip

  @staticmethod
  def _clip(x, lb, ub):
    return torch.min(torch.max(x, lb), ub)

  def clip_method(self, x: torch.Tensor) -> torch.Tensor:
    return self.clip(x, self.pos_lower_bound, self.pos_upper_bound)

  def detach(self, x: Any) -> Any:
    return x.cpu().detach().numpy() if torch.is_tensor(x) else x
  
  def build(
      self,
      timestep: float,
      pos_upper_bound: float | list | tuple | np.ndarray | torch.Tensor,
      pos_lower_bound: float | list | tuple | np.ndarray | torch.Tensor,
      vel_upper_bound: float | list | tuple | np.ndarray | torch.Tensor,
      vel_lower_bound: float | list | tuple | np.ndarray | torch.Tensor,
      ) -> None:
    """This method should be called by the initialization method of the 
    :class:`motornet.effector.Effector` object class or subclass.

    Args:
      timestep: `Float`, size of a single timestep (sec).
      pos_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of joint position. Should
        be a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
        degrees-of-freedom arm, we would have `n=2`.
      pos_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of joint position. Should
        be a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
        degrees-of-freedom arm, we would have `n=2`.
      vel_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of joint velocity. Should
        be a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
        degrees-of-freedom arm, we would have `n=2`.
      vel_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of joint velocity. Should
        be a `n`-elements vector or list, with `n` the number of joints of the skeleton. For instance, for a two
        degrees-of-freedom arm, we would have `n=2`.
    """
    
    for _attr in ('pos_upper_bound', 'pos_lower_bound', 'vel_upper_bound', 'vel_lower_bound'):
      self.__dict__.pop(_attr, None)
    self.register_buffer('pos_upper_bound', torch.as_tensor(pos_upper_bound, dtype=torch.float32).reshape(1, -1))
    self.register_buffer('pos_lower_bound', torch.as_tensor(pos_lower_bound, dtype=torch.float32).reshape(1, -1))
    self.register_buffer('vel_upper_bound', torch.as_tensor(vel_upper_bound, dtype=torch.float32).reshape(1, -1))
    self.register_buffer('vel_lower_bound', torch.as_tensor(vel_lower_bound, dtype=torch.float32).reshape(1, -1))
    self.dt = timestep
    self.clip_position = self.clip_method
    self.built = True

  def path2cartesian(self, path_coordinates: torch.Tensor, path_fixation_body: torch.Tensor, joint_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transforms muscle paths into cartesian paths for each muscle's fixation points, given a joint configuration.
    This method is used by the wrapper :class:`motornet.plants.plants.Plant` object class or subclass to then
    calculate musculotendon complex length and velocity, as well as moment arms. See `[1]` for more details.

    References:
      [1] `Sherman MA, Seth A, Delp SL. What Is Moment Arm? Calculating Muscle Effectiveness in Biomechanical
      Models Using Generalized Coordinates. Proc ASME Des Eng Tech Conf. 2013 Aug; DOI: 10.1115/DETC2013-13633.
      PMID: 25905111; PMCID: PMC4404026.`

    Args:
      path_coordinates: The coordinates of each muscle's fixation point. This is an attribute held by the wrapper
        :class:`motornet.effector.Effector` object class or subclass.
      path_fixation_body: The body part (bone) to which each muscle fixation point is attached to. This is an
        attribute held by the wrapper :class:`motornet.plants.plants.Plant` object class or subclass.
      joint_state: `Tensor`, the current joint configuration.

    Returns:
      - The position of each fixation point in cartesian space. The dimensionality is `n_batch * space_dim *
        n_fixation_points`, with `space_dim` the dimensionality of the worldspace and `n_fixation_points` the
        number of fixation points across all muscles.
      - The derivative with respect to time of each fixation point's position in cartesian space (velocity). The
        dimensionality is `n_batch * space_dim * n_fixation_points`, with `space_dim` the dimensionality of the
        worldspace and `n_fixation_points` the number of fixation points across all muscles.
      - The derivative with respect to joint angle of each fixation point's position in cartesian space. The
        dimensionality is `n_batch * space_dim * n_dof * n_fixation_points`, with `space_dim` the dimensionality
        of the worldspace, `n_dof` the number of degrees of freedoms of the :class:`Skeleton` object class or
        subclass, and `n_fixation_points` the number of fixation points across all muscles.
    """
    return self._path2cartesian(path_coordinates, path_fixation_body, joint_state)
  
  def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
    raise NotImplementedError
  
  def integrate(self, dt: float, state_derivative: torch.Tensor, joint_state: torch.Tensor) -> torch.Tensor:
    """Performs one integration step. This method is usually called by the
    :meth:`motornet.effector.Effector.integration_step` during numerical integration by the
    :class:`motornet.effector.Effector` wrapper class or subclass.

    Args:
      dt: `Float`, timestep (sec) of the effector. The skeleton object's :attr:`dt` attribute is not used here
        because for the Runge-Kutta method, half-timesteps can occasionally be passed.
      state_derivative: `Tensor`, derivative of joint state. This input and `joint_state` should have identical
        dimensionality.
      joint_state: `Tensor`, joint state that serves as the initial value. This input and `state_derivative`
        should have identical dimensionality.

    Returns:
      The new state following integration. The dimensionality is identical to that of the `joint_state` input.
    """
    return self._integrate(dt, state_derivative, joint_state)

  def _integrate(self, dt, state_derivative, joint_state):
    raise NotImplementedError

  def joint2cartesian(self, joint_state: torch.Tensor) -> torch.Tensor:
    """Computes the cartesian state given the joint state.

    Args:
      joint_state: `Tensor`, the current joint configuration.

    Returns:
      A `tensor` containing the current cartesian configuration (position, velocity).
    """
    return self._joint2cartesian(joint_state)

  def _joint2cartesian(self, joint_state):
    raise NotImplementedError

  def ode(self, inputs: torch.Tensor, joint_state: torch.Tensor, endpoint_load: torch.Tensor) -> torch.Tensor:
    """Evaluates the Ordinary Differential Equation (ODE) function.

    Args:
      inputs: `Tensor`, the control input to the skeleton object (typically torques).
      joint_state: `Tensor`, the current joint configuration.
      endpoint_load: `Tensor`, the loads applied to the skeleton's endpoint. The dimensionality should match the
        skeleton object or subclass' `space_dim` attribute.

    Returns:
      The derivatives of the joint state. The dimensionality is identical to that of the `joint_state` input.
    """
    return self._ode(inputs, joint_state, endpoint_load)
  
  def _ode(self, inputs, joint_state, endpoint_load):
    raise NotImplementedError

  def clip_velocity(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    """Clips the joint velocities input based on the velocity boundaries as well as the joint positions.
    Specifically, if a velocity is past the boundary values, it is set to the boundary value exactly. Then, if
    the position is past or equal to the position boundary, and the clipped velocity would result in moving further
    past that boundary, then the velocity is set to `0`.

    Args:
      pos: A `tensor` containing positions of the plant in joint space.
      vel: A `tensor` containing velocities of the plant in joint space.

    Returns:
      A `tensor` containing the clipped velocities, with same dimensionality as the `vel` input argument.
    """
    vel = self.clip(vel, self.vel_lower_bound, self.vel_upper_bound)
    vel = torch.where(torch.logical_and(vel < 0, pos <= self.pos_lower_bound), 0., vel)
    vel = torch.where(torch.logical_and(vel > 0, pos >= self.pos_upper_bound), 0., vel)
    return vel

  def get_base_config(self) -> dict:
    """Get the object instance's base configuration. This is the set of configuration entries that will be useful
    for any :class:`Skeleton` class or subclass. This method should be called by the :meth:`get_save_config`
    method. Users wanting to save additional configuration entries specific to a :class:`Skeleton` subclass should
    then do so in the :meth:`get_save_config` method, using this method's output `dictionary` as a base.

    Returns:
      A `dictionary` containing the skeleton's degrees of freedom, timestep size, and space dimensionality.
    """
    cfg = {'dof': self.dof, 'dt': str(self.dt), 'space_dim': self.space_dim}
    return cfg

  def get_save_config(self) -> dict:
    """Get the skeleton object's configuration as a `dictionary`. This method should be overwritten by subclass
    objects, and used to add configuration entries specific to that subclass, on top of the output of the
    :meth:`get_base_config` method.

    Returns:
      By default, this method returns the output of the :meth:`get_base_config` method.
    """
    return self.get_base_config()
  
  def setattr(self, name: str, value: Any) -> None:
    """Changes the value of an attribute held by this object.

    Args:
      name: `String`, attribute to set to a new value.
      value: Value that the attribute should take.
    """
    self.__setattr__(name, value)

  def to(self, *args, **kwargs):
    if args and isinstance(args[0], (str, torch.device)):
      self._device = torch.device(args[0])
    elif args and isinstance(args[0], torch.Tensor):
      self._device = args[0].device
    elif 'device' in kwargs:
      self._device = torch.device(kwargs['device'])
    return super().to(*args, **kwargs)

  @property
  def device(self) -> torch.device:
    return self._device


class PointMass(Skeleton):
  """A simple point-mass skeleton. The point of the point-mass is considered as a bone at the implementation level, so
  this can be conceptualized as a "one-bone" skeleton with a length of `0` and no joint. However, note that the number
  of degrees of freedom is not `0` like the number of joints, but is equal to the `space_dim` input (see below for
  details).

  Args:
    space_dim: `Integer`, the dimensionality of the space in which the point-mass evolves. For instance, this would
      be `2` for a point-mass evolving in a cartesian, planar `xy` space.
    mass: `Float`, the mass (kg) of the point-mass.
    name: `String`, the name of the skeleton.
    **kwargs: This is passed as-is to the parent :class:`Skeleton` class. This also
      allows for some backward compatibility. Do not try to pass a `dof` input value as this is automatically
      taken by the `space_dim` input.
  """

  def __init__(self, space_dim: int = 2, mass: float = 1., name: str = "point_mass", **kwargs):
    super().__init__(dof=space_dim, space_dim=space_dim, name=name, **kwargs)
    self.mass = mass
    # to speed up runtime during `self._path2cartesian` method
    dpos_ddof = torch.nn.functional.one_hot(torch.arange(0, self.dof) % self.dof).to(torch.float32)[None, :, :, None]
    self.register_buffer('dpos_ddof', dpos_ddof)

  def _ode(self, inputs, joint_state, endpoint_load):
    return inputs + endpoint_load

  def _integrate(self, dt, state_derivative, joint_state):
    old_pos, old_vel = joint_state.chunk(2, dim=1)
    new_vel = old_vel + state_derivative * dt / self.mass
    new_pos = old_pos + old_vel * dt
    new_vel = self.clip_velocity(new_pos, new_vel)
    new_pos = self.clip_position(new_pos)
    return torch.cat([new_pos, new_vel], dim=1)
  
  def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
    pos, vel = joint_state[:, :, None].chunk(2, dim=1)
    # if fixed on the point mass, then add the point-mass position / velocity to the fixation point coordinate
    pos = torch.where(path_fixation_body == 0, 0., pos) + path_coordinates
    vel = torch.where(path_fixation_body == 0, 0., vel)
    dpos_ddof = torch.where(path_fixation_body == 0, 0., self.dpos_ddof)
    return pos, vel, dpos_ddof

  def _joint2cartesian(self, joint_state):
    return joint_state

  def get_save_config(self) -> dict:
    """Gets the base configuration from the :meth:`Skeleton.get_base_config` method, and adds the point mass' mass.

    Returns:
      A `dictionary` containing the object instance's full configuration.
    """
    cfg = self.get_base_config()
    cfg['mass'] = self.mass
    return cfg


class TwoDofArm(Skeleton):
  """A two degrees-of-freedom planar arm.

  Args:
    name: `String`, the name of the skeleton.
    m1: `Float`, mass (kg) of the first bone.
    m2: `Float`, mass (kg) of the second bone.
    l1g: `Float`, position of the center of gravity of the first bone (m).
    l2g: `Float`, position of the center of gravity of the second bone (m).
    i1: `Float`, inertia (kg.m^2) of the first bone.
    i2: `Float`, inertia (kg.m^2) of the second bone.
    l1: `Float`, length (m) of the first bone.
    l2: `Float`, length (m) of the second bone.
    **kwargs: All contents are passed to the parent :class:`Skeleton` base class. Also allows for some backward
      compatibility.
  """

  def __init__(self, name: str = 'two_dof_arm', m1: float = 1.864572, m2: float = 1.534315, l1g: float = 0.180496,
         l2g: float = 0.181479, i1: float = 0.013193, i2: float = 0.020062, l1: float = 0.309,
         l2: float = 0.26, viscosity: float = 0., **kwargs):

    sho_limit = np.deg2rad([-0, 140])  # mechanical constraints - used to be -90 180
    elb_limit = np.deg2rad([0, 160])
    lb = [sho_limit[0], elb_limit[0]]
    ub = [sho_limit[1], elb_limit[1]]
    super().__init__(dof=2, space_dim=2, pos_lower_bound=lb, pos_upper_bound=ub, name=name, **kwargs)

    self.m1 = m1 # masses of arm links
    self.m2 = m2
    self.L1g = kwargs.get('L1g', l1g)  # center of mass of the links
    self.L2g = kwargs.get('L2g', l2g)
    self.I1 = kwargs.get('I1', i1)  # moment of inertia around center of mass
    self.I2 = kwargs.get('I2', i2)
    self.L1 = kwargs.get('L1', l1)  # length of links
    self.L2 = kwargs.get('L2', l2)

    # for consistency with args & backward compatibility
    self.l1g = self.L1g
    self.l2g = self.L2g
    self.i1 = self.I1
    self.i2 = self.I2
    self.l1 = self.L1
    self.l2 = self.L2

    # pre-compute values for mass, coriolis, and gravity matrices
    inertia_11_c = self.m1 * self.L1g ** 2 + self.I1 + self.m2 * (self.L2g ** 2 + self.L1 ** 2) + self.I2
    inertia_12_c = self.m2 * (self.L2g ** 2) + self.I2
    inertia_22_c = self.m2 * (self.L2g ** 2) + self.I2
    inertia_11_m = 2 * self.m2 * self.L1 * self.L2g
    inertia_12_m = self.m2 * self.L1 * self.L2g
    inertia_c = np.array([[[inertia_11_c, inertia_12_c], [inertia_12_c, inertia_22_c]]]).astype(np.float32)
    inertia_m = np.array([[[inertia_11_m, inertia_12_m], [inertia_12_m, 0.]]]).astype(np.float32)
    self.register_buffer('inertia_m', torch.tensor(inertia_m.reshape((1, 2, 2)), dtype=torch.float32))
    self.register_buffer('inertia_c', torch.tensor(inertia_c.reshape((1, 2, 2)), dtype=torch.float32))

    self.coriolis_1 = - self.m2 * self.L1 * self.L2g
    self.coriolis_2 = self.m2 * self.L1 * self.L2g
    self.c_viscosity = viscosity  # put at zero but available if implemented later on

  def _ode(self, inputs, joint_state, endpoint_load):
    # first two elements of state are joint position, last two elements are joint angular velocities
    pos0, pos1, vel0, vel1 = joint_state[:, 0], joint_state[:, 1], joint_state[:, 2], joint_state[:, 3]
    pos_sum = pos0 + pos1
    c1 = torch.cos(pos0)
    c2 = torch.cos(pos1)
    c12 = torch.cos(pos_sum)
    s1 = torch.sin(pos0)
    s2 = torch.sin(pos1)
    s12 = torch.sin(pos_sum)

    # inertia matrix (batch_size x 2 x 2)
    inertia = self.inertia_c + c2[:, None, None] * self.inertia_m

    # coriolis torques (batch_size x 2) plus a damping term (scaled by self.c_viscosity)
    coriolis_1 = (self.coriolis_1 * s2 * (2 * vel0 + vel1)) * vel1 + self.c_viscosity * vel0
    coriolis_2 = (self.coriolis_2 * s2 * vel0) * vel0 + self.c_viscosity * vel1
    coriolis = torch.stack([coriolis_1, coriolis_2], dim=1)

    # jacobian to distribute external loads (forces) applied at endpoint to the two rigid links
    jacobian_11 = -self.L1*s1 - self.L2*s12
    jacobian_12 = -self.L2*s12
    jacobian_21 = self.L1*c1 + self.L2*c12
    jacobian_22 = self.L2*c12

    # apply external loads
    # torque = jacobian.T @ endpoint_load
    r_col = (jacobian_11 * endpoint_load[:, 0]) + (jacobian_21 * endpoint_load[:, 1]) # these are torques
    l_col = (jacobian_12 * endpoint_load[:, 0]) + (jacobian_22 * endpoint_load[:, 1])
    torques = inputs + torch.stack([r_col, l_col], dim=1)

    rhs = -coriolis[:, :, None] + torques[:, :, None]

    denom = 1 / (inertia[:, 0, 0] * inertia[:, 1, 1] - inertia[:, 0, 1] * inertia[:, 1, 0])
    l_col = torch.stack([inertia[:, 1, 1], -inertia[:, 1, 0]], dim=1)
    r_col = torch.stack([-inertia[:, 0, 1], inertia[:, 0, 0]], dim=1)
    inertia_inv = denom[:, None, None] * torch.stack([l_col, r_col], dim=2)
    new_acc_3d = torch.matmul(inertia_inv, rhs)
    return new_acc_3d[:, :, 0]  # somehow tf.squeeze doesn't work well in a Lambda wrap...

  def _integrate(self, dt, state_derivative, joint_state):
    old_pos, old_vel = joint_state.chunk(2, dim=1)
    new_vel = old_vel + state_derivative * dt
    new_pos = old_pos + old_vel * dt

    # Clip to ensure values don't get off-hand.
    # We clip position after velocity to ensure any off-space position is taken into account when clipping velocity.
    new_vel = self.clip_velocity(new_pos, new_vel)
    new_pos = self.clip_position(new_pos)
    return torch.cat([new_pos, new_vel], dim=1)

  def _joint2cartesian(self, joint_state):
    # compute cartesian state from joint state
    # reshape to have all time steps lined up in 1st dimension
    j = torch.reshape(joint_state, shape=(-1, self.state_dim))
    pos0, pos1, vel0, vel1 = j[:, 0], j[:, 1], j[:, 2], j[:, 3]
    pos_sum = pos0 + pos1

    c1 = torch.cos(pos0)
    s1 = torch.sin(pos0)
    c12 = torch.cos(pos_sum)
    s12 = torch.sin(pos_sum)

    end_pos_x = self.L1 * c1 + self.L2 * c12
    end_pos_y = self.L1 * s1 + self.L2 * s12
    end_vel_x = - (self.L1 * s1 + self.L2 * s12) * vel0 - self.L2 * s12 * vel1
    end_vel_y = (self.L1 * c1 + self.L2 * c12) * vel0 + self.L2 * c12 * vel1
    return torch.stack([end_pos_x, end_pos_y, end_vel_x, end_vel_y], dim=1)

  def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
    n_points = path_fixation_body.numel()
    joint_angles, joint_vel = joint_state.chunk(2, dim=1)

    sho, elb_wrt_sho = joint_angles.chunk(2, axis=1)
    elb = elb_wrt_sho + sho
    elb_y = self.L1 * torch.sin(sho)[:, :, None]
    elb_x = self.L1 * torch.cos(sho)[:, :, None]

    # If we want the position of a fixation point relative to the origin of its bone given global cartesian
    # coordinates, then we use the joint angles; here we are trying to do the inverse of that, that is getting the
    # fixation point in global cartesian coordinates given its position relative to the origin of the bone it is
    # fixed on. Therefore we use a minus in front of the angles since we are doing the inverse rotation.
    # This line picks no rotation angle if the muscle path point is fixed on the extrinstic workspace
    # (path_fixation_body = 0), the shoulder angle if it is fixed on the upper arm (path_fixation_body = 1) and the
    # eblow angle if it is fixed on the forearm (path_fixation_body = 2).
    flat_path_fixation_body = path_fixation_body.reshape(-1)
    ang = torch.where(flat_path_fixation_body == 0., 0., torch.where(flat_path_fixation_body == 1., -sho, -elb))
    ca = torch.cos(ang)
    sa = torch.sin(ang)

    # rotation matrix to transform the bone-relative coordinates into global coordinates
    rot1 = torch.cat([ca, sa], dim=1).reshape(-1, 2, n_points)
    rot2 = torch.cat([-sa, ca], dim=1).reshape(-1, 2, n_points)

    # derivative of each fixation point's position wrt the angle of the bone they are fixed on
    dx_da = torch.sum(-path_coordinates * rot2, dim=1, keepdims=True)
    dy_da = torch.sum(path_coordinates * rot1, dim=1, keepdims=True)

    # Derivative of each fixation point's position wrt each angle
    # This is counter-intuitive but the derivative of any point wrt the shoulder angle (da1) is equal to the
    # derivative of that point wrt the angle of the bone they are actually fixed on (dx_da or dy_da), even if that
    # bone is the forearm and not the upper arm. However, if the bone is indeed the forearm, then an additional term
    # must be added (see below).
    dx_da1 = torch.where(path_fixation_body == 0., 0., dx_da) + torch.where(path_fixation_body == 2., -elb_y, 0.)
    dy_da1 = torch.where(path_fixation_body == 0., 0., dy_da) + torch.where(path_fixation_body == 2., elb_x, 0.)
    dx_da2 = torch.where(path_fixation_body == 2., dx_da, 0.)
    dy_da2 = torch.where(path_fixation_body == 2., dy_da, 0.)

    dxy_da1 = torch.cat([dx_da1, dy_da1], dim=1)
    dxy_da2 = torch.cat([dx_da2, dy_da2], dim=1)
    dxy_da = torch.cat([dxy_da1[:, :, None, :], dxy_da2[:, :, None, :]], dim=2)

    sho_vel_3d = joint_vel[:, 0, None, None]
    elb_vel_3d = joint_vel[:, 1, None, None] + sho_vel_3d
    dxy_dt = dxy_da1 * sho_vel_3d + dxy_da2 * elb_vel_3d # by virtue of the chain rule

    bone_origin = torch.where(path_fixation_body == 2, torch.cat([elb_x, elb_y], dim=1), 0.)
    xy = torch.cat([dy_da, -dx_da], dim=1) + bone_origin
    return xy, dxy_dt, dxy_da

  def get_save_config(self) -> dict:
    """Gets the base configuration from the :meth:`Skeleton.get_base_config` method, and adds the arm's properties
    to the configuration: each bone length, mass, center of gravity, coriolis forces, and viscosity parameters.

    Returns:
      A `dictionary` containing the object instance's full configuration.
    """
    cfg = self.get_base_config()
    cfg.update(
      {
        'I1': self.detach(self.I1),
        'I2': self.detach(self.I2),
        'L1': self.detach(self.L1),
        'L2': self.detach(self.L2),
        'L1g': self.detach(self.L1g),
        'L2g': self.detach(self.L2g),
        'c_viscosity': self.detach(self.c_viscosity),
        'coriolis_1': self.detach(self.coriolis_1),
        'coriolis_2': self.detach(self.coriolis_2),
        'm1': self.detach(self.m1),
        'm2': self.detach(self.m2)
        }
       )
    return cfg
