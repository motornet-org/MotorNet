import numpy as np
import torch
import gymnasium as gym
from gymnasium.utils import seeding
from typing import Any


class Environment(gym.Env, torch.nn.Module):
  """Base class for environments.

  Args:
    effector: :class:`motornet.effector.Effector` object class or subclass. This is the effector that will evolve in
      the environment.
    q_init: `Tensor` or `numpy.ndarray`, the desired initial joint states for the environment, if a single set
      of pre-defined initial joint states is desired. If `None`, the initial joint states will be drawn from the
      :class:`motornet.nets.layers.Network.get_initial_state` method at each call of :meth:`generate`. This parameter
      will be ignored on :meth:`generate` calls where a `joint_state` is provided as input
      argument.
    name: `String`, the name of the environment object instance.
    differentiable: `Boolean`, whether the environment will be differentiable or not. This will usually be useful for
      reinforcement learning, where the differentiability is not needed.
    max_ep_duration: `Float`, the maximum duration of an episode, in seconds.
    action_noise: `Float`, the standard deviation of the Gaussian noise added to the action input at each step of the 
      simulation.
    obs_noise: `Float` or `list`, the standard deviation of the Gaussian noise added to the observation vector at each
      step of the simulation. If this is a `list`, it should have as many elements as the observation vector and will 
      indicate the standard deviation of each observation element independently.
    action_frame_stacking: `Integer`, the number of past action steps to add to the observation vector.
    proprioception_delay: `Float`, the delay in seconds for the proprioceptive feedback to be added to the observation 
      vector. If `None`, no delay will occur.
    vision_delay: `Float`, the delay in seconds for the visual feedback to be added to the observation vector. If 
      `None`, no delay will occur.
    proprioception_noise: `Float`, the standard deviation of the Gaussian noise added to the proprioceptive feedback at
      each step of the simulation.
    vision_noise: `Float`, the standard deviation of the Gaussian noise added to the visual feedback at each step of the
      simulation.
    **kwargs: This is passed as-is to the :class:`torch.nn.Module` parent class.
  """
  def __init__(
    self,
    effector,
    q_init=None,
    name: str = 'Env',
    differentiable: bool = True,
    max_ep_duration: float = 1.,
    action_noise: float = 0.,
    obs_noise: float | list = 0.,
    action_frame_stacking: int = 0,
    proprioception_delay: float = None,
    vision_delay: float = None,
    proprioception_noise: float = 0.,
    vision_noise: float = 0.,
    **kwargs,
  ):
    
    super().__init__(**kwargs)

    self._device = torch.device('cpu')
    self.__name__ = name
    self.effector = effector.to(self.device)
    self.dt = self.effector.dt
    self.differentiable = differentiable
    self.max_ep_duration = max_ep_duration
    self.elapsed = None

    self.delay_range = [0, 0]

    if q_init is not None:
      q_init = np.array(q_init)
      if len(q_init.shape) == 1:
        q_init = q_init.reshape(1, -1)
      self.nq_init = q_init.shape[0]
    else:
      self.nq_init = None
    self.q_init = q_init
    self.goal = None

    self._action_noise = action_noise
    self._obs_noise = obs_noise
    self.action_noise = 0.
    self.obs_noise = 0.
    self.proprioception_noise = [proprioception_noise]
    self.vision_noise = [vision_noise]
    self.action_frame_stacking = action_frame_stacking

    # default is no delay
    proprioception_delay = self.dt if proprioception_delay is None else proprioception_delay
    vision_delay = self.dt if vision_delay is None else vision_delay

    def floating_precision(x): return x < np.finfo(x).resolution
    assert floating_precision(np.mod(vision_delay / self.dt, 1.)), f"delay was {vision_delay} and dt was {self.dt}"
    assert floating_precision(np.mod(proprioception_delay / self.dt, 1.)), f"delay was {proprioception_delay} and " + \
      f"dt was {self.dt}"

    self.proprioception_delay = int(proprioception_delay / self.dt)
    self.vision_delay = int(vision_delay / self.dt)
    
    self.obs_buffer = {
      "proprioception": [None] * self.proprioception_delay,
      "vision":  [None] * self.vision_delay,
      "action": [None] * self.action_frame_stacking
    }

    self.seed = None
    self._build_spaces()

  def detach(self, x):
    return x.cpu().detach().numpy() if torch.is_tensor(x) else x
  
  def _build_spaces(self):
    self.action_space = gym.spaces.Box(low=0., high=1., shape=(self.effector.n_muscles,), dtype=np.float32)

    obs, _ = self.reset(options={"deterministic": True})
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32)

    def handle_noise_arg(noise, space):
      return [noise] * space.shape[0] if (type(noise) is float or int) else noise

    self.action_noise = handle_noise_arg(self._action_noise, self.action_space)
    self.obs_noise = handle_noise_arg(self._obs_noise, self.observation_space)
  
  # # # ========================================================================================
  # # # ========================================================================================
  # # # The methods below are those MOST likely to be overwritten by users creating custom tasks
  # # # ========================================================================================
  # # # ========================================================================================
  def get_proprioception(self) -> torch.Tensor:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the instantaneous (non-delayed) proprioceptive 
    feedback. By default, this is the normalized muscle length for each muscle, followed by the normalized
    muscle velocity for each muscle as well. `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
    using the :attr:`proprioception_noise` attribute.
    """
    mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
    mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
    prop = torch.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
    return self.apply_noise(prop, self.proprioception_noise)

  def get_vision(self) -> torch.Tensor:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the instantaneous (non-delayed) visual 
    feedback. By default, this is the cartesian position of the end-point effector, that is, the fingertip.
    `.i.i.d.` Gaussian noise is added to each element in the `tensor`, using the
    :attr:`vision_noise` attribute.
    """
    vis = self.states["fingertip"]
    return self.apply_noise(vis, self.vision_noise)

  def get_obs(self, action=None, deterministic: bool = False) -> torch.Tensor | np.ndarray:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the (potientially time-delayed) observations.
    By default, this is the task goal, followed by the output of the :meth:`get_proprioception()` method, 
    the output of the :meth:`get_vision()` method, and finally the last :attr:`action_frame_stacking` action sets,
    if a non-zero `action_frame_stacking` keyword argument was passed at initialization of this class instance.
    `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
    using the :attr:`obs_noise` attribute.
    """
    self.update_obs_buffer(action=action)

    obs_as_list = [
      self.goal,
      self.obs_buffer["vision"][0],
      self.obs_buffer["proprioception"][0],
      ] + self.obs_buffer["action"][:self.action_frame_stacking]
    
    obs = torch.cat(obs_as_list, dim=-1)

    if deterministic is False:
      obs = self.apply_noise(obs, noise=self.obs_noise)

    return obs if self.differentiable else self.detach(obs)

  def step(
      self,
      action: torch.Tensor | np.ndarray,
      deterministic: bool = False,
      **kwargs,
    ) -> tuple[torch.Tensor | np.ndarray, bool, bool, dict[str, Any]]:
    """
    Perform one simulation step. This method is likely to be overwritten by any subclass to implement user-defined 
    computations, such as reward value calculation for reinforcement learning, custom truncation or termination
    conditions, or time-varying goals.
    
    Args:
      action: `Tensor` or `numpy.ndarray`, the input drive to the actuators.
      deterministic: `Boolean`, whether observation, action, proprioception, and vision noise are applied.
      **kwargs: This is passed as-is to the :meth:`motornet.effector.Effector.step()` call. This is maily useful to pass
      `endpoint_load` or `joint_load` kwargs.
  
    Returns:
      - The observation vector as `tensor` or `numpy.ndarray`, if the :class:`Environment` is set as differentiable or 
        not, respectively. It has dimensionality `(batch_size, n_features)`.
      - A `numpy.ndarray` with the reward information for the step, with dimensionality `(batch_size, 1)`. This is 
        `None` if the :class:`Environment` is set as differentiable. By default this always returns `0.` in the 
        :class:`Environment`.
      - A `boolean` indicating if the simulation has been terminated or truncated. If the :class:`Environment` is set as
        differentiable, this returns `True` when the simulation time reaches `max_ep_duration` provided at 
        initialization.
      - A `boolean` indicating if the simulation has been truncated early or not. This always returns `False` if the
        :class:`Environment` is set as differentiable.
      - A `dictionary` containing this step's information.
    """
    
    self.elapsed += self.dt

    action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.float32).to(self.device)
    noisy_action = action
    if deterministic is False:
      noisy_action = self.apply_noise(noisy_action, noise=self.action_noise)
    
    self.effector.step(noisy_action, **kwargs)

    obs = self.get_obs(action=noisy_action)
    reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
    truncated = False if self.differentiable else False
    terminated = bool(self.elapsed >= self.max_ep_duration) or truncated
    self.goal = self.goal.clone()
    info = {
      "states": self._maybe_detach_states(),
      "action": action,
      "noisy action": noisy_action,
      "goal": self.goal if self.differentiable else self.detach(self.goal),
      }

    return obs, reward, terminated, truncated, info

  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
    """
    Initialize the task goal and :attr:`effector` states for a (batch of) simulation episode(s). The :attr:`effector`
    states (joint, cartesian, muscle, geometry) are initialized to be biomechanically compatible with each other.
    This method is likely to be overwritten by any subclass to implement user-defined computations, such as defining 
    a custom initial goal or initial states.

    Args:
      seed: `Integer`, the seed that is used to initialize the environment's PRNG (`np_random`).
        If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
        a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
        However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
        If you pass an integer, the PRNG will be reset even if it already exists.
        Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
      options: `Dictionary`, optional kwargs specific to motornet environments. This is mainly useful to pass
        `batch_size`, `joint_state`, and `deterministic` kwargs if desired, as described below.

    Options:
      - **batch_size**: `Integer`, the desired batch size. Default: `1`.
      - **joint_state**: The joint state from which the other state values are inferred. If `None`, the `q_init` value 
        declared during the class instantiation will be used. If `q_init` is also `None`, random initial joint
        states are drawn, from which the other state values are inferred. Default: `None`.
      - **deterministic**: `Boolean`, whether observation, proprioception, and vision noise are applied.
        Default: `False`.

    Returns:
      - The observation vector as `tensor` or `numpy.ndarray`, if the :class:`Environment` is set as differentiable or 
        not, respectively. It has dimensionality `(batch_size, n_features)`.
      - A `dictionary` containing the initial step's information.
    """
    self._set_generator(seed=seed)

    options = {} if options is None else options
    batch_size: int = options.get("batch_size", 1)
    joint_state: torch.Tensor | np.ndarray | None = options.get("joint_state", None)
    deterministic: bool = options.get("deterministic", False)

    if joint_state is not None:
      joint_state_shape = np.shape(self.detach(joint_state))
      if joint_state_shape[0] > 1:
        batch_size = joint_state_shape[0]
    else:
      joint_state = self.q_init

    #self.muscle_normalizer = self.detach(self.muscle.max_iso_force / torch.mean(self.muscle.max_iso_force))
    # goal = self.joint2cartesian(self.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)
    self.goal = torch.zeros((batch_size, self.skeleton.space_dim)).to(self.device)
    self.elapsed = 0.

    self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})
    
    # initialize buffer
    action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
    self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
    self.obs_buffer["action"] = [action] * self.action_frame_stacking

    action = action if self.differentiable else self.detach(action)

    obs = self.get_obs(deterministic=deterministic)
    info = {
        "states": self._maybe_detach_states(),
        "action": action,
        "noisy action": action,
        "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
    return obs, info
  
  # # # ========================================================================================
  # # # ========================================================================================
  # # # The methods below are those LESS likely to be overwritten by users creating custom tasks
  # # # ========================================================================================
  # # # ========================================================================================
  def update_obs_buffer(self, action=None):
    self.obs_buffer["proprioception"] = self.obs_buffer["proprioception"][1:] + [self.get_proprioception()]
    self.obs_buffer["vision"] = self.obs_buffer["vision"][1:] + [self.get_vision()]

    if action is not None:
      self.obs_buffer["action"] = self.obs_buffer["action"][1:] + [action.reshape(-1, self.action_space.shape[0])]

  @property
  def muscle(self):
    """Shortcut to the :class:`motornet.effector.Effector`'s `muscle` attribute."""
    return self.effector.muscle
  
  @property
  def n_muscles(self):
    """Shortcut to the :class:`motornet.muscle.Muscle`'s `n_muscles` attribute."""
    return self.effector.muscle.n_muscles
  
  @property
  def skeleton(self):
    """Shortcut to the :class:`motornet.effector.Effector`'s `skeleton` attribute."""
    return self.effector.skeleton
  
  @property
  def space_dim(self):
    """Shortcut to the :class:`motornet.skeleton.Skeleton`'s `space_dim` attribute."""
    return self.effector.skeleton.space_dim
  
  @property
  def states(self):
    """Shortcut to the :class:`motornet.effector.Effector`'s `states` attribute."""
    return self.effector.states

  def _maybe_detach_states(self):
    return self.states if self.differentiable else {key: self.detach(val) for key, val in self.states.items()}
  
  def joint2cartesian(self, joint_states):
    """Shortcut to :meth:`motornet.effector.Effector.joint2cartesian()` method."""
    return self.effector.joint2cartesian(joint_states)
  
  def _set_generator(self, seed: int | None):
    if seed is not None:
      self.effector.reset(seed=seed)

  @property
  def np_random(self) -> np.random.Generator:
    """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

    Returns:
      Instances of `np.random.Generator`
    """
    return self.effector.np_random

  @np_random.setter
  def np_random(self, rng: np.random.Generator) -> None:
      self.effector.np_random = rng

  def apply_noise(self, loc, noise: float | list) -> torch.Tensor:
    """Applies element-wise Gaussian noise to the input `loc`.

    Args:
      loc: input on which the Gaussian noise is applied, which in probabilistic terms make it the mean of the
        Gaussian distribution.
      noise: `Float` or `list`, the standard deviation (spread or “width”) of the distribution. Must be 
        non-negative. If this is a `list, it must contain as many elements as the second axis of `loc`, and the 
        Gaussian distribution for each column of `loc` will have a different standard deviation. Note that the 
        elements within each column of `loc` will still be independent and identically distributed (`i.i.d.`).

    Returns:
      A noisy version of `loc` as a `tensor`.
    """
    if isinstance(noise[0], float):
      white_noise = self.np_random.normal(size=(loc.shape[0], loc.shape[1]), scale=noise)
    else:
      white_noise = self.np_random.normal(size=(loc.shape[0], len(noise[0])), scale=noise)
    return loc + torch.tensor(white_noise, dtype=torch.float32).to(self.device)

  def get_attributes(self):
    """Gets all non-callable attributes declared in the object instance, excluding `gym.spaces.Space` attributes,
    the effector, muscle, and skeleton attributes.

    Returns:
      - A `list` of attribute names as `string` elements.
      - A `list` of attribute values.
    """
    blacklist = ["effector", "muscle", "skeleton", "np_random", "states", "goal", "obs_buffer", "unwrapped"]
    attributes = [
      a for a in dir(self)
      if not a.startswith('_')
      and not callable(getattr(self, a))
      and not blacklist.__contains__(a)
      and not isinstance(getattr(self, a), gym.spaces.Space)  # Spaces are not JSON serializable
    ]
    values = [getattr(self, a) for a in attributes]
    return attributes, values

  def print_attributes(self):
    """Prints all non-callable attributes declared in the object instance, excluding `gym.spaces.Space` attributes,
      the effector, muscle, and skeleton attributes."""
    attributes = [a for a in dir(self) if not a.startswith('_') and not callable(getattr(self, a))]
    blacklist = []

    for a in attributes:
      if not blacklist.__contains__(a):
        print(a + ": ", getattr(self, a))

      for elem in blacklist:
        print("\n" + elem + ":\n", getattr(self, elem))

  def get_save_config(self):
    """Gets the environment object's configuration as a `dictionary`.

    Returns:
      A `dictionary` containing the  parameters of the environment's configuration. All parameters held as 
      non-callable attributes by the object instance will be included in the `dictionary`, excluding
      `gym.spaces.Space` attributes, the effector, muscle, and skeleton attributes.
    """

    cfg = {'name': self.__name__}
    attributes, values = self.get_attributes()
    for attribute, value in zip(attributes, values):
      value = self.detach(value)  # tensors are not JSON serializable
      if isinstance(value, np.ndarray):
        value = value.tolist()
      if attribute in ("T_destination", "device"):
        value = str(value)
      cfg[attribute] = value

    cfg["effector"] = self.effector.get_save_config()
    return cfg
  
  def to(self, *args, **kwargs):
    if args and isinstance(args[0], (str, torch.device)):
      self._device = torch.device(args[0])
    elif args and isinstance(args[0], torch.Tensor):
      self._device = args[0].device
    elif 'device' in kwargs:
      self._device = torch.device(kwargs['device'])
    return super().to(*args, **kwargs)

  @property
  def device(self):
    return self._device


class RandomTargetReach(Environment):
  """A reach to a random target from a random starting position.

  Args:
    network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
      the task.
    name: `String`, the name of the task object instance.
    deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
      loss.
    **kwargs: This is passed as-is to the parent :class:`Task` class.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless

  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """
    Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to 
    change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
    the full joint space.
    """
    self._set_generator(seed=seed)

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: torch.Tensor | np.ndarray | None = options.get('joint_state', None)
    deterministic: bool = options.get('deterministic', False)
    
    if joint_state is not None:
      joint_state_shape = np.shape(self.detach(joint_state))
      if joint_state_shape[0] > 1:
        batch_size = joint_state_shape[0]
    else:
      joint_state = self.q_init

    self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

    self.goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
    self.elapsed = 0.

    action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)

    self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
    self.obs_buffer["action"] = [action] * self.action_frame_stacking

    action = action if self.differentiable else self.detach(action)

    obs = self.get_obs(deterministic=deterministic)
    info = {
      "states": self._maybe_detach_states(),
      "action": action,
      "noisy action": action,
      "goal": self.goal if self.differentiable else self.detach(self.goal),
      }
    return obs, info
