import torch
import numpy as np


class Muscle(torch.nn.Module):
  """Base class for `Muscle` objects. If a effector contains several muscles, this object will contain all of
  those in a vectorized format, meaning for any given effector there will always be only `one` muscle object, 
  regardless of the number of muscles wrapped around the skeleton.

  The dimensionality of the muscle states produced by this object and subclasses will always be
  `n_batches * n_timesteps * n_states * n_muscles`.

  Args:
    input_dim: `Integer`, the dimensionality of the drive input to the muscle. For instance, if the muscle is only
      driven by an excitation signal, then this value should be `1`.
    output_dim: `Integer`, the dimensionality of the output. Since this object does not rely on a ``call`` method,
      but on an :meth:`integrate` method, this is the dimensionality of the :meth:`integrate` method's output.
      The output of that method should be a `muscle state`, so `output_dim` is usually the number of states
      of the :class:`Muscle` object class or subclass.
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    tau_activation: `Float`, the time constant for activation of the muscle. This is used for the Ordinary
      Differential Equation of the muscle activation method :meth:`activation_ode`.
    tau_deactivation: `Float`, the time constant for deactivation of the muscle. This is used for the Ordinary
      Differential Equation of the muscle activation method :meth:`activation_ode`.
  """

  def __init__(
    self, input_dim: int = 1,
    output_dim: int = 1,
    min_activation: float = 0.,
    tau_activation: float = 0.015,
    tau_deactivation: float = 0.05
  ):
    
    super().__init__()

    self._device = torch.device('cpu')
    self.input_dim = input_dim
    self.state_name = []
    self.output_dim = output_dim
    self.register_buffer('min_activation', torch.tensor(min_activation, dtype=torch.float32))
    self.register_buffer('tau_activation', torch.tensor(tau_activation, dtype=torch.float32))
    self.register_buffer('tau_deactivation', torch.tensor(tau_deactivation, dtype=torch.float32))
    self.to_build_dict = {'max_isometric_force': []}
    self.to_build_dict_default = {}
    self.dt = None
    self.n_muscles = None
    self.register_buffer('max_iso_force', None)
    self.register_buffer('vmax', None)
    self.register_buffer('l0_se', None)
    self.register_buffer('l0_ce', None)
    self.register_buffer('l0_pe', None)
    self.built = False

  def clip_activation(self, a):
    return torch.clamp(a, self.min_activation, 1.)
  
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

  def build(self, timestep, max_isometric_force, **kwargs):
    """Build the muscle given parameters from the ``motornet.effector.Effector`` wrapper object. This should be 
    called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle scructure according to 
    the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a `list` containing as many
        elements as there are muscles.
      **kwargs: Optional keyword arguments. This allows for extra parameters to be passed in the :meth:`build`
        method for a :class:`Muscle` subclass, if needed.
    """
    max_isometric_force = np.array(max_isometric_force).reshape(1, 1, -1)
    self.n_muscles = np.array(max_isometric_force).size
    self.register_buffer('max_iso_force', torch.tensor(max_isometric_force, dtype=torch.float32))
    self.dt = timestep
    self.register_buffer('vmax', torch.ones((1, 1, self.n_muscles)))
    self.register_buffer('l0_se', torch.ones((1, 1, self.n_muscles)))
    self.register_buffer('l0_ce', torch.ones((1, 1, self.n_muscles)))
    self.register_buffer('l0_pe', torch.ones((1, 1, self.n_muscles)))
    self.built = True

  def get_initial_muscle_state(self, batch_size, geometry_state):
    """Infers the `muscle state` matching a provided `geometry state` array.

    Args:
      batch_size: `Integer`, the size of the batch passed in `geometry state`.
      geometry_state: `Tensor`, the `geometry state` array from which the matching initial `muscle state` is
        inferred.

    Returns:
      A `tensor` containing the initial `muscle state` matching the input `geometry state` array.
    """
    return self._get_initial_muscle_state(batch_size, geometry_state)

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    raise NotImplementedError

  def integrate(self, dt, state_derivative, muscle_state, geometry_state):
    """Performs one integration step for the muscle step.

    Args:
      dt: `Float`, size of the timestep in seconds for this integration step.
      state_derivative: `Tensor`, the derivatives of the `muscle state`. These are usually obtained using this
        object's :meth:`ode` method.
      muscle_state: `Tensor`, the `muscle state` used as the initial state value for the numerical integration.
      geometry_state: `Tensor`, the `geometry state` used as the initial state value for the numerical
        integration.

    Returns:
      A `tensor` containing the new `muscle state` following numerical integration.
    """
    return self._integrate(dt, state_derivative, muscle_state, geometry_state)
  
  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    raise NotImplementedError

  def ode(self, action, muscle_state):
    """Computes the derivatives of `muscle state` using the corresponding Ordinary Differential Equations.

    Args:
      action: `Tensor`, the descending excitation drive to the muscle(s).
      muscle_state: `Tensor`, the `muscle state` used as the initial value for the evaluation of the Ordinary
        Differential Equations.

    Returns:
      A `tensor` containing the derivatives of the `muscle state`.
    """
    return self._ode(action, muscle_state)

  def _ode(self, action, muscle_state):
    activation = muscle_state[:, :1, :]
    return self.activation_ode(action, activation)

  def activation_ode(self, action, activation):
    """Computes the new activation value of the (set of) muscle(s) according to the Ordinary Differential Equation
    shown in equations 1-2 in `[1]`. Note that this is incidentally the same activation function as used for the 
    `muscle` actuators in MuJoCo `[2]`.

    References:
      [1] `Thelen DG. Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older
      adults. J Biomech Eng. 2003 Feb;125(1):70-7. doi: 10.1115/1.1531112. PMID: 12661198.`
      [2] https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators

    Args:
      excitation: `Float` or `list` of `float`, the descending excitation drive to the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a `list` containing as many
        elements as there are muscles in that parent effector object.
      muscle_state: `Tensor`, the `muscle state` that provides the initial activation value for the Ordinary
        Differential Equation.

    Returns:
      A `tensor` containing the updated activation values.
    """
    action = self.clip_activation(torch.reshape(action, (-1, 1, self.n_muscles)))
    activation = self.clip_activation(activation)
    tmp = 0.5 + 1.5 * activation
    tau = torch.where(action > activation, self.tau_activation * tmp, self.tau_deactivation / tmp)
    return (action - activation) / tau

  def setattr(self, name: str, value):
    """Changes the value of an attribute held by this object.

    Args:
      name: `String`, attribute to set to a new value.
      value: Value that the attribute should take.
    """
    self.__setattr__(name, value)

  def get_save_config(self):
    """Gets the object instance's configuration. This is the set of configuration entries that will be useful
    for any muscle objects or subclasses.

    Returns:
       - A `dictionary` containing the muscle object's name and state names.
    """
    cfg = {'name': str(self.__name__), 'state names': self.state_name}
    return cfg


class ReluMuscle(Muscle):
  """A "rectified linear" muscle whose force output :math:`F` is a linear function of its activation value, which
  itself is bounded between `0` and `1`. Specifically:

  .. math::
    F = m * activation

  with :math:`m` the maximum isometric force of the muscle. Note that the maximum isometric force is not declared at
  initialization but via the :meth:`Muscle.build` call, which is inherited from the parent :class:`Muscle` class.
  The :math:`activation` value is the result of an Ordinary Differential Equation computed by the
  :meth:`Muscle.activation_ode` method. It is not directly the `action` input drive.

  Args:
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.__name__ = 'ReluMuscle'
    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force'
      ]
    self.state_dim = len(self.state_name)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = muscle_state[:, :1, :] + state_derivative * dt
    activation = self.clip_activation(activation)
    forces = activation * self.max_iso_force
    len_vel = geometry_state[:, :2, :]
    return torch.cat([activation, len_vel, forces], dim=1)

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    activation0 = torch.ones((batch_size, 1, self.n_muscles), device=self.device) * self.min_activation
    force0 = torch.zeros((batch_size, 1, self.n_muscles), device=self.device)
    len_vel = geometry_state[:, 0:2, :]
    return torch.cat([activation0, len_vel, force0], dim=1)



class MujocoHillMuscle(Muscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in the MuJoCo 
  documentation`[1]`. It is a rigid tendon Hill-type model.

  References:
    [1] https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    passive_forces: `Float`, a scalar coefficient to tune the contribution of passive forces to the total force
      output.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(
    self,
    min_activation: float = 0.,
    passive_forces: float = 1.,
    tau_activation: float = 0.01,
    tau_deactivation: float = 0.04,
    **kwargs,
  ):
    
    super().__init__(
      min_activation=min_activation,
      tau_activation=tau_activation,
      tau_deactivation=tau_deactivation,
      **kwargs,
      )
    self.__name__ = 'MujocoHillMuscle'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length CE',
      'force-velocity CE',
      'force'
      ]
    self.state_dim = len(self.state_name)

    self.to_build_dict = {
      'max_isometric_force': [],
      'optimal_muscle_length': [],
      "tendon_length": [],
      'normalized_slack_muscle_length': [],
      "lmin": [],
      "lmax": [],
      "vmax": [],
      "fvmax": [],
      }
    self.to_build_dict_default = {
      'normalized_slack_muscle_length': 1.3,
      "lmin": 0.5,
      "lmax": 1.6,
      "vmax": 1.5,
      "fvmax": 1.2,
      }
    self.built = False
    self.passive_forces = passive_forces

  def build(
    self,
    timestep,
    max_isometric_force,
    tendon_length,
    optimal_muscle_length,
    normalized_slack_muscle_length,
    lmin,
    lmax,
    vmax,
    fvmax,
  ):
    """Build the muscle using arguments from the :class:`motornet.effector.Effector` wrapper object. This
    should be called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle 
    structure according to the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a list containing as many
        elements as there are muscles.
      tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a list containing as many 
        elements as there are muscles in that parent effector object.
      optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
        the length at which the muscle will output the maximum amount of force given the same excitation. If
        several muscles are declared in the parent effector object, then this should be a list containing as
        many elements as there are muscles in that parent effector object.
      normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
        muscle(s) will start to developp passive forces. If several muscles are declared in the parent effector
        object, then this should be a list containing as many elements as there are muscles in that parent
        effector object.
      lmin: `Float`, lower bound on the operating range of the muscle length (normalized by its optimal length).
      lmax: `Float`, upper bound on the operating range of the muscle length (normalized by its optimal length).
      vmax: `Float`, shortening velocity at which muscle force drops to zero (normalized by its optimal length per
        sec).
      fvmax: `Float`, active force generated at saturating lengthening velocity, relative to its maximum isometric
        force.
    """
    self.n_muscles = np.array(tendon_length).size
    
    def to_tensor(x):
      tensor = torch.tensor(x, dtype=torch.float32).reshape((1, 1, -1))
      if tensor.numel() != 1 and tensor.numel() != self.n_muscles:
        raise ValueError(
          f"Expected a scalar or a vector of length n_muscles={self.n_muscles}, "
          f"got {tensor.numel()} elements."
        )
      return tensor

    self.register_buffer('max_iso_force', to_tensor(max_isometric_force))
    self.register_buffer('l0_pe', to_tensor(normalized_slack_muscle_length))
    self.register_buffer('l0_ce', to_tensor(optimal_muscle_length))
    self.register_buffer('l0_se', to_tensor(tendon_length))
    self.register_buffer('lmin', to_tensor(lmin))
    self.register_buffer('lmax', to_tensor(lmax))
    self.register_buffer('vmax', to_tensor(vmax))
    self.register_buffer('fvmax', to_tensor(fvmax))

    self.dt = timestep


    # derived quantities
    # a = 0.5*(lmin+1)
    self.register_buffer('b', 0.5 * (1 + self.lmax))
    self.register_buffer('c', self.fvmax - 1)
    self.register_buffer('p1', self.b - 1)
    self.register_buffer('p2', 0.25 * self.l0_pe)
    self.register_buffer('zero_as_tensor', torch.tensor(0., dtype=torch.float32))
    self.register_buffer('mid', 0.5 * (self.lmin + 0.95))

    self.built = True

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    shape = geometry_state[:, :1, :].shape
    muscle_state = torch.ones(shape, dtype=torch.float32, device=self.device) * self.min_activation
    state_derivatives = torch.zeros(shape, dtype=torch.float32, device=self.device)
    return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = muscle_state[:, :1, :] + state_derivative * dt
    activation = self.clip_activation(activation)

    # musculotendon geometry
    musculotendon_len = geometry_state[:, :1, :]
    muscle_len = torch.clip(musculotendon_len - self.l0_se, min=0.001) / self.l0_ce
    muscle_vel = geometry_state[:, 1:2, :] / self.vmax

    # muscle forces
    x = torch.where(
      muscle_len <= 1,
      input=self.zero_as_tensor,
      other=torch.where(
        torch.less_equal(muscle_len, self.b),
        input=(muscle_len - 1) / self.p1,
        other=(muscle_len - self.b) / self.p1,
        )
      )
    flpe = torch.where(
      torch.less_equal(muscle_len, 1),
      input=self.zero_as_tensor,
      other=torch.where(
        muscle_len <= self.b,
        input=self.p2 * x**3,
        other=self.p2 * (1 + 3*x),
        )
      )

    # length-active
    flce = self._bump(muscle_len, mid=1, lmax=self.lmax) + 0.15 * self._bump(muscle_len, mid=self.mid, lmax=0.95)

    # velocity-active
    fvce = torch.where(
      torch.less_equal(muscle_vel, -1),
      input=self.zero_as_tensor,
      other=torch.where(
        torch.less_equal(muscle_vel, 0.),
        input=(muscle_vel+1) * (muscle_vel+1),
        other=torch.where(
          torch.less_equal(muscle_vel, self.c),
          input=self.fvmax - (self.c-muscle_vel)*(self.c-muscle_vel)/self.c,
          other=self.fvmax,
          )
        )
      )
    force = (activation * flce * fvce + self.passive_forces * flpe) * self.max_iso_force
    return torch.cat([activation, muscle_len * self.l0_ce, muscle_vel * self.vmax, flpe, flce, fvce, force], dim=1)

    
  def _bump(self, L, mid, lmax):
    """Skewed bump function: quadratic spline."""

    left = 0.5*(self.lmin+mid)
    right = 0.5*(mid+lmax)

    out_of_range = torch.logical_or(torch.less_equal(L, self.lmin), torch.greater_equal(L, lmax))
    less_than_left = torch.less(L, left)
    less_than_mid = torch.less(L, mid)
    less_than_right = torch.less(L, right)

    x = torch.where(out_of_range, input=self.zero_as_tensor,
      other=torch.where(less_than_left, input=(L-self.lmin) / (left-self.lmin),
        other=torch.where(less_than_mid, input=(mid-L) / (mid-left),
          other=torch.where(less_than_right, input=(L-mid) / (right-mid),
            other=(lmax-L) / (lmax-right),
            )
          )
        )
      )
    pfivexx = 0.5 * x * x
    y = torch.where(out_of_range, input=self.zero_as_tensor,
      other=torch.where(less_than_left, input=pfivexx,
        other=torch.where(less_than_mid, input=1 - pfivexx,
          other=torch.where(less_than_right, input=1 - pfivexx,
            other=pfivexx,
            )
          )
        )
      )
    return y


class RigidTendonHillMuscle(Muscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`, adjusted to
  behave as a rigid tendon version of the original model.

  References:
    [1] `Kistemaker DA, Wong JD, Gribble PL. The central nervous system does not minimize energy cost in arm
    movements. J Neurophysiol. 2010 Dec;104(6):2985-94. doi: 10.1152/jn.00483.2010. Epub 2010 Sep 8. PMID:
    20884757.`

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, min_activation=0.001, **kwargs):
    super().__init__(min_activation=min_activation, **kwargs)
    self.__name__ = 'RigidTendonHillMuscle'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length CE',
      'force-velocity CE',
      'force'
    ]
    self.state_dim = len(self.state_name)

    # parameters for the passive element (PE) and contractile element (CE)
    self.pe_k = 5.
    self.pe_1 = self.pe_k / 0.66
    self.pe_den = np.exp(self.pe_k) - 1
    self.ce_gamma = 0.45
    self.ce_Af = 0.25
    self.ce_fmlen = 1.4

    # pre-define attributes:
    self.register_buffer('musculotendon_slack_len', None)
    self.register_buffer('k_pe', None)
    self.s_as = 0.001
    self.f_iso_n_den = .66 ** 2
    self.k_se = 1 / (0.04 ** 2)
    self.q_crit = 0.3
    self.b_rel_st_den = 5e-3 - self.q_crit
    self.min_flce = 0.01

    self.to_build_dict = {
      'max_isometric_force': [],
      'tendon_length': [],
      'optimal_muscle_length': [],
      'normalized_slack_muscle_length': [],
      }
    self.to_build_dict_default = {'normalized_slack_muscle_length': 1.4}

    self.built = False

  def build(
    self,
    timestep,
    max_isometric_force,
    tendon_length,
    optimal_muscle_length,
    normalized_slack_muscle_length,
  ):
    """Build the muscle using arguments from the :class:`motornet.effector.Effector` wrapper object. This
    should be called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle
    structure according to the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a list containing as many
        elements as there are muscles.
      tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a list containing as many
        elements as there are muscles in that parent effector object.
      optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
        the length at which the muscle will output the maximum amount of force given the same excitation. If
        several muscles are declared in the parent effector object, then this should be a list containing as 
        many elements as there are muscles in that parent effector object.
      normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
        muscle(s) will start to developp passive forces. If several muscles are declared in the parent effector
        object, then this should be a list containing as many elements as there are muscles in that parent 
        effector object.
    """
    self.n_muscles = np.array(tendon_length).size
    shape = (1, 1, self.n_muscles)

    self.dt = timestep
    self.register_buffer('max_iso_force', torch.tensor(max_isometric_force, dtype=torch.float32).reshape(shape))
    self.register_buffer('l0_ce', torch.tensor(optimal_muscle_length, dtype=torch.float32).reshape(shape))
    self.register_buffer('l0_se', torch.tensor(tendon_length, dtype=torch.float32).reshape(shape))
    self.register_buffer('l0_pe', torch.tensor(normalized_slack_muscle_length, dtype=torch.float32)*self.l0_ce)
    self.register_buffer('k_pe', 1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2))
    self.register_buffer('musculotendon_slack_len', self.l0_pe + self.l0_se)
    self.register_buffer('vmax', 10 * self.l0_ce)
    self.built = True

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    shape = geometry_state[:, :1, :].shape
    muscle_state = torch.ones(shape, device=self.device) * self.min_activation
    state_derivatives = torch.zeros(shape, device=self.device)
    return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = self.clip_activation(muscle_state[:, :1, :] + state_derivative * dt)

    # musculotendon geometry
    musculotendon_len = geometry_state[:, :1, :]
    muscle_vel = geometry_state[:, 1:2, :]
    muscle_len = torch.clip(musculotendon_len - self.l0_se, min=0.)
    muscle_strain = torch.clip((muscle_len - self.l0_pe) / self.l0_ce, min=0.)
    muscle_len_n = muscle_len / self.l0_ce
    muscle_vel_n = muscle_vel / self.vmax

    # muscle forces
    # flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 3.)
    flpe = self.k_pe * (muscle_strain ** 2)
    flce = torch.clip(1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, min=self.min_flce)

    a_rel_st = torch.where(
      condition=muscle_len_n > 1.,
      input=.41 * flce,
      other=.41
      )
    b_rel_st = torch.where(
      condition=activation < self.q_crit,
      input=5.2 * (1 - .9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
      other=5.2
      )
    dfdvcon0 = activation * (flce + a_rel_st) / b_rel_st  # inv of slope at isometric point wrt concentric curve

    f_x_a = flce * activation  # to speed up computation

    tmp_p_nom = f_x_a * .5
    tmp_p_den = self.s_as - dfdvcon0 * 2.

    p1 = - tmp_p_nom / tmp_p_den
    p2 = (tmp_p_nom ** 2) / tmp_p_den
    p3 = - 1.5 * f_x_a

    nom = torch.where(
      condition=muscle_vel_n < 0,
      input=muscle_vel_n * activation * a_rel_st + f_x_a * b_rel_st,
      other=-p1 * p3 + p1 * self.s_as * muscle_vel_n + p2 - p3 * muscle_vel_n + self.s_as * muscle_vel_n ** 2
      )
    den = torch.where(
      condition=muscle_vel_n < 0,
      input=b_rel_st - muscle_vel_n,
      other=p1 + muscle_vel_n
      )

    active_force = torch.clip(nom / den, min=0.)
    force = (active_force + flpe) * self.max_iso_force
    return torch.cat([activation, muscle_len, muscle_vel, flpe, flce, active_force, force], dim=1)


class RigidTendonHillMuscleThelen(Muscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`, adjusted to
  behave as a rigid tendon version of the original model.

  References:
    [1] `Thelen DG. Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older
    adults. J Biomech Eng. 2003 Feb;125(1):70-7. doi: 10.1115/1.1531112. PMID: 12661198.`

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, min_activation=0.001, **kwargs):
    super().__init__(min_activation=min_activation, **kwargs)
    self.__name__ = 'RigidTendonHillMuscleThelen'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length CE',
      'force-velocity CE',
      'force'
      ]
    self.state_dim = len(self.state_name)

    # parameters for the passive element (PE) and contractile element (CE)
    self.register_buffer('pe_k', torch.tensor(5., dtype=torch.float32))
    self.register_buffer('pe_1', self.pe_k / 0.6)  # divided by epsilon_0^M in Thelen (2003) eq. 3
    self.register_buffer('pe_den', torch.exp(self.pe_k) - 1)
    self.register_buffer('ce_gamma', torch.tensor(0.45, dtype=torch.float32))
    self.register_buffer('ce_Af', torch.tensor(0.25, dtype=torch.float32))
    self.register_buffer('ce_fmlen', torch.tensor(1.4, dtype=torch.float32))

    # pre-define attributes:
    self.register_buffer('musculotendon_slack_len', None)
    self.register_buffer('ce_0', None)
    self.register_buffer('ce_1', None)
    self.register_buffer('ce_2', None)
    self.register_buffer('ce_3', None)
    self.register_buffer('ce_4', None)
    self.register_buffer('ce_5', None)

    self.to_build_dict = {
      'max_isometric_force': [],
      'tendon_length': [],
      'optimal_muscle_length': [],
      'normalized_slack_muscle_length': []
      }
    self.to_build_dict_default = {'normalized_slack_muscle_length': 1.}
    self.built = False

  def build(
    self,
    timestep,
    max_isometric_force,
    tendon_length,
    optimal_muscle_length,
    normalized_slack_muscle_length
  ):
    """Build the muscle using arguments from the :class:`motornet.effector.Effector` wrapper object. This
    should be called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle 
    structure according to the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a list containing as many
        elements as there are muscles.
      tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a list containing as many
        elements as there are muscles in that parent effector object.
      optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
        the length at which the muscle will output the maximum amount of force given the same excitation. If
        several muscles are declared in the parent effector object, then this should be a list containing as 
        many elements as there are muscles in that parent effector object.
      normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
        muscle(s) will start to developp passive forces. If several muscles are declared in the parent effector
        object, then this should be a list containing as many elements as there are muscles in that parent 
        effector object.
    """
    self.n_muscles = np.array(tendon_length).size
    self.dt = timestep

    max_isometric_force = torch.tensor(max_isometric_force, dtype=torch.float32).reshape(1, 1, self.n_muscles)
    optimal_muscle_length = torch.tensor(optimal_muscle_length, dtype=torch.float32).reshape(1, 1, self.n_muscles)
    tendon_length = torch.tensor(tendon_length, dtype=torch.float32).reshape(1, 1, self.n_muscles)
    normalized_slack_muscle_length = torch.tensor(normalized_slack_muscle_length, dtype=torch.float32)

    self.register_buffer('max_iso_force', max_isometric_force)
    self.register_buffer('l0_ce', optimal_muscle_length)
    self.register_buffer('l0_se', tendon_length)
    self.register_buffer('l0_pe', self.l0_ce * normalized_slack_muscle_length)
    self.register_buffer('musculotendon_slack_len', self.l0_pe + self.l0_se)
    self.register_buffer('vmax', 10 * self.l0_ce)

    # pre-computed for speed
    self.register_buffer('ce_0', 3 * self.vmax)
    self.register_buffer('ce_1', self.ce_Af * self.vmax)
    self.register_buffer('ce_2', 3 * self.ce_Af * self.vmax * self.ce_fmlen - 3. * self.ce_Af * self.vmax)
    self.register_buffer('ce_3', 8 * self.ce_Af * self.ce_fmlen + 8. * self.ce_fmlen)
    self.register_buffer('ce_4', self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1)
    self.register_buffer('ce_5', 8 * (self.ce_Af + 1.))

    self.built = True

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    shape = geometry_state[:, :1, :].shape
    muscle_state = torch.ones(shape, device=self.device) * self.min_activation
    state_derivatives = torch.zeros(shape, device=self.device)
    return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = muscle_state[:, :1, :] + state_derivative * dt
    activation = self.clip_activation(activation)

    # musculotendon geometry
    musculotendon_len = geometry_state[:, :1, :]
    muscle_len = torch.clip(musculotendon_len - self.l0_se, min=0.001)
    muscle_vel = geometry_state[:, 1:2, :]

    # muscle forces
    a3 = activation * 3.
    condition = muscle_vel <= 0
    nom = torch.where(
      condition,
      input=self.ce_Af * (activation * self.ce_0 + 4. * muscle_vel + self.vmax),
      other=self.ce_2 * activation + self.ce_3 * muscle_vel + self.ce_4
      )
    den = torch.where(
      condition,
      input=a3 * self.ce_1 + self.ce_1 - 4. * muscle_vel,
      other=self.ce_4 * a3 + self.ce_5 * muscle_vel + self.ce_4
      )
    fvce = torch.clip(nom / den, min=0.)
    flpe = torch.clip((torch.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1) / self.pe_den, min=0.)
    flce = torch.exp((- ((muscle_len / self.l0_ce) - 1) ** 2) / self.ce_gamma)
    force = (activation * flce * fvce + flpe) * self.max_iso_force
    return torch.cat([activation, muscle_len, muscle_vel, flpe, flce, fvce, force], dim=1)


class CompliantTendonHillMuscle(RigidTendonHillMuscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`. Unlike its
  parent class, this class implements a full compliant tendon version of the model, as formulated in the
  reference article.

  References:
    [1] `Kistemaker DA, Wong JD, Gribble PL. The central nervous system does not minimize energy cost in arm
    movements. J Neurophysiol. 2010 Dec;104(6):2985-94. doi: 10.1152/jn.00483.2010. Epub 2010 Sep 8. PMID:
    20884757.`

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, min_activation=0.01, **kwargs):
    super().__init__(min_activation=min_activation, **kwargs)
    self.__name__ = 'CompliantTendonHillMuscle'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length SE',
      'active force',
      'force']
    self.state_dim = len(self.state_name)
    self.built = False

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    # Compute musculotendon geometry
    muscle_len = muscle_state[:, 1:2, :]
    muscle_len_n = muscle_len / self.l0_ce
    musculotendon_len = geometry_state[:, :1, :]
    tendon_len = musculotendon_len - muscle_len
    tendon_strain = torch.clip((tendon_len - self.l0_se) / self.l0_se, min=0.)
    muscle_strain = torch.clip((muscle_len - self.l0_pe) / self.l0_ce, min=0.)

    # Compute forces
    flse = torch.clip(self.k_se * (tendon_strain ** 2), max=1.)
    # flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 1.)
    flpe = self.k_pe * (muscle_strain ** 2)
    active_force = torch.clip(flse - flpe, min=0.)

    # Integrate
    d_activation = state_derivative[:, 0:1, :]
    muscle_vel_n = state_derivative[:, 1:2, :]
    activation = muscle_state[:, 0:1, :] + d_activation * dt
    activation = self.clip_activation(activation)
    new_muscle_len = (muscle_len_n + dt * muscle_vel_n) * self.l0_ce

    muscle_vel = muscle_vel_n * self.vmax
    force = flse * self.max_iso_force
    return torch.cat([activation, new_muscle_len, muscle_vel, flpe, flse, active_force, force], dim=1)

  def _ode(self, excitation, muscle_state):
    activation = muscle_state[:, 0:1, :]
    d_activation = self.activation_ode(excitation, activation)
    muscle_len_n = muscle_state[:, 1:2, :] / self.l0_ce
    active_force = muscle_state[:, 5:6, :]
    new_muscle_vel_n = self._normalized_muscle_vel(muscle_len_n, activation, active_force)
    return torch.cat([d_activation, new_muscle_vel_n], dim=1)

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    musculotendon_len = geometry_state[:, 0:1, :]
    activation = torch.ones_like(musculotendon_len, device=self.device) * self.min_activation
    d_activation = torch.zeros_like(musculotendon_len, device=self.device)

    # if musculotendon length is negative, raise an error.
    # if musculotendon length is less than tendon slack length, assign all (most of) the length to the tendon.
    # if musculotendon length is more than tendon slack length and less than musculotendon slack length, assign to
    #  the tendon up to the tendon slack length, and the rest to the muscle lengtorch.
    # if musculotendon length is more than tendon slack length and muscle slack length combined, find the muscle
    #  length that satisfies equilibrium between tendon passive forces and muscle passive forces.
    muscle_len = torch.where(
      condition=torch.less(musculotendon_len, 0.),
      input=torch.tensor(-1., dtype=torch.float32, device=self.device),
      other=torch.where(
        condition=torch.less(musculotendon_len, self.l0_se),
        input=0.001 * self.l0_ce,
        other=torch.where(
          condition=torch.less(musculotendon_len, self.l0_se + self.l0_pe),
          input=musculotendon_len - self.l0_se,
          other=(self.k_pe * self.l0_pe * self.l0_se ** 2 -
            self.k_se * (self.l0_ce ** 2) * musculotendon_len +
            self.k_se * self.l0_ce ** 2 * self.l0_se -
            self.l0_ce * self.l0_se * torch.sqrt(self.k_pe * self.k_se)
            * (-musculotendon_len + self.l0_pe + self.l0_se)) /
           (self.k_pe * self.l0_se ** 2 - self.k_se * self.l0_ce ** 2))))

    # tf.debugging.assert_non_negative(muscle_len, message='initial muscle length was < 0.')
    tendon_len = musculotendon_len - muscle_len
    tendon_strain = torch.clip((tendon_len - self.l0_se) / self.l0_se, min=0.)
    muscle_strain = torch.clip((muscle_len - self.l0_pe) / self.l0_ce, min=0.)

    # Compute forces
    flse = torch.clip(self.k_se * (tendon_strain ** 2), max=1.)
    flpe = torch.clip(self.k_pe * (muscle_strain ** 2), max=1.)
    active_force = torch.clip(flse - flpe, min=0.)

    muscle_vel_n = self._normalized_muscle_vel(muscle_len / self.l0_ce, activation, active_force)
    muscle_state = torch.cat([activation, muscle_len], dim=1)
    state_derivative = torch.cat([d_activation, muscle_vel_n], dim=1)

    return self.integrate(self.dt, state_derivative, muscle_state, geometry_state)

  def _normalized_muscle_vel(self, muscle_len_n, activation, active_force):
    flce = torch.clip(1. + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, min=self.min_flce)
    a_rel_st = torch.where(torch.less(muscle_len_n, 1.), input=.41 * flce, other=.41)
    b_rel_st = torch.where(
      condition=torch.less(activation, self.q_crit),
      input=5.2 * (1 - .9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
      other=5.2)
    # inv of slope at isometric point wrt concentric curve
    f_x_a = flce * activation  # to speed up computation
    dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

    p1 = - f_x_a * .5 / (self.s_as - dfdvcon0 * 2.)
    p3 = - 1.5 * f_x_a
    p2_containing_term = (4 * ((f_x_a * 0.5) ** 2) * (- self.s_as)) / (self.s_as - dfdvcon0 * 2)

    # defensive code to avoid propagation of negative square root in the non-selected tf.where outcome
    # the assertion is to ensure that any selected item is indeed not a negative root.
    sqrt_term = active_force ** 2 + 2 * active_force * p1 * self.s_as + \
      2 * active_force * p3 + p1 ** 2 * self.s_as ** 2 + 2 * p1 * p3 * self.s_as +\
      p2_containing_term + p3 ** 2
    #cond = torch.logical_or(torch.less(sqrt_term, 0.), torch.greater_equal(active_force, f_x_a))
    #torch._assert(cond, message='root that should be used is negative.')
    sqrt_term = torch.clip(sqrt_term, min=0.)

    new_muscle_vel_nom = torch.where(
      condition=torch.less(active_force, f_x_a),
      input=b_rel_st * (active_force - f_x_a),
      other=- active_force + p1 * self.s_as - p3 - torch.sqrt(sqrt_term))
    new_muscle_vel_den = torch.where(
      condition=torch.less(active_force, f_x_a),
      input=active_force + activation * a_rel_st,
      other=- 2 * self.s_as)

    return new_muscle_vel_nom / new_muscle_vel_den
