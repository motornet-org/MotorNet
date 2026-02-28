"""
Environment module for motor control tasks.

Provides a Gym-like interface for motor control tasks,
built on top of the effector simulation.

Matches the PyTorch MotorNet API including:
- Proprioceptive feedback (normalized muscle length + velocity)
- Vision feedback (fingertip position)
- Configurable delays
- Observation buffer management
- Noise application
"""

from typing import NamedTuple, Tuple, Optional, Any, Dict, List
import jax
import jax.numpy as jnp

from motornet_jax.types import (
    JointState,
    CartesianState,
    MuscleState,
    GeometryState,
    EffectorState,
)


class ObsBuffer(NamedTuple):
    """Observation buffer for implementing delays.

    Stores past values of proprioception, vision, and actions.
    Oldest values are at index 0.

    Attributes:
        proprioception: List of past proprioceptive observations. Shape per entry: (batch, n_prop)
        vision: List of past visual observations. Shape per entry: (batch, n_vis)
        action: List of past actions. Shape per entry: (batch, n_muscles)
    """
    proprioception: jnp.ndarray  # (delay_steps, batch, n_prop)
    vision: jnp.ndarray          # (delay_steps, batch, n_vis)
    action: jnp.ndarray          # (stack_size, batch, n_muscles)


class EnvState(NamedTuple):
    """Environment state.

    Contains the effector state plus task-specific information.

    Attributes:
        effector: Current effector state.
        goal: Target position(s). Shape: (batch, n_dim)
        obs_buffer: Observation buffer for delays.
        step_count: Current step count.
        elapsed: Elapsed time in seconds.
    """
    effector: EffectorState
    goal: jnp.ndarray
    obs_buffer: ObsBuffer
    step_count: int
    elapsed: float


class EnvParams(NamedTuple):
    """Environment parameters.

    Attributes:
        max_ep_duration: Maximum episode duration in seconds.
        dt: Timestep size.
        proprioception_delay: Delay for proprioceptive feedback (in timesteps).
        vision_delay: Delay for visual feedback (in timesteps).
        action_frame_stacking: Number of past actions to include in observation.
        proprioception_noise: Noise std for proprioception.
        vision_noise: Noise std for vision.
        action_noise: Noise std for actions.
        obs_noise: Noise std for final observation.
        l0_ce: Optimal contractile element length per muscle. Shape: (n_muscles,)
        vmax: Maximum velocity per muscle. Shape: (n_muscles,)
    """
    max_ep_duration: float = 1.0
    dt: float = 0.01
    proprioception_delay: int = 1
    vision_delay: int = 1
    action_frame_stacking: int = 0
    proprioception_noise: float = 0.0
    vision_noise: float = 0.0
    action_noise: float = 0.0
    obs_noise: float = 0.0
    l0_ce: jnp.ndarray = None  # For muscle length normalization
    vmax: jnp.ndarray = None   # For muscle velocity normalization


class Environment:
    """Motor control environment matching PyTorch MotorNet API.

    Provides observations in the format:
    [goal, vision (delayed), proprioception (delayed), action_history]

    Where:
    - goal: Target position (n_dim)
    - vision: Fingertip position with delay (n_dim)
    - proprioception: Normalized muscle length + velocity with delay (2 * n_muscles)
    - action_history: Past actions if action_frame_stacking > 0
    """

    def __init__(
        self,
        effector,
        max_ep_duration: float = 1.0,
        proprioception_delay: float = None,
        vision_delay: float = None,
        action_frame_stacking: int = 0,
        proprioception_noise: float = 0.0,
        vision_noise: float = 0.0,
        action_noise: float = 0.0,
        obs_noise: float = 0.0,
    ):
        """Initialize environment.

        Args:
            effector: Effector instance (e.g., Arm26).
            max_ep_duration: Maximum episode duration (seconds).
            proprioception_delay: Proprioceptive feedback delay (seconds). Default: dt (1 timestep).
            vision_delay: Visual feedback delay (seconds). Default: dt (1 timestep).
            action_frame_stacking: Number of past actions to include in observation.
            proprioception_noise: Std of noise added to proprioception.
            vision_noise: Std of noise added to vision.
            action_noise: Std of noise added to actions.
            obs_noise: Std of noise added to final observation.
        """
        self.effector = effector
        self.dt = effector.params.dt
        self.n_muscles = effector.n_muscles
        self.n_dim = effector.n_dim

        # Handle delays (default is 1 timestep)
        proprioception_delay = self.dt if proprioception_delay is None else proprioception_delay
        vision_delay = self.dt if vision_delay is None else vision_delay

        # Convert delays to timesteps
        prop_delay_steps = max(1, int(proprioception_delay / self.dt))
        vis_delay_steps = max(1, int(vision_delay / self.dt))

        # Get muscle normalization parameters
        # For Arm26, use optimal_fiber_length as l0_ce and compute vmax
        l0_ce = getattr(effector.params, 'optimal_fiber_length', jnp.ones(self.n_muscles))
        vmax = 10.0 * l0_ce  # Standard vmax = 10 * l0

        self.params = EnvParams(
            max_ep_duration=max_ep_duration,
            dt=self.dt,
            proprioception_delay=prop_delay_steps,
            vision_delay=vis_delay_steps,
            action_frame_stacking=action_frame_stacking,
            proprioception_noise=proprioception_noise,
            vision_noise=vision_noise,
            action_noise=action_noise,
            obs_noise=obs_noise,
            l0_ce=l0_ce,
            vmax=vmax,
        )

        # Compute observation dimension
        self._obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension."""
        # goal: n_dim
        # vision: n_dim
        # proprioception: 2 * n_muscles (length + velocity)
        # action_history: action_frame_stacking * n_muscles
        return (
            self.n_dim +  # goal
            self.n_dim +  # vision (fingertip)
            2 * self.n_muscles +  # proprioception (muscle length + velocity)
            self.params.action_frame_stacking * self.n_muscles  # action history
        )

    @property
    def observation_dim(self) -> int:
        """Dimension of observation vector."""
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        """Dimension of action vector."""
        return self.n_muscles

    def get_params(self) -> EnvParams:
        """Get environment parameters."""
        return self.params

    @staticmethod
    def get_proprioception(
        effector_state: EffectorState,
        params: EnvParams,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Get proprioceptive feedback (normalized muscle length + velocity).

        This matches PyTorch MotorNet's get_proprioception() method.

        Args:
            effector_state: Current effector state.
            params: Environment parameters.
            key: Random key for noise.

        Returns:
            Proprioceptive feedback. Shape: (batch, 2 * n_muscles)
        """
        # Normalized muscle length: fiber_length / l0_ce
        muscle_length_norm = effector_state.muscle.fiber_length / params.l0_ce

        # Normalized muscle velocity: fiber_velocity / vmax
        muscle_velocity_norm = effector_state.muscle.fiber_velocity / params.vmax

        # Concatenate length and velocity
        prop = jnp.concatenate([muscle_length_norm, muscle_velocity_norm], axis=-1)

        # Add noise if specified (check outside of traced function)
        if key is not None and float(params.proprioception_noise) > 0:
            noise = jax.random.normal(key, prop.shape) * params.proprioception_noise
            prop = prop + noise

        return prop

    @staticmethod
    def get_vision(
        effector_state: EffectorState,
        params: EnvParams,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Get visual feedback (fingertip position).

        This matches PyTorch MotorNet's get_vision() method.

        Args:
            effector_state: Current effector state.
            params: Environment parameters.
            key: Random key for noise.

        Returns:
            Visual feedback. Shape: (batch, n_dim)
        """
        vision = effector_state.fingertip

        # Add noise if specified (check outside of traced function)
        if key is not None and float(params.vision_noise) > 0:
            noise = jax.random.normal(key, vision.shape) * params.vision_noise
            vision = vision + noise

        return vision

    @staticmethod
    def update_obs_buffer(
        obs_buffer: ObsBuffer,
        new_proprioception: jnp.ndarray,
        new_vision: jnp.ndarray,
        new_action: jnp.ndarray,
    ) -> ObsBuffer:
        """Update observation buffer with new values.

        Shifts old values and adds new ones.

        Args:
            obs_buffer: Current observation buffer.
            new_proprioception: New proprioceptive observation.
            new_vision: New visual observation.
            new_action: New action.

        Returns:
            Updated observation buffer.
        """
        # Shift buffer and append new value (more efficient than roll+set)
        new_prop = jnp.concatenate([obs_buffer.proprioception[1:], new_proprioception[None]], axis=0)
        new_vis = jnp.concatenate([obs_buffer.vision[1:], new_vision[None]], axis=0)
        new_act = jnp.concatenate([obs_buffer.action[1:], new_action[None]], axis=0)

        return ObsBuffer(
            proprioception=new_prop,
            vision=new_vis,
            action=new_act,
        )

    @staticmethod
    def get_obs(
        env_state: EnvState,
        params: EnvParams,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Get observation from environment state.

        Observation format: [goal, vision (delayed), proprioception (delayed), action_history]

        This matches PyTorch MotorNet's get_obs() method.

        Args:
            env_state: Current environment state.
            params: Environment parameters.
            key: Random key for observation noise.

        Returns:
            Observation vector. Shape: (batch, obs_dim)
        """
        # Get delayed proprioception (oldest = most delayed)
        delayed_prop = env_state.obs_buffer.proprioception[0]  # (batch, 2*n_muscles)

        # Get delayed vision (oldest = most delayed)
        delayed_vision = env_state.obs_buffer.vision[0]  # (batch, n_dim)

        # Build observation: goal, vision, proprioception, [action_history]
        obs_parts = [
            env_state.goal,       # (batch, n_dim)
            delayed_vision,       # (batch, n_dim)
            delayed_prop,         # (batch, 2*n_muscles)
        ]

        # Add action history if configured
        if int(params.action_frame_stacking) > 0:
            # Flatten action history: (stack_size, batch, n_muscles) -> (batch, stack_size * n_muscles)
            action_history = jnp.transpose(env_state.obs_buffer.action, (1, 0, 2))
            action_history = action_history.reshape(action_history.shape[0], -1)
            obs_parts.append(action_history)

        obs = jnp.concatenate(obs_parts, axis=-1)

        # Add observation noise
        if key is not None and float(params.obs_noise) > 0:
            # Don't add noise to goal (first n_dim elements)
            noise = jax.random.normal(key, obs.shape) * params.obs_noise
            noise = noise.at[:, :params.l0_ce.shape[0]].set(0)  # No noise on goal
            obs = obs + noise

        return obs

    def reset(
        self,
        key: jax.random.PRNGKey,
        batch_size: int = 1,
        joint_state: Optional[JointState] = None,
        goal: Optional[jnp.ndarray] = None,
    ) -> Tuple[EnvState, jnp.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            key: Random key for initialization.
            batch_size: Batch size.
            joint_state: Optional initial joint state.
            goal: Optional goal position. Shape: (batch, n_dim)

        Returns:
            env_state: Initial environment state.
            obs: Initial observation.
            info: Info dictionary.
        """
        key, key_effector, key_goal, key_prop, key_vis = jax.random.split(key, 5)

        # Reset effector
        effector_state = self.effector.reset(
            batch_size=batch_size,
            joint_state=joint_state,
            key=key_effector,
        )

        # Generate goal if not provided
        if goal is None:
            # Random goal in reachable workspace
            goal = self._generate_random_goal(key_goal, batch_size)

        # Initialize observation buffer
        # Get initial proprioception and vision
        init_prop = self.get_proprioception(effector_state, self.params, key_prop)
        init_vis = self.get_vision(effector_state, self.params, key_vis)
        init_action = jnp.zeros((batch_size, self.n_muscles))

        # Initialize buffers with repeated values
        prop_buffer = jnp.tile(init_prop[None, :, :], (self.params.proprioception_delay, 1, 1))
        vis_buffer = jnp.tile(init_vis[None, :, :], (self.params.vision_delay, 1, 1))
        action_buffer_size = max(1, self.params.action_frame_stacking)
        action_buffer = jnp.tile(init_action[None, :, :], (action_buffer_size, 1, 1))

        obs_buffer = ObsBuffer(
            proprioception=prop_buffer,
            vision=vis_buffer,
            action=action_buffer,
        )

        env_state = EnvState(
            effector=effector_state,
            goal=goal,
            obs_buffer=obs_buffer,
            step_count=0,
            elapsed=0.0,
        )

        obs = self.get_obs(env_state, self.params)

        info = {
            "states": {
                "joint": effector_state.joint,
                "cartesian": effector_state.cartesian,
                "muscle": effector_state.muscle,
                "geometry": effector_state.geometry,
                "fingertip": effector_state.fingertip,
            },
            "action": init_action,
            "goal": goal,
        }

        return env_state, obs, info

    def _generate_random_goal(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> jnp.ndarray:
        """Generate random goal position.

        Can be overridden by subclasses for different goal generation.
        """
        # Default: random position in approximate workspace
        return jax.random.uniform(
            key,
            (batch_size, self.n_dim),
            minval=jnp.array([0.1, -0.3]),
            maxval=jnp.array([0.5, 0.3]),
        )

    def step(
        self,
        env_state: EnvState,
        action: jnp.ndarray,
        endpoint_load: Optional[jnp.ndarray] = None,
        joint_load: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, bool, bool, Dict]:
        """Take one environment step.

        Returns match PyTorch MotorNet: (obs, reward, terminated, truncated, info)

        Args:
            env_state: Current environment state.
            action: Action to take. Shape: (batch, n_muscles)
            endpoint_load: External endpoint load. Shape: (batch, n_dim)
            joint_load: External joint load. Shape: (batch, n_joints)
            key: Random key for noise.

        Returns:
            new_state: New environment state.
            obs: New observation.
            reward: Reward (None for differentiable training).
            terminated: Whether episode is done.
            truncated: Whether episode was truncated.
            info: Additional information.
        """
        batch_size = action.shape[0]

        # Default loads
        if endpoint_load is None:
            endpoint_load = jnp.zeros((batch_size, self.n_dim))
        if joint_load is None:
            joint_load = jnp.zeros((batch_size, self.effector.n_joints))

        # Add action noise
        noisy_action = action
        if key is not None and self.params.action_noise > 0:
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, action.shape) * self.params.action_noise
            noisy_action = jnp.clip(action + noise, 0.0, 1.0)

        # Step effector
        new_effector = self.effector.__class__.step(
            env_state.effector,
            noisy_action,
            endpoint_load,
            joint_load,
            self.effector.params,
        )

        # Get new proprioception and vision
        if key is not None:
            key, prop_key, vis_key, obs_key = jax.random.split(key, 4)
        else:
            prop_key, vis_key, obs_key = None, None, None

        new_prop = self.get_proprioception(new_effector, self.params, prop_key)
        new_vis = self.get_vision(new_effector, self.params, vis_key)

        # Update observation buffer
        new_obs_buffer = self.update_obs_buffer(
            env_state.obs_buffer,
            new_prop,
            new_vis,
            noisy_action,
        )

        # Update elapsed time
        new_elapsed = env_state.elapsed + self.dt

        # Update state
        new_state = EnvState(
            effector=new_effector,
            goal=env_state.goal,
            obs_buffer=new_obs_buffer,
            step_count=env_state.step_count + 1,
            elapsed=new_elapsed,
        )

        # Get observation
        obs = self.get_obs(new_state, self.params, obs_key)

        # Reward (None for differentiable)
        reward = None

        # Check termination
        terminated = new_elapsed >= self.params.max_ep_duration
        truncated = False

        info = {
            "states": {
                "joint": new_effector.joint,
                "cartesian": new_effector.cartesian,
                "muscle": new_effector.muscle,
                "geometry": new_effector.geometry,
                "fingertip": new_effector.fingertip,
            },
            "action": action,
            "noisy_action": noisy_action,
            "goal": env_state.goal,
        }

        return new_state, obs, reward, terminated, truncated, info

    def step_training(
        self,
        env_state: EnvState,
        action: jnp.ndarray,
        endpoint_load: Optional[jnp.ndarray] = None,
        joint_load: Optional[jnp.ndarray] = None,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, bool, bool, Dict]:
        """Lightweight step for training loops (no noise, minimal info).

        Returns the same tuple signature as step() but with a minimal info
        dict containing only fingertip position. This reduces the amount of
        data carried through lax.scan outputs during training.

        Args:
            env_state: Current environment state.
            action: Action to take. Shape: (batch, n_muscles)
            endpoint_load: External endpoint load. Shape: (batch, n_dim)
            joint_load: External joint load. Shape: (batch, n_joints)

        Returns:
            new_state, obs, reward, terminated, truncated, info
        """
        batch_size = action.shape[0]

        if endpoint_load is None:
            endpoint_load = jnp.zeros((batch_size, self.n_dim))
        if joint_load is None:
            joint_load = jnp.zeros((batch_size, self.effector.n_joints))

        # Step effector (no action noise for training)
        new_effector = self.effector.__class__.step(
            env_state.effector,
            action,
            endpoint_load,
            joint_load,
            self.effector.params,
        )

        # Get new proprioception and vision (no noise)
        new_prop = self.get_proprioception(new_effector, self.params)
        new_vis = self.get_vision(new_effector, self.params)

        # Update observation buffer
        new_obs_buffer = self.update_obs_buffer(
            env_state.obs_buffer,
            new_prop,
            new_vis,
            action,
        )

        new_elapsed = env_state.elapsed + self.dt

        new_state = EnvState(
            effector=new_effector,
            goal=env_state.goal,
            obs_buffer=new_obs_buffer,
            step_count=env_state.step_count + 1,
            elapsed=new_elapsed,
        )

        obs = self.get_obs(new_state, self.params)

        reward = None
        terminated = new_elapsed >= self.params.max_ep_duration
        truncated = False

        # Minimal info dict for training - only what's needed for loss
        info = {
            "states": {
                "fingertip": new_effector.fingertip,
            },
        }

        return new_state, obs, reward, terminated, truncated, info


class RandomTargetReach(Environment):
    """Random target reaching task.

    Reaches from random starting position to random target.
    Matches PyTorch MotorNet RandomTargetReach.
    """

    def __init__(self, effector, **kwargs):
        super().__init__(effector, **kwargs)

    def _generate_random_goal(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> jnp.ndarray:
        """Generate random goal by sampling random joint state and converting to cartesian."""
        # Sample random joint state
        key, joint_key = jax.random.split(key)
        random_joint_pos = jax.random.uniform(
            joint_key,
            (batch_size, self.effector.n_joints),
            minval=self.effector.params.skeleton.pos_lower_bound,
            maxval=self.effector.params.skeleton.pos_upper_bound,
        )
        random_joint_state = JointState(
            position=random_joint_pos,
            velocity=jnp.zeros((batch_size, self.effector.n_joints)),
        )

        # Convert to cartesian to get fingertip position as goal
        from motornet_jax.skeleton import TwoDofArm
        cartesian = TwoDofArm.joint2cartesian(random_joint_state, self.effector.params.skeleton)
        return cartesian.position  # Fingertip position as goal


class CenterOutReach(Environment):
    """Center-out reaching task.

    Reaches from center position to targets arranged in a circle.
    """

    def __init__(
        self,
        effector,
        center_joint: Optional[jnp.ndarray] = None,
        n_targets: int = 8,
        target_radius: float = 0.1,
        **kwargs
    ):
        """Initialize center-out reaching task.

        Args:
            effector: Effector instance.
            center_joint: Center joint position. Default: [0.8, 1.2] rad.
            n_targets: Number of targets around circle.
            target_radius: Radius of target circle from center.
            **kwargs: Additional environment arguments.
        """
        super().__init__(effector, **kwargs)

        if center_joint is None:
            center_joint = jnp.array([0.8, 1.2])
        self.center_joint = center_joint
        self.n_targets = n_targets
        self.target_radius = target_radius

        # Compute center position in cartesian
        from motornet_jax.skeleton import TwoDofArm
        center_state = JointState(
            position=center_joint[None, :],
            velocity=jnp.zeros((1, 2)),
        )
        center_cartesian = TwoDofArm.joint2cartesian(center_state, self.effector.params.skeleton)
        self.center_pos = center_cartesian.position[0]

    def _generate_random_goal(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> jnp.ndarray:
        """Generate random target on circle around center."""
        # Random target indices
        target_indices = jax.random.randint(key, (batch_size,), 0, self.n_targets)
        angles = target_indices * (2 * jnp.pi / self.n_targets)

        # Target positions
        targets = self.center_pos + self.target_radius * jnp.stack([
            jnp.cos(angles),
            jnp.sin(angles),
        ], axis=1)

        return targets

    def reset(
        self,
        key: jax.random.PRNGKey,
        batch_size: int = 1,
        joint_state: Optional[JointState] = None,
        goal: Optional[jnp.ndarray] = None,
    ) -> Tuple[EnvState, jnp.ndarray, Dict]:
        """Reset to center position."""
        # Always start from center
        if joint_state is None:
            joint_state = JointState(
                position=jnp.tile(self.center_joint[None, :], (batch_size, 1)),
                velocity=jnp.zeros((batch_size, 2)),
            )
        return super().reset(key, batch_size, joint_state, goal)


class TrackingEnv(Environment):
    """Target tracking task.

    The agent must track a moving target.
    """

    def __init__(self, effector, target_speed: float = 0.1, **kwargs):
        super().__init__(effector, **kwargs)
        self.target_speed = target_speed

    def step(self, env_state, action, endpoint_load=None, joint_load=None, key=None):
        """Step with moving target."""
        # Move target
        if key is not None:
            key, target_key, step_key = jax.random.split(key, 3)
            target_vel = jax.random.normal(target_key, env_state.goal.shape) * self.target_speed
            new_goal = env_state.goal + target_vel * self.dt
            # Clip to workspace
            new_goal = jnp.clip(
                new_goal,
                jnp.array([0.1, -0.3]),
                jnp.array([0.5, 0.3]),
            )
        else:
            new_goal = env_state.goal
            step_key = None

        # Update goal in state
        env_state = EnvState(
            effector=env_state.effector,
            goal=new_goal,
            obs_buffer=env_state.obs_buffer,
            step_count=env_state.step_count,
            elapsed=env_state.elapsed,
        )

        # Call parent step
        return super().step(env_state, action, endpoint_load, joint_load, step_key)
