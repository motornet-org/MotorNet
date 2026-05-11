"""Tests for motornet.environment — Environment base class and RandomTargetReach."""

import numpy as np
import pytest
import torch

from motornet.effector import ReluPointMass24, RigidTendonArm26
from motornet.environment import Environment, RandomTargetReach
from motornet.muscle import RigidTendonHillMuscleThelen


# =============================================================================
# Environment base class
# =============================================================================

class TestEnvironmentReset:

    def test_reset_returns_obs_and_info(self, base_env):
        result = base_env.reset(options={"deterministic": True})
        assert len(result) == 2
        obs, info = result
        assert obs is not None
        assert isinstance(info, dict)

    def test_obs_is_tensor_in_differentiable_mode(self, base_env):
        obs, _ = base_env.reset(options={"deterministic": True})
        assert torch.is_tensor(obs)

    def test_obs_shape_matches_observation_space(self, base_env):
        obs, _ = base_env.reset(options={"batch_size": 5, "deterministic": True})
        assert obs.shape == (5, base_env.observation_space.shape[0])

    def test_obs_size_for_relu_point_mass_no_stacking(self):
        # goal(2) + vision(2) + prop(4 muscles × 2) = 12
        env = Environment(effector=ReluPointMass24())
        obs, _ = env.reset(options={"deterministic": True})
        assert obs.shape[-1] == 12

    def test_info_contains_required_keys(self, base_env):
        _, info = base_env.reset(options={"deterministic": True})
        for key in ('states', 'action', 'noisy action', 'goal'):
            assert key in info

    def test_info_states_contains_all_keys(self, base_env):
        _, info = base_env.reset(options={"deterministic": True})
        for key in ('joint', 'muscle', 'geometry', 'cartesian', 'fingertip'):
            assert key in info['states']

    def test_elapsed_reset_to_zero(self, base_env):
        base_env.reset()
        base_env.step(torch.zeros(1, base_env.effector.n_muscles))
        base_env.reset()
        assert base_env.elapsed == 0.0

    def test_goal_shape_after_reset(self, base_env):
        obs, info = base_env.reset(options={"batch_size": 4, "deterministic": True})
        goal = info["goal"]
        assert goal.shape == (4, base_env.skeleton.space_dim)

    def test_seeded_reset_is_reproducible(self, base_env):
        obs_a, _ = base_env.reset(seed=42, options={"deterministic": True})
        obs_b, _ = base_env.reset(seed=42, options={"deterministic": True})
        assert torch.allclose(obs_a, obs_b)

    def test_different_seeds_give_different_states(self, base_env):
        base_env.reset(seed=0)
        state_0 = base_env.states["joint"].clone()
        base_env.reset(seed=99)
        state_99 = base_env.states["joint"].clone()
        assert not torch.allclose(state_0, state_99)

    def test_obs_buffer_initialized_at_reset(self, base_env):
        base_env.reset(options={"deterministic": True})
        assert all(v is not None for v in base_env.obs_buffer["proprioception"])
        assert all(v is not None for v in base_env.obs_buffer["vision"])


class TestEnvironmentStep:

    def test_step_returns_five_tuple(self, base_env):
        base_env.reset(options={"deterministic": True})
        action = torch.zeros(1, base_env.effector.n_muscles)
        result = base_env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self, base_env):
        base_env.reset(options={"batch_size": 3, "deterministic": True})
        action = torch.zeros(3, base_env.effector.n_muscles)
        obs, reward, terminated, truncated, info = base_env.step(action)
        assert obs.shape == (3, base_env.observation_space.shape[0])

    def test_step_terminated_false_before_max_duration(self, base_env):
        base_env.reset(options={"deterministic": True})
        action = torch.zeros(1, base_env.effector.n_muscles)
        _, _, terminated, _, _ = base_env.step(action)
        assert not terminated  # one step is far from max duration

    def test_step_terminated_true_at_max_duration(self):
        env = Environment(effector=ReluPointMass24(), max_ep_duration=0.02)
        env.reset(options={"deterministic": True})
        action = torch.zeros(1, env.effector.n_muscles)
        n_steps = int(0.02 / env.dt)
        terminated = False
        for _ in range(n_steps):
            _, _, terminated, _, _ = env.step(action)
        assert terminated

    def test_elapsed_increments_each_step(self, base_env):
        base_env.reset(options={"deterministic": True})
        action = torch.zeros(1, base_env.effector.n_muscles)
        base_env.step(action)
        assert base_env.elapsed == pytest.approx(base_env.dt)
        base_env.step(action)
        assert base_env.elapsed == pytest.approx(2 * base_env.dt)

    def test_step_info_contains_states(self, base_env):
        base_env.reset(options={"deterministic": True})
        action = torch.zeros(1, base_env.effector.n_muscles)
        _, _, _, _, info = base_env.step(action)
        assert 'states' in info

    def test_step_no_nan_over_many_steps(self, base_env):
        base_env.reset(seed=0, options={"deterministic": True})
        action = torch.ones(1, base_env.effector.n_muscles) * 0.5
        for _ in range(100):
            obs, _, _, _, _ = base_env.step(action, deterministic=True)
        assert not torch.isnan(obs).any()

    def test_differentiable_mode_returns_tensor(self, base_env):
        assert base_env.differentiable is True
        base_env.reset(options={"deterministic": True})
        action = torch.zeros(1, base_env.effector.n_muscles)
        obs, _, _, _, _ = base_env.step(action)
        assert torch.is_tensor(obs)

    def test_non_differentiable_mode_returns_numpy(self):
        env = Environment(effector=ReluPointMass24(), differentiable=False)
        env.reset(options={"deterministic": True})
        action = np.zeros((1, env.effector.n_muscles), dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        assert isinstance(obs, np.ndarray)


class TestObsBuffer:

    def test_proprioception_buffer_length_matches_delay(self):
        dt = 0.01
        delay = 0.05  # 5 steps
        env = Environment(effector=ReluPointMass24(), proprioception_delay=delay)
        assert len(env.obs_buffer["proprioception"]) == int(delay / dt)

    def test_vision_buffer_length_matches_delay(self):
        dt = 0.01
        delay = 0.09  # 9 steps
        env = Environment(effector=ReluPointMass24(), vision_delay=delay)
        assert len(env.obs_buffer["vision"]) == int(delay / dt)

    def test_default_proprioception_buffer_length_is_one(self):
        env = Environment(effector=ReluPointMass24())
        assert len(env.obs_buffer["proprioception"]) == 1

    def test_default_vision_buffer_length_is_one(self):
        env = Environment(effector=ReluPointMass24())
        assert len(env.obs_buffer["vision"]) == 1

    def test_action_frame_stacking_expands_obs(self):
        env_no_stack = Environment(effector=ReluPointMass24(), action_frame_stacking=0)
        env_stacked = Environment(effector=ReluPointMass24(), action_frame_stacking=3)
        obs_no, _ = env_no_stack.reset(options={"deterministic": True})
        obs_st, _ = env_stacked.reset(options={"deterministic": True})
        # stacked adds n_muscles × stacking extra features
        n_muscles = env_stacked.effector.n_muscles
        assert obs_st.shape[-1] == obs_no.shape[-1] + 3 * n_muscles

    def test_action_buffer_length_matches_stacking(self):
        env = Environment(effector=ReluPointMass24(), action_frame_stacking=4)
        assert len(env.obs_buffer["action"]) == 4

    def test_proprioception_delay_shifts_obs(self):
        # With large delay, proprioception in obs should still reflect initial state even after steps
        env = Environment(effector=ReluPointMass24(), proprioception_delay=0.05)
        env.reset(seed=0, options={"deterministic": True})
        prop_at_reset = env.obs_buffer["proprioception"][0].clone()

        action = torch.ones(1, env.effector.n_muscles) * 0.8
        # After 3 steps, proprioception in obs[0] should still be the initial value
        # (because the buffer is 5 steps deep)
        for _ in range(3):
            env.step(action, deterministic=True)

        prop_in_obs = env.obs_buffer["proprioception"][0]
        assert torch.allclose(prop_in_obs, prop_at_reset)


class TestObsNoise:

    def test_deterministic_true_disables_obs_noise(self):
        env = Environment(effector=ReluPointMass24(), obs_noise=1.0)
        obs_a, _ = env.reset(seed=42, options={"deterministic": True})
        obs_b, _ = env.reset(seed=42, options={"deterministic": True})
        assert torch.allclose(obs_a, obs_b)

    def test_obs_noise_produces_different_obs_than_deterministic(self):
        env = Environment(effector=ReluPointMass24(), obs_noise=1.0)
        obs_det, _ = env.reset(seed=0, options={"deterministic": True})
        obs_noisy, _ = env.reset(seed=0, options={"deterministic": False})
        # With std=1.0 noise, the two observations should differ
        assert not torch.allclose(obs_det, obs_noisy)

    def test_action_space_matches_n_muscles(self, base_env):
        assert base_env.action_space.shape[0] == base_env.effector.n_muscles

    def test_action_space_bounds(self, base_env):
        assert base_env.action_space.low.min() == pytest.approx(0.0)
        assert base_env.action_space.high.max() == pytest.approx(1.0)

    def test_apply_noise_adds_noise(self, base_env):
        base_env.reset(seed=0)
        loc = torch.zeros(1, 4)
        noisy = base_env.apply_noise(loc, noise=[1.0, 1.0, 1.0, 1.0])
        assert not torch.allclose(noisy, loc)

    def test_apply_noise_zero_std_no_effect(self, base_env):
        base_env.reset(seed=0)
        loc = torch.ones(1, 4) * 3.14
        noisy = base_env.apply_noise(loc, noise=[0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(noisy, loc)


class TestEnvironmentProperties:

    def test_muscle_shortcut(self, base_env):
        assert base_env.muscle is base_env.effector.muscle

    def test_skeleton_shortcut(self, base_env):
        assert base_env.skeleton is base_env.effector.skeleton

    def test_n_muscles_shortcut(self, base_env):
        assert base_env.n_muscles == base_env.effector.n_muscles

    def test_space_dim_shortcut(self, base_env):
        assert base_env.space_dim == base_env.effector.skeleton.space_dim

    def test_states_shortcut(self, base_env):
        base_env.reset(options={"deterministic": True})
        assert base_env.states is base_env.effector.states

    def test_dt_matches_effector(self, base_env):
        assert base_env.dt == base_env.effector.dt

    def test_get_vision_returns_fingertip_shape(self, base_env):
        base_env.reset(options={"batch_size": 3, "deterministic": True})
        vis = base_env.get_vision()
        assert vis.shape == (3, base_env.skeleton.space_dim)

    def test_get_proprioception_returns_muscle_features(self, base_env):
        base_env.reset(options={"batch_size": 3, "deterministic": True})
        prop = base_env.get_proprioception()
        # prop = [normalized muscle length, normalized muscle vel] for each muscle
        assert prop.shape == (3, 2 * base_env.n_muscles)

    def test_get_save_config_returns_dict(self, base_env):
        cfg = base_env.get_save_config()
        assert isinstance(cfg, dict)
        assert 'name' in cfg
        assert 'effector' in cfg

    def test_get_obs_size_matches_observation_space(self, base_env):
        assert base_env._get_obs_size() == base_env.observation_space.shape[0]

    def test_get_obs_size_formula_relu_point_mass(self):
        # goal(2) + vision(2) + prop(4×2=8) + stacking(0) = 12
        env = Environment(effector=ReluPointMass24())
        assert env._get_obs_size() == 12

    def test_get_obs_size_with_action_stacking(self):
        env = Environment(effector=ReluPointMass24(), action_frame_stacking=3)
        # 12 base + 4 muscles × 3 stacking = 24
        assert env._get_obs_size() == 24

    def test_obs_size_matches_actual_obs_after_reset(self):
        env = Environment(effector=ReluPointMass24())
        obs, _ = env.reset(options={"deterministic": True})
        assert obs.shape[-1] == env._get_obs_size()

    def test_init_does_not_call_reset(self):
        """States should be None immediately after init — reset is no longer called in _build_spaces."""
        env = Environment(effector=ReluPointMass24())
        assert env.states["joint"] is None
        assert env.goal is None
        assert env.elapsed is None


# =============================================================================
# __init_subclass__ warning
# =============================================================================

class TestSubclassWarning:

    def test_warns_when_get_obs_overridden_without_get_obs_size(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class BadEnv(Environment):
                def get_obs(self, action=None, deterministic=False):
                    return super().get_obs(action, deterministic)

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "_get_obs_size" in str(w[0].message)

    def test_no_warning_when_both_overridden(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class GoodEnv(Environment):
                def get_obs(self, action=None, deterministic=False):
                    return super().get_obs(action, deterministic)

                def _get_obs_size(self):
                    return super()._get_obs_size()

        assert len(w) == 0

    def test_no_warning_when_neither_overridden(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class PlainEnv(Environment):
                pass

        assert len(w) == 0

    def test_no_warning_for_get_obs_size_only_override(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class SizeOnlyEnv(Environment):
                def _get_obs_size(self):
                    return 42

        assert len(w) == 0

    def test_warning_fires_at_class_definition_not_instantiation(self):
        """The warning should fire when the class body is executed, before any instance exists."""
        import warnings
        fired_during_definition = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class EarlyWarnEnv(Environment):
                def get_obs(self, action=None, deterministic=False):
                    return super().get_obs(action, deterministic)

            fired_during_definition.append(len(w))

        assert fired_during_definition[0] == 1  # fired before any instance was created


# =============================================================================
# RandomTargetReach
# =============================================================================

class TestRandomTargetReach:

    @pytest.fixture
    def reach_env(self):
        effector = RigidTendonArm26(muscle=RigidTendonHillMuscleThelen())
        return RandomTargetReach(effector=effector)

    def test_reset_returns_obs_and_info(self, reach_env):
        obs, info = reach_env.reset(options={"deterministic": True})
        assert obs is not None
        assert isinstance(info, dict)

    def test_goal_is_2d_cartesian(self, reach_env):
        _, info = reach_env.reset(options={"deterministic": True})
        goal = info["goal"]
        assert goal.shape[-1] == 2  # x, y position

    def test_goal_shape_matches_batch(self, reach_env):
        obs, info = reach_env.reset(options={"batch_size": 5, "deterministic": True})
        assert info["goal"].shape == (5, 2)

    def test_goal_within_arm_reach(self, reach_env):
        """Target should be reachable — within max arm lengtorch."""
        arm = reach_env.skeleton
        max_reach = arm.L1 + arm.L2
        obs, info = reach_env.reset(
            options={"batch_size": 50, "deterministic": True}
        )
        goal = info["goal"]
        dist = goal.norm(dim=-1)
        assert (dist <= max_reach + 1e-3).all()

    def test_goal_randomized_across_resets(self, reach_env):
        _, info_a = reach_env.reset(seed=0, options={"deterministic": True})
        _, info_b = reach_env.reset(seed=99, options={"deterministic": True})
        # Different seeds should (almost certainly) produce different targets
        assert not torch.allclose(info_a["goal"], info_b["goal"])

    def test_obs_includes_goal_features(self, reach_env):
        # The goal is the first `space_dim` features of the observation
        obs, info = reach_env.reset(options={"deterministic": True})
        goal = info["goal"]
        obs_goal = obs[:, :reach_env.skeleton.space_dim]
        assert torch.allclose(obs_goal, goal, atol=1e-5)

    def test_step_goal_unchanged(self, reach_env):
        """Goal should remain constant during an episode."""
        _, info = reach_env.reset(seed=0, options={"deterministic": True})
        goal_before = info["goal"].clone()
        action = torch.zeros(1, reach_env.effector.n_muscles)
        for _ in range(10):
            _, _, _, _, info = reach_env.step(action, deterministic=True)
        assert torch.allclose(info["goal"], goal_before)

    def test_no_nan_over_episode(self, reach_env):
        reach_env.reset(seed=0, options={"deterministic": True})
        action = torch.ones(1, reach_env.effector.n_muscles) * 0.05
        n_steps = int(reach_env.max_ep_duration / reach_env.dt)
        for _ in range(n_steps):
            obs, _, terminated, _, _ = reach_env.step(action, deterministic=True)
            assert not torch.isnan(obs).any(), "NaN in observation during episode"
            if terminated:
                break


# =============================================================================
# Non-differentiable (RL) mode
# =============================================================================

class TestNonDifferentiable:

    @pytest.fixture
    def rl_env(self):
        """Environment in non-differentiable mode — the standard RL setup."""
        return Environment(effector=ReluPointMass24(), differentiable=False)

    # --- reset ---

    def test_reset_obs_is_numpy(self, rl_env):
        obs, _ = rl_env.reset(options={"deterministic": True})
        assert isinstance(obs, np.ndarray)

    def test_reset_action_in_info_is_numpy(self, rl_env):
        _, info = rl_env.reset(options={"deterministic": True})
        assert isinstance(info["action"], np.ndarray)

    def test_reset_goal_in_info_is_numpy(self, rl_env):
        _, info = rl_env.reset(options={"deterministic": True})
        assert isinstance(info["goal"], np.ndarray)

    def test_reset_states_in_info_are_numpy(self, rl_env):
        _, info = rl_env.reset(options={"deterministic": True})
        for key, val in info["states"].items():
            if val is not None:
                assert isinstance(val, np.ndarray), f"states['{key}'] is not numpy"

    # --- step outputs ---

    def test_step_obs_is_numpy(self, rl_env):
        rl_env.reset(options={"deterministic": True})
        obs, _, _, _, _ = rl_env.step(np.zeros((1, rl_env.n_muscles), dtype=np.float32))
        assert isinstance(obs, np.ndarray)

    def test_reward_is_numpy_zeros(self, rl_env):
        rl_env.reset(options={"deterministic": True})
        _, reward, _, _, _ = rl_env.step(np.zeros((1, rl_env.n_muscles), dtype=np.float32))
        assert isinstance(reward, np.ndarray)
        assert reward.item() == pytest.approx(0.0)

    def test_reward_shape_is_batch_by_one(self, rl_env):
        rl_env.reset(options={"batch_size": 4, "deterministic": True})
        action = np.zeros((4, rl_env.n_muscles), dtype=np.float32)
        _, reward, _, _, _ = rl_env.step(action)
        assert reward.shape == (4, 1)

    def test_reward_is_none_in_differentiable_mode(self, base_env):
        base_env.reset(options={"deterministic": True})
        _, reward, _, _, _ = base_env.step(torch.zeros(1, base_env.n_muscles))
        assert reward is None

    def test_step_states_in_info_are_numpy(self, rl_env):
        rl_env.reset(options={"deterministic": True})
        _, _, _, _, info = rl_env.step(np.zeros((1, rl_env.n_muscles), dtype=np.float32))
        for key, val in info["states"].items():
            if val is not None:
                assert isinstance(val, np.ndarray), f"states['{key}'] is not numpy after step"

    def test_step_goal_in_info_is_numpy(self, rl_env):
        rl_env.reset(options={"deterministic": True})
        _, _, _, _, info = rl_env.step(np.zeros((1, rl_env.n_muscles), dtype=np.float32))
        assert isinstance(info["goal"], np.ndarray)

    # --- action input ---

    def test_numpy_action_accepted_by_step(self, rl_env):
        rl_env.reset(options={"deterministic": True})
        action = np.ones((1, rl_env.n_muscles), dtype=np.float32) * 0.5
        obs, _, _, _, _ = rl_env.step(action)  # must not raise
        assert obs is not None

    def test_tensor_action_also_accepted_in_non_diff_mode(self, rl_env):
        rl_env.reset(options={"deterministic": True})
        action = torch.ones(1, rl_env.n_muscles) * 0.5
        obs, _, _, _, _ = rl_env.step(action)
        assert isinstance(obs, np.ndarray)

    # --- episode viability ---

    def test_full_episode_runs_without_error(self):
        env = Environment(effector=ReluPointMass24(), differentiable=False,
                          max_ep_duration=0.1)
        env.reset(seed=0, options={"deterministic": True})
        action = np.ones((1, env.n_muscles), dtype=np.float32) * 0.3
        terminated = False
        steps = 0
        while not terminated:
            obs, reward, terminated, _, _ = env.step(action, deterministic=True)
            steps += 1
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, np.ndarray)
        assert terminated
        assert steps == pytest.approx(int(0.1 / env.dt), abs=1)

    def test_multi_episode_cycling(self):
        """Reset after termination must produce a valid next episode."""
        env = Environment(effector=ReluPointMass24(), differentiable=False,
                          max_ep_duration=0.05)
        action = np.zeros((1, env.n_muscles), dtype=np.float32)
        for episode in range(3):
            obs, _ = env.reset(seed=episode, options={"deterministic": True})
            assert isinstance(obs, np.ndarray)
            terminated = False
            while not terminated:
                obs, _, terminated, _, _ = env.step(action, deterministic=True)
            assert not np.isnan(obs).any(), f"NaN in obs at end of episode {episode}"

    def test_batched_non_diff_full_episode(self):
        """Batched RL episode should complete without error and return arrays."""
        batch = 4
        env = Environment(effector=ReluPointMass24(), differentiable=False,
                          max_ep_duration=0.05)
        env.reset(options={"batch_size": batch, "deterministic": True})
        action = np.zeros((batch, env.n_muscles), dtype=np.float32)
        terminated = False
        while not terminated:
            obs, reward, terminated, _, info = env.step(action, deterministic=True)
        assert obs.shape == (batch, env.observation_space.shape[0])
        assert reward.shape == (batch, 1)
        assert all(isinstance(v, np.ndarray)
                   for v in info["states"].values() if v is not None)

    # --- noise in RL mode ---

    def test_action_noise_changes_noisy_action(self):
        env = Environment(effector=ReluPointMass24(), differentiable=False,
                          action_noise=1.0)
        env.reset(seed=0, options={"deterministic": True})
        action = np.ones((1, env.n_muscles), dtype=np.float32) * 0.5
        _, _, _, _, info = env.step(action, deterministic=False)
        # noisy action is a tensor internally; convert for comparison
        clean = torch.tensor(action)
        noisy = info["noisy action"]
        assert not torch.allclose(clean, noisy)

    def test_deterministic_true_suppresses_action_noise(self):
        env = Environment(effector=ReluPointMass24(), differentiable=False,
                          action_noise=5.0)
        env.reset(seed=0, options={"deterministic": True})
        action = np.ones((1, env.n_muscles), dtype=np.float32) * 0.5
        _, _, _, _, info = env.step(action, deterministic=True)
        assert torch.allclose(torch.tensor(action), info["noisy action"])
