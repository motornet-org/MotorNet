"""Simulation tests derived from the examples/ notebooks.

Covers end-to-end physics behaviour: directional activation, numerical
stability under prolonged stepping, environment step dynamics, and
policy forward pass / training loop correctness.
"""
import numpy as np
import torch

from motornet.effector import (
    Effector,
    ReluPointMass24,
    RigidTendonArm26,
)
from motornet.environment import Environment, RandomTargetReach
from motornet.muscle import (
    CompliantTendonHillMuscle,
    RigidTendonHillMuscle,
    RigidTendonHillMuscleThelen,
)
from motornet.policy import PolicyGRU
from motornet.skeleton import PointMass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIAG = np.sqrt(8.0)  # length of each arm in the X-shaped PointMass effector


def _make_x_effector(muscle_cls, **muscle_kwargs):
    """Four-muscle X-shaped PointMass effector as used in notebook 0."""
    eff = Effector(skeleton=PointMass(space_dim=2), muscle=muscle_cls())
    L = 2.0
    for coords, name in [
        ([[L,  L], [0, 0]], "UpRight"),
        ([[-L, L], [0, 0]], "UpLeft"),
        ([[-L, -L], [0, 0]], "DownLeft"),
        ([[L, -L], [0, 0]], "DownRight"),
    ]:
        kw = dict(path_fixation_body=[0, 1], path_coordinates=coords,
                  name=name, max_isometric_force=500)
        kw.update(muscle_kwargs)
        eff.add_muscle(**kw)
    return eff


def _run_steps(eff, action_list, n_steps):
    action = torch.tensor([action_list], dtype=torch.float32)
    for _ in range(n_steps):
        eff.step(action)


def _make_env_and_policy(hidden_dim=16):
    env = RandomTargetReach(effector=ReluPointMass24(), max_ep_duration=0.1)
    obs_dim = env.observation_space.shape[0]
    policy = PolicyGRU(input_dim=obs_dim, hidden_dim=hidden_dim,
                       output_dim=env.n_muscles, device="cpu")
    return env, policy


# ---------------------------------------------------------------------------
# Notebook 0 – directional simulation tests for Hill-type muscles
# ---------------------------------------------------------------------------

class TestHillMuscleDirectionalSimulation:
    """Notebook 0: activating one muscle should move the mass in the expected direction."""

    def test_rigid_hill_upright_activation_moves_up_right(self):
        eff = _make_x_effector(RigidTendonHillMuscle, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        _run_steps(eff, [1.0, 0.0, 0.0, 0.0], n_steps=100)
        x = eff.states["joint"][0, 0].item()
        y = eff.states["joint"][0, 1].item()
        assert x > 0
        assert y > 0

    def test_rigid_hill_isometric_cocontraction_minimal_displacement(self):
        eff = _make_x_effector(RigidTendonHillMuscle, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        _run_steps(eff, [1.0, 1.0, 1.0, 1.0], n_steps=150)
        pos = eff.states["joint"][0, :2]
        assert pos.abs().max().item() < 0.05

    def test_rigid_hill_no_nan_over_simulation(self):
        eff = _make_x_effector(RigidTendonHillMuscle, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        _run_steps(eff, [0.5, 0.2, 0.1, 0.3], n_steps=100)
        assert not torch.isnan(eff.states["joint"]).any()
        assert not torch.isnan(eff.states["muscle"]).any()

    def test_thelen_upright_activation_moves_up_right(self):
        eff = _make_x_effector(RigidTendonHillMuscleThelen, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        _run_steps(eff, [1.0, 0.0, 0.0, 0.0], n_steps=100)
        x = eff.states["joint"][0, 0].item()
        y = eff.states["joint"][0, 1].item()
        assert x > 0
        assert y > 0

    def test_thelen_isometric_cocontraction_minimal_displacement(self):
        eff = _make_x_effector(RigidTendonHillMuscleThelen, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        _run_steps(eff, [1.0, 1.0, 1.0, 1.0], n_steps=150)
        pos = eff.states["joint"][0, :2]
        assert pos.abs().max().item() < 0.05

    def test_thelen_no_nan_over_simulation(self):
        eff = _make_x_effector(RigidTendonHillMuscleThelen, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        _run_steps(eff, [0.4, 0.3, 0.2, 0.1], n_steps=100)
        assert not torch.isnan(eff.states["joint"]).any()

    def test_compliant_tendon_tug_of_war_cocontraction_stays_at_origin(self):
        """Notebook 0, section V.4: symmetric co-contraction should not displace the mass."""
        eff = Effector(skeleton=PointMass(space_dim=2),
                       muscle=CompliantTendonHillMuscle(),
                       timestep=1e-4, integration_method="rk4")
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[6, 0], [0, 0]],
                       max_isometric_force=100, tendon_length=5, optimal_muscle_length=0.8)
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[-6, 0], [0, 0]],
                       max_isometric_force=100, tendon_length=5, optimal_muscle_length=0.8)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        action = torch.tensor([[0.5, 0.5]])
        for _ in range(int(0.2 / 1e-4)):
            eff.step(action)
        x = eff.states["joint"][0, 0].item()
        assert abs(x) < 0.05

    def test_compliant_tendon_one_sided_activation_moves_mass(self):
        """Activating only the right-pulling muscle should move the mass rightward."""
        eff = Effector(skeleton=PointMass(space_dim=2),
                       muscle=CompliantTendonHillMuscle(),
                       timestep=1e-4, integration_method="rk4")
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[6, 0], [0, 0]],
                       max_isometric_force=100, tendon_length=5, optimal_muscle_length=0.8)
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[-6, 0], [0, 0]],
                       max_isometric_force=100, tendon_length=5, optimal_muscle_length=0.8)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        action = torch.tensor([[0.5, 0.0]])
        for _ in range(int(0.2 / 1e-4)):
            eff.step(action)
        x = eff.states["joint"][0, 0].item()
        assert x > 0


# ---------------------------------------------------------------------------
# Notebooks 1 & 2 – effector numerical stability over time
# ---------------------------------------------------------------------------

class TestEffectorSimulation:
    """Long-horizon stepping: numerical stability for arm26 and extended effectors."""

    def test_arm26_passive_drift_remains_finite(self):
        """Notebook 1: zero-input simulation stays finite across the full workspace."""
        arm26 = RigidTendonArm26(muscle=RigidTendonHillMuscleThelen())
        n = 5
        sho = torch.linspace(float(arm26.pos_lower_bound[0]),
                              float(arm26.pos_upper_bound[0]), n)
        elb = torch.linspace(float(arm26.pos_lower_bound[1]),
                              float(arm26.pos_upper_bound[1]), n)
        sho_g, elb_g = torch.meshgrid(sho, elb, indexing="ij")
        joint_states = torch.stack([sho_g.reshape(-1), elb_g.reshape(-1)], dim=1)
        arm26.reset(options={"joint_state": joint_states})
        batch = joint_states.shape[0]
        action = torch.zeros(batch, arm26.n_muscles)
        for _ in range(int(0.2 / arm26.dt)):
            arm26.step(action)
        assert torch.isfinite(arm26.states["joint"]).all()
        assert torch.isfinite(arm26.states["fingertip"]).all()

    def test_fifth_muscle_no_nan_after_steps(self):
        """Notebook 2: a fifth muscle added to ReluPointMass24 should not produce NaN."""
        eff = ReluPointMass24()
        eff.add_muscle(path_fixation_body=[0, 1],
                       path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=100)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        action = torch.ones(1, 5) * 0.5
        for _ in range(50):
            eff.step(action)
        assert not torch.isnan(eff.states["muscle"]).any()


# ---------------------------------------------------------------------------
# Notebook 3 – environment step dynamics
# ---------------------------------------------------------------------------

class TestEnvironmentStep:
    """Notebook 3: custom step overrides and step-level behaviour."""

    def test_custom_step_returns_custom_reward(self):
        class DistanceRewardEnv(Environment):
            def step(self, action, deterministic=False, **kwargs):
                obs, _, terminated, truncated, info = super().step(
                    action, deterministic=deterministic, **kwargs)
                info["distance"] = self.states["fingertip"].norm(dim=-1).mean().item()
                return obs, info["distance"], terminated, truncated, info

        env = DistanceRewardEnv(effector=ReluPointMass24())
        env.reset(options={"deterministic": True})
        _, reward, _, _, info = env.step(torch.zeros(1, env.n_muscles),
                                         deterministic=True)
        assert "distance" in info
        assert isinstance(reward, float)
        assert reward >= 0.0

    def test_custom_env_step_no_nan(self):
        class NoisyEnv(Environment):
            pass

        env = NoisyEnv(effector=ReluPointMass24(), obs_noise=0.1)
        env.reset(seed=0)
        for _ in range(20):
            obs, _, _, _, _ = env.step(torch.zeros(1, env.n_muscles))
        assert not torch.isnan(obs).any()


# ---------------------------------------------------------------------------
# Notebook 4 – PolicyGRU forward pass and training
# ---------------------------------------------------------------------------

class TestPolicyGRUSimulation:
    """Notebook 4: forward pass correctness, training loop, and gradient flow."""

    def test_forward_output_shape(self):
        env, policy = _make_env_and_policy()
        obs_dim = env.observation_space.shape[0]
        batch = 4
        h = policy.init_hidden(batch_size=batch)
        x = torch.zeros(batch, obs_dim)
        u, h_new = policy.forward(x, h)
        assert u.shape == (batch, env.n_muscles)
        assert h_new.shape == (1, batch, 16)

    def test_forward_output_in_unit_interval(self):
        env, policy = _make_env_and_policy()
        obs_dim = env.observation_space.shape[0]
        h = policy.init_hidden(batch_size=8)
        x = torch.randn(8, obs_dim)
        u, _ = policy.forward(x, h)
        assert (u >= 0.0).all()
        assert (u <= 1.0).all()

    def test_forward_output_changes_with_hidden_state(self):
        """Sequential steps should change the hidden state."""
        env, policy = _make_env_and_policy()
        obs_dim = env.observation_space.shape[0]
        h = policy.init_hidden(batch_size=1)
        x = torch.randn(1, obs_dim)
        u1, h1 = policy.forward(x, h)
        u2, h2 = policy.forward(x, h1)
        assert not torch.allclose(u1, u2) or not torch.allclose(h1, h2)

    def test_short_training_loop_runs_without_error(self):
        """Two gradient updates should complete without error."""
        env, policy = _make_env_and_policy(hidden_dim=16)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        for _ in range(2):
            h = policy.init_hidden(batch_size=4)
            obs, _ = env.reset(options={"batch_size": 4, "deterministic": True})
            terminated = False
            loss = torch.zeros(1)
            while not terminated:
                action, h = policy.forward(obs, h)
                obs, _, terminated, _, info = env.step(action, deterministic=True)
                loss = loss + info["states"]["fingertip"].norm(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradients_flow_through_policy(self):
        """Policy parameters should receive gradients after a full episode loss.backward()."""
        env, policy = _make_env_and_policy(hidden_dim=16)
        batch_size = 2
        h = policy.init_hidden(batch_size=batch_size)
        obs, info = env.reset(options={"batch_size": batch_size})
        terminated = False
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]
        while not terminated:
            action, h = policy(obs, h)
            obs, _, terminated, _, info = env.step(action=action)
            xy.append(info["states"]["fingertip"][:, None, :])
            tg.append(info["goal"][:, None, :])
        xy = torch.cat(xy, dim=1)
        tg = torch.cat(tg, dim=1)
        loss = torch.mean(torch.sum(torch.abs(xy - tg), dim=-1))
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
