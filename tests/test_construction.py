"""Construction tests derived from the examples/ notebooks.

Covers building effectors and environments, inspecting their configuration,
initial state values after reset, custom subclassing, and policy save/load —
none of these tests advance the simulation with step().
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from motornet.effector import (
    Effector,
    ReluPointMass24,
)
from motornet.environment import Environment, RandomTargetReach
from motornet.muscle import (
    ReluMuscle,
    RigidTendonHillMuscle,
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


def _make_env_and_policy(hidden_dim=16):
    env = RandomTargetReach(effector=ReluPointMass24(), max_ep_duration=0.1)
    obs_dim = env.observation_space.shape[0]
    policy = PolicyGRU(input_dim=obs_dim, hidden_dim=hidden_dim,
                       output_dim=env.n_muscles, device="cpu")
    return env, policy


# ---------------------------------------------------------------------------
# Notebook 0 – muscle demo: initial states at origin
# ---------------------------------------------------------------------------

class TestInitialStatesAtOrigin:
    """Notebook 0: muscle states after reset at the workspace origin."""

    def test_relu_muscle_lengths_equal_at_origin(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"joint_state": torch.zeros(2, 4)})
        lengths = eff.states["muscle"][:, 1, :]  # (batch, n_muscles)
        # Both batch rows at the same position → identical
        assert torch.allclose(lengths[0], lengths[1], atol=1e-5)
        # Symmetric X layout → all four muscles have the same length
        assert torch.allclose(lengths[:, 0], lengths[:, 1], atol=1e-5)
        assert torch.allclose(lengths[:, 0], lengths[:, 2], atol=1e-5)

    def test_relu_muscle_length_value_at_origin(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        lengths = eff.states["muscle"][:, 1, :]
        assert lengths.mean().item() == pytest.approx(_DIAG, abs=0.01)

    def test_relu_initial_activation_at_min(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        activations = eff.states["muscle"][:, 0, :]
        min_act = eff.muscle.min_activation.item()
        assert torch.allclose(activations, torch.full_like(activations, min_act), atol=1e-6)

    def test_relu_initial_velocity_zero(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        velocities = eff.states["muscle"][:, 2, :]
        assert velocities.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_hill_muscle_lengths_equal_at_origin(self):
        eff = _make_x_effector(RigidTendonHillMuscle, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(2, 4)})
        lengths = eff.states["muscle"][:, 1, :]
        assert torch.allclose(lengths[0], lengths[1], atol=1e-5)
        assert torch.allclose(lengths[:, 0], lengths[:, 1], atol=1e-5)

    def test_hill_initial_activation_at_min(self):
        eff = _make_x_effector(RigidTendonHillMuscle, tendon_length=0.0,
                                optimal_muscle_length=_DIAG)
        eff.reset(options={"joint_state": torch.zeros(1, 4)})
        activations = eff.states["muscle"][:, 0, :]
        min_act = eff.muscle.min_activation.item()
        assert (activations - min_act).abs().max().item() < 1e-5

    def test_batch_rows_identical_at_same_position(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"joint_state": torch.zeros(3, 4)})
        mstate = eff.states["muscle"]
        assert torch.allclose(mstate[0], mstate[1], atol=1e-5)
        assert torch.allclose(mstate[1], mstate[2], atol=1e-5)

    def test_muscle_state_shape(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"batch_size": 5})
        assert eff.states["muscle"].shape == (5, eff.muscle_state_dim, 4)

    def test_geometry_state_shape(self):
        eff = _make_x_effector(ReluMuscle)
        eff.reset(options={"batch_size": 5})
        assert eff.states["geometry"].shape == (5, eff.geometry_state_dim, 4)


# ---------------------------------------------------------------------------
# Notebook 1 – effector inspection
# ---------------------------------------------------------------------------

class TestEffectorInspection:
    """Notebook 1: print_muscle_wrappings, get_muscle_cfg, geometry state names."""

    def test_print_muscle_wrappings_output_contains_name(self, capsys):
        eff = ReluPointMass24()
        eff.print_muscle_wrappings()
        out = capsys.readouterr().out
        assert "UpperRight" in out

    def test_print_muscle_wrappings_shows_all_muscles(self, capsys):
        eff = ReluPointMass24()
        eff.print_muscle_wrappings()
        out = capsys.readouterr().out
        for name in eff.muscle_name:
            assert name in out

    def test_get_muscle_cfg_keys_match_muscle_names(self):
        eff = ReluPointMass24()
        cfg = eff.get_muscle_cfg()
        assert set(cfg.keys()) == set(eff.muscle_name)

    def test_get_muscle_cfg_n_fixation_points(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle())
        eff.add_muscle(path_fixation_body=[0, 0, 1],
                       path_coordinates=[[4, -2], [2, -2], [0, 0]],
                       max_isometric_force=100)
        cfg = eff.get_muscle_cfg()
        assert cfg["muscle_1"]["n_fixation_points"] == 3

    def test_geometry_state_name_first_two_entries(self):
        eff = ReluPointMass24()
        assert eff.geometry_state_name[0] == "musculotendon length"
        assert eff.geometry_state_name[1] == "musculotendon velocity"

    def test_geometry_state_name_length_matches_dof(self):
        eff = ReluPointMass24()
        assert len(eff.geometry_state_name) == 2 + eff.dof


# ---------------------------------------------------------------------------
# Notebook 2 – adding a fifth muscle to a pre-built effector
# ---------------------------------------------------------------------------

class TestAddExtraMuscle:
    """Notebook 2: building on a pre-built effector by adding an extra muscle."""

    def test_fifth_muscle_increments_count(self):
        eff = ReluPointMass24()
        assert eff.n_muscles == 4
        eff.add_muscle(path_fixation_body=[0, 1],
                       path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=100)
        assert eff.n_muscles == 5

    def test_fifth_muscle_state_shape_after_reset(self):
        eff = ReluPointMass24()
        eff.add_muscle(path_fixation_body=[0, 1],
                       path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=100)
        eff.reset(options={"batch_size": 3})
        assert eff.states["muscle"].shape == (3, eff.muscle_state_dim, 5)

    def test_fifth_muscle_geometry_state_shape_after_reset(self):
        eff = ReluPointMass24()
        eff.add_muscle(path_fixation_body=[0, 1],
                       path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=100)
        eff.reset(options={"batch_size": 3})
        assert eff.states["geometry"].shape[-1] == 5

    def test_muscle_state_feature_names_unchanged(self):
        eff = ReluPointMass24()
        names_before = list(eff.muscle.state_name)
        eff.add_muscle(path_fixation_body=[0, 1],
                       path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=100)
        assert eff.muscle.state_name == names_before


# ---------------------------------------------------------------------------
# Notebook 3 – environment construction and introspection
# ---------------------------------------------------------------------------

class TestEnvironmentConstruction:
    """Notebook 3: subclassing Environment and using introspection methods."""

    def test_custom_reset_sets_goal(self):
        class ConstantGoalEnv(Environment):
            def reset(self, *, seed=None, options=None):
                obs, info = super().reset(seed=seed, options=options)
                self.goal = torch.ones_like(self.goal) * 0.1
                return obs, info

        env = ConstantGoalEnv(effector=ReluPointMass24())
        env.reset(options={"deterministic": True})
        assert torch.allclose(env.goal, torch.ones_like(env.goal) * 0.1)

    def test_print_attributes_produces_output(self, capsys):
        env = Environment(effector=ReluPointMass24())
        env.print_attributes()
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_get_attributes_returns_matching_lists(self):
        env = Environment(effector=ReluPointMass24())
        names, values = env.get_attributes()
        assert isinstance(names, list)
        assert isinstance(values, list)
        assert len(names) == len(values)
        assert all(isinstance(n, str) for n in names)

    def test_get_attributes_includes_dt(self):
        env = Environment(effector=ReluPointMass24())
        names, _ = env.get_attributes()
        assert "dt" in names


# ---------------------------------------------------------------------------
# Notebook 4 – PolicyGRU construction, save/load
# ---------------------------------------------------------------------------

class TestPolicyGRUConstruction:
    """Notebook 4: PolicyGRU initialisation, hidden state, and save/load."""

    def test_init_hidden_shape(self):
        _, policy = _make_env_and_policy()
        h = policy.init_hidden(batch_size=7)
        assert h.shape == (1, 7, 16)

    def test_init_hidden_is_zero(self):
        _, policy = _make_env_and_policy()
        h = policy.init_hidden(batch_size=3)
        assert h.abs().max().item() == pytest.approx(0.0)

    def test_save_and_load_weights_produce_same_output(self):
        """save state_dict and reload → identical forward pass."""
        env, policy = _make_env_and_policy(hidden_dim=16)
        obs_dim = env.observation_space.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pt")
            torch.save(policy.state_dict(), path)

            policy2 = PolicyGRU(input_dim=obs_dim, hidden_dim=16,
                                 output_dim=env.n_muscles, device="cpu")
            policy2.load_state_dict(torch.load(path, weights_only=True))

        h1 = policy.init_hidden(1)
        h2 = policy2.init_hidden(1)
        x = torch.randn(1, obs_dim)
        u1, _ = policy.forward(x, h1)
        u2, _ = policy2.forward(x, h2)
        assert torch.allclose(u1, u2)

    def test_get_save_config_is_json_serializable(self):
        """Environment config should round-trip through JSON."""
        env = RandomTargetReach(effector=ReluPointMass24())
        cfg = env.get_save_config()
        json_str = json.dumps(cfg)
        loaded = json.loads(json_str)
        assert isinstance(loaded, dict)
        assert "effector" in loaded
        assert "name" in loaded
