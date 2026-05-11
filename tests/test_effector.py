"""Tests for motornet.effector — Effector base class and all pre-built effectors."""

import numpy as np
import pytest
import torch

from motornet.effector import Effector, FreePointMass24, Reacher
from motornet.muscle import ReluMuscle, RigidTendonHillMuscle
from motornet.skeleton import PointMass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_relu_effector_with_two_muscles():
    """Minimal effector: PointMass + 2 ReluMuscles pulling in opposite x-directions."""
    eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle())
    eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[2, 0], [0, 0]],
                   name='right', max_isometric_force=100.0)
    eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[-2, 0], [0, 0]],
                   name='left', max_isometric_force=100.0)
    return eff


# =============================================================================
# Effector base class
# =============================================================================

class TestEffectorBase:

    def test_add_muscle_increments_n_muscles(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle())
        assert eff.n_muscles == 0
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=100.0)
        assert eff.n_muscles == 1
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[-1, 0], [0, 0]],
                       max_isometric_force=100.0)
        assert eff.n_muscles == 2

    def test_add_muscle_stores_name(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle())
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[1, 0], [0, 0]],
                       name='myMuscle', max_isometric_force=50.0)
        assert eff.muscle_name[0] == 'myMuscle'

    def test_add_muscle_default_name(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle())
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[1, 0], [0, 0]],
                       max_isometric_force=50.0)
        assert eff.muscle_name[0] == 'muscle_1'

    def test_add_muscle_missing_required_kwarg_raises_type_error(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=RigidTendonHillMuscle())
        with pytest.raises(TypeError):
            # tendon_length and optimal_muscle_length are required but not provided
            eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[1, 0], [0, 0]],
                           max_isometric_force=100.0)

    def test_invalid_integration_method_raises_value_error(self):
        with pytest.raises(ValueError):
            Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle(),
                     integration_method='adams')

    def test_euler_method_accepted(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle(),
                       integration_method='euler')
        assert eff.integration_method == 'euler'

    def test_rk4_method_accepted(self):
        for name in ('rk4', 'rungekutta4', 'runge-kutta4', 'runge-kutta-4'):
            eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle(),
                           integration_method=name)
            assert eff.integration_method == name.casefold()

    def test_n_ministeps_sets_minidt(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle(),
                       timestep=0.01, n_ministeps=5)
        assert eff.minidt == pytest.approx(0.002)

    def test_reset_initializes_all_state_keys(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(seed=0)
        for key in ('joint', 'muscle', 'geometry', 'cartesian', 'fingertip'):
            assert eff.states[key] is not None

    def test_reset_joint_state_shape(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(options={"batch_size": 4})
        assert eff.states["joint"].shape == (4, 4)  # (batch, dof*2)

    def test_reset_muscle_state_shape(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(options={"batch_size": 4})
        assert eff.states["muscle"].shape == (4, 4, 2)  # (batch, state_dim, n_muscles)

    def test_reset_geometry_state_shape(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(options={"batch_size": 4})
        # geometry = (batch, 2+dof, n_muscles) = (4, 4, 2)
        assert eff.states["geometry"].shape == (4, 4, 2)

    def test_reset_cartesian_state_shape(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(options={"batch_size": 4})
        # PointMass: cartesian == joint → (batch, space_dim*2)
        assert eff.states["cartesian"].shape == (4, 4)

    def test_reset_fingertip_shape(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(options={"batch_size": 4})
        assert eff.states["fingertip"].shape == (4, 2)  # (batch, space_dim)

    def test_reset_with_explicit_full_joint_state(self):
        eff = make_relu_effector_with_two_muscles()
        joint = torch.tensor([[0.3, -0.2, 0.0, 0.0]])  # pos + vel
        eff.reset(options={"joint_state": joint})
        pos = eff.states["joint"][:, :2]
        assert torch.allclose(pos, joint[:, :2], atol=1e-5)

    def test_reset_with_explicit_position_only(self):
        eff = make_relu_effector_with_two_muscles()
        pos = torch.tensor([[0.3, -0.2]])  # position only (no velocity)
        eff.reset(options={"joint_state": pos})
        stored_pos = eff.states["joint"][:, :2]
        assert torch.allclose(stored_pos, pos, atol=1e-5)

    def test_reset_batch_size_tiles_state(self):
        eff = make_relu_effector_with_two_muscles()
        joint = torch.tensor([[0.3, -0.2, 0.0, 0.0]])
        eff.reset(options={"joint_state": joint, "batch_size": 5})
        assert eff.states["joint"].shape[0] == 5
        # All rows should be the same position
        assert torch.allclose(eff.states["joint"][:, :2],
                           joint[:, :2].expand(5, -1), atol=1e-5)

    def test_step_changes_joint_state(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(seed=0)
        state_before = eff.states["joint"].clone()
        action = torch.tensor([[1.0, 0.0]])  # activate right muscle only
        # Step 1: activation builds from 0, but force is still 0 (Euler uses prior state).
        # Step 2: force is now non-zero → velocity changes.
        # Step 3: velocity is non-zero → position changes.
        for _ in range(3):
            eff.step(action)
        assert not torch.allclose(eff.states["joint"], state_before)

    def test_step_no_nan(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(seed=0)
        action = torch.ones(1, 2) * 0.5
        for _ in range(50):
            eff.step(action)
        for key, val in eff.states.items():
            if val is not None:
                assert not torch.isnan(val).any(), f"NaN in state '{key}'"
                assert not torch.isinf(val).any(), f"Inf in state '{key}'"

    def test_draw_fixed_states_position_matches(self):
        eff = make_relu_effector_with_two_muscles()
        # Small position well within bounds
        pos = np.array([[0.2, -0.1]])
        states = eff.draw_fixed_states(position=pos, batch_size=1)
        assert states[0, 0].item() == pytest.approx(0.2, abs=1e-5)
        assert states[0, 1].item() == pytest.approx(-0.1, abs=1e-5)

    def test_draw_fixed_states_tiled_for_batch(self):
        eff = make_relu_effector_with_two_muscles()
        pos = np.array([[0.2, -0.1]])
        states = eff.draw_fixed_states(position=pos, batch_size=6)
        assert states.shape[0] == 6
        assert torch.allclose(states[0, :2], states[5, :2])

    def test_draw_fixed_states_zero_velocity_by_default(self):
        eff = make_relu_effector_with_two_muscles()
        pos = np.array([[0.2, -0.1]])
        states = eff.draw_fixed_states(position=pos, batch_size=1)
        assert states[0, 2:].abs().max().item() == pytest.approx(0.0)

    def test_draw_random_uniform_states_shape(self):
        eff = make_relu_effector_with_two_muscles()
        states = eff.draw_random_uniform_states(batch_size=10)
        assert states.shape == (10, 4)  # (batch, dof*2)

    def test_draw_random_uniform_states_within_pos_bounds(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(seed=42)
        states = eff.draw_random_uniform_states(batch_size=100)
        pos = states[:, :2]
        lb = eff.skeleton.pos_lower_bound
        ub = eff.skeleton.pos_upper_bound
        assert (pos >= lb).all()
        assert (pos <= ub).all()

    def test_draw_random_uniform_states_velocity_is_zero(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(seed=0)
        states = eff.draw_random_uniform_states(batch_size=10)
        assert states[:, 2:].abs().max().item() == pytest.approx(0.0)

    def test_euler_runs_many_steps_without_error(self):
        eff = make_relu_effector_with_two_muscles()
        eff.reset(seed=0)
        action = torch.tensor([[0.8, 0.2]])
        for _ in range(100):
            eff.step(action)
        assert torch.isfinite(eff.states["joint"]).all()

    def test_rk4_runs_many_steps_without_error(self):
        eff = Effector(skeleton=PointMass(space_dim=2), muscle=ReluMuscle(),
                       integration_method='rk4', timestep=0.01)
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[2, 0], [0, 0]],
                       name='right', max_isometric_force=100.0)
        eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[-2, 0], [0, 0]],
                       name='left', max_isometric_force=100.0)
        eff.reset(seed=0)
        action = torch.tensor([[0.8, 0.2]])
        for _ in range(100):
            eff.step(action)
        assert torch.isfinite(eff.states["joint"]).all()

    def test_rk4_and_euler_agree_after_many_steps_small_dt(self):
        # Both methods should produce similar trajectories at small timestep.
        # We compare after 20 ms of simulation (200 steps at dt=0.0001).
        def make_eff(method):
            e = Effector(skeleton=PointMass(space_dim=2, mass=1.0), muscle=ReluMuscle(),
                         integration_method=method, timestep=0.0001)
            e.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[2, 0], [0, 0]],
                         name='right', max_isometric_force=10.0)
            e.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[-2, 0], [0, 0]],
                         name='left', max_isometric_force=10.0)
            return e

        joint = torch.zeros(1, 4)
        e_euler = make_eff('euler')
        e_rk4 = make_eff('rk4')
        e_euler.reset(options={"joint_state": joint})
        e_rk4.reset(options={"joint_state": joint})
        action = torch.tensor([[0.7, 0.3]])
        for _ in range(200):
            e_euler.step(action)
            e_rk4.step(action)
        assert torch.allclose(e_euler.states["joint"], e_rk4.states["joint"], atol=1e-3)

    def test_batch_rows_are_independent(self):
        eff = make_relu_effector_with_two_muscles()
        joint_a = torch.tensor([[0.5, 0.3, 0.0, 0.0]])
        joint_b = torch.tensor([[-0.5, -0.3, 0.0, 0.0]])
        batch_joint = torch.cat([joint_a, joint_b], dim=0)

        eff.reset(options={"joint_state": batch_joint})
        action = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        eff.step(action)
        pos_a_from_batch = eff.states["joint"][0, :2].clone()
        pos_b_from_batch = eff.states["joint"][1, :2].clone()

        # Each batch element should evolve independently
        eff.reset(options={"joint_state": joint_a})
        eff.step(torch.tensor([[1.0, 0.0]]))
        pos_a_single = eff.states["joint"][0, :2]

        eff.reset(options={"joint_state": joint_b})
        eff.step(torch.tensor([[0.0, 1.0]]))
        pos_b_single = eff.states["joint"][0, :2]

        assert torch.allclose(pos_a_from_batch, pos_a_single, atol=1e-5)
        assert torch.allclose(pos_b_from_batch, pos_b_single, atol=1e-5)


# =============================================================================
# ReluPointMass24
# =============================================================================

class TestReluPointMass24:

    def test_has_4_muscles_after_init(self, relu_point_mass):
        assert relu_point_mass.n_muscles == 4

    def test_muscle_names(self, relu_point_mass):
        assert set(relu_point_mass.muscle_name) == {'UpperRight', 'UpperLeft', 'LowerRight', 'LowerLeft'}

    def test_joint_state_shape_after_reset(self, relu_point_mass):
        relu_point_mass.reset(options={"batch_size": 7})
        assert relu_point_mass.states["joint"].shape == (7, 4)

    def test_muscle_state_shape_after_reset(self, relu_point_mass):
        relu_point_mass.reset(options={"batch_size": 7})
        # (batch, state_dim=4, n_muscles=4)
        assert relu_point_mass.states["muscle"].shape == (7, 4, 4)

    def test_geometry_state_shape_after_reset(self, relu_point_mass):
        relu_point_mass.reset(options={"batch_size": 7})
        # (batch, 2+dof=4, n_muscles=4)
        assert relu_point_mass.states["geometry"].shape == (7, 4, 4)

    def test_fingertip_shape_after_reset(self, relu_point_mass):
        relu_point_mass.reset(options={"batch_size": 7})
        assert relu_point_mass.states["fingertip"].shape == (7, 2)

    def test_co_contraction_minimal_net_displacement(self, relu_point_mass):
        """Symmetric co-contraction should produce near-zero net movement."""
        relu_point_mass.reset(options={"joint_state": torch.zeros(1, 4)})
        pos_before = relu_point_mass.states["joint"][:, :2].clone()
        action = torch.ones(1, 4)  # all muscles equally activated
        for _ in range(50):
            relu_point_mass.step(action)
        pos_after = relu_point_mass.states["joint"][:, :2]
        displacement = (pos_after - pos_before).abs().max().item()
        assert displacement < 0.05  # symmetric system → minimal drift

    def test_asymmetric_activation_produces_movement(self, relu_point_mass):
        relu_point_mass.reset(options={"joint_state": torch.zeros(1, 4)})
        action = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # UpperRight + LowerRight active
        for _ in range(100):
            relu_point_mass.step(action)
        x = relu_point_mass.states["joint"][0, 0].item()
        assert x > 0.0  # should drift rightward

    def test_upward_activation_moves_point_mass_up(self, relu_point_mass):
        relu_point_mass.reset(options={"joint_state": torch.zeros(1, 4)})
        action = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # UpperRight + UpperLeft
        for _ in range(100):
            relu_point_mass.step(action)
        y = relu_point_mass.states["joint"][0, 1].item()
        assert y > 0.0

    def test_get_save_config_includes_n_muscles(self, relu_point_mass):
        cfg = relu_point_mass.get_save_config()
        assert cfg['n_muscles'] == 4

    def test_input_dim_equals_n_muscles(self, relu_point_mass):
        assert relu_point_mass.input_dim == relu_point_mass.n_muscles


# =============================================================================
# RigidTendonArm26
# =============================================================================

class TestRigidTendonArm26:

    def test_has_6_muscles_after_init(self, thelen_arm26):
        assert thelen_arm26.n_muscles == 6

    def test_muscle_names(self, thelen_arm26):
        expected = ['pectoralis', 'deltoid', 'brachioradialis', 'tricepslat', 'biceps', 'tricepslong']
        assert thelen_arm26.muscle_name == expected

    def test_joint_state_shape_after_reset(self, thelen_arm26):
        thelen_arm26.reset(options={"batch_size": 3})
        assert thelen_arm26.states["joint"].shape == (3, 4)  # 2 joints × 2 (pos + vel)

    def test_muscle_state_shape_after_reset(self, thelen_arm26):
        thelen_arm26.reset(options={"batch_size": 3})
        # (batch, state_dim=7, n_muscles=6)
        assert thelen_arm26.states["muscle"].shape == (3, 7, 6)

    def test_geometry_state_shape_after_reset(self, thelen_arm26):
        thelen_arm26.reset(options={"batch_size": 3})
        # (batch, 2+dof=4, n_muscles=6)
        assert thelen_arm26.states["geometry"].shape == (3, 4, 6)

    def test_fingertip_shape_after_reset(self, thelen_arm26):
        thelen_arm26.reset(options={"batch_size": 3})
        assert thelen_arm26.states["fingertip"].shape == (3, 2)

    def test_no_nan_after_passive_simulation(self, thelen_arm26):
        thelen_arm26.reset(seed=0)
        action = torch.zeros(1, 6)
        for _ in range(100):
            thelen_arm26.step(action)
        for key, val in thelen_arm26.states.items():
            if val is not None:
                assert not torch.isnan(val).any(), f"NaN in state '{key}'"

    def test_geometry_polynomial_moment_arms_finite(self, thelen_arm26):
        thelen_arm26.reset(options={"batch_size": 5})
        geom = thelen_arm26.states["geometry"]
        # moment arms are in slots [2:] of the geometry state
        moment_arms = geom[:, 2:, :]
        assert torch.isfinite(moment_arms).all()

    def test_musculotendon_lengths_positive(self, thelen_arm26):
        thelen_arm26.reset(options={"batch_size": 5})
        lengths = thelen_arm26.states["geometry"][:, 0, :]
        assert (lengths > 0).all()

    def test_endpoint_within_arm_reach(self, thelen_arm26):
        thelen_arm26.reset(seed=0)
        fingertip = thelen_arm26.states["fingertip"]
        arm = thelen_arm26.skeleton
        max_reach = arm.L1 + arm.L2
        dist = fingertip.norm(dim=-1)
        assert (dist <= max_reach + 1e-3).all()


# =============================================================================
# CompliantTendonArm26
# =============================================================================

class TestCompliantTendonArm26:

    def test_has_6_muscles_after_init(self, compliant_arm26):
        assert compliant_arm26.n_muscles == 6

    def test_uses_rk4_by_default(self, compliant_arm26):
        assert compliant_arm26.integration_method in ('rk4', 'rungekutta4', 'runge-kutta4', 'runge-kutta-4')

    def test_joint_state_shape_after_reset(self, compliant_arm26):
        compliant_arm26.reset(options={"batch_size": 2})
        assert compliant_arm26.states["joint"].shape == (2, 4)

    def test_muscle_state_shape_after_reset(self, compliant_arm26):
        compliant_arm26.reset(options={"batch_size": 2})
        assert compliant_arm26.states["muscle"].shape == (2, 7, 6)

    def test_geometry_state_shape_after_reset(self, compliant_arm26):
        compliant_arm26.reset(options={"batch_size": 2})
        assert compliant_arm26.states["geometry"].shape == (2, 4, 6)

    def test_no_nan_after_short_simulation(self, compliant_arm26):
        compliant_arm26.reset(seed=0)
        action = torch.ones(1, 6) * 0.1
        for _ in range(50):
            compliant_arm26.step(action)
        for key, val in compliant_arm26.states.items():
            if val is not None:
                assert not torch.isnan(val).any(), f"NaN in state '{key}'"

    def test_timestep_is_small(self, compliant_arm26):
        # CompliantTendonArm26 uses a small dt by default (0.0002) for numerical stability
        assert compliant_arm26.dt <= 0.001

    def test_muscle_lengths_positive_after_reset(self, compliant_arm26):
        compliant_arm26.reset(options={"batch_size": 2})
        muscle_lengths = compliant_arm26.states["muscle"][:, 1, :]
        assert (muscle_lengths > 0).all()


# =============================================================================
# FreePointMass24
# =============================================================================

class TestFreePointMass24:

    def test_has_4_muscles_after_init(self, free_point_mass):
        assert free_point_mass.n_muscles == 4

    def test_muscle_names(self, free_point_mass):
        assert free_point_mass.muscle_name == ['r', 'u', 'l', 'd']

    def test_input_dim_equals_n_muscles(self, free_point_mass):
        assert free_point_mass.input_dim == 4

    def test_joint_state_shape_after_reset(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 3})
        # PointMass dof=2 → joint = (batch, 4) = pos(2) + vel(2)
        assert free_point_mass.states["joint"].shape == (3, 4)

    def test_muscle_state_shape_after_reset(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 3})
        # ReluMuscle state_dim == 4; FreePointMass24 has 4 muscles
        assert free_point_mass.states["muscle"].shape == (3, 4, 4)

    def test_geometry_state_shape_after_reset(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 3})
        # (batch, 2+dof=4, n_muscles=4)
        assert free_point_mass.states["geometry"].shape == (3, 4, 4)

    def test_fingertip_shape_after_reset(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 3})
        assert free_point_mass.states["fingertip"].shape == (3, 2)

    def test_musculotendon_lengths_are_zero(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 2})
        assert torch.all(free_point_mass.states["geometry"][:, 0, :] == 0.0)

    def test_musculotendon_velocities_are_zero(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 2})
        assert torch.all(free_point_mass.states["geometry"][:, 1, :] == 0.0)

    def test_moment_arms_match_a0(self, free_point_mass):
        free_point_mass.reset(options={"batch_size": 5})
        moment_arms = free_point_mass.states["geometry"][:, 2:, :]  # (5, 2, 4)
        expected = np.array([-1., 0., 1., 0., 0., -1., 0., 1.],
                             dtype=np.float32).reshape(2, 4)
        expected_t = torch.tensor(expected).unsqueeze(0).expand(5, -1, -1)
        assert torch.allclose(moment_arms, expected_t)

    def test_geometry_is_constant_across_joint_states(self, free_point_mass):
        pos_a = torch.zeros(1, 4)
        pos_b = torch.tensor([[0.3, -0.2, 0.0, 0.0]])
        free_point_mass.reset(options={"joint_state": pos_a})
        geom_a = free_point_mass.states["geometry"].clone()
        free_point_mass.reset(options={"joint_state": pos_b})
        geom_b = free_point_mass.states["geometry"].clone()
        assert torch.allclose(geom_a, geom_b)

    def test_default_workspace_bounds(self, free_point_mass):
        lb = free_point_mass.skeleton.pos_lower_bound.flatten().tolist()
        ub = free_point_mass.skeleton.pos_upper_bound.flatten().tolist()
        assert lb == pytest.approx([-0.6, -0.6])
        assert ub == pytest.approx([0.6, 0.6])

    def test_custom_skeleton_mass_is_respected(self):
        from motornet.skeleton import PointMass
        r = FreePointMass24(muscle=ReluMuscle(), skeleton=PointMass(space_dim=2, mass=3.25))
        assert r.skeleton.mass == pytest.approx(3.25)

    def test_default_max_isometric_force_is_1000(self):
        r = FreePointMass24(muscle=ReluMuscle())
        assert r.tobuild__muscle["max_isometric_force"] == [[1000] * 4]


# =============================================================================
# Reacher
# =============================================================================

class TestReacher:

    def test_has_4_muscles_after_init(self, reacher):
        assert reacher.n_muscles == 4

    def test_muscle_names(self, reacher):
        assert reacher.muscle_name == ['sf', 'se', 'ef', 'ee']

    def test_input_dim_equals_n_muscles(self, reacher):
        assert reacher.input_dim == 4

    def test_joint_state_shape_after_reset(self, reacher):
        reacher.reset(options={"batch_size": 3})
        assert reacher.states["joint"].shape == (3, 4)  # 2 joints × (pos + vel)

    def test_muscle_state_shape_after_reset(self, reacher):
        reacher.reset(options={"batch_size": 3})
        # ReluMuscle state_dim == 4; Reacher has 4 muscles
        assert reacher.states["muscle"].shape == (3, 4, 4)

    def test_geometry_state_shape_after_reset(self, reacher):
        reacher.reset(options={"batch_size": 3})
        # (batch, 2+dof=4, n_muscles=4)
        assert reacher.states["geometry"].shape == (3, 4, 4)

    def test_fingertip_shape_after_reset(self, reacher):
        reacher.reset(options={"batch_size": 3})
        assert reacher.states["fingertip"].shape == (3, 2)

    def test_musculotendon_lengths_are_zero(self, reacher):
        reacher.reset(options={"batch_size": 2})
        lengths = reacher.states["geometry"][:, 0, :]
        assert torch.all(lengths == 0.0)

    def test_musculotendon_velocities_are_zero(self, reacher):
        reacher.reset(options={"batch_size": 2})
        velocities = reacher.states["geometry"][:, 1, :]
        assert torch.all(velocities == 0.0)

    def test_moment_arms_match_a0(self, reacher):
        import numpy as np
        reacher.reset(options={"batch_size": 5})
        moment_arms = reacher.states["geometry"][:, 2:, :]  # (5, 2, 4)
        expected = np.array([-1., 1., 0., 0., 0., 0., -1., 1.],
                             dtype=np.float32).reshape(2, 4)
        expected_t = torch.tensor(expected).unsqueeze(0).expand(5, -1, -1)
        assert torch.allclose(moment_arms, expected_t)

    def test_geometry_is_constant_across_joint_states(self, reacher):
        """Moment arms must not change regardless of joint configuration."""
        pos_a = torch.zeros(1, 4)
        pos_b = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        reacher.reset(options={"joint_state": pos_a})
        geom_a = reacher.states["geometry"].clone()
        reacher.reset(options={"joint_state": pos_b})
        geom_b = reacher.states["geometry"].clone()
        assert torch.allclose(geom_a, geom_b)

    def test_custom_max_isometric_force_via_muscle_kwargs(self):
        import torch
        r = Reacher(muscle=ReluMuscle(), muscle_kwargs={"max_isometric_force": [500] * 4})
        r.reset(seed=0)
        # The muscle was built successfully; activation at max should produce a finite state
        assert torch.isfinite(r.states["muscle"]).all()

    def test_default_max_isometric_force_is_1000(self):
        r = Reacher(muscle=ReluMuscle())
        # _merge_muscle_kwargs appends the list, so the stored value is [[1000]*4]
        assert r.tobuild__muscle["max_isometric_force"] == [[1000] * 4]
