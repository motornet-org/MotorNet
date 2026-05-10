"""Tests for motornet.skeleton — PointMass and TwoDofArm."""

import numpy as np
import pytest
import torch

from motornet.skeleton import PointMass, TwoDofArm
from motornet.effector import ReluPointMass24, RigidTendonArm26
from motornet.muscle import RigidTendonHillMuscleThelen


# =============================================================================
# PointMass
# =============================================================================

class TestPointMass:

    @pytest.fixture
    def pm_effector(self):
        """PointMass already built inside an effector."""
        return ReluPointMass24()

    @pytest.fixture
    def pm(self, pm_effector):
        """The built skeleton from the effector."""
        return pm_effector.skeleton

    def test_dof_equals_space_dim(self):
        skel = PointMass(space_dim=2)
        assert skel.dof == skel.space_dim == 2

    def test_dof_three(self):
        skel = PointMass(space_dim=3)
        assert skel.dof == 3

    def test_joint2cartesian_is_identity(self, pm):
        state = torch.tensor([[0.3, -0.5, 0.1, -0.2]])
        out = pm.joint2cartesian(state)
        assert torch.allclose(out, state)

    def test_joint2cartesian_preserves_batch_dimension(self, pm):
        state = torch.randn(5, 4)
        out = pm.joint2cartesian(state)
        assert out.shape == (5, 4)

    def test_path2cartesian_worldspace_fixation_is_fixed(self, pm_effector, pm):
        """Fixation on worldspace (body index 0) should stay at its coordinates."""
        # Set point-mass at some position
        pos = torch.tensor([[0.3, 0.7, 0.0, 0.0]])
        path_coords = torch.tensor([[[2.0], [3.0]]], dtype=torch.float32)  # (1, 2, 1)
        path_body = torch.tensor([[[0.0]]], dtype=torch.float32)            # body index 0 = worldspace
        xy, dxy_dt, _ = pm.path2cartesian(path_coords, path_body, pos)
        # World-fixed point has velocity zero and position equal to its coords
        assert xy[0, 0, 0].item() == pytest.approx(2.0, abs=1e-5)
        assert xy[0, 1, 0].item() == pytest.approx(3.0, abs=1e-5)
        assert dxy_dt[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert dxy_dt[0, 1, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_path2cartesian_body_fixation_moves_with_mass(self, pm):
        """Fixation on the body (index 1, coord=[0,0]) should be at the mass's position."""
        pos = torch.tensor([[0.4, -0.3, 0.0, 0.0]])
        path_coords = torch.tensor([[[0.0], [0.0]]], dtype=torch.float32)  # origin of the body
        path_body = torch.tensor([[[1.0]]], dtype=torch.float32)            # body index 1 = the mass
        xy, _, _ = pm.path2cartesian(path_coords, path_body, pos)
        assert xy[0, 0, 0].item() == pytest.approx(0.4, abs=1e-5)
        assert xy[0, 1, 0].item() == pytest.approx(-0.3, abs=1e-5)

    def test_integrate_zero_force_zero_velocity_no_change(self, pm):
        state = torch.tensor([[0.5, -0.3, 0.0, 0.0]])  # [x, y, vx, vy]
        derivative = torch.zeros(1, 2)  # zero force
        new_state = pm.integrate(dt=0.01, state_derivative=derivative, joint_state=state)
        # Position should not change (vel=0); velocity stays 0
        assert new_state[0, 0].item() == pytest.approx(0.5, abs=1e-5)
        assert new_state[0, 1].item() == pytest.approx(-0.3, abs=1e-5)

    def test_integrate_velocity_updates_position(self, pm):
        # Starting at [0, 0] with velocity [1, 0], after dt=0.1 → x ≈ 0.1
        state = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        derivative = torch.zeros(1, 2)
        new_state = pm.integrate(dt=0.1, state_derivative=derivative, joint_state=state)
        assert new_state[0, 0].item() == pytest.approx(0.1, abs=1e-5)

    def test_clip_position_clamps_to_bounds(self, pm):
        # PointMass bounds are [-1, 1] for ReluPointMass24
        state_over = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
        derivative = torch.zeros(1, 2)
        new_state = pm.integrate(dt=0.01, state_derivative=derivative, joint_state=state_over)
        assert new_state[0, 0].item() <= pm.pos_upper_bound[0, 0].item() + 1e-4

    def test_clip_velocity_zero_at_lower_bound(self, pm):
        # At lower position boundary with negative velocity → velocity zeroed
        lb = pm.pos_lower_bound[0, 0].item()
        pos = torch.tensor([[lb, 0.0]])
        vel = torch.tensor([[-1.0, 0.0]])  # moving toward lower boundary
        clipped = pm.clip_velocity(pos, vel)
        assert clipped[0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_clip_velocity_zero_at_upper_bound(self, pm):
        ub = pm.pos_upper_bound[0, 0].item()
        pos = torch.tensor([[ub, 0.0]])
        vel = torch.tensor([[1.0, 0.0]])  # moving toward upper boundary
        clipped = pm.clip_velocity(pos, vel)
        assert clipped[0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_clip_velocity_allows_motion_away_from_lower_bound(self, pm):
        lb = pm.pos_lower_bound[0, 0].item()
        pos = torch.tensor([[lb, 0.0]])
        vel = torch.tensor([[1.0, 0.0]])  # moving AWAY from lower boundary
        clipped = pm.clip_velocity(pos, vel)
        assert clipped[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_clip_velocity_allows_motion_away_from_upper_bound(self, pm):
        ub = pm.pos_upper_bound[0, 0].item()
        pos = torch.tensor([[ub, 0.0]])
        vel = torch.tensor([[-1.0, 0.0]])  # moving AWAY from upper boundary
        clipped = pm.clip_velocity(pos, vel)
        assert clipped[0, 0].item() == pytest.approx(-1.0, abs=1e-5)

    def test_state_dim_after_build(self, pm):
        assert pm.state_dim == 4  # [x, y, vx, vy]

    def test_geometry_state_dim(self, pm):
        # 2 (len + vel) + dof (moments) = 2 + 2 = 4
        assert pm.geometry_state_dim == 4


# =============================================================================
# TwoDofArm
# =============================================================================

class TestTwoDofArm:

    @pytest.fixture
    def arm_effector(self):
        """TwoDofArm inside a RigidTendonArm26 effector — already built."""
        return RigidTendonArm26(muscle=RigidTendonHillMuscleThelen())

    @pytest.fixture
    def arm(self, arm_effector):
        return arm_effector.skeleton

    def test_dof_equals_two(self):
        assert TwoDofArm().dof == 2

    def test_space_dim_equals_two(self):
        assert TwoDofArm().space_dim == 2

    def test_default_bone_lengths(self):
        a = TwoDofArm()
        assert a.L1 == pytest.approx(0.309)
        assert a.L2 == pytest.approx(0.26)

    def test_joint2cartesian_at_zero_angles_gives_L1_plus_L2(self):
        arm = TwoDofArm()
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        cart = arm.joint2cartesian(state)
        expected_x = arm.L1 + arm.L2  # cos(0) + cos(0) = 1 + 1
        assert cart[0, 0].item() == pytest.approx(expected_x, abs=1e-5)
        assert cart[0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_joint2cartesian_at_90deg_shoulder_gives_vertical(self):
        arm = TwoDofArm()
        state = torch.tensor([[np.pi / 2, 0.0, 0.0, 0.0]])
        cart = arm.joint2cartesian(state)
        # end_pos_x = L1*cos(π/2) + L2*cos(π/2) ≈ 0
        # end_pos_y = L1*sin(π/2) + L2*sin(π/2) = L1 + L2
        assert cart[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert cart[0, 1].item() == pytest.approx(arm.L1 + arm.L2, abs=1e-5)

    def test_joint2cartesian_at_zero_velocity_gives_zero_endpoint_velocity(self):
        arm = TwoDofArm()
        state = torch.tensor([[0.5, 0.3, 0.0, 0.0]])
        cart = arm.joint2cartesian(state)
        # With vel=0, endpoint velocity should be 0
        assert cart[0, 2].item() == pytest.approx(0.0, abs=1e-5)
        assert cart[0, 3].item() == pytest.approx(0.0, abs=1e-5)

    def test_joint2cartesian_batch(self):
        arm = TwoDofArm()
        state = torch.zeros(5, 4)
        cart = arm.joint2cartesian(state)
        assert cart.shape == (5, 4)
        # All should give same result
        assert torch.allclose(cart[0], cart[1])

    def test_ode_zero_torques_zero_velocity_zero_acceleration(self):
        arm = TwoDofArm()
        state = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        torques = torch.zeros(1, 2)
        endpoint_load = torch.zeros(1, 2)
        acc = arm.ode(torques, state, endpoint_load)
        assert torch.allclose(acc, torch.zeros(1, 2), atol=1e-5)

    def test_ode_nonzero_torque_gives_nonzero_acceleration(self):
        arm = TwoDofArm()
        state = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        torques = torch.tensor([[10.0, 0.0]])
        endpoint_load = torch.zeros(1, 2)
        acc = arm.ode(torques, state, endpoint_load)
        assert acc.abs().sum().item() > 0

    def test_inertia_matrix_positive_definite(self):
        arm = TwoDofArm()
        # c2 = cos(elbow angle) — test at a few angles
        for elb in [0.0, 0.5, 1.0, 1.5]:
            c2 = np.cos(elb)
            M = arm.inertia_c.numpy()[0] + c2 * arm.inertia_m.numpy()[0]
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            assert det > 0, f"Inertia matrix not positive definite at elbow={elb}"

    def test_integrate_updates_position_and_velocity(self, arm):
        state = torch.tensor([[0.3, 0.5, 0.1, -0.1]])
        derivative = torch.tensor([[5.0, -5.0]])
        new_state = arm.integrate(dt=0.01, state_derivative=derivative, joint_state=state)
        # Position = old_pos + old_vel * dt
        assert new_state[0, 0].item() == pytest.approx(0.3 + 0.1 * 0.01, abs=1e-4)
        assert new_state[0, 1].item() == pytest.approx(0.5 + (-0.1) * 0.01, abs=1e-4)

    def test_integrate_respects_position_bounds(self, arm):
        ub = arm.pos_upper_bound[0, 0].item()
        state = torch.tensor([[ub + 0.5, 0.5, 0.0, 0.0]])
        derivative = torch.zeros(1, 2)
        new_state = arm.integrate(dt=0.01, state_derivative=derivative, joint_state=state)
        assert new_state[0, 0].item() <= ub + 1e-5

    def test_clip_velocity_zeroed_at_lower_pos_bound(self, arm):
        lb = arm.pos_lower_bound[0, 0].item()
        pos = torch.tensor([[lb, 0.5]])
        vel = torch.tensor([[-1.0, 0.0]])
        clipped = arm.clip_velocity(pos, vel)
        assert clipped[0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_joint_limits_sho_and_elb(self, arm_effector):
        eff = arm_effector
        sho_lb = eff.pos_lower_bound[0]
        sho_ub = eff.pos_upper_bound[0]
        elb_lb = eff.pos_lower_bound[1]
        elb_ub = eff.pos_upper_bound[1]
        assert sho_lb.item() == pytest.approx(np.deg2rad(0), abs=1e-4)
        assert sho_ub.item() == pytest.approx(np.deg2rad(135), abs=1e-4)
        assert elb_lb.item() == pytest.approx(np.deg2rad(0), abs=1e-4)
        assert elb_ub.item() == pytest.approx(np.deg2rad(155), abs=1e-4)

    def test_cartesian_position_from_effector(self, arm_effector):
        arm_effector.reset(seed=0, options={"batch_size": 3})
        cart = arm_effector.states["cartesian"]
        joint = arm_effector.states["joint"]
        # Manually compute for one batch element
        pos0, pos1 = joint[0, 0].item(), joint[0, 1].item()
        arm = arm_effector.skeleton
        expected_x = arm.L1 * np.cos(pos0) + arm.L2 * np.cos(pos0 + pos1)
        assert cart[0, 0].item() == pytest.approx(expected_x, abs=1e-4)
