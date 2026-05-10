"""Tests for motornet.muscle — all muscle classes and the shared activation ODE."""

import numpy as np
import pytest
import torch as th

from motornet.muscle import (
    CompliantTendonHillMuscle,
    MujocoHillMuscle,
    ReluMuscle,
    RigidTendonHillMuscle,
    RigidTendonHillMuscleThelen,
)
from tests.conftest import make_geometry_state


# =============================================================================
# Shared activation ODE  (tested via ReluMuscle, which uses the base _ode)
# =============================================================================

class TestActivationODE:
    """Tests for Muscle.activation_ode — shared by all muscle types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.muscle = ReluMuscle()
        self.muscle.build(timestep=0.01, max_isometric_force=[1.0])

    def _ode(self, excitation, activation):
        exc = th.tensor([[[excitation]]], dtype=th.float32)
        act = th.tensor([[[activation]]], dtype=th.float32)
        return self.muscle.activation_ode(exc, act).item()

    def test_derivative_positive_when_excited_above_activation(self):
        assert self._ode(0.8, 0.3) > 0

    def test_derivative_negative_when_excited_below_activation(self):
        assert self._ode(0.2, 0.8) < 0

    def test_derivative_zero_when_excitation_equals_activation(self):
        d = self._ode(0.5, 0.5)
        assert abs(d) < 1e-5

    def test_rise_uses_tau_activation(self):
        # d = (u - a) / (tau_activation * (0.5 + 1.5*a))  when u > a
        u, a = 0.8, 0.3
        tmp = 0.5 + 1.5 * a
        expected = (u - a) / (self.muscle.tau_activation.item() * tmp)
        assert self._ode(u, a) == pytest.approx(expected, rel=1e-4)

    def test_fall_uses_tau_deactivation(self):
        # d = (u - a) / (tau_deactivation / (0.5 + 1.5*a))  when u < a
        u, a = 0.2, 0.8
        tmp = 0.5 + 1.5 * a
        expected = (u - a) / (self.muscle.tau_deactivation.item() / tmp)
        assert self._ode(u, a) == pytest.approx(expected, rel=1e-4)

    def test_excitation_clipped_to_min_activation(self):
        # Sending excitation below min_activation should behave as if excitation == min_activation
        d_at_min = self._ode(self.muscle.min_activation.item(), 0.5)
        d_below = self._ode(-1.0, 0.5)
        assert d_at_min == pytest.approx(d_below, rel=1e-4)

    def test_excitation_clipped_to_one(self):
        d_at_one = self._ode(1.0, 0.5)
        d_above = self._ode(2.0, 0.5)
        assert d_at_one == pytest.approx(d_above, rel=1e-4)

    def test_activation_clipped_to_min_before_ode(self):
        # activation below min gets clipped; result should equal the min-clipped case
        d_at_min = self._ode(0.5, self.muscle.min_activation.item())
        d_below = self._ode(0.5, -1.0)
        assert d_at_min == pytest.approx(d_below, rel=1e-4)

    def test_batch_consistent(self):
        # Both batch elements with the same inputs should produce the same derivative
        exc = th.tensor([[[0.7]], [[0.7]]], dtype=th.float32)
        act = th.tensor([[[0.4]], [[0.4]]], dtype=th.float32)
        d = self.muscle.activation_ode(exc, act)
        assert d[0].item() == pytest.approx(d[1].item(), rel=1e-5)

    def test_multi_muscle_independent(self):
        # Different excitations for different muscles produce independent derivatives
        m = ReluMuscle()
        m.build(timestep=0.01, max_isometric_force=[1.0, 1.0])
        exc = th.tensor([[[0.9, 0.1]]], dtype=th.float32)
        act = th.tensor([[[0.3, 0.7]]], dtype=th.float32)
        d = m.activation_ode(exc, act)
        assert d[0, 0, 0].item() > 0   # 0.9 > 0.3 → rising
        assert d[0, 0, 1].item() < 0   # 0.1 < 0.7 → falling


# =============================================================================
# clip_activation
# =============================================================================

class TestClipActivation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.muscle = ReluMuscle(min_activation=0.05)
        self.muscle.build(timestep=0.01, max_isometric_force=[1.0])

    def test_clips_below_min(self):
        x = th.tensor([[[0.0]]])
        out = self.muscle.clip_activation(x)
        assert out.item() == pytest.approx(0.05)

    def test_clips_above_one(self):
        x = th.tensor([[[1.5]]])
        out = self.muscle.clip_activation(x)
        assert out.item() == pytest.approx(1.0)

    def test_passes_through_valid(self):
        x = th.tensor([[[0.5]]])
        out = self.muscle.clip_activation(x)
        assert out.item() == pytest.approx(0.5)

    def test_clips_at_exactly_min(self):
        x = th.tensor([[[0.05]]])
        out = self.muscle.clip_activation(x)
        assert out.item() == pytest.approx(0.05)

    def test_clips_at_exactly_one(self):
        x = th.tensor([[[1.0]]])
        out = self.muscle.clip_activation(x)
        assert out.item() == pytest.approx(1.0)


# =============================================================================
# ReluMuscle
# =============================================================================

class TestReluMuscle:

    def test_state_names(self):
        m = ReluMuscle()
        assert m.state_name == ['activation', 'muscle length', 'muscle velocity', 'force']

    def test_state_dim(self):
        m = ReluMuscle()
        assert m.state_dim == 4

    def test_build_sets_n_muscles_single(self):
        m = ReluMuscle()
        m.build(timestep=0.01, max_isometric_force=[50.0])
        assert m.n_muscles == 1

    def test_build_sets_n_muscles_multi(self):
        m = ReluMuscle()
        m.build(timestep=0.01, max_isometric_force=[50.0, 100.0, 150.0])
        assert m.n_muscles == 3

    def test_build_stores_max_iso_force(self):
        m = ReluMuscle()
        m.build(timestep=0.01, max_isometric_force=[42.0])
        assert m.max_iso_force.item() == pytest.approx(42.0)

    def test_build_marks_built_flag(self, built_relu_muscle):
        assert built_relu_muscle.built is True

    def test_initial_state_shape_batch1(self, built_relu_muscle):
        g = make_geometry_state(1, 0.0, 2)
        s = built_relu_muscle.get_initial_muscle_state(batch_size=1, geometry_state=g)
        assert s.shape == (1, 4, 2)  # (batch, state_dim, n_muscles)

    def test_initial_state_shape_batch_n(self, built_relu_muscle):
        g = make_geometry_state(7, 0.0, 2)
        s = built_relu_muscle.get_initial_muscle_state(batch_size=7, geometry_state=g)
        assert s.shape == (7, 4, 2)

    def test_initial_activation_equals_min(self, built_relu_muscle):
        g = make_geometry_state(1, 0.0, 2)
        s = built_relu_muscle.get_initial_muscle_state(1, g)
        min_act = built_relu_muscle.min_activation.item()
        assert s[:, 0, :].min().item() == pytest.approx(min_act)

    def test_force_equals_activation_times_max_iso_force(self, built_relu_muscle):
        # muscle force = activation * max_iso_force
        activation = 0.6
        g = make_geometry_state(1, 0.0, 2)
        muscle_state = th.zeros(1, 4, 2)
        muscle_state[:, 0, :] = activation
        deriv = built_relu_muscle.ode(th.tensor([[[activation, activation]]]), muscle_state)
        new_state = built_relu_muscle.integrate(
            dt=0.01, state_derivative=deriv, muscle_state=muscle_state, geometry_state=g
        )
        force_muscle0 = new_state[0, 3, 0].item()
        expected = pytest.approx(activation * built_relu_muscle.max_iso_force[0, 0, 0].item(), rel=1e-3)
        assert force_muscle0 == expected

    def test_force_non_negative(self, built_relu_muscle):
        g = make_geometry_state(5, 0.1, 2)
        s = built_relu_muscle.get_initial_muscle_state(5, g)
        assert (s[:, 3, :] >= 0).all()

    def test_force_zero_when_activation_is_zero(self, built_relu_muscle):
        g = make_geometry_state(1, 0.0, 2)
        muscle_state = th.zeros(1, 4, 2)
        deriv = built_relu_muscle.ode(th.zeros(1, 1, 2), muscle_state)
        new_state = built_relu_muscle.integrate(0.01, deriv, muscle_state, g)
        # With zero activation, force should be at the minimum (min_act * max_iso)
        assert (new_state[:, 3, :] >= 0).all()

    def test_force_max_at_full_activation(self, built_relu_muscle):
        g = make_geometry_state(1, 0.0, 2)
        muscle_state = th.ones(1, 4, 2)  # activation=1 in slot 0
        deriv = built_relu_muscle.ode(th.ones(1, 1, 2), muscle_state)
        new_state = built_relu_muscle.integrate(0.01, deriv, muscle_state, g)
        # force = 1.0 * max_iso_force
        expected_m0 = built_relu_muscle.max_iso_force[0, 0, 0].item()
        assert new_state[0, 3, 0].item() == pytest.approx(expected_m0, rel=1e-3)

    def test_integrate_output_shape(self, built_relu_muscle):
        g = make_geometry_state(3, 0.0, 2)
        s = th.zeros(3, 4, 2)
        d = built_relu_muscle.ode(th.zeros(3, 1, 2), s)
        out = built_relu_muscle.integrate(0.01, d, s, g)
        assert out.shape == (3, 4, 2)

    def test_activation_stays_in_bounds_over_many_steps(self, built_relu_muscle):
        g = make_geometry_state(1, 0.0, 2)
        s = built_relu_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 2)
        for _ in range(200):
            d = built_relu_muscle.ode(action, s)
            s = built_relu_muscle.integrate(0.01, d, s, g)
        activation = s[:, 0, :]
        assert (activation >= 0).all()
        assert (activation <= 1.0).all()

    def test_no_nan_over_many_steps(self, built_relu_muscle):
        g = make_geometry_state(1, 0.0, 2)
        s = built_relu_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 2) * 0.7
        for _ in range(200):
            d = built_relu_muscle.ode(action, s)
            s = built_relu_muscle.integrate(0.01, d, s, g)
        assert not th.isnan(s).any()
        assert not th.isinf(s).any()

    def test_different_muscles_independent(self, built_relu_muscle):
        # Muscle 0 excited, muscle 1 at rest — their activations evolve independently
        g = make_geometry_state(1, 0.0, 2)
        s = th.zeros(1, 4, 2)
        action = th.tensor([[[1.0, 0.0]]])
        for _ in range(50):
            d = built_relu_muscle.ode(action, s)
            s = built_relu_muscle.integrate(0.01, d, s, g)
        assert s[0, 0, 0].item() > s[0, 0, 1].item()  # muscle 0 more activated


# =============================================================================
# RigidTendonHillMuscle
# =============================================================================

class TestRigidTendonHillMuscle:

    def test_state_names(self):
        m = RigidTendonHillMuscle()
        assert m.state_name == [
            'activation', 'muscle length', 'muscle velocity',
            'force-length PE', 'force-length CE', 'force-velocity CE', 'force'
        ]

    def test_state_dim(self):
        m = RigidTendonHillMuscle()
        assert m.state_dim == 7

    def test_build_sets_n_muscles(self, built_rigid_tendon_muscle):
        assert built_rigid_tendon_muscle.n_muscles == 1

    def test_build_sets_l0_ce(self, built_rigid_tendon_muscle):
        assert built_rigid_tendon_muscle.l0_ce.item() == pytest.approx(0.1)

    def test_build_sets_l0_se(self, built_rigid_tendon_muscle):
        assert built_rigid_tendon_muscle.l0_se.item() == pytest.approx(0.0)

    def test_build_sets_l0_pe(self, built_rigid_tendon_muscle):
        # l0_pe = normalized_slack * l0_ce = 1.4 * 0.1 = 0.14
        assert built_rigid_tendon_muscle.l0_pe.item() == pytest.approx(0.14)

    def test_build_sets_vmax(self, built_rigid_tendon_muscle):
        # vmax = 10 * l0_ce = 10 * 0.1 = 1.0
        assert built_rigid_tendon_muscle.vmax.item() == pytest.approx(1.0)

    def test_initial_state_shape(self, built_rigid_tendon_muscle):
        g = make_geometry_state(1, 0.1, 1)
        s = built_rigid_tendon_muscle.get_initial_muscle_state(1, g)
        assert s.shape == (1, 7, 1)

    def test_flce_peak_at_optimal_length(self, built_rigid_tendon_muscle):
        # With tendon_length=0, musculotendon_len == muscle_len.
        # At muscle_len = l0_ce = 0.1, muscle_len_n = 1.0, flce should be 1.0.
        l0_ce = built_rigid_tendon_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)
        s = th.zeros(1, 7, 1)
        d = built_rigid_tendon_muscle.ode(th.ones(1, 1, 1), s)
        out = built_rigid_tendon_muscle.integrate(0.01, d, s, g)
        flce = out[0, 4, 0].item()
        assert flce == pytest.approx(1.0, abs=1e-5)

    def test_flpe_zero_below_slack_length(self, built_rigid_tendon_muscle):
        # muscle_len = l0_ce (optimal, below slack l0_pe = 0.14) → flpe = 0
        l0_ce = built_rigid_tendon_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)
        s = th.zeros(1, 7, 1)
        d = built_rigid_tendon_muscle.ode(th.ones(1, 1, 1), s)
        out = built_rigid_tendon_muscle.integrate(0.01, d, s, g)
        flpe = out[0, 3, 0].item()
        assert flpe == pytest.approx(0.0, abs=1e-6)

    def test_flpe_positive_above_slack_length(self, built_rigid_tendon_muscle):
        # muscle_len = 0.16 > l0_pe = 0.14 → flpe > 0
        g = make_geometry_state(1, 0.16, 1)
        s = th.zeros(1, 7, 1)
        d = built_rigid_tendon_muscle.ode(th.ones(1, 1, 1), s)
        out = built_rigid_tendon_muscle.integrate(0.01, d, s, g)
        assert out[0, 3, 0].item() > 0.0

    def test_force_non_negative(self, built_rigid_tendon_muscle):
        lengths = [0.0, 0.05, 0.1, 0.14, 0.2]
        for length in lengths:
            g = make_geometry_state(1, length, 1)
            s = th.zeros(1, 7, 1)
            d = built_rigid_tendon_muscle.ode(th.ones(1, 1, 1), s)
            out = built_rigid_tendon_muscle.integrate(0.01, d, s, g)
            assert out[0, 6, 0].item() >= 0.0, f"negative force at length={length}"

    def test_fvce_decreases_with_shortening_velocity(self, built_rigid_tendon_muscle):
        # Shortening (vel < 0) reduces force; lengthening (vel > 0) increases it.
        l0_ce = built_rigid_tendon_muscle.l0_ce.item()
        g_iso = make_geometry_state(1, l0_ce, 1, vel=0.0)
        g_sho = make_geometry_state(1, l0_ce, 1, vel=-0.5)
        g_len = make_geometry_state(1, l0_ce, 1, vel=0.5)
        s = th.zeros(1, 7, 1)
        s[:, 0, :] = 1.0  # full activation
        d = built_rigid_tendon_muscle.ode(th.ones(1, 1, 1), s)
        out_iso = built_rigid_tendon_muscle.integrate(0.01, d, s, g_iso)
        out_sho = built_rigid_tendon_muscle.integrate(0.01, d, s, g_sho)
        out_len = built_rigid_tendon_muscle.integrate(0.01, d, s, g_len)
        fvce_iso = out_iso[0, 5, 0].item()
        fvce_sho = out_sho[0, 5, 0].item()
        fvce_len = out_len[0, 5, 0].item()
        assert fvce_sho < fvce_iso
        assert fvce_len > fvce_iso

    def test_activation_in_bounds_over_many_steps(self, built_rigid_tendon_muscle):
        l0_ce = built_rigid_tendon_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)
        s = built_rigid_tendon_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 1) * 0.8
        for _ in range(200):
            d = built_rigid_tendon_muscle.ode(action, s)
            s = built_rigid_tendon_muscle.integrate(0.01, d, s, g)
        activation = s[0, 0, 0].item()
        assert built_rigid_tendon_muscle.min_activation.item() <= activation <= 1.0

    def test_no_nan_over_many_steps(self, built_rigid_tendon_muscle):
        l0_ce = built_rigid_tendon_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)
        s = built_rigid_tendon_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 1) * 0.5
        for _ in range(200):
            d = built_rigid_tendon_muscle.ode(action, s)
            s = built_rigid_tendon_muscle.integrate(0.01, d, s, g)
        assert not th.isnan(s).any()
        assert not th.isinf(s).any()

    def test_multi_muscle_build(self):
        m = RigidTendonHillMuscle()
        m.build(
            timestep=0.01,
            max_isometric_force=[100.0, 200.0, 300.0],
            tendon_length=[0.0, 0.0, 0.0],
            optimal_muscle_length=[0.1, 0.12, 0.08],
            normalized_slack_muscle_length=[1.4, 1.4, 1.4],
        )
        assert m.n_muscles == 3
        assert m.max_iso_force.shape == (1, 1, 3)

    def test_multi_muscle_independent_forces(self):
        m = RigidTendonHillMuscle()
        m.build(
            timestep=0.01,
            max_isometric_force=[100.0, 200.0],
            tendon_length=[0.0, 0.0],
            optimal_muscle_length=[0.1, 0.1],
            normalized_slack_muscle_length=[1.4, 1.4],
        )
        g = make_geometry_state(1, 0.1, 2)
        s = th.zeros(1, 7, 2)
        d = m.ode(th.ones(1, 1, 2), s)
        out = m.integrate(0.01, d, s, g)
        # Muscle 1 has 2× the max force → its force output at full activation should be ~2×
        assert out[0, 6, 1].item() > out[0, 6, 0].item()


# =============================================================================
# RigidTendonHillMuscleThelen
# =============================================================================

class TestRigidTendonHillMuscleThelen:

    def test_state_names(self):
        m = RigidTendonHillMuscleThelen()
        assert m.state_name == [
            'activation', 'muscle length', 'muscle velocity',
            'force-length PE', 'force-length CE', 'force-velocity CE', 'force'
        ]

    def test_state_dim(self):
        assert RigidTendonHillMuscleThelen().state_dim == 7

    def test_build_sets_l0_ce(self, built_thelen_muscle):
        assert built_thelen_muscle.l0_ce.item() == pytest.approx(0.1)

    def test_flce_equals_one_at_optimal_length(self, built_thelen_muscle):
        # flce = exp(0) = 1.0 when muscle_len / l0_ce == 1
        l0_ce = built_thelen_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)
        s = th.zeros(1, 7, 1)
        d = built_thelen_muscle.ode(th.ones(1, 1, 1), s)
        out = built_thelen_muscle.integrate(0.01, d, s, g)
        assert out[0, 4, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_flce_less_than_one_away_from_optimal(self, built_thelen_muscle):
        for length in [0.05, 0.07, 0.13, 0.16]:
            g = make_geometry_state(1, length, 1)
            s = th.zeros(1, 7, 1)
            d = built_thelen_muscle.ode(th.ones(1, 1, 1), s)
            out = built_thelen_muscle.integrate(0.01, d, s, g)
            assert out[0, 4, 0].item() <= 1.0 + 1e-5

    def test_flpe_zero_below_slack(self, built_thelen_muscle):
        l0_ce = built_thelen_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)  # l0_ce < l0_pe = 1.4 * l0_ce
        s = th.zeros(1, 7, 1)
        d = built_thelen_muscle.ode(th.ones(1, 1, 1), s)
        out = built_thelen_muscle.integrate(0.01, d, s, g)
        assert out[0, 3, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_flpe_positive_above_slack(self, built_thelen_muscle):
        # l0_pe = 1.4 * 0.1 = 0.14; stretch to 0.16
        g = make_geometry_state(1, 0.16, 1)
        s = th.zeros(1, 7, 1)
        d = built_thelen_muscle.ode(th.ones(1, 1, 1), s)
        out = built_thelen_muscle.integrate(0.01, d, s, g)
        assert out[0, 3, 0].item() > 0.0

    def test_force_non_negative(self, built_thelen_muscle):
        l0_ce = built_thelen_muscle.l0_ce.item()
        for length in [0.0, 0.05, l0_ce, 0.14, 0.20]:
            g = make_geometry_state(1, length, 1)
            s = th.zeros(1, 7, 1)
            d = built_thelen_muscle.ode(th.ones(1, 1, 1), s)
            out = built_thelen_muscle.integrate(0.01, d, s, g)
            assert out[0, 6, 0].item() >= 0.0, f"negative force at length={length}"

    def test_no_nan_over_many_steps(self, built_thelen_muscle):
        l0_ce = built_thelen_muscle.l0_ce.item()
        g = make_geometry_state(1, l0_ce, 1)
        s = built_thelen_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 1) * 0.6
        for _ in range(200):
            d = built_thelen_muscle.ode(action, s)
            s = built_thelen_muscle.integrate(0.01, d, s, g)
        assert not th.isnan(s).any()
        assert not th.isinf(s).any()

    def test_multi_muscle_independent(self):
        m = RigidTendonHillMuscleThelen()
        m.build(
            timestep=0.01,
            max_isometric_force=[100.0, 50.0],
            tendon_length=[0.0, 0.0],
            optimal_muscle_length=[0.1, 0.1],
            normalized_slack_muscle_length=[1.4, 1.4],
        )
        g = make_geometry_state(1, 0.1, 2)
        s = th.zeros(1, 7, 2)
        # Activate only muscle 0
        d = m.ode(th.tensor([[[1.0, 0.0]]]), s)
        out = m.integrate(0.01, d, s, g)
        # Muscle 0 should have higher activation than muscle 1
        assert out[0, 0, 0].item() > out[0, 0, 1].item()


# =============================================================================
# MujocoHillMuscle
# =============================================================================

class TestMujocoHillMuscle:

    def test_state_names(self):
        m = MujocoHillMuscle()
        assert m.state_name == [
            'activation', 'muscle length', 'muscle velocity',
            'force-length PE', 'force-length CE', 'force-velocity CE', 'force'
        ]

    def test_state_dim(self):
        assert MujocoHillMuscle().state_dim == 7

    def test_build_sets_n_muscles(self, built_mujoco_muscle):
        assert built_mujoco_muscle.n_muscles == 1

    def test_bump_zero_at_lower_boundary(self, built_mujoco_muscle):
        lmin = built_mujoco_muscle.lmin
        L = lmin.clone()
        result = built_mujoco_muscle._bump(L, mid=th.tensor(1.0), lmax=built_mujoco_muscle.lmax)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_bump_zero_at_upper_boundary(self, built_mujoco_muscle):
        lmax = built_mujoco_muscle.lmax
        L = lmax.clone()
        result = built_mujoco_muscle._bump(L, mid=th.tensor(1.0), lmax=built_mujoco_muscle.lmax)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_bump_peak_at_mid(self, built_mujoco_muscle):
        L = th.tensor([[[1.0]]])
        result = built_mujoco_muscle._bump(L, mid=th.tensor(1.0), lmax=built_mujoco_muscle.lmax)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_bump_non_negative(self, built_mujoco_muscle):
        lengths = th.linspace(0.3, 1.8, 100).reshape(100, 1, 1)
        result = built_mujoco_muscle._bump(lengths, mid=th.tensor(1.0), lmax=built_mujoco_muscle.lmax)
        assert (result >= 0).all()

    def test_flce_peak_at_normalized_optimal_length(self, built_mujoco_muscle):
        # At muscle_len_n = 1.0 (optimal), flce should peak at 1.0 (main bump)
        # + small secondary bump (which is 0 at muscle_len_n=1.0 since lmax=0.95 < 1.0)
        l0_ce = built_mujoco_muscle.l0_ce.item()
        l0_se = built_mujoco_muscle.l0_se.item()
        g = make_geometry_state(1, l0_ce + l0_se, 1)
        s = th.zeros(1, 7, 1)
        d = built_mujoco_muscle.ode(th.ones(1, 1, 1), s)
        out = built_mujoco_muscle.integrate(0.01, d, s, g)
        flce = out[0, 4, 0].item()
        assert flce == pytest.approx(1.0, abs=1e-4)

    def test_flce_zero_outside_operating_range(self, built_mujoco_muscle):
        # At muscle_len_n = lmax, the main bump is zero; only secondary bump matters (also ~0)
        l0_ce = built_mujoco_muscle.l0_ce.item()
        l0_se = built_mujoco_muscle.l0_se.item()
        lmax = built_mujoco_muscle.lmax.item()
        g = make_geometry_state(1, lmax * l0_ce + l0_se, 1)
        s = th.zeros(1, 7, 1)
        d = built_mujoco_muscle.ode(th.ones(1, 1, 1), s)
        out = built_mujoco_muscle.integrate(0.01, d, s, g)
        assert out[0, 4, 0].item() == pytest.approx(0.0, abs=1e-4)

    def test_flpe_zero_at_or_below_optimal(self, built_mujoco_muscle):
        l0_ce = built_mujoco_muscle.l0_ce.item()
        l0_se = built_mujoco_muscle.l0_se.item()
        # muscle_len_n = 1.0 ≤ 1.0 → flpe should be 0 (x=0 in the formula)
        g = make_geometry_state(1, l0_ce + l0_se, 1)
        s = th.zeros(1, 7, 1)
        d = built_mujoco_muscle.ode(th.zeros(1, 1, 1), s)
        out = built_mujoco_muscle.integrate(0.01, d, s, g)
        assert out[0, 3, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_force_non_negative(self, built_mujoco_muscle):
        l0_ce = built_mujoco_muscle.l0_ce.item()
        l0_se = built_mujoco_muscle.l0_se.item()
        for length in [0.001 + l0_se, 0.05 + l0_se, l0_ce + l0_se, 0.15 + l0_se]:
            g = make_geometry_state(1, length, 1)
            s = th.zeros(1, 7, 1)
            d = built_mujoco_muscle.ode(th.ones(1, 1, 1), s)
            out = built_mujoco_muscle.integrate(0.01, d, s, g)
            assert out[0, 6, 0].item() >= 0.0, f"negative force at length={length}"

    def test_no_nan_over_many_steps(self, built_mujoco_muscle):
        l0_ce = built_mujoco_muscle.l0_ce.item()
        l0_se = built_mujoco_muscle.l0_se.item()
        g = make_geometry_state(1, l0_ce + l0_se, 1)
        s = built_mujoco_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 1) * 0.5
        for _ in range(200):
            d = built_mujoco_muscle.ode(action, s)
            s = built_mujoco_muscle.integrate(0.01, d, s, g)
        assert not th.isnan(s).any()
        assert not th.isinf(s).any()


# =============================================================================
# CompliantTendonHillMuscle
# =============================================================================

class TestCompliantTendonHillMuscle:

    def test_state_names(self):
        m = CompliantTendonHillMuscle()
        assert m.state_name == [
            'activation', 'muscle length', 'muscle velocity',
            'force-length PE', 'force-length SE', 'active force', 'force'
        ]

    def test_state_dim(self):
        assert CompliantTendonHillMuscle().state_dim == 7

    def test_build_sets_n_muscles(self, built_compliant_muscle):
        assert built_compliant_muscle.n_muscles == 1

    def test_ode_returns_two_components(self, built_compliant_muscle):
        # CompliantTendon ODE returns [d_activation, normalized_muscle_vel]
        s = built_compliant_muscle.get_initial_muscle_state(
            1, make_geometry_state(1, 0.6, 1)
        )
        d = built_compliant_muscle.ode(th.zeros(1, 1, 1), s)
        assert d.shape[1] == 2

    def test_initial_state_shape(self, built_compliant_muscle):
        g = make_geometry_state(1, 0.6, 1)
        s = built_compliant_muscle.get_initial_muscle_state(1, g)
        assert s.shape == (1, 7, 1)

    def test_force_non_negative(self, built_compliant_muscle):
        g = make_geometry_state(1, 0.6, 1)
        s = built_compliant_muscle.get_initial_muscle_state(1, g)
        for _ in range(20):
            d = built_compliant_muscle.ode(th.ones(1, 1, 1) * 0.5, s)
            s = built_compliant_muscle.integrate(0.0001, d, s, g)
        assert s[0, 6, 0].item() >= 0.0

    def test_tendon_force_reported_in_state(self, built_compliant_muscle):
        # slot 4 = flse (tendon force), slot 6 = total force = flse * max_iso_force
        g = make_geometry_state(1, 0.6, 1)
        s = built_compliant_muscle.get_initial_muscle_state(1, g)
        d = built_compliant_muscle.ode(th.ones(1, 1, 1) * 0.5, s)
        out = built_compliant_muscle.integrate(0.0001, d, s, g)
        flse = out[0, 4, 0].item()
        force = out[0, 6, 0].item()
        max_iso = built_compliant_muscle.max_iso_force.item()
        assert force == pytest.approx(flse * max_iso, rel=1e-4)

    def test_no_nan_over_many_steps(self, built_compliant_muscle):
        g = make_geometry_state(1, 0.6, 1)
        s = built_compliant_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 1) * 0.4
        for _ in range(500):
            d = built_compliant_muscle.ode(action, s)
            s = built_compliant_muscle.integrate(0.0001, d, s, g)
        assert not th.isnan(s).any()
        assert not th.isinf(s).any()

    def test_activation_in_bounds_over_many_steps(self, built_compliant_muscle):
        g = make_geometry_state(1, 0.6, 1)
        s = built_compliant_muscle.get_initial_muscle_state(1, g)
        action = th.ones(1, 1, 1) * 0.8
        for _ in range(500):
            d = built_compliant_muscle.ode(action, s)
            s = built_compliant_muscle.integrate(0.0001, d, s, g)
        activation = s[0, 0, 0].item()
        assert built_compliant_muscle.min_activation.item() <= activation <= 1.0
