"""
Tests for muscle models.
"""

import jax
import jax.numpy as jnp
import pytest

from motornet_jax.muscle import RigidTendonMuscle, CompliantTendonMuscle, ReluMuscle
from motornet_jax.muscle import ThelenMuscle, ThelenMuscleParams, ThelenMuscleState
from motornet_jax.muscle import MujocoHillMuscle, MujocoHillMuscleParams, MujocoHillMuscleState
from motornet_jax.types import MuscleState, GeometryState


class TestRigidTendonMuscle:
    """Tests for the RigidTendonMuscle."""

    @pytest.fixture
    def muscle(self):
        """Create a muscle for testing."""
        return RigidTendonMuscle(
            max_isometric_force=jnp.array([100.0, 150.0]),
            optimal_fiber_length=jnp.array([0.1, 0.12]),
            tendon_slack_length=jnp.array([0.05, 0.06]),
        )

    @pytest.fixture
    def params(self, muscle):
        """Get muscle parameters."""
        return muscle.get_params()

    def test_initialization(self, muscle, params):
        """Test initialization."""
        assert muscle.n_muscles == 2
        assert params.max_isometric_force.shape == (2,)
        assert params.optimal_fiber_length.shape == (2,)

    def test_initial_state(self, muscle, params):
        """Test initial state generation."""
        batch_size = 4
        geometry = GeometryState(
            musculotendon_length=jnp.ones((batch_size, 2)) * 0.15,
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        state = RigidTendonMuscle.get_initial_state(batch_size, geometry, params)

        assert state.activation.shape == (batch_size, 2)
        assert state.fiber_length.shape == (batch_size, 2)
        assert state.fiber_velocity.shape == (batch_size, 2)

        # Activation should be at minimum
        assert jnp.allclose(state.activation, params.min_activation)

    def test_force_computation(self, muscle, params):
        """Test force computation."""
        batch_size = 2
        muscle_state = MuscleState(
            activation=jnp.array([[0.5, 0.5], [1.0, 1.0]]),
            fiber_length=jnp.array([[0.1, 0.12], [0.1, 0.12]]),
            fiber_velocity=jnp.zeros((batch_size, 2)),
        )
        geometry = GeometryState(
            musculotendon_length=jnp.array([[0.15, 0.18], [0.15, 0.18]]),
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        force, flpe, flce, fvce = RigidTendonMuscle.compute_force(
            muscle_state, geometry, params
        )

        assert force.shape == (batch_size, 2)
        assert jnp.all(force >= 0)  # Force should be non-negative

        # Higher activation should produce more force
        assert jnp.all(force[1] >= force[0])

    def test_ode(self, muscle, params):
        """Test ODE computation."""
        batch_size = 2
        excitation = jnp.array([[0.5, 0.8], [0.1, 0.3]])
        muscle_state = MuscleState(
            activation=jnp.array([[0.3, 0.3], [0.5, 0.5]]),
            fiber_length=jnp.ones((batch_size, 2)) * 0.1,
            fiber_velocity=jnp.zeros((batch_size, 2)),
        )

        d_activation = RigidTendonMuscle.ode(excitation, muscle_state, params)

        assert d_activation.shape == (batch_size, 2)

        # Excitation > activation should give positive derivative
        assert d_activation[0, 0] > 0
        assert d_activation[0, 1] > 0

        # Excitation < activation should give negative derivative
        assert d_activation[1, 0] < 0
        assert d_activation[1, 1] < 0


class TestReluMuscle:
    """Tests for the simple ReluMuscle."""

    @pytest.fixture
    def muscle(self):
        return ReluMuscle(
            max_isometric_force=jnp.array([100.0, 200.0]),
        )

    @pytest.fixture
    def params(self, muscle):
        return muscle.get_params()

    def test_force_linear(self, muscle, params):
        """Test that force is linear in activation."""
        muscle_state = MuscleState(
            activation=jnp.array([[0.5, 0.5]]),
            fiber_length=jnp.ones((1, 2)) * 0.1,
            fiber_velocity=jnp.zeros((1, 2)),
        )

        force = ReluMuscle.compute_force(muscle_state, params)

        expected = 0.5 * params.max_isometric_force
        assert jnp.allclose(force[0], expected)



class TestCompliantTendonMuscle:
    """Tests for the CompliantTendonMuscle."""

    @pytest.fixture
    def muscle(self):
        """Create a compliant tendon muscle for testing."""
        return CompliantTendonMuscle(
            max_isometric_force=jnp.array([100.0, 150.0]),
            optimal_fiber_length=jnp.array([0.1, 0.12]),
            tendon_slack_length=jnp.array([0.05, 0.06]),
        )

    @pytest.fixture
    def params(self, muscle):
        """Get muscle parameters."""
        return muscle.get_params()

    def test_initialization(self, muscle, params):
        """Test that parameters are set correctly."""
        assert muscle.n_muscles == 2
        assert params.max_isometric_force.shape == (2,)
        assert params.optimal_fiber_length.shape == (2,)
        assert params.tendon_slack_length.shape == (2,)
        assert params.max_velocity.shape == (2,)
        assert params.passive_slack_length.shape == (2,)
        assert params.k_pe.shape == (2,)

        # Default activation bounds
        assert params.tau_activation == 0.015
        assert params.tau_deactivation == 0.05
        assert params.min_activation == 0.01

        # Derived: passive_slack_length = 1.4 * optimal_fiber_length
        expected_psl = 1.4 * params.optimal_fiber_length
        assert jnp.allclose(params.passive_slack_length, expected_psl)

        # Derived: max_velocity = 10 * optimal_fiber_length
        expected_vmax = 10.0 * params.optimal_fiber_length
        assert jnp.allclose(params.max_velocity, expected_vmax)

    def test_initial_state(self, muscle, params):
        """Test initial state generation from geometry."""
        batch_size = 4
        geometry = GeometryState(
            musculotendon_length=jnp.ones((batch_size, 2)) * 0.20,
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        state = CompliantTendonMuscle.get_initial_state(batch_size, geometry, params)

        assert state.activation.shape == (batch_size, 2)
        assert state.fiber_length.shape == (batch_size, 2)
        assert state.fiber_velocity.shape == (batch_size, 2)

        # Activation should start at min_activation
        assert jnp.allclose(state.activation, params.min_activation)

        # Fiber length should be positive
        assert jnp.all(state.fiber_length > 0)

    def test_initial_state_short_mt(self, muscle, params):
        """Test initial state when musculotendon length is short."""
        batch_size = 2
        geometry = GeometryState(
            musculotendon_length=jnp.ones((batch_size, 2)) * 0.03,
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        state = CompliantTendonMuscle.get_initial_state(batch_size, geometry, params)

        # Fiber length should still be positive (clamped)
        assert jnp.all(state.fiber_length > 0)

    def test_ode_computation(self, muscle, params):
        """Test ODE returns correct shapes and signs."""
        batch_size = 2
        excitation = jnp.array([[0.8, 0.9], [0.1, 0.2]])
        muscle_state = MuscleState(
            activation=jnp.array([[0.3, 0.3], [0.5, 0.5]]),
            fiber_length=jnp.array([[0.10, 0.12], [0.10, 0.12]]),
            fiber_velocity=jnp.zeros((batch_size, 2)),
        )
        geometry = GeometryState(
            musculotendon_length=jnp.ones((batch_size, 2)) * 0.20,
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        d_activation, fiber_velocity_n = CompliantTendonMuscle.ode(
            excitation, muscle_state, geometry, params
        )

        assert d_activation.shape == (batch_size, 2)
        assert fiber_velocity_n.shape == (batch_size, 2)

        # Excitation > activation should give positive d_activation
        assert jnp.all(d_activation[0] > 0)

        # Excitation < activation should give negative d_activation
        assert jnp.all(d_activation[1] < 0)

    def test_force_computation(self, muscle, params):
        """Test force computation produces non-negative forces."""
        batch_size = 2
        muscle_state = MuscleState(
            activation=jnp.array([[0.5, 0.5], [1.0, 1.0]]),
            fiber_length=jnp.array([[0.10, 0.12], [0.10, 0.12]]),
            fiber_velocity=jnp.zeros((batch_size, 2)),
        )
        geometry = GeometryState(
            musculotendon_length=jnp.ones((batch_size, 2)) * 0.20,
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        force, flpe, flse, active_force = CompliantTendonMuscle.compute_force(
            muscle_state, geometry, params
        )

        assert force.shape == (batch_size, 2)
        assert jnp.all(force >= 0)

    def test_integrate(self, muscle, params):
        """Test integration updates state correctly."""
        batch_size = 2
        dt = 0.01
        muscle_state = MuscleState(
            activation=jnp.array([[0.3, 0.3], [0.5, 0.5]]),
            fiber_length=jnp.array([[0.10, 0.12], [0.10, 0.12]]),
            fiber_velocity=jnp.zeros((batch_size, 2)),
        )
        geometry = GeometryState(
            musculotendon_length=jnp.ones((batch_size, 2)) * 0.20,
            musculotendon_velocity=jnp.zeros((batch_size, 2)),
            moment_arm=jnp.zeros((batch_size, 2, 2)),
        )

        d_activation = jnp.array([[0.1, 0.2], [-0.1, -0.2]])
        fiber_velocity_n = jnp.zeros((batch_size, 2))

        new_state = CompliantTendonMuscle.integrate(
            dt, d_activation, fiber_velocity_n, muscle_state, geometry, params
        )

        assert new_state.activation.shape == (batch_size, 2)
        assert new_state.fiber_length.shape == (batch_size, 2)

        # Activation should have changed
        expected_act = muscle_state.activation + d_activation * dt
        expected_act = jnp.clip(expected_act, params.min_activation, 1.0)
        assert jnp.allclose(new_state.activation, expected_act, atol=1e-5)


class TestThelenMuscle:
    """Tests for the ThelenMuscle."""

    @pytest.fixture
    def muscle(self):
        """Create a Thelen muscle for testing."""
        return ThelenMuscle(
            n_muscles=4,
            max_iso_force=500.0,
            optimal_muscle_length=0.08,
            tendon_length=0.05,
        )

    @pytest.fixture
    def params(self, muscle):
        """Get muscle parameters."""
        return muscle.get_params()

    def test_initialization(self, muscle, params):
        """Test that parameters are set correctly."""
        assert muscle.n_muscles == 4
        assert params.n_muscles == 4
        assert params.max_iso_force.shape == (4,)
        assert params.l0_ce.shape == (4,)
        assert params.l0_se.shape == (4,)
        assert params.l0_pe.shape == (4,)
        assert params.vmax.shape == (4,)

        # Check default activation params
        assert params.tau_activation == 0.01
        assert params.tau_deactivation == 0.04
        assert params.min_activation == 0.001

        # vmax should be 10 * l0_ce
        expected_vmax = 10.0 * params.l0_ce
        assert jnp.allclose(params.vmax, expected_vmax)

    def test_initial_state(self, muscle, params):
        """Test initial state generation."""
        batch_size = 3
        # geometry_state: shape (batch, 2, n_muscles)
        geometry = jnp.stack([
            jnp.ones((batch_size, 4)) * 0.13,   # musculotendon length
            jnp.zeros((batch_size, 4)),           # musculotendon velocity
        ], axis=1)

        state = ThelenMuscle.get_initial_state(batch_size, geometry, params)

        assert state.activation.shape == (batch_size, 4)
        assert state.fiber_length.shape == (batch_size, 4)
        assert state.fiber_velocity.shape == (batch_size, 4)
        assert state.force.shape == (batch_size, 4)

        # Activation should start at min_activation
        assert jnp.allclose(state.activation, params.min_activation)

        # Fiber length should be positive
        assert jnp.all(state.fiber_length > 0)

        # Force should be non-negative
        assert jnp.all(state.force >= 0)

    def test_force_gaussian_fl(self, muscle, params):
        """Test that force-length follows Gaussian profile."""
        batch_size = 1
        n = params.n_muscles

        # At optimal length, flce should be ~1.0
        optimal_mt_length = params.l0_ce + params.l0_se
        geometry_optimal = jnp.stack([
            jnp.ones((batch_size, n)) * optimal_mt_length,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        # Far from optimal length, flce should be smaller
        long_mt_length = 1.5 * params.l0_ce + params.l0_se
        geometry_long = jnp.stack([
            jnp.ones((batch_size, n)) * long_mt_length,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        activation = jnp.ones((batch_size, n))
        dummy_state = ThelenMuscleState(
            activation=activation,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )

        state_opt = ThelenMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy_state, geometry_optimal, params
        )
        state_long = ThelenMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy_state, geometry_long, params
        )

        # At optimal length, CE force-length should be near 1.0
        assert jnp.allclose(state_opt.force_length_ce, 1.0, atol=0.01)

        # Away from optimal, CE force-length should be smaller
        assert jnp.all(state_long.force_length_ce < state_opt.force_length_ce)

    def test_passive_force_exponential(self, muscle, params):
        """Test that passive force follows exponential curve."""
        batch_size = 1
        n = params.n_muscles

        # Below passive slack length: no passive force
        short_mt = 0.9 * params.l0_ce + params.l0_se
        geometry_short = jnp.stack([
            jnp.ones((batch_size, n)) * short_mt,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        # Above passive slack length: passive force present
        long_mt = 1.3 * params.l0_ce + params.l0_se
        geometry_long = jnp.stack([
            jnp.ones((batch_size, n)) * long_mt,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        activation = jnp.zeros((batch_size, n)) + params.min_activation
        dummy_state = ThelenMuscleState(
            activation=activation,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )

        state_short = ThelenMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy_state, geometry_short, params
        )
        state_long = ThelenMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy_state, geometry_long, params
        )

        # Short muscle should have zero or near-zero passive force
        assert jnp.allclose(state_short.force_length_pe, 0.0, atol=1e-5)

        # Long muscle should have positive passive force
        assert jnp.all(state_long.force_length_pe > 0)

    def test_activation_ode(self, muscle, params):
        """Test activation dynamics."""
        batch_size = 2
        action = jnp.array([[0.8, 0.9, 0.7, 0.6], [0.1, 0.2, 0.0, 0.1]])
        activation = jnp.array([[0.3, 0.3, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5]])

        d_act = ThelenMuscle.activation_ode(action, activation, params)

        assert d_act.shape == (batch_size, 4)

        # Excitation > activation -> positive derivative
        assert jnp.all(d_act[0] > 0)

        # Excitation < activation -> negative derivative
        assert jnp.all(d_act[1] < 0)

    def test_higher_activation_more_force(self, muscle, params):
        """Test that higher activation produces more force at same length."""
        batch_size = 1
        n = params.n_muscles

        mt_len = params.l0_ce + params.l0_se
        geometry = jnp.stack([
            jnp.ones((batch_size, n)) * mt_len,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        low_act = jnp.full((batch_size, n), 0.3)
        high_act = jnp.full((batch_size, n), 0.9)

        dummy_low = ThelenMuscleState(
            activation=low_act,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )
        dummy_high = ThelenMuscleState(
            activation=high_act,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )

        state_low = ThelenMuscle.integrate(
            0.01, jnp.zeros_like(low_act), dummy_low, geometry, params
        )
        state_high = ThelenMuscle.integrate(
            0.01, jnp.zeros_like(high_act), dummy_high, geometry, params
        )

        assert jnp.all(state_high.force > state_low.force)


class TestMujocoHillMuscle:
    """Tests for the MujocoHillMuscle."""

    @pytest.fixture
    def muscle(self):
        """Create a MuJoCo Hill muscle for testing."""
        return MujocoHillMuscle(
            n_muscles=4,
            max_iso_force=800.0,
            optimal_muscle_length=0.1,
            tendon_length=0.08,
        )

    @pytest.fixture
    def params(self, muscle):
        """Get muscle parameters."""
        return muscle.get_params()

    def test_initialization(self, muscle, params):
        """Test that parameters are set correctly."""
        assert muscle.n_muscles == 4
        assert params.n_muscles == 4
        assert params.max_iso_force.shape == (4,)
        assert params.l0_ce.shape == (4,)
        assert params.l0_se.shape == (4,)
        assert params.l0_pe.shape == (4,)
        assert params.lmin.shape == (4,)
        assert params.lmax.shape == (4,)
        assert params.vmax.shape == (4,)
        assert params.fvmax.shape == (4,)

        # Check default params
        assert params.tau_activation == 0.01
        assert params.tau_deactivation == 0.04
        assert params.min_activation == 0.0

    def test_initial_state(self, muscle, params):
        """Test initial state generation."""
        batch_size = 3
        geometry = jnp.stack([
            jnp.ones((batch_size, 4)) * 0.18,
            jnp.zeros((batch_size, 4)),
        ], axis=1)

        state = MujocoHillMuscle.get_initial_state(batch_size, geometry, params)

        assert state.activation.shape == (batch_size, 4)
        assert state.fiber_length.shape == (batch_size, 4)
        assert state.fiber_velocity.shape == (batch_size, 4)
        assert state.force.shape == (batch_size, 4)
        assert state.force_length_ce.shape == (batch_size, 4)
        assert state.force_length_pe.shape == (batch_size, 4)
        assert state.force_velocity_ce.shape == (batch_size, 4)

        # Fiber length should be positive
        assert jnp.all(state.fiber_length > 0)

        # Force should be non-negative
        assert jnp.all(state.force >= 0)

    def test_force_computation(self, muscle, params):
        """Test force computation at different activations."""
        batch_size = 1
        n = params.n_muscles

        mt_len = params.l0_ce + params.l0_se
        geometry = jnp.stack([
            jnp.ones((batch_size, n)) * mt_len,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        low_act = jnp.full((batch_size, n), 0.2)
        high_act = jnp.full((batch_size, n), 0.8)

        dummy_low = MujocoHillMuscleState(
            activation=low_act,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )
        dummy_high = MujocoHillMuscleState(
            activation=high_act,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )

        state_low = MujocoHillMuscle.integrate(
            0.01, jnp.zeros_like(low_act), dummy_low, geometry, params
        )
        state_high = MujocoHillMuscle.integrate(
            0.01, jnp.zeros_like(high_act), dummy_high, geometry, params
        )

        assert jnp.all(state_low.force >= 0)
        assert jnp.all(state_high.force >= 0)

        # Higher activation should produce more force
        assert jnp.all(state_high.force > state_low.force)

    def test_bump_function_peak(self, muscle, params):
        """Test bump function peaks at midpoint."""
        lmin = jnp.array([0.5])
        mid = jnp.array([1.0])
        lmax = jnp.array([1.6])

        # At peak (mid), bump should be 1.0
        y_mid = MujocoHillMuscle._bump(mid, lmin, mid, lmax)
        assert jnp.allclose(y_mid, 1.0, atol=1e-5)

    def test_bump_function_zero_outside(self, muscle, params):
        """Test bump function is zero outside range."""
        lmin = jnp.array([0.5])
        mid = jnp.array([1.0])
        lmax = jnp.array([1.6])

        # Below lmin
        y_below = MujocoHillMuscle._bump(jnp.array([0.3]), lmin, mid, lmax)
        assert jnp.allclose(y_below, 0.0, atol=1e-6)

        # Above lmax
        y_above = MujocoHillMuscle._bump(jnp.array([2.0]), lmin, mid, lmax)
        assert jnp.allclose(y_above, 0.0, atol=1e-6)

        # At lmin (boundary)
        y_lmin = MujocoHillMuscle._bump(lmin, lmin, mid, lmax)
        assert jnp.allclose(y_lmin, 0.0, atol=1e-6)

        # At lmax (boundary)
        y_lmax = MujocoHillMuscle._bump(lmax, lmin, mid, lmax)
        assert jnp.allclose(y_lmax, 0.0, atol=1e-6)

    def test_bump_function_symmetry(self, muscle, params):
        """Test bump function has expected shape."""
        lmin = jnp.array([0.5])
        mid = jnp.array([1.0])
        lmax = jnp.array([1.5])

        # Symmetric range: left and right half-points should give same value
        left_quarter = jnp.array([0.75])
        right_quarter = jnp.array([1.25])

        y_left = MujocoHillMuscle._bump(left_quarter, lmin, mid, lmax)
        y_right = MujocoHillMuscle._bump(right_quarter, lmin, mid, lmax)

        assert jnp.allclose(y_left, y_right, atol=1e-5)

        # Both should be between 0 and 1
        assert jnp.all(y_left > 0)
        assert jnp.all(y_left <= 1)

    def test_force_velocity_relationship(self, muscle, params):
        """Test force-velocity: shortening reduces force, lengthening increases it."""
        batch_size = 1
        n = params.n_muscles

        mt_len = params.l0_ce + params.l0_se
        activation = jnp.ones((batch_size, n))

        # Zero velocity
        geom_static = jnp.stack([
            jnp.ones((batch_size, n)) * mt_len,
            jnp.zeros((batch_size, n)),
        ], axis=1)

        # Shortening velocity (negative)
        geom_shortening = jnp.stack([
            jnp.ones((batch_size, n)) * mt_len,
            -jnp.ones((batch_size, n)) * 0.05,
        ], axis=1)

        # Lengthening velocity (positive)
        geom_lengthening = jnp.stack([
            jnp.ones((batch_size, n)) * mt_len,
            jnp.ones((batch_size, n)) * 0.05,
        ], axis=1)

        dummy = MujocoHillMuscleState(
            activation=activation,
            fiber_length=jnp.zeros((batch_size, n)),
            fiber_velocity=jnp.zeros((batch_size, n)),
            force_length_pe=jnp.zeros((batch_size, n)),
            force_length_ce=jnp.zeros((batch_size, n)),
            force_velocity_ce=jnp.zeros((batch_size, n)),
            force=jnp.zeros((batch_size, n)),
        )

        state_static = MujocoHillMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy, geom_static, params
        )
        state_short = MujocoHillMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy, geom_shortening, params
        )
        state_long = MujocoHillMuscle.integrate(
            0.01, jnp.zeros_like(activation), dummy, geom_lengthening, params
        )

        # fvce: shortening < static < lengthening
        assert jnp.all(state_short.force_velocity_ce <= state_static.force_velocity_ce)
        assert jnp.all(state_static.force_velocity_ce <= state_long.force_velocity_ce)

    def test_activation_ode(self, muscle, params):
        """Test activation dynamics."""
        batch_size = 2
        action = jnp.array([[0.8, 0.9, 0.7, 0.6], [0.1, 0.2, 0.0, 0.1]])
        activation = jnp.array([[0.3, 0.3, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5]])

        d_act = MujocoHillMuscle.activation_ode(action, activation, params)

        assert d_act.shape == (batch_size, 4)

        # Excitation > activation -> positive derivative
        assert jnp.all(d_act[0] > 0)

        # Excitation < activation -> negative derivative
        assert jnp.all(d_act[1] < 0)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
