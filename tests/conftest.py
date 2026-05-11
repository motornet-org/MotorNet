import numpy as np
import pytest
import torch

from motornet.muscle import (
    CompliantTendonHillMuscle,
    MujocoHillMuscle,
    ReluMuscle,
    RigidTendonHillMuscle,
    RigidTendonHillMuscleThelen,
)
from motornet.skeleton import PointMass
from motornet.effector import (
    CompliantTendonArm26,
    ReluPointMass24,
    RigidTendonArm26,
)
from motornet.environment import Environment


# ---------------------------------------------------------------------------
# Muscle fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def built_relu_muscle():
    """ReluMuscle built with two muscles directly."""
    m = ReluMuscle()
    m.build(timestep=0.01, max_isometric_force=[100.0, 200.0])
    return m


@pytest.fixture
def built_rigid_tendon_muscle():
    """RigidTendonHillMuscle with a single muscle, tendon_length=0 so muscle len == musculotendon len."""
    m = RigidTendonHillMuscle()
    m.build(
        timestep=0.01,
        max_isometric_force=[100.0],
        tendon_length=[0.0],
        optimal_muscle_length=[0.1],
        normalized_slack_muscle_length=[1.4],
    )
    return m


@pytest.fixture
def built_thelen_muscle():
    """RigidTendonHillMuscleThelen with a single muscle, tendon_length=0."""
    m = RigidTendonHillMuscleThelen()
    m.build(
        timestep=0.01,
        max_isometric_force=[100.0],
        tendon_length=[0.0],
        optimal_muscle_length=[0.1],
        normalized_slack_muscle_length=[1.4],
    )
    return m


@pytest.fixture
def built_mujoco_muscle():
    """MujocoHillMuscle with a single muscle, tendon_length=0."""
    m = MujocoHillMuscle()
    m.build(
        timestep=0.01,
        max_isometric_force=[100.0],
        tendon_length=[0.0],
        optimal_muscle_length=[0.1],
        normalized_slack_muscle_length=[1.3],
        lmin=[0.5],
        lmax=[1.6],
        vmax=[1.5],
        fvmax=[1.2],
    )
    return m


@pytest.fixture
def built_compliant_muscle():
    """CompliantTendonHillMuscle with a single muscle. Uses a long tendon to isolate force curves."""
    m = CompliantTendonHillMuscle()
    m.build(
        timestep=0.0001,
        max_isometric_force=[100.0],
        tendon_length=[0.5],
        optimal_muscle_length=[0.1],
        normalized_slack_muscle_length=[1.4],
    )
    return m


# ---------------------------------------------------------------------------
# Effector fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def relu_point_mass():
    """ReluPointMass24 ready for use."""
    return ReluPointMass24()


@pytest.fixture
def thelen_arm26():
    """RigidTendonArm26 with RigidTendonHillMuscleThelen."""
    return RigidTendonArm26(muscle=RigidTendonHillMuscleThelen())


@pytest.fixture
def compliant_arm26():
    """CompliantTendonArm26 (uses rk4 integration by default)."""
    return CompliantTendonArm26()


# ---------------------------------------------------------------------------
# Skeleton fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def point_mass_2d():
    """PointMass in 2D, not yet built (build() is called by Effector)."""
    return PointMass(space_dim=2)


# ---------------------------------------------------------------------------
# Environment fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base_env():
    """Environment wrapping a ReluPointMass24, with no delays beyond one step."""
    effector = ReluPointMass24()
    return Environment(effector=effector)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def make_geometry_state(batch_size, musculotendon_len, n_muscles, dof=2, vel=0.0):
    """Build a geometry_state tensor with given musculotendon lengths (scalar or list)."""
    lens = np.broadcast_to(np.array(musculotendon_len, dtype=np.float32), (batch_size, 1, n_muscles))
    vels = np.full((batch_size, 1, n_muscles), vel, dtype=np.float32)
    moments = np.zeros((batch_size, dof, n_muscles), dtype=np.float32)
    return torch.tensor(np.concatenate([lens, vels, moments], axis=1))
