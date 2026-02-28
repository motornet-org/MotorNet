"""
MotorNet-JAX: A JAX-based neuromechanical simulation framework.

This is a high-performance rewrite of MotorNet using JAX for XLA compilation,
automatic vectorization (vmap), and efficient gradient computation through
physics simulations.
"""

from motornet_jax.types import (
    JointState,
    MuscleState,
    GeometryState,
    EffectorState,
    CartesianState,
)

from motornet_jax.skeleton import (
    PointMass,
    TwoDofArm,
)

from motornet_jax.muscle import (
    RigidTendonMuscle,
    CompliantTendonMuscle,
    ReluMuscle,
    ThelenMuscle,
    MujocoHillMuscle,
)

from motornet_jax.effector import Effector, Arm26, Arm26Compliant

from motornet_jax.environment import (
    Environment,
    RandomTargetReach,
    CenterOutReach,
    TrackingEnv,
)

from motornet_jax.policy import GRUPolicy, MLPPolicy, ModularPolicyGRU, create_modular_policy

from motornet_jax.integration import euler_step, rk4_step

from motornet_jax import plotor

__version__ = "0.1.0"

__all__ = [
    # Types
    "JointState",
    "MuscleState",
    "GeometryState",
    "EffectorState",
    "CartesianState",
    # Skeleton
    "PointMass",
    "TwoDofArm",
    # Muscle
    "RigidTendonMuscle",
    "CompliantTendonMuscle",
    "ReluMuscle",
    "ThelenMuscle",
    "MujocoHillMuscle",
    # Effector
    "Effector",
    "Arm26",
    "Arm26Compliant",
    # Environment
    "Environment",
    "RandomTargetReach",
    "CenterOutReach",
    "TrackingEnv",
    # Policy
    "GRUPolicy",
    "MLPPolicy",
    "ModularPolicyGRU",
    "create_modular_policy",
    # Integration
    "euler_step",
    "rk4_step",
    # Plotting
    "plotor",
]
