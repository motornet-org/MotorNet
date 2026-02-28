# MotorNet-JAX

A high-performance JAX rewrite of MotorNet for neuromechanical simulation.

## Key Features

- **XLA Compilation**: Entire simulation graph compiled to optimized machine code
- **Automatic Vectorization**: `vmap` for parallel environment simulation
- **Efficient Integration**: `lax.scan` for trajectory rollouts without Python loops
- **End-to-End Differentiability**: Gradients flow through the full simulation

## Installation

```bash
pip install jax jaxlib equinox optax

# For GPU support:
# pip install jax[cuda12_pip] jaxlib equinox optax
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from motornet_jax import TwoDofArm, GRUPolicy
from motornet_jax.types import JointState

# Create skeleton
skeleton = TwoDofArm()
params = skeleton.get_params()

# Test forward kinematics
joint_state = JointState(
    position=jnp.array([[0.5, 0.5]]),
    velocity=jnp.zeros((1, 2)),
)

elbow, hand, _, _ = TwoDofArm.forward_kinematics(joint_state, params)
print(f"Hand position: {hand}")

# Simulate with lax.scan for efficiency
@jax.jit
def simulate(initial_pos, torques, n_steps, dt):
    state = JointState(
        position=initial_pos[None, :],
        velocity=jnp.zeros((1, 2)),
    )
    endpoint_load = jnp.zeros((1, 2))

    def step(state, _):
        _, acc = TwoDofArm.ode(state, torques, endpoint_load, params)
        new_state = TwoDofArm.integrate(state, acc, dt, params)
        return new_state, new_state.position[0]

    final, trajectory = jax.lax.scan(step, state, None, length=n_steps)
    return trajectory

# Run simulation
trajectory = simulate(
    jnp.array([0.5, 0.5]),
    jnp.array([[1.0, -0.5]]),
    100,
    0.01
)
```

## Architecture

```
motornet_jax/
├── types.py          # State types (JointState, MuscleState, etc.)
├── skeleton/         # Skeleton models (PointMass, TwoDofArm)
├── muscle/           # Muscle models (RigidTendon, CompliantTendon)
├── effector/         # Muscle-skeleton coupling
├── environment/      # Gym-like environment interface
├── policy/           # Neural network policies (GRU, MLP)
├── integration/      # Numerical integrators
└── training/         # Training utilities with Optax
```

## Performance Comparison

Benchmarked on Apple M4 Max (CPU) with the Arm26 effector, batch size 128, 200-step episodes, GRU policy (hidden=256):

| Metric | PyTorch MotorNet | MotorNet-JAX | Speedup |
|--------|------------------|--------------|---------|
| Single step | ~0.38 ms | ~0.17 ms | **~2.2x** |
| Episode rollout | ~153 ms | ~65 ms | **~2.4x** |
| Training step (fwd+bwd) | ~427 ms | ~252 ms | **~1.7x** |

Speedups increase with smaller policy networks (where Python loop overhead dominates):

| GRU hidden size | Rollout speedup |
|----------------|-----------------|
| 32 | **~10.6x** |
| 64 | **~5.2x** |
| 128 | **~4.4x** |
| 256 | **~2.9x** |

**Component breakdown** (batch=128, hidden=256, 200-step episode rollout):

| Component | PyTorch | JAX | Speedup |
|-----------|---------|-----|---------|
| Physics simulation only | ~47 ms | ~2 ms | **~20x** |
| GRU policy only | ~68 ms | ~18 ms | **~3.5x** |

At hidden=256 the GRU dominates total time. JAX achieves ~20x on the physics via `lax.scan` +
XLA fusion, while the GRU speedup is bounded by BLAS throughput for sequential matmuls.
Speedups are expected to be larger on GPU (CUDA or Apple MPS).

## Key Differences from PyTorch Version

1. **Functional Design**: All functions are pure with explicit parameters
2. **NamedTuple States**: Pytree-compatible state types for JAX transformations
3. **Vectorized Geometry**: Uses `segment_sum` instead of Python loops
4. **Equinox for NNs**: Clean functional neural network library

## Running Tests

```bash
pytest motornet_jax/tests/ -v
```

## Running Examples

```bash
python examples/jax_reaching_example.py
```

## License

Same as MotorNet.
