# MotorNet-JAX Tutorials

This folder contains Jupyter notebook tutorials for MotorNet-JAX.

## Tutorials

| Notebook | Description |
|----------|-------------|
| `0-muscle-demo.ipynb` | Muscle types (ReLU, Hill-type), force-length/velocity relationships, activation dynamics |
| `1-build-effector.ipynb` | Building effectors, TwoDofArm skeleton, Arm26 model, moment arms, muscle lengths |
| `2-states-and-simulation.ipynb` | State types, running simulations, time-varying actions, external loads, gradients |
| `3-training.ipynb` | Training neural network policies, GRU networks, loss functions, evaluation |

## Running the Tutorials

1. **Install dependencies:**
   ```bash
   pip install jax jaxlib equinox optax matplotlib jupyter
   ```

2. **Navigate to project root and install:**
   ```bash
   cd /path/to/MotorNet-JAM-staging
   pip install -e .
   ```

3. **Launch Jupyter:**
   ```bash
   cd examples/jax
   jupyter notebook
   ```

## Key Differences from PyTorch MotorNet

| Feature | PyTorch MotorNet | MotorNet-JAX |
|---------|------------------|--------------|
| State storage | In object (`effector.states`) | Explicit (passed to functions) |
| Compilation | Eager execution | XLA JIT compilation |
| Batching | Manual | Automatic with `jax.vmap` |
| Loops | Python for-loops | `jax.lax.scan` (no Python overhead) |
| Gradients | `torch.autograd` | `jax.grad` (through entire simulation) |
| Neural networks | `torch.nn` | Equinox |
| Optimization | `torch.optim` | Optax |

## Performance

Benchmarked on Apple M4 Max (CPU) with the Arm26 effector, 200-step episodes, GRU policy (hidden=256),
batch size 128:

| Metric | PyTorch | JAX | Speedup |
|--------|---------|-----|---------|
| Single step | ~0.32 ms | ~0.11 ms | **~3x** |
| Episode rollout | ~126 ms | ~50 ms | **~2.5x** |
| Training step (fwd+bwd) | ~347 ms | ~185 ms | **~2x** |

For GPU acceleration with CUDA, speedups are expected to be significantly larger.

## Quick Example

```python
import jax
import jax.numpy as jnp
from motornet_jax.effector import Arm26
from motornet_jax.types import JointState

# Create arm
arm = Arm26(dt=0.01)
params = arm.get_params()

# Initialize
state = arm.reset(batch_size=64, key=jax.random.PRNGKey(0))

# Simulate with JIT + lax.scan
@jax.jit
def simulate(state, action, n_steps=100):
    def step_fn(state, _):
        new_state = Arm26.step(state, action,
                               jnp.zeros((64, 2)), jnp.zeros((64, 2)), params)
        return new_state, new_state.fingertip
    return jax.lax.scan(step_fn, state, None, length=n_steps)

action = jnp.ones((64, 6)) * 0.3
final_state, trajectory = simulate(state, action)
```
