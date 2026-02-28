# MotorNet

**MotorNet** is a Python package for training neural networks to produce biomechanically realistic motor behaviour.
It provides a differentiable simulation framework with realistic muscle models, skeletal dynamics, and training
utilities, enabling researchers to study motor control through computational modelling.

## Key Features

- **Biomechanical Realism**: Multiple muscle types (rigid tendon, compliant tendon, Hill-type, MuJoCo Hill),
  realistic skeletal dynamics, and online moment arm computation from user-defined muscle paths.
- **High Performance (JAX backend)**: XLA compilation, automatic vectorization via `vmap`, efficient trajectory
  rollouts with `lax.scan`, and end-to-end differentiability through the full simulation.
- **Flexible API**: Easily define custom skeletons, muscle configurations, environments, and policies.
  Gymnasium-compatible environment interface for standard RL workflows.
- **Open Source**: GPLv3 licensed. Contributions and bug reports are welcome.

## Installation

### From source (recommended)

```bash
pip install git+https://github.com/OlivierCodol/MotorNet.git
```

### Requirements

- Python >= 3.10
- [JAX](https://github.com/jax-ml/jax) and JAXlib
- [Equinox](https://github.com/patrick-kidger/equinox) (functional neural networks for JAX)
- [Optax](https://github.com/google-deepmind/optax) (gradient-based optimizers)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Gymnasium](https://gymnasium.farama.org/)

For GPU support, install JAX with CUDA:
```bash
pip install jax[cuda12_pip] jaxlib
```

## Performance (JAX vs PyTorch)

Benchmarked on Apple M4 Max (CPU) with the Arm26 effector, 200-step episodes, GRU policy (hidden=256),
batch size 128:

| Metric | PyTorch | JAX | Speedup |
|--------|---------|-----|---------|
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

At hidden=256 the GRU dominates total time in both frameworks. JAX achieves ~20x on the physics
through `lax.scan` + XLA fusion, while the GRU speedup is limited by BLAS throughput for the
sequential matmuls. On GPU (CUDA or Apple MPS), speedups are expected to be significantly larger.

## Quick Start

```python
import jax
from motornet_jax import Arm26, RandomTargetReach, GRUPolicy

# Build an effector (arm with 6 muscles)
effector = Arm26()

# Wrap it in an environment
env = RandomTargetReach(effector, max_ep_duration=1.0)

# Create a policy network
key = jax.random.PRNGKey(0)
policy = GRUPolicy(obs_dim=env.observation_dim, action_dim=env.action_dim, hidden_size=128, key=key)

# Run a single episode
env_state, obs, info = env.reset(key, batch_size=32)
hidden = policy.init_hidden(batch_size=32)

for t in range(100):
    action, hidden = policy(obs, hidden)
    env_state, obs, reward, terminated, truncated, info = env.step(env_state, action)
```

## Architecture

```
motornet_jax/
├── types.py          # State types (JointState, MuscleState, EffectorState, ...)
├── skeleton/         # Skeleton models (PointMass, TwoDofArm)
├── muscle/           # Muscle models (RigidTendon, CompliantTendon, ReLU, Thelen, MujocoHill)
├── effector/         # Muscle-skeleton coupling (Effector, Arm26)
├── environment/      # Gymnasium-compatible environments (RandomTargetReach, CenterOutReach, Tracking)
├── policy/           # Neural network policies (GRU, MLP, ModularGRU)
├── integration/      # Numerical integrators (Euler, RK4)
├── training/         # Training utilities and losses
├── plotor.py         # Visualization tools
└── tests/            # Test suite
```

## Tutorials

Example notebooks are available in the [`examples/jax/`](examples/jax/) directory:

0. **Muscle Demo** — Explore muscle model properties
1. **Build Effector** — Construct custom effectors with arbitrary muscle configurations
2. **States and Simulation** — Understand state representations and run simulations
3. **Training** — Train a policy network to perform reaching movements
4. **Environment Training** — Full environment-based training pipeline

## Running Tests

```bash
pytest motornet_jax/tests/ -v
```

## Legacy Backends

Previous implementations using PyTorch (`motornet/`) and TensorFlow (`motornet_tf/`) are included in this
repository for reference but are no longer actively developed.

## License

GNU General Public License v3 (GPLv3). See [LICENSE](LICENSE) for details.

## Citation

If you use MotorNet in your research, please cite the original paper.
