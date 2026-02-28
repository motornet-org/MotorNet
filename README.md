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

## Quick Start

```python
import jax
import jax.numpy as jnp
from motornet_jax import Arm26, RandomTargetReach, GRUPolicy

# Build an effector (arm with 6 muscles)
effector = Arm26()

# Wrap it in an environment
env = RandomTargetReach(effector=effector, episode_length=100)

# Create a policy network
key = jax.random.PRNGKey(0)
policy = GRUPolicy(key=key, input_size=env.obs_size, hidden_size=128, output_size=env.action_size)

# Run a single episode
env_state = env.reset(key=key)
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
