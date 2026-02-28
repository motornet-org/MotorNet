"""
Benchmark: PyTorch MotorNet vs JAX MotorNet

Compares performance on three tasks:
1. Single effector step
2. Full episode rollout
3. Training iteration (forward + backward)

Note: PyTorch MotorNet uses @th.compile(mode='max-autotune', backend='inductor')
on all muscle and skeleton functions internally, giving it a strong compiled baseline.
JAX achieves its speedup through whole-graph XLA compilation via @jax.jit + lax.scan.
"""

import time
import numpy as np

BATCH_SIZE = 128
N_STEPS = 200
HIDDEN_SIZE = 256
N_WARMUP = 5
N_REPEATS = 50


def benchmark_fn(fn, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """Run a function multiple times, return median time in ms."""
    # Warmup
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return np.median(times), np.std(times)


# ============================================================
# PyTorch benchmarks
# ============================================================
def run_pytorch_benchmarks():
    import torch as th
    from motornet.muscle import RigidTendonHillMuscle
    from motornet.effector import RigidTendonArm26
    from motornet.environment import RandomTargetReach
    from motornet.policy import PolicyGRU

    device = th.device("cpu")

    # --- Setup effector (pre-built Arm26) ---
    effector = RigidTendonArm26(muscle=RigidTendonHillMuscle(), timestep=0.01, n_ministeps=1)
    env = RandomTargetReach(effector, max_ep_duration=2.0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = PolicyGRU(obs_dim, HIDDEN_SIZE, act_dim, device=device)
    policy.eval()

    print(f"  PyTorch obs_dim={obs_dim}, act_dim={act_dim}")

    # --- Benchmark 1: Single step ---
    def single_step():
        env.effector.reset(options={"batch_size": BATCH_SIZE})
        action = th.rand(BATCH_SIZE, act_dim)
        env.effector.step(action)

    med, std = benchmark_fn(single_step)
    print(f"  Single step:       {med:.3f} ms (std={std:.3f})")
    pt_single = med

    # --- Benchmark 2: Episode rollout ---
    def episode_rollout():
        with th.no_grad():
            obs, info = env.reset(options={"batch_size": BATCH_SIZE})
            h = policy.init_hidden(BATCH_SIZE)
            for t in range(N_STEPS):
                obs_t = obs if th.is_tensor(obs) else th.tensor(obs, dtype=th.float32)
                action, h = policy(obs_t, h)
                obs, reward, terminated, truncated, info = env.step(action)

    med, std = benchmark_fn(episode_rollout)
    print(f"  Episode rollout:   {med:.3f} ms (std={std:.3f})")
    pt_rollout = med

    # --- Benchmark 3: Training step (forward + backward) ---
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

    def train_step():
        policy.train()
        obs, info = env.reset(options={"batch_size": BATCH_SIZE})
        h = policy.init_hidden(BATCH_SIZE)

        fingertips = []
        actions = []
        for t in range(N_STEPS):
            obs_t = obs if th.is_tensor(obs) else th.tensor(obs, dtype=th.float32)
            action, h = policy(obs_t, h)
            obs, reward, terminated, truncated, info = env.step(action)
            fp = info["states"]["fingertip"]
            if not th.is_tensor(fp):
                fp = th.tensor(fp, dtype=th.float32)
            fingertips.append(fp)
            actions.append(action)

        # Loss
        fp_stack = th.stack(fingertips)
        act_stack = th.stack(actions)
        target = th.zeros(BATCH_SIZE, 2)
        pos_loss = ((fp_stack - target) ** 2).mean()
        effort_loss = (act_stack ** 2).mean()
        loss = pos_loss + 0.01 * effort_loss

        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    med, std = benchmark_fn(train_step)
    print(f"  Training step:     {med:.3f} ms (std={std:.3f})")
    pt_train = med

    return pt_single, pt_rollout, pt_train


# ============================================================
# JAX benchmarks
# ============================================================
def run_jax_benchmarks(backend=None):
    import os
    if backend:
        os.environ["JAX_PLATFORMS"] = backend
    import jax
    import jax.numpy as jnp
    from jax import random
    import equinox as eqx
    import optax

    from motornet_jax.effector import Arm26
    from motornet_jax.environment import RandomTargetReach
    from motornet_jax.policy import GRUPolicy

    print(f"  Backend: {jax.default_backend()}, Devices: {jax.devices()}")

    # --- Setup ---
    arm = Arm26(dt=0.01, n_ministeps=1)
    env = RandomTargetReach(arm, max_ep_duration=2.0)
    obs_dim = env.observation_dim
    act_dim = env.action_dim

    key = random.PRNGKey(0)
    policy = GRUPolicy(obs_dim=obs_dim, action_dim=act_dim, hidden_size=HIDDEN_SIZE, key=key)

    print(f"  JAX obs_dim={obs_dim}, act_dim={act_dim}")

    # --- Benchmark 1: Single step ---
    @jax.jit
    def jax_single_step(key):
        state = arm.reset(batch_size=BATCH_SIZE, key=key)
        action = jnp.ones((BATCH_SIZE, act_dim)) * 0.3
        endpoint_load = jnp.zeros((BATCH_SIZE, 2))
        joint_load = jnp.zeros((BATCH_SIZE, 2))
        new_state = Arm26.step(state, action, endpoint_load, joint_load, arm.params)
        return new_state

    def single_step():
        nonlocal key
        key, k = random.split(key)
        result = jax_single_step(k)
        jax.block_until_ready(result)

    med, std = benchmark_fn(single_step)
    print(f"  Single step:       {med:.3f} ms (std={std:.3f})")
    jax_single = med

    # --- Benchmark 2: Episode rollout ---
    @jax.jit
    def jax_episode_rollout(key):
        key, reset_key = random.split(key)
        env_state, obs, info = env.reset(reset_key, batch_size=BATCH_SIZE)
        hidden = policy.init_hidden(BATCH_SIZE)

        def step_fn(carry, _):
            env_state, obs, hidden = carry
            action, new_hidden = policy(obs, hidden)
            new_env_state, new_obs, reward, terminated, truncated, info = env.step_training(
                env_state, action
            )
            return (new_env_state, new_obs, new_hidden), info["states"]["fingertip"]

        (final_state, final_obs, final_hidden), fingertips = jax.lax.scan(
            step_fn, (env_state, obs, hidden), None, length=N_STEPS
        )
        return final_state, fingertips

    def episode_rollout():
        nonlocal key
        key, k = random.split(key)
        result = jax_episode_rollout(k)
        jax.block_until_ready(result)

    med, std = benchmark_fn(episode_rollout)
    print(f"  Episode rollout:   {med:.3f} ms (std={std:.3f})")
    jax_rollout = med

    # --- Benchmark 3: Training step (forward + backward) ---
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-3),
    )
    policy_arrays, policy_static = eqx.partition(policy, eqx.is_array)
    opt_state = optimizer.init(policy_arrays)

    @jax.jit
    def jax_train_step(policy_arrays, opt_state, key):
        key, reset_key = random.split(key)

        def loss_fn(p_arrays):
            pol = eqx.combine(p_arrays, policy_static)
            env_state, obs, info = env.reset(reset_key, batch_size=BATCH_SIZE)
            hidden = pol.init_hidden(BATCH_SIZE)

            def step_fn(carry, _):
                env_state, obs, hidden = carry
                action, new_hidden = pol(obs, hidden)
                new_env_state, new_obs, reward, terminated, truncated, info = env.step_training(
                    env_state, action
                )
                return (new_env_state, new_obs, new_hidden), (info["states"]["fingertip"], action)

            (final_state, final_obs, final_hidden), (fingertips, actions) = jax.lax.scan(
                step_fn, (env_state, obs, hidden), None, length=N_STEPS
            )

            target = jnp.zeros((BATCH_SIZE, 2))
            pos_loss = jnp.mean((fingertips - target) ** 2)
            effort_loss = jnp.mean(actions ** 2)
            return pos_loss + 0.01 * effort_loss

        loss, grads = jax.value_and_grad(loss_fn)(policy_arrays)
        updates, new_opt_state = optimizer.update(grads, opt_state, policy_arrays)
        new_arrays = optax.apply_updates(policy_arrays, updates)
        return new_arrays, new_opt_state, loss

    def train_step():
        nonlocal key, policy_arrays, opt_state
        key, k = random.split(key)
        policy_arrays, opt_state, loss = jax_train_step(policy_arrays, opt_state, k)
        jax.block_until_ready(loss)

    med, std = benchmark_fn(train_step)
    print(f"  Training step:     {med:.3f} ms (std={std:.3f})")
    jax_train = med

    return jax_single, jax_rollout, jax_train


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"Benchmark: batch_size={BATCH_SIZE}, n_steps={N_STEPS}, "
          f"hidden={HIDDEN_SIZE}")
    print(f"Warmup={N_WARMUP}, Repeats={N_REPEATS}")
    print("=" * 60)

    print("\n--- PyTorch MotorNet ---")
    pt_single, pt_rollout, pt_train = run_pytorch_benchmarks()

    print("\n--- JAX MotorNet ---")
    jax_single, jax_rollout, jax_train = run_jax_benchmarks()

    print("\n" + "=" * 60)
    print("RESULTS (median times, lower is better)")
    print("=" * 60)
    print(f"{'Metric':<35} {'PyTorch':>10} {'JAX':>10} {'Speedup':>10}")
    print("-" * 65)
    print(f"{'Single step':<35} {pt_single:>8.3f}ms {jax_single:>8.3f}ms {pt_single/jax_single:>8.1f}x")
    print(f"{'Episode rollout':<35} {pt_rollout:>8.3f}ms {jax_rollout:>8.3f}ms {pt_rollout/jax_rollout:>8.1f}x")
    print(f"{'Training step (fwd+bwd)':<35} {pt_train:>8.3f}ms {jax_train:>8.3f}ms {pt_train/jax_train:>8.1f}x")
    print("=" * 65)
