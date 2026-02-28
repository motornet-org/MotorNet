"""
Modular GRU-based policy network using Equinox.

Biologically-inspired modular architecture with:
- Configurable module connectivity
- Dale's principle (excitatory/inhibitory constraints)
- Connectivity delays between modules
- Spectral scaling of recurrent weights
- Sparse connectivity masks

Matches PyTorch MotorNet's ModularPolicyGRU API.
"""

from typing import NamedTuple, Tuple, Optional, List
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import numpy as np


class ModularGRUParams(NamedTuple):
    """Configuration for ModularPolicyGRU."""
    input_size: int
    module_size: List[int]
    output_size: int
    hidden_size: int  # sum of module_size
    num_modules: int


class ModularPolicyGRU(eqx.Module):
    """Modular GRU-based recurrent policy with biologically-inspired constraints.

    This implements a GRU network with:
    - Multiple independent modules with configurable sizes
    - Sparse connectivity between modules (binomial sampling)
    - Dale's principle: neurons are either excitatory or inhibitory
    - Connectivity delays between modules
    - Input routing: vision, proprioception, and task inputs to specific modules
    - Output masking: which modules contribute to output
    - Spectral scaling of recurrent weights

    Architecture per GRU step:
        z = sigmoid(Wz @ [x, h_delayed] + bz)  # Update gate
        r = sigmoid(Wr @ [x, h_delayed] + br)  # Reset gate
        h_tilde = activation(Wh @ [x, r*h_delayed] + bh)  # Candidate
        h_new = (1 - z) * h_delayed + z * h_tilde
        y = sigmoid(Y @ h + bY)  # Output
    """

    # GRU parameters
    h0: jnp.ndarray  # Learnable initial hidden state (1, hidden_size)
    Wz: jnp.ndarray  # Update gate weights (hidden_size, input_size + hidden_size)
    bz: jnp.ndarray  # Update gate bias (hidden_size,)
    Wr: jnp.ndarray  # Reset gate weights
    br: jnp.ndarray  # Reset gate bias
    Wh: jnp.ndarray  # Candidate weights
    bh: jnp.ndarray  # Candidate bias
    Y: jnp.ndarray   # Output weights (output_size, hidden_size)
    bY: jnp.ndarray  # Output bias (output_size,)

    # Masks (non-trainable)
    mask_Wz: jnp.ndarray
    mask_Wr: jnp.ndarray
    mask_Wh: jnp.ndarray
    mask_Y: jnp.ndarray

    # Dale's principle masks (optional)
    unittype_W: Optional[jnp.ndarray]  # +1 for excitatory, -1 for inhibitory

    # Static fields
    input_size: int = eqx.field(static=True)
    module_size: tuple = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    num_modules: int = eqx.field(static=True)
    module_dims: tuple = eqx.field(static=True)
    connectivity_delay: tuple = eqx.field(static=True)  # tuple of tuples (int)
    output_delay: int = eqx.field(static=True)
    max_delay: int = eqx.field(static=True)
    use_dale: bool = eqx.field(static=True)
    activation_type: str = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        module_size: List[int],
        output_size: int,
        vision_mask: List[float],
        proprio_mask: List[float],
        task_mask: List[float],
        connectivity_mask: np.ndarray,
        output_mask: List[float],
        vision_dim: List[int],
        proprio_dim: List[int],
        task_dim: List[int],
        connectivity_delay: Optional[np.ndarray] = None,
        spectral_scaling: Optional[float] = None,
        proportion_excitatory: Optional[List[float]] = None,
        input_gain: float = 1.0,
        activation: str = 'tanh',
        output_delay: int = 0,
        last_task_proprio_only: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """Initialize ModularPolicyGRU.

        Args:
            input_size: Total input dimension
            module_size: List of sizes for each module
            output_size: Output dimension (number of muscles)
            vision_mask: Connection probability from vision to each module
            proprio_mask: Connection probability from proprioception to each module
            task_mask: Connection probability from task inputs to each module
            connectivity_mask: Module-to-module connectivity matrix (num_modules, num_modules)
            output_mask: Connection probability from each module to output
            vision_dim: Indices of vision inputs in observation
            proprio_dim: Indices of proprioception inputs in observation
            task_dim: Indices of task inputs in observation
            connectivity_delay: Delay matrix between modules (timesteps)
            spectral_scaling: Scale recurrent weights to this spectral radius
            proportion_excitatory: Fraction of excitatory neurons per module (for Dale's law)
            input_gain: Scaling for input weight initialization
            activation: 'tanh' or 'rect_tanh' (rectified tanh)
            output_delay: Output delay in timesteps
            last_task_proprio_only: Route last task input only to proprio-receiving modules
            key: Random key for initialization
        """
        if key is None:
            key = random.PRNGKey(0)

        # Store configuration
        self.input_size = input_size
        self.module_size = tuple(module_size)
        self.hidden_size = sum(module_size)
        self.output_size = output_size
        self.num_modules = len(module_size)
        self.activation_type = activation
        self.output_delay = output_delay

        # Connectivity delay
        if connectivity_delay is None:
            connectivity_delay = np.zeros((self.num_modules, self.num_modules), dtype=np.int32)
        self.connectivity_delay = tuple(map(tuple, connectivity_delay.astype(np.int32)))
        max_conn_delay = int(np.max(connectivity_delay))
        self.max_delay = max(max_conn_delay, output_delay)

        # Create module dimension indices
        module_dims = []
        current_idx = 0
        for size in module_size:
            module_dims.append(tuple(range(current_idx, current_idx + size)))
            current_idx += size
        self.module_dims = tuple(module_dims)

        # Convert lists to arrays for indexing
        vision_dim = np.array(vision_dim, dtype=np.int32)
        proprio_dim = np.array(proprio_dim, dtype=np.int32)
        task_dim = np.array(task_dim, dtype=np.int32)

        # Validate inputs
        assert len(vision_mask) == self.num_modules
        assert len(proprio_mask) == self.num_modules
        assert len(task_mask) == self.num_modules
        assert connectivity_mask.shape == (self.num_modules, self.num_modules)
        assert len(output_mask) == self.num_modules
        assert len(vision_dim) + len(proprio_dim) + len(task_dim) == input_size

        # Use numpy RNG for mask generation (reproducible)
        key, subkey = random.split(key)
        np_seed = int(random.randint(subkey, (), 0, 2**30))
        rng = np.random.default_rng(seed=np_seed)

        # Create sparsity probability mask for GRU weights
        h_probability_mask = np.zeros((self.hidden_size, input_size + self.hidden_size), dtype=np.float32)

        # Populate mask module-by-module
        for i_mod in range(self.num_modules):
            rows = np.array(self.module_dims[i_mod])

            # Input connections
            if len(vision_dim) > 0:
                h_probability_mask[np.ix_(rows, vision_dim)] = vision_mask[i_mod]
            if len(proprio_dim) > 0:
                h_probability_mask[np.ix_(rows, proprio_dim)] = proprio_mask[i_mod]

            if len(task_dim) > 0:
                if last_task_proprio_only:
                    # General task inputs
                    general_task_dims = task_dim[:-1]
                    if len(general_task_dims) > 0:
                        h_probability_mask[np.ix_(rows, general_task_dims)] = task_mask[i_mod]
                    # Last task input only to proprio modules
                    last_task_dim = task_dim[-1:]
                    h_probability_mask[np.ix_(rows, last_task_dim)] = proprio_mask[i_mod]
                else:
                    h_probability_mask[np.ix_(rows, task_dim)] = task_mask[i_mod]

            # Recurrent connections between modules
            for j_mod in range(self.num_modules):
                p = connectivity_mask[i_mod, j_mod]
                if p > 0:
                    all_presynaptic = np.array(self.module_dims[j_mod])
                    num_projecting = int(np.ceil(p * len(all_presynaptic)))
                    selected = rng.choice(all_presynaptic, size=num_projecting, replace=False)
                    selected_global = selected + input_size
                    h_probability_mask[np.ix_(rows, selected_global)] = 1.0

        # Create output mask
        y_probability_mask = np.zeros((output_size, self.hidden_size), dtype=np.float32)
        for j_mod in range(self.num_modules):
            cols = np.array(self.module_dims[j_mod])
            y_probability_mask[:, cols] = output_mask[j_mod]

        # Sample binary masks
        mask_connectivity = rng.binomial(1, h_probability_mask).astype(np.float32)
        mask_output = rng.binomial(1, y_probability_mask).astype(np.float32)

        # Initialize weights
        key, *subkeys = random.split(key, 8)

        # GRU weights: input part (xavier) + recurrent part (normal)
        Wz_input = jax.nn.initializers.glorot_uniform()(subkeys[0], (self.hidden_size, input_size)) * input_gain
        Wz_recur = random.normal(subkeys[1], (self.hidden_size, self.hidden_size)) / np.sqrt(self.hidden_size)
        Wz = jnp.concatenate([Wz_input, Wz_recur], axis=1)

        Wr_input = jax.nn.initializers.glorot_uniform()(subkeys[2], (self.hidden_size, input_size)) * input_gain
        Wr_recur = random.normal(subkeys[3], (self.hidden_size, self.hidden_size)) / np.sqrt(self.hidden_size)
        Wr = jnp.concatenate([Wr_input, Wr_recur], axis=1)

        Wh_input = jax.nn.initializers.glorot_uniform()(subkeys[4], (self.hidden_size, input_size)) * input_gain
        Wh_recur = random.normal(subkeys[5], (self.hidden_size, self.hidden_size)) / np.sqrt(self.hidden_size)
        Wh = jnp.concatenate([Wh_input, Wh_recur], axis=1)

        # Output weights
        Y = jax.nn.initializers.glorot_uniform()(subkeys[6], (output_size, self.hidden_size))

        # Biases
        bz = jnp.zeros(self.hidden_size)
        br = jnp.zeros(self.hidden_size)
        bh = jnp.zeros(self.hidden_size)
        bY = jnp.full(output_size, -3.0)  # Initialize low for sigmoid

        # Learnable initial hidden state
        h0 = jnp.zeros((1, self.hidden_size))

        # Convert masks to JAX arrays
        self.mask_Wz = jnp.array(mask_connectivity)
        self.mask_Wr = jnp.array(mask_connectivity)
        self.mask_Wh = jnp.array(mask_connectivity)
        self.mask_Y = jnp.array(mask_output)

        # Handle Dale's principle
        self.use_dale = proportion_excitatory is not None
        if self.use_dale:
            assert len(proportion_excitatory) == self.num_modules
            unittype_W = np.zeros((self.hidden_size, self.hidden_size), dtype=np.float32)
            for m in range(self.num_modules):
                indices = np.array(self.module_dims[m])
                unit_types = np.array([1 if rng.random() < proportion_excitatory[m] else -1
                                       for _ in range(len(indices))], dtype=np.float32)
                unittype_W[:, indices] = unit_types[np.newaxis, :]

            # Remove cross-module inhibitory connections
            for i_mod in range(self.num_modules):
                for j_mod in range(self.num_modules):
                    if i_mod != j_mod:
                        pre_indices = np.array(self.module_dims[j_mod])
                        post_indices = np.array(self.module_dims[i_mod])
                        inhib_mask = unittype_W[0, pre_indices] == -1
                        inhib_global = pre_indices[inhib_mask] + input_size
                        if len(inhib_global) > 0:
                            mask_connectivity[np.ix_(post_indices, inhib_global)] = 0

            # Update masks after Dale modification
            self.mask_Wz = jnp.array(mask_connectivity)
            self.mask_Wr = jnp.array(mask_connectivity)
            self.mask_Wh = jnp.array(mask_connectivity)
            self.unittype_W = jnp.array(unittype_W)
        else:
            self.unittype_W = None

        # Apply masks to weights
        self.Wz = Wz * self.mask_Wz
        self.Wr = Wr * self.mask_Wr
        self.Wh = Wh * self.mask_Wh
        self.Y = Y * self.mask_Y
        self.bz = bz
        self.br = br
        self.bh = bh
        self.bY = bY
        self.h0 = h0

        # Apply Dale's principle to initial weights
        if self.use_dale:
            self.Wz, self.Wr, self.Wh = self._enforce_dale_weights(
                self.Wz, self.Wr, self.Wh, self.unittype_W, self.input_size
            )

        # Spectral scaling
        if spectral_scaling is not None:
            Wh_recur = self.Wh[:, input_size:]
            # Compute spectral radius (max eigenvalue magnitude)
            eigvals = jnp.linalg.eigvals(Wh_recur)
            spectral_radius = jnp.max(jnp.abs(eigvals))
            if spectral_radius > 1e-6:
                scale = spectral_scaling / spectral_radius
                Wh_scaled = jnp.concatenate([
                    self.Wh[:, :input_size],
                    Wh_recur * scale
                ], axis=1)
                self.Wh = Wh_scaled

    @staticmethod
    def _enforce_dale_weights(Wz, Wr, Wh, unittype_W, input_size):
        """Enforce Dale's principle: excitatory weights >= 0, inhibitory <= 0."""
        # Split input and recurrent parts
        Wz_i, Wz_r = Wz[:, :input_size], Wz[:, input_size:]
        Wr_i, Wr_r = Wr[:, :input_size], Wr[:, input_size:]
        Wh_i, Wh_r = Wh[:, :input_size], Wh[:, input_size:]

        # Enforce signs based on unit type
        excit_mask = unittype_W == 1
        inhib_mask = unittype_W == -1

        # Excitatory: make positive
        Wz_r = jnp.where(excit_mask & (Wz_r < 0), jnp.abs(Wz_r), Wz_r)
        Wr_r = jnp.where(excit_mask & (Wr_r < 0), jnp.abs(Wr_r), Wr_r)
        Wh_r = jnp.where(excit_mask & (Wh_r < 0), jnp.abs(Wh_r), Wh_r)

        # Inhibitory: make negative
        Wz_r = jnp.where(inhib_mask & (Wz_r > 0), -jnp.abs(Wz_r), Wz_r)
        Wr_r = jnp.where(inhib_mask & (Wr_r > 0), -jnp.abs(Wr_r), Wr_r)
        Wh_r = jnp.where(inhib_mask & (Wh_r > 0), -jnp.abs(Wh_r), Wh_r)

        return (
            jnp.concatenate([Wz_i, Wz_r], axis=1),
            jnp.concatenate([Wr_i, Wr_r], axis=1),
            jnp.concatenate([Wh_i, Wh_r], axis=1),
        )

    def _activation(self, x):
        """Apply activation function."""
        if self.activation_type == 'tanh':
            return jnp.tanh(x)
        elif self.activation_type == 'rect_tanh':
            return jnp.maximum(0.0, jnp.tanh(x))
        else:
            return jnp.tanh(x)

    def __call__(
        self,
        obs: jnp.ndarray,
        hidden: jnp.ndarray,
        h_buffer: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: Observation, shape (batch, input_size) or (input_size,)
            hidden: Hidden state, shape (batch, hidden_size) or (hidden_size,)
            h_buffer: Hidden state history for delays, shape (batch, hidden_size, max_delay+1)

        Returns:
            action: Output action, shape (batch, output_size)
            new_hidden: Updated hidden state
            new_h_buffer: Updated history buffer
        """
        # Handle unbatched input
        batched = obs.ndim > 1
        if not batched:
            obs = obs[None, :]
            hidden = hidden[None, :]
            if h_buffer is not None:
                h_buffer = h_buffer[None, :, :]

        batch_size = obs.shape[0]

        # Initialize buffer if needed
        if h_buffer is None:
            h_buffer = jnp.tile(hidden[:, :, None], (1, 1, self.max_delay + 1))

        # Update buffer: shift and prepend current hidden
        new_h_buffer = jnp.concatenate([hidden[:, :, None], h_buffer[:, :, :-1]], axis=-1)

        # Get delayed hidden states for connectivity
        max_conn_delay = max(max(row) for row in self.connectivity_delay)

        if max_conn_delay > 0:
            # Module-by-module computation with delays
            h_new = jnp.zeros_like(hidden)

            for i in range(self.num_modules):
                rows = jnp.array(self.module_dims[i])

                # Collect delayed hidden states
                h_delayed = jnp.zeros_like(hidden)
                for j in range(self.num_modules):
                    cols = jnp.array(self.module_dims[j])
                    delay = self.connectivity_delay[i][j]
                    h_delayed = h_delayed.at[:, cols].set(new_h_buffer[:, cols, delay])

                # GRU step for this module
                concat = jnp.concatenate([obs, h_delayed], axis=-1)

                z = jax.nn.sigmoid(concat @ self.Wz[rows, :].T + self.bz[rows])
                r = jax.nn.sigmoid(concat @ self.Wr.T + self.br)

                concat_reset = jnp.concatenate([obs, r * h_delayed], axis=-1)
                h_tilda = self._activation(concat_reset @ self.Wh[rows, :].T + self.bh[rows])

                h_mod = (1 - z) * h_delayed[:, rows] + z * h_tilda
                h_new = h_new.at[:, rows].set(h_mod)
        else:
            # Single pass (no delays)
            concat = jnp.concatenate([obs, hidden], axis=-1)

            z = jax.nn.sigmoid(concat @ self.Wz.T + self.bz)
            r = jax.nn.sigmoid(concat @ self.Wr.T + self.br)

            concat_reset = jnp.concatenate([obs, r * hidden], axis=-1)
            h_tilda = self._activation(concat_reset @ self.Wh.T + self.bh)

            h_new = (1 - z) * hidden + z * h_tilda

        # Output layer (with optional delay)
        if self.output_delay > 0:
            h_for_output = new_h_buffer[:, :, self.output_delay - 1]
        else:
            h_for_output = h_new

        action = jax.nn.sigmoid(h_for_output @ self.Y.T + self.bY)

        # Remove batch dim if needed
        if not batched:
            action = action[0]
            h_new = h_new[0]
            new_h_buffer = new_h_buffer[0]

        return action, h_new, new_h_buffer

    def init_hidden(self, batch_size: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize hidden state and buffer.

        Args:
            batch_size: Batch size

        Returns:
            hidden: Initial hidden state, shape (batch, hidden_size)
            h_buffer: Initial history buffer, shape (batch, hidden_size, max_delay+1)
        """
        h0 = self._activation(jnp.tile(self.h0, (batch_size, 1)))
        h_buffer = jnp.tile(h0[:, :, None], (1, 1, self.max_delay + 1))
        return h0, h_buffer

    def apply_masks(self) -> 'ModularPolicyGRU':
        """Return new policy with masks applied to weights.

        Use this after gradient updates to maintain sparsity.
        """
        return eqx.tree_at(
            lambda p: (p.Wz, p.Wr, p.Wh, p.Y),
            self,
            (
                self.Wz * self.mask_Wz,
                self.Wr * self.mask_Wr,
                self.Wh * self.mask_Wh,
                self.Y * self.mask_Y,
            )
        )

    def enforce_dale(self) -> 'ModularPolicyGRU':
        """Return new policy with Dale's principle enforced.

        Use this after gradient updates to maintain E/I constraints.
        """
        if not self.use_dale:
            return self

        Wz, Wr, Wh = self._enforce_dale_weights(
            self.Wz, self.Wr, self.Wh, self.unittype_W, self.input_size
        )
        return eqx.tree_at(
            lambda p: (p.Wz, p.Wr, p.Wh),
            self,
            (Wz, Wr, Wh)
        )


def create_modular_policy(
    input_size: int,
    module_size: List[int],
    output_size: int,
    vision_dim: List[int],
    proprio_dim: List[int],
    task_dim: List[int],
    vision_mask: Optional[List[float]] = None,
    proprio_mask: Optional[List[float]] = None,
    task_mask: Optional[List[float]] = None,
    connectivity_mask: Optional[np.ndarray] = None,
    output_mask: Optional[List[float]] = None,
    key: jax.random.PRNGKey = None,
    **kwargs,
) -> ModularPolicyGRU:
    """Create a modular GRU policy with sensible defaults.

    Args:
        input_size: Total input dimension
        module_size: Size of each module
        output_size: Output dimension
        vision_dim: Indices of vision inputs
        proprio_dim: Indices of proprioception inputs
        task_dim: Indices of task inputs
        vision_mask: Vision connectivity per module (default: all 1.0)
        proprio_mask: Proprio connectivity per module (default: all 1.0)
        task_mask: Task connectivity per module (default: all 1.0)
        connectivity_mask: Module connectivity (default: full connectivity)
        output_mask: Output connectivity per module (default: all 1.0)
        key: Random key
        **kwargs: Additional arguments for ModularPolicyGRU

    Returns:
        Initialized ModularPolicyGRU
    """
    num_modules = len(module_size)

    if vision_mask is None:
        vision_mask = [1.0] * num_modules
    if proprio_mask is None:
        proprio_mask = [1.0] * num_modules
    if task_mask is None:
        task_mask = [1.0] * num_modules
    if connectivity_mask is None:
        connectivity_mask = np.ones((num_modules, num_modules))
    if output_mask is None:
        output_mask = [1.0] * num_modules
    if key is None:
        key = random.PRNGKey(0)

    return ModularPolicyGRU(
        input_size=input_size,
        module_size=module_size,
        output_size=output_size,
        vision_mask=vision_mask,
        proprio_mask=proprio_mask,
        task_mask=task_mask,
        connectivity_mask=connectivity_mask,
        output_mask=output_mask,
        vision_dim=vision_dim,
        proprio_dim=proprio_dim,
        task_dim=task_dim,
        key=key,
        **kwargs,
    )
