import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PolicyGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.gru = th.nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                th.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError

        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x[:, None, :], h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class ModularPolicyGRU(nn.Module):
    def __init__(self, input_size: int, module_size: list, output_size: int,
                 vision_mask: list, proprio_mask: list, task_mask: list,
                 connectivity_mask: np.ndarray, output_mask: list,
                 vision_dim: list, proprio_dim: list, task_dim: list,
                 connectivity_delay: np.ndarray, spectral_scaling=None,
                 proportion_excitatory=None, input_gain=1.,
                 device=th.device("cpu"), random_seed=None, activation='tanh', output_delay=0,
                 cancelation_matrix=None):
        super(ModularPolicyGRU, self).__init__()

        # Store class info
        hidden_size = sum(module_size)
        self.outfun = lambda hidden: th.min(th.ones_like(hidden), th.relu(hidden))
        assert activation == 'tanh' or activation == 'rect_tanh'
        if activation == 'tanh':
            self.activation = lambda hidden: th.tanh(hidden)
            self.d_hidden = lambda hidden: 1 - th.square(hidden)
        elif activation == 'rect_tanh':
            self.activation = lambda hidden: th.max(th.zeros_like(hidden), th.tanh(hidden))
            self.d_hidden = lambda hidden: th.where(th.tanh(hidden) > 0, 1 - th.tanh(hidden) ** 2, th.zeros_like(hidden))
        self.spectral_scaling = spectral_scaling
        self.device = device
        self.num_modules = len(module_size)
        self.input_size = input_size
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.connectivity_delay = connectivity_delay
        self.output_delay = output_delay
        self.max_connectivity_delay = np.max(connectivity_delay)
        self.max_delay = np.max([self.max_connectivity_delay, self.output_delay]).astype(np.integer)
        self.h_buffer = []
        self.counter = 0
        self.cancel_times = None

        if cancelation_matrix is None:
            self.cancelation_matrix = cancelation_matrix
        else:
            self.cancelation_matrix = th.tensor(cancelation_matrix, dtype=th.float32)

        # Set the random seed
        if random_seed:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = np.random.default_rng()

        # Make sure that all sizes check out
        assert len(vision_mask) == self.num_modules
        assert len(proprio_mask) == self.num_modules
        assert len(task_mask) == self.num_modules
        assert connectivity_mask.shape[0] == connectivity_mask.shape[1] == self.num_modules
        assert len(output_mask) == self.num_modules
        assert len(vision_dim) + len(proprio_dim) + len(task_dim) == self.input_size
        if proportion_excitatory:
            assert len(proportion_excitatory) == self.num_modules

        # Initialize all GRU parameters
        # Initial hidden state
        self.h0 = nn.Parameter(th.zeros(1, hidden_size))
        # Update gate
        self.Wz = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=input_gain),
                                       nn.init.normal_(th.Tensor(hidden_size, hidden_size), 0, 1/np.sqrt(hidden_size))), dim=1))
        self.bz = nn.Parameter(th.zeros(hidden_size))
        # Reset gate
        self.Wr = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=input_gain),
                                       nn.init.normal_(th.Tensor(hidden_size, hidden_size), 0, 1/np.sqrt(hidden_size))), dim=1))
        self.br = nn.Parameter(th.zeros(hidden_size))
        # Candidate hidden state
        self.Wh = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=input_gain),
                                       nn.init.normal_(th.Tensor(hidden_size, hidden_size), 0, 1/np.sqrt(hidden_size))), dim=1))
        self.bh = nn.Parameter(th.zeros(hidden_size))

        # Initialize all output parameters
        self.Y = nn.Parameter(nn.init.xavier_uniform_(th.Tensor(output_size, hidden_size), gain=1))
        self.bY = nn.Parameter(nn.init.constant_(th.Tensor(output_size), -3.)) #-6

        # Create indices for indexing modules
        module_dims = []
        for m in range(self.num_modules):
            if m > 0:
                module_dims.append(np.arange(module_size[m]) + module_dims[-1][-1] + 1)
            else:
                module_dims.append(np.arange(module_size[m]))
        self.module_dims = module_dims

        # Create sparsity mask for GRU
        h_probability_mask = np.zeros((hidden_size, input_size + hidden_size), dtype=np.float32)
        i_module = 0
        j_module = 0
        for i in range(self.Wz.shape[0]):
            for m in range(self.num_modules):
                if i in module_dims[m]:
                    i_module = m
            for j in range(self.Wz.shape[1]):
                module_type = 'hidden'
                if j < input_size:
                    if j in vision_dim:
                        module_type = 'vision'
                    elif j in proprio_dim:
                        module_type = 'proprio'
                    elif j in task_dim:
                        module_type = 'task'
                if module_type == 'hidden':
                    for m in range(self.num_modules):
                        if j in (module_dims[m] + input_size):
                            j_module = m
                        h_probability_mask[i, j] = connectivity_mask[i_module, j_module]
                elif module_type == 'vision':
                    h_probability_mask[i, j] = vision_mask[i_module]
                elif module_type == 'proprio':
                    h_probability_mask[i, j] = proprio_mask[i_module]
                elif module_type == 'task':
                    h_probability_mask[i, j] = task_mask[i_module]
                    if j == task_dim[-1]:
                        h_probability_mask[i, j] = proprio_mask[i_module]

        # Create sparsity mask for output
        y_probability_mask = np.zeros((output_size, hidden_size), dtype=np.float32)
        j_module = 0
        for j in range(self.Y.shape[1]):
            for m in range(self.num_modules):
                if j in module_dims[m]:
                    j_module = m
            y_probability_mask[:, j] = output_mask[j_module]

        # Initialize masks with desired sparsity
        mask_connectivity = self.rng.binomial(1, h_probability_mask)
        mask_output = self.rng.binomial(1, y_probability_mask)

        # Masks for weights and biases
        self.mask_Wz = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Wr = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Wh = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Y = nn.Parameter(th.tensor(mask_output), requires_grad=False)
        # No need to mask any biases for now
        self.mask_bz = nn.Parameter(th.ones_like(self.bz), requires_grad=False)
        self.mask_br = nn.Parameter(th.ones_like(self.br), requires_grad=False)
        self.mask_bh = nn.Parameter(th.ones_like(self.bh), requires_grad=False)
        self.mask_bY = nn.Parameter(th.ones_like(self.bY), requires_grad=False)

        # initialize cache
        self.Wr_cached = self.Wr.detach()
        self.Wz_cached = self.Wz.detach()
        self.Wh_cached = self.Wh.detach()
        self.Y_cached = self.Y.detach()

        # Create unit type masks if required
        if proportion_excitatory:
            type_list = np.array([-1, 1])
            self.unittype_W = nn.Parameter(th.zeros((hidden_size, hidden_size)), requires_grad=False)
            for m in range(self.num_modules):
                self.unittype_W[:, module_dims[m]] = (
                    th.tensor(type_list[self.rng.binomial(1, np.ones(module_size[m])*proportion_excitatory[m])],
                              dtype=th.float32))
            #self.mask_Y[:, self.unittype_W[0, :] != 1] = 0  # only allow excitatory units to produce output
            # Eliminate inhibitory connections across modules
            for i in range(hidden_size):
                for m in range(self.num_modules):
                    if i in module_dims[m]:
                        i_module = m
                for j in range(hidden_size):
                    for m in range(self.num_modules):
                        if j in module_dims[m]:
                            j_module = m
                    if j_module != i_module and self.unittype_W[i, j] == -1 and False:
                        self.mask_Wz[i, j + input_size] = 0
                        self.mask_Wr[i, j + input_size] = 0
                        self.mask_Wh[i, j + input_size] = 0
            self.enforce_dale()


        # Zero out weights and biases that we don't want to exist
        self.Wz = nn.Parameter(th.mul(self.Wz, self.mask_Wz))
        self.Wr = nn.Parameter(th.mul(self.Wr, self.mask_Wr))
        self.Wh = nn.Parameter(th.mul(self.Wh, self.mask_Wh))
        self.Y = nn.Parameter(th.mul(self.Y, self.mask_Y))
        self.bz = nn.Parameter(th.mul(self.bz, self.mask_bz))
        self.br = nn.Parameter(th.mul(self.br, self.mask_br))
        self.bh = nn.Parameter(th.mul(self.bh, self.mask_bh))
        self.bY = nn.Parameter(th.mul(self.bY, self.mask_bY))

        Wh_i, Wh = th.split(self.Wh.detach(), [input_size, hidden_size], dim=1)
        _, mask_Wh = th.split(self.mask_Wh.detach(), [input_size, hidden_size], dim=1)
        #Wh = self.orthogonalize_with_sparsity(Wh.numpy(), mask_Wh.numpy())
        #self.Wh = nn.Parameter(th.mul(th.cat((Wh_i, th.tensor(Wh)), dim=1), self.mask_Wh))
        if proportion_excitatory:
            self.enforce_dale()
            # Restoring E/I balance
            Wh_i, Wh = th.split(self.Wh.detach(), [input_size, hidden_size], dim=1)
            Wh[self.unittype_W == -1] = Wh[self.unittype_W == -1] / th.abs(th.sum(Wh[self.unittype_W == -1]))
            Wh[self.unittype_W == 1] = Wh[self.unittype_W == 1] / th.sum(Wh[self.unittype_W == 1])
            self.Wh = nn.Parameter(th.cat((Wh_i, Wh), dim=1))

        # Optional rescaling of Wh eigenvalues
        if self.spectral_scaling:
            Wh_i, Wh = th.split(self.Wh.detach(), [input_size, hidden_size], dim=1)
            eig_norm = th.max(th.real(th.linalg.eigvals(Wh)))
            Wh = self.spectral_scaling * (Wh / eig_norm)
            self.Wh = nn.Parameter(th.cat((Wh_i, Wh), dim=1))

        # Registering a backward hook to apply mask on gradients during backward pass
        self.Wz.register_hook(lambda grad: grad * self.mask_Wz.data)
        self.Wr.register_hook(lambda grad: grad * self.mask_Wr.data)
        self.Wh.register_hook(lambda grad: grad * self.mask_Wh.data)
        self.bz.register_hook(lambda grad: grad * self.mask_bz.data)
        self.br.register_hook(lambda grad: grad * self.mask_br.data)
        self.bh.register_hook(lambda grad: grad * self.mask_bh.data)
        self.Wh.register_hook(lambda grad: grad * self.mask_Wh.data)
        self.Y.register_hook(lambda grad: grad * self.mask_Y.data)
        self.bY.register_hook(lambda grad: grad * self.mask_bY.data)

        self.to(device)

    def set_cancelation_matrix(self, cancelation_matrix):
        self.cancelation_matrix = th.tensor(cancelation_matrix, dtype=th.float32)

    def reset_counter(self):
        self.counter = 0

    def set_cancel_times(self, times):
        self.cancel_times = times

    @th.compile(mode='max-autotune')
    def update_buffer(self, h_buffer, h_prev):
        # Create a new tensor by concatenating h_prev (reshaped appropriately) with the older values
        # Skip the last value to maintain the buffer size
        new_h_buffer = th.cat((h_prev.unsqueeze(-1), h_buffer[:, :, :-1]), dim=-1)
        return new_h_buffer

    @th.compile(mode='max-autotune')
    def forward(self, x, h_prev):
        # Update hidden state buffer
        self.h_buffer = self.update_buffer(self.h_buffer, h_prev)

        # If there are delays between modules we need to go module-by-module (this is slower)
        if self.max_connectivity_delay > 0:
            # Forward pass
            h_new = th.zeros_like(h_prev)
            for i in range(self.num_modules):
                # Prepare delayed hidden states for each module
                h_prev_delayed = th.zeros_like(h_prev)
                for j in range(self.num_modules):
                    h_prev_delayed[:, self.module_dims[j]] = self.h_buffer[:, self.module_dims[j],
                                                                           self.connectivity_delay[i, j]]
                concat = th.cat((x, h_prev_delayed), dim=1)
                z = th.sigmoid(F.linear(concat, self.Wz[self.module_dims[i], :], self.bz[self.module_dims[i]]))
                r = th.sigmoid(F.linear(concat, self.Wr, self.br))
                concat_hidden = th.cat((x, r * h_prev_delayed), dim=1)
                h_tilda = self.activation(F.linear(concat_hidden, self.Wh[self.module_dims[i], :],
                                                   self.bh[self.module_dims[i]]) +
                                          (th.randn(len(self.module_dims[i])) * 1e-3))
                h = (1 - z) * h_prev_delayed[:, self.module_dims[i]] + z * h_tilda
                # Store new hidden states to correct module
                h_new[:, self.module_dims[i]] = h

        # If there are no delays between modules we can do a single pass
        else:
            concat = th.cat((x, h_prev), dim=1)
            z = th.sigmoid(F.linear(concat, self.Wz, self.bz))
            r = th.sigmoid(F.linear(concat, self.Wr, self.br))
            concat_hidden = th.cat((x, r * h_prev), dim=1)
            h_tilda = self.activation(F.linear(concat_hidden, self.Wh, self.bh) + (th.randn(self.Wh.shape[0]) * 1e-3))
            h_new = (1 - z) * h_prev + z * h_tilda

        if self.cancelation_matrix is not None and self.counter in self.cancel_times:
            print('cancelling')
            h_new += h_new @ self.cancelation_matrix

        # Output layer
        if self.output_delay == 0:
            y = th.sigmoid(F.linear(h_new, self.Y, self.bY))
        else:
            y = th.sigmoid(F.linear(self.h_buffer[:, :, self.output_delay-1], self.Y, self.bY))
        self.counter += 1
        return y, h_new

    def init_hidden(self, batch_size):
        # Tile learnable hidden state
        h0 = th.tile(self.activation(self.h0), (batch_size, 1))
        # Create initial hidden state buffer
        self.h_buffer = th.tile(h0.unsqueeze(dim=2), (1, 1, self.max_delay+1))
        return h0

    def cache_policy(self):
        self.Wr_cached = self.Wr.detach()
        self.Wz_cached = self.Wz.detach()
        self.Wh_cached = self.Wh.detach()
        self.Y_cached = self.Y.detach()

    def enforce_dale(self, zero_out=False):
        with th.no_grad():
            unittype = self.unittype_W
            Wr_i, Wr = th.split(self.Wr.detach(), [self.input_size, self.hidden_size], dim=1)
            Wr_i_cached, Wr_cached = th.split(self.Wr_cached, [self.input_size, self.hidden_size], dim=1)
            print(th.sum((Wr < 0) & (unittype == 1)))
            print(th.sum((Wr > 0) & (unittype == -1)))
            if zero_out:
                Wr[(Wr < 0) & (unittype == 1)] = 0
                Wr[(Wr > 0) & (unittype == -1)] = 0
                #Wr_i[Wr_i < 0] = 0
            else:
                Wr[(Wr < 0) & (unittype == 1)] = th.abs(Wr_cached[(Wr < 0) & (unittype == 1)])
                Wr[(Wr > 0) & (unittype == -1)] = -th.abs(Wr_cached[(Wr > 0) & (unittype == -1)])
                #Wr_i[Wr_i < 0] = th.abs(Wr_i[Wr_i < 0])

            Wz_i, Wz = th.split(self.Wz.detach(), [self.input_size, self.hidden_size], dim=1)
            Wz_i_cached, Wz_cached = th.split(self.Wz_cached, [self.input_size, self.hidden_size], dim=1)
            print(th.sum((Wz < 0) & (unittype == 1)))
            print(th.sum((Wz > 0) & (unittype == -1)))
            if zero_out:
                Wz[(Wz < 0) & (unittype == 1)] = 0
                Wz[(Wz > 0) & (unittype == -1)] = 0
                #Wz_i[Wz_i < 0] = 0
            else:
                Wz[(Wz < 0) & (unittype == 1)] = th.abs(Wz_cached[(Wz < 0) & (unittype == 1)])
                Wz[(Wz > 0) & (unittype == -1)] = -th.abs(Wz_cached[(Wz > 0) & (unittype == -1)])
                #Wz_i[Wz_i < 0] = th.abs(Wz_i[Wz_i < 0])

            Wh_i, Wh = th.split(self.Wh.detach(), [self.input_size, self.hidden_size], dim=1)
            Wh_i_cached, Wh_cached = th.split(self.Wh_cached, [self.input_size, self.hidden_size], dim=1)
            print(th.sum((Wh < 0) & (unittype == 1)))
            print(th.sum((Wh > 0) & (unittype == -1)))
            if zero_out:
                Wh[(Wh < 0) & (unittype == 1)] = 0
                Wh[(Wh > 0) & (unittype == -1)] = 0
                #Wh_i[Wh_i < 0] = 0
            else:
                Wh[(Wh < 0) & (unittype == 1)] = th.abs(Wh_cached[(Wh < 0) & (unittype == 1)])
                Wh[(Wh > 0) & (unittype == -1)] = -th.abs(Wh_cached[(Wh > 0) & (unittype == -1)])
                #Wh_i[Wh_i < 0] = th.abs(Wh_i[Wh_i < 0])

            Y = self.Y.detach()
            #if zero_out:
                #Y[Y < 0] = 0
            #else:
                #Y[Y < 0] = th.abs(Y[Y < 0])

        self.Wr.data = th.cat((Wr_i, Wr), dim=1)
        self.Wz.data = th.cat((Wz_i, Wz), dim=1)
        self.Wh.data = th.cat((Wh_i, Wh), dim=1)
        self.Y.data = Y
        self.cache_policy()

    def orthogonalize_with_sparsity(self, matrix, sparsity_matrix):
        # Ensure sparsity_matrix is binary (0 or 1)
        assert np.all(np.isin(sparsity_matrix, [0, 1]))

        # Copy matrix so as not to modify the original
        Q = matrix.copy()

        # Loop over columns
        for i in range(Q.shape[1]):
            # Subtract projections onto previous columns
            for j in range(i):
                # Check if the column is a zero vector
                if np.dot(Q[:, j], Q[:, j]) < 1e-10:
                    continue

                proj = np.dot(Q[:, j], Q[:, i]) / np.dot(Q[:, j], Q[:, j])
                Q[:, i] -= proj * Q[:, j]

                # Reset the undesired entries to zero using sparsity matrix
                Q[:, i] *= sparsity_matrix[:, i]

            # Normalize current column, avoiding division by zero
            norm = np.linalg.norm(Q[:, i])
            if norm > 1e-10:
                Q[:, i] /= norm

                # Reset the undesired entries to zero again to ensure structure
                Q[:, i] *= sparsity_matrix[:, i]

        return Q
