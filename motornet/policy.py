import torch
import torch.nn as nn


class PolicyGRU(nn.Module):
  """A single-layer GRU policy network with a sigmoid-activated linear readout.

  Args:
    input_dim: `Integer`, dimensionality of the input observation vector.
    hidden_dim: `Integer`, number of hidden units in the GRU layer.
    output_dim: `Integer`, dimensionality of the output (motor command) vector.
    device: The device on which to place the network parameters.
  """

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
    super().__init__()
    self.device = device
    self.hidden_dim = hidden_dim
    self.n_layers = 1

    self.gru = torch.nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
    self.fc = torch.nn.Linear(hidden_dim, output_dim)
    self.sigmoid = torch.nn.Sigmoid()

    # the default initialization in torch isn't ideal
    for name, param in self.named_parameters():
      if name == "gru.weight_ih_l0":
        torch.nn.init.xavier_uniform_(param)
      elif name == "gru.weight_hh_l0":
        torch.nn.init.orthogonal_(param)
      elif name == "gru.bias_ih_l0":
        torch.nn.init.zeros_(param)
      elif name == "gru.bias_hh_l0":
        torch.nn.init.zeros_(param)
      elif name == "fc.weight":
        torch.nn.init.xavier_uniform_(param)
      elif name == "fc.bias":
        torch.nn.init.constant_(param, -5.)
      else:
        raise ValueError

    self.to(device)

  def forward(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs one step of the GRU and returns the motor command and updated hidden state.

    Args:
      x: `Tensor`, the input observation vector with shape `(batch_size, input_dim)`.
      h0: `Tensor`, the initial hidden state with shape `(n_layers, batch_size, hidden_dim)`.

    Returns:
      - The motor command `tensor` with shape `(batch_size, output_dim)`, passed through a sigmoid.
      - The updated hidden state `tensor` with shape `(n_layers, batch_size, hidden_dim)`.
    """
    y, h = self.gru(x[:, None, :], h0)
    u = self.sigmoid(self.fc(y)).squeeze(dim=1)
    return u, h

  def init_hidden(self, batch_size: int) -> torch.Tensor:
    """Creates a zero-initialized hidden state tensor on the correct device.

    Args:
      batch_size: `Integer`, the number of parallel sequences in the batch.

    Returns:
      A zero `tensor` with shape `(n_layers, batch_size, hidden_dim)`.
    """
    weight = next(self.parameters()).data
    hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
    return hidden
