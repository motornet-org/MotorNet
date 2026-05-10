import torch
import torch.nn as nn


class PolicyGRU(nn.Module):
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

  def forward(self, x, h0):
    y, h = self.gru(x[:, None, :], h0)
    u = self.sigmoid(self.fc(y)).squeeze(dim=1)
    return u, h

  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
    return hidden
