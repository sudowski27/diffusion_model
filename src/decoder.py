"""version 0.1.1"""
import torch
from torch import nn


class Decoder(nn.Module):
    """
    Encoder module
    """
    def __init__(self, timesteps: int, layer_widths: list):
        super().__init__()
        self.network_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layer_widths[0], layer_widths[1]),
                    nn.ReLU(),
                    nn.Linear(layer_widths[1], layer_widths[2]),
                    nn.ReLU(),
                    nn.Linear(layer_widths[2], layer_widths[3]),
                    nn.ReLU(),
                    nn.Linear(layer_widths[3], layer_widths[4]),
                    nn.ReLU(),
                    nn.Linear(layer_widths[4], layer_widths[5])
                ) for _ in range(timesteps)
            ]
        )

    def forward(self, x: torch.Tensor, t: int):
        """
        Forward method
        """
        x = self.network_list[t](x)

        mu, h = torch.chunk(x, 2)

        var = torch.exp(h)
        std = torch.sqrt(var)

        return mu, std
