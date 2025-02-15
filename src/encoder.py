"""version 0.1.0"""
import torch
from torch import nn


class Encoder(nn.Module):
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
                    nn.Linear(layer_widths[3], layer_widths[4])
                ) for _ in range(timesteps)
            ]
        )

    def forward(self, x: torch.Tensor, t: int):
        """
        Forward method
        """
        x = self.network_list[t](x)

        return x
