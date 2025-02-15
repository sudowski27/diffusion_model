"""version 0.1.0"""
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Mlp(nn.Module):
    """
    Mlp class
    """
    def __init__(self, timesteps: int,
                 layer_widths_encoder: list,
                 layer_widths_decoder: list):
        super().__init__()
        # encoder,
        self.encoder = Encoder(
            timesteps=timesteps,
            layer_widths=layer_widths_encoder
        )
        self.decoder = Decoder(
            timesteps=timesteps,
            layer_widths=layer_widths_decoder
        )

    def forward(self, x, t, cond):
        x_encoder = self.encoder(
            x, t
        )

        x_with_cond = torch.cat((x_encoder, cond))

        x_decoder = self.decoder(
            x_with_cond, t
        )

        return x_decoder
