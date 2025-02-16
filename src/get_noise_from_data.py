"""version 0.1.0"""
import torch


def get_noise_from_data(
    ref_data: torch.tensor,
    alphas_bar: torch.tensor,
    time: int
):
    """
    Function gets noise from data
    """
    mu = torch.sqrt(alphas_bar[time]) * ref_data
    std = torch.sqrt(1 - alphas_bar[time])

    epsilon = torch.randn_like(ref_data)

    noised_data = mu + epsilon * std
    return noised_data
