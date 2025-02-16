"""version 0.1.0"""
import torch


def get_data_from_noise(
    noise: torch.tensor,
    model: torch.nn.Module,
    condition: torch.tensor,
    time: int
):
    """
    Function get_data_from_noise
    """
    samples = [noise]
    prev_data = noise
    model.eval()
    for t in range(time, 0, -1):
        mu, std = model(
            x=prev_data,
            t=t,
            cond=condition
        )
        epsilon = torch.randn_like(noise)
        data = mu + epsilon * std
        samples.append(data)
        prev_data = data

    final_data = samples[-1]
    model.train()

    return final_data
