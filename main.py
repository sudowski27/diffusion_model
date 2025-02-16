"""version 0.1.3"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import get_dataset
from src.mlp import Mlp
from src.get_noise_from_data import get_noise_from_data
from src.get_data_from_noise import get_data_from_noise


def main():
    """
    Main function
    """
    number_of_samples = 1200
    param = 1.0
    param_1 = 0.9
    param_2 = 0.8
    param_3 = 0.7
    param_4 = 0.6
    param_5 = 0.5
    number_of_diffrent_sets = 2
    device = "cuda"
    timesteps = 40
    plot_0_fname = "plot_0.png"
    plot_01_fname = "plot_01.png"
    plot_1_fname = "plot_1.png"
    lr = 1e-4
    epoch_num = 300_000
    data, data_debug = get_dataset(
        number_of_samples,
        param,
        number_of_diffrent_sets
    )

    data_1, data_debug_1 = get_dataset(
        number_of_samples,
        param_1,
        number_of_diffrent_sets
    )
    for db_data in data_debug:
        plt.plot(db_data, 'b-', linewidth=2.5)
    plt.savefig(plot_0_fname)
    plt.close()
    for db_data in data_debug_1:
        plt.plot(db_data, 'b-', linewidth=2.5)
    plt.savefig(plot_01_fname)
    plt.close()
    layer_widths_encoder = [1200, 600, 300, 150, 100]
    layer_widths_decoder = [101, 150, 300, 600, 1200, 2400]
    param_tensor_0 = torch.tensor([param], dtype=torch.float32, device=device)
    param_tensor_1 = torch.tensor([param_1], dtype=torch.float32, device=device)
    data_tensor_0 = torch.tensor(data, dtype=torch.float32, device=device)
    data_tensor_1 = torch.tensor(data_1, dtype=torch.float32, device=device)

    plt.plot(data)
    plt.savefig(plot_1_fname)
    plt.close()

    mlp_model = Mlp(
        timesteps=timesteps,
        layer_widths_encoder=layer_widths_encoder,
        layer_widths_decoder=layer_widths_decoder
    ).to(device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)
    epoch_offset = 0

    betas = (torch.sigmoid(torch.linspace(-18, 10, timesteps)) * (3e-1 - 1e-5) + 1e-5).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0).to(device)
    global_loss = []
    FIRST_DATA = 0
    SECOND_DATA = 1
    for epoch in range(epoch_offset, epoch_offset + epoch_num):
        i = int(np.random.uniform(0, 40))
        param_choice = np.random.randint(0, 2)
        if param_choice == FIRST_DATA:
            data_tensor = data_tensor_0
            param_tensor = param_tensor_0
        elif param_choice == SECOND_DATA:
            data_tensor = data_tensor_1
            param_tensor = param_tensor_1

        noise = torch.randn_like(data_tensor, device=device)

        active_index = i
        for j, network in enumerate(mlp_model.encoder.network_list):
            for param in network.parameters():
                param.requires_grad = j == active_index

        for j, network in enumerate(mlp_model.decoder.network_list):
            for param in network.parameters():
                param.requires_grad = j == active_index

        mu = torch.sqrt(alphas_bar[i]) * data_tensor
        std = torch.sqrt(1 - alphas_bar[i])
        epsilon = torch.randn_like(data_tensor)
        xt = mu + epsilon * std

        std_q = torch.sqrt((1 - alphas_bar[i-1]) / (1 - alphas_bar[i]) * betas[i])
        m1 = torch.sqrt(alphas_bar[i-1]) * betas[i] / (1 - alphas_bar[i])
        m2 = torch.sqrt(alphas[i]) * (1 - alphas_bar[i-1]) / (1 - alphas_bar[i])
        mu_q = m1 * data_tensor + m2 * xt

        mu, std = mlp_model(xt, i, param_tensor)

        kl = torch.log(std) - torch.log(std_q) + (
            std_q**2 + (mu_q - mu)**2) / (2 * std**2)
        k = - kl.mean()
        loss = -k

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_loss.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch: {epoch} loss: {loss.item()}")

        if epoch % 1000 == 0:
            noise = get_noise_from_data(
                data_tensor,
                alphas_bar,
                timesteps - 1
            )
            noise_np = noise.cpu().numpy()
            with torch.no_grad():
                data_np0 = get_data_from_noise(
                    noise,
                    mlp_model,
                    param_tensor_0,
                    timesteps - 1
                ).cpu().numpy()
                data_np1 = get_data_from_noise(
                    noise,
                    mlp_model,
                    param_tensor_1,
                    timesteps - 1
                ).cpu().numpy()
            plt.plot(noise_np)
            plt.savefig(f"results/noise/{epoch}")
            plt.close()
            plt.plot(data_np0)
            plt.savefig(f"results/data/{epoch}_param0.png")
            plt.close()
            plt.plot(data_np1)
            plt.savefig(f"results/data/{epoch}_param1.png")
            plt.close()

            plt.plot(np.abs(data_np0 - noise_np))
            plt.savefig(f"results/abs/{epoch}")
            plt.close()

    plt.plot(global_loss)
    plt.show()


if __name__ == "__main__":
    main()
