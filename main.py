"""version 0.1.1"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import get_dataset
from src.mlp import Mlp


def main():
    """
    Main function
    """
    number_of_samples = 2000
    param = 1.0
    number_of_diffrent_sets = 5
    device = "cuda"
    timesteps = 40
    plot_0_fname = "plot_0.png"
    plot_1_fname = "plot_1.png"
    lr = 1e-4
    epoch_num = 2000
    data, data_debug = get_dataset(
        number_of_samples,
        param,
        number_of_diffrent_sets
    )
    for db_data in data_debug:
        plt.plot(db_data, 'b-', linewidth=2.5)
    plt.savefig(plot_0_fname)
    plt.close()
    layer_widths_encoder = [2000, 1000, 500, 250, 125]
    layer_widths_decoder = [126, 250, 500, 1000, 2000]
    param_tensor = torch.tensor([param], dtype=torch.float32, device=device)
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    plt.plot(data)
    plt.savefig(plot_1_fname)
    plt.close()

    # TODO Add load from file if exists
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

    print(betas)

    global_loss = []
    for epoch in range(epoch_offset, epoch_offset + epoch_num):
        local_loss = []
        for i in range(2, timesteps):
            noise = torch.randn_like(data_tensor, device=device)
            noised_data = torch.sqrt(alphas_bar[i]) * data_tensor + torch.sqrt(1 - alphas_bar[i]) * noise

            active_index = i
            for j, network in enumerate(mlp_model.encoder.network_list):
                for param in network.parameters():
                    param.requires_grad = j == active_index

            for j, network in enumerate(mlp_model.decoder.network_list):
                for param in network.parameters():
                    param.requires_grad = j == active_index

            prediction = mlp_model(
                x=noised_data,
                t=i,
                cond=param_tensor
            )
            error = torch.nn.MSELoss()(noised_data, prediction)
            local_loss.append(error.item())
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
        local_loss = np.array(local_loss)
        local_loss_mean = np.mean(local_loss)
        global_loss.append(local_loss_mean)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")

    plt.plot(global_loss)
    plt.show()


if __name__ == "__main__":
    main()
