import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, z_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, z_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(z_channels // 2),
            nn.Conv2d(z_channels // 2, z_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(z_channels),
            nn.Conv2d(z_channels, z_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, z_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_channels, z_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(z_channels),
            nn.ConvTranspose2d(z_channels, z_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(z_channels // 2),
            nn.Conv2d(z_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous(), loss


class VQVAE(nn.Module):
    def __init__(self, num_channels, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.encoder = Encoder(num_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, qloss = self.quantizer(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, qloss
