import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from model import VQVAE
from utils import save_imgs
import config as cfg


class Trainer:
    def __init__(self, dataloader, debug=True, save_path=None, save_examples=True):
        super().__init__()
        self.debug = debug
        self.save_path = save_path
        self.save_examples = save_examples
        self.model = VQVAE(cfg.num_channels, cfg.embedding_dim, cfg.num_embeddings, cfg.commitment_cost).to(cfg.device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def run(self):
        self.model.train()
        for epoch in range(cfg.num_epochs):
            for idx, (imgs, _) in enumerate(self.dataloader):
                x = imgs.to(cfg.device)
                x_hat, quantizer_loss = self.model(imgs)
                recon_loss = F.mse_loss(x_hat, x)
                loss = quantizer_loss + recon_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch: {epoch}, RLoss: {recon_loss.item():.3f}, QLoss: {quantizer_loss.item():.3f}')

        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)
            print(f'Model saved to {self.save_path}!')

        if self.save_examples:
            self.model.eval()
            with torch.no_grad():
                imgs, _ = next(iter(self.dataloader))
                imgs_recon, _ = self.model(imgs.to(cfg.device))
            save_imgs(make_grid(torch.cat([imgs.cpu()[:32], imgs_recon.cpu()[:32]])))

