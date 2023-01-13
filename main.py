from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from trainer import Trainer
import config as cfg

dataset = datasets.CIFAR10(root='.',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

dataloader = DataLoader(dataset,
                        batch_size=cfg.batch_size,
                        shuffle=True)

vqvae_trainer = Trainer(dataloader, debug=True, save_examples=True)
vqvae_trainer.run()
