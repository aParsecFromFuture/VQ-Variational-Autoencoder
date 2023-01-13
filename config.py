import torch

img_dim = 32
num_channels = 3
embedding_dim = 32
num_embeddings = 512

batch_size = 256
lr = 1e-3
commitment_cost = 0.25

num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')