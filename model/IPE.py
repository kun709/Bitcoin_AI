import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolePositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_atoms=51):
        super(InterpolePositionalEmbedding, self).__init__()
        self.atoms = embed_atoms
        self.embedding = nn.Embedding(embed_atoms, embed_dim)

    def forward(self, x: torch.Tensor):
        x *= self.atoms

        x_floor = torch.floor(x).long().clip(0, self.atoms - 1)
        x_ceil = torch.ceil(x).long().clip(0, self.atoms - 1)
        x_deci = (x - torch.floor(x)).unsqueeze(-1).float()

        x_floor_embed = self.embedding(x_floor)
        x_ceil_embed = self.embedding(x_ceil)

        result = x_floor_embed * (1 - x_deci) + x_ceil_embed * x_deci
        return result

    def loss(self):
        return F.mse_loss(self.embedding.weight[1:], self.embedding.weight[:-1])


if __name__ == "__main__":
    ipe = InterpolePositionalEmbedding(2)
    test_input = torch.rand(1, 2)
    print(ipe(test_input))
