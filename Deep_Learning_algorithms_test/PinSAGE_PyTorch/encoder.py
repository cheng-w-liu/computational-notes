import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input dimension of the linear layer, input_dim = 2 * feat_dim
        :param output_dim: output dimension of the linear layer
        """
        super(Encoder, self).__init__()
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        :param inputs: a tuple of (nodes_embeddings, neighbors_embeddings)
            nodes_embeddings: torch.Tensor, shape: (n_nodes, feat_dim)
            neighbors_embeddings: torch.Tensor, shape: (n_nodes, feat_dim)

        :return:
            tensor representing the k-layer's embeddings
        """
        embed = torch.cat(inputs, dim=1)
        h = self.linear_layer(embed)
        h = F.relu(h)
        h = F.normalize(h, p=2, dim=1)
        return h


def test_encoder():
    feat_dim = 3
    output_dim = 5
    n = 4
    nodes_embed = torch.randn(size=(n, feat_dim))
    neighs_embed = torch.randn(size=(n, feat_dim))
    encoder_inputs = (nodes_embed, neighs_embed)

    encoder = Encoder(2*feat_dim, output_dim)
    h = encoder(encoder_inputs)
    print('\nProcessed embeddings:')
    print(h)

    for params in encoder.parameters():
        print(params)


if __name__ == '__main__':
    test_encoder()

