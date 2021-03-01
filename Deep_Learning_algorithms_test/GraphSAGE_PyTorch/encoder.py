import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim, activation='relu'):
        assert activation in ['relu', 'sigmoid', 'none']
        super(Encoder, self).__init__()
        self.activation = activation
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        :param inputs: a tuple of (nodes_embed, neighs_embed)
           nodes_embed: nodes embeddings, shape (num_of_nodes, hidden_dim)
           neighs_embed: embedding of each node's neighbors, shape (num_of_nodes, hidden_dim)

        :return:
          processed embeddings, shape (num_of_nodes, output_dim)
        """
        nodes_embed, neighs_embed = inputs
        embed = torch.cat([nodes_embed, neighs_embed], dim=1)
        h = self.linear_layer(embed)
        if self.activation == 'relu':
            h = F.relu(h)
        elif self.activation == 'sigmoid':
            h = F.sigmoid(h)
        else:
            raise ValueError(f'Unknown activation: {self.activation}')

        h = F.normalize(h, p=2, dim=1)
        return h



def test_encoder():
    feat_dim = 3
    output_dim = 5
    n = 4
    nodes_embed = torch.randn(size=(n, feat_dim))
    neighs_embed = torch.randn(size=(n, feat_dim))

    encoder = Encoder(2*feat_dim, output_dim)

    encoder_inputs = (nodes_embed, neighs_embed)
    embed = encoder(encoder_inputs)

    print('\nProcessed embedding:')
    print(embed)
    print('\nTrainable variables:')
    for params in encoder.parameters():
        print(params)


if __name__ == '__main__':
    test_encoder()

