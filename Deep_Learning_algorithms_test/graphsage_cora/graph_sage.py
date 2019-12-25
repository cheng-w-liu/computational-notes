import numpy as np
import torch
import torch.nn as nn
from typing import List

from aggregator import Aggregator
from encoder import Encoder


class GraphSAGEclassifier(nn.Module):
    """
    Apply GraphSAGE to classify nodes
    """
    def __init__(self, num_classes: int, encoder: Encoder):
        """
        :param num_classes: number of classes
        :param encoder: an Encoder instance
        """
        super(GraphSAGEclassifier, self).__init__()
        self.encoder = encoder
        self.cross_entropy = nn.CrossEntropyLoss()
        self.projection = nn.Linear(
            in_features=encoder.embed_dim,
            out_features=num_classes
        )


    def forward(self, nodes: List[int]) -> torch.Tensor:
        """
        :param nodes: a batch of nodes, shape (batch_size,)
        :return: hidden vectors before applying the softmax function
        """
        embeds = self.encoder(nodes)  # (batch_size, embed_dim)
        hiddens = self.projection(embeds)  # (batch_size, num_classes)
        return hiddens


    def loss(self, nodes: List[int], labels):
        """
        :param nodes: a batch of nodes, shape (batch_size,)
        :param labels: the corresponding labels for the nodes, shape (N,)
        :return: a real-number corresponding to the cross entropy loss of this batch
        """
        hiddens = self.forward(nodes) # shape (batch_size, num_classes)
        return self.cross_entropy(hiddens, labels)


def test_graphsage():
    custom_weights = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.0],
        dtype=np.float32
    ).reshape(6, 2)
    V = 5
    feat_dim = 2
    embeddings = nn.Embedding(V+1, feat_dim, padding_idx=V)
    embeddings.weight = nn.Parameter(
        torch.tensor(custom_weights),
        requires_grad=False
    )
    adj_list = [set([0, 2]), set([1, 3, 4]), set([3]), set([0, 1, 2]), set([3])]
    labels = np.array([0, 1, 0, 1, 0])

    agg = Aggregator(embeddings, feat_dim)

    embed_dim = 3
    enc = Encoder(embeddings, agg, adj_list, embed_dim)

    graphsage = GraphSAGEclassifier(2, enc)
    print('parameters:')
    for name, param in graphsage.named_parameters():
        if param.requires_grad:
            print('--------------\n')
            print(f'name: {name}')
            print(param)
            print('\n')

    hiddens = graphsage.forward([0, 1, 2, 3, 4])
    print('\n')
    print('hiddens:')
    print(hiddens)
    loss = graphsage.loss([0, 1, 2, 3, 4], torch.tensor(labels))
    print(f'loss: {loss}')
    loss.backward()


if __name__ == '__main__':
    test_graphsage()
