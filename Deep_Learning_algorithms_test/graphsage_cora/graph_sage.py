import torch
import torch.nn as nn
from typing import List

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
            in_features=encoder.feat_dim,
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


