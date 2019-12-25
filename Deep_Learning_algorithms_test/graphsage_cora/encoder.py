from aggregator import Aggregator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set

class Encoder(nn.Module):
    """
    Encodes a list of nodes to the embedding vectors
    """
    def __init__(self, feature_map, aggregator: Aggregator, adj_list: List[Set],
                 embed_dim: int, base_model = None,
                 num_sample: int = 10, gpu: bool = False):
        """
        :param feature_map: used to generate feature vectors for a list of inputs
        :param aggregator: used to aggregate neighbors' features
        :param adj_list: adjacency list, each entry is a set of neighbors of a given node
        :param embed_dim: int, embedding dimension
        :param num_sample: number of neighbors to sample in the aggregation process
        :param gpu: indicates whether to use GPU or not
        """
        super(Encoder, self).__init__()

        self.feature_map = feature_map
        self.feat_dim = feature_map.weight.shape[1]
        self.aggregator = aggregator
        self.aggregator.gpu = gpu
        self.adj_list = adj_list
        self.embed_dim = embed_dim
        self.base_model = base_model
        self.num_sample = num_sample
        self.gpu = gpu
        self.W_k = nn.Linear(
            in_features=2 * self.feat_dim,
            out_features=self.embed_dim
        )

    def forward(self, nodes: List[int]) -> torch.Tensor:
        """
        :param nodes: a batch of nodes
        :return: encoded features for the nodes
        """
        nodes_features = self.feature_map(
            torch.tensor(nodes, dtype=torch.long)
        )  # shape: (numb_of_nodes, feat_dim)
        if self.gpu:
            nodes_features = nodes_features.cuda()
        neighbors = [self.adj_list[node] for node in nodes]
        neighbors_features = self.aggregator(neighbors, self.num_sample)  # shape: (numb_of_nodes, feat_dim)

        print("neighbors_features:")
        print(neighbors_features)

        features = torch.cat(
            tensors=(nodes_features[:], neighbors_features[:]),
            dim=1
        ) # shape: (numb_of_nodes, 2 * feat_dim)

        print('\n')
        print('combined features:')
        print(features)

        h = F.relu(self.W_k(features)) # shape: (numb_of_nodes, embed_dim)

        return h


def test_encoder():
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

    agg = Aggregator(embeddings, feat_dim)

    embed_dim = 3
    enc = Encoder(embeddings, agg, adj_list, embed_dim)
    h = enc([0, 1, 2])


if __name__ == '__main__':
    test_encoder()
