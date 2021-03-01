import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class Aggregator(nn.Module):

    def __init__(self, input_dim, agg_method='mean'):
        super(Aggregator, self).__init__()
        assert agg_method in ['mean', 'max', 'pool']
        self.input_dim = input_dim
        self.agg_method = agg_method
        self.linear_layer = None
        if agg_method == 'pool':
            self.linear_layer = nn.Linear(
                in_features=input_dim,
                out_features=input_dim
            )

    def forward(self, inputs: Tuple[int, List[int], Dict[int, torch.Tensor]]) -> torch.Tensor:
        """
        :param inputs: a tuple of (u, neighs, features)
          u: int, the node whose neighbors we care about
          neighs: List[int], a list of sampled nodes representing u's neighbors
          features_map: Dict[int, torch.Tensor], a lookup table that maps to a node's feature. Acts as h^{k-1}_u

        :return:
          an aggregated neighbor embedding. Acts as h^{k}_{N(u)}
        """
        u, neighs, features_map = inputs
        neighs_features = torch.stack(
            [features_map[v] for v in neighs],
            dim=0
        )

        if self.agg_method == 'mean':
            h = torch.mean(neighs_features, dim=0)
        elif self.agg_method == 'max':
            h = torch.max(neighs_features, dim=0)[0]
        else:
            z = torch.max(
                self.linear_layer(neighs_features),
                dim=0
            )[0]
            h = F.sigmoid(z)

        return h


def test_aggregator():
    adjList = {
        0: [1, 2, 3],
        1: [2, 3, 2],
        2: [2, 2, 2],
        3: [1, 1, 1]
    }

    V = len(adjList)
    feat_dim = 2
    weights = torch.randn(size=(V, feat_dim))

    embeddings = nn.Embedding(
        num_embeddings=V,
        embedding_dim=feat_dim
    )
    embeddings.weight = torch.nn.Parameter(weights)
    embeddings.weight.requires_grad = False

    features_map = {v: embeddings(torch.tensor(v)) for v in range(V)}
    print(features_map)

    print('\nTest mean aggregator:')
    mean_aggregator = Aggregator(feat_dim, 'mean')
    for u in adjList.keys():
        agg_inputs = (u, adjList[u], features_map)
        h = mean_aggregator(agg_inputs)
        print(f'h_N({u})={h}')

    print('\nTest max aggregator:')
    max_aggregator = Aggregator(feat_dim, 'max')
    for u in adjList.keys():
        agg_inputs = (u, adjList[u], features_map)
        h = max_aggregator(agg_inputs)
        print(f'h_N({u})={h}')


    print('\nTest pooling aggregator:')
    pool_aggregator = Aggregator(feat_dim, 'pool')
    for u in adjList.keys():
        agg_inputs = (u, adjList[u], features_map)
        h = pool_aggregator(agg_inputs)
        print(f'h_N({u})={h}')
    print('Pooling layer weights:')
    for params in pool_aggregator.parameters():
        print(params)


if __name__ == '__main__':
    test_aggregator()

