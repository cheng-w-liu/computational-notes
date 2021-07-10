import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class Aggregator(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Aggregator, self).__init__()
        self.linear_layer = nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )
        self.input_dim = input_dim
        self.output_dim = output_dim


    def forward(self, neighbors: List[Tuple[int, float]], features_map: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        :param neighbors: List[Tuple[int, float]], a list of neighbor nodes along with the weights
        :param embeddings: mappings from each of the node id to its embedding
        :return: aggregated embedding
        """
        neighbor_indices = [v for v, _ in neighbors]
        weights = torch.tensor([w for _, w in neighbors])
        weights = weights / weights.sum()
        id2node = {i: v for i, (v, w) in enumerate(neighbors)}

        n_neighbors = len(neighbors)
        if n_neighbors == 0:
            return torch.zeros(self.input_dim)

        assert features_map[neighbor_indices[0]].size(0) == self.input_dim
        hiddens = torch.zeros(n_neighbors, self.output_dim)

        for i, v in id2node.items():
            hiddens[i] = F.relu(self.linear_layer(features_map[v]))

        """
        hiddens: (n_nodes, output_dim)
        hiddens.unsqueeze(1): (n_nodes, 1, output_dim)
        
        weights: (n_nodes,)
        weights.unsqueeze(1).unsqueeze(1): (n_nodes, 1, 1)
        
        *: element-wise multiplication
        
        hiddens.unsqueeze(1) *  weights.unsqueeze(1).unsqueeze(1): (n_nodes, 1, output_dim)
        
        """
        agg = (hiddens.unsqueeze(1) *  weights.unsqueeze(1).unsqueeze(1)).squeeze(1)  # (n_nodes, output_dim)
        agg = agg.sum(dim=0, keepdim=False)

        return agg


def test_aggregator():
    neighbors = [(3, 0.3), (1, 0.1), (4, 0.4)]

    V = 5
    feat_dim = 20
    weights = torch.randn(size=(V, feat_dim))

    embeddings = nn.Embedding(
        num_embeddings=V,
        embedding_dim=feat_dim
    )
    embeddings.weight = torch.nn.Parameter(weights)
    embeddings.weight.requires_grad = False

    features_map = {v: embeddings(torch.tensor(v)) for v, prob in neighbors}

    aggregator = Aggregator(input_dim=feat_dim, output_dim=3)
    n = aggregator(neighbors, features_map)
    print('input:')
    print(embeddings)
    print('\n\noutput:')
    print(n)

if __name__ == '__main__':
    test_aggregator()





