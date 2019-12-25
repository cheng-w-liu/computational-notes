import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set
import numpy as np


class Aggregator(nn.Module):
    def __init__(self, feature_map: nn.Embedding, agg_type='mean', gpu=False):
        """
        :param feature_map:
        :param feat_dim:
        :param agg_type:
        :param gpu:
        """
        super(Aggregator, self).__init__()

        if agg_type not in ['max', 'mean']:
            raise ValueError(f'Expect agg_type to be "max" or "mean", but got : {agg_type}')

        self.feature_map = feature_map
        self.feat_dim = feature_map.weight.shape[1]
        self.agg_type = agg_type
        self.gpu = gpu


    def forward(self, neighbors: List[Set], num_sample=10) -> torch.Tensor:
        """
        :param neighbors: a list of sets, each set corresponds to the neighbors of a given node
        :param num_sample: number of neighbors to sample
        :return: torch.Tensor representing the aggregated features from the neighbors
                shape: (numb_of_nodes, feat_dim)
        """
        if num_sample is not None:
            sampled_neighbors = [set(np.random.choice(neigh, num_sample))
                                 if len(neigh) > num_sample else neigh for neigh in neighbors]
        else:
            sampled_neighbors = neighbors

        if self.agg_type == 'max':
            return self._maxForward(sampled_neighbors)
        else:
            return self._meanForward(sampled_neighbors)


    def _maxForward(self, neighbors: List[Set]) -> torch.Tensor:
        agg_neighbors_features = torch.zeros(len(neighbors), self.feat_dim)
        for i, neigh in enumerate(neighbors):
            node_indices = torch.tensor(list(neigh), dtype=torch.long)
            n_features = self.feature_map(node_indices)  # shape (numb_of_neighbors, feat_dim)
            agg_neighbors_features[i, :] = torch.max(n_features, dim=0)
        return agg_neighbors_features


    def _meanForward(self, neighbors: List[Set]) -> torch.Tensor:
        """
        :param neighbors: a list of sets, each set corresponds to the neighbors of a given node
        :return:
        """
        # all_nodes specifies a temporary order that the nodes that appear in this batch will appear
        all_nodes = list(set.union(*neighbors))

        node_indices = torch.tensor(all_nodes, dtype=torch.long)
        neighbors_features = self.feature_map(node_indices)
        if self.cuda:
            neighbors_features = neighbors_features.cuda()

        node2id = {n: i for i, n in enumerate(all_nodes)}
        mask = torch.zeros(len(neighbors), len(all_nodes), dtype=torch.double)
        row_indices = [i for i in range(len(neighbors)) for _ in range(len(neighbors[i]))]
        col_indices = [node2id[n] for n_neighbors in neighbors for n in n_neighbors]
        mask[row_indices, col_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        num_neigh = torch.sum(mask, 1).unsqueeze(dim=1)
        mask /= num_neigh

        agg_neighbors_features = mask.mm(neighbors_features)

        return agg_neighbors_features

def test_aggregator():
    custom_weights = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.0]).reshape(6, 2)
    V = 5
    feat_dim = 3
    embeddings = nn.Embedding(V+1, feat_dim, padding_idx=V)
    embeddings.weight = nn.Parameter(
        torch.tensor(custom_weights),
        requires_grad=False
    )
    agg = Aggregator(embeddings)
    neighbors = [set([0, 2]), set([1, 3, 4]), set([3])]
    agg_neighbors_features = agg(neighbors)
    print(agg_neighbors_features)

if __name__ == '__main__':
    test_aggregator()

