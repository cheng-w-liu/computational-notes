import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from aggregators import Aggregator
from encoder import Encoder


class PinSAGE(nn.Module):

    def __init__(self, dims: List[int], k_layers=2):
        """
        dims[4*k + 0] = aggregators[k] input_dimension
        dims[4*k + 1] = aggregators[k] output_dimension
        dims[4*k + 2] = encoders[k] input_dimension
        dims[4*k + 3] = encoders[k] output_dimension

        assert dims[4*k + 2] == 2 * dims[4*k + 1]

        if k < K-1:
            assert dims[4*k + 3] == dims[4*(k+1)]
        """
        super(PinSAGE, self).__init__()
        assert len(dims) == 4 * k_layers

        self.k_layers = k_layers
        self.dims = dims[:]
        self.aggregators = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for k in range(k_layers): # k = 0, ..., K-1
            assert dims[4*k+2] == 2 * dims[4*k+1]
            if k < k_layers - 1:
                assert dims[4*k+3] == dims[4*(k+1)]
            self.aggregators.append( Aggregator(dims[4*k], dims[4*k+1]) )
            self.encoders.append( Encoder(dims[4*k+2], dims[4*k+3]) )


    def forward(self,
                batch_nodes: List[List[int]],
                batch_adjList: List[Dict[int, Tuple[int, float]]],
                embeddings: nn.Embedding
                ) -> torch.Tensor:

        features_map = {u: embeddings(torch.tensor(u)) for u in batch_nodes[0]}

        for k in range(1, self.k_layers+1):  # k = 1, ..., K
            """
            Generate embeddings for nodes in b^k = batch_nodes[k]
            We need nodes in b^{k-1}, which is represented by the graph g^{k-1} = batch_adjList[k-1]
            """
            nodes = {}
            neighs = {}
            for u in batch_nodes[k]:
                nodes[u] = features_map[u]
                neighs[u] = self.aggregators[k-1](batch_adjList[k-1][u], features_map)

            id2node = {i: u for i, u in enumerate(batch_nodes[k])}

            nodes_embeddings = torch.stack(
                [nodes[u] for u in batch_nodes[k]],
                dim=0
            )

            neighs_embeddings = torch.stack(
                [neighs[u] for u in batch_nodes[k]],
                dim=0
            )

            h = self.encoders[k-1]((nodes_embeddings, neighs_embeddings))

            # refresh features_map
            features_map = {id2node[i]: h[i] for i in range(h.size(0))}

        # convert features_map to a tensor
        h = torch.stack(
            [features_map[u] for u in batch_nodes[-1]],
            dim=0
        )

        return h


class NodeClassifier(nn.Module):

    def __init__(self, num_classes, dims: List[int], k_layers=2, output_dim=64):
        """
        dims[4*k + 0] = aggregators[k] input_dimension
        dims[4*k + 1] = aggregators[k] output_dimension
        dims[4*k + 2] = encoders[k] input_dimension
        dims[4*k + 3] = encoders[k] output_dimension

        assert dims[4*k + 2] == 2 * dims[4*k + 1]

        if k < K-1:
            assert dims[4*k + 3] == dims[4*(k+1)]
        """
        super(NodeClassifier, self).__init__()
        assert len(dims) == 4 * k_layers
        self.sage = PinSAGE(dims, k_layers)
        self.linear1 = nn.Linear(dims[-1], output_dim)
        self.linear2 = nn.Linear(output_dim, num_classes, bias=False)


    def forward(self,
                batch_nodes: List[List[int]],
                batch_adjList: List[Dict[int, Tuple[int, float]]],
                embeddings: nn.Embedding
                ) -> torch.Tensor:

        h = self.sage(batch_nodes, batch_adjList, embeddings)
        return self.linear2(F.relu(self.linear1(h)))


def test_pinsage():
    import numpy as np
    from graph_sampling import get_unique_nodes, random_sampling, topK_sampling

    np.random.seed(42)

    adjList = {
        0: [1, 2],
        1: [11, 0, 3],
        2: [0, 3, 7],
        3: [1, 2, 4, 9, 10],
        4: [3, 11],
        5: [9],
        6: [7, 10],
        7: [6, 10, 2],
        8: [9],
        9: [5, 3, 8],
        10: [3, 6, 7],
        11: [1, 4]
    }
    nodes = get_unique_nodes(adjList, has_weight=False)
    V = len(nodes)

    feat_dim = 100
    weights = torch.randn(size=(V, feat_dim))

    embeddings = nn.Embedding(
        num_embeddings=V,
        embedding_dim=feat_dim
    )
    embeddings.weight = nn.Parameter(weights)
    embeddings.weight.requires_grad = False

    k_layers = 2
    num_neighbors = 3
    target_nodes = list(np.random.choice(nodes, replace=False, size=(2,)))
    batch_nodes, batch_adjList = random_sampling(adjList, target_nodes, k_layers, num_neighbors)

    dims = [feat_dim, feat_dim, 2*feat_dim, 64,
            64, 64, 2*64, 32
            ]

    print('Test PinSAGE')
    gcn = PinSAGE(dims, k_layers)
    h = gcn(batch_nodes, batch_adjList, embeddings)
    print(h.size())
    print(h)
    print('\n-------\n')

    print('Test NodeClassifier')
    num_classes = 5
    cls = NodeClassifier(num_classes, dims, k_layers, 32)
    logits = cls(batch_nodes, batch_adjList, embeddings)
    print(logits.size())
    print(logits)


if __name__ == '__main__':
    test_pinsage()





