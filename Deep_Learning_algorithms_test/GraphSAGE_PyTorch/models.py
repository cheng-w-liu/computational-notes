import torch
import torch.nn as nn
from typing import Tuple, List, Dict

from aggregator import Aggregator
from encoder import Encoder

class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dims, K=2, agg_method='mean'):
        super(GraphSAGE, self).__init__()
        assert len(hidden_dims) == K
        assert agg_method in ['mean', 'max', 'pool']
        dims = [input_dim] + hidden_dims
        self.aggregators = nn.ModuleList([Aggregator(in_dim, agg_method) for in_dim in dims[:-1]])
        self.encoders = nn.ModuleList([Encoder(2*in_dim, out_dim) for (in_dim, out_dim) in zip(dims[:-1], dims[1:])])
        self.K = K

    def forward(self, inputs: Tuple[List[List[int]], List[Dict[int, List[int]]], torch.nn.Embedding]) -> torch.Tensor:
        """
        :param inputs: a tuple of (batch_nodes, batch_adjList, features_map)

          - batch_nodes: List[List[int]], outer list has length K.
                      Each list contains the nodes we want to generate representation for

          - batch_adjList: List[Dict[int, List[int]]], length K.
                      Each entry is an adjacency-list representation of a sampled graph.

            Notes:
            batch_adjList[k]:
               -- The sampled graph needed in order to generate representations for the nodes
                    specified in batch_nodes[k+1]
               -- In other words, batch_nodes[k'] is generated from the graph batch_adjList[k'-1]

            batch_adjList[K] is a dummy entry

            batch_nodes[0] corresponds to `B^0`, the initial nodes needed in order to generate
                             representation for the nodes in B^K

          - features_map: nn.Embedding

        :return:
          encoded node representation. shape (num_nodes, hidden_dim)
        """
        batch_nodes, batch_adjList, features_map = inputs
        assert len(batch_nodes) == len(batch_adjList)
        assert len(batch_nodes) == self.K + 1

        h = {u: features_map(torch.tensor(u)) for u in batch_nodes[0]}

        for k in range(1, self.K + 1):  # k = 1,..., K
            neighs = {}
            nodes = {}
            for u in batch_nodes[k]:
                aggre_inputs = (u, batch_adjList[k-1][u], h)
                neighs[u] = self.aggregators[k-1](aggre_inputs)  # h^k_{N(u)}
                nodes[u] = h[u]  # h^{k-1}_u

            temp_id2node = {i: node for i, node in enumerate(batch_nodes[k])}

            nodes_embed = torch.stack(
                [nodes[u] for u in batch_nodes[k]],
                dim=0
            )

            neighs_embed = torch.stack(
                [neighs[u] for u in batch_nodes[k]],
                dim=0
            )

            encoder_inputs = (nodes_embed, neighs_embed)
            embed = self.encoders[k-1](encoder_inputs)
            h = {temp_id2node[i]: embed[i] for i in range(embed.size(0))}

        h = torch.stack(
            [h[u] for u in batch_nodes[self.K]],
            dim=0
        )

        return h



class NodeClassifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dims, K=2, agg_method='mean'):
        super(NodeClassifier, self).__init__()
        self.num_classes = num_classes
        self.graph_sage = GraphSAGE(input_dim, hidden_dims, K, agg_method)
        self.linear_layer = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, inputs: Tuple[List[List[int]], List[Dict[int, List[int]]], torch.nn.Embedding]) -> torch.Tensor:
        """
        :param inputs: the same input format as in GraphSAGE's forward parameter
        :return: logits, shape (num_nodes, num_classes)
        """
        h = self.graph_sage(inputs)
        logits = self.linear_layer(h)
        return logits

