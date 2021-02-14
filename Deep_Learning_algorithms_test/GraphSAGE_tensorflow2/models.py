import tensorflow as tf
from aggregator import Aggregator
from encoder import Encoder
from dense import Dense

from typing import List


class GraphSAGE(tf.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], K: int=2, agg_method: str='mean', name=None):
        super(GraphSAGE, self).__init__(name=name)
        assert agg_method in ['mean', 'max', 'pool']
        assert len(hidden_dims) == K
        dims = [input_dim] + hidden_dims
        self.aggregators = [Aggregator(in_dim, agg_method) for in_dim in dims[:-1]]
        self.encoders = [Encoder(2*in_dim, out_dim) for (in_dim, out_dim) in zip(dims[:-1], dims[1:])]
        self.K = K

    def __call__(self, inputs):
        """
        :param inputs: a tuple of (batch_nodes, batch_adjList, features_map)

          batch_nodes: List[List[int]], outer list has length K.
                      Each list contains the nodes we want to generate representation for

          batch_adjList: List[Dict[int, List[int]]], length K.
                      Each entry is an adjacency-list representation of a sampled graph.

            Notes:
            batch_adjList[k]: the sampled graph needed in order to generate representations for the nodes
                              specified in batch_nodes[k+1]

            batch_adjList[K] is a dummy entry

            batch_nodes[0] corresponds to `B^0`, the initial nodes needed in order to generate
                             representation for the nodes in B^K

          features_map: Embedding

        :return:
          encoded node representation
        """
        batch_nodes, batch_adjList, features_map = inputs
        assert len(batch_nodes) == len(batch_adjList)
        assert len(batch_nodes) == self.K + 1

        h = {int(u): features_map(int(u)) for u in batch_nodes[0]}  # acts as h^{k-1}, i.e., the features in the previous "layer"

        for k in range(1, self.K + 1):  # k = 1, ..., K
            neighs = {}
            nodes = {}
            for u in batch_nodes[k]:
                aggre_inputs = (u, batch_adjList[k-1][u], h)
                neighs[u] = self.aggregators[k-1](aggre_inputs)  # h^k_{N(u)}
                nodes[u] = h[u]  # h^{k-1}_u

            temp_id2node = {i: node for i, node in enumerate(batch_nodes[k])}

            nodes_embed = tf.stack(
                [nodes[u] for u in batch_nodes[k]],
                axis=0
            )

            neighs_embed = tf.stack(
                [neighs[u] for u in batch_nodes[k]],
                axis=0
            )

            encoder_inputs = (nodes_embed, neighs_embed)
            embed = self.encoders[k-1](encoder_inputs)
            h = {temp_id2node[i]: embed[i] for i in range(embed.shape[0])}

        h = tf.stack(
            [h[u] for u in batch_nodes[self.K]],
            axis=0
        )

        return h


class NodeClassifier(tf.Module):
    """
    A classifier that uses GraphSAGE to derive the representation of the node and then uses
    the embedding to predict the class of a node
    """

    def __init__(self, num_classes: int, input_dim: int, hidden_dims: List[int], K: int=2, agg_method: str='mean', name=None):
        super(NodeClassifier, self).__init__(name=name)
        self.graph_sage = GraphSAGE(input_dim, hidden_dims, K, agg_method, name)
        self.dense = Dense(hidden_dims[-1], num_classes)

    def __call__(self, inputs):
        """
        :param inputs: a tuple of (batch_nodes, batch_adjList, features_map)

          batch_nodes: List[List[int]], outer list has length K.
                      Each list contains the nodes we want to generate representation for

          batch_adjList: List[Dict[int, List[int]]], length K.
                      Each entry is an adjacency-list representation of a sampled graph.

            Notes:
            batch_adjList[k]: the sampled graph needed in order to generate representations for the nodes
                              specified in batch_nodes[k+1]

            batch_adjList[K] is a dummy entry

            batch_nodes[0] corresponds to `B^0`, the initial nodes needed in order to generate
                             representation for the nodes in B^K

          features_map: Embedding

        :return:
          logits corresponding to the class probabilities
        """
        h = self.graph_sage(inputs)
        logits = self.dense(h)
        return logits
