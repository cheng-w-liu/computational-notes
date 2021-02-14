import tensorflow as tf
from typing import Dict, List, Tuple
from dense import Dense

class Aggregator(tf.Module):

    def __init__(self, input_dim, agg_method='mean', name=None):
        super(Aggregator, self).__init__(name=name)
        assert agg_method in ['mean', 'max', 'pool']
        self.agg_method = agg_method
        if agg_method == 'pool':
            self.dense = Dense(input_dim, input_dim, activation='sigmoid')

    def __call__(self, inputs: Tuple[int, List[int], Dict[int, tf.Tensor]]):
        """
        :param inputs: a tuple of (u, neighs, features)
          u: int, the node whose neighbors we care about
          neighs: List[int], a list of sampled nodes representing u's neighbors
          features: Dict[int, tf.Tensor], a lookup table that maps to a node's feature. Acts as h^{k-1}_u
        :param training

        :return:
          aggregated neighbor embedding. Acts as h^{k}_{N(u)}
        """

        u, neighs, features = inputs

        neighs_features = tf.stack(
            [features[v] for v in neighs],
            axis=0
        )

        if self.agg_method == 'mean':
            aggr_neighs = tf.reduce_mean(neighs_features, axis=0)
        elif self.agg_method == 'max':
            aggr_neighs = tf.reduce_max(neighs_features, axis=0)
        elif self.agg_method == 'pool':
            aggr_neighs = tf.reduce_max(self.dense(neighs_features), axis=0)
        else:
            raise ValueError(f'Unknown aggregate method: {self.agg_method} ')

        return aggr_neighs


def test_aggregator():
    adjList = {
        0: [1, 2, 3],
        1: [2, 3, 2],
        2: [2, 2, 2],
        3: [1, 1, 1]
    }

    V = len(adjList)
    feat_dim = 2
    weights = tf.random.normal(shape=(V, feat_dim))
    embeddings = tf.keras.layers.Embedding(
        V,
        feat_dim,
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=False
    )
    features_map = {v: embeddings(v) for v in range(V)}
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
    print(pool_aggregator.trainable_variables)


if __name__ == '__main__':
    test_aggregator()

