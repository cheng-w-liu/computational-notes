import tensorflow as tf
from typing import Tuple
from dense import Dense

class Encoder(tf.Module):

    def __init__(self, input_dim, output_dim, name=None):
        super(Encoder, self).__init__(name=name)
        self.dense = Dense(input_dim, output_dim)

    def __call__(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        """
        :param inputs: a tuple of (nodes_embed, neighs_embed)
           nodes_embed: nodes embeddings, shape (num_of_nodes, hidden_dim)
           neighs_embed: embedding of each node's neighbors, shape (num_of_nodes, hidden_dim)
        :param training: bool
        :param mask:

        :return:
          processed embeddings, shape (num_of_nodes, output_dim)
        """

        nodes_embed, neighs_embed = inputs

        embed = tf.concat([nodes_embed, neighs_embed], axis=1)
        h = self.dense(embed)
        h, _ = tf.linalg.normalize(h, ord=2, axis=1)
        return h


def test_encoder():
    feat_dim = 3
    output_dim = 5
    n = 4
    nodes_embed = tf.random.normal(shape=(n, feat_dim))
    neighs_embed = tf.random.normal(shape=(n, feat_dim))

    encoder = Encoder(feat_dim, output_dim)

    encoder_inputs = (nodes_embed, neighs_embed)
    embed = encoder(encoder_inputs)

    print('\nProcessed embedding:')
    print(embed)
    print('\nTrainable variables:')
    print(encoder.trainable_variables)


if __name__ == '__main__':
    test_encoder()
