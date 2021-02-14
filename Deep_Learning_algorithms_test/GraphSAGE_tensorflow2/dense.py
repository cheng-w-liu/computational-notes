import tensorflow as tf

class Dense(tf.Module):

    def __init__(self, input_dim, output_dim, activation='none', name=None):
        super(Dense, self).__init__(name=name)
        assert activation in ['none', 'sigmoid', 'relu']
        init = tf.keras.initializers.glorot_uniform()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = tf.Variable(
            initial_value=init(shape=(input_dim, output_dim)),
            trainable=True,
            name='w'
        )
        self.b = tf.Variable(
            initial_value=tf.zeros(shape=[output_dim]),
            trainable=True,
            name='b'
        )
        self.activation = activation

    def __call__(self, x: tf.Tensor):
        """
        :param x: tf.Tensor
        :return: tf.Tensor
        """
        y = tf.matmul(x, self.w) + self.b
        if self.activation == 'none':
            return y
        elif self.activation == 'sigmoid':
            return tf.math.sigmoid(y)
        elif self.activation == 'relu':
            return tf.nn.relu(y)
        else:
            raise ValueError(f'Unsupported activation method: {self.activation} ')
