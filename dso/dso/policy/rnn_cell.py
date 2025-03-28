import tensorflow as tf


class NoisyLSTMCell(tf.nn.rnn_cell.LSTMCell):
    """LSTM Cell with Factorised Gaussian Noise."""

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            outputs, new_state = super().__call__(inputs, state, scope)

            output_size = outputs.get_shape()[-1]

            # Parameters for factorised Gaussian noise
            mu = tf.get_variable("mu", shape=[output_size], initializer=tf.zeros_initializer())
            sigma = tf.get_variable("sigma", shape=[output_size], initializer=tf.constant_initializer(0.017))

            epsilon = tf.random.normal(shape=[output_size])
            noisy_output = outputs + mu + sigma * epsilon

        return noisy_output, new_state

class NoisyRNNCell(tf.contrib.rnn.LayerRNNCell):
    """RNN cell wrapper that adds factorised Gaussian noise to its parameters."""

    def __init__(self, cell, output_size, sigma_init=0.017):
        super().__init__()
        self.cell = cell
        self._output_size = output_size
        self.sigma_init = sigma_init

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(type(self).__name__):
            outputs, state = self.cell(inputs, state, scope=scope)

            with tf.variable_scope('perturbation_layer'):
                # Standard parameters
                theta = tf.get_variable("theta", shape=[outputs.shape[-1], self._output_size],
                                        initializer=tf.glorot_uniform_initializer())
                bias_theta = tf.get_variable("bias_theta", shape=[self._output_size],
                                             initializer=tf.zeros_initializer())

                # Factorised perturbation parameters
                sigma_w = tf.get_variable("sigma_w", shape=[outputs.shape[-1], 1],
                                          initializer=tf.constant_initializer(self.sigma_init))
                sigma_b = tf.get_variable("sigma_b", shape=[1, self._output_size],
                                          initializer=tf.constant_initializer(self.sigma_init))

                # Sample factorised noise
                epsilon_w = tf.random.normal([outputs.shape[-1], 1])
                epsilon_b = tf.random.normal([1, self._output_size])

                # Factorised perturbed weights and biases
                w = theta + sigma_w * epsilon_w
                b = bias_theta + sigma_b * epsilon_b

                logits = tf.matmul(outputs, w) + b

        return logits, state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)
