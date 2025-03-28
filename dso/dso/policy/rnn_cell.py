import tensorflow as tf


class NoisyLSTMCell(tf.nn.rnn_cell.LSTMCell):
    """LSTM Cell with Factorised Gaussian Noise."""

    def __init__(self, num_units, initializer=None, sigma_init=0.017, **kwargs):
        super().__init__(num_units, initializer=initializer, **kwargs)
        self.sigma_init = sigma_init

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            outputs, new_state = super().__call__(inputs, state, scope)

            output_size = int(outputs.get_shape()[-1])

            # Parameters for factorised Gaussian noise
            mu = tf.get_variable("mu", shape=[output_size], initializer=tf.zeros_initializer())
            sigma = tf.get_variable("sigma", shape=[output_size], initializer=tf.constant_initializer(self.sigma_init))

            epsilon = tf.random.normal(shape=[output_size])
            noisy_output = outputs + mu + sigma * epsilon

        return noisy_output, new_state
