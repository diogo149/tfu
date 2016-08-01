"""
from "Layer Normalization"
http://arxiv.org/abs/1607.06450
"""

import tensorflow as tf
import tfu


def layer_normalization(name, tensor, epsilon=1e-5):
    # default epsilon taken from
    # https://github.com/ryankiros/layer-norm/blob/master/layers.py
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(
            x=tensor,
            axes=[dim for dim in range(tfu.ndim(tensor))
                  if dim != 0],
            keep_dims=True)
        z = (tensor - mean) / tf.sqrt(variance + epsilon)
        z = tfu.learned_scaling("scale", z)
        z = tfu.add_bias("bias", z)
        return z


class LNSimpleRNNStep(tfu.RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return self.num_units

    def call(self, inputs, state):
        x, = inputs
        h = state
        logit = tfu.add_bias(
            "bias",
            tfu.linear("x_to_h", x, self.num_units) +
            tfu.linear("h_to_h", h, self.num_units))
        logit = layer_normalization("ln", logit)

        @tfu.hooked
        def nonlinearity(logit):
            return tf.tanh(logit)

        return nonlinearity(logit=logit)
