"""
from "Layer Normalization"
http://arxiv.org/abs/1607.06450
"""

import numpy as np
import tensorflow as tf
import tfu


def layer_normalization(name, tensor, epsilon=1e-4):
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
