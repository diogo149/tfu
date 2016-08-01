"""
from "Layer Normalization"
http://arxiv.org/abs/1607.06450
"""

import tensorflow as tf
import tfu


@tfu.hooked
def layer_normalization(name,
                        tensor,
                        epsilon=1e-5,
                        include_bias=False,
                        include_scale=True):
    # default epsilon taken from
    # https://github.com/ryankiros/layer-norm/blob/master/layers.py
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(
            x=tensor,
            axes=[dim for dim in range(tfu.ndim(tensor))
                  if dim != 0],
            keep_dims=True)
        z = (tensor - mean) / tf.sqrt(variance + epsilon)
        if include_scale:
            z = tfu.learned_scaling("scale", z)
        if include_bias:
            z = tfu.add_bias("bias", z)
        return z


class LNSimpleRNNStep(tfu.RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return self.num_units

    def call(self, inputs, state):
        with tf.variable_scope("simple_rnn"):
            x, = inputs
            h = state
            with tf.variable_scope("x_to_h"):
                x_to_h = tfu.linear("linear", x, self.num_units)
                x_to_h = layer_normalization("ln", x_to_h)
            with tf.variable_scope("h_to_h"):
                h_to_h = tfu.linear("linear", h, self.num_units)
                h_to_h = layer_normalization("ln", h_to_h)
            logit = tfu.add_bias("bias", x_to_h + h_to_h)

            @tfu.hooked
            def nonlinearity(logit):
                return tf.tanh(logit)

            return nonlinearity(logit=logit)


class LNLSTMStep(tfu.RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return {"h": self.num_units, "c": self.num_units}

    def call(self, inputs, state):
        with tf.variable_scope("lstm"):
            x, = inputs
            h = state["h"]
            c = state["c"]

            multi_names = ["forget", "input", "output", "update"]
            multi_units = [self.num_units] * 4
            with tf.variable_scope("x_to_h"):
                x_logits = tfu.multi_linear(names=multi_names,
                                            tensor=x,
                                            num_units=multi_units,
                                            split_output=False)
                x_logits = layer_normalization("ln", x_logits)
                x_logits = tfu.split_axis(x_logits, axis=-1, sizes=multi_units)
            with tf.variable_scope("h_to_h"):
                h_logits = tfu.multi_linear(names=multi_names,
                                            tensor=h,
                                            num_units=multi_units,
                                            split_output=False)
                h_logits = layer_normalization("ln", h_logits)
                h_logits = tfu.split_axis(h_logits, axis=-1, sizes=multi_units)
            logits = []
            for name, x_logit, h_logit in zip(multi_names, x_logits, h_logits):
                with tf.variable_scope(name):
                    logit = tfu.add_bias("bias", x_logit + h_logit)
                logits.append(logit)
            f = tf.nn.sigmoid(logits[0])
            i = tf.nn.sigmoid(logits[1])
            o = tf.nn.sigmoid(logits[2])
            u = tf.tanh(logits[3])
            new_c = f * c + i * u
            new_h = tf.tanh(layer_normalization("cell_ln", new_c)) * o
            return {"h": new_h, "c": new_c}
