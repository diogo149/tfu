"""
from "Layer Normalization"
http://arxiv.org/abs/1607.06450
"""

import tensorflow as tf
import tfu


@tfu.hooked
def layer_normalization(x,
                        epsilon=1e-5,
                        bias_axis=-1,
                        scale_axis=-1,
                        name=None):
    # default epsilon taken from
    # https://github.com/ryankiros/layer-norm/blob/master/layers.py
    with tfu.variable_scope(name):
        with tfu.variable_scope("layer_normalization"):
            mean, variance = tf.nn.moments(
                x=x,
                axes=[dim for dim in range(tfu.ndim(x))
                      if dim != 0],
                keep_dims=True)
            z = (x - mean) / tf.sqrt(variance + epsilon)
            if scale_axis is not None:
                z = tfu.learned_scaling(z, axis=scale_axis)
            if bias_axis is not None:
                z = tfu.add_bias(z, axis=bias_axis)
            return z


class LNSimpleRNNStep(tfu.RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return self.num_units

    def call(self, inputs, state):
        with tfu.variable_scope("simple_rnn"):
            x, = inputs
            h = state
            with tfu.variable_scope("x_to_h"):
                x_to_h = tfu.linear(x, self.num_units, "linear")
                x_to_h = layer_normalization(x_to_h, "ln")
            with tfu.variable_scope("h_to_h"):
                h_to_h = tfu.linear(h, self.num_units, "linear")
                h_to_h = layer_normalization(h_to_h, "ln")
            logit = tfu.add_bias(x_to_h + h_to_h)

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
        with tfu.variable_scope("lstm"):
            x, = inputs
            h = state["h"]
            c = state["c"]

            multi_names = ["forget", "input", "output", "update"]
            multi_units = [self.num_units] * 4
            with tfu.variable_scope("x_to_h"):
                x_logits = tfu.multi_linear(names=multi_names,
                                            tensor=x,
                                            num_units=multi_units,
                                            split_output=False)
                x_logits = layer_normalization(x_logits, "ln")
                x_logits = tfu.split_axis(x_logits, axis=-1, sizes=multi_units)
            with tfu.variable_scope("h_to_h"):
                h_logits = tfu.multi_linear(names=multi_names,
                                            tensor=h,
                                            num_units=multi_units,
                                            split_output=False)
                h_logits = layer_normalization(h_logits, "ln")
                h_logits = tfu.split_axis(h_logits, axis=-1, sizes=multi_units)
            logits = []
            for name, x_logit, h_logit in zip(multi_names, x_logits, h_logits):
                with tfu.variable_scope(name):
                    logit = tfu.add_bias(x_logit + h_logit)
                logits.append(logit)
            f = tf.nn.sigmoid(logits[0])
            i = tf.nn.sigmoid(logits[1])
            o = tf.nn.sigmoid(logits[2])
            u = tf.tanh(logits[3])
            new_c = f * c + i * u
            new_h = tf.tanh(layer_normalization(new_c, "cell_ln")) * o
            return {"h": new_h, "c": new_c}
