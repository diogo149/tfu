import tensorflow as tf

from . import base
from . import tf_utils
from .rnn_step import RNNStep


class SimpleRNNStep(RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return self.num_units

    def call(self, inputs, state):
        x, = inputs
        h = state
        logit = tf_utils.add_bias(
            "bias",
            tf_utils.linear("x_to_h", x, self.num_units) +
            tf_utils.linear("h_to_h", h, self.num_units))

        @base.hooked
        def nonlinearity(logit):
            return tf.tanh(logit)

        return nonlinearity(logit=logit)


class LSTMStep(RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return {"h": self.num_units, "c": self.num_units}

    def call(self, inputs, state):
        x, = inputs
        h = state["h"]
        c = state["c"]
        forget_logit = tf_utils.add_bias(
            "forget_bias",
            tf_utils.linear("forget_x", x, self.num_units) +
            tf_utils.linear("forget_h", h, self.num_units))
        input_logit = tf_utils.add_bias(
            "input_bias",
            tf_utils.linear("input_x", x, self.num_units) +
            tf_utils.linear("input_h", h, self.num_units))
        output_logit = tf_utils.add_bias(
            "output_bias",
            tf_utils.linear("output_x", x, self.num_units) +
            tf_utils.linear("output_h", h, self.num_units))
        update_logit = tf_utils.add_bias(
            "update_bias",
            tf_utils.linear("update_x", x, self.num_units) +
            tf_utils.linear("update_h", h, self.num_units))
        f = tf.nn.sigmoid(forget_logit)
        i = tf.nn.sigmoid(input_logit)
        o = tf.nn.sigmoid(output_logit)
        u = tf.tanh(update_logit)
        new_c = f * c + i * u
        new_h = tf.tanh(new_c) * o
        return {"h": new_h, "c": new_c}
