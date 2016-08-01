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
        with tf.variable_scope("simple_rnn"):
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
        with tf.variable_scope("lstm"):
            x, = inputs
            h = state["h"]
            c = state["c"]
            logits = []
            for name in ["forget", "input", "output", "update"]:
                with tf.variable_scope(name):
                    logit = tf_utils.add_bias(
                        "bias",
                        tf_utils.linear("x_to_h", x, self.num_units) +
                        tf_utils.linear("h_to_h", h, self.num_units))
                logits.append(logit)
            f = tf.nn.sigmoid(logits[0])
            i = tf.nn.sigmoid(logits[1])
            o = tf.nn.sigmoid(logits[2])
            u = tf.tanh(logits[3])
            new_c = f * c + i * u
            new_h = tf.tanh(new_c) * o
            return {"h": new_h, "c": new_c}

