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
        with base.variable_scope("simple_rnn"):
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


class LSTMStep_v1(RNNStep):

    """
    slightly slower implementation of LSTM that still
    may be useful to fork off of
    """

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return {"h": self.num_units, "c": self.num_units}

    def call(self, inputs, state):
        with base.variable_scope("lstm"):
            x, = inputs
            h = state["h"]
            c = state["c"]
            logits = []
            for name in ["forget", "input", "output", "update"]:
                with base.variable_scope(name):
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


class LSTMStep(RNNStep):

    def __init__(self, num_units):
        self.num_units = num_units

    def state_specification(self):
        return {"h": self.num_units, "c": self.num_units}

    def call(self, inputs, state):
        with base.variable_scope("lstm"):
            x, = inputs
            h = state["h"]
            c = state["c"]

            multi_names = ["forget", "input", "output", "update"]
            multi_units = [self.num_units] * 4
            with base.variable_scope("x_to_h"):
                x_logits = tf_utils.multi_linear(names=multi_names,
                                                 tensor=x,
                                                 num_units=multi_units)
            with base.variable_scope("h_to_h"):
                h_logits = tf_utils.multi_linear(names=multi_names,
                                                 tensor=h,
                                                 num_units=multi_units)
            logits = []
            for name, x_logit, h_logit in zip(multi_names, x_logits, h_logits):
                with base.variable_scope(name):
                    logit = tf_utils.add_bias("bias", x_logit + h_logit)
                logits.append(logit)
            f = tf.nn.sigmoid(logits[0])
            i = tf.nn.sigmoid(logits[1])
            o = tf.nn.sigmoid(logits[2])
            u = tf.tanh(logits[3])
            new_c = f * c + i * u
            new_h = tf.tanh(new_c) * o
            return {"h": new_h, "c": new_c}
