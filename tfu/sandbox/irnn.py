"""
http://arxiv.org/abs/1504.00941
"""
import numpy as np
import tensorflow as tf
import tfu


def irnn_hook(nonlin=tf.nn.relu):
    def identity_init(**_):
        def inner(shape, dtype):
            assert len(shape) == 2
            assert shape[0] == shape[1]
            return np.identity(shape[0])

        return inner

    def replace_nonlinearity(hs):
        return nonlin(hs.kwargs["logit"])

    def inner(hs):
        hs.hooks += [
            tfu.inits.set_weight_init(identity_init,
                                      variable_scope=["simple_rnn", "h_to_h"]),
            tfu.filter_dsl(replace_nonlinearity,
                           key=["nonlinearity"],
                           variable_scope=["simple_rnn"]),
        ]
        return hs()

    return inner
