"""
http://arxiv.org/abs/1504.00941
"""
import numpy as np
import tensorflow as tf
import tfu


def irnn_hook(nonlin=tf.nn.relu):
    def identity_init(metadata):
        shape = metadata["shape"]
        assert len(shape) == 2
        assert shape[0] == shape[1]
        return np.identity(shape[0])

    def replace_nonlinearity(hs):
        return nonlin(hs.kwargs["logit"])

    def irnn_hook_inner(hs):
        # prepend so that weight init applies after other hooks (and overwrites
        # other weight inits)
        hs.hooks = [
            tfu.inits.set_weight_init(
                identity_init,
                variable_scope=["simple_rnn", "h_to_h"]),
            tfu.filter_dsl(replace_nonlinearity,
                           key=["nonlinearity"],
                           variable_scope=["simple_rnn"]),
        ] + hs.hooks
        return hs()

    return irnn_hook_inner
