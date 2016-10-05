"""
from "Weight Normalization: A Simple Reparameterization to Accelerate Training
of Deep Neural Networks"
http://arxiv.org/abs/1602.07868
"""

import numpy as np
import tensorflow as tf
import tfu


def weight_normalization_hook(**filter_dsl_kwargs):
    """
    applies weight normalization using a custom initialization (using the norm
    of the original weight tensor, so that WN cancels out initially)

    NOTE: requires sequentially initializing variables
    (eg. with tfu.sequentially_initialize_all_variables or
    tfu.hooks.auto_initialize_variables)
    """

    def apply_weight_normalization(hs):
        if "in_axes" in hs.kwargs and "out_axes" in hs.kwargs:
            out_axes = hs.kwargs["out_axes"]
            w = hs()
            pattern = []
            axes_to_sum = []
            g_shape = []
            for dim, s in enumerate(hs.kwargs["shape"]):
                if dim not in out_axes:
                    axes_to_sum.append(dim)
                    pattern.append("x")
                else:
                    pattern.append(out_axes.index(dim))
                    g_shape.append(s)

            norm = tf.sqrt(tf.reduce_sum(tf.square(w), axes_to_sum))

            with tfu.variable_scope("weight_normalization"):
                g = tfu.get_variable("g",
                                     # don't need to provide shape if
                                     # initializer is a constant
                                     # shape=g_shape,
                                     dtype=w.dtype,
                                     initializer=norm,
                                     trainable=True,
                                     weight_normalization_g=True)

            scale = g / norm
            return w * tfu.dimshuffle(scale, pattern=pattern)
        else:
            return hs()

    return tfu.filter_dsl(apply_weight_normalization,
                          key="get_variable",
                          **filter_dsl_kwargs)
