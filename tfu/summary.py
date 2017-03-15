"""
wrapped versions of tf.summary that also register_summary
"""

import tensorflow as tf

from . import base


def scalar(name, tensor, required=False, **metadata):
    summ = tf.summary.scalar(name, tensor)
    base.register_summary(summ, required=required, **metadata)
    return summ


def histogram(name, values, required=False, **metadata):
    summ = tf.summary.histogram(name, values)
    base.register_summary(summ, required=required, **metadata)
    return summ

# TODO add more wrappers
