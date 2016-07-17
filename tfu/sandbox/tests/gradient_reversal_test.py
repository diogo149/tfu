import numpy as np
import tensorflow as tf
import nose.tools as nt
import tfu
from tfu.sandbox import gradient_reversal


def test_gradient_reversal():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            x = tf.constant(0.)
            y = gradient_reversal.gradient_reversal(x)
            g1, = tf.gradients(x, [x])
            g2, = tf.gradients(y, [x])
            nt.assert_equal(1.,
                            sess.run(g1))
            nt.assert_equal(-1.,
                            sess.run(g2))
