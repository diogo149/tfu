import numpy as np
import tensorflow as tf
import nose.tools as nt
import tfu
from tfu.sandbox import gradient


def test_gradient_reversal():
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            x = tf.constant(0.)
            y = gradient.gradient_reversal(x)
            g1, = tf.gradients(x, [x])
            g2, = tf.gradients(y, [x])
            nt.assert_equal(1.,
                            sess.run(g1))
            nt.assert_equal(-1.,
                            sess.run(g2))


def test_elementwise_gradient_clip():
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            x = tf.constant(3.)
            y1 = gradient.elementwise_gradient_clip(x, 0, 0.5)
            y2 = -gradient.elementwise_gradient_clip(x, -0.1, 1)
            g1, = tf.gradients(y1, [x])
            g2, = tf.gradients(y2, [x])
            nt.assert_equal(0.5,
                            sess.run(g1))
            np.testing.assert_almost_equal(-0.1,
                                           sess.run(g2))
