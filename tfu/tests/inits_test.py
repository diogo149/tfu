import numpy as np
import tensorflow as tf
import nose.tools as nt
import tfu


def test_scale_inits():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tfu.temporary_hook(tfu.inits.scale_inits(3.0)):
                b = tfu.get_variable("b",
                                     shape=(),
                                     initializer=tf.constant_initializer(2.0))
            sess.run(tf.initialize_all_variables())
            res = sess.run(b)
            assert res == 6.0
