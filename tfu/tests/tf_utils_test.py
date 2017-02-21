import numpy as np
import tensorflow as tf
import nose.tools as nt
import tfu


def test_initialize_uninitialized_variables():
    with tf.Session() as sess:
        foo = tf.get_variable("foo",
                              shape=(),
                              initializer=tf.random_uniform_initializer())
        tfu.initialize_uninitialized_variables(sess)
        v1 = foo.eval()
        sess.run(tf.global_variables_initializer())
        v2 = foo.eval()
        nt.assert_not_equal(v1, v2)
        tfu.initialize_uninitialized_variables(sess)
        v3 = foo.eval()
        nt.assert_equal(v2, v3)
