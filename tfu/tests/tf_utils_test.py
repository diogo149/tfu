import numpy as np
import tensorflow as tf
import nose.tools as nt
import tfu


def test_dimshuffle():
    for input_shape, pattern, output_shape in [
            [[1, 2, 3], [1, 2], [2, 3]],
            [[4, 5, 6], [2, "x", 1, "x", 0], [6, 1, 5, 1, 4]],
            [[4, 1, 6], [2, 0, "x"], [6, 4, 1]],
    ]:
        with tf.Session() as sess:
            in_val = tf.constant(np.random.rand(*input_shape))
            res_shape = tfu.get_shape_values(tfu.dimshuffle(in_val,
                                                            pattern))
            nt.assert_equal(output_shape,
                            res_shape)


def test_initialize_uninitialized_variables():
    with tf.Session() as sess:
        foo = tf.get_variable("foo",
                              shape=(),
                              initializer=tf.random_uniform_initializer())
        tfu.initialize_uninitialized_variables(sess)
        v1 = foo.eval()
        sess.run(tf.initialize_all_variables())
        v2 = foo.eval()
        nt.assert_not_equal(v1, v2)
        tfu.initialize_uninitialized_variables(sess)
        v3 = foo.eval()
        nt.assert_equal(v2, v3)
