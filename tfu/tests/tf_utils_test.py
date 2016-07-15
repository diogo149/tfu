import numpy as np
import tensorflow as tf
import nose.tools as nt
import tf_utils as tfu


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
