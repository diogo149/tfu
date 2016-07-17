import tensorflow as tf


@tf.RegisterGradient("GradientReversal")
def _gradient_reversal_grad(unused_op, grad):
    return [tf.neg(grad)]


def gradient_reversal(tensor):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "GradientReversal"}):
        return tf.identity(tensor)
