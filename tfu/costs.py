import tensorflow as tf

from . import utils
from . import base


def l2(params=None):
    if params is None:
        params = base.find_variables(weight=True)
    return utils.smart_sum([tf.nn.l2_loss(x) for x in params])
