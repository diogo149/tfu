import six
import du
import tensorflow as tf

from . import base


def dump_variables(filename=None, metadata=None):
    if metadata is None:
        metadata = {}
    result = {}
    variables = base.find_variables(**metadata)
    # HACK assumes default session
    values = tf.get_default_session().run(variables)
    for var, value in zip(variables, values):
        result[var.name] = value
    if filename is not None:
        du.io_utils.pickle_dump(result, filename)
    return result


def load_variables(filename_or_obj):
    # TODO flag to make sure all keys are present
    # TODO flag to verify all shareds are loaded
    # TODO flag to verify shapes
    if isinstance(filename_or_obj, six.string_types):
        obj = du.io_utils.pickle_load(filename_or_obj)
    else:
        obj = filename_or_obj

    # TODO batch loading
    for var in base.find_variables():
        if var.name in obj:
            # HACK assumes default session
            tf.assign(var, obj[var.name]).eval()

    return obj
