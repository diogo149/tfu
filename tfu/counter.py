import tensorflow as tf

from . import base

# ################################### base ###################################


class GlobalStepCounter(object):

    def __init__(self,
                 sess=None,
                 expected_count=None,
                 progress_dtype=tf.float32):
        self.sess = sess
        self.expected_count = expected_count
        self.progress_dtype = progress_dtype

        self._count = base.get_variable(name="global_step_counter",
                                        dtype=tf.int64,
                                        initial_value=0)
        self._count_value = 0
        self._new_count = self._count + 1
        self._step_op = tf.assign(self._count, self._new_count)
        if expected_count is not None:
            self._progress = (tf.cast(self._count, progress_dtype) /
                              tf.cast(expected_count, progress_dtype))
            self._bounded_progress = tf.clip_by_value(self._progress, 0, 1)

    def set_session(self, sess):
        self.sess = sess

    def step(self, sess=None):
        if sess is None:
            sess = self.sess
        if sess is None:
            sess = tf.get_default_session()
        _, self._count_value = sess.run([self._step_op, self._new_count])

    def sync_count_value(self, sess=None):
        if sess is None:
            sess = self.sess
        if sess is None:
            sess = tf.get_default_session()
        self._count_value = sess.run(self._count)

    def as_default(self):
        # TODO implement
        raise NotImplementedError()


def make_default_counter(expected_count=None):
    assert base.default_graph_state().global_step_counter is None
    base.default_graph_state().global_step_counter = GlobalStepCounter(
        expected_count=expected_count,
    )
    return base.default_graph_state().global_step_counter


def get_default_counter():
    return base.default_graph_state().global_step_counter


def set_session(sess):
    return get_default_counter().set_session(sess)


def step(sess=None):
    """
    note: should be done at the beginning of each iteration
    (so the count starts at 1)
    """
    return get_default_counter().step(sess=sess)


def get_count():
    """
    returns the count as a tf int64 tensor
    """
    return get_default_counter()._count


def get_count_value():
    """
    returns the count as a python value
    """
    return get_default_counter()._count_value


def get_progress(dtype=tf.float32):
    """
    returns progress as a tensor (mapping the first count to 1/expected_count
    and expected_count to 1)
    """
    assert get_default_counter().expected_count is not None
    return get_default_counter()._progress


def get_bounded_progress():
    """
    returns progress as a tensor between 0 and 1
    """
    assert get_default_counter().expected_count is not None
    return get_default_counter()._bounded_progress


def sync_count_value(sess=None):
    """
    loads count value from tensorflow session

    use case: loading previous count value from serialized state
    """
    get_default_counter().sync_count_value()
    pass

# ############################ utility functions ############################


def to_scaled_bounded_progress(end_progress):
    """
    given a target end progress ratio, converts to a scale when 0 maps to 0
    progress and 1 maps to the end progress
    """
    return tf.clip_by_value(get_progress() / end_progress, 0, 1)


def progress_cond(progress, if_above, if_below):
    return tf.cond(get_progress() > progress, if_above, if_below)


def discrete_scale_schedule(x, scale, thresholds):
    """
    multiplies x by scale  every time a threshold is crossed
    """
    for threshold in thresholds:
        x = progress_cond(threshold, lambda: scale * x, lambda: x)
    return x
