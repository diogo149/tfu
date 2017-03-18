"""
functions for wrapping a TensorFlowFunction
"""

import time
import numpy as np
import du

from . import utils
from . import base
from . import serialization

# ################################### misc ###################################


def add_callback(fn, callback):
    """
    calls a callback with the functions input and output
    after each call
    """

    def add_callback_inner(datamap, *args, **kwargs):
        res = fn(datamap, *args, **kwargs)
        callback(datamap, res)
        return res

    return add_callback_inner


def format_output_keys(fn, format):
    """
    transforms the keys of the resulting datamap by applying string formatting
    """

    def format_output_keys_inner(datamap, *args, **kwargs):
        res = fn(datamap, *args, **kwargs)
        return {(format % k): v for k, v in res.items()}

    return format_output_keys_inner


def output_to_ndarray(fn):
    """
    converts every value in the output of the function into a numpy array

    use case:
    - theano sometimes returns cuda arrays
    """

    def output_to_ndarray_inner(*args, **kwargs):
        res = fn(*args, **kwargs)
        return {k: np.array(v) for k, v in res.items()}

    return output_to_ndarray_inner


# ################################# monitor #################################


def time_call(fn, key="time"):
    """
    times the function call and adds a corresponding key with the time to
    the output
    """

    def time_call_inner(*args, **kwargs):
        start_time = time.time()
        res = fn(*args, **kwargs)
        assert key not in res
        res[key] = time.time() - start_time
        return res

    return time_call_inner


def update_summary_printer(fn, summary_printer):
    """
    updates a summary printer with results of the fn
    """
    def update_summary_printer_inner(*args, **kwargs):
        res = fn(*args, **kwargs)
        summary_printer.update(res)
        return res

    return update_summary_printer_inner

# ################################## batch ##################################


def split_input(fn, split_size, keys=None, scalar_merge="mean", strict=False):
    """
    Splits the input into multiple smaller inputs along axis 0,
    applies the inner function, and concatenates the results.

    Size of input must be a mulitple of split_size.

    scalar_merge:
    how scalar outputs should be merged together

    """

    def split_input_inner(datamap, *args, **kwargs):
        _keys = keys or datamap.keys()
        input_size = None
        for input_key in _keys:
            input_val = datamap[input_key]
            if input_size is None:
                input_size = len(input_val)
            else:
                assert len(input_val) == input_size
        assert input_size is not None
        # optimization to prevent copying and simply pass computation through
        # when splitting is a no-op
        if input_size == split_size:
            return fn(datamap, *args, **kwargs)
        if strict:
            assert (input_size % split_size) == 0
        inner_map = dict(datamap)
        results = []
        for i in range(int(np.ceil(input_size / split_size))):
            split_slice = slice(i * split_size, (i + 1) * split_size)
            for key in _keys:
                inner_map[key] = datamap[key][split_slice]
            result = fn(inner_map, *args, **kwargs)
            results.append(result)
        return utils.datamap_merge(results, scalar_merge=scalar_merge)

    return split_input_inner


def batch_pad(fn, batch_size, keys, axis=0):
    """
    pads variables with 0's to the specified batch size
    """

    def pad(arr):
        rem = arr.shape[axis] % batch_size
        if rem == 0:
            return arr
        else:
            pad_shape = [s if i != axis else (batch_size - rem)
                         for i, s in enumerate(arr.shape)]
            to_pad = np.zeros(pad_shape, dtype=arr.dtype)
            return np.concatenate([arr, to_pad], axis=axis)

    def batch_pad_inner(datamap, *args, **kwargs):
        new_datamap = dict(datamap)
        for key in keys:
            new_datamap[key] = pad(datamap[key])

        return fn(new_datamap, *args, **kwargs)

    return batch_pad_inner


def shuffle_input(fn, random_state=None):
    """
    randomly reorders items in input datamaps along 1st axis
    """
    # TODO take in subset of keys to shuffle
    def shuffle_input_inner(datamap, *args, **kwargs):
        new_datamap = du.tasks.tasks_utils.shuffle_datamap(
            datamap=datamap,
            random_state=random_state)
        return fn(new_datamap, *args, **kwargs)

    return shuffle_input_inner

# ################################### debug ###################################


def _check_nan_fn(name, nan_is_error, inf_is_error, big_is_error):
    def handle_error(error_type, k, v):
        msg = dict(
            msg="%s: found nan" % name,
            error_type=error_type,
            key=k,
            value=v
        )
        raise Exception(msg)

    def check_nan(key, arr):
        if nan_is_error:
            if np.isnan(np.min(arr)):
                handle_error("nan", key, arr)
        if inf_is_error:
            # OPTIMIZE could do np.isinf(np.max(np.abs(x)))
            # next check can also use np.max(np.abs(x)
            if np.any(np.isinf(arr)):
                handle_error("inf", key, arr)
        if big_is_error:
            if np.any(np.abs(arr) > 1e10):
                handle_error("big", key, arr)

    return check_nan


def output_nan_guard(fn,
                     nan_is_error=True,
                     inf_is_error=True,
                     big_is_error=True):
    """
    checks outputs for nan and raises an exception if any contain nan

    should be more efficient than theano.compile.nanguardmode.NanGuardMode,
    since it is only done on outputs and not intermediate computations
    """
    check_nan = _check_nan_fn(name="output_nan_guard",
                              nan_is_error=nan_is_error,
                              inf_is_error=inf_is_error,
                              big_is_error=big_is_error)

    def output_nan_guard_inner(*args, **kwargs):
        res = fn(*args, **kwargs)
        for k, v in res.items():
            check_nan(k, v)
        return res

    return output_nan_guard_inner


def variable_nan_guard(fn,
                       nan_is_error=True,
                       inf_is_error=True,
                       big_is_error=True):
    """
    checks variables for nan and raises an exception if any contain nan
    """

    check_nan = _check_nan_fn(name="variable_nan_guard",
                              nan_is_error=nan_is_error,
                              inf_is_error=inf_is_error,
                              big_is_error=big_is_error)

    def variable_nan_guard_inner(*args, **kwargs):
        res = fn(*args, **kwargs)
        for var in base.find_variables():
            # HACK assumes default session
            check_nan(var.name, var.eval())
        return res

    return variable_nan_guard_inner


class SaveLastInputsAndVariables(object):

    """
    keeps a history of inputs and variables (before calling
    the function)

    example:
    >>> fn = save_wrapper = thu.save_last_inputs_and_variables(fn, 5)
    >>> # use the fn
    >>> fn(some_input)
    >>> # to view the saved inputs:
    >>> save_handler.inputs_
    >>> # to view the final variable state
    >>> save_handler.variables_[-1]
    """

    def __init__(self,
                 fn,
                 num_inputs_to_save=5,
                 num_variables_to_save=None):
        """
        num_inputs_to_save:
        the number of inputs to save

        num_variables_to_save:
        the number of variables to save
        """
        # TODO take in metadata for queries
        self.fn = fn
        # TODO split into 2 wrappers that independently save inputs
        # and value dicts
        self.num_inputs_to_save = num_inputs_to_save
        if num_variables_to_save is None:
            num_variables_to_save = num_inputs_to_save
        self.num_variables_to_save = num_variables_to_save
        self.inputs_ = []
        self.shareds_ = []

    def __call__(self, datamap, *args, **kwargs):
        self.inputs_.append(datamap)
        if len(self.inputs_) > self.num_inputs_to_save:
            self.inputs_.pop(0)
        # TODO optimize when num_variables_to_save == 0
        self.shareds_.append(serialization.dump_variables())
        if len(self.shareds_) > self.num_variables_to_save:
            self.shareds_.pop(0)
        return self.fn(datamap, *args, **kwargs)


save_last_inputs_and_variables = SaveLastInputsAndVariables
