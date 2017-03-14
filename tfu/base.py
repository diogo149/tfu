import collections
import functools
import contextlib
import six
import tensorflow as tf

import du
import du.sandbox.summary

from . import utils

# ############################### state class ###############################


class GraphState(object):

    def __init__(self, graph=None):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.variable_metadata = {}
        self.current_metadata = {}
        self.hooks = []
        self.updates_accumulators = []
        self.summary_accumulators = []
        self.global_step_counter = None
        self.misc = {}


DEFAULT_GRAPH_STATE = [GraphState(graph=tf.get_default_graph())]


def default_graph_state():
    return DEFAULT_GRAPH_STATE[0]

# ################################ base hooks ############################


class HookedState(object):
    """
    container for the state of hooked function calls
    """

    def __init__(self, key, fn, args, kwargs):
        self.key = key
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        # make a copy of hooks
        # ---
        # rationale: this allows a hook to edit the list of hooks,
        # thus allowing hooks to be composed of other hooks
        self.hooks = list(default_graph_state().hooks)

    def __call__(self):
        if self.hooks:
            hook = self.hooks.pop()
            return hook(self)
        else:
            return self.fn(*self.args, **self.kwargs)

    def __repr__(self):
        return "HookedState({})".format(self.__dict__)


def hooked(fn):
    """
    decorator for hooked functions
    """

    @functools.wraps(fn)
    def hooked_inner(*args, **kwargs):
        hs = HookedState(
            key=fn.func_name,
            fn=fn,
            args=args,
            kwargs=kwargs
        )
        return hs()

    return hooked_inner


def add_hook(fn, location="outer"):
    # TODO does it ever make sense to have a hook twice?
    assert fn not in default_graph_state().hooks
    if location == "outer":
        default_graph_state().hooks.append(fn)
    elif location == "inner":
        default_graph_state().hooks.insert(0, fn)
    else:
        raise ValueError("Invalid hook location: %s" % location)


def remove_hook(fn):
    while fn in default_graph_state().hooks:
        default_graph_state().hooks.remove(fn)


@contextlib.contextmanager
def temporary_hook(fn, location="outer"):
    try:
        add_hook(fn, location=location)
        yield
    finally:
        remove_hook(fn)


@contextlib.contextmanager
def temporary_hooks(fns, location="outer"):
    try:
        for fn in fns:
            add_hook(fn, location=location)
        yield
    finally:
        for fn in fns:
            remove_hook(fn)

# ################## modified variable_scope / get_variable ##################


@contextlib.contextmanager
def variable_scope(name_or_scope=None,
                   default_name=None,
                   reuse=None,
                   **metadata):
    """
    wrapped version of variable_scope; differences:
    - less arguments available
    - allows specifying optional metadata
    """
    if default_name is None and name_or_scope is None:
        name_or_scope = tf.get_variable_scope()
    # HACK for variable names as scopes
    if isinstance(name_or_scope, six.string_types) and name_or_scope.endswith(":0"):
        name_or_scope = name_or_scope[:-len(":0")]

    with tf.variable_scope(name_or_scope=name_or_scope,
                           default_name=default_name,
                           reuse=reuse):
        prev_metadata = default_graph_state().current_metadata
        new_metadata = dict(prev_metadata)  # make a copy
        new_metadata.update(metadata)
        try:
            default_graph_state().current_metadata = new_metadata
            yield
        finally:
            default_graph_state().current_metadata = prev_metadata


@hooked
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initial_value=0,
                 **metadata):
    """
    wrapped version of tf.get_variable; differences:
    - less arguments available
    - allows specifying optional metadata
    - allows hooking of init
    - default to zero init
    - auto-converting dtype
    - only supporting initial value instead of initializer function
    """
    # make a copy of currrent metadata
    new_metadata = dict(default_graph_state().current_metadata)
    # overwrite with given metadata
    new_metadata.update(metadata)
    # also including these fields in the metadata because they may be
    # useful
    # e.g. for initialization
    new_metadata.update(dict(
        name=name,
        shape=shape,
        dtype=dtype,
        initial_value=initial_value,
    ))

    @hooked
    def get_initial_value(metadata):
        shape = metadata["shape"]
        initial_value = metadata["initial_value"]
        dtype = metadata["dtype"]
        if dtype is None:
            dtype = tf.float32
        # NOTE: we cast to dtype in this function as well, because the behavior
        # of arithmetic on some inits may change depending on the dtype
        # e.g. for int dtype, we might want to do integer arithmetic on it
        if shape is None:
            # assume initial value is already a tensor or numpy array
            assert (utils.is_tensor(initial_value) or
                    utils.is_ndarray(initial_value) or
                    utils.is_number(initial_value))
            return tf.cast(initial_value, dtype)
        else:
            # if shape is given, assume that the output is a scalar
            # to be created into the desired shape
            if initial_value == 0:
                return tf.zeros(shape, dtype)
            else:
                return initial_value * tf.ones(shape, dtype)

    new_initial_value = get_initial_value(new_metadata)
    if dtype is not None:
        new_initial_value = tf.cast(new_initial_value, dtype)

    # TODO stop using get_variable
    # only important functionality is adding to global and trainable variables
    # collections
    var = tf.get_variable(name=name,
                          shape=None,
                          dtype=None,
                          initializer=new_initial_value,
                          # defaulting trainable to False is not specified
                          trainable=new_metadata.get("trainable", False))
    default_graph_state().variable_metadata[var] = new_metadata
    return var

# ############################# variable search #############################


def variables(name=None, variable_scope=None, **metadata):
    """
    metadata based variable search
    """
    result = []
    for var, var_meta in default_graph_state().variable_metadata.items():
        var_path = var.name.split("/")

        # for name match, we require an exact match
        if name is not None:
            if var_path[-1] != name + ":0":
                continue

        if variable_scope is not None:
            vs_path = var_path[:-1]
            if isinstance(variable_scope, six.string_types):
                if variable_scope not in vs_path:
                    continue
            elif isinstance(variable_scope, (list, tuple)):
                matched = True
                for subscope in variable_scope:
                    if subscope in vs_path:
                        vs_path = vs_path[vs_path.index(subscope) + 1:]
                    else:
                        matched = False
                        break
                if not matched:
                    continue
            else:
                raise ValueError("wrong variable_scope type: %s" %
                                 variable_scope)

        for k, v in metadata.items():
            if var_meta.get(k) != v:
                break
        else:
            result.append(var)
    return result

# ################################# filters #################################


def wrap_filter(hook, filter_fn):
    """
    takes in a hook and only applies it when the given filter function
    returns a truthy value
    """
    def wrap_filter_inner(hs):
        if filter_fn(hs):
            return hook(hs)
        else:
            return hs()

    return wrap_filter_inner


def filter_dsl(hook,
               key=None,
               variable_scope=None,
               filter_fn=None):
    """
    takes in a hook and only applies it when the hooked state matches a
    DSL

    key:
    if a string, matches for an exact key match
    if a list, matches if the key is in the list

    variable_scope:
    if a string, matches if it is a subset of the current variable scope's name
    if a list, matches if it is a subsequence of the current variable scope's
    name split by "/"
    """
    # TODO do we want it to work on regexes instead of exact string matches

    def filter_dsl_inner(hs):
        if key is not None:
            if isinstance(key, six.string_types):
                if key != hs.key:
                    return hs()
            elif isinstance(key, list):
                if hs.key not in key:
                    return hs()
            else:

                raise ValueError("wrong key type: %s" % key)

        if variable_scope is not None:
            vs_name = tf.get_variable_scope().name
            if isinstance(variable_scope, six.string_types):
                if variable_scope not in vs_name:
                    return hs()
            elif isinstance(variable_scope, list):
                vs_list = vs_name.split("/")
                for subset in variable_scope:
                    if subset not in vs_list:
                        return hs()
                    vs_list = vs_list[vs_list.index(subset) + 1:]
            else:
                # TODO allow searching by metadata
                raise ValueError("wrong variable_scope type: %s" %
                                 variable_scope)

        if filter_fn is not None:
            if not filter_fn(hs):
                return hs()

        return hook(hs)

    return filter_dsl_inner


# ########################### tensorflow function ###########################

class TensorFlowFunctionOp(object):

    """
    base class for objects which want to be treated as an op by
    TensorFlowFunction
    """

    def to_op(self):
        raise NotImplementedError()

    def handle_result(self, op_res):
        pass


class TensorFlowFunction(object):

    def __init__(self,
                 sess=None,
                 inputs=None,
                 outputs=None,
                 ops=None,
                 options=None,
                 run_metadata=None,
                 name=None):
        """
        if name is given, function is timed
        """
        if sess is None:
            sess = tf.get_default_session()
        if inputs is None:
            inputs = {}
        if outputs is None:
            outputs = {}
        if ops is None:
            ops = []
        assert sess is not None
        assert isinstance(inputs, dict)
        assert isinstance(outputs, dict)
        assert isinstance(ops, list)

        self.sess = sess
        self.inputs = inputs
        self.outputs = outputs
        self.ops = ops
        self.options = options
        self.run_metadata = run_metadata
        self.name = name

        self._input_keys = list(inputs.keys())
        self._output_keys = []
        self._output_values = []
        for k, v in outputs.items():
            self._output_keys.append(k)
            self._output_values.append(v)

        self._op_fetches = []
        for op in ops:
            if isinstance(op, TensorFlowFunctionOp):
                op = op.to_op()
            self._op_fetches.append(op)

        self._fetches = self._output_values + self._op_fetches

    def __call__(self, datamap):
        if self.name is None:
            return self._internal_call(datamap)
        else:
            with du.timer(self.name):
                return self._internal_call(datamap)

    def _internal_call(self, datamap):
        if not self._input_keys:
            feed_dict = None
        else:
            feed_dict = {self.inputs[k]: datamap[k] for k in self._input_keys}

        res = self.sess.run(fetches=self._fetches,
                            feed_dict=feed_dict,
                            options=self.options,
                            run_metadata=self.run_metadata)

        output_res = res[:len(self.outputs)]
        ops_res = res[-len(self.ops):]

        for op, op_res in zip(self.ops, ops_res):
            if isinstance(op, TensorFlowFunctionOp):
                op.handle_result(op_res)

        return {k: v for k, v in zip(self._output_keys, output_res)}


tf_fn = TensorFlowFunction

# ########################### updates accumulator ###########################


class UpdatesAccumulator(TensorFlowFunctionOp):

    def __init__(self,
                 merge_strategy="raise",
                 variable_scope=None,
                 **metadata):
        # TODO implement filtering based on variable scope
        assert variable_scope is None
        self.merge_strategy = merge_strategy
        self.metadata = metadata
        self.updates = collections.OrderedDict()

    def __enter__(self):
        assert self not in default_graph_state().updates_accumulators
        default_graph_state().updates_accumulators.append(self)
        return self

    def __exit__(self, type, value, tb):
        while self in default_graph_state().updates_accumulators:
            default_graph_state().updates_accumulators.remove(self)
        # don't supress any exception
        return False

    def to_op(self):
        ops = []
        for k, v in self.updates.items():
            ops.append(tf.assign(k, v))
        return tf.group(*ops)


def register_updates(updates, required=False, **metadata):
    registered = False
    if isinstance(updates, list):
        updates = collections.OrderedDict(updates)
    assert isinstance(updates, collections.OrderedDict)
    for acc in default_graph_state().updates_accumulators:
        for k, v in acc.metadata.items():
            if metadata.get(k) != v:
                break
        else:
            for k, v in updates.items():
                # make sure we aren't overwritting anything
                if k not in acc.updates:
                    acc.updates[k] = v
                else:
                    if acc.merge_strategy == "raise":
                        raise ValueError("key already in updates accumulator: %s" %
                                         k)
                    elif acc.merge_strategy == "add":
                        acc.updates[k] = acc.updates[k] + v - k
                    elif acc.merge_strategy == "overwrite":
                        acc.updates[k] = v
                    else:
                        raise ValueError("unknown merge_strategy: %s" %
                                         acc.merge_strategy)
            registered = True
    if required and not registered:
        raise Exception("required update was not registered")

# ########################### summary accumulator ###########################


class SummaryAccumulator(TensorFlowFunctionOp):

    def __init__(self,
                 variable_scope=None,
                 **metadata):
        # TODO implement filtering based on variable scope
        assert variable_scope is None
        self.metadata = metadata
        self.summaries = []
        self.file_writers = []
        self.summary_printers = []
        self.history = []

    def __enter__(self):
        assert self not in default_graph_state().summary_accumulators
        default_graph_state().summary_accumulators.append(self)
        return self

    def __exit__(self, type, value, tb):
        while self in default_graph_state().summary_accumulators:
            default_graph_state().summary_accumulators.remove(self)
        # don't supress any exception
        return False

    def to_op(self):
        return tf.summary.merge(self.summaries)

    def handle_result(self, op_res):
        summ = tf.summary.Summary(op_res)
        summ_dict = utils.scalar_summary_to_dict(summ)
        for file_writer in self.file_writers:
            global_step = default_graph_state().global_step_counter._count_value
            file_writer.add_summary(summary=summ, global_step=global_step)
        for summary_printer in self.summary_printers:
            summary_printer.update(summ_dict)
            print(summary_printer.to_org_list())

    def add_file_writer(self, file_writer):
        if isinstance(file_writer, six.string_types):
            # if string, then it is a logdir
            file_writer = tf.summary.FileWriter(logdir=file_writer)
        self.file_writers.append(file_writer)

    def add_summary_printer(self, summary_printer):
        assert isinstance(summary_printer, du.sandbox.summary.Summary)
        self.summary_printers.append(summary_printer)


def register_summary(summary, required=False, **metadata):
    registered = False
    for acc in default_graph_state().summary_accumulators:
        for k, v in acc.metadata.items():
            if metadata.get(k) != v:
                break
        else:
            acc.summaries.append(summary)
            registered = True
    if required and not registered:
        raise Exception("required update was not registered")
