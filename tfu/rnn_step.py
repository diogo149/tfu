import tensorflow as tf

from . import utils
from . import base
from . import tf_utils


class RNNStep(object):

    """
    more general abstraction for RNN's than tf.nn.rnn_cell.RNNCell
    - allows keeping multiple states
      - eg. both cell and hidden in LSTM
    """

    def state_specification(self):
        """
        returns state specification as a DSL
        - int: corresponds to hidden state that is just a tensor with that many
               units
        - list: corresponds to a list of tensors with contents specifying shape
        - dict: corresponds to a dictionary with string keys and values
                specifying shape
        - tuple / tensorshape: hidden state with that exact shape (TODO)
        """
        raise NotImplementedError

    def call(self, inputs, state):
        """
        takes in state in an arbitrary format and returns state in the same
        format
        """
        raise NotImplementedError

    def __call__(self, inputs, state, scope=None):
        hs = base.HookedState(
            key=type(self).__name__,
            fn=self.call,
            args=(),
            kwargs={"inputs": inputs,
                    "state": state},
        )
        with tf.variable_scope(scope or type(self).__name__):
            return hs()

    def to_cell(self):
        return RNNStepToCell(self)

    def zero_state(self, state_specification=None, **kwargs):
        ss = state_specification
        if ss is None:
            ss = self.state_specification()
        if isinstance(ss, int):
            return tf.zeros(shape=[1, ss], **kwargs)
        elif isinstance(ss, (tuple, tf.TensorShape)):
            # TODO implement this
            raise NotImplementedError
        elif isinstance(ss, list):
            return [self.zero_state(s, **kwargs)
                    for s in ss]
        elif isinstance(ss, dict):
            return {k: self.zero_state(v, **kwargs)
                    for k, v in ss.items()}
        else:
            raise ValueError("Incorrect shape specification: %s" % ss)

    def variable_state(self,
                       state_specification=None,
                       batch_size=1,
                       **kwargs):
        ss = state_specification
        if ss is None:
            ss = self.state_specification()
        if isinstance(ss, int):
            return tf.get_variable(name="state",
                                   shape=[batch_size, ss],
                                   **kwargs)
        elif isinstance(ss, (tuple, tf.TensorShape)):
            # TODO implement this
            raise NotImplementedError
        elif isinstance(ss, list):
            results = []
            for idx, s in enumerate(ss):
                with tf.variable_scope(str(idx)):
                    results.append(self.variable_state(s, **kwargs))
            return results
        elif isinstance(ss, dict):
            results = {}
            for k, v in ss.items():
                with tf.variable_scope(k):
                    results[k] = self.variable_state(v, **kwargs)
            return results
        else:
            raise ValueError("Incorrect shape specification: %s" % ss)

    def state_to_cell_state(self, state, state_specification=None):
        ss = state_specification
        if ss is None:
            ss = self.state_specification()
        if isinstance(ss, (int, tuple, tf.TensorShape)):
            return (state,)
        elif isinstance(ss, list):
            results = ()
            for state_, s in zip(state, ss):
                results += self.state_to_cell_state(state_, s)
            return results
        elif isinstance(ss, dict):
            results = ()
            for k, v in sorted(ss.items(), key=lambda x: x[0]):
                state_ = state[k]
                results += self.state_to_cell_state(state_, v)
            return results
        else:
            raise ValueError("Incorrect shape specification: %s" % ss)

    def state_to_cell_output(self, state, state_specification=None):
        ss = state_specification
        if ss is None:
            ss = self.state_specification()
        if isinstance(ss, (int, tuple, tf.TensorShape)):
            return state
        elif isinstance(ss, list):
            return state[0]
        elif isinstance(ss, dict):
            return state["h"]
        else:
            raise ValueError("Incorrect shape specification: %s" % ss)

    def state_from_cell_state(self, cell_state, state_specification=None):
        ss = state_specification
        if ss is None:
            ss = self.state_specification()
        if isinstance(ss, (int, tuple, tf.TensorShape)):
            return cell_state
        elif isinstance(ss, list):
            # won't work for nested
            assert len(ss) == len(cell_state)
            return [self.state_from_cell_state(state_, s)
                    for state_, s in zip(cell_state, ss)]
        elif isinstance(ss, dict):
            # won't work for nested
            assert len(ss) == len(cell_state)
            results = {}
            for state_, (k, v) in zip(cell_state,
                                      sorted(ss.items(), key=lambda x: x[0])):
                results[k] = self.state_from_cell_state(state_, v)
            return results
        else:
            raise ValueError("Incorrect shape specification: %s" % ss)

    def cell_state_size(self):
        return self.state_to_cell_state(self.state_specification())

    def cell_output_size(self):
        return self.state_to_cell_output(self.state_specification())

    def apply_layer(self,
                    name,
                    inputs,
                    initial_state=None,
                    evaluation_type="unrolled"):
        if initial_state is None:
            initial_state = self.zero_state()

        if evaluation_type == "unrolled":
            results = tf_utils.rnn_reduce(name=name,
                                          rnn_fn=self,
                                          tensors=inputs,
                                          initial_state=initial_state)
            return results
        elif evaluation_type in {"rnn", "dynamic_rnn"}:
            cell = self.to_cell()
            initial_cell_state = self.state_to_cell_state(initial_state)

            # preprocess initial_cell_state for shape
            # by broadcasting 1 dim into batch size
            initial_cell_state = list(initial_cell_state)
            batch_size = utils.get_shape_symbolic(inputs[0])[1]
            for idx, s in enumerate(initial_cell_state):
                if utils.get_shape_values(s)[0] == 1:
                    s_shape = utils.get_shape_symbolic(s)
                    s_shape[0] = batch_size
                    zeros = tf.zeros(s_shape, dtype=s.dtype)
                    initial_cell_state[idx] = s + zeros
            initial_cell_state = tuple(initial_cell_state)

            if evaluation_type == "rnn":
                outputs, final_cell_state = tf.nn.rnn(
                    cell,
                    inputs=zip(*map(tf.unpack, inputs)),
                    initial_state=initial_cell_state)
            elif evaluation_type == "dynamic_rnn":
                outputs, final_cell_state = tf.nn.dynamic_rnn(
                    cell,
                    inputs=inputs,
                    initial_state=initial_cell_state,
                    time_major=True)
            return outputs, final_cell_state
        else:
            raise ValueError("Unknown evaluation_type: %s" % evaluation_type)

        return results

    def update_state_op(self, variable_state, new_state, **kwargs):
        variable_cell_state = self.state_to_cell_state(variable_state)
        new_cell_state = self.state_to_cell_state(new_state)
        ops = [tf.assign(vs, ns, **kwargs)
               for vs, ns in zip(variable_cell_state, new_cell_state)]
        return tf.group(*ops)

    def reset_state_op(self, variable_state, **kwargs):
        variable_cell_state = self.state_to_cell_state(variable_state)
        ops = [tf.assign(vs, tf.zeros_like(vs), **kwargs)
               for vs in variable_cell_state]
        return tf.group(*ops)


class RNNStepToCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, step):
        self.step = step

    def __call__(self, inputs, state, scope=None):
        step_state = self.step.state_from_cell_state(state)
        new_step_state = self.step(inputs, step_state, scope=scope)
        new_cell_state = self.step.state_to_cell_state(new_step_state)
        output = self.step.state_to_cell_output(new_step_state)
        return output, new_cell_state

    @property
    def state_size(self):
        return self.step.cell_state_size()

    @property
    def output_size(self):
        return self.step.cell_output_size()
