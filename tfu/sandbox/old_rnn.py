"""
older RNN implementations + deprecated (slower?) implementations of lstm_layer
"""


@base.hooked
def simple_rnn_step(tensors, state):
    # TODO have different fn to also precompute input
    with tf.variable_scope("simple_rnn"):
        x, = tensors
        h = state
        assert is_symbolic(x)
        assert is_symbolic(h)
        num_units = get_shape_values(h)[-1]
        logit = add_bias("bias",
                         linear("x_to_h", x, num_units) +
                         linear("h_to_h", h, num_units))

        @base.hooked
        def nonlinearity(logit):
            return tf.tanh(logit)

        return nonlinearity(logit=logit)


@base.hooked
def lstm_step(tensors, state):
    # TODO group linear operations for more efficiency
    with tf.variable_scope("lstm"):
        x, = tensors
        h = state["h"]
        c = state["c"]
        assert is_symbolic(x)
        assert is_symbolic(h)
        assert is_symbolic(c)
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units
        forget_logit = add_bias("forget_bias",
                                linear("forget_x", x, num_units) +
                                linear("forget_h", h, num_units))
        input_logit = add_bias("input_bias",
                               linear("input_x", x, num_units) +
                               linear("input_h", h, num_units))
        output_logit = add_bias("output_bias",
                                linear("output_x", x, num_units) +
                                linear("output_h", h, num_units))
        update_logit = add_bias("update_bias",
                                linear("update_x", x, num_units) +
                                linear("update_h", h, num_units))
        f = tf.nn.sigmoid(forget_logit)
        i = tf.nn.sigmoid(input_logit)
        o = tf.nn.sigmoid(output_logit)
        u = tf.tanh(update_logit)
        new_c = f * c + i * u
        new_h = tf.tanh(new_c) * o
        return {"h": new_h, "c": new_c}


@base.hooked
def simple_rnn_layer(name, tensor, state):
    with tf.variable_scope(name):
        x = tensor
        h = state
        assert is_symbolic(x)
        assert is_symbolic(h)
        num_units = get_shape_values(h)[-1]
        z = affine("x_to_h", x, num_units=num_units)

        @base.hooked
        def nonlinearity(logit):
            return tf.tanh(logit)

        def _step(tensors, state):
            z_, = tensors
            h_ = state
            logit = z_ + linear("h_to_h", h_, num_units=num_units)
            return nonlinearity(logit=logit)

        outputs = rnn_reduce("rnn", _step, [z], h)
        return outputs


@base.hooked
def lstm_layer(name, tensor, state):
    with tf.variable_scope(name):
        x = tensor
        h = state["h"]
        c = state["c"]
        assert is_symbolic(x)
        assert is_symbolic(h)
        assert is_symbolic(c)
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units

        # broadcast initial state to have batch size
        batch_size = get_shape_symbolic(x)[1]
        zero_state = tf.zeros((batch_size, num_units), dtype=x.dtype)
        state = {"h": zero_state + h, "c": zero_state + c}

        def _step(tensors, state_):
            x_, = tensors
            h_ = state_["h"]
            c_ = state_["c"]
            results = multi_affine(["h_to_forgetgate",
                                    "h_to_cellgate",
                                    "h_to_outputgate",
                                    "h_to_inputgate"],
                                   tf.concat(concat_dim=1, values=[x_, h_]),
                                   num_units=num_units)

            f = tf.nn.sigmoid(results[0])
            u = tf.tanh(results[1])
            o = tf.nn.sigmoid(results[2])
            i = tf.nn.sigmoid(results[3])
            new_c = f * c_ + i * u
            new_h = tf.tanh(new_c) * o
            return {"h": new_h, "c": new_c}

        outputs = rnn_reduce("rnn", _step, [x], state)
        return outputs


def lstm_layer_old1(name, tensor, state):
    """
    lstm with precomputing input and multi affine for hidden to hidden
    connections
    """
    with tf.variable_scope(name):
        x = tensor
        h = state["h"]
        c = state["c"]
        assert is_symbolic(x)
        assert is_symbolic(h)
        assert is_symbolic(c)
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units
        zs_ = multi_affine(["x_to_forgetgate",
                            "x_to_cellgate",
                            "x_to_outputgate",
                            "x_to_inputgate"],
                           x,
                           num_units=num_units)

        def _step(tensors, state_):
            precomp = tensors
            h_ = state_["h"]
            c_ = state_["c"]
            results = multi_linear(["h_to_forgetgate",
                                    "h_to_cellgate",
                                    "h_to_outputgate",
                                    "h_to_inputgate"],
                                   h_,
                                   num_units=num_units)

            f = tf.nn.sigmoid(precomp[0] + results[0])
            u = tf.tanh(precomp[1] + results[1])
            o = tf.nn.sigmoid(precomp[2] + results[2])
            i = tf.nn.sigmoid(precomp[3] + results[3])
            new_c = f * c_ + i * u
            new_h = tf.tanh(new_c) * o
            return {"h": new_h, "c": new_c}

        outputs = rnn_reduce("rnn", _step, zs_, state)
        return outputs


def lstm_layer_old2(name, tensor, state):
    """
    using tf.nn.rnn_cell.LSTMCell
    """
    x = tensor
    h = state["h"]
    c = state["c"]
    num_units = get_shape_values(h)[-1]
    assert get_shape_values(c)[-1] == num_units
    cell = tf.nn.rnn_cell.LSTMCell(num_units,
                                   forget_bias=0,
                                   state_is_tuple=True)

    # broadcast initial state to have batch size
    batch_size = get_shape_symbolic(x)[1]
    zero_state = tf.zeros((batch_size, num_units), dtype=x.dtype)
    state = {"h": zero_state + h, "c": zero_state + c}

    def _step(tensors, state_):
        x_, = tensors
        h_ = state_["h"]
        c_ = state_["c"]
        outputs, new_state = cell(x_, (c_, h_))
        new_c, new_h = new_state
        return {"h": new_h, "c": new_c}

    outputs = rnn_reduce("rnn", _step, [x], state)
    return outputs


def lstm_layer_old3(name, tensor, state):
    """
    using tf.nn.rnn_cell.LSTMCell and tf.nn.rnn
    """
    with tf.variable_scope(name):
        x = tensor
        h = state["h"]
        c = state["c"]
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units
        cell = tf.nn.rnn_cell.LSTMCell(num_units,
                                       forget_bias=0,
                                       state_is_tuple=True)

        # broadcast initial state to have batch size
        batch_size = get_shape_symbolic(x)[1]
        zero_state = tf.zeros((batch_size, num_units), dtype=x.dtype)
        state = (c + zero_state, h + zero_state)

        outputs, final_state = tf.nn.rnn(cell,
                                         inputs=tf.unpack(x),
                                         initial_state=state)
        return [{"h": output} for output in outputs]


def lstm_layer_old4(name, tensor, state):
    """
    using tf.nn.rnn_cell.LSTMCell and tf.nn.dynamic_rnn
    """
    with tf.variable_scope(name):
        x = tensor
        h = state["h"]
        c = state["c"]
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units
        cell = tf.nn.rnn_cell.LSTMCell(num_units,
                                       forget_bias=0,
                                       state_is_tuple=True)

        # broadcast initial state to have batch size
        batch_size = get_shape_symbolic(x)[1]
        zero_state = tf.zeros((batch_size, num_units), dtype=x.dtype)
        state = tf.nn.rnn_cell.LSTMStateTuple(c + zero_state, h + zero_state)

        outputs, final_state = tf.nn.dynamic_rnn(cell,
                                                 inputs=x,
                                                 initial_state=state,
                                                 time_major=True)
        # outputs is a 3-tensor
        # HACK only returning final output
        return [{"h": final_state[1]}]
