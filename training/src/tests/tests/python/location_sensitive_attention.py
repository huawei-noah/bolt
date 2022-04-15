# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# !pip install tensorflow==1.13.2

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import array_ops, math_ops, variable_scope

tf.enable_eager_execution()


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
    This attention is described in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.

    #############################################################################
              hybrid attention (content-based + location-based)
                               f = F * α_{i-1}
       energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
    #############################################################################

    Args:
        W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
        W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
        W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
    Returns:
        A '[batch_size, max_time]' attention score (energy)
    """
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable(
        "attention_variable_projection",
        shape=[num_units],
        dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True,
    )
    print(v_a)
    b_a = tf.get_variable(
        "attention_bias",
        shape=[num_units],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
    )

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
    Introduced in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.

    ############################################################################
                        Smoothing normalization function
                a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
    ############################################################################

    Args:
        e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
            values of an attention mechanism
    Returns:
        matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
            attendance to multiple memory time steps.
    """
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function.
      Usually referred to as "hybrid" attention (content-based + location-based)
      Extends the additive attention described in:
      "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
    tion by jointly learning to align and translate,” in Proceedings
    of ICLR, 2015."
      to use previous alignments as additional location features.

      This attention is described in:
      J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
    gio, “Attention-based models for speech recognition,” in Ad-
    vances in Neural Information Processing Systems, 2015, pp.
    577–585.
    """

    def __init__(
        self,
        num_units,
        memory,
        hparams,
        is_training,
        mask_encoder=True,
        memory_sequence_length=None,
        normalize=False,
        smoothing=False,
        cumulate_weights=True,
        use_score_bias=False,
        score_bias_init=0.0,
        sigmoid_noise=0.0,
        use_forward=False,
        name="LocationSensitiveAttention",
    ):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            mask_encoder (optional): Boolean, whether to mask encoder paddings.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths. Only relevant if mask_encoder = True.
            smoothing (optional): Boolean. Determines which normalization function to use.
                Default normalization function (probablity_fn) is softmax. If smoothing is
                enabled, we replace softmax with:
                        a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
                Introduced in:
                    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
                  gio, “Attention-based models for speech recognition,” in Ad-
                  vances in Neural Information Processing Systems, 2015, pp.
                  577–585.
                This is mainly used if the model wants to attend to multiple input parts
                at the same decoding step. We probably won't be using it since multiple sound
                frames may depend on the same character/phone, probably not the way around.
                Note:
                    We still keep it implemented in case we want to test it. They used it in the
                    paper in the context of speech recognition, where one phoneme may depend on
                    multiple subsequent sound frames.
            name: Name to use when creating ops.
        """
        # Create normalization function
        # Setting it to None defaults in using softmax
        normalization_function = _smoothing_normalization if smoothing is True else None
        memory_length = memory_sequence_length if mask_encoder is True else None
        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_length,
            normalize=normalize,
            probability_fn=normalization_function,
            name=name,
        )

        self.location_convolution = tf.layers.Conv1D(
            filters=hparams.attention_filters,
            kernel_size=hparams.attention_kernel,
            padding="same",
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            name="location_features_convolution",
        )
        self.location_layer = tf.layers.Dense(
            units=num_units,
            use_bias=False,
            dtype=tf.float32,
            name="location_features_layer",
        )

        self.transition_agent_layer = None
        if hparams.use_transition_agent:
            self.transition_agent_layer = tf.layers.Dense(
                units=1,
                use_bias=True,
                bias_initializer=tf.zeros_initializer(),
                dtype=tf.float32,
                name="transition_agent_layer",
            )

        self._cumulate = cumulate_weights
        self._use_score_bias = use_score_bias
        self._score_bias_init = score_bias_init
        self._sigmoid_noise = sigmoid_noise
        self._use_forward = use_forward

        self.synthesis_constraint = hparams.synthesis_constraint and not is_training
        self.attention_win_size = tf.convert_to_tensor(
            hparams.attention_win_size, dtype=tf.int32
        )
        self.constraint_type = hparams.synthesis_constraint_type

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return array_ops.one_hot(
            array_ops.zeros((batch_size,), dtype=tf.int32), max_time, dtype=dtype
        )

    def __call__(self, query, state, prev_max_attentions):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state (previous alignments): Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        previous_alignments = state
        with variable_scope.variable_scope(
            None, "Location_Sensitive_Attention", [query]
        ):
            # transition agent
            transit_proba = 0.5
            if self.transition_agent_layer is not None:
                expanded_alignments = array_ops.expand_dims(previous_alignments, 1)
                previous_context = math_ops.matmul(expanded_alignments, self.values)
                previous_context = array_ops.squeeze(previous_context, [1])
                ta_input = tf.concat([query, previous_context], axis=-1)
                transit_proba = tf.sigmoid(self.transition_agent_layer(ta_input))

            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)

            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(
                processed_query, processed_location_features, self.keys
            )

    """ 
    # [A. Misevich]: This part will not be supported now.
    # It will be implemented if will be needed.
        if self.synthesis_constraint:
            Tx = tf.shape(energy)[-1]
            # prev_max_attentions = tf.squeeze(prev_max_attentions, [-1])
            if self.constraint_type == 'forward':
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(Tx - 2 - prev_max_attentions, Tx)[:, ::-1]
            elif self.constraint_type == 'monotonic':
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(Tx - self.attention_win_size - prev_max_attentions, Tx)[:, ::-1]
            else:
                assert self.constraint_type == 'window'
                key_masks = tf.sequence_mask(prev_max_attentions - (self.attention_win_size // 2 + (self.attention_win_size % 2 != 0)), Tx)
                reverse_masks = tf.sequence_mask(Tx - (self.attention_win_size // 2) - prev_max_attentions, Tx)[:, ::-1]

            masks = tf.logical_or(key_masks, reverse_masks)
            paddings = tf.ones_like(energy) * (-2 ** 32 + 1)  # (N, Ty/r, Tx)
            energy = tf.where(tf.equal(masks, False), energy, paddings)
    """

    # alignments shape = energy shape = [batch_size, max_time]
    if self.constraint_type == "stepwise_monotonic":
        noise = tf.random_normal(array_ops.shape(energy), dtype=energy.dtype)
        p = tf.sigmoid(energy + self._sigmoid_noise * noise)
        alignments = tf.multiply(previous_alignments, p) + tf.pad(
            tf.multiply(previous_alignments, 1 - p)[:, :-1], [[0, 0], [1, 0]]
        )
    else:
        print("here")
        alignments = self._probability_fn(energy, previous_alignments)

    if self._use_forward:
        # [batch_size, max_time] -- shift and add to get alpha(n-1) + alpha(n)
        previous_alignments_shifted = tf.pad(
            previous_alignments[:, :-1], [[0, 0], [1, 0]]
        )
        alignments = (
            (1 - transit_proba) * previous_alignments
            + transit_proba * previous_alignments_shifted
        ) * alignments
        alignments = alignments / tf.reduce_sum(alignments, axis=-1, keepdims=True)

    max_attentions = tf.argmax(alignments, -1, output_type=tf.int32)  # (N, Ty/r)

    # Cumulate alignments
    if self._cumulate:
        next_state = alignments + previous_alignments
    else:
        next_state = alignments

    return alignments, next_state, max_attentions


# TESTS
# TEST 1
# No mask, no transition agent, synthesis_constraint=False, use_forward=False, cumulate=False, normalize=False, smoothing=False, constraint_type=None
tf.set_random_seed(0)


class params:
    attention_filters = 3
    attention_kernel = 3
    use_transition_agent = False
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [2, 3, 5], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [2, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [2, 5], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(4, memory, param, False, cumulate_weights=False)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 2
# No mask, no transition agent, synthesis_constraint=False, use_forward=False, cumulate=True, smoothing=True, constraint_type=None
tf.set_random_seed(1)


class params:
    attention_filters = 2
    attention_kernel = 5
    use_transition_agent = False
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [3, 4, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    5, memory, param, False, cumulate_weights=True, smoothing=True
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 3 (like TEST 2, but with mask)
# Use mask, no transition agent, synthesis_constraint=False, use_forward=False, cumulate=True, smoothing=True, constraint_type=None
tf.set_random_seed(1)


class params:
    attention_filters = 2
    attention_kernel = 5
    use_transition_agent = False
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [3, 4, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    5,
    memory,
    param,
    False,
    memory_sequence_length=[2, 3, 4],
    cumulate_weights=True,
    smoothing=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)
print("keys:", attn.keys)

# TEST 4
# Use mask, no transition agent, synthesis_constraint=False, use_forward=True, cumulate=True, smoothing=True, constraint_type=None
tf.set_random_seed(1)


class params:
    attention_filters = 2
    attention_kernel = 5
    use_transition_agent = False
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [3, 4, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
weights = np.array(
    [
        0.01492912,
        -0.08340913,
        -0.13510555,
        0.7274594,
        0.27867514,
        -0.22769517,
        -0.09443533,
        0.14927697,
        -0.06407022,
        0.3732596,
        -0.46085864,
        0.0720188,
        0.07225263,
        0.0731563,
        -0.32511705,
        -0.5776096,
        0.19336885,
        0.5521658,
        0.04765564,
        0.09024948,
        -0.3043793,
        0.6329126,
        -0.57645786,
        0.68469685,
        0.27464074,
        -0.32275102,
        0.4352892,
        -0.59377813,
        0.703242,
        0.49236614,
    ]
)
attn = LocationSensitiveAttention(
    5,
    memory,
    param,
    False,
    memory_sequence_length=[2, 3, 4],
    cumulate_weights=True,
    smoothing=True,
    use_forward=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 5
# Use mask, no transition agent, synthesis_constraint=False, use_forward=False, cumulate=True, smoothing=True, constraint_type=None
tf.set_random_seed(1)


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = False
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = "stepwise_monotonic"


memory = tf.Variable(
    tf.random.uniform(
        [3, 5, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 5], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 8], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    5,
    memory,
    param,
    False,
    memory_sequence_length=[3, 1, 2],
    cumulate_weights=True,
    smoothing=True,
    use_forward=False,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 6 (Like TEST 5, but use forward)
# Use mask, no transition agent, synthesis_constraint=False, use_forward=False, cumulate=True, smoothing=True, constraint_type='stepwise_monotonic'
tf.set_random_seed(1)


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = False
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = "stepwise_monotonic"


memory = tf.Variable(
    tf.random.uniform(
        [3, 5, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 5], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 8], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    5,
    memory,
    param,
    False,
    memory_sequence_length=[3, 1, 2],
    cumulate_weights=True,
    smoothing=True,
    use_forward=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 7
# Use mask, use transition agent, synthesis_constraint=False, use_forward=False, cumulate=True, smoothing=True, constraint_type=None
tf.set_random_seed(1)


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = True
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [3, 3, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    7,
    memory,
    param,
    False,
    memory_sequence_length=[2, 2, 2],
    cumulate_weights=True,
    smoothing=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 8
# Use mask, use transition agent, synthesis_constraint=False, use_forward=True, cumulate=True, smoothing=True, constraint_type=None
tf.set_random_seed(1)


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = True
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [3, 3, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    7,
    memory,
    param,
    False,
    memory_sequence_length=[2, 2, 2],
    cumulate_weights=True,
    smoothing=True,
    use_forward=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 9
# Enable everything, but no smoothing (and no constraints)
tf.set_random_seed(1)


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = True
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = None


memory = tf.Variable(
    tf.random.uniform(
        [3, 3, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    7,
    memory,
    param,
    False,
    memory_sequence_length=[2, 2, 2],
    cumulate_weights=True,
    smoothing=False,
    use_forward=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 10
# Enable everything
tf.set_random_seed(1)


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = True
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = "stepwise_monotonic"


memory = tf.Variable(
    tf.random.uniform(
        [3, 3, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    7,
    memory,
    param,
    False,
    memory_sequence_length=[2, 2, 2],
    cumulate_weights=True,
    smoothing=True,
    use_forward=True,
)

result = attn(query, state, prev_max)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 11
# Double step, enable everything
tf.set_random_seed(1)


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
    This attention is described in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.

    #############################################################################
              hybrid attention (content-based + location-based)
                               f = F * α_{i-1}
       energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
    #############################################################################

    Args:
        W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
        W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
        W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
    Returns:
        A '[batch_size, max_time]' attention score (energy)
    """
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]
    v_a = tf.get_variable(
        "attention_variable_projection",
        shape=[num_units],
        dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True,
    )

    # PLUG IN ORDER TO KEEP THE SAME VALUE FOR THIS VARIABLE
    v_a = tf.Variable(
        [
            -0.21917742,
            0.19094306,
            0.26173806,
            -0.31035906,
            0.06741166,
            0.22481245,
            -0.6266,
        ]
    )
    b_a = tf.get_variable(
        "attention_bias",
        shape=[num_units],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
    )
    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


class params:
    attention_filters = 5
    attention_kernel = 5
    use_transition_agent = True
    synthesis_constraint = False
    attention_win_size = 1
    synthesis_constraint_type = "stepwise_monotonic"


memory = tf.Variable(
    tf.random.uniform(
        [3, 3, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query1 = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query1",
)

query2 = tf.Variable(
    tf.random.uniform(
        [3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query2",
)
bias = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="bias",
)
multiplier = tf.Variable(
    tf.random.uniform(
        [3, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="multiplier",
)

prev_max = tf.Variable(tf.constant([1.0, 1.0]), name="prev_max")
param = params
attn = LocationSensitiveAttention(
    7,
    memory,
    param,
    False,
    memory_sequence_length=[2, 2, 2],
    cumulate_weights=True,
    smoothing=True,
    use_forward=True,
)

# First step: calculate attention
result1 = attn(query1, state, prev_max)
# Second step: Add bias to produced alignments
new_state = result1[0] + bias
# Third step: calculate new attention based on previous
result2 = attn(query2, new_state, result1[2])
# Fourth step: multiply new alignments by multiplier
final = result2[0] * multiplier

print("query1: ", query1)
print("query2: ", query2)
print("state: ", state)
print("memory: ", memory)
print("result1: ", result1)
print("bias: ", bias)
print("new state: ", new_state)
print("result2: ", result2)
print("multiplier: ", multiplier)
print("final result: ", final)
