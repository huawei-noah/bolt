# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import functools

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import (
    BahdanauAttention,
    _BaseMonotonicAttentionMechanism,
    _monotonic_probability_fn,
    BahdanauMonotonicAttention,
)
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import (
    array_ops,
    math_ops,
    nn_ops,
    variable_scope,
    random_ops,
)

tf.enable_eager_execution()

# Preparation


def _bahdanau_score(
    processed_query, keys, attention_v, attention_g=None, attention_b=None
):
    """Implements Bahdanau-style (additive) scoring function.
    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    To enable the second form, set please pass in attention_g and attention_b.
    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to
        keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      attention_v: Tensor, shape `[num_units]`.
      attention_g: Optional scalar tensor for normalization.
      attention_b: Optional tensor with shape `[num_units]` for normalization.
    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = (
            attention_g
            * attention_v
            * tf.math.rsqrt(tf.reduce_sum(tf.square(attention_v)))
        )
        return tf.reduce_sum(
            normed_v * tf.tanh(keys + processed_query + attention_b), [2]
        )
    else:
        return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query), [2])


def _maybe_mask_score(
    score, memory_sequence_length=None, memory_mask=None, score_mask_value=None
):
    """Mask the attention score based on the masks."""
    if memory_sequence_length is None and memory_mask is None:
        return score
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided at same time."
        )
    if memory_sequence_length is not None:
        message = "All values in memory_sequence_length must greater than zero."
        with tf.control_dependencies(
            [
                tf.debugging.assert_positive(  # pylint: disable=bad-continuation
                    memory_sequence_length, message=message
                )
            ]
        ):
            memory_mask = tf.sequence_mask(
                memory_sequence_length, maxlen=tf.shape(score)[1]
            )
    score_mask_value = 1.1
    print(score)
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(memory_mask, score, score_mask_values)

def monotonic_stepwise_attention(p_choose_i, previous_attention, mode):
    # p_choose_i, previous_alignments, previous_score: [batch_size, memory_size]
    # p_choose_i: probability to keep attended to the last attended entry i
    if mode == "parallel":
        stay_part = previous_attention * p_choose_i
        go_part = previous_attention * (1.0 - p_choose_i)

        shifted = go_part[:, :-1]
        eos = go_part[:, -1][:, tf.newaxis]

        # shift
        go_part = tf.concat([tf.zeros_like(eos), shifted], axis=1)
        go_part += tf.concat([tf.zeros_like(shifted), eos], axis=1)

        attention = stay_part + go_part
    elif mode == "hard":
        # Given that previous_alignments is one_hot
        move_next_mask = tf.concat(
            [tf.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]],
            axis=1,
        )
        stay_prob = tf.reduce_sum(p_choose_i * previous_attention, axis=1)  # [B]
        attention = tf.where(stay_prob > 0.5, previous_attention, move_next_mask)
    else:
        raise ValueError("mode must be 'parallel', or 'hard'.")
    return attention


def _stepwise_monotonic_probability_fn(
    score, previous_alignments, sigmoid_noise, mode, seed=None
):
    if sigmoid_noise > 0:
        noise = random_ops.random_normal(
            array_ops.shape(score), dtype=score.dtype, seed=seed
        )
        score += sigmoid_noise * noise
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = math_ops.cast(score > 0, score.dtype)
    else:
        p_choose_i = math_ops.sigmoid(score)
    alignments = monotonic_stepwise_attention(p_choose_i, previous_alignments, mode)
    return alignments


class BahdanauStepwiseMonotonicAttention(BahdanauMonotonicAttention):
    def __init__(
        self,
        num_units,
        memory,
        memory_sequence_length=None,
        normalize=True,
        score_mask_value=None,
        sigmoid_noise=2.0,
        sigmoid_noise_seed=None,
        score_bias_init=3.5,
        mode="parallel",
        dtype=None,
        name="BahdanauStepwiseMonotonicAttention",
    ):
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _stepwise_monotonic_probability_fn,
            sigmoid_noise=sigmoid_noise,
            mode=mode,
            seed=sigmoid_noise_seed,
        )
        super(BahdanauMonotonicAttention, self).__init__(
            query_layer=tf.layers.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype
            ),
            memory_layer=tf.layers.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype
            ),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name,
        )
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init

        self.attention_v = tf.Variable(
            tf.random.uniform(
                [num_units],
                minval=0,
                maxval=None,
                dtype=tf.dtypes.float32,
                seed=0,
                name=None,
            ),
            name="attention_v",
        )
        self.attention_g = None
        self.attention_b = None
        if normalize:
            self.attention_g = 0.5
            self.attention_b = 0.0

    def __call__(self, query, state, prev_max_attentions):
        # Adding max_attentions to the call method for compability
        processed_query = self.query_layer(query) if self.query_layer else query
        score = _bahdanau_score(
            processed_query,
            self.keys,
            self.attention_v,
            self.attention_g,
            self.attention_b,
        )
        score += self._score_bias_init
        alignments = _stepwise_monotonic_probability_fn(
            _maybe_mask_score(score, memory_sequence_length=[1.0, 2.0, 3.0]),
            state,
            0.0,
            "parallel",
        )
        max_attentions = tf.argmax(alignments, -1, output_type=tf.int32)
        return alignments, max_attentions


# TEST 1 (normalize=False)

tf.random.set_random_seed(0)

memory = tf.Variable(
    tf.random.uniform(
        [3, 6, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 5], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0, 1.0]), name="prev_max")

attn = BahdanauStepwiseMonotonicAttention(
    4,
    memory,
    memory_sequence_length=tf.Variable(tf.constant([1.0, 2.0, 3.0])),
    normalize=False,
    sigmoid_noise=0.0,
    score_mask_value=tf.Variable(tf.constant([1.1])),
)

result = attn(query, state, prev_max)

print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)

# TEST 2 (normalize=True)

tf.random.set_random_seed(0)

memory = tf.Variable(
    tf.random.uniform(
        [3, 6, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="memory",
)
state = tf.Variable(
    tf.random.uniform(
        [3, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="state",
)
query = tf.Variable(
    tf.random.uniform(
        [3, 5], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)
prev_max = tf.Variable(tf.constant([1.0, 1.0, 1.0]), name="prev_max")

attn = BahdanauStepwiseMonotonicAttention(
    4,
    memory,
    memory_sequence_length=tf.Variable(tf.constant([1.0, 2.0, 3.0])),
    normalize=True,
    sigmoid_noise=0.0,
    score_mask_value=tf.Variable(tf.constant([1.1])),
)

result = attn(query, state, prev_max)

print("query: ", query)
print("state: ", state)
print("memory: ", memory)
print("result: ", result)
