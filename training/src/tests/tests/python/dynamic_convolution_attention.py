# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Needed tf version - 1.13.2

import functools

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import (
    BahdanauAttention,
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

import scipy.special as sc
from scipy.special import comb
from scipy.stats import beta

import numpy as np

tf.enable_eager_execution()


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


def _dca_score(W_location, W_dynamic, prior, W_keys):
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_location.dtype
    print(dtype)
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable(
        "attention_variable_projection",
        shape=[num_units],
        dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer(),
    )
    b_a = tf.get_variable(
        "attention_bias",
        shape=[num_units],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
    )

    return tf.reduce_sum(v_a * tf.tanh(W_location + W_dynamic + b_a), [2]) + prior


class DynamicConvolutionalAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function
    The attention is location-based.
    """

    def beta_pdf(self, k):
        alpha_ = self.hparams.prior_alpha
        beta_ = self.hparams.prior_beta
        n = self.hparams.prior_filter_size - 1
        return comb(n, k) * sc.beta(k + alpha_, n - k + beta_) / sc.beta(alpha_, beta_)

    def _build_prior_filters(self, prior_filter_size):
        return [self.beta_pdf(i) for i in range(prior_filter_size)]

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
        name="DynamicConvolutionalAttention",
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
        self.hparams = hparams
        super(DynamicConvolutionalAttention, self).__init__(
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

        self.dynamic_fc1 = tf.layers.Dense(
            units=128,
            use_bias=True,
            dtype=tf.float32,
            name="dynamic_fc1",
            activation=tf.tanh,
        )

        self.dynamic_fc2 = tf.layers.Dense(
            units=21 * 8,
            use_bias=False,
            dtype=tf.float32,
            name="dynamic_fc2",
            activation=None,
        )

        self.dynamic_projection = tf.layers.Dense(
            units=num_units, use_bias=False, dtype=tf.float32, name="dynamic_projection"
        )

        prior_filters = tf.convert_to_tensor(
            self._build_prior_filters(hparams.prior_filter_size), dtype=tf.float32
        )
        prior_filters = tf.reverse(prior_filters, axis=[0])
        self.prior_filters = tf.reshape(
            prior_filters, [hparams.prior_filter_size, 1, 1]
        )

        self.synthesis_constraint = hparams.synthesis_constraint and not is_training
        self.attention_win_size = tf.convert_to_tensor(
            hparams.attention_win_size, dtype=tf.int32
        )
        self.constraint_type = hparams.synthesis_constraint_type
        self._cumulate = cumulate_weights

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return array_ops.one_hot(
            array_ops.zeros((batch_size,), dtype=tf.int32), max_time, dtype=dtype
        )

    def _dynamic_conv(self, dynamic_filters, dynamic_input):
        """
        Dynamic conv implementation using depthwise_conv2d.
        See details here:
        """
        sequence_length = tf.shape(dynamic_input)[1]

        dynamic_input = tf.expand_dims(dynamic_input, axis=1)

        dynamic_input = tf.transpose(dynamic_input, [1, 2, 0, 3])
        dynamic_input = tf.reshape(dynamic_input, [1, 1, sequence_length, -1])

        dynamic_features = tf.nn.depthwise_conv2d(
            input=dynamic_input,
            filter=dynamic_filters,
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="dynamic_convolution",
            data_format="NHWC",
        )

        dynamic_features = tf.reshape(dynamic_features, [1, sequence_length, -1, 1, 8])

        dynamic_features = tf.transpose(dynamic_features, [2, 0, 1, 3, 4])

        dynamic_features = tf.reduce_sum(dynamic_features, axis=3)

        dynamic_features = tf.squeeze(dynamic_features, axis=1)

        return dynamic_features

    def _apply_prior_filters(self, expanded_alignments):
        padded_alignments = tf.pad(expanded_alignments, [[0, 0], [10, 0], [0, 0]])
        prior = tf.nn.conv1d(
            padded_alignments, self.prior_filters, stride=1, padding="VALID"
        )
        prior = tf.squeeze(prior, axis=2)

        MIN_INPUT = (
            1.775e-38  # Smallest value that doesn't lead to log underflow for float32.
        )
        MIN_OUTPUT = -1e6
        prior_output = tf.math.log(tf.maximum(prior, MIN_INPUT))
        prior_output = tf.where(
            prior >= MIN_INPUT,
            prior_output,
            tf.fill(tf.shape(prior_output), MIN_OUTPUT),
        )
        return prior_output

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
        with tf.GradientTape() as g:
            with variable_scope.variable_scope(None, "DynamicConvolution", [query]):

                # processed_location_features shape [batch_size, max_time, attention dimension]
                # [batch_size, max_time] -> [batch_size, max_time, 1]
                expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
                # Projected location features [batch_size, max_time, attention_dim]
                f = self.location_convolution(expanded_alignments)
                processed_location_features = self.location_layer(f)
                dynamic_filters = self.dynamic_fc2(self.dynamic_fc1(query))
                dynamic_filters = tf.reshape(dynamic_filters, [-1, 1, 21, 8])

                # Do not mix from various samples in the same batch
                dynamic_filters = tf.transpose(dynamic_filters, [1, 2, 0, 3])

                # Apply dynamic filters
                dynamic_features = self._dynamic_conv(
                    dynamic_filters, expanded_alignments
                )

                # dynamic_features = tf.matmul(previous_alignments, dynamic_filters)
                processed_dynamic_features = self.dynamic_projection(dynamic_features)

                # energy shape [batch_size, max_time]
                # apply prior filters to ensure mo
                prior_output = self._apply_prior_filters(expanded_alignments)

                energy = _dca_score(
                    processed_location_features,
                    processed_dynamic_features,
                    prior_output,
                    self.keys,
                )
                g.watch(
                    tf.get_variable(
                        "attention_variable_projection",
                        shape=[4],
                        dtype=tf.dtypes.float32,
                        initializer=tf.contrib.layers.xavier_initializer(),
                    )
                )
            alignments = tf.nn.softmax(energy)

            max_attentions = tf.argmax(
                alignments, -1, output_type=tf.int32
            )  # (N, Ty/r)
            # Cumulate alignments
            if self._cumulate:
                next_state = alignments + previous_alignments
            else:
                next_state = alignments
        print(
            g.gradient(
                alignments,
                tf.get_variable(
                    "attention_variable_projection",
                    shape=[4],
                    dtype=tf.dtypes.float32,
                    initializer=tf.contrib.layers.xavier_initializer(),
                ),
            )
        )
        return alignments, next_state, max_attentions


def save_weights(path, data, name):
    import os

    if not os.path.exists(path):
        os.mkdir(path)

    if data.ndim == 3:
        with open(path + name + ".txt", "w") as outfile:
            for i in range(0, data.shape[0]):
                np.savetxt(outfile, data[i])
    elif data.ndim == 0:
        with open(path + name + ".txt", "w") as outfile:
            print(data, file=outfile)
    else:
        with open(path + name + ".txt", "w") as outfile:
            np.savetxt(outfile, data)


# TEST 1


class params:
    attention_filters = 3
    attention_kernel = 3
    prior_filter_size = 11
    synthesis_constraint = False
    synthesis_constraint_type = 2
    prior_alpha = 0.6
    prior_beta = 0.2
    attention_win_size = 1


tf.random.set_random_seed(0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1

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
attn = DynamicConvolutionalAttention(4, memory, param, False, cumulate_weights=False)
result = attn(query, state, prev_max)
print("query", query)
print("state", state)
print("memory", memory)
print("Result", result)
""" save all params
save_weights('./', attn.location_convolution.weights[0].numpy(), "location_convolution.weights")
save_weights('./', attn.location_convolution.weights[1].numpy(), "location_convolution.bias")
save_weights('./', attn.location_layer.weights[0].numpy(), "location_layer.weights")
save_weights('./', attn.dynamic_fc1.weights[0].numpy(), "dynamic_fc1.weights")
save_weights('./', attn.dynamic_fc1.weights[1].numpy(), "dynamic_fc1.bias")
save_weights('./', attn.dynamic_fc2.weights[0].numpy(), "dynamic_fc2.weights")
save_weights('./', attn.dynamic_projection.weights[0].numpy(), "dynamic_projection.weights")
save_weights('./', attn.prior_filters.numpy(), "apply_prior_filters.weights")
"""

# TEST 2
class params:
    attention_filters = 5
    attention_kernel = 3
    prior_filter_size = 14
    synthesis_constraint = False
    synthesis_constraint_type = 2
    prior_alpha = 0.9
    prior_beta = 0.1
    attention_win_size = 1


# Session setting
tf.random.set_random_seed(1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1

memory = tf.Variable(
    tf.random.uniform(
        [3, 4, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
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
        [3, 7], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="query",
)

prev_max = tf.Variable(tf.constant([1.0, 1.0, 1.0]), name="prev_max")
param = params
attn = DynamicConvolutionalAttention(
    5,
    memory,
    param,
    False,
    cumulate_weights=True,
    memory_sequence_length=tf.constant([2.0, 4.0, 5.0]),
    smoothing=True,
    normalize=True,
)
result = attn(query, state, prev_max)
print("query", query)
print("state", state)
print("memory", memory)
print("Result", result)

""" save all params
save_weights('./', attn.location_convolution.weights[0].numpy(), "location_convolution.weights_2")
save_weights('./', attn.location_convolution.weights[1].numpy(), "location_convolution.bias_2")
save_weights('./', attn.location_layer.weights[0].numpy(), "location_layer.weights_2")
save_weights('./', attn.dynamic_fc1.weights[0].numpy(), "dynamic_fc1.weights_2")
save_weights('./', attn.dynamic_fc1.weights[1].numpy(), "dynamic_fc1.bias_2")
save_weights('./', attn.dynamic_fc2.weights[0].numpy(), "dynamic_fc2.weights_2")
save_weights('./', attn.dynamic_projection.weights[0].numpy(), "dynamic_projection.weights_2")
"""
