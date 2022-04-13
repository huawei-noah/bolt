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

import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()


def range_with_max(r, max_range):
    range_values = tf.cond(
        tf.greater(r, 0.0), lambda: tf.range(r), lambda: -1 * tf.ones(1, tf.float32)
    )  # when duration is padded
    # pad to max range in sample
    paddings = [[0, max_range - tf.shape(range_values)[0]]]
    return tf.pad(range_values, paddings, "CONSTANT", constant_values=-1)


def durations_range(duration, max_mel_len):
    # for one sample
    # cur_range = tf.ragged.range(tf.ones(tf.shape(duration)[0]), duration+tf.ones(tf.shape(duration)[0])).flat_values
    max_range = tf.cast(tf.reduce_max(duration), tf.int32)
    cur_range = tf.map_fn(
        fn=lambda r: range_with_max(r, max_range), elems=duration
    )  # input_length x max_range
    cur_range = tf.reshape(cur_range, [-1])
    # drop zeros
    # mask = tf.cast(cur_range, dtype=tf.bool)
    mask = tf.greater(cur_range, -1)
    cur_range = tf.boolean_mask(cur_range, mask)
    # pad to max_mel_len in batch
    pad_num = tf.cond(
        tf.greater(max_mel_len - tf.shape(cur_range)[0], 0),
        lambda: max_mel_len - tf.shape(cur_range)[0],
        lambda: 0,
    )

    paddings = [[0, pad_num]]
    return tf.cond(
        tf.greater(max_mel_len - tf.shape(cur_range)[0], 0),
        lambda: tf.pad(cur_range, paddings, "CONSTANT", constant_values=149),
        lambda: cur_range[:max_mel_len],
    )


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding:
    """"""

    def __init__(self, embed_dim):
        """
        Args:
            input_dim: int, encoder_output vector size corresponding to one phoneme/char
        """
        super(PositionalEncoding, self).__init__()
        # self.embed_dim = embed_dim
        # self.w = tf.constant([1/(pow(10000,2*i/embed_dim)) for i in range(embed_dim)], tf.float32)
        n = 150
        pos_encoding = positional_encoding(n, embed_dim)
        self.pos_encoding_table = tf.reshape(pos_encoding[0], (n, embed_dim))

    def __call__(self, durations, max_mel_len=None):
        # durations - durations in frames (batch_size x seq_len)
        # раскрыть все длительности in range()
        # mel_len = tf.reduce_max(tf.reduce_sum(durations, axis = 1))
        if max_mel_len is not None:
            max_mel_len = tf.cast(max_mel_len, tf.int32)
        else:
            max_mel_len = tf.cast(
                tf.reduce_max(tf.reduce_sum(durations, axis=1)), tf.int32
            )
        ranges = tf.map_fn(
            fn=lambda t: durations_range(t, max_mel_len), elems=durations
        )
        return tf.nn.embedding_lookup(
            self.pos_encoding_table, tf.cast(ranges, tf.int32)
        )


d = 100
p = PositionalEncoding(32)
durations = tf.Variable(
    [
        [
            0,
            3,
            2,
            3,
            1,
            4,
            1,
            2,
            3,
            3,
            2,
            2,
            3,
            1,
            2,
            2,
            2,
            3,
            6,
            6,
            3,
            5,
            1,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            3,
            3,
            2,
            1,
            3,
            1,
            1,
            3,
            3,
            2,
            2,
            6,
            4,
            2,
            5,
            7,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            4,
            1,
            1,
            3,
            1,
            1,
            2,
            3,
            2,
            3,
            4,
            2,
            4,
            5,
            3,
            2,
            3,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            1,
            2,
            3,
            2,
            1,
            1,
            1,
            2,
            3,
            1,
            2,
            5,
            2,
            2,
            3,
            4,
            3,
            1,
            2,
            3,
            3,
            6,
            7,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    ],
    dtype=tf.float32,
)
res = p(durations, 222)
tf.print(res, summarize=-1)
