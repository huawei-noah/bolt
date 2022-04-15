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

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class GaussianUpsampling:
    def __init__(self):
        super(GaussianUpsampling, self).__init__()

    def __call__(self, x, durations, ranges, mel_len=None):
        eps = 0.00001
        # x: batch_size x seq_len x input_dim
        # durations: batch_size x seq_len
        # missing elements to fit dimensions of both arrays should be 0
        if mel_len is not None:
            mel_len = tf.cast(mel_len, tf.float32)
        else:
            mel_len = tf.reduce_max(tf.reduce_sum(durations, axis=1))
        c = tf.add(
            tf.cast(tf.divide(durations, 2), tf.float32),
            tf.concat(
                [
                    tf.zeros([tf.shape(durations)[0], 1], tf.float32),
                    tf.cast(tf.cumsum(durations, axis=1)[:, :-1], tf.float32),
                ],
                axis=1,
            ),
        )
        dist = tfd.Normal(loc=c, scale=ranges + eps)
        bs = tf.shape(durations)[0]
        t = tf.range(1, mel_len + 1, dtype=tf.float32)
        t = t[:, tf.newaxis, tf.newaxis]
        p = tf.transpose(
            tf.exp(dist.log_prob(t)), [1, 0, 2]
        )  # bs x seq_len (t) x len(t)=seq_len (i)
        s = tf.expand_dims(tf.reduce_sum(p, axis=2), axis=2) + eps
        w = tf.divide(p, s)
        u = tf.matmul(w, x)
        return u


tf.random.set_seed(0)
x = tf.random.uniform([2, 5, 4], minval=0.0, maxval=1.0)
durations = tf.random.uniform([2, 5], minval=0.0, maxval=1.0)
ranges = tf.random.uniform([2, 5], minval=0.0, maxval=1.0)
gauss = GaussianUpsampling()
print("Input: ", x)
print("Durations: ", durations)
print("Ranges: ", ranges)
with tf.GradientTape() as g:
    g.watch(x)
    y = gauss(x, durations, ranges, 3.0)
    print("Output: ", y)
dy_dx = g.gradient(y, x)
print("Gradient for input: ", dy_dx)
with tf.GradientTape() as g:
    g.watch(durations)
    y = gauss(x, durations, ranges, 3.0)
dy_dx = g.gradient(y, durations)
print("Gradient for durations: ", dy_dx)
with tf.GradientTape() as g:
    g.watch(ranges)
    y = gauss(x, durations, ranges, 3.0)
dy_dx = g.gradient(y, ranges)
print("Gradient for ranges: ", dy_dx)
