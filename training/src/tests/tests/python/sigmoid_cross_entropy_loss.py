# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import tensorflow as tf

logits = tf.constant(
    [
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        ]
    ]
)
labels = tf.constant(
    [
        [
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[5, 5, 5], [8, 8, 8], [17, 16, 15]],
            [[14, 21, 22], [2, 3, 4], [5, 6, 7]],
        ]
    ]
)
print("Logits: ", logits)
print("Labels: ", labels)

# None
with tf.GradientTape() as g:
    g.watch(logits)
    res = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
    print("Result (None): ", res)
dy_dx = g.gradient(res, logits)
print("Gradient (None): ", dy_dx)

# Sum
with tf.GradientTape() as g:
    g.watch(logits)
    res = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
    print("Result (Sum): ", res)
dy_dx = g.gradient(res, logits)
print("Gradient (Sum): ", dy_dx)

# Mean
with tf.GradientTape() as g:
    g.watch(logits)
    res = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
    print("Result (Mean): ", res)
dy_dx = g.gradient(res, logits)
print("Gradient (Mean): ", dy_dx)
