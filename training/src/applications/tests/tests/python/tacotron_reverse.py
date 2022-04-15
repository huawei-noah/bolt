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

seq_lengths = [0, 3, 5]
input = tf.Variable(
    [
        [[1.0, 2, 3], [4, 5, 0], [3, 6, 0], [0, 0, 0], [0, 0, 0]],
        [[1.0, 2, 3], [4, 5, 2], [5, 6, 1], [0, 0, 0], [0, 0, 0]],
        [[1.0, 2, 3], [4, 5, 0], [5, 6, 9], [7, 8, 10], [9, 10, 11]],
    ]
)
gradient = tf.Variable(
    [
        [[1.0, 2, 3], [4, 5, 0], [3, 6, 0], [0, 0, 0], [0, 0, 0]],
        [[1.0, 2, 3], [4, 5, 2], [5, 6, 1], [0, 0, 0], [0, 0, 0]],
        [[1.0, 2, 3], [4, 5, 0], [5, 6, 9], [7, 8, 10], [9, 10, 11]],
    ]
)
with tf.GradientTape() as g:
    g.watch(input)
    output = tf.reverse_sequence(input, seq_lengths, seq_axis=1, batch_axis=0)
dy_dx = g.gradient(output, input, gradient)

print("Input: ", input)
print("Output: ", output)
print("Gradient for input: ", dy_dx)
