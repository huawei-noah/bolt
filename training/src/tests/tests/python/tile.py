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

tf.random.set_seed(0)
x = tf.random.uniform([1, 2, 3, 4])
print("Input: ", x)
# Depth
with tf.GradientTape() as g:
    g.watch(x)
    y = tf.tile(x, [1, 3, 1, 1])
    print("Output (Depth * 3) = ", y)
print("Gradient (Depth * 3) = ", g.gradient(y, x))
# Height
with tf.GradientTape() as g:
    g.watch(x)
    y = tf.tile(x, [1, 1, 3, 1])
    print("Output (Height * 3) = ", y)
print("Gradient (Height * 3) = ", g.gradient(y, x))
# Width
with tf.GradientTape() as g:
    g.watch(x)
    y = tf.tile(x, [1, 1, 1, 2])
    print("Output (Width * 2) = ", y)
print("Gradient (Width * 2) = ", g.gradient(y, x))
with tf.GradientTape() as g:
    g.watch(x)
    y = tf.tile(x, [1, 3, 2, 1])
    print("Output (Depth * 3, Height * 2, Width * 1) = ", y)
print("Gradient (Depth * 3, Height * 2, Width * 1) = ", g.gradient(y, x))
