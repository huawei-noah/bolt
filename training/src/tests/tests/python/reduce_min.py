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
tf.random.set_seed(42)
x = tf.random.uniform(shape=[2, 3, 4, 5])
print("Input = ", x)
# Default
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x)
  print("Output (Default) = ", y)
print("Gradient (Default) = ", g.gradient(y, x) 
# Batch
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 0)
  print("Output (Batch) = ", y)
print("Gradient (Batch) = ", g.gradient(y, x) 
# Depth
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 1)
  print("Output (Depth) = ", y)
print("Gradient (Depth) = ", g.gradient(y, x) 
# Height
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 2)
  print("Output (Height) = ", y)
print("Gradient (Height) = ", g.gradient(y, x) 
# Width
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 3)
  print("Output (Width) = ", y)
print("Gradient (Width) = ", g.gradient(y, x)

# Extremum repeats case:
x = tf.constant([[[[1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0],
                [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0],
                [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0]]]])
print("Input = ", x)
# Default
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x)
  print("Output (Default) = ", y)
print("Gradient (Default) = ", g.gradient(y, x) 
# Batch
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 0)
  print("Output (Batch) = ", y)
print("Gradient (Batch) = ", g.gradient(y, x) 
# Depth
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 1)
  print("Output (Depth) = ", y)
print("Gradient (Depth) = ", g.gradient(y, x) 
# Height
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 2)
  print("Output (Height) = ", y)
print("Gradient (Height) = ", g.gradient(y, x) 
# Width
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.math.reduce_min(x, 3)
  print("Output (Width) = ", y)
print("Gradient (Width) = ", g.gradient(y, x)