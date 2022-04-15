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
import numpy as np

# Zero clip
t_list = [tf.ones([1, 1, 2, 2]) for i in range(1)]
print("Input:", t_list)
t_list_clipped, _ = tf.clip_by_global_norm(t_list, clip_norm=-0.5)
print("Result:", t_list_clipped)

# Inf clip
t_list = [tf.ones([1, 1, 2, 2]) for i in range(1)]
print("Input:", t_list)
t_list_clipped, _ = tf.clip_by_global_norm(t_list, clip_norm=np.inf)
print("Result:", t_list_clipped)

# Process Gradients main test
tf.random.set_seed(0)
t_list = [tf.random.uniform([2, 1, 2, 2], minval=0.0, maxval=1.0) for i in range(3)]
for t in t_list:
    print("Input:", t)

t_list_clipped, global_norm = tf.clip_by_global_norm(t_list, clip_norm=100.0)
# No changes expected
for i in range(len(t_list)):
    assert tf.reduce_all(tf.equal(t_list, t_list_clipped))

t_list_clipped, global_norm = tf.clip_by_global_norm(t_list, clip_norm=0.5)
print("Calculated global norm:", global_norm)
for t in t_list_clipped:
    print("Result:", t)
