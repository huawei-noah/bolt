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

tf.random.set_seed(42)

query = tf.random.uniform(shape=[2, 3, 3])
value = tf.random.uniform(shape=[2, 2, 3])
key = tf.random.uniform(shape=[2, 2, 3])
attn = tf.keras.layers.AdditiveAttention(False)
print("Attention: ", attn)
# Calculate attention and gradients
# Query
with tf.GradientTape() as g:
    g.watch(query)
    y = attn([query, value, key])
print("Query gradient", g.gradient(y, query))
# Value
with tf.GradientTape() as g:
    g.watch(value)
    y = attn([query, value, key])
print("Value gradient", g.gradient(y, value))
# Key
with tf.GradientTape() as g:
    g.watch(key)
    y = attn([query, value, key])
print("Key gradient", g.gradient(y, key))
