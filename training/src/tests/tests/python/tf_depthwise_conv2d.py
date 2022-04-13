# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Convolution (BatchedUnit)
tf.set_random_seed(0)

dynamic_input = tf.Variable(
    tf.random.uniform(
        [5, 3, 2, 2], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="input",
)

dynamic_filters = tf.Variable(
    tf.random.uniform(
        [5, 3, 2, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="input",
)

dynamic_features1 = tf.nn.depthwise_conv2d(
    input=dynamic_input,
    filter=dynamic_filters,
    strides=[1, 1, 1, 1],
    padding="SAME",
    name="dynamic_convolution",
    data_format="NHWC",
)

print("Input:", dynamic_input)
print("Filters:", dynamic_filters)
print("Result:", dynamic_features1)

# Same result as
dynamic_input_padded = tf.pad(dynamic_input, [[0, 0], [2, 2], [1, 1], [0, 0]])
with tf.GradientTape() as g:
    g.watch(dynamic_input_padded)
    dynamic_features2 = tf.nn.depthwise_conv2d(
        input=dynamic_input_padded,
        filter=dynamic_filters,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="dynamic_convolution",
        data_format="NHWC",
    )
deltas = tf.random.uniform(
    [5, 3, 2, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
)

print("Result:", dynamic_features2)
print("Deltas:", deltas)
print(
    "Gradient for input:", g.gradient(dynamic_features2, dynamic_input_padded, deltas)
)
with tf.GradientTape() as g:
    g.watch(dynamic_filters)
    dynamic_features2 = tf.nn.depthwise_conv2d(
        input=dynamic_input_padded,
        filter=dynamic_filters,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="dynamic_convolution",
        data_format="NHWC",
    )
print("Gradient for weights:", g.gradient(dynamic_features2, dynamic_filters, deltas))

# Convolution (NonBatchedUnit)
dynamic_input = tf.Variable(
    tf.random.uniform(
        [1, 4, 5, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="input",
)

dynamic_filters = tf.Variable(
    tf.random.uniform(
        [1, 21, 3, 2], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="input",
)

print("Input:", dynamic_input)
print("Filters:", dynamic_filters)

dynamic_features1 = tf.nn.depthwise_conv2d(
    input=dynamic_input,
    filter=dynamic_filters,
    strides=[1, 1, 1, 1],
    padding="SAME",
    name="dynamic_convolution",
    data_format="NHWC",
)
print("Result:", dynamic_features1)

dynamic_input_padded = tf.pad(dynamic_input, [[0, 0], [0, 0], [10, 10], [0, 0]])
with tf.GradientTape() as g:
    g.watch(dynamic_input_padded)
    dynamic_features2 = tf.nn.depthwise_conv2d(
        input=dynamic_input_padded,
        filter=dynamic_filters,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="dynamic_convolution",
        data_format="NHWC",
    )
deltas = tf.random.uniform(
    [1, 4, 5, 6], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
)
print("Result:", dynamic_features2)
print("Deltas", deltas)
print(
    "Gradient for input:", g.gradient(dynamic_features2, dynamic_input_padded, deltas)
)
with tf.GradientTape() as g:
    g.watch(dynamic_filters)
    dynamic_features2 = tf.nn.depthwise_conv2d(
        input=dynamic_input_padded,
        filter=dynamic_filters,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="dynamic_convolution",
        data_format="NHWC",
    )
print("Gradient for filters:", g.gradient(dynamic_features2, dynamic_filters, deltas))
