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
import torch
import numpy as np

# Simple
c_out = 3
w = 3
c_in = 2
N = 2

np.random.seed(0)
weights = np.random.randn(c_out * c_in * 2)

tf.compat.v1.enable_eager_execution()

input = tf.constant(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[N, w, c_in]
)
tf_c = tf.keras.layers.Conv1D(
    filters=c_out, kernel_size=2, activation=None, use_bias=False, dilation_rate=1
)
tf_c.build(input.shape)
tf_c.set_weights([weights.reshape(tf_c.weights[0].shape)])

print("TF Input: ", input)
print("TF weights: ", tf_c.weights)
print("TF Result: ", tf_c(input))

# Equal torch variant
torch.set_printoptions(precision=8)
input = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
    requires_grad=True,
)
input1 = input.transpose(1, 2)
torch_c = torch.nn.Conv1d(
    in_channels=c_in, out_channels=c_out, kernel_size=2, padding=0, bias=False
)
torch_c.weight.data = torch.FloatTensor(
    np.transpose(weights.reshape(tf_c.weights[0].shape), (2, 1, 0)).reshape(
        torch_c.weight.shape
    )
)
result = torch_c(input1)
result = result.transpose(1, 2)

print("Torch result:", result)
# Gradients
deltas = torch.tensor([[[12, 11, 10], [9, 8, 7]], [[6, 5, 4], [3, 2, 1]]])
result.backward(deltas)
print("Gradient for input:", input.grad)
print("Gradient for weights:", torch.transpose(torch_c.weight.grad, 0, 2))

# More complex
c_out = 2
w = 5
c_in = 3
N = 2

np.random.seed(0)
weights = np.random.randn(c_out * c_in * 3)

tf.compat.v1.enable_eager_execution()

input = tf.constant(
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
        26.0,
        27.0,
        28.0,
        29.0,
        30.0,
    ],
    shape=[N, w, c_in],
)

tf_c = tf.keras.layers.Conv1D(
    filters=c_out,
    kernel_size=3,
    activation=None,
    padding="same",
    use_bias=False,
    dilation_rate=2,
    groups=1,
)
tf_c.build(input.shape)
tf_c.set_weights([weights.reshape(tf_c.weights[0].shape)])

print("TF input:", input.shape)
print("TF weights:", tf_c.weights[0].shape)
print("TF result:", tf_c(input).shape)

# The same thing, but in torch
torch.set_printoptions(precision=8)
input = torch.tensor(
    [
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        [
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ],
    ],
    requires_grad=True,
)
input1 = input.transpose(1, 2)
torch_c = torch.nn.Conv1d(
    in_channels=c_in,
    out_channels=c_out,
    kernel_size=3,
    padding=2,
    bias=False,
    dilation=2,
)
torch_c.weight.data = torch.FloatTensor(
    np.transpose(weights.reshape(tf_c.weights[0].shape), (2, 1, 0)).reshape(
        torch_c.weight.shape
    )
)
result = torch_c(input1)
result = result.transpose(1, 2)

print("Torch result:", result)

# Gradients
deltas = torch.tensor(
    [
        [[20.0, 19.0], [18.0, 17.0], [16.0, 15.0], [14.0, 13.0], [12.0, 11.0]],
        [[10.0, 9.0], [8.0, 7.0], [6.0, 5.0], [4.0, 3.0], [2.0, 1.0]],
    ]
)
result.backward(deltas)
print("Gradient for input:", input.grad)
print("Gradient for weights:", torch.transpose(torch_c.weight.grad, 0, 2))
