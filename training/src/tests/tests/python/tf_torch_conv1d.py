# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch
import tensorflow as tf
import numpy as np

c_out = 64
c_in = 1500
w = 20
N = 8
kernel = 5

np.random.seed(0)
weights = np.random.randn(c_out * w * kernel)

tf.compat.v1.enable_eager_execution()

input = tf.ones([N, c_in, w])
tf_c = tf.keras.layers.Conv1D(
    filters=c_out,
    kernel_size=kernel,
    activation=None,
    padding="same",
    use_bias=False,
    kernel_initializer=tf.keras.initializers.uniform,
    dilation_rate=1,
)
tf_c.build(input.shape)
tf_c.set_weights([weights.reshape(tf_c.weights[0].shape)])
result = tf_c(input)

print("TF Input: ", input.shape)
print("TF weights: ", tf_c.weights[0].shape)
print("TF Result: ", result.shape)

print(result)

# First
torch.set_printoptions(precision=8)
input = torch.ones(N, c_in, w, requires_grad=True)
input1 = input.transpose(1, 2)
torch_c = torch.nn.Conv1d(
    in_channels=w, out_channels=c_out, kernel_size=5, padding=2, bias=False
)
torch_c.weight.data = torch.FloatTensor(
    np.transpose(weights.reshape(tf_c.weights[0].shape), (2, 1, 0)).reshape(
        torch_c.weight.shape
    )
)
result = torch_c(input1)
result = result.transpose(1, 2)
print("Torch Input: ", input.shape)
print("Torch weights: ", torch_c.weight.shape)
print("Torch Result: ", result.shape)
print(result)
