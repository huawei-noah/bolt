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

# C-tor Check
torch.manual_seed(42)
torch.set_printoptions(precision=8)
m = torch.nn.ConvTranspose1d(3, 2, 3, 3, 2, 1, 1, True, 3)
input = torch.rand(2, 3, 3)
print("Weights shape:", m.weight.shape)
print("Biases shape:", m.bias.shape)
print("Output shape:", m(input).shape)

# Simple Unit
## Setup
m = torch.nn.ConvTranspose1d(1, 1, 2, bias=False)
input = torch.rand(2, 1, 3, requires_grad=True)
print("Convolution weights:", m.weight)
print("Input:", input)
## Forward
result = m(input)
print("Result:", result)
## Backward
result.sum().backward()
print("Gradient for input:", input.grad)
print("Gradient for weights:", m.weight.grad)

# Non-trivial Unit
m = torch.nn.ConvTranspose1d(4, 6, 3, 2, 2, 1, 2, True, 3)
input = torch.rand(2, 4, 3, requires_grad=True)
print("Convolution weights:", m.weight)
print("Convolution bias:", m.bias)
print("Input:", input)
## Forward
result = m(input)
print("Result:", result)
## Backward
deltas = torch.rand(2, 6, 8)
result.backward(deltas)
print("Incoming deltas:", deltas)
print("Gradient for input:", input.grad)
print("Gradient for weights:", m.weight.grad)
print("Gradient for biases:", m.bias.grad)
