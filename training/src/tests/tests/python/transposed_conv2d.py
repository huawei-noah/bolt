# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Bias unit
import torch

m = torch.nn.ConvTranspose2d(1, 2, 3)
input = torch.ones(1, 1, 3, 3, requires_grad=True)
m.weight = torch.nn.Parameter(torch.ones(1, 2, 3, 3))
m.bias = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
result = m(input)

## Input
print(input)

## Result
print(result)

# Simple unit
import torch

torch.manual_seed(0)
## Create convolution and input
m = torch.nn.ConvTranspose2d(1, 1, 2, bias=False)
input = torch.randn(1, 1, 2, 2, requires_grad=True)
print("Weights:", m.weight)
print("Bias:", m.bias)
print("Input:", input)
## Forward
result = m(input)
print("Result:", result)
## Backward
result.sum().backward()
print("Gradient for input:", input.grad)
print("Gradient for weights:", m.weight.grad)

# Non-trivial Unit
torch.manual_seed(0)
torch.set_printoptions(precision=8)
## Create convolution and input
m = torch.nn.ConvTranspose2d(3, 2, (3, 1), (2, 3), (2, 1), dilation=(2, 3), bias=True)
print("Weights:", m.weight)
print("Bias:", m.bias)
torch.manual_seed(0)
input = torch.randn(2, 3, 3, 3, requires_grad=True)
print("Input:", input)
## Forward
result = m(input)
print("Result:", result)
## Backward
deltas = torch.randn(2, 2, 5, 5)
result.backward(deltas)
print("Gradient for input:", input.grad)
print("Gradient for weights:", m.weight.grad)
print("Gradient for bias:", m.bias.grad)
