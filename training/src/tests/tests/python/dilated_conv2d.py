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

# First
torch.manual_seed(0)
m = torch.nn.Conv2d(
    1, 1, (2, 2), stride=(1, 1), padding=(0, 0), dilation=(2, 2), bias=False
)
torch.manual_seed(42)
torch.set_printoptions(precision=8)
input = torch.randn(1, 1, 5, 5, requires_grad=True)
print("Input: ", input)
result = m(input)
print("Result: ", result)
result.sum().backward()
print("Gradient for input: ", input.grad)
print("Gradient for weights: ", m.weight.grad)

# Second
torch.manual_seed(0)
m = torch.nn.Conv2d(
    2, 3, (3, 3), stride=(2, 2), padding=(2, 2), dilation=(3, 3), bias=False
)

torch.manual_seed(42)
torch.set_printoptions(precision=8)
input = torch.randn(2, 2, 5, 5, requires_grad=True)

print("Input: ", input)
result = m(input)
print("Result: ", result)
result.sum().backward()
print("Gradient for input: ", input.grad)
print("Gradient for weights: ", m.weight.grad)

# Third
torch.manual_seed(0)
m = torch.nn.Conv2d(
    3, 2, (1, 3), stride=(1, 3), padding=(3, 1), dilation=(2, 3), bias=True
)

torch.manual_seed(42)
torch.set_printoptions(precision=8)
input = torch.randn(2, 3, 5, 5, requires_grad=True)

print("Input: ", input)
result = m(input)
print("Result: ", result)
result.sum().backward()
print("Gradient for input: ", input.grad)
print("Gradient for weights: ", m.weight.grad)
