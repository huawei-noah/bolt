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
torch.set_printoptions(precision=8)
m = torch.nn.Conv1d(3, 3, 2, stride=2, padding=1, dilation=2, bias=True)
input = torch.randn(2, 3, 5, requires_grad=True)
deltas = torch.randn(2, 3, 2)
print("Weights: ", m.weight)
print("Biases: ", m.bias)
print("Input: ", input)
result1 = m(input)
print("Result after first step: ", result1)
result2 = m(result1)
print("Result after second step: ", result2)
result2.backward(deltas)
print("Incoming deltas: ", deltas)
print("Gradient for weights: ", m.weight.grad)
print("Gradient for biases: ", m.bias.grad)
