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


import torch

# Simple unit
input = torch.rand(2, 3, requires_grad=True)
target = torch.rand(2, 3)
loss = torch.nn.BCELoss(reduction="none")
print("Input: ", input)
print("Target: ", target)
output = loss(input, target)
print("Loss: ", output)
output.sum().backward()
print("Gradient for input: ", input.grad)

# Corner cases
loss = torch.nn.BCELoss(reduction="none")
input = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.998, 0.002, 1.0], [0.999, 0.0001, 0.11]],
    requires_grad=True,
)
target = torch.tensor(
    [[0.0, 1.0, 0.99], [0.0, 1.0, 0.5], [0.0001, 0.997, 1.0], [0.999, 0.001, 0.00001]]
)
print("Input: ", input)
print("Target: ", target)
output = loss(input, target)
print("Loss: ", loss)
output.sum().backward()
print("Gradient for input: ", input.grad)
