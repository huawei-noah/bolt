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

torch.manual_seed(0)
torch.set_printoptions(precision=8)
# Default
x = torch.rand(2, 3, 4, 5, requires_grad=True)
z = torch.sum(x)
z.backward()
print("Input = ", x)
print("Result (Default) = ", z)
print("Gradient for input (Default) = ", x.grad)

# By dimension
dimensions = [0, 1, 2, 3]
names = ["Batch", "Depth", "Height", "Width"]
for dim in dimensions:
    x = torch.rand(2, 3, 4, 5, requires_grad=True)
    z = torch.sum(x, dim)
    z.sum().backward()
    print("Result (" + names[dim] + ") = ", z)
    print("Gradient for input (" + names[dim] + ") = ", x.grad)
