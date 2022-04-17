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

# Simple example
a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
b = torch.tensor([[-5.0, -0.527, 1.0, 1.2, 2.5]], requires_grad=True)
z = torch.max(a, b)
z.requires_grad_ = True
z.sum().backward()
print("Result = ", z)
print("Gradient for first input: ", a.grad)
print("Gradient for second input: ", b.grad)
# Equal values
c1 = torch.tensor([-5.0, -1.0, 0.0, 1.2, 2.5], requires_grad=True)
c2 = torch.tensor([[-5.0, -1.0, 0.0, 1.2, 2.5]], requires_grad=True)
z = torch.max(c1, c2)
z.requires_grad_ = True
z.sum().backward()
print("Result = ", z)
print("Gradient for first input: ", c1.grad)
print("Gradient for second input: ", c2.grad)
# broadcasting
torch.manual_seed(0)
x = torch.rand(3, 1, 2, 1, requires_grad=True)
y = torch.rand(3, 2, 1, 3, requires_grad=True)
z = torch.max(x, y)
z.requires_grad_ = True
z.sum().backward()
print("First input tensor = ", torch.flatten(x))
print("Second input tensor = ", torch.flatten(y))
print("Result = ", torch.flatten(z))
print("Result shape = ", z.shape)
print("Gradient for first input = ", torch.flatten(x.grad))
print("Gradient for second input = ", torch.flatten(y.grad))
