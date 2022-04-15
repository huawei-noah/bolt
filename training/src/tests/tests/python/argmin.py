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

# Default
torch.manual_seed(0)
x = torch.rand(2, 3, 4, 5, requires_grad=True)
z = torch.max(x)
z.requires_grad_ = True
z.backward()
print("Input x = ", x)
print("Result (Default) = ", z)
print("Gradient for input = ", x.grad)

# Batch dimension
torch.manual_seed(0)
x = torch.rand(2, 3, 4, 5, requires_grad=True)
z0 = torch.min(x, 0)
print("Result (Batch) = ", z0)
z0.values.requires_grad_ = True
z0.values.sum().backward()
print("Gradient for input = ", x.grad)

# Depth dimension
torch.manual_seed(0)
x = torch.rand(2, 3, 4, 5, requires_grad=True)
z1 = torch.min(x, 1)
print("Result (Depth) = ", z1)
z1.values.requires_grad_ = True
z1.values.sum().backward()
print("Gradient for input = ", x.grad)

# Height dimension
torch.manual_seed(0)
x = torch.rand(2, 3, 4, 5, requires_grad=True)
z2 = torch.min(x, 2)
print("Result (Depth) = ", z2)
z2.values.requires_grad_ = True
z2.values.sum().backward()
print("Gradient for input = ", x.grad)

# Width dimension
torch.manual_seed(0)
x = torch.rand(2, 3, 4, 5, requires_grad=True)
z3 = torch.min(x, 3)
print("Result (Depth) = ", z3)
z3.values.requires_grad_ = True
z3.values.sum().backward()
print("Gradient for input = ", x.grad)
