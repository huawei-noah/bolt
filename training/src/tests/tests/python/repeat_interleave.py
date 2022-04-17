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

torch.manual_seed(21)
torch.set_printoptions(precision=8)
x = torch.rand(1, 2, 3, 4, requires_grad=True)
print("Input: ", x)
# Depth
z = torch.repeat_interleave(x, torch.tensor([2, 3]), 1)
print("Result(Depth): ", z)
z_grad = torch.rand(1, 5, 3, 4)
z.backward(z_grad)
print("Gradient for input(Depth): ", x.grad)
# Height
x = torch.rand(1, 2, 3, 4, requires_grad=True)
z = torch.repeat_interleave(x, torch.tensor([2, 2, 3]), 2)
print("Result(Height): ", z)
z_grad = torch.rand(1, 2, 7, 4)
z.backward(z_grad)
print("Gradient for input(Height): ", x.grad)
# Width
x = torch.rand(1, 2, 3, 4, requires_grad=True)
z = torch.repeat_interleave(x, torch.tensor([3, 4, 5, 6]), 3)
print("Result(Width): ", z)
z_grad = torch.rand(1, 2, 3, 18)
z.backward(z_grad)
print("Gradient for input(Width): ", x.grad)
# Width (1 number)
x = torch.rand(1, 2, 3, 4, requires_grad=True)
z = torch.repeat_interleave(x, torch.tensor([3]), 3)
print("Result(Width): ", z)
z_grad = torch.rand(1, 2, 3, 12)
z.backward(z_grad)
print("Gradient for input(Width): ", x.grad)
