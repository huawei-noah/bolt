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
import torch.nn as nn

x = torch.tensor(
    [
        [[[1.0, 1.0, 2.0, 0.0, 5.0], [-1.0, 2.0, 2.0, 0.0, 5.0]]],
        [[[-1.0, 4.0, 1.0, 2.0, 1], [-3.0, 4.0, 5.0, 2.0, 1]]],
    ],
    requires_grad=True,
)

l = nn.Linear(5, 5)
l.weight.data = torch.tensor(
    [
        [-0.2381, 0.1714, -0.0612, -0.1329, -0.3701],
        [0.0283, -0.2147, -0.0502, 0.2090, 0.4333],
        [-0.1200, 0.1664, -0.3021, -0.2250, 0.3329],
        [-0.1200, 0.1664, -0.3021, -0.2250, 0.3329],
        [0.1200, 0.1664, 0.3021, 0.2250, 0.3329],
    ],
    requires_grad=True,
)
l.bias.data = torch.tensor([0.3548, 0.2879, 0.0343, 0.1269, 0.2234], requires_grad=True)
print(l.weight)
print(l.bias)

res = l(x)
res = l(res)
res = l(res)
res.backward(
    torch.tensor(
        [
            [[[1.0, 2.0, -1, 2.0, 1], [0.5, 1, 1.0, 0.4, 0.8]]],
            [[[0.5, 6.0, 1, 1, 2], [2, -1.0, 1.5, -0.5, 0.1]]],
        ]
    )
)

print("res", res)
print("x.grad", x.grad)
print("weight.grad", l.weight.grad)
print("bias.grad", l.bias.grad)
