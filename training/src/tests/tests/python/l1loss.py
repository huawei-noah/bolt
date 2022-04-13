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
import numpy as np
import torchvision

# Mean reduction
loss_mean = nn.L1Loss(reduction="mean")
loss_none = nn.L1Loss(reduction="none")
loss_sum = nn.L1Loss(reduction="sum")
inp = torch.tensor(
    [
        [1.3, 1.2, 0.1, -4.0, -0.3],
        [-10.0, 1.0, -1.0, 2.0, -2.3],
    ],
    requires_grad=True,
)
target = torch.tensor([[0.1, 1.2, 1.0, 0.1, 7.7], [0.2, 0.2, 0.2, -1.3, -2.3]])

output_mean = loss_mean(inp, target)
print("Loss (mean reduction) = ", output_mean)
output_mean.backward()
print("Gradient = ", inp.grad)

# None reduction
inp = torch.tensor(
    [
        [1.3, 1.2, 0.1, -4.0, -0.3],
        [-10.0, 1.0, -1.0, 2.0, -2.3],
    ],
    requires_grad=True,
)
target = torch.tensor([[0.1, 1.2, 1.0, 0.1, 7.7], [0.2, 0.2, 0.2, -1.3, -2.3]])
output_none = loss_none(inp, target)
print("Loss (none reduction) = ", output_none)
output_none.sum().backward()
print("Gradient = ", inp.grad)

# Sum reduction
inp = torch.tensor(
    [
        [1.3, 1.2, 0.1, -4.0, -0.3],
        [-10.0, 1.0, -1.0, 2.0, -2.3],
    ],
    requires_grad=True,
)
target = torch.tensor([[0.1, 1.2, 1.0, 0.1, 7.7], [0.2, 0.2, 0.2, -1.3, -2.3]])
output_sum = loss_sum(inp, target)
print("Loss (sum reduction) = ", output_sum)
output_sum.backward()
print("Gradient = ", inp.grad)
