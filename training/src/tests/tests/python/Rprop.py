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

# Step unit
torch.manual_seed(0)
param = torch.rand(1, 2, 3, 4)
grad = torch.normal(mean=0.0, std=1.0, size=(1, 2, 3, 4))
param.grad = grad
optimizer = torch.optim.Rprop([param], lr=0.1)
print("Initial param: ", param)
print("Gradient: ", grad)
optimizer.step()
print("Updated param: ", param)
optimizer.step()
print("Updated param: ", param)

# N steps
torch.manual_seed(0)

param = torch.rand(1, 2, 3, requires_grad=True)
multipliers = [
    torch.rand(1, 2, 3),
    torch.rand(1, 2, 3),
    torch.rand(1, 2, 3),
    (1.0 - (-1.0)) * torch.rand(1, 2, 3) - 1.0,
    (1.0 - (-1.0)) * torch.rand(1, 2, 3) - 1.0,
    torch.rand(1, 2, 3),
    torch.zeros(1, 2, 3),
    (1.0 - (-1.0)) * torch.rand(1, 2, 3) - 1.0,
]
optimizer = torch.optim.Rprop([param], lr=0.1, etas=(0.1, 20), step_sizes=(0.001, 1.2))
for i in range(len(multipliers)):
    c = param * multipliers[i]
    c.sum().backward()
    print("Param before optimization: ", param)
    print("Grad for param: ", param.grad)
    optimizer.step()
    optimizer.zero_grad()
    print("State of optimizer: ", optimizer.state_dict())
    print("Param after optimization: ", param)
