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
from torch.optim import rmsprop

# Test 1
torch.manual_seed(0)
torch.set_printoptions(precision=6)
param = torch.rand(1, 2, 3, 4)
param.grad = torch.rand(1, 2, 3, 4)
print("Parameter: ", param)
print("Gradeient: ", param.grad)
# First step
opt = rmsprop.RMSprop(
    [param], lr=0.1, alpha=0.9, eps=0.1, weight_decay=0.1, momentum=0.1, centered=True
)
opt.step()
print("Parameter (after first step): ", param)
# Second step
opt.step()
print("Parameter (after second step): ", param)

# Test 2
param = torch.rand(1, 2, 3, 4)
param.grad = torch.rand(1, 2, 3, 4)
print("Parameter: ", param)
print("Gradeient: ", param.grad)

# First step
opt = rmsprop.RMSprop(
    [param], lr=0.1, alpha=0.9, eps=0.1, weight_decay=0.1, momentum=0.1, centered=False
)
opt.step()
print("Parameter (after first step): ", param)
# Second step
opt.step()
print("Parameter (after second step): ", param)

# Test 3
param = torch.rand(1, 2, 3, 4)
param.grad = torch.rand(1, 2, 3, 4)
print("Parameter: ", param)
print("Gradeient: ", param.grad)

# First step
opt = rmsprop.RMSprop(
    [param], lr=0.1, alpha=0.9, eps=0.1, weight_decay=0.1, momentum=0.0, centered=False
)
opt.step()
print("Parameter (after first step): ", param)
# Second step
opt.step()
print("Parameter (after second step): ", param)
