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

torch.manual_seed(42)
torch.set_printoptions(precision=7)

# MSELoss variants
loss_none = torch.nn.MSELoss(reduction="none")
loss_mean = torch.nn.MSELoss(reduction="mean")
loss_sum = torch.nn.MSELoss(reduction="sum")
# None
x = torch.tensor(
    [[[0.8822693, 0.9150040, 0.3828638], [0.9593056, 0.3904482, 0.6008953]]],
    requires_grad=True,
)
y = torch.tensor(
    [[[0.2565725, 0.7936413, 0.9407715], [0.1331859, 0.9345981, 0.5935796]]],
    requires_grad=True,
)
z_none = loss_none(x, y)
z_none.requires_grad_ = True
z_none.sum().backward()
print("input = ", x)
print("target = ", y)
print("Result (none) = ", z_none)
print("Gradient for input (none) = ", x.grad)
# Mean
x = torch.tensor(
    [[[0.8822693, 0.9150040, 0.3828638], [0.9593056, 0.3904482, 0.6008953]]],
    requires_grad=True,
)
y = torch.tensor(
    [[[0.2565725, 0.7936413, 0.9407715], [0.1331859, 0.9345981, 0.5935796]]],
    requires_grad=True,
)
z_mean = loss_mean(x, y)
z_mean.requires_grad_ = True
z_mean.backward()
print("Result (mean) = ", z_mean)
print("Gradient for input (mean) = ", x.grad)
# None
x = torch.tensor(
    [[[0.8822693, 0.9150040, 0.3828638], [0.9593056, 0.3904482, 0.6008953]]],
    requires_grad=True,
)
y = torch.tensor(
    [[[0.2565725, 0.7936413, 0.9407715], [0.1331859, 0.9345981, 0.5935796]]],
    requires_grad=True,
)
z_sum = loss_sum(x, y)
z_sum.requires_grad_ = True
z_sum.backward()
print("Result (sum) = ", z_sum)
print("Gradient for input (sum) = ", x.grad)
