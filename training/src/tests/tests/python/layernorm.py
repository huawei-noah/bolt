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
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.autograd import Variable

#####################################################################################
#####################################################################################


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        print("mean", mean)
        print("std", std)
        print("t", std.grad_fn(x))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


x = torch.tensor(
    [[[-1.0, 0, 1.0], [4.0, 5.0, 6.0]], [[0.0, 4.0, 7.0], [-1.0, 2.0, 5.0]]],
    requires_grad=True,
)
target = torch.tensor([1, 0])

L = LayerNorm(3, 0)
fc = nn.Linear(6, 2)
torch.manual_seed(0)
np.random.seed(0)
nn.init.uniform_(fc.weight)
nn.init.ones_(fc.bias)
soft = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss(reduction="mean")

# forward
x_l = L(x)
x_fc = fc(x_l.reshape([2, 6]))
x_sm = soft(x_fc)
x_loss = loss(x_sm, target)

torch.set_printoptions(precision=6)
print("fc.weight", fc.weight)

print("x_l", x_l, "\n")
print("fc", x_fc, "\n")
print("sm", x_sm, "\n")
print("loss", x_loss)

# backward

x.retain_grad()
x_l.retain_grad()
x_fc.retain_grad()
x_sm.retain_grad()
x_loss.backward()

print("x.grad", x.grad)
print("x_l.grad", x_l.grad)
print("x_fc.grad", x_fc.grad)
print("x_sm.grad", x_sm.grad)
print("L.a_2.grad", L.a_2.grad)
print("L.b_2.grad", L.b_2.grad)
