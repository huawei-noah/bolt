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
import math
import torch.nn.functional as F
from torch.autograd import Variable


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('scores', scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores.retain_grad()
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    res = torch.matmul(p_attn, value)
    return res, p_attn, scores


Q = torch.tensor(
    [[[1.0, 1.0, 2.0, 0.0, 5.0], [-1.0, 2.0, 2.0, 0.0, 5.0]]], requires_grad=True
)
K = torch.tensor(
    [[[-1.0, 4.0, 1.0, 2.0, 1], [-3.0, 4.0, 5.0, 2.0, 1]]], requires_grad=True
)
V = torch.tensor([[[0.0, 2.0], [1.0, 1.0]]], requires_grad=True)

mask = torch.tensor([[[1, 1], [1, 1]]])

res, p_attn, s = attention(Q, K, V, mask)


print("res", res)
print()
print("p_attn", p_attn)
p_attn.retain_grad()
res.backward(torch.tensor([[[0.9360, 1.0640], [0.9887, 1.0113]]]))

print(Q.grad)

print(K.grad)

print(V.grad)
