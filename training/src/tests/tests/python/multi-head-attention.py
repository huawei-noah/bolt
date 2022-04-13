# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import seaborn

seaborn.set_context(context="talk")


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print('Before heads splitting')
        # print('Q', query)
        # print('K', key)
        # print('V', value)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # print('After heads splitting')
        # print('Q', query)
        # print('K', key)
        # print('V', value)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # print('x', x)
        # print('self.attn', self.attn)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# Train the simple copy task.
MODEL_SIZE = 9  # 512
H = 3  # 8
attn = MultiHeadedAttention(H, MODEL_SIZE, 0)

for p in attn.parameters():
    nn.init.ones_(p)

X = torch.tensor(
    np.array(
        [
            [
                [
                    [
                        1111.0,
                        1112.0,
                        1113.0,
                        1114.0,
                        1115.0,
                        1116.0,
                        1117.0,
                        1118.0,
                        1119.0,
                    ],
                    [
                        5121.0,
                        1122.0,
                        1123.0,
                        1124.0,
                        1125.0,
                        1126.0,
                        1127.0,
                        1128.0,
                        1129.0,
                    ],
                ],
            ],
            [
                [
                    [
                        2111.0,
                        2112.0,
                        2113.0,
                        2114.0,
                        2115.0,
                        2116.0,
                        2117.0,
                        2118.0,
                        2119.0,
                    ],
                    [
                        8121.0,
                        2122.0,
                        2123.0,
                        2124.0,
                        2125.0,
                        2126.0,
                        2127.0,
                        2128.0,
                        2129.0,
                    ],
                ],
            ],
        ]
    )
    / 10000.0,
    requires_grad=True,
    dtype=torch.float32,
)

print("X", X)
Y = attn(X, X, X)

print("Y", Y)

Y.backward(
    torch.tensor(
        np.array(
            [
                [
                    [
                        1111.0,
                        1112.0,
                        1113.0,
                        1114.0,
                        1115.0,
                        1116.0,
                        1117.0,
                        1118.0,
                        1119.0,
                    ],
                    [
                        8121.0,
                        1122.0,
                        1123.0,
                        1124.0,
                        1125.0,
                        1126.0,
                        1127.0,
                        1128.0,
                        1129.0,
                    ],
                ],
                [
                    [
                        2111.0,
                        2112.0,
                        2113.0,
                        2114.0,
                        2115.0,
                        2116.0,
                        2117.0,
                        2118.0,
                        2119.0,
                    ],
                    [
                        5121.0,
                        2122.0,
                        2123.0,
                        2124.0,
                        2125.0,
                        2126.0,
                        2127.0,
                        2128.0,
                        2129.0,
                    ],
                ],
            ]
        )
        / 10000.0,
        requires_grad=False,
        dtype=torch.float32,
    )
)

print("grad", X.grad)
