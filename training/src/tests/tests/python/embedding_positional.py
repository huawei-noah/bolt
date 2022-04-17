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
from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        p = Variable(self.pe[:, : x.size(1)], requires_grad=False)
        x = x + p
        x = x.squeeze()
        return self.dropout(x)


V = 3
d_model = 6

inp = torch.LongTensor([[2, 1, 0, 0]])
target = torch.tensor([1])

torch.manual_seed(0)
np.random.seed(0)
e = Embeddings(d_model, V)
pe = PositionalEncoding(d_model, 0, 4)
fc = nn.Linear(24, 2)

nn.init.uniform_(fc.weight)
nn.init.ones_(fc.bias)
torch.set_printoptions(precision=6)
print("fc.weight", fc.weight)

f = fc.weight

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss(reduction="mean")

t1 = e(inp)
t2 = pe(t1)
t3 = fc(t2.reshape([1, 24]))
t4 = m(t3)

t1.retain_grad()
t3.retain_grad()
t4.retain_grad()
output = loss(t4, target)
output.backward()


print(t1)
print(t2)
print(t3)
print(t4)
print(output)

print()
print("t1grad", t1.grad)
print("t2grad", t2.grad)
print("t3grad", t3.grad)
print("t4grad", t4.grad)

print(e.lut.weight)
print(e.lut.weight.grad)
