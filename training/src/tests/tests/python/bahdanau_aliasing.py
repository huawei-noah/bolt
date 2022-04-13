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
import texar.torch

torch.manual_seed(42)
torch.set_printoptions(precision=10)
# First
query1 = torch.rand(2, 2, requires_grad=True)
state1 = torch.rand(2, 2, requires_grad=True)
memory1 = torch.rand(2, 2, 2, requires_grad=True)
# Second
query2 = torch.rand(2, 2, requires_grad=True)
state2 = torch.rand(2, 2, requires_grad=True)
memory2 = torch.rand(2, 2, 2, requires_grad=True)

attn = texar.torch.core.BahdanauMonotonicAttention(
    3,
    2,
    2,
    False,
    score_mask_value=torch.tensor([1.1]),
    sigmoid_noise=0.0,
    score_bias_init=1.7,
)
queryWeights = torch.tensor(
    [[-0.144869, 0.342418], [-0.365538, 0.267883], [0.123214, 0.565657676]]
)
memoryWeights = torch.tensor(
    [[-0.0409466, -0.260044], [0.0764357, -0.269967], [0.675676, 0.234354545]]
)
attention_v = torch.tensor([-0.633191, 0.234963, -0.391515])
attn.attention_v = torch.nn.Parameter(attention_v, requires_grad=True)
attn.query_layer.weight = torch.nn.Parameter(queryWeights, requires_grad=True)
attn.memory_layer.weight = torch.nn.Parameter(memoryWeights, requires_grad=True)
memorySeqLength = torch.tensor([1.0, 1.0])

# Forward steps
y1 = attn.forward(query1, state1, memory1, memorySeqLength)[0]
z1 = y1.unsqueeze(1) * attn.values
y2 = attn.forward(query2, state2, z1, memorySeqLength)[0]
z2 = attn.values * y2.unsqueeze(1)

# Backward
z2.sum().backward(retain_graph=True)
z1.sum().backward(retain_graph=True)

print(query1.grad)
print(query2.grad)

print(state1.grad)
print(state2.grad)

print(memory1.grad)

print(attn.attention_v.grad)
print(attn.attention_score_bias.grad)
