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

import texar.torch
import torch

# Default
torch.manual_seed(42)
query = torch.rand(2, 3, requires_grad=True)
state = torch.rand(2, 5, requires_grad=True)
memory = torch.rand(2, 5, 7, requires_grad=True)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
## Create attention
attn = texar.torch.core.BahdanauMonotonicAttention(
    3, 3, 7, False, score_mask_value=1.1, sigmoid_noise=0.0, score_bias_init=1.7
)
## Hint to reproduce the result in Raul
queryWeights = torch.tensor(
    [
        [-0.144869, 0.342418, 0.52044],
        [-0.365538, 0.267883, 0.322959],
        [0.113921, 0.111833, -0.397195],
    ]
)
memoryWeights = torch.tensor(
    [
        [-0.0409466, -0.260044, -0.302391, -0.334057, -0.0308049, 0.276803, -0.125704],
        [0.0764357, -0.269967, 0.157288, 0.114061, -0.362404, -0.335321, 0.355218],
        [0.167815, 0.251303, 0.331515, -0.217451, -0.377376, -0.240518, 0.372077],
    ]
)
attention_v = torch.tensor([-0.633191, 0.234963, -0.391515])
attn.attention_v = torch.nn.Parameter(attention_v, requires_grad=True)
attn.query_layer.weight = torch.nn.Parameter(queryWeights, requires_grad=True)
attn.memory_layer.weight = torch.nn.Parameter(memoryWeights, requires_grad=True)

## With mask logic
masks = [[5, 5], [2, 4], [3, 2]]
for i in range(len(masks)):
    print("Mask: ", masks[i])
    z = attn.forward(query, state, memory, torch.tensor(masks[i]))[0]
    print("Result: ", z)
    # Backward
    z.sum().backward()
    print("Gradient for attention_v:", attn.attention_v.grad)
    print("Gradient for attention_score_bias:", attn.attention_score_bias.grad)

# Normalized
torch.manual_seed(0)
torch.set_printoptions(precision=10)
query = torch.rand(3, 5, requires_grad=True)
state = torch.rand(3, 6, requires_grad=True)
memory = torch.rand(3, 6, 7, requires_grad=True)
print("query: ", query)
print("state: ", state)
print("memory: ", memory)
## Hint to reproduce the result
queryWeights = torch.tensor(
    [
        [-0.112215, 0.265236, 0.403131, -0.283145, 0.207502],
        [0.250163, 0.0882428, 0.0866254, -0.307666, -0.0484487],
        [-0.307688, -0.357793, -0.395262, -0.0364489, 0.327518],
        [-0.148736, 0.09044, -0.31943, 0.186106, 0.134959],
    ]
)
memoryWeights = torch.tensor(
    [
        [-0.362404, -0.335321, 0.355218, 0.167815, 0.251303, 0.331515, -0.217451],
        [-0.377376, -0.240518, 0.372077, -0.239324, 0.0888077, -0.147979, 0.0844018],
        [0.0187141, -0.372623, -0.0514447, -0.360531, -0.157816, 0.0187279, 0.0845528],
        [-0.075698, -0.272517, -0.342689, -0.157124, 0.358126, -0.101021, -0.202006],
    ]
)
attention_v = torch.tensor([-0.076089, -0.70909, 0.493939, 0.205051])
attn.attention_v = torch.nn.Parameter(attention_v, requires_grad=True)
attn.query_layer.weight = torch.nn.Parameter(queryWeights, requires_grad=True)
attn.memory_layer.weight = torch.nn.Parameter(memoryWeights, requires_grad=True)

## Mask logic
masks = [[2, 3, 4], [4, 3, 2], [7, 7, 2]]
values = [[1], [1, 2, 3, 4, 5, 6], [10.0, -10.0, 0.013213, 1.0, 1.123, -1]]
for i in range(len(masks)):
    print("Mask: ", masks[i])
    attn = texar.torch.core.BahdanauAttention(
        4, 5, 7, True, score_mask_value=torch.tensor(values[i])
    )  # , 0.0, 1.7
    attn.attention_v = torch.nn.Parameter(attention_v, requires_grad=True)
    attn.query_layer.weight = torch.nn.Parameter(queryWeights, requires_grad=True)
    attn.memory_layer.weight = torch.nn.Parameter(memoryWeights, requires_grad=True)
    z = attn.forward(query, state, memory, torch.tensor(masks[i]))[0]
    print("Result: ", z)
    # Backward
    z.sum().backward()
    print("Gradient for attention_v:", attn.attention_v.grad)
    print("Gradient for attention_b:", attn.attention_b.grad)
    print("Gradient for attention_g:", attn.attention_g.grad)
    print("Gradient for attention_score_bias:", attn.attention_score_bias.grad)

# Multiple forward

## First
query1 = torch.rand(2, 2, requires_grad=True)
state1 = torch.rand(2, 3, requires_grad=True)
memory1 = torch.rand(2, 3, 2, requires_grad=True)
## Second
query2 = torch.rand(2, 2, requires_grad=True)
state2 = torch.rand(2, 3, requires_grad=True)
memory2 = torch.rand(2, 3, 2, requires_grad=True)

## Attention
attn = texar.torch.core.BahdanauMonotonicAttention(
    3, 2, 2, False, sigmoid_noise=0.0, score_bias_init=1.7
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

## Run
z1 = attn.forward(query1, state1, memory1)
print("1st step result: ", z1[0])

z2 = attn.forward(query2, state2, memory2)
print("2nd step result: ", z2[0])

z2[0].sum().backward(retain_graph=True)
print("Current gradient for attention_v:", attn.attention_v.grad)
print("Current gradient for attention_score_bias:", attn.attention_score_bias.grad)

z1[0].sum().backward()
print("Final gradient for attention_v:", attn.attention_v.grad)
print("Final gradient for attention_score_bias:", attn.attention_score_bias.grad)
