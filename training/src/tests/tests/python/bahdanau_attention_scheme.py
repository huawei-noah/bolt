# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Step by step example of all versions of Bahdanau Attention
import torch

torch.manual_seed(0)

# Inputs
torch.set_printoptions(precision=10)
query = torch.rand(3, 5, requires_grad=True)
state = torch.rand(3, 6, requires_grad=True)
memory = torch.rand(3, 6, 7, requires_grad=True)
print("query:", query)
print("state:", state)
print("memory:", memory)

# Settings
num_units = 4
sigmoid_noise = 0.0
score_bias_init = 1.7
normalize = False
stepwise = False
score_mask_value = torch.tensor([1.1])

# Size: [query.shape[1], num_units]
queryWeights = torch.tensor(
    [
        [-0.112215, 0.265236, 0.403131, -0.283145, 0.207502],
        [0.250163, 0.0882428, 0.0866254, -0.307666, -0.0484487],
        [-0.307688, -0.357793, -0.395262, -0.0364489, 0.327518],
        [-0.148736, 0.09044, -0.31943, 0.186106, 0.134959],
    ]
)
# Size: [memory.shape[1], num_units]
memoryWeights = torch.tensor(
    [
        [-0.362404, -0.335321, 0.355218, 0.167815, 0.251303, 0.331515, -0.217451],
        [-0.377376, -0.240518, 0.372077, -0.239324, 0.0888077, -0.147979, 0.0844018],
        [0.0187141, -0.372623, -0.0514447, -0.360531, -0.157816, 0.0187279, 0.0845528],
        [-0.075698, -0.272517, -0.342689, -0.157124, 0.358126, -0.101021, -0.202006],
    ]
)
attention_v = torch.tensor([-0.076089, -0.70909, 0.493939, 0.205051])
attention_g = torch.sqrt(torch.tensor(1.0 / num_units))
memory_seq_length = torch.tensor([4, 3, 2])

# Forward path

## Step1:
queryProcessed = torch.matmul(query, torch.transpose(queryWeights, 0, 1))

## Mask to hide some elems of memory tensor
max_len = memory.shape[1]
rank = memory.dim()
size = memory_seq_length.size()
row_vector = torch.arange(max_len).view(*([1] * len(size)), -1).expand(*size, max_len)
mask = row_vector < memory_seq_length.unsqueeze(-1)

## Step2:
memoryProcessed = torch.matmul(
    memory * mask.unsqueeze(2), torch.transpose(memoryWeights, 0, 1)
)
memoryProcessed

## Step3:
sum_ = memoryProcessed + torch.unsqueeze(queryProcessed, 1)

## Step 4:
tanh_ = torch.tanh(sum_)

## Step 5:
if not normalize:
    mul_ = attention_v * tanh_
else:
    mul_ = attention_g * attention_v * torch.rsqrt(torch.sum(attention_v ** 2)) * tanh_

## Step 6:
rsum_ = torch.sum(mul_, 2)

## Step 7:
biased_sum_ = rsum_ + score_bias_init

## Step 8:
score_mask_values = score_mask_value * torch.ones_like(mask)
res_ = torch.where(mask, biased_sum_, score_mask_values)

## Step 9:
noise = torch.randn(res_.shape)
res_ += sigmoid_noise * noise
res_

## Step 10:
sigm_ = torch.sigmoid(res_)

## Step 11:
rsigm_ = 1 - sigm_[:, :-1]

if stepwise:
    ## Step 12:
    batch_size = sigm_.shape[0]
    pad = sigm_.new_zeros(batch_size, 1)

    ## Step 13:
    tmp_state = state[:, :-1] * rsigm_

    ## Step 14:
    tmp_conc = torch.cat((pad, tmp_state), dim=1)

    ## Step 15:
    pre_final = state * sigm_

    # First output
    attn = pre_final + tmp_conc

    # Second output
    indices = torch.argmax(attn, -1)

else:
    ## Step 12:
    batch_size = sigm_.shape[0]
    shifted_1 = torch.cat((sigm_.new_ones(batch_size, 1), 1 - sigm_[:, :-1]), 1)

    ## Step 13:
    tiny = torch.finfo(shifted_1.dtype).tiny
    z1 = torch.clamp(shifted_1, tiny, 1)

    ## Step 14:
    z2 = torch.log(z1)

    ## Step 15:
    z3 = torch.cumsum(z2, dim=1)

    ## Step 16:
    z4 = torch.exp(z3)

    ## Step 17:
    z5 = state / z4.clamp(min=1e-10, max=1.0)

    ## Output
    attn = sigm_ * z4 * z5

# Backward

attn.sum().backward()

print("query grad: ", query.grad)
print("state grad: ", state.grad)
print("memory grad: ", memory.grad)
