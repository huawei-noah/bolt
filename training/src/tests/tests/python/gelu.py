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
import math
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelua(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


KK = torch.tensor(
    [
        [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]],
        [
            [[1.0, 1.0], [3.0, 3.0]],
            [[4.0, 4.0], [3.0, 3.0]],
            [[2.0, 1.0], [3.0, 7.0]],
        ],
    ],
    requires_grad=True,
)

grad = torch.tensor(
    [
        [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]],
        [
            [[1.0, 1.0], [3.0, 3.0]],
            [[4.0, 4.0], [3.0, 3.0]],
            [[2.0, 1.0], [3.0, 7.0]],
        ],
    ]
)

res = gelu(KK)
print("gelu", res.view(24))
res.backward(grad)
print("grad", KK.grad.view(24))
torch.nn.init.zeros_(KK.grad)

res = gelua(KK)
print("gelua", res.view(24))
res.backward(grad)
print("grada", KK.grad.view(24))
torch.nn.init.zeros_(KK.grad)
