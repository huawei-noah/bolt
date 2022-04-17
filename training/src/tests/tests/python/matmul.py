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

Q = torch.tensor(
    [
        [[1.0, 1.0, 2.0, 0.0, 5.0], [-1.0, 2.0, 2.0, 0.0, 5.0]],
        [[1.0, 1.0, 2.0, 0.0, 5.0], [-1.0, 2.0, 2.0, 0.0, 5.0]],
    ],
    requires_grad=True,
)
K1 = torch.tensor(
    [
        [[-1.0, 4.0, 1.0, 2.0, 1], [-3.0, 4.0, 5.0, 2.0, 1], [-3.0, 3.0, 5.0, 1.0, 1]],
        [
            [-2.0, 8.0, 2.0, 4.0, 2],
            [-6.0, 8.0, 10.0, 4.0, 2],
            [-6.0, 6.0, 10.0, 2.0, 2],
        ],
    ],
    requires_grad=True,
)

K = K1.transpose(-2, -1)
K.retain_grad()
print("K", K.reshape(30))

res = torch.matmul(Q, K)
res.backward(
    torch.tensor([[[1.0, 2.0, -1], [1.0, 0.5, 6.0]], [[1.0, 2.0, -1], [1.0, 0.5, 6.0]]])
)

print(Q.grad)
print(K.grad)

print("res", res.reshape(12))
