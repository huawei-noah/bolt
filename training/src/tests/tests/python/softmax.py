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

for i in range(0, 4):
    x = F.softmax(KK, dim=i)
    print("dim = " + str(i), x.view(24))
    x.backward(grad)
    print("grad(dim = " + str(i) + ")", KK.grad.view(24))
    torch.nn.init.zeros_(KK.grad)
# print('dim = 1', F.softmax(KK, dim = 1))
# print('dim = 2', F.softmax(KK, dim = 2))
# print('dim = 3', F.softmax(KK, dim = 3))
