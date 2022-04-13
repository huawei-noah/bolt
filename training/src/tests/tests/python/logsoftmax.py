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

for i in range(1, 4):
    x = F.log_softmax(KK, dim=i)
    print("dim = " + str(i), x.view(24))
    x.backward(grad)
    print("grad(dim = " + str(i) + ")", KK.grad.view(24))
    torch.nn.init.zeros_(KK.grad)


KK = torch.tensor(
    [
        [
            [
                [-0.6226, 0.4617],
                [-1.4773, 0.6215],
                [-0.1313, 0.3401],
                [-0.1132, 0.3223],
                [-0.6378, 0.4601],
                [-0.9472, 0.8423],
                [-0.4721, 0.7331],
                [-0.9548, 0.7353],
                [-0.7862, 0.9551],
                [-0.2192, 0.7782],
                [-0.6109, 0.9250],
                [-0.4453, 0.5458],
                [-0.7532, 0.2978],
                [-0.5299, 0.6211],
                [-0.5609, 0.8073],
                [0.1008, 0.8441],
                [-0.6271, 0.2552],
                [-0.3445, 0.8781],
                [-0.5200, 0.6886],
                [-0.5070, 0.5257],
                [-0.2960, 0.3555],
                [-1.1601, 0.5381],
                [-0.5758, 0.6153],
                [-1.0586, 0.7958],
                [-1.0672, 0.5650],
                [-0.7843, 1.1598],
                [-0.7141, 0.3838],
                [-0.0126, 0.5464],
                [0.0409, 0.5775],
                [-0.8350, 0.7043],
                [-0.7880, 0.1792],
                [-0.1886, 0.6843],
            ]
        ]
    ],
    requires_grad=True,
)

grad = torch.tensor(
    [
        [
            [
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [0.0000, -0.0312],
                [0.0000, -0.0312],
                [0.0000, -0.0312],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
                [0.0000, -0.0312],
                [-0.0312, 0.0000],
                [-0.0312, 0.0000],
                [0.0000, -0.0312],
            ]
        ]
    ]
)

x = F.log_softmax(KK, dim=3)
print("dim = " + str(3), x.view(64))
x.backward(grad)
print("grad(dim = " + str(3) + ")", KK.grad.view(64))

# print('dim = 1', F.softmax(KK, dim = 1))
# print('dim = 2', F.softmax(KK, dim = 2))
# print('dim = 3', F.softmax(KK, dim = 3))
