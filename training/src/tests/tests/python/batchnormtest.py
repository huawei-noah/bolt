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
from torch.nn import Module


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch = nn.BatchNorm2d(5)

    def forward(self, x):

        out = self.batch(x)
        return out


x = np.array(
    [
        0.2992,
        0.0614,
        0.3442,
        0.4992,
        0.1848,
        0.3404,
        0.3627,
        0.6232,
        0.5426,
        0.1261,
        0.9982,
        0.7149,
        0.8062,
        0.6040,
        0.0333,
        0.3870,
        0.2276,
        0.0830,
        0.0222,
        0.9375,
        0.9395,
        0.4894,
        0.4846,
        0.3932,
        0.3220,
    ]
)

x = np.reshape(x, (1, 5, 1, 5))

x = torch.Tensor(x)
model = Model()
model.eval()
out = model(x)
print(out)
