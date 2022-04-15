# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#!/usr/bin/env python
import torch

from ..activation_functions import Softmax


class MLP(torch.nn.Module):
    """
    Multilayer Perceptron

    Args:
        in: input size of the network
        hidden_1: size of the 1st hidden layer
        hidden_2: size of the 2nd hidden layer
        out: output size of the network (aka amount of classes)

    """

    def __init__(self, **config):
        super(MLP, self).__init__()

        assert "in" in config, "Config must contain `in` argument"
        assert "hidden_1" in config, "Config must contain `hidden_1` argument"
        assert "hidden_2" in config, "Config must contain `hidden_2` argument"
        assert "out" in config, "Config must contain `out` argument"

        net_size_in: int = config["in"]
        net_size_hidden_1: int = config["hidden_1"]
        net_size_hidden_2: int = config["hidden_2"]
        net_size_out: int = config["out"]

        self.fc1 = torch.nn.Linear(net_size_in, net_size_hidden_1)
        self.tanh1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(net_size_hidden_1, net_size_hidden_2)
        self.sigm1 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(net_size_hidden_2, net_size_out)
        self.softmax = Softmax()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.sigm1(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
