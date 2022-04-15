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
from torch.utils.data import DataLoader
from typing import Callable, Optional


def accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    preprocessor: Optional[Callable] = None,
    device: str = "cpu",
    squeeze_target: bool = False,
    **kwargs,
) -> float:
    """
    The function returns an accuracy score in percentages.

    Accuracy = correct answer / total answers

    :param model: Neural network model
    :param dataset: Wrapping object that contains data loaders
    :param preprocessor: Callable object which is preprocess data
    :param kwargs: Other arguments in dictionary
    :return:
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            if preprocessor:
                data = preprocessor(data)
            data = data.to(device)
            labels = labels.to(device)
            if squeeze_target:
                labels = labels.squeeze()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += outputs.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total
