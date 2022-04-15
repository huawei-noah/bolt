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
import torchvision
import torchvision.transforms as transforms
from typing import Optional, List
from .logging import logging


class Dataset:
    def __init__(
        self,
        name: str,
        batch_size: int,
        root: str = "./data",
        shuffle: bool = False,
        drop_last: bool = True,
        train_transform: Optional[List] = None,
        test_transform: Optional[List] = None,
        **kwargs,
    ):
        train_transform = train_transform or [transforms.ToTensor()]
        test_transform = test_transform or [transforms.ToTensor()]

        if name not in torchvision.datasets.__all__:
            datasets = "\n".join(torchvision.datasets.__all__)
            raise RuntimeError(
                f"Unknown dataset name `{name}`. Availible datasets:\n{datasets}"
            )
        logging.info(f"Loading {name} dataset...")

        dataset_object = getattr(torchvision.datasets, name)

        self.train_dataset = dataset_object(
            root=f"{root}/{name}",
            train=True,
            transform=transforms.Compose(train_transform),
            download=True,
        )

        self.test_dataset = dataset_object(
            root=f"{root}/{name}",
            train=False,
            transform=transforms.Compose(test_transform),
            download=True,
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
        )
