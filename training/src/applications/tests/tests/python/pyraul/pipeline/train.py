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
import os
from enum import Enum
from typing import Callable, Optional, List
from ..losses import Loss
from ..tools.logging import logging
from ..tools.dataset import Dataset
from ..tools.timing import profile_acc
from ..tools.dumping import dump_weights, dump_loss, DumpConfig, add_index_to_filepath

_default_total_steps = 100


class TrainHistory(Enum):
    LOSS = 0


def train(
    model: torch.nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    loss_func: Callable[[torch.Tensor, torch.Tensor], Loss],
    trace_loss: Optional[DumpConfig] = None,
    trace_weights: Optional[DumpConfig] = None,
    preprocessor: Optional[Callable] = None,
    history: TrainHistory = TrainHistory.LOSS,
    device="cpu",
    **kwargs,
) -> List:
    f"""
    Train function
    :param model: Neural network model
    :param dataset: Wrapping object that contains data loaders
    :param optimizer: Pytorch optimizer
    :param loss_func: Loss function
    :param trace_loss: enables loss tracing if config is provided
    :param trace_weights: enables weights tracing if config is provided
    :param preprocessor: Callable object which is preprocess data
    :param kwargs: Other arguments in dictionary
    :return: None
    """
    total_steps = kwargs.get("total_steps", _default_total_steps)
    history_data = []

    if trace_loss and os.path.exists(trace_loss.filename):
        os.remove(trace_loss.filename)

    # Note: This is a workaround.
    # Python doesn't have references, so it is impossible
    # to pass an integer counter to the function for incrementing.
    # We use here for this purpose list which is mutable entity.
    # Think about better solutions, e.g. keep accumulator inside a wrapped function and so on.
    total_time = [0]

    model.train()

    for i, (data, labels) in enumerate(dataset.train_loader):
        loss = None
        if preprocessor:
            data = preprocessor(data)
        data = data.to(device)

        if trace_weights and i % trace_weights.step == 0:
            indexed_filename = add_index_to_filepath(trace_weights.filename, i)
            dump_weights(model, indexed_filename)

        @profile_acc(total_time)
        def _train_step():
            nonlocal loss
            outputs = model(data)
            loss, _ = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _train_step()

        # It can be extended
        history_val = loss if history == TrainHistory.LOSS else None
        history_data.append(history_val)

        if trace_loss and i % trace_loss.step == 0:
            dump_loss(loss, trace_loss.filename)
            logging.info(f"Step [{i:4d}/{total_steps}], Loss: {loss:.10f}")

    logging.info(f"Time taken = {total_time[0]:.4f}")

    return history_data
