# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import time
from collections import namedtuple
from typing import Callable, Optional, List
from ..tools.logging import get_fixedwide_str


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, history: bool = False):
        self.use_history = history
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.use_history:
            self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.use_history:
            self.history.append(val)


TrainStepResult = namedtuple(
    "TrainStepResult", ["loss", "time_batch_load", "time_batch_full"]
)


def train_step(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    print_freq=1,
    verbose: bool = True,
    loss_history: bool = False,
    preprocessor: Optional[Callable] = None,
):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(history=loss_history)

    model.train()

    n = len(train_loader)
    n_wide = len(str(n))

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if preprocessor:
            input = preprocessor(input)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % print_freq == 0:
            print(
                f"Step {get_fixedwide_str(str(i), n_wide)}/{n}\t"
                f"Loss: {losses.val:.6f} ({losses.avg:.6f})\t"
                f"Time.step: {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Time.load: {data_time.val:.3f} ({data_time.avg:.3f})"
            )
    return TrainStepResult(
        loss=losses, time_batch_load=data_time, time_batch_full=batch_time
    )
