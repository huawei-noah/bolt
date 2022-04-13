# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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
from pyraul.nn import MLP
from pyraul.pipeline import train, predict
from pyraul.losses import cross_entropy
from pyraul.tools.dataset import Dataset
from pyraul.tools.logging import logging
from pyraul.tools.seed import use_seed
from pyraul.tools.dumping import DumpConfig
from typing import Callable, Optional

_seed = 0

_config = {
    "in": 784,
    "hidden_1": 500,
    "hidden_2": 100,
    "out": 10,  # number of classes
    "batch_size": 50,
    "sgd": {
        "lr": 0.1,
    },
    "sgd_momentum": {"lr": 0.1, "momentum": 0.5},
    "sgd_momentum_nesterov": {"lr": 0.1, "momentum": 0.5},
    "adam": {},
    "adamax": {},
    "adagrad": {},
    "adadelta": {},
}


@use_seed(_seed)
def trace_optimizer(
    optimizer_func: Callable,
    loss_filename: Optional[str] = None,
    weights_filename: Optional[str] = None,
) -> None:
    """
    Function runs training MLP network with specified optimizzer
    :param optimizer_func: Function returning optimuzer, takes model as argument
    :param loss_filename: Path to loss trace file
    :param weights_filename: Path to weights trace file
    :return:
    """
    logging.info(f"IEEE 754 Precision: {torch.Tensor().type()}")
    # Configuration
    model = MLP(**_config)
    logging.info(f"Architecture:\n{model}")
    ds = Dataset("MNIST", **_config)
    optimizer = optimizer_func(model)
    # Inference
    accuracy_before = predict(
        model=model,
        dataset=ds,
        preprocessor=lambda images: images.reshape(-1, 28 * 28),
        **_config,
    )
    logging.info(
        f"Accuracy of the network on the 10000 test images: {accuracy_before:.2f} %"
    )
    # Train
    steps = len(ds.train_loader)
    train(
        model=model,
        dataset=ds,
        optimizer=optimizer,
        loss_func=cross_entropy,
        total_steps=steps,
        trace_loss=loss_filename and DumpConfig(filename=loss_filename, step=100),
        trace_weights=weights_filename
        and DumpConfig(filename=weights_filename, step=100),
        preprocessor=lambda images: images.reshape(-1, 28 * 28),
        **_config,
    )
    # Inference
    accuracy_after = predict(
        model=model,
        dataset=ds,
        preprocessor=lambda images: images.reshape(-1, 28 * 28),
        **_config,
    )
    logging.info(
        f"Accuracy of the network on the 10000 test images: {accuracy_after:.2f} %"
    )


def main():
    # Default parameters
    optimizers = {
        "sgd": lambda model: torch.optim.SGD(model.parameters(), **_config["sgd"]),
        "sgd_momentum": lambda model: torch.optim.SGD(
            model.parameters(), **_config["sgd_momentum"]
        ),
        "sgd_momentum_nesterov": lambda model: torch.optim.SGD(
            model.parameters(), **_config["sgd_momentum_nesterov"], nesterov=True
        ),
        "adadelta": lambda model: torch.optim.Adadelta(
            model.parameters(),
            **_config["adadelta"],
        ),
        "adagrad": lambda model: torch.optim.Adagrad(
            model.parameters(),
            **_config["adagrad"],
        ),
        "adam": lambda model: torch.optim.Adam(
            model.parameters(),
            **_config["adam"],
        ),
        "adamax": lambda model: torch.optim.Adamax(
            model.parameters(), **_config["adamax"]
        ),
    }

    for name in optimizers.keys():
        logging.info(f"Trace optimizer {name}")
        optimizer_func = optimizers[name]
        trace_optimizer(
            optimizer_func,
            loss_filename=f"{name}_loss.txt",
            weights_filename=f"{name}_weights.txt",
        )


if __name__ == "__main__":
    main()
