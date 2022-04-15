# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from pyraul.tools.seed import set_seed
from enum import Enum
from pyraul.tools.dataset import Dataset
from pyraul.tools.dumping import save_checkpoint
from pyraul.pipeline.train_step import train_step
from pyraul.pipeline.inference import accuracy
import torchvision.transforms as transforms

set_seed(0)

# inplace=True means that x will be overwritten which reduces memory usage
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


# inplace=True means that x will be overwritten which reduces memory usage
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Output 1x1xC (parameters adapts automatically)
            nn.Conv2d(
                in_size,
                in_size // reduction,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),  # FC
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_size // reduction,
                in_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),  # FC
            nn.BatchNorm2d(in_size),
            hsigmoid(),
        )

    def forward(self, x):
        # Important: here we weight original features x using weights self.se(x)
        return x * self.se(x)


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride
    ):
        super().__init__()
        self.stride = stride
        self.se = semodule

        # 1x1, NL
        self.conv1 = nn.Conv2d(
            in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear

        # Dwise
        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear

        # Lineaer
        self.conv3 = nn.Conv2d(
            expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_size)

        # For stride=2 no shorcut
        self.shortcut = nn.Sequential()

        # For stride=1 blocks
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        # 1x1, NL
        out = self.nolinear1(self.bn1(self.conv1(x)))
        # Dwise
        out = self.nolinear2(self.bn2(self.conv2(out)))
        # Lineaer
        out = self.bn3(self.conv3(out))

        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()

        self.linear3 = nn.Linear(960, 1280)
        self.hs3 = hswish()

        self.linear4 = nn.Linear(1280, num_classes)

        # Initialization of some layers
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # conv2d
        out = self.hs1(self.bn1(self.conv1(x)))
        # bneck stack
        out = self.bneck(out)
        # conv2d
        out = self.hs2(self.bn2(self.conv2(out)))
        # pool
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        # conv2d, NBN
        out = self.hs3(self.linear3(out))
        # conv2d, NBN
        out = self.linear4(out)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()

        self.linear3 = nn.Linear(576, 1024)
        self.hs3 = hswish()

        self.linear4 = nn.Linear(1024, num_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # conv2d
        out = self.hs1(self.bn1(self.conv1(x)))
        # bneck stack
        out = self.bneck(out)
        # conv2d
        out = self.hs2(self.bn2(self.conv2(out)))
        # pool
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        # conv2d, NBN
        out = self.hs3(self.linear3(out))
        # conv2d, NBN
        out = self.linear4(out)
        return out


class ClassifierType(Enum):
    Small = "small"
    Large = "large"


class Classifier(nn.Module):
    def __init__(self, model: ClassifierType, num_classes):
        super().__init__()
        self.mobilenetv3 = None
        if model == ClassifierType.Small:
            self.mobilenetv3 = MobileNetV3_Small(num_classes)
        if model == ClassifierType.Large:
            self.mobilenetv3 = MobileNetV3_Large(num_classes)
        assert self.mobilenetv3
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.mobilenetv3(x)
        x = self.softmax(x)
        return x


def run_training(
    optimizer_factory,
    model_type,
    batch_size=64,
    epochs=10,
    device: str = "cpu",
    multigpu: bool = False,
    filename="checkpoint.pth.tar",
):
    ds = Dataset(
        "CIFAR10",
        batch_size,
        train_transform=[
            transforms.Resize(224, interpolation=0),
            transforms.ToTensor(),
        ],
        test_transform=[transforms.Resize(224, interpolation=0), transforms.ToTensor()],
    )

    model = Classifier(model_type, 10)
    if multigpu:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optimizer_factory(model)

    criterion = nn.NLLLoss()

    history_acc = []
    history_train = []

    def get_acc(model):
        acc_val = accuracy(model, ds.test_loader, device=device)
        print(f"Accuracy: {acc_val:.6f}")
        return acc_val

    get_acc(model)

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:03}/{epochs:03}")
        train_val = train_step(
            ds.train_loader, model, criterion, optimizer, device, print_freq=100
        )
        history_train.append(train_val)
        acc = get_acc(model)
        history_acc.append(acc)
        save_checkpoint(
            {"epoch": epoch, "acc": acc, "state_dict": model.state_dict()},
            filename=filename,
        )
    return history_acc, history_train


parser = argparse.ArgumentParser(description="MobileNetV3 Explorer")
parser.add_argument("-g", "--gpu-id", default=0, type=int, metavar="N", help="GPU Id")
parser.add_argument(
    "-e",
    "--epochs",
    default=50,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")


def main():
    args = parser.parse_args()
    cuda_device = f"cuda:{args.gpu_id}"
    device = cuda_device if torch.cuda.is_available() else "cpu"

    optimizers = {
        "sgd": lambda model: torch.optim.SGD(model.parameters(), lr=args.lr),
        "sgd_momentum": lambda model: torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        ),
        "sgd_momentum_nesterov": lambda model: torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True
        ),
        "adadelta": lambda model: torch.optim.Adadelta(
            model.parameters(),
        ),
        "adagrad": lambda model: torch.optim.Adagrad(
            model.parameters(),
        ),
        "adam": lambda model: torch.optim.Adam(model.parameters()),
        "adamax": lambda model: torch.optim.Adamax(model.parameters()),
    }

    for optimizer_name in optimizers.keys():
        optimizer = optimizers[optimizer_name]
        for net_type in (ClassifierType.Small, ClassifierType.Large):
            print(
                f"Run {net_type=}, {optimizer_name=}, {args.batch_size=} on {args.epochs=}"
            )
            train_id = f"bs.{args.batch_size}_opt.{optimizer_name}_t.{net_type}_e.{args.epochs}"
            results = run_training(
                optimizer,
                net_type,
                args.batch_size,
                args.epochs,
                device=device,
                multigpu=False,
                filename=f"{train_id}_checkpoint.pth.tar",
            )
            save_checkpoint(results, filename=f"{train_id}_results.pth.tar")


if __name__ == "__main__":
    main()
