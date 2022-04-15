# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import math

curdir = "./weights/"

# https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# https://github.com/d-li14/mobilenetv2.pytorch/blob/master/imagenet.py

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.Conv2d(inp, oup, 3, stride, 1),
        # nn.BatchNorm2d(oup),
        nn.BatchNorm2d(oup, momentum=0.1),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.Conv2d(inp, oup, 1, 1, 0),
        # nn.BatchNorm2d(oup),
        nn.BatchNorm2d(oup, momentum=0.1),
        nn.ReLU6(inplace=True),
    )


def make_divisible(x, divisible_by=8):
    import numpy as np

    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
                # nn.BatchNorm2d(hidden_dim),
                nn.BatchNorm2d(hidden_dim, momentum=0.1),
                nn.ReLU6(inplace=True),
                # pw-linear
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0),
                # nn.BatchNorm2d(oup),
                nn.BatchNorm2d(oup, momentum=0.1),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.Conv2d(inp, hidden_dim, 1, 1, 0),
                # nn.BatchNorm2d(hidden_dim),
                nn.BatchNorm2d(hidden_dim, momentum=0.1),
                nn.ReLU6(inplace=True),
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
                # nn.BatchNorm2d(hidden_dim),
                nn.BatchNorm2d(hidden_dim, momentum=0.1),
                nn.ReLU6(inplace=True),
                # pw-linear
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0),
                # nn.BatchNorm2d(oup),
                nn.BatchNorm2d(oup, momentum=0.1),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        block = InvertedResidual
        width_mult = 1.0
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        self.last_channel = (
            make_divisible(last_channel * width_mult)
            if width_mult > 1.0
            else last_channel
        )
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t)
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t)
                    )
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.avg = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, 10)

        self.softmax = nn.LogSoftmax(dim=1)

        self._initialize_weights()

    def forward(self, x):
        # out = nn.functional.interpolate(x, (224, 224))
        out = x
        out = self.features(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.softmax(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def predict(test_loader, model):
    correct = 0
    total = 0
    # ~ with torch.no_grad():
    for images, labels in test_loader:
        model.eval()
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(
        "Accuracy of the network on the 10000 test images: {:.2f} %".format(
            100 * correct / total
        )
    )


def saveWeights(index, model):

    if not os.path.exists(curdir):
        os.mkdir(curdir)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.data.dim() == 4:
                for i in range(0, param.data.shape[0]):
                    with open(
                        curdir + str(index) + "_" + name + "_" + str(i) + ".txt", "w"
                    ) as outfile:
                        for j in range(0, param.data.shape[1]):
                            np.savetxt(outfile, param.data[i, j].cpu())
            else:
                with open(curdir + str(index) + "_" + name + ".txt", "w") as outfile:
                    np.savetxt(outfile, np.transpose(param.data.cpu()))


def saveState(index, model):

    if not os.path.exists(curdir):
        os.mkdir(curdir)

    for name, param in model.state_dict().items():
        if param.data.dim() == 4:
            for i in range(0, param.data.shape[0]):
                with open(
                    curdir + str(index) + "_" + name + "_" + str(i) + ".txt", "w"
                ) as outfile:
                    for j in range(0, param.data.shape[1]):
                        np.savetxt(outfile, param.data[i, j].cpu())
        elif param.data.dim() != 0:
            with open(curdir + str(index) + "_" + name + ".txt", "w") as outfile:
                np.savetxt(outfile, np.transpose(param.data.cpu()))


device = torch.device("cuda")
torch.cuda.set_device(1)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transforms.Compose(
        [transforms.Resize(224, interpolation=0), transforms.ToTensor()]
    ),
    download=True,
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transforms.Compose(
        [transforms.Resize(224, interpolation=0), transforms.ToTensor()]
    ),
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=50, drop_last=True, shuffle=False, num_workers=6
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=50, drop_last=True, shuffle=False, num_workers=6
)

model = NeuralNet()
model.cuda()
epoch_start = 1
criterion = nn.NLLLoss()
base_lr = 0.05
# base_lr = 0.005

# model.load_state_dict(torch.load('modelMobileNetv2_20'))
# model.to(device)

# print(model)

model.eval()
predict(test_loader, model)

optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

total_step = len(train_loader)

if os.path.exists(curdir + "loss.txt"):
    os.remove(curdir + "loss.txt")

for epoch in range(epoch_start, 2):
    print("Epoch: {:d}".format(epoch))
    model.train()

    # if epoch % 5 == 0:
    #    torch.save(model.state_dict(), 'modelMobileNetv2_{:d}'.format(epoch))

    for i, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        outputs = model(images)
        # if i == 0 and epoch == 1:
        #    saveWeights(i, model)
        #    saveState(i, model)
        # exit()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Step [{:4d}/{}], Loss: {:.6f}".format(i, total_step, loss.item()))

    model.eval()
    predict(test_loader, model)
