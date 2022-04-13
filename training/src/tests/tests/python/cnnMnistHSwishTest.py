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
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import os

curdir = "./weights/"
num_classes = 10


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 1, 1, 0)
        self.hsigm = h_sigmoid()
        self.fc = nn.Linear(1568, 10)
        self.softmax = nn.Softmax()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.hsigm(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        out = self.softmax(out)
        return out


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def predict(test_loader, model):
    correct = 0
    total = 0
    # ~ with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(
        "Accuracy of the network on the 10000 test images: {:.2f} %".format(
            100 * correct / total
        )
    )


def printModel(model, file):
    for i in model.state_dict():
        file.write(len(model.state_dict()[i]).to_bytes(4, byteorder="big"))
        np.ndarray.tofile(model.state_dict()[i].detach().numpy(), file, format="%f")


if __name__ == "__main__":
    batch_size = 50
    train_dataset = mnist.MNIST(root="./train", train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root="./test", train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model()
    sgd = SGD(model.parameters(), lr=1e-2)
    cross_error = CrossEntropyLoss()
    epoch = 1
    predict(test_loader, model)
    for _epoch in range(epoch):

        for i, (images, labels) in enumerate(train_loader):

            outputs = model(images)

            loss, lossInput = CrossEntropy(outputs, labels)
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            """if i % 100 == 0:
                with open(curdir + 'loss.txt', 'a') as outfile:
                    print(loss.item(), file=outfile)"""

            if i % 100 == 0:
                print("Step [{:4d}], Loss: {:.6f}".format(i, loss.item()))

        print("Epocha: ", _epoch)
        predict(test_loader, model)

    with open("dump.bin", "wb") as file:
        printModel(model, file)
