# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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
import time

num_classes = 10
batch_size = 50
learning_rate = 0.01

curdir = "./weights/"


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()

        torch.manual_seed(0)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        self.batch1 = nn.BatchNorm2d(16, momentum=0.1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.batch2 = nn.BatchNorm2d(32, momentum=0.1)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        self.conv1Input = x
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = out.reshape(-1, 32 * 6 * 6)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


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
        "Accuracy of the network on the 10000 test images: {:.6f} %".format(
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
                            np.savetxt(outfile, param.data[i, j])
            else:
                with open(curdir + str(index) + "_" + name + ".txt", "w") as outfile:
                    np.savetxt(outfile, np.transpose(param.data))


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def main():

    train_dataset = torchvision.datasets.MNIST(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data/", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = NeuralNet(num_classes)

    model.eval()
    predict(test_loader, model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    if os.path.exists(curdir + "loss.txt"):
        os.remove(curdir + "loss.txt")

    timeTaken = 0
    start = time.time()

    for epoch in range(1, 3):
        for i, (images, labels) in enumerate(train_loader):

            model.train()
            outputs = model(images)
            # print(outputs)
            # if i < 1:
            #    saveWeights(i, model)

            loss, lossInput = CrossEntropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i % 100 == 0:
            #    with open(curdir + 'loss.txt', 'a') as outfile:
            #        print(loss.item(), file = outfile)

            if i % 100 == 0:
                print(
                    "Step [{:4d}/{}], Loss: {:.6f}".format(i, total_step, loss.item())
                )

        model.eval()
        predict(test_loader, model)

    timeTaken += time.time() - start
    print("Time taken = {:.4f}".format(timeTaken))


if __name__ == "__main__":
    main()
