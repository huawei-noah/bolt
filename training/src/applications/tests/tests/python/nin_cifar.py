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

        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.relu6 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.drop2 = nn.Dropout(0.5)

        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.relu9 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.conv9.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.conv5.bias)
        nn.init.zeros_(self.conv6.bias)
        nn.init.zeros_(self.conv7.bias)
        nn.init.zeros_(self.conv8.bias)
        nn.init.zeros_(self.conv9.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.mp(out)
        # out = self.drop1(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.avg1(out)
        # out = self.drop2(out)

        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.avg2(out)

        out = out.view(out.size(0), num_classes)
        out = self.softmax(out)
        return out


def predict(test_loader, model):
    correct = 0
    total = 0
    # ~ with torch.no_grad():
    for images, labels in test_loader:
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
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
                            np.savetxt(outfile, param.data[i, j].cpu())
            else:
                with open(curdir + str(index) + "_" + name + ".txt", "w") as outfile:
                    np.savetxt(outfile, np.transpose(param.data.cpu()))


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    ones = Variable(ones.cuda())
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def main():

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = NeuralNet(num_classes)
    model.cuda()

    predict(test_loader, model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    if os.path.exists(curdir + "loss.txt"):
        os.remove(curdir + "loss.txt")

    for q in range(0, 100):

        print("Epoch = {:d}".format(q + 1))
        timeTaken = 0

        for i, (images, labels) in enumerate(train_loader):

            images, labels = Variable(images.cuda()), Variable(labels.cuda())

            start = time.time()

            outputs = model(images)
            #            print(outputs)
            #            if i == 0 and q == 9:
            #                saveWeights(i, model)

            loss, lossInput = CrossEntropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            timeTaken += time.time() - start

            #            if i % 100 == 0:
            #                 with open(curdir + 'loss.txt', 'a') as outfile:
            #                    print(loss.item(), file = outfile)

            if i % 100 == 0:
                print(
                    "Step [{:4d}/{}], Loss: {:.6f}".format(i, total_step, loss.item())
                )

        predict(test_loader, model)

        print("Time taken = {:.4f}".format(timeTaken))


if __name__ == "__main__":
    main()
