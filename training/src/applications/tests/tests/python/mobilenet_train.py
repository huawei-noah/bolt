# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from matplotlib import pyplot as plt
import torch
from mobilenetv3 import mobilenetv3_small
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import cv2
import numpy as np
import torch.nn as nn

curdir = "./weight/"


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


_batch = 128
_threads = 12
learning_rate = 0.1
num_classes = 10
_epochs = 10


def printModel(index, model):
    for name, param in model.named_parameters():
        print(name)


def predict(test_loader, model):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    ones = Variable(ones.cuda())
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def main():
    net_small = mobilenetv3_small()
    # net_small.load_state_dict(torch.load('mobilenetv3.pth'))

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/",
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(224, 0), transforms.ToTensor()]
        ),
        download=True,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/",
        train=False,
        transform=transforms.Compose(
            [transforms.Resize(224, 0), transforms.ToTensor()]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=_batch, shuffle=False, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=_batch, shuffle=False, drop_last=True
    )

    print("Number of samples: ", len(train_loader))

    net_small.cuda()
    predict(test_loader, net_small)
    optimizer = torch.optim.Adam(net_small.parameters())

    total_step = len(train_loader)

    acc = predict(test_loader, net_small)

    print("Accuracy of the network on the 10000 test images: {:.6f} %".format(acc))

    for q in range(0, _epochs):

        print("Epoch = {:d}".format(q + 1))
        timeTaken = 0
        start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            """for img in images:
            img = np.transpose(img,(1,2,0))
            plt.imshow(img)
            plt.show()"""
            images, labels = Variable(images.cuda()), Variable(labels.cuda())

            outputs = net_small(images)
            # print(outputs)
            # if i < 1:
            criterian = nn.NLLLoss()

            # loss = criterian(outputs, labels)
            loss = criterian(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #        if i % 100 == 0:
            #            with open(curdir + 'loss.txt', 'a') as outfile:
            #                print(loss.item(), file = outfile)

            if i % 100 == 0:
                print(
                    "Step [{:4d}/{}], Loss: {:.6f}".format(i, total_step, loss.item())
                )

        print("time of train: ", time.time() - start)

        acc = predict(test_loader, net_small)

        print("Accuracy of the network on the 10000 test images: {:.6f} %".format(acc))

        torch.save(net_small, "model/mobilenetv3_small" + "batch128_" + str(q) + ".pt")
        print("Time taken = {:.4f}".format(timeTaken))
    # print(net_small)


if __name__ == "__main__":
    main()
