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

input_size = 784
hidden_size = 500
hidden_size2 = 100
num_classes = 10
batch_size = 50

curdir = "./weights/"


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        torch.manual_seed(0)

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

        # self.softmax = Softmax()
        self.softmax = nn.LogSoftmax(dim=1)

        # ~ print(self.classifier)

        # nn.init.xavier_uniform_(self.classifier[0].weight)
        # nn.init.xavier_uniform_(self.classifier[2].weight)
        # nn.init.xavier_uniform_(self.classifier[4].weight)

        # nn.init.xavier_uniform_(self.classifier[8].weight)
        # nn.init.xavier_uniform_(self.classifier[10].weight)
        # nn.init.xavier_uniform_(self.classifier[12].weight)

        # nn.init.xavier_uniform_(self.classifier[16].weight)
        # nn.init.xavier_uniform_(self.classifier[18].weight)
        # nn.init.xavier_uniform_(self.classifier[20].weight)

        # nn.init.zeros_(self.classifier[0].bias)
        # nn.init.zeros_(self.classifier[2].bias)
        # nn.init.zeros_(self.classifier[4].bias)

        # nn.init.zeros_(self.classifier[8].bias)
        # nn.init.zeros_(self.classifier[10].bias)
        # nn.init.zeros_(self.classifier[12].bias)

        # nn.init.zeros_(self.classifier[16].bias)
        # nn.init.zeros_(self.classifier[18].bias)
        # nn.init.zeros_(self.classifier[20].bias)

    def forward(self, x):
        out = self.classifier(x)
        out = out.view(out.size(0), num_classes)
        out = self.softmax(out)
        return out


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


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    ones = Variable(ones.cuda())
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss


device = torch.device("cuda")
torch.cuda.set_device(0)

# CIFAR dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=128, drop_last=True, shuffle=False, num_workers=6
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=100, drop_last=True, shuffle=False, num_workers=6
)


model = NeuralNet(input_size, hidden_size, num_classes)
model.cuda()

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.05)
        m.bias.data.normal_(0, 0.00001)

epoch_start = 1
# model.load_state_dict(torch.load('model_{:d}'.format(epoch_start)))
# model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

# param_dict = dict(model.named_parameters())
# params = []

base_lr = 0.1

# for key, value in param_dict.items():
#    if key == 'classifier.20.weight':
#        params += [{'params':[value], 'lr':0.1 * base_lr,
#            'momentum':0.95, 'weight_decay':0.0001}]
#    elif key == 'classifier.20.bias':
#        params += [{'params':[value], 'lr':0.1 * base_lr,
#            'momentum':0.95, 'weight_decay':0.0000}]
#    elif 'weight' in key:
#        params += [{'params':[value], 'lr':1.0 * base_lr,
#            'momentum':0.95, 'weight_decay':0.0001}]
#    else:
#        params += [{'params':[value], 'lr':2.0 * base_lr,
#            'momentum':0.95, 'weight_decay':0.0000}]


predict(test_loader, model)


optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
# optimizer = torch.optim.SGD(params, lr=base_lr, momentum=0.9)


total_step = len(train_loader)

if os.path.exists(curdir + "loss.txt"):
    os.remove(curdir + "loss.txt")

for epoch in range(epoch_start, 320):

    print("Epoch: {:d}".format(epoch))

    model.train()

    #    if epoch % 5 == 0:
    #        torch.save(model.state_dict(), 'model_{:d}'.format(epoch))

    if epoch % 80 == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1
            print("LR: {:.6f}".format(param_group["lr"]))

    for i, (images, labels) in enumerate(train_loader):

        images, labels = Variable(images.cuda()), Variable(labels.cuda())

        outputs = model(images)

        #        if i == 0 and epoch == 80:
        #            saveWeights(i, model)

        # loss = CrossEntropy(outputs, labels)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #        if i % 100 == 0:
        #            with open(curdir + 'loss.txt', 'a') as outfile:
        #                print(loss.item(), file = outfile)

        if i % 100 == 0:
            print("Step [{:4d}/{}], Loss: {:.4f}".format(i, total_step, loss.item()))

    predict(test_loader, model)
