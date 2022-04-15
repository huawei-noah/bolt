import sys
sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pytorch_to_caffe


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print("in forward function, type of x is: ")
        print(type(x))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

if __name__ == '__main__':
    model_path = "./lenet.pth"
    model = LeNet()
    model = torch.load(model_path)
    model.eval()

    name = "LeNet"
    input = torch.ones([1, 1, 28, 28])
    pytorch_to_caffe.trans_net(model, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
