from model2 import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from torch.autograd import Variable

num_classes = 10
import os

curdir = "./weights/"


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


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


if __name__ == "__main__":
    batch_size = 50
    train_dataset = mnist.MNIST(root="./train", train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root="./test", train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model()
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss()
    epoch = 1

    for _epoch in range(epoch):
        correct = 0
        _sum = 0
        for idx, (test_x, test_label) in enumerate(test_loader):
            if idx < 1:
                saveWeights(idx, model)
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print("accuracy: {:.4f}".format(correct / _sum))
        print("error:\n")
        for idx, (train_x, train_label) in enumerate(train_loader):

            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            _error, _ = CrossEntropy(predict_y, train_label)
            predict_y = model(train_x.float()).detach()
            if idx % 100 == 0:
                print("{:.8f}".format(_error.item()))

            _error.backward()
            sgd.step()

        correct = 0
        _sum = 0
        # model.train(False)
        for idx, (test_x, test_label) in enumerate(test_loader):

            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            # print(predict_ys)
            # plt.imshow(test_x[0].reshape(28, 28), cmap=plt.cm.binary)
            # plt.show()
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print("accuracy: {:.4f}".format(correct / _sum))
        # torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))

plt.imshow(test_x[0].reshape(28, 28), cmap=plt.cm.binary)
plt.show()
