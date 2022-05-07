import torch
from torch import nn
import cv2
import numpy as np

from lenet5_net import LeNet5
from load_data import load_data_mnist
from utils import Accumulator, Timer


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device

    metric = Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_net(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)
        print("epoch {}, loss {:.3f}, train acc {:.3f}, test acc {:.3f}".format(epoch, train_l, train_acc, test_acc))
    print("{:.1f} examples/sec on {}".format(metric[2] * num_epochs / timer.sum(), str(device)))
    img = read_data("./data/5.jpg")
    input_tensor = torch.from_numpy(img)
    traced = torch.jit.trace(net, input_tensor)
    traced.save('./model/lenet.zip')
    print('-' * 100)


def read_data(im_path):
    img = cv2.imread(im_path, 0)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 10)
    x = binary_img / 255
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    return x


if __name__ == '__main__':
    train_iter, test_iter = load_data_mnist(batch_size=256, resize=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LeNet5()
    train_net(net=net,
              train_iter=train_iter,
              test_iter=test_iter,
              num_epochs=5,
              lr=0.25,
              device=device)

