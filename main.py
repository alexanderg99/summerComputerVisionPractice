# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim

def downloadData():
    data_path = '../data-unversioned/p1ch7/'
    cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform= transforms.ToTensor())
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform= transforms.ToTensor())




def transformData(dataset):
    imgs = torch.stack([img_t for img_t, _ in dataset], dim=3)
    u = imgs.view(3,-1).mean(dim=1)
    sig = imgs.view(3,-1).std(dim=1)

    return u, sig

def main():
    downloadData()

    n_out= 2
    model = nn.Sequential(nn.Linear(3072,512),
                          nn.Tanh(),
                          nn.Linear(512,2),
                          nn.Softmax(dim=1))

    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    n_epochs = 100

    for epoch in range(n_epochs):
        for img,label in 1,1:
            out = model(img.view(-1).unsqueeze(0))
            #evaluating the model
            loss = loss_fn(out, torch.tensor)
            #computing the loss
            optimizer.zero_grad()
            #zero the gradient
            loss.backward()
            #propagate the gradient
            optimizer.step()
            # update weights







class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()


    def model(self):
        seq_model = nn.Sequential(nn.Linear(1,5), nn.ReLU, nn.Linear(5,1),
                                  nn.Sigmoid)
        return seq_model

    def train(self):
        pass

    def loss(self, x):
        return nn.MSELoss(x)





#dataloader shuffles and organizes the
#data in minibatches





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
