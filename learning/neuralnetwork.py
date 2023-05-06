from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch
import matplotlib.pyplot as plt

import process_data.process_cards as process_cards
from sklearn.model_selection import train_test_split


class AutoEncoder(nn.Module):
    """
    Autoencoder class
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the autoencoder
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass of the autoencoder
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(model: AutoEncoder, lr: float, train_data: np.ndarray, val_data: np.ndarray, epochs: int,
          batch_size: int):
    """
    Train the autoencoder
    :param model: the autoencoder model
    :param lr:
    :param train_data:
    :param val_data:
    :param epochs:
    :param batch_size:
    :return:
    """
    model.train()
    # user Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # use Binary Cross Entropy loss
    criterion = nn.BCELoss()
    # convert the data to torch tensors
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    # create a data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # keep track of the loss
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        # keep track of the loss
        running_loss = 0.0
        # train the model
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs = data
            # wrap them in Variable
            inputs = Variable(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
        train_loss.append(running_loss / len(train_loader))
        # validate the model
        running_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            # get the inputs
            inputs = data
            # wrap them in Variable
            inputs = Variable(inputs)
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            # print statistics
            running_loss += loss.data[0]
        val_loss.append(running_loss / len(val_loader))
        print("Epoch: {}, Train Loss: {}, Validation Loss: {}".format(epoch, train_loss[-1], val_loss[-1]))



# load the data
# data = process_cards.load_data(50000)
# np.save("matrix.npy", data)


# reload data
data = np.load("matrix.npy")
# split the data into train, validation and test sets
train, test = train_test_split(data, train_size=0.7)
test, val = train_test_split(test, test_size=0.5)
