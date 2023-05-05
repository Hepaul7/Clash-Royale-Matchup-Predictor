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


def train(lr: float, train_data: np.ndarray, val_data: np.ndarray, epochs: int, batch_size: int):
    """
    Train the autoencoder
    :param lr:
    :param train_data:
    :param val_data:
    :param epochs:
    :param batch_size:
    :return:
    """




# load the data
# data = process_cards.load_data(50000)
# np.save("matrix.npy", data)


# reload data
data = np.load("matrix.npy")
# split the data into train, validation and test sets
train, test = train_test_split(data, train_size=0.7)
test, val = train_test_split(test, test_size=0.5)
