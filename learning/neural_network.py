import torch.nn as nn
import torch.utils.data
from torch.nn import init
from learning.matrix_loader import *


NUM_CARDS = 109
MATRIX_PATH = "../matrix.npy"
MODEL_PATH = "../model.pth"


class Net(nn.Module):
    def __init__(self):
        """ Initialize the neural network model
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2 * NUM_CARDS, NUM_CARDS)
        self.fc2 = nn.Linear(NUM_CARDS, 1)

        init.xavier_uniform_(self.fc1.weight)
        # self.initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def initialize_weights(self):
        weights = torch.zeros_like(self.fc1.weight.data)
        weights[:NUM_CARDS, :] = 3.0  # Assign higher weights to indices 0-218
        self.fc1.weight.data.copy_(weights)

    def get_weight_norm(self):
        """
        Get the norm of the weights
        """
        return torch.norm(self.fc1.weight, 2) ** 2  + torch.norm(self.fc2.weight) ** 2


