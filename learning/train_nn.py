import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable

from learning.matrix_loader import *
from learning.interactions_nn import *

NUM_CARDS = 109


def train_model(model: CardInteractionNet, train: Tensor, epochs: int, lr: float,
                interactions: Tensor):

    model.train()
    num_data = train.shape[0]
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for row in range(num_data):
            inputs = Variable(train[row, 0:2 * NUM_CARDS])
            target = Variable(train[row, 2 * NUM_CARDS]).unsqueeze(0)
            optimizer.zero_grad()

            outputs = model(inputs, interactions)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()


matrix = load_battle_matrix()
train, validation, test = split_data(matrix)

train = ndarray_tensor(train)
interactions = ndarray_tensor(load_interaction_matrix())

model = CardInteractionNet(50, 64)
train_model(model, train, 1, 0.01, interactions)


