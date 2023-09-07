import torch.nn as nn
import torch.utils.data
from sklearn import metrics
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from learning.matrix_loader import *

NUM_CARDS = 109
MATRIX_PATH = "../matrix.npy"
MODEL_PATH = "../model.pth"

INTERACTION_MATRIX = load_interaction_matrix()


class Net(nn.Module):
    def __init__(self):
        """ Initialize the neural network model
        """
        super().__init__()
        self.fc1 = nn.Linear(20, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

    def get_weight_norm(self):
        """
        Get the norm of the weights
        """
        return torch.norm(self.fc1.weight, 2) ** 2 + torch.norm(self.fc2.weight) ** 2


def train_model(model: Net, data: torch.Tensor, epochs: int, lr: float, lamb: float):
    """

    """
    model.train()
    num_data = data.shape[0]
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for row in range(num_data):
            inputs = data[row, 0:20].unsqueeze(0)
            target = data[row, 20].unsqueeze(0)
            print(inputs.shape)
            print(target.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)
            print(outputs.item())
            loss = criterion(input=torch.tensor(outputs.item()).view(1), target=target) - (lamb / 2) * torch.norm(model.get_weight_norm())

            loss.backward()
            optimizer.step()


def evaluate(model, data):
    """

    """
    model.eval()
    num_data = data.shape[0]
    true_labels = []
    predicted = []

    for row in range(num_data):
        inputs = data[row, 0:20].unsqueeze(0)
        target = data[row, 20].unsqueeze(0)

        output = model(inputs)


        true_labels.append(target.item())
        predicted.append(output.item())

    true_labels = np.array(true_labels)
    predicted = np.array(predicted)
    binary_predicted_labels = (predicted >= 0.5).astype(int)

    print(true_labels)
    print(binary_predicted_labels)

    accuracy = metrics.accuracy_score(true_labels, binary_predicted_labels)
    return accuracy


data = load_battle_matrix()
np.random.shuffle(data)
train, test = train_test_split(data, train_size=0.7)
train, test = ndarray_tensor(train), ndarray_tensor(test)
model = Net()
train_model(model, train, 10, 0.01, 0.01)

acc = evaluate(model, test)
