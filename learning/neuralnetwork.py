from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch

from sklearn.model_selection import train_test_split

NUM_CARDS = 109


def load_matrix() -> np.ndarray:
    """Load the saved matrix of labeled data
    return: the matrix of labeled data
    """
    return np.load("matrix.npy")


def split_data(matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Split the data into train, validation and test sets
    :param matrix: the matrix of labeled data
    :return: train, validation and test sets
    """
    # split the data into train, validation and test sets
    train, test = train_test_split(matrix, train_size=0.7)
    test, validation = train_test_split(train, test_size=0.5)
    return train, validation, test


def ndarray_tensor(matrix: np.ndarray) -> torch.Tensor:
    """Convert a numpy ndarray to a torch Tensor
    :param matrix: the numpy ndarray
    :return: the torch Tensor
    """
    return torch.from_numpy(matrix).float()  # convert to float tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2 * NUM_CARDS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


def train_model(model: Net, data: torch.Tensor, epochs: int, lr: float, lamb: float):
    """Train the neural network model
    :param model: the neural network model
    :param data: the training data
    :param epochs: the number of epochs to train
    :param lr: the learning rate
    :param lamb: the regularization parameter
    """
    model.train()
    num_data = data.shape[0]
    print("num_data: ", num_data)
    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # train the model
    for epoch in range(epochs):

        for row in range(num_data):
            # get the inputs
            inputs = Variable(data[row, 0:2 * NUM_CARDS])
            target = Variable(data[row, 2 * NUM_CARDS]).unsqueeze(0)
            print(inputs)
            print(target)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, target)  # + (lamb / 2) * model.get_weight_norm()
            loss.backward()
            optimizer.step()

            # print statistics
            # print("Epoch: %d, Loss: %.3f" % (epoch + 1, loss.item()))


def evaluate_model(model: Net, data: torch.Tensor):
    """ Evaluate the model
    """
    num_data = data.shape[0]
    correct = 0
    num_win = 0
    num_defeat = 0
    positive_predictions = 0
    for row in range(num_data):
        inputs = Variable(data[row, 0:2 * NUM_CARDS])
        target = Variable(data[row, 2 * NUM_CARDS])
        output = model(inputs)

        if target.item() == 1:
            num_win += 1
        else:
            num_defeat += 1

        prediction = 1 if output[0].item() > 0.5 else 0
        if prediction == target.item():
            correct += 1
        else:
            print(output[0].item(), target.item())

        if prediction == 1:
            positive_predictions += 1

    print("Accuracy: %.3f" % (correct / num_data))
    print("num_wins: ", num_win)
    print("num_defeats: ", num_defeat)
    print("num_positive_predictions: ", positive_predictions)


matrix = load_matrix()
train, validation, test = split_data(matrix)

# convert the data to torch Tensors
train = ndarray_tensor(train)
validation = ndarray_tensor(validation)
test = ndarray_tensor(test)

# create the neural network model
model = Net()

# train the model
train_model(model, train, 1, 0.01, 0)
evaluate_model(model, test)
