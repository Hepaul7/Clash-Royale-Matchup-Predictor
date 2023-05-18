from typing import List, Tuple

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
from matrix_loader import *

NUM_CARDS = 109


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2 * NUM_CARDS, 1)

    def forward(self, inputs):
        x = F.sigmoid(self.fc1(inputs))

        return x

    def get_weight_norm(self):
        """
        Get the norm of the weights
        """
        return \
            torch.norm(self.fc1.weight)


def train_model(model: Net, data: torch.Tensor, epochs: int, lr: float, lamb: float,
                c_p: float, c_n: float) -> None:
    """Train the neural network model
    :param model: the neural network model
    :param data: the training data
    :param epochs: the number of epochs to train
    :param lr: the learning rate
    :param lamb: the regularization parameter
    :param c_p: the cost of false positive
    :param c_n: the cost of false negative
    """
    model.train()
    num_data = data.shape[0]
    # print("num_data: ", num_data)
    # define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # train the model
    num_pos = 0
    for epoch in range(epochs):
        print("Epoch: %d" % (epoch + 1))
        for row in range(num_data):
            # get the inputs
            inputs = Variable(data[row, 0:2 * NUM_CARDS])
            target = Variable(data[row, 2 * NUM_CARDS]).unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)[-1].unsqueeze(0)
            print(outputs, target)
            loss = criterion(outputs, target) * c_p if target.item() == 1 \
                else criterion(outputs, target) * c_n

            loss += (lamb / 2) * model.get_weight_norm()

            loss.backward()
            optimizer.step()

            # print statistics
            # print("Epoch: %d, Loss: %.3f" % (epoch + 1, loss.item()))


def evaluate_model(model: Net, data: torch.Tensor) -> float:
    """ Evaluate the model
    :param model: the neural network model
    :param data: the test data
    :return: Accuracy of the model
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

        prediction = 1 if output[0].item() > 0.52 else 0
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

    if positive_predictions == num_data or positive_predictions == 0:
        print("Model is overfitting")

    return correct / num_data


def tune_hyperparameters(lr: List[float], lamb: List[float], c_p: List[float], c_n: List[float],
                         num_epochs: List[int], train_data: torch.Tensor, valid_data: torch.Tensor)\
        -> Tuple[float, float, float, float, int, Net]:

    """Tune the hyperparameters of the neural network model
    :param lr: the list of learning rates
    :param lamb: the list of regularization parameters
    :param c_p: the list of costs of false positive
    :param c_n: the list of costs of false negative
    :param num_epochs: the list of number of epochs to train
    :param train_data: the training data
    :param valid_data: the validation data
    :return: the best hyperparameters as a tuple (lr, lamb, c_p, c_n, num_epochs)
    """

    best_model = None
    best_accuracy, best_lr, best_lamb, best_c_p, best_c_n, best_num_epochs = 0, 0, 0, 0, 0, 0
    for lr_i in lr:
        for lamb_i in lamb:
            for c_p_i in c_p:
                for c_n_i in c_n:
                    for num_epochs_i in num_epochs:
                        model = Net()
                        train_model(model, train_data, num_epochs_i, lr_i, lamb_i, c_p_i, c_n_i)
                        accuracy = evaluate_model(model, valid_data)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_lr = lr_i
                            best_lamb = lamb_i
                            best_c_p = c_p_i
                            best_c_n = c_n_i
                            best_num_epochs = num_epochs_i
                            best_model = model
    if best_model is not None:
        # Specify the file path to save the model
        model_path = "trained_model.pt"

        # Save the model's state dictionary
        torch.save(best_model.state_dict(), model_path)

        print("Trained model saved successfully.")

    return best_lr, best_lamb, best_c_p, best_c_n, best_num_epochs, best_model


def run() -> None:
    """Run the program
    """
    matrix = load_matrix()
    train, validation, test = split_data(matrix)

    # convert the data to torch Tensors
    train = ndarray_tensor(train)
    validation = ndarray_tensor(validation)
    test = ndarray_tensor(test)

    # tune the hyperparameters
    lrs = [0.1, 0.01, 0.001]
    lambs = [0.1, 0.01, 0.001]
    c_ps = [1]
    c_ns = [1.7, 1.75, 1.65]
    nums_epochs = [10, 20, 30]

    f_lr, f_lamb, f_c_p, f_c_n, f_best_num_epochs, model = tune_hyperparameters(
        lrs, lambs, c_ps, c_ns, nums_epochs, train, validation)

    evaluate_model(model, test)


matrix = load_matrix()
train, validation, test = split_data(matrix)

# convert the data to torch Tensors
train = ndarray_tensor(train)
validation = ndarray_tensor(validation)
test = ndarray_tensor(test)

model = Net()
train_model(model, train, 100, 0.01, 0.01, 1.0, 1.8)
evaluate_model(model, test)
