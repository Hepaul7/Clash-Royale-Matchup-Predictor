import torch.nn as nn
import torch.utils.data
from sklearn import metrics
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from learning.matrix_loader import *

NUM_CARDS = 109
MATRIX_PATH = "../matrix_updated.npy"
MODEL_PATH = "../model.pth"

INTERACTION_MATRIX = load_interaction_matrix()


class Net(nn.Module):
    def __init__(self):
        """ Initialize the neural network model
        """
        super().__init__()
        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        # self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = F.relu(x)
        return x

    def get_weight_norm(self):
        """
        Get the norm of the weights
        """
        return torch.norm(self.fc1.weight, 2) ** 2 + torch.norm(self.fc2.weight) ** 2 + \
               torch.norm(self.fc3.weight, 2) ** 2


def train_model(model: Net, data: torch.Tensor, epochs: int, lr: float, lamb: float):
    """

    """
    model.train()
    num_data = data.shape[0]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for row in range(num_data):
            inputs = data[row, 0:20].unsqueeze(0)
            target = data[row, 20].unsqueeze(0)
            # print(inputs.shape)
            # print(target.shape)
            print(inputs)
            print(target)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            # print(outputs.item())
            loss = criterion(input=torch.tensor(outputs.item()).view(1), target=target) - (
                        lamb / 2) * torch.norm(model.get_weight_norm())

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

    accuracy = metrics.accuracy_score(true_labels, binary_predicted_labels)
    precision = metrics.precision_score(true_labels, binary_predicted_labels)
    recall = metrics.recall_score(true_labels, binary_predicted_labels)
    f1_score = metrics.f1_score(true_labels, binary_predicted_labels)
    roc_auc = metrics.roc_auc_score(true_labels, predicted)

    print("Accuracy: %.3f" % accuracy)
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1 Score: %.3f" % f1_score)
    print("ROC-AUC: %.3f" % roc_auc)

    print(true_labels)
    print(binary_predicted_labels)

    accuracy = metrics.accuracy_score(true_labels, binary_predicted_labels)
    return accuracy


def run_simple():
    data = load_battle_matrix()
    # np.random.shuffle(data)
    train, test = train_test_split(data, train_size=0.8)
    train, test = ndarray_tensor(train), ndarray_tensor(test)
    model = Net()
    train_model(model, train, 10, 0.1, 0.01)

    acc = evaluate(model, test)
    print(acc)


def tune_hyperparameters():
    lrs = [0.1, 0.01, 0.001, 0.05]
    lambs = [0.01, 0.001]
    epochs = [4, 7, 10, 13, 16]

    max_acc = 0
    max_lamb = 0
    max_epoch = 0
    max_lr = 0

    data = load_battle_matrix()
    np.random.shuffle(data)
    train, test = train_test_split(data, train_size=0.7)
    train, test = ndarray_tensor(train), ndarray_tensor(test)

    best_model = None
    for lr in lrs:
        for lamb in lambs:
            for epoch in epochs:
                model = Net()
                train_model(model, train, epoch, lr, lamb)
                acc = evaluate(model, test)
                print(lr, lamb, epoch, acc)
                if acc > max_acc:
                    max_acc = acc
                    max_lamb = lamb
                    max_epoch = epoch
                    max_lr = lr
                    best_model = model

    print(max_acc, max_lr, max_lamb, max_epoch)
    evaluate(best_model, test)
    return best_model


