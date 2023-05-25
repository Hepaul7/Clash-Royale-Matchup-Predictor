from learning.neural_network import *
from torch.autograd import Variable
from typing import List, Tuple
import torch.optim as optim
import numpy as np

from sklearn import metrics


def train_model(model: Net, train: torch.Tensor, valid: torch.Tensor,
                test: torch.Tensor, epochs: int, lr: float, lamb: float,
                c_p: float, c_n: float) -> Tuple[int, float | int]:
    """Train the neural network model
    :param model: the neural network model
    :param train: the training data
    :param valid: the validation data
    :param epochs: the number of epochs to train
    :param test: the test data
    :param lr: the learning rate
    :param lamb: the regularization parameter
    :param c_p: the cost of false positive
    :param c_n: the cost of false negative
    """
    model.train()
    num_data = train.shape[0]
    # print("num_data: ", num_data)
    # define the loss function and optimizer
    class_weights = torch.tensor([IMBALANCE_RATIO])

    criterion = nn.BCELoss(weight=class_weights)

    # criterion = nn.SoftMarginLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train the model
    max_acc = 0
    max_epoch = 0
    for epoch in range(epochs):
        print("Epoch: %d" % (epoch + 1))
        for row in range(num_data):
            # get the inputs
            inputs = Variable(train[row, 0:2 * NUM_CARDS])
            target = Variable(train[row, 2 * NUM_CARDS]).unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # Calculate class weights dynamically
            imbalance_ratio = 0.67  # Update with the correct imbalance ratio
            class_weights = torch.tensor([1.0, 1.0 / imbalance_ratio])
            class_weights = class_weights[target.long()].to(target.device)

            # Apply class weights to the loss function
            criterion.weight = class_weights

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, target) * c_p if target.item() == 1 \
                else criterion(outputs, target) * c_n

            loss += (lamb / 2) * torch.norm(model.get_weight_norm())

            # TODO: monitor on validation set to see if we need to stop

            loss.backward()
            optimizer.step()

        accuracy = evaluate_model(model, test)
        print("Epoch: %d, Accuracy: %.3f" % (epoch + 1, accuracy))
        if accuracy > max_acc:
            max_acc = accuracy
            max_epoch = epoch + 1
    return max_epoch, max_acc


def evaluate_model(model: Net, data: torch.Tensor) -> float:
    """ Evaluate the model
    :param model: the neural network model
    :param data: the test data
    :return: Accuracy, Precision, Recall, F1 score, ROC-AUC of the model
    """
    num_data = data.shape[0]
    true_labels = []
    predicted_labels = []

    for row in range(num_data):
        inputs = Variable(data[row, 0:2 * NUM_CARDS])
        target = Variable(data[row, 2 * NUM_CARDS])
        output = model(inputs)

        true_labels.append(target.item())
        predicted_labels.append(output.item())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Convert labels to binary values using a threshold (0.5)
    binary_predicted_labels = (predicted_labels >= 0.5).astype(int)

    accuracy = metrics.accuracy_score(true_labels, binary_predicted_labels)
    precision = metrics.precision_score(true_labels, binary_predicted_labels)
    recall = metrics.recall_score(true_labels, binary_predicted_labels)
    f1_score = metrics.f1_score(true_labels, binary_predicted_labels)
    roc_auc = metrics.roc_auc_score(true_labels, predicted_labels)

    print("Accuracy: %.3f" % accuracy)
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1 Score: %.3f" % f1_score)
    print("ROC-AUC: %.3f" % roc_auc)

    return accuracy


def tune_hyperparameters(lr: List[float], lamb: List[float], c_p: List[float], c_n: List[float],
                         num_epochs: List[int], train_data: torch.Tensor, valid_data: torch.Tensor,
                         test_data: torch.Tensor) \
        -> Tuple[float, float, float, float, int, Net]:
    """Tune the hyperparameters of the neural network model
    :param lr: the list of learning rates
    :param lamb: the list of regularization parameters
    :param c_p: the list of costs of false positive
    :param c_n: the list of costs of false negative
    :param num_epochs: the list of number of epochs to train
    :param train_data: the training data
    :param valid_data: the validation data
    :param test_data: the test data
    :return: the best hyperparameters as a tuple (lr, lamb, c_p, c_n, num_epochs)
    """

    best_model = None
    best_accuracy, best_lr, best_lamb, best_c_p, best_c_n, best_num_epochs = 0, 0, 0, 0, 0, 0
    for lr_i in lr:
        for lamb_i in lamb:
            for c_p_i in c_p:
                for c_n_i in c_n:
                    model = Net()
                    res = train_model(model, train_data, valid_data, test_data, num_epochs[0], lr_i,
                                      lamb_i, c_p_i, c_n_i)
                    accuracy = evaluate_model(model, valid_data)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_lr = lr_i
                        best_lamb = lamb_i
                        best_c_p = c_p_i
                        best_c_n = c_n_i
                        best_num_epochs = res[0]
                        best_model = model
    if best_model is not None:
        # Specify the file path to save the model
        model_path = MODEL_PATH

        # Save the model's state dictionary
        torch.save(best_model.state_dict(), model_path)

    return best_lr, best_lamb, best_c_p, best_c_n, best_num_epochs, best_model


def save_model(model: torch.nn) -> None:
    """ Save the model
    """
    torch.save(model, MODEL_PATH)


def run() -> torch.nn:
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
    return model


def run_simple():
    matrix = load_matrix()
    train, validation, test = split_data(matrix)

    # convert the data to torch Tensors
    train = ndarray_tensor(train)
    validation = ndarray_tensor(validation)
    test = ndarray_tensor(test)

    model = Net()
    train_model(model, train, validation, test, 1, 0.01, 0.1, 1.0, 1.7)
    save_model(model)
