from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from matrix_loader import *


def load_data() -> (torch.nn, torch.nn, torch.nn):
    """
    Load the data from a saved matrix, split into
    train, validation and test sets (70%, 15%, 15%)
    """
    matrix = load_battle_matrix()
    train, validation, test = split_data(matrix)
    train = ndarray_tensor(train)
    validation = ndarray_tensor(validation)
    test = ndarray_tensor(test)
    return train, validation, test


def split_labels(data: torch.nn) -> (torch.nn, torch.nn):
    # split labels
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def run_logistic() -> float:
    train, _, test = load_data()
    x_train, y_train = split_labels(train)
    x_test, y_test = split_labels(test)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)


def run_naive_bayes() -> float:
    train, _, test = load_data()
    x_train, y_train = split_labels(train)
    x_test, y_test = split_labels(test)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)



