import numpy as np
from sklearn.model_selection import train_test_split
import torch


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

