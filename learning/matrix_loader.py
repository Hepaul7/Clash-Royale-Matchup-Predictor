from sklearn.model_selection import train_test_split
import torch
from process_data.data_process import *


def load_interaction_matrix() -> np.ndarray:
    """
    Load the saved matrix of card interactions
    """
    matrix = np.load(INTERACTION_PATH)
    return matrix


def load_battle_matrix() -> np.ndarray:
    """Load the saved matrix of labeled data
    return: the matrix of labeled data
    """
    matrix = np.load(MATRIX_PATH)
    # np.random.shuffle(matrix)
    return matrix


def create_battle_matrix() -> np.ndarray:
    """Create the matrix of labeled data
    return: the matrix of labeled data
    """
    matrix = load_data()
    np.shuffle(matrix)
    # save the matrix
    np.save(MATRIX_PATH, matrix)
    return matrix


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

