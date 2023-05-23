import torch
import os
from learning.neuralnetwork import run,run_simple, save_model

MODEL_PATH = "../model.pth"


def run_model() -> bool:
    """ Run and save the model
    """
    model = run_simple()
    save_model(model)
    if os.path.exists(MODEL_PATH):
        return True
    return False


def load_model() -> bool:
    """ Load an existing model
    """
    if os.path.exists(MODEL_PATH):
        torch.load(MODEL_PATH)
        return True
    return False




