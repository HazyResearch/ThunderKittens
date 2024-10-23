import numpy as np


def count_parameters(model, requires_grad: bool = True):
    """
    Return total # of trainable parameters
    """
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    try:
        return sum([np.prod(p.size()) for p in model_parameters]).item()
    except:
        return sum([np.prod(p.size()) for p in model_parameters])
