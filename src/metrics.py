import numpy as np


def mbe(y_true, prob):
    y_true = np.mean(y_true)
    y_pred = np.mean(prob)
    error = abs(y_true - y_pred)
    return error
