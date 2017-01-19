import numpy as np

def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x
