import numpy as np

def softmax(x):
    shiftx = x - np.max(x, axis=x.ndim-1, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=x.ndim-1, keepdims=True)
