import numpy as np

def initialize_vectors(xdim, ydim):
    random_mat = np.random.randn(xdim, ydim)
    # Make each row have unit length
    return random_mat / random_mat.max(axis = 0)
