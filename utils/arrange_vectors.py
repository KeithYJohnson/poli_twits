import numpy as np
from ipdb import set_trace as st

def stack_vecs(input_vecs, output_vecs):
    return np.vstack([input_vecs, output_vecs])

def split_vecs(word_vecs):
    num_train = word_vecs.shape[0]
    input_vectors  = word_vecs[:num_train / 2, :]
    output_vectors = word_vecs[num_train / 2:, :]

    return input_vectors, output_vectors
