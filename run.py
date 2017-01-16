from ipdb import set_trace as st
from hyperparams import num_features, batch_size, window_size, num_noise, filename, num_train
from scripts.initialize_vectors import initialize_vectors
from utils.softmax import softmax
from algos.skipgram import skipgram
from run_batch import run_batch
import pandas as pd
import random
import numpy as np

words_df = pd.read_csv(
    filename,
    header=None,
    index_col=False,
    names=['body']
)

input_vectors     = initialize_vectors(words_df.shape[0], num_features)
output_vectors = initialize_vectors(words_df.shape[0], num_features)

cost = 0
input_grad  = np.zeros(input_vectors.shape)
output_grad = np.zeros(output_vectors.shape)

for i in range(batch_size):
    print('running batch: ', i)

    [batch_loss,
     batch_input_grad,
     batch_output_grad
    ] = run_batch(input_vectors, output_vectors)

    cost        += batch_loss        / batch_size
    input_grad  += batch_input_grad  / batch_size
    output_grad += batch_output_grad / batch_size



print('final cost: ', cost)
print('final input grad: ', input_grad)
print('final output grad: ', output_grad)
