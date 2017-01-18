from ipdb import set_trace as st
from hyperparams import num_features, batch_size, window_size, num_noise, filename, num_train, num_iterations, learning_rate
from scripts.initialize_vectors import initialize_vectors
from utils.softmax import softmax
from algos.skipgram import skipgram
from run_batch import run_batch
from utils.pickling import save_data
import pandas as pd
import random
import numpy as np

words_df = pd.read_csv(
    filename,
    header=None,
    index_col=False,
    names=['body']
)

unique_words = words_df.body.unique()
num_unique_words = len(unique_words)
print('num_unique_words: ', num_unique_words)

i = iter(unique_words)
words_dict = dict(zip(i, np.arange(num_unique_words)))

cost = 0
input_grad  = np.zeros(input_vectors.shape)
output_grad = np.zeros(output_vectors.shape)


for j in range(num_iterations):
    print('iteration number: ', j)
    for i in range(batch_size):
        if i % 15 == 0:
            print('running batch: ', i)
        batch_loss        = 0
        batch_input_grad  = np.zeros(input_grad.shape)
        batch_output_grad = np.zeros(input_grad.shape)

        [batch_iter_loss,
         batch_iter_input_grad,
         batch_iter_output_grad
        ] = run_batch(input_vectors, output_vectors)

        batch_loss        += batch_iter_loss
        batch_input_grad  += batch_iter_input_grad
        batch_output_grad += batch_iter_output_grad


    batch_loss        /= batch_size
    batch_input_grad  /= batch_size
    batch_output_grad /= batch_size

    print('batch_loss: ', batch_loss)

    cost           += batch_loss
    input_vectors  -= batch_input_grad  * learning_rate
    output_vectors -= batch_output_grad * learning_rate

save_data(input_vectors, output_vectors)

print('final cost: ', cost)
print('final input grad: ', input_grad)
print('final output grad: ', output_grad)
