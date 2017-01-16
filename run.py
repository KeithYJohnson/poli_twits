from ipdb import set_trace as st
from hyperparams import num_features, batch_size, window_size, num_noise
from scripts.initialize_vectors import initialize_vectors
from utils.softmax import softmax
from algos.skipgram import skipgram
import pandas as pd
import random
import numpy as np

words_df = pd.read_csv(
    'cleaned_data/cleanedwords_from_all_tweets.csv',
    header=None,
    index_col=False,
    names=['body']
)
num_train = words_df.shape[0]

input_vectors     = initialize_vectors(words_df.shape[0], num_features)
output_vectors = initialize_vectors(words_df.shape[0], num_features)

cost = 0
input_grad  = np.zeros(input_vectors.shape)
output_grad = np.zeros(output_vectors.shape)

for i in range(batch_size):
    print('running batch: ', i)
    center_word_index  = random.randint(0, num_train - 1)

    context_word_indices = list(range(center_word_index - window_size, center_word_index + window_size + 1))
    context_word_indices.remove(center_word_index)

    batch_loss = 0
    batch_input_grad  = np.zeros(input_grad.shape)
    batch_output_grad = np.zeros(output_grad.shape)

    [skipgram_loss,
     skipgram_input_grad_loss,
     skipgram_output_grad_loss
    ] = skipgram(center_word_index, context_word_indices, batch_input_grad, batch_output_grad)

    batch_loss        += skipgram_loss
    batch_input_grad  += skipgram_input_grad_loss
    batch_output_grad += skipgram_output_grad_loss

    cost        += batch_loss        / batch_size
    input_grad  += batch_input_grad  / batch_size
    output_grad += batch_output_grad / batch_size


print('final cost: ', cost)
print('final input grad: ', input_grad)
print('final output grad: ', output_grad)
