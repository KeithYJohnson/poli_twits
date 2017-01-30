from ipdb import set_trace as st
from hyperparams import num_features, batch_size, window_size, num_noise, num_train, num_iterations, learning_rate, filename
from scripts.initialize_vectors import initialize_vectors
from run_batch import run_batch
from utils.pickling import save_data
from utils.noisy_vec_generator import NoisyVecGenerator
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
generator  = NoisyVecGenerator(words_df, words_dict)

word_vectors = initialize_vectors(2 * num_unique_words, num_features)

hparams = {
    'words_dict':  words_dict,
    'noise_gen':   generator,
    'words_df':    words_df,
    'num_noise':   num_noise,
    'batch_size':  batch_size,
    'window_size': window_size,
    'num_train':   num_train,
    'batch_size':  batch_size
}

cost = 0
grad = np.zeros(word_vectors.shape)
def batch_wrapper(word_vectors, hparams={}):
    batch_loss = 0
    batch_grad = np.zeros(word_vectors.shape)
    for i in range(hparams['batch_size']):
        if i % 50 == 0:
            print('running batch: ', i)

        [batch_iter_loss,
         batch_iter_grad,
        ] = run_batch(word_vectors, hparams=hparams)

        batch_loss += batch_iter_loss
        batch_grad += batch_iter_grad

    batch_loss  /= batch_size
    batch_grad  /= batch_size

    return batch_loss, batch_grad


for j in range(num_iterations):
    print('iteration number: ', j)
    batch_loss, batch_grad = batch_wrapper(word_vectors, hparams=hparams)
    print('batch_loss: ', batch_loss)

    cost += batch_loss
    grad -= batch_grad

save_data(input_vectors, output_vectors)

print('final cost: ', cost)
print('final grad: ', grad)
