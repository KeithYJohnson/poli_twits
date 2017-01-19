from utils.noisy_vec_generator import NoisyVecGenerator
import pandas as pd
import numpy as np

num_features = 100
batch_size   = 50
window_size  = 2 # How many words on both the left and right of the target word to include in the context.
num_noise    = 3
filename = 'cleaned_data/cleanedwords_from_all_tweets.csv'
learning_rate = 0.001
num_iterations = 1000

import os, re
string = re.findall(r'\d+', os.popen('wc -l {}'.format(filename)).read())
num_train = int(string[0])

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
global generator
generator  = NoisyVecGenerator(words_df, words_dict)
