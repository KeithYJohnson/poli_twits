import random
import numpy as np
import pandas as pd
import math
from hyperparams import num_train, window_size, words_dict
from algos.skipgram import skipgram
from ipdb import set_trace as st

tweets_df = pd.read_csv(
    'cleaned_data/cleanedtext_of_all_tweets.csv',
    header=None,
    names=['body']
)

def run_batch(word_vectors):
    sentence_index            = random.randint(0, len(tweets_df) - 1)
    sentence_words            = tweets_df.body[sentence_index].split()
    sentence_words_wv_indices = [words_dict[word] for word in sentence_words if word in words_dict]

    if len(sentence_words_wv_indices) > 2 * window_size + 1:
        center_word_index = random.randint(window_size, len(sentence_words_wv_indices) - window_size)
        context_word_indices = list(range(center_word_index - window_size, center_word_index + window_size + 1))
        context_word_indices.remove(center_word_index)
    else:
        center_word_index = math.ceil(len(sentence_words_wv_indices) / 2)
        context_word_indices = list(range(0, len(sentence_words_wv_indices)))
        context_word_indices.remove(center_word_index)

    [skipgram_loss,
     skipgram_grad
    ] = skipgram(center_word_index, context_word_indices, word_vectors)

    return skipgram_loss, skipgram_grad
