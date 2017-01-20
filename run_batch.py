import random
import numpy as np
from hyperparams import num_train, window_size
from algos.skipgram import skipgram

tweets_df = pd.read_csv(
    'cleaned_data/cleanedtext_of_all_tweets.csv',
    header=None,
    names=['body']
)

def run_batch(input_vectors, output_vectors):
    # Need center and context word indices for the algorithms
    center_word_index  = random.randint(0, input_vectors.shape[0] - 1)

    context_word_indices = list(range(center_word_index - window_size, center_word_index + window_size + 1))
    context_word_indices.remove(center_word_index)

    [skipgram_loss,
     skipgram_input_grad_loss,
     skipgram_output_grad_loss
    ] = skipgram(center_word_index, context_word_indices, input_vectors, output_vectors)

    return skipgram_loss, skipgram_input_grad_loss, skipgram_output_grad_loss
