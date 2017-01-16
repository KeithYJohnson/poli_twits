from hyperparams import num_features, batch_size, window_size, num_noise
from scripts.initialize_vectors import initialize_vectors
import pandas as pd
import numpy as np
words_df = pd.read_csv(
    'cleaned_data/cleanedwords_from_all_tweets.csv',
    header=None,
    index_col=False,
    names=['body']
)
num_train = words_df.shape[0]

embeddings     = initialize_vectors(words_df.shape[0], num_features)
output_vectors = initialize_vectors(words_df.shape[0], num_features)

cost = 0
input_grad  = np.zeros(embeddings.shape)
output_grad = np.zeros(output_vectors.shape)

