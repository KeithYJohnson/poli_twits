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

for i in range(batch_size):
    center_word_index  = random.randint(0, num_train - 1)

    center_word        = words_df.iloc[center_word_index]
    center_word_vector = embeddings[i]

    context_word_indices = list(range(center_word_index - window_size, center_word_index + window_size + 1))
    context_word_indices.remove(center_word_index)

    batch_loss = 0
    batch_input_grad  = np.zeros(input_grad.shape)
    batch_output_grad = np.zeros(output_grad.shape)

    for context_word_idx in context_word_indices:
        probabilities = softmax(center_word_vector.dot(output_vectors.T))
        context_word_loss = -np.log(probabilities)

        output_layer_error = probabilities
        output_layer_error[context_word_idx] -= 1

        num_predictions = output_layer_error.shape[0]

        input_grad_at_context_word = np.dot(
            output_layer_error.reshape(1, num_predictions),
            output_vectors
        ).flatten()

        output_grad_at_context_word = np.multiply(
            output_layer_error.reshape(num_predictions, 1),
            center_word_vector.reshape(1, num_features)
        )

        batch_loss        += context_word_loss
        batch_input_grad  += input_grad_at_context_word
        batch_output_grad += output_grad_at_context_word

    cost        += batch_loss        / batch_size
    input_grad  += batch_input_grad  / batch_size
    output_grad += batch_output_grad / batch_size


print('final cost: ', cost)
print('final input grad: ', input_grad)
print('final output grad: ', output_grad)
